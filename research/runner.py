"""Run independent portfolio rules, feature recipes and fixed prediction models."""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import zipfile
import numpy as np
import pandas as pd
from .adjustments import identity, loss_probability, upside_probability
from .accounting import portfolio_with_cash
from .data import prepare_data
from .features import FeatureSpec
from .legacy import KEYS, PROJECT_ROOT, downside_statistics, load_panel, performance, portfolio_returns
from .models import PredictionBatch, default_model_registry, evaluate_predictions, predict_model, event_fields
from .strategies import StrategyContext, default_registry


@dataclass(frozen=True)
class RunConfig:
    strategies: tuple[str, ...] = ("equal_weight_universe", "momentum_weighted")
    evaluation: str = "training_validation"
    validation_history_months: int = 60
    include_ml: bool = False
    adjustment_strength: float = 0.5
    models: tuple[str, ...] = ("logistic",)
    feature_spec: FeatureSpec = field(default_factory=FeatureSpec)
    refit_months: int = 12
    fit_policy: str = "expanding"
    cash_momentum_months: int = 12
    prediction_target: str = "loss"
    model_scope: str = "pooled"
    upside_threshold: float = .6
    min_stock_history: int = 36
    min_stock_class: int = 6

    def __post_init__(self):
        if self.evaluation not in {"training_validation", "test"}:
            raise ValueError("Choose training_validation or test.")
        if not isinstance(self.validation_history_months, int) or self.validation_history_months < 24:
            raise ValueError("Use at least 24 initial training months.")
        if not self.strategies or len(set(self.strategies)) != len(self.strategies):
            raise ValueError("Choose at least one strategy, without duplicates.")
        if not np.isfinite(self.adjustment_strength) or not 0 <= self.adjustment_strength < 1:
            raise ValueError("Adjustment strength must lie in [0, 1).")
        if (self.include_ml and not self.models) or len(set(self.models)) != len(self.models):
            raise ValueError("Choose distinct models and at least one when ML is enabled.")
        if not isinstance(self.feature_spec, FeatureSpec):
            raise ValueError("feature_spec must be a FeatureSpec.")
        if not isinstance(self.refit_months, int) or self.refit_months < 1:
            raise ValueError("refit_months must be a positive integer.")
        if self.fit_policy not in {"expanding", "frozen_at_origin"}:
            raise ValueError("Choose expanding or frozen_at_origin model fits.")
        if type(self.cash_momentum_months) is not int or not 2 <= self.cash_momentum_months <= 12:
            raise ValueError("Cash-strategy momentum lookback must be an integer from 2 to 12 months.")
        event_fields(self.prediction_target)
        if self.model_scope not in {"pooled", "per_stock"}:
            raise ValueError("Choose pooled or per_stock model scope.")
        if not np.isfinite(self.upside_threshold) or not 0 <= self.upside_threshold <= 1:
            raise ValueError("Upside threshold must lie in [0, 1].")
        if type(self.min_stock_history) is not int or self.min_stock_history < 24:
            raise ValueError("Per-stock models need at least 24 history months.")
        if type(self.min_stock_class) is not int or self.min_stock_class < 1:
            raise ValueError("Per-stock models need a positive minimum class count.")


def _json(value):
    return json.dumps(value, indent=2, default=str, allow_nan=False)


@dataclass
class RunResult:
    config: RunConfig
    metrics: pd.DataFrame
    paired: pd.DataFrame
    monthly: pd.DataFrame
    diagnostics: pd.DataFrame
    weights: dict[str, pd.DataFrame]
    books: dict[str, pd.DataFrame]
    signals: pd.DataFrame
    screen: pd.DataFrame
    predictions: PredictionBatch | None
    manifest: dict
    screens: dict[str, pd.DataFrame] = field(default_factory=dict)
    strategy_screen_keys: dict[str, str] = field(default_factory=dict)
    cases: pd.DataFrame = field(default_factory=pd.DataFrame)
    prediction_batches: dict[str, PredictionBatch] = field(default_factory=dict)
    data_audit: dict = field(default_factory=dict)

    def case_id(self, strategy, variant="baseline", model_id=None):
        rows = self.cases.loc[self.cases.strategy.eq(strategy) & self.cases.variant.eq(variant)]
        if variant != "baseline":
            model_id = model_id or self.config.models[0]
            rows = rows.loc[rows.model_id.eq(model_id)]
        if len(rows) != 1:
            raise ValueError("Choose exactly one existing strategy/model case.")
        return rows.case_id.iloc[0]

    def allocation(self, strategy, variant, snapshot, model_id=None):
        return self.allocation_case(self.case_id(strategy, variant, model_id), snapshot)

    def allocation_case(self, case_id, snapshot):
        case = self.cases.set_index("case_id").loc[case_id]
        weight = self.weights[case_id]
        allocation = weight.loc[weight.date.eq(pd.Timestamp(snapshot))].merge(
            self.signals, on=KEYS, how="left", validate="one_to_one")
        if case.get("allows_cash", False) and "cash_trailing_momentum" in allocation:
            columns = ["trailing_momentum", "trailing_volatility", "history_months", "eligible"]
            allocation = allocation.drop(columns=columns).rename(columns={f"cash_{c}": c for c in columns})
        baseline = self.weights[self.case_id(case.strategy)].rename(columns={"weight": "baseline_weight"})
        allocation = allocation.merge(baseline, on=KEYS, validate="one_to_one")
        screen = self.screens.get(self.strategy_screen_keys.get(case.strategy), pd.DataFrame())
        if not screen.empty:
            columns = [c for c in ["stock_id", "selected", "training_volatility", "training_rank",
                "training_sharpe", "training_sortino", "screen_metric", "screen_score", "screen_asof",
                "complete_years", "positive_years", "negative_years", "flat_years", "positive_year_fraction",
                "last_complete_year", "exclusion_reason"] if c in screen]
            allocation = allocation.merge(screen[columns], on="stock_id", how="left", validate="many_to_one")
        batch = self.prediction_batches.get(case.model_id)
        if batch is not None:
            allocation = allocation.merge(batch.predictions, on=KEYS, validate="one_to_one")
        allocation["weight_change"] = allocation.weight - allocation.baseline_weight
        return allocation.sort_values(["weight", "stock_id"], ascending=[False, True]).reset_index(drop=True)

    @property
    def model_metrics(self):
        rows = []
        for index, (key, batch) in enumerate(self.prediction_batches.items()):
            table = batch.metrics.copy()
            table["model_id"] = [key, event_fields(self.config.prediction_target)["control_id"]]
            if index:
                table = table.iloc[:1]
            rows.append(table)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _files(self, include_weights=False):
        yield "manifest.json", _json(self.manifest).encode()
        yield "data_audit.json", _json(self.data_audit).encode()
        tables = {"cases": self.cases, "strategy_comparison": self.metrics,
                  "paired_changes": self.paired, "monthly_returns": self.monthly,
                  "allocation_diagnostics": self.diagnostics}
        if not self.screen.empty:
            tables["frozen_training_screen"] = self.screen
        for metric, screen in self.screens.items():
            if metric != "volatility":
                tables[f"frozen_{metric}_screen"] = screen
        if self.predictions is not None:
            # Keep the original filenames for existing single-model consumers.
            tables.update({"loss_predictions" if self.config.prediction_target == "loss" else "upside_predictions": self.predictions.predictions, "model_fits": self.predictions.fits,
                           "model_metrics": self.model_metrics, "probability_calibration": self.predictions.calibration})
        for key, batch in self.prediction_batches.items():
            tables[f"models/{key}/predictions"] = batch.predictions.assign(model_id=key, feature_spec_id=self.config.feature_spec.feature_id)
            tables[f"models/{key}/fits"] = batch.fits
            tables[f"models/{key}/calibration"] = batch.calibration
        for name, table in tables.items():
            yield f"{name}.csv", table.to_csv(index=False).encode()
        if include_weights:
            for name, weights in self.weights.items():
                yield f"weights/{name.replace('::', '__')}.csv", weights.to_csv(index=False).encode()

    def to_zip(self, include_weights=False):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in self._files(include_weights):
                archive.writestr(name, content)
        return buffer.getvalue()

    def save(self, destination, include_weights=False):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        for name, content in self._files(include_weights):
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return destination


class ExperimentRunner:
    def __init__(self, root=PROJECT_ROOT, registry=None, model_registry=None):
        self.root = Path(root).resolve()
        self.registry = default_registry() if registry is None else dict(registry)
        self.model_registry = default_model_registry() if model_registry is None else dict(model_registry)

    def run(self, config=RunConfig(), progress=None):
        if os.environ.get("MLFACTOR_READ_ONLY") == "1":
            raise RuntimeError("Computation is disabled during artifact-only rendering. Compute and save the experiment first.")
        missing = set(config.strategies) - self.registry.keys()
        if missing or any("::" in key for key in config.strategies):
            raise ValueError(f"Unknown or invalid strategy keys: {sorted(missing)}")
        if config.include_ml and set(config.models) - self.model_registry.keys():
            raise ValueError("Unknown prediction model.")
        notify = progress or (lambda message: None)
        specs = [self.registry[key] for key in config.strategies]
        screen_keys = {s.key: s.screen_metric or "volatility" for s in specs if s.needs_screen}
        notify("Preparing shared inputs, date schedule, lagged signals and frozen screens")
        data = prepare_data(self.root, config.evaluation, config.validation_history_months,
                            screen_keys.values(), any(s.needs_risk for s in specs), loader=load_panel,
                            cash_momentum_months=config.cash_momentum_months)
        pools = {key: tuple(frame.loc[frame.selected, "stock_id"]) for key, frame in data.screens.items()}
        context = StrategyContext(data.signals, self.registry, pools.get("volatility", ()),
                                  data.risk_signals, frozen_pools=pools, cash_momentum_months=config.cash_momentum_months)
        batches = {}
        if config.include_ml:
            for key in config.models:
                model_spec = self.model_registry[key]
                pred, fits = predict_model(data.model, config.feature_spec, model_spec, config.refit_months, notify,
                    fit_policy=config.fit_policy, target=config.prediction_target, model_scope=config.model_scope,
                    min_stock_history=config.min_stock_history, min_stock_class=config.min_stock_class)
                batches[key] = evaluate_predictions(pred, data.evaluation_returns, fits, model_spec.label, config.prediction_target)

        weights, books, metric_rows, monthly, audits, cases = {}, {}, [], [], [], []
        benchmark = portfolio_returns(data.evaluation_returns).simple_return
        for spec in specs:
            notify(f"Running {spec.label}")
            base, base_audit = context.allocation(spec.key)
            variants = [(None, identity(base))]
            adjust = loss_probability if config.prediction_target == "loss" else upside_probability
            adjust_options = {} if config.prediction_target == "loss" else {"threshold": config.upside_threshold}
            variants += [(key, adjust(base, batch.predictions, config.adjustment_strength,
                                     allow_cash=spec.allows_cash, **adjust_options))
                         for key, batch in batches.items()]
            for model_id, allocation in variants:
                variant = "baseline" if model_id is None else (
                    "loss_adjusted" if config.prediction_target == "loss" else "upside_adjusted")
                # Legacy IDs remain valid; all consumers use case metadata, not string splitting.
                key = f"{spec.key}::baseline" if model_id is None else (
                    f"{spec.key}::loss_adjusted" if model_id == "logistic" and config.feature_spec == FeatureSpec()
                        and config.prediction_target == "loss" and config.model_scope == "pooled"
                    else f"{spec.key}::{model_id}::{config.feature_spec.feature_id}" + (
                        f"::{variant}::{config.model_scope}" if config.prediction_target != "loss" or config.model_scope != "pooled" else ""))
                case = {
                    "case_id": key, "strategy": spec.key, "label": spec.label, "variant": variant,
                    "model_id": model_id or "none",
                    "model_label": self.model_registry[model_id].label if model_id else "Original",
                    "feature_spec_id": config.feature_spec.feature_id if model_id else "",
                    "feature_label": config.feature_spec.label if model_id else "",
                    "adjustment_id": ("loss_probability" if config.prediction_target == "loss" else "upside_probability") if model_id else "identity",
                    "prediction_target": config.prediction_target if model_id else "none",
                    "model_scope": config.model_scope if model_id else "none",
                    "allows_cash": spec.allows_cash,
                    "cash_momentum_months": config.cash_momentum_months if spec.allows_cash else 0,
                }
                cases.append(case)
                book = (portfolio_with_cash(data.evaluation_returns, allocation) if spec.allows_cash
                        else portfolio_returns(data.evaluation_returns, allocation))
                if not book.index.equals(benchmark.index):
                    raise AssertionError("All cases must share earned-return months.")
                weights[key], books[key] = allocation, book
                held = allocation.loc[allocation.weight.gt(0)]
                audit = held.groupby("date").weight.agg(n_held="size", max_weight="max",
                    effective_n=lambda w: 1 / w.pow(2).sum()).reset_index()
                if spec.allows_cash:
                    # Stock concentration is measured within the invested sleeve.
                    # Cash-only months still need a diagnostic row with zero names.
                    exposure = allocation.groupby("date").weight.sum()
                    audit = exposure.rename("equity_weight").reset_index().merge(audit, on="date", how="left")
                    audit[["n_held", "max_weight", "effective_n"]] = audit[["n_held", "max_weight", "effective_n"]].fillna(0)
                    audit["effective_n"] *= audit.equity_weight.pow(2)
                    audit["cash_weight"] = (1 - audit.equity_weight).clip(lower=0)
                extra = base_audit.drop(columns=["strategy", "n_held", "max_weight", "effective_n"], errors="ignore")
                audit = audit.merge(extra, on="date", validate="one_to_one").assign(**case)
                audits.append(audit)
                row = performance(book.simple_return).to_dict()
                row.update(downside_statistics(book.simple_return))
                active = book.simple_return - benchmark
                row.update({**case, "first_return_month": book.index.min(), "last_return_month": book.index.max(),
                    "avg_one_way_turnover": book.turnover_one_way.iloc[1:].mean(),
                    "avg_names": book.n_held.mean(), "avg_effective_n": audit.effective_n.mean(),
                    "largest_single_weight": audit.max_weight.max(),
                    "fallback_months": int(audit["fallback"].sum()) if "fallback" in audit else 0,
                    "annualized_mean_active_return": active.mean() * 12,
                    "tracking_error": active.std(ddof=1) * np.sqrt(12)})
                metric_rows.append(row)
                if spec.allows_cash:
                    row.update(avg_cash_weight=book.cash_weight.mean(),
                               max_cash_weight=book.cash_weight.max(),
                               cash_only_months=int(book.cash_weight.ge(1 - 1e-10).sum()))
                monthly.append(book.reset_index().assign(**case))
        metrics, case_table = pd.DataFrame(metric_rows), pd.DataFrame(cases)
        pairs = []
        for spec in specs:
            rows = metrics.loc[metrics.strategy.eq(spec.key)]
            baseline = rows.loc[rows.variant.eq("baseline")].iloc[0]
            for _, adjusted in rows.loc[rows.variant.ne("baseline")].iterrows():
                pair = {c: adjusted[c] for c in ["case_id", "strategy", "label", "model_id", "model_label", "feature_spec_id"]}
                for metric in ["cagr", "sharpe", "sortino", "max_drawdown", "annualized_volatility",
                               "avg_one_way_turnover", "largest_single_weight"]:
                    pair[f"baseline_{metric}"] = baseline[metric]
                    pair[f"adjusted_{metric}"] = adjusted[metric]
                    pair[f"change_{metric}"] = adjusted[metric] - baseline[metric]
                pairs.append(pair)
        signals = data.signals if data.risk_signals is None else data.signals.merge(data.risk_signals, on=KEYS, validate="one_to_one")
        manifest = self._manifest(config, data, specs, case_table)
        notify("Comparison ready")
        return RunResult(config, metrics, pd.DataFrame(pairs), pd.concat(monthly, ignore_index=True),
            pd.concat(audits, ignore_index=True), weights, books, signals,
            data.screens.get("volatility", pd.DataFrame()), next(iter(batches.values()), None), manifest,
            data.screens, screen_keys, case_table, batches, data.audit)

    def _manifest(self, config, data, specs, cases):
        import sklearn
        paths = [*Path(__file__).parent.glob("*.py"), PROJECT_ROOT / "part0_data_and_benchmark/pipeline.py",
                 *[PROJECT_ROOT / "0_1_Basic_Strategies" / f"{name}.py" for name in ["strategies", "hybrid", "downside"]]]
        return {
            "schema_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config), "feature_spec": config.feature_spec.metadata(), "data_audit": data.audit,
            **{key: data.audit[key] for key in ["first_snapshot", "first_return_month", "last_return_month", "evaluation_rows"]},
            "source_sha256": {f"mlfactor_long_only_{s}.csv": hashlib.sha256(
                (self.root / f"mlfactor_long_only_{s}.csv").read_bytes()).hexdigest() for s in data.source_splits},
            "code_sha256": {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "strategies": [{"key": s.key, "label": s.label, "description": s.description,
                "allows_cash": s.allows_cash,
                "screen_metric": (s.screen_metric or "volatility") if s.needs_screen else None,
                "explanation": asdict(s.explanation) if s.explanation else None} for s in specs],
            "models": [self.model_registry[key].metadata(config.feature_spec, config.prediction_target) for key in config.models] if config.include_ml else [],
            "model": {"enabled": config.include_ml, "target": "R1M_Usd < 0" if config.prediction_target == "loss" else "R1M_Usd > 0", "refit_months": config.refit_months,
                      "scope": config.model_scope, "min_stock_history": config.min_stock_history,
                      "min_stock_class": config.min_stock_class,
                      "per_stock_fallback": "same-origin pooled model for insufficient history or class counts",
                      "training": config.fit_policy, "tuning": "none", "feature_selection": config.feature_spec.selection},
            "cases": cases.to_dict("records"),
            "assumptions": {"risk_free_monthly": 0, "sortino_target_monthly": 0, "costs": "gross; none deducted",
                "portfolio": ("monthly, long-only; cash permitted for explicitly marked strategies"
                              if any(s.allows_cash for s in specs) else "monthly, long-only, fully invested; no cash"),
                "cash_return_monthly": 0,
                "cash_allocation": "residual total capital minus stock weights; cash included in drift-aware turnover",
                "cash_ml_adjustment": ("stock weight * (1 - strength * probability of loss); released capital stays in cash"
                    if config.prediction_target == "loss" else "threshold boost within original equity budget; original cash preserved"),
                "upside_adjustment": {"threshold": config.upside_threshold,
                    "rule": "raw weight = base * (1 + strength * I(P(up) >= threshold)); normalize to original equity budget"},
                "annual_consistency_screen": {"minimum_complete_years": 3, "minimum_positive_year_fraction": "strictly > 0.5",
                    "years": "complete January--December earned-return years known at origin; flat years nonpositive",
                    "equity_budget": "positive eligible pool stocks / currently available pool stocks"},
                "signal_lookback_months": 12,
                "cash_momentum_months": config.cash_momentum_months,
                "risk_lookback_months": 3, "basket_fraction": .2,
                "screen": "fixed at first evaluation snapshot using realized training labels only",
                "ratio_screen": {"fraction": .2, "minimum_observed_months": 12,
                    "sharpe_risk_free_monthly": 0, "sortino_target_monthly": 0, "undefined_ratios": "excluded"},
                "timing": "labels earned through month-end are available at that rebalance (course convention)",
                "test_status": "test period was inspected earlier; not an untouched holdout"},
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "sklearn": sklearn.__version__},
        }
