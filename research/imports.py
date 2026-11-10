"""Upgrade an earlier CSV report into a complete saved run without fitting models."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .adjustments import loss_probability, upside_probability
from .accounting import portfolio_with_cash
from .data import prepare_data
from .legacy import PROJECT_ROOT, portfolio_returns
from .models import PredictionBatch, event_fields
from .runner import RunResult
from .store import config_from_dict, digest_file
from .strategies import StrategyContext, default_registry


def import_csv_report(store, directory, label=None):
    """Replay existing predictions with verified unchanged allocation rules.

    This is a historical import, not a claim that old predictions were fitted by
    today's model code. Source manifests remain intact; replayed books must match
    every saved monthly return and turnover before the run is catalogued.
    """
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("Only case-aware schema-2 CSV reports can be imported.")
    for name, digest in manifest["source_sha256"].items():
        if digest_file(store.project_root / name) != digest:
            raise ValueError(f"Source data no longer match this saved report: {name}")
    allocation_files = ["research/data.py", "research/strategies.py", "research/screens.py", "research/legacy.py",
        "part0_data_and_benchmark/pipeline.py", *[f"0_1_Basic_Strategies/{name}.py" for name in ["strategies", "hybrid", "downside"]]]
    if any(s.get("allows_cash", False) for s in manifest.get("strategies", [])):
        allocation_files += ["research/cash_strategy.py", "research/accounting.py", "research/adjustments.py"]
    for name in allocation_files:
        if digest_file(PROJECT_ROOT / name) != manifest["code_sha256"].get(name):
            raise ValueError(f"Cannot replay changed allocation code: {name}")

    def read(name, dates=()):
        return pd.read_csv(directory / name, parse_dates=list(dates), float_precision="round_trip")

    config = config_from_dict(manifest["config"])
    cases = read("cases.csv").fillna({"feature_spec_id": "", "feature_label": ""})
    registry = default_registry()
    screen_keys = {key: registry[key].screen_metric or "volatility" for key in config.strategies if registry[key].needs_screen}
    data = prepare_data(store.project_root, config.evaluation, config.validation_history_months,
                        screen_keys.values(), any(registry[key].needs_risk for key in config.strategies),
                        cash_momentum_months=config.cash_momentum_months)
    pools = {key: tuple(frame.loc[frame.selected, "stock_id"]) for key, frame in data.screens.items()}
    context = StrategyContext(data.signals, registry, pools.get("volatility", ()), data.risk_signals,
                              frozen_pools=pools, cash_momentum_months=config.cash_momentum_months)
    global_metrics = read("model_metrics.csv") if config.include_ml else pd.DataFrame()
    batches = {}
    for key in config.models if config.include_ml else ():
        folder = f"models/{key}"
        pred = read(f"{folder}/predictions.csv", ["date", "fit_snapshot", "trained_through_return_month"])
        # The original predictor contract excludes the export-only identifiers.
        pred = pred.drop(columns=["model_id", "feature_spec_id"], errors="ignore")
        fits = read(f"{folder}/fits.csv", ["fit_snapshot", "trained_through_return_month"])
        metrics = pd.concat([global_metrics.loc[global_metrics.model_id.eq(key)],
                             global_metrics.loc[global_metrics.model_id.eq(event_fields(config.prediction_target)["control_id"])]], ignore_index=True)
        if len(metrics) != 2:
            raise ValueError("A saved model must have predictive metrics and a historical loss-rate control.")
        batches[key] = PredictionBatch(pred, fits, metrics.drop(columns="model_id"), read(f"{folder}/calibration.csv"))
    monthly = read("monthly_returns.csv", ["return_month", "snapshot_date"])
    book_columns = ["snapshot_date", "n_assets", "n_held", "simple_return", "turnover_one_way", "log_return", "wealth", "drawdown"]
    weights, books = {}, {}
    for _, case in cases.iterrows():
        base, _ = context.allocation(case.strategy)
        cash_enabled = registry[case.strategy].allows_cash
        adjust = loss_probability if config.prediction_target == "loss" else upside_probability
        options = {} if config.prediction_target == "loss" else {"threshold": config.upside_threshold}
        allocation = base if case.variant == "baseline" else adjust(
            base, batches[case.model_id].predictions, config.adjustment_strength, allow_cash=cash_enabled, **options)
        replayed = (portfolio_with_cash(data.evaluation_returns, allocation) if cash_enabled
                    else portfolio_returns(data.evaluation_returns, allocation))
        columns = book_columns + (["equity_weight", "cash_weight"] if cash_enabled else [])
        saved = monthly.loc[monthly.case_id.eq(case.case_id)].set_index("return_month")[columns]
        if not replayed.index.equals(saved.index):
            raise ValueError("Saved and replayed return calendars differ.")
        for column in ["simple_return", "turnover_one_way", "wealth", "drawdown"] + (["equity_weight", "cash_weight"] if cash_enabled else []):
            np.testing.assert_allclose(replayed[column], saved[column], rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=case.case_id)
        weights[case.case_id], books[case.case_id] = allocation, saved
    manifest["saved_result_import"] = {"report": str(directory), "manifest_sha256": digest_file(directory / "manifest.json"),
        "method": "saved probabilities replayed; allocation code/source hashes and monthly books verified; no model fitting"}
    signals = data.signals if data.risk_signals is None else data.signals.merge(data.risk_signals, on=["date", "stock_id"], validate="one_to_one")
    metrics = read("strategy_comparison.csv", ["first_return_month", "last_return_month"]).fillna({"feature_spec_id": "", "feature_label": ""})
    result = RunResult(config, metrics, read("paired_changes.csv"), monthly,
        read("allocation_diagnostics.csv", ["date"]), weights, books, signals,
        data.screens.get("volatility", pd.DataFrame()), next(iter(batches.values()), None), manifest,
        data.screens, screen_keys, cases, batches, json.loads((directory / "data_audit.json").read_text()))
    return store.save(result, label=label or f"Imported: {directory.name}", origin="imported_csv")
