"""Shared preparation of date schedules, available inputs, labels and screens."""
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from .legacy import FEATURES, KEYS, TARGET, build_risk_signals, build_signals, load_panel, rank_training_volatility, return_month
from .screens import rank_training_ratio
from .cash_strategy import annual_consistency_screen


@dataclass
class ModelData:
    inputs: pd.DataFrame
    labels: pd.DataFrame
    evaluation_inputs: pd.DataFrame

    def training_slice(self, asof, columns):
        """Join labels only AFTER filtering on when their outcomes become known."""
        known = self.labels.loc[self.labels.label_available_at.le(pd.Timestamp(asof))]
        joined = self.inputs[KEYS + list(columns)].merge(
            known[KEYS + [TARGET, "earned_month", "label_available_at"]],
            on=KEYS, how="inner", validate="one_to_one", sort=False)
        if joined.date.nunique() < 24:
            raise ValueError("Need at least 24 months of realized training labels before prediction.")
        return joined

    def fit_schedule(self, refit_months=12):
        if not isinstance(refit_months, int) or refit_months < 1:
            raise ValueError("refit_months must be a positive integer.")
        months = pd.DatetimeIndex(self.evaluation_inputs.date.unique()).sort_values()
        if len(months) == 0 or not months.to_period("M").equals(pd.period_range(months.min(), months.max(), freq="M")):
            raise ValueError("Prediction months must be nonempty and consecutive.")
        for offset in range(0, len(months), refit_months):
            yield months[offset], months[offset:offset + refit_months]


def model_data(history, evaluation_inputs):
    """Separate future returns from predictors; also supports the original API."""
    for frame in [history, evaluation_inputs]:
        if frame.empty or not set(KEYS).issubset(frame):
            raise ValueError("Need nonempty keyed model inputs.")
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError("Need unique, nonmissing stock-month keys.")
        if not pd.api.types.is_datetime64_any_dtype(frame.date) or not frame.date.dt.is_month_end.all():
            raise ValueError("Need parsed month-end dates.")
    inputs = history.drop(columns=[TARGET]).copy()
    labels = history[KEYS + [TARGET]].copy()
    labels["earned_month"] = return_month(labels.date)
    labels["label_available_at"] = labels.earned_month
    return ModelData(inputs, labels, evaluation_inputs.drop(columns=[TARGET], errors="ignore").copy())


@dataclass
class PreparedData:
    model: ModelData
    evaluation_returns: pd.DataFrame
    signals: pd.DataFrame
    risk_signals: pd.DataFrame | None
    screens: dict[str, pd.DataFrame]
    source_splits: tuple[str, ...]
    audit: dict = field(default_factory=dict)

    @property
    def decision_start(self):
        return self.evaluation_returns.date.min()


def prepare_data(root, evaluation="training_validation", validation_history_months=60,
                 screen_metrics=(), needs_risk=False, loader=None, cash_momentum_months=12):
    """Prepare once; training validation never opens the test CSV."""
    loader = load_panel if loader is None else loader
    root = Path(root)
    train = loader("train", root)
    if evaluation == "training_validation":
        dates = pd.DatetimeIndex(train.date.unique()).sort_values()
        if not isinstance(validation_history_months, int) or validation_history_months < 24:
            raise ValueError("Use at least 24 initial training months.")
        if len(dates) - validation_history_months < 2:
            raise ValueError("Need at least two evaluation months after the initial history.")
        panel = train.loc[train.date.ge(dates[validation_history_months])].copy()
        history, splits = train, ("train",)
    elif evaluation == "test":
        panel = loader("test", root)
        if train.date.max() >= panel.date.min():
            raise ValueError("Training snapshots must precede test snapshots.")
        history = pd.concat([train, panel], ignore_index=True).sort_values(KEYS).reset_index(drop=True)
        splits = ("train", "test")
    else:
        raise ValueError("Choose training_validation or test.")
    start = panel.date.min()
    signals = build_signals(history)
    if cash_momentum_months != 12 and "annual_consistency" in screen_metrics:
        shorter = build_signals(history, lookback=cash_momentum_months)
        columns = ["trailing_momentum", "trailing_volatility", "history_months", "eligible"]
        shorter = shorter[KEYS + columns].rename(columns={c: f"cash_{c}" for c in columns})
        signals = signals.merge(shorter, on=KEYS, validate="one_to_one")
    signals = signals.loc[signals.date.isin(panel.date.unique())].reset_index(drop=True)
    screens = {metric: (annual_consistency_screen(train, asof=start) if metric == "annual_consistency"
                        else rank_training_volatility(train, asof=start) if metric == "volatility"
                        else rank_training_ratio(train, metric, asof=start)) for metric in sorted(set(screen_metrics))}
    risk = None
    if needs_risk:
        risk = build_risk_signals(history)
        risk = risk.loc[risk.date.isin(panel.date.unique())].reset_index(drop=True)
    models = model_data(history, panel[KEYS + FEATURES])
    initial = models.training_slice(start, FEATURES)
    audit = {
        "source_splits": list(splits), "training_source_rows": len(train),
        "training_source_stocks": int(train.stock_id.nunique()), "input_rows": len(history),
        "evaluation_rows": len(panel), "evaluation_stocks": int(panel.stock_id.nunique()),
        "evaluation_months": int(panel.date.nunique()), "first_snapshot": str(start.date()),
        "last_snapshot": str(panel.date.max().date()),
        "first_return_month": str(return_month([start])[0].date()),
        "last_return_month": str(return_month([panel.date.max()])[0].date()),
        "initial_known_label_rows": len(initial),
        "initial_labels_available_through": str(initial.label_available_at.max().date()),
        "source_features": FEATURES, "missing_input_cells": int(models.inputs[FEATURES].isna().sum().sum()),
        "label_rule": "row t return is earned and assumed available at t+1 month-end",
        "missing_returns": "never imputed; gaps break rolling eligibility",
        "cash_momentum_months": cash_momentum_months,
        "fitted_screens": {key: {"asof": str(frame.screen_asof.iloc[0].date()), "pool_size": int(frame.selected.sum())}
                           for key, frame in screens.items()},
    }
    return PreparedData(models, panel[KEYS + [TARGET]].copy(), signals, risk, screens, splits, audit)
