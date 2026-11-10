"""Register weight-producing functions, with dependencies computed once per run."""

from dataclasses import dataclass, field, replace
from typing import Callable

import pandas as pd
from .explanations import StrategyExplanation, explanation_for
from .cash_strategy import STRATEGY_KEY, consistency_momentum_weights

from .legacy import (
    HYBRID_LABELS, KEYS, LABELS, SIZING_LABELS, hybrid_weights,
    risk_adjust_weights, strategy_weights,
)


@dataclass(frozen=True)
class StrategySpec:
    key: str
    label: str
    description: str
    build_weights: Callable
    needs_screen: bool = False
    needs_risk: bool = False
    screen_metric: str | None = None
    explanation: StrategyExplanation | None = None
    allows_cash: bool = False


@dataclass
class StrategyContext:
    """Only available-at-snapshot signals are exposed to allocation functions.

    Forward returns and model-fitting data do not belong in this object.
    Adapters operate on whole keyed panels but inherit the existing lag tests.
    New functions must likewise make each month's weights depend only on inputs
    available at that snapshot, never on later signal rows.
    """

    signals: pd.DataFrame
    registry: dict[str, StrategySpec]
    frozen_ids: tuple = ()
    risk_signals: pd.DataFrame | None = None
    basket_fraction: float = 0.20
    frozen_pools: dict[str, tuple] = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)
    cash_momentum_months: int = 12

    def allocation(self, key):
        if key not in self._cache:
            self._cache[key] = self.registry[key].build_weights(self)
        weights, audit = self._cache[key]
        # A new adjustment must not mutate the baseline used by another variant.
        return weights.copy(), audit.copy()


def _basic(key):
    def allocate(context):
        return strategy_weights(context.signals, key, context.basket_fraction)
    return allocate


def _hybrid(key):
    def allocate(context):
        return hybrid_weights(context.signals, context.frozen_ids, key, context.basket_fraction)
    return allocate


def _risk(column):
    def allocate(context):
        base, audit = context.allocation("training_high_vol_momentum_weighted")
        fallback = audit.loc[audit.fallback, "date"]
        weights, risk_audit = risk_adjust_weights(
            base, context.risk_signals, column, fallback_dates=fallback,
        )
        return weights, audit.merge(risk_audit[["date", "risk_floor_hits"]], on="date", validate="one_to_one")
    return allocate


def _ratio_momentum(key, metric):
    def allocate(context):
        weights, audit = hybrid_weights(
            context.signals, context.frozen_pools[metric],
            "training_high_vol_momentum_weighted", context.basket_fraction,
        )
        audit["strategy"] = key
        audit["screen_metric"] = metric
        return weights, audit
    return allocate


def _consistency_cash(context):
    signals = context.signals
    if context.cash_momentum_months != 12:
        columns = ["trailing_momentum", "trailing_volatility", "history_months", "eligible"]
        signals = signals[KEYS + [f"cash_{c}" for c in columns]].rename(
            columns={f"cash_{c}": c for c in columns})
    weights, audit = consistency_momentum_weights(signals, context.frozen_pools["annual_consistency"])
    audit["momentum_lookback_months"] = context.cash_momentum_months
    return weights, audit


def default_registry():
    """An ordinary dictionary is the strategy registry; no subclass is needed."""
    descriptions = {
        "equal_weight_universe": "Equal capital in every available stock.",
        "equal_weight_eligible": "Equal capital after requiring 12 consecutive known returns.",
        "high_volatility": "Highest 20% by trailing 12-month volatility, equally weighted.",
        "low_volatility": "Lowest 20% by trailing 12-month volatility, equally weighted.",
        "momentum_equal_weight": "Top positive 12-month momentum basket, equally weighted.",
        "momentum_weighted": "Same positive-momentum basket, weighted by momentum.",
    }
    registry = {
        key: StrategySpec(key, label, descriptions[key], _basic(key))
        for key, label in LABELS.items()
    }
    for key, label in HYBRID_LABELS.items():
        registry[key] = StrategySpec(
            key, label, "Frozen high-volatility screen fitted before this evaluation; monthly allocation inside it.",
            _hybrid(key), needs_screen=True,
        )
    for key, label in SIZING_LABELS.items():
        column = "ordinary_volatility" if key == "hybrid_volatility_3m_sizing" else "downside_weighted_risk"
        registry[key] = StrategySpec(
            key, label, "Hybrid momentum weights divided by trailing 3-month risk, with a 1% monthly risk floor.",
            _risk(column), needs_screen=True, needs_risk=True,
        )
    for metric, name in [("sharpe", "Sharpe"), ("sortino", "Sortino")]:
        key = f"training_{metric}_momentum_weighted"
        registry[key] = StrategySpec(
            key, f"Training {name} pool + momentum",
            f"Freeze the top 20% by historical {name} before evaluation. Each month, select the strongest positive 12-month momentum inside that pool (up to 20% of eligible pool stocks) and weight by momentum.",
            _ratio_momentum(key, metric), needs_screen=True, screen_metric=metric,
        )
    registry[STRATEGY_KEY] = StrategySpec(
        STRATEGY_KEY, "Annual consistency + positive momentum + cash",
        "Freeze stocks with at least three complete historical calendar years and a strict majority positive. "
        "Buy positive momentum over the configured window (default 12 months); inactive pool slots stay in cash.",
        _consistency_cash,
        needs_screen=True, screen_metric="annual_consistency", allows_cash=True)
    return {key: replace(spec, explanation=explanation_for(key)) for key, spec in registry.items()}
