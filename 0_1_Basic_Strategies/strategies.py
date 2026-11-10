"""Simple historical-return strategies using Part 0's unchanged accounting."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FOUNDATION = HERE.parent / "part0_data_and_benchmark"
if str(FOUNDATION) not in sys.path:
    sys.path.insert(0, str(FOUNDATION))
from pipeline import KEYS, TARGET, load_panel, performance, portfolio_returns, return_month

LOOKBACK_MONTHS = 12
BASKET_FRACTION = 0.20
LABELS = {
    "equal_weight_universe": "Equal-weight universe",
    "equal_weight_eligible": "Equal-weight eligible stocks",
    "high_volatility": "High volatility, equal weight",
    "low_volatility": "Low volatility, equal weight",
    "momentum_equal_weight": "Positive momentum, equal weight",
    "momentum_weighted": "Positive momentum, score weights",
}


def build_signals(history, lookback=LOOKBACK_MONTHS):
    """Map labels to their earned months BEFORE building trailing histories.

    At snapshot t, only returns earned through t may enter a signal. The label
    stored on row t is earned in t+1 and therefore cannot enter its own signal.
    Reindex to calendar months before rolling: a missing month is never treated
    as adjacent to an earlier available observation. No missing returns are filled.
    The result contains keys and signals, without the forward target.
    """
    if not isinstance(lookback, int) or lookback < 2:
        raise ValueError("lookback must be an integer of at least two months")
    required = set(KEYS + [TARGET])
    if not required.issubset(history.columns) or history.empty:
        raise ValueError("Need stock-month keys and forward returns.")
    history = history[KEYS + [TARGET]].copy().sort_values(KEYS)
    if history[KEYS].isna().any().any() or history.duplicated(KEYS).any():
        raise ValueError("Missing or duplicated stock-month keys.")
    if not pd.api.types.is_datetime64_any_dtype(history.date) or not history.date.dt.is_month_end.all():
        raise ValueError("Dates must be parsed calendar month-ends.")
    if not np.isfinite(history[TARGET]).all() or (history[TARGET] < -1).any():
        raise ValueError("Need finite simple returns of at least -100%.")

    realized = history.assign(earned_month=return_month(history.date))
    realized = realized.pivot(index="earned_month", columns="stock_id", values=TARGET)
    # Include every decision month, even if all stocks lack a preceding label.
    calendar = pd.date_range(history.date.min(), history.date.max(), freq="ME", name="date")
    realized = realized.reindex(calendar)
    rolling = realized.rolling(lookback, min_periods=lookback)
    volatility = rolling.std(ddof=1) * np.sqrt(12)
    # Direct gross-return products also handle a historical -100% stock return.
    momentum = (1 + realized).rolling(lookback, min_periods=lookback).apply(np.prod, raw=True) - 1
    counts = realized.rolling(lookback, min_periods=1).count().fillna(0)

    def long_form(values, name):
        values.columns.name = "stock_id"
        return values.stack(future_stack=True).rename(name).reset_index()

    signals = history[KEYS].copy()
    for values, name in [(volatility, "trailing_volatility"), (momentum, "trailing_momentum"), (counts, "history_months")]:
        signals = signals.merge(long_form(values, name), on=KEYS, how="left", validate="one_to_one")
    signals["history_months"] = signals.history_months.astype(int)
    signals["eligible"] = (
        signals.history_months.eq(lookback)
        & np.isfinite(signals.trailing_volatility)
        & np.isfinite(signals.trailing_momentum)
    )
    return signals.sort_values(KEYS).reset_index(drop=True)


def strategy_weights(signals, strategy, basket_fraction=BASKET_FRACTION):
    """Return full keyed weights and a monthly explanation of each allocation.

    Volatility baskets select ceil(fraction * eligible stocks). Momentum selects
    up to that many POSITIVE-momentum stocks. Ties resolve by lower stock_id.
    If momentum has no positive candidates, hold all eligible stocks equally.
    If there are no eligible stocks, hold the current universe equally. Initial
    warm-up months are later excluded from every strategy's comparison window.
    """
    if strategy not in LABELS:
        raise ValueError(f"Unknown strategy: {strategy}")
    if not 0 < basket_fraction <= 1:
        raise ValueError("basket_fraction must lie in (0, 1]")
    required = set(KEYS + ["eligible", "trailing_volatility", "trailing_momentum"])
    if not required.issubset(signals.columns) or signals.empty or signals.duplicated(KEYS).any():
        raise ValueError("Need unique keyed signals and eligibility.")
    allocations, diagnostics = [], []
    for date, block in signals.sort_values(KEYS).groupby("date", sort=True):
        eligible = block.loc[block.eligible]
        k = max(1, int(np.ceil(basket_fraction * len(eligible))))
        positive = eligible.loc[eligible.trailing_momentum > 0]
        reason = ""
        if strategy == "equal_weight_universe":
            selected = block
        elif eligible.empty:
            selected = block
            reason = "no eligible history: equal-weight current universe"
        elif strategy == "equal_weight_eligible":
            selected = eligible
        elif strategy in {"high_volatility", "low_volatility"}:
            selected = eligible.sort_values(
                ["trailing_volatility", "stock_id"],
                ascending=[strategy == "low_volatility", True],
            ).head(k)
        elif positive.empty:
            selected = eligible
            reason = "no positive momentum: equal-weight eligible stocks"
        else:
            selected = positive.sort_values(["trailing_momentum", "stock_id"], ascending=[False, True]).head(k)

        weights = block[KEYS].copy()
        weights["weight"] = 0.0
        if strategy == "momentum_weighted" and not reason:
            scores = selected.trailing_momentum
            # Rescaling prevents a large score sum from overflowing.
            scaled = scores / scores.max()
            allocation = scaled / scaled.sum()
        else:
            allocation = pd.Series(1 / len(selected), index=selected.index)
        weights.loc[selected.index, "weight"] = allocation
        allocations.append(weights)
        nonzero = weights.loc[weights.weight > 0, "weight"]
        diagnostics.append({
            "date": date, "strategy": strategy, "n_universe": len(block),
            "n_eligible": len(eligible), "n_positive_momentum": len(positive),
            "basket_target": k, "n_held": len(nonzero),
            "effective_n": 1 / nonzero.pow(2).sum(), "max_weight": nonzero.max(),
            "fallback": bool(reason), "fallback_reason": reason,
        })
    return pd.concat(allocations, ignore_index=True), pd.DataFrame(diagnostics)


def run_comparison(root=None, lookback=LOOKBACK_MONTHS, basket_fraction=BASKET_FRACTION):
    """Fixed-rule training and test comparisons; no parameter search or ML.

    Earlier test returns can update later test signals after they are realized.
    Training provides history for the first test allocation. Later observations
    cannot influence an earlier weight, as covered by the causality tests.
    """
    root = Path(root) if root else HERE.parent
    panels = {split: load_panel(split, root) for split in ["train", "test"]}
    history = pd.concat(panels.values(), ignore_index=True).sort_values(KEYS)
    signals = build_signals(history, lookback)
    ready = signals.loc[signals.eligible, "date"]
    if ready.empty:
        raise ValueError("No stock has a complete history window.")
    start = ready.min()
    # Every strategy and both equal-weight controls share the same warm-up cut.
    signals = signals.loc[signals.date >= start].reset_index(drop=True)
    weights, diagnostics = {}, []
    wide_weights = signals[KEYS].copy()
    for strategy in LABELS:
        weight, audit = strategy_weights(signals, strategy, basket_fraction)
        weights[strategy] = weight
        diagnostics.append(audit)
        wide_weights = wide_weights.merge(weight.rename(columns={"weight": strategy}), on=KEYS, validate="one_to_one")
    diagnostics = pd.concat(diagnostics, ignore_index=True)
    books, metric_rows, monthly_frames, active_frames = {}, [], [], []
    for split, panel in panels.items():
        panel = panel.loc[panel.date >= start].copy()
        dates = panel.date.unique()
        books[split] = {}
        for strategy, weight in weights.items():
            book = portfolio_returns(panel, weight.loc[weight.date.isin(dates)])
            books[split][strategy] = book
        benchmark = books[split]["equal_weight_universe"].simple_return
        for strategy, book in books[split].items():
            if not book.index.equals(benchmark.index):
                raise AssertionError("Strategies must share earned-return months.")
            row = performance(book.simple_return).to_dict()
            audit = diagnostics.loc[diagnostics.strategy.eq(strategy) & diagnostics.date.isin(dates)]
            difference = book.simple_return - benchmark
            row.update({
                "split": split, "strategy": strategy,
                "first_return_month": book.index.min(), "last_return_month": book.index.max(),
                "avg_one_way_turnover": book.turnover_one_way.iloc[1:].mean(),
                "avg_names": book.n_held.mean(), "avg_effective_n": audit.effective_n.mean(),
                "largest_single_weight": audit.max_weight.max(), "fallback_months": int(audit.fallback.sum()),
                "annualized_mean_active_return": difference.mean() * 12,
                "tracking_error": difference.std(ddof=1) * np.sqrt(12),
                "months_beating_universe": (difference > 0).mean() if strategy != "equal_weight_universe" else np.nan,
            })
            metric_rows.append(row)
            monthly_frames.append(book.reset_index().assign(split=split, strategy=strategy))
            active_frames.append(pd.DataFrame({
                "return_month": book.index, "split": split, "strategy": strategy,
                "active_return": difference.to_numpy(),
                "relative_wealth": book.wealth.to_numpy() / books[split]["equal_weight_universe"].wealth.to_numpy(),
            }))
    return {
        "signals": signals, "weights": wide_weights, "diagnostics": diagnostics,
        "books": books, "metrics": pd.DataFrame(metric_rows),
        "monthly": pd.concat(monthly_frames, ignore_index=True),
        "active": pd.concat(active_frames, ignore_index=True),
        "config": {"lookback_months": lookback, "basket_fraction": basket_fraction},
    }
