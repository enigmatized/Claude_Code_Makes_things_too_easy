"""Freeze a training volatility universe, then allocate by monthly momentum."""

import numpy as np
import pandas as pd

from strategies import (
    HERE, KEYS, TARGET, BASKET_FRACTION, LOOKBACK_MONTHS, build_signals,
    load_panel, performance, portfolio_returns, return_month,
    run_comparison, strategy_weights,
)

HYBRID_LABELS = {
    "training_high_vol_equal_weight": "Training high-vol universe, equal weight",
    "training_high_vol_momentum_equal": "Training high vol + momentum, equal weight",
    "training_high_vol_momentum_weighted": "Training high vol + momentum, score weights",
}
UNDERLYING_RULES = dict(zip(HYBRID_LABELS, [
    "equal_weight_eligible", "momentum_equal_weight", "momentum_weighted",
]))


def rank_training_volatility(training, fraction=BASKET_FRACTION, min_months=LOOKBACK_MONTHS, asof=None):
    """Select the most volatile stocks using only returns realized by asof.

    Full training-history sample volatility defines membership, not future
    survival, test returns, or monthly rolling test volatility. At least
    min_months observed training returns are required. Ties use lower stock_id.
    """
    if not 0 < fraction <= 1 or not isinstance(min_months, int) or min_months < 2:
        raise ValueError("Need a fraction in (0, 1] and at least two observations.")
    if training.empty or not set(KEYS + [TARGET]).issubset(training.columns):
        raise ValueError("Need training keys and forward returns.")
    if training[KEYS].isna().any().any() or training.duplicated(KEYS).any():
        raise ValueError("Missing or duplicate training stock-month keys.")
    if not pd.api.types.is_datetime64_any_dtype(training.date) or not training.date.dt.is_month_end.all():
        raise ValueError("Training dates must be parsed month-ends.")
    known = training[KEYS + [TARGET]].copy()
    known["earned_month"] = return_month(known.date)
    asof = pd.Timestamp(asof) if asof is not None else known.earned_month.max()
    known = known.loc[known.earned_month.le(asof)]
    if not np.isfinite(known[TARGET]).all() or (known[TARGET] < -1).any():
        raise ValueError("Known returns must be finite and at least -100%.")
    ranking = known.groupby("stock_id").agg(
        training_months=(TARGET, "size"),
        training_volatility=(TARGET, "std"),
        first_training_return=("earned_month", "min"),
        last_training_return=("earned_month", "max"),
    ).reset_index()
    ranking = ranking.loc[ranking.training_months.ge(min_months)].copy()
    ranking["training_volatility"] *= np.sqrt(12)
    ranking = ranking.sort_values(["training_volatility", "stock_id"], ascending=[False, True]).reset_index(drop=True)
    if ranking.empty:
        raise ValueError("No stock has enough known training observations.")
    ranking["training_rank"] = np.arange(1, len(ranking) + 1)
    ranking["selected"] = ranking.training_rank.le(int(np.ceil(fraction * len(ranking))))
    ranking["screen_asof"] = asof
    return ranking


def hybrid_weights(signals, frozen_ids, strategy, basket_fraction=BASKET_FRACTION):
    """Apply an existing allocation rule strictly inside the frozen universe.

    Stocks outside the training screen always have zero weight, even during
    fallback. If no screened stock is currently present, fail explicitly rather
    than expanding the universe. With no eligible history, equal-weight available
    screened stocks; with no positive momentum, equal-weight eligible screened
    stocks. All fallbacks are recorded by the underlying rule.
    """
    if strategy not in UNDERLYING_RULES:
        raise ValueError("Unknown hybrid strategy.")
    frozen_ids = set(frozen_ids)
    restricted = signals.loc[signals.stock_id.isin(frozen_ids)].copy()
    missing_dates = set(signals.date.unique()) - set(restricted.date.unique())
    if signals.empty or not frozen_ids or missing_dates:
        raise ValueError("No frozen-universe stock available in one or more months; cash is not modeled.")
    selected_weights, audit = strategy_weights(restricted, UNDERLYING_RULES[strategy], basket_fraction)
    weights = signals[KEYS].merge(selected_weights, on=KEYS, how="left", validate="one_to_one")
    # These are explicitly unheld stocks, not missing return observations.
    weights["weight"] = weights.weight.fillna(0)
    audit["strategy"] = strategy
    audit = audit.rename(columns={"n_universe": "n_available_frozen"})
    audit["fallback_reason"] = audit.fallback_reason.str.replace(
        "current universe", "available frozen universe", regex=False,
    ).str.replace("eligible stocks", "eligible frozen stocks", regex=False)
    audit["frozen_universe_size"] = len(frozen_ids)
    if weights.loc[~weights.stock_id.isin(frozen_ids), "weight"].ne(0).any():
        raise AssertionError("Hybrid allocation escaped the training universe.")
    return weights, audit


def evaluate_hybrid(test_panel, test_signals, ranking, basket_fraction=BASKET_FRACTION):
    """Evaluate a previously fitted training screen in later months only."""
    if test_panel.date.min() < ranking.screen_asof.max():
        raise ValueError("Do not apply a final training screen to earlier training months.")
    frozen = ranking.loc[ranking.selected, "stock_id"].tolist()
    weights = test_signals[KEYS].copy()
    books, metrics, audits, monthly = {}, [], [], []
    universe = portfolio_returns(test_panel).simple_return
    for name in HYBRID_LABELS:
        allocation, audit = hybrid_weights(test_signals, frozen, name, basket_fraction)
        book = portfolio_returns(test_panel, allocation)
        weights = weights.merge(allocation.rename(columns={"weight": name}), on=KEYS, validate="one_to_one")
        books[name] = book
        row = performance(book.simple_return).to_dict()
        difference = book.simple_return - universe
        row.update({
            "split": "test", "strategy": name,
            "first_return_month": book.index.min(), "last_return_month": book.index.max(),
            "avg_one_way_turnover": book.turnover_one_way.iloc[1:].mean(),
            "avg_names": book.n_held.mean(), "avg_effective_n": audit.effective_n.mean(),
            "largest_single_weight": audit.max_weight.max(), "fallback_months": int(audit.fallback.sum()),
            "annualized_mean_active_return": difference.mean() * 12,
            "tracking_error": difference.std(ddof=1) * np.sqrt(12),
            "months_beating_universe": (difference > 0).mean(),
        })
        metrics.append(row)
        audits.append(audit)
        monthly.append(book.reset_index().assign(strategy=name, split="test"))
    return {"books": books, "metrics": pd.DataFrame(metrics), "weights": weights,
            "diagnostics": pd.concat(audits, ignore_index=True), "monthly": pd.concat(monthly, ignore_index=True)}


def compare_hybrid(root=None, ranking=None, basket_fraction=BASKET_FRACTION):
    """Return existing baselines and the new test-period hybrid; no file writes.

    The caller may supply an already saved training ranking. Original baseline
    rules and output files are unchanged. No backtest on the final screen's own
    training window is produced, since that would use future membership.
    """
    root = root or HERE.parent
    train = load_panel("train", root)
    if ranking is None:
        ranking = rank_training_volatility(train)
    test = load_panel("test", root)
    combined = pd.concat([train, test], ignore_index=True)
    all_signals = build_signals(combined)
    test_signals = all_signals.loc[all_signals.date.isin(test.date.unique())].reset_index(drop=True)
    result = evaluate_hybrid(test, test_signals, ranking, basket_fraction)
    baseline = run_comparison(root)
    result["baseline_books"] = baseline["books"]["test"]
    result["comparison"] = pd.concat([
        baseline["metrics"].loc[baseline["metrics"].split.eq("test")], result["metrics"],
    ], ignore_index=True)
    result["ranking"] = ranking.copy()
    result["signals"] = test_signals
    return result
