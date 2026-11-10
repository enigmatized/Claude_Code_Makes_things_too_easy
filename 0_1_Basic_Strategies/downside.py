"""Three-month volatility with an explicit extra penalty for negative returns."""

import numpy as np
import pandas as pd

from strategies import KEYS, TARGET, return_month

RISK_MONTHS = 3
DOWNSIDE_PENALTY = 1.0
MONTHLY_RISK_FLOOR = 0.01
SIZING_LABELS = {
    "hybrid_volatility_3m_sizing": "Hybrid, momentum / ordinary 3m volatility",
    "hybrid_downside_3m_sizing": "Hybrid, momentum / downside-weighted 3m risk",
}


def build_risk_signals(history, months=RISK_MONTHS, downside_penalty=DOWNSIDE_PENALTY):
    """Use only returns earned by each decision snapshot.

    volatility = sample standard deviation of the last three monthly returns.
    downside = sqrt(mean(min(return, 0)**2)), keeping ALL three observations.
    custom_risk = sqrt(volatility**2 + penalty * downside**2).
    All three measures are annualized with sqrt(12) for consistent units.
    This is an explicit custom risk measure, not a Sharpe or Sortino ratio.
    """
    if not isinstance(months, int) or months < 2:
        raise ValueError("Need at least two monthly observations.")
    if not np.isfinite(downside_penalty) or downside_penalty < 0:
        raise ValueError("Downside penalty must be finite and nonnegative.")
    if history.empty or not set(KEYS + [TARGET]).issubset(history.columns):
        raise ValueError("Need stock-month keys and forward return labels.")
    if history[KEYS].isna().any().any() or history.duplicated(KEYS).any():
        raise ValueError("Missing or duplicate stock-month keys.")
    if not pd.api.types.is_datetime64_any_dtype(history.date) or not history.date.dt.is_month_end.all():
        raise ValueError("Dates must be parsed calendar month-ends.")
    if not np.isfinite(history[TARGET]).all() or (history[TARGET] < -1).any():
        raise ValueError("Known simple returns must be finite and at least -100%.")
    realized = history[KEYS + [TARGET]].assign(earned_month=return_month(history.date))
    grid = realized.pivot(index="earned_month", columns="stock_id", values=TARGET)
    calendar = pd.date_range(history.date.min(), history.date.max(), freq="ME", name="date")
    grid = grid.reindex(calendar)
    variance = grid.rolling(months, min_periods=months).var(ddof=1)
    downside_square = grid.clip(upper=0).pow(2).rolling(months, min_periods=months).mean()
    measures = {
        "risk_history_months": grid.rolling(months, min_periods=1).count().fillna(0),
        "ordinary_volatility": np.sqrt(variance * 12),
        "downside_deviation": np.sqrt(downside_square * 12),
        "downside_weighted_risk": np.sqrt((variance + downside_penalty * downside_square) * 12),
    }
    signals = history[KEYS].copy()
    for name, values in measures.items():
        values.columns.name = "stock_id"
        long = values.stack(future_stack=True).rename(name).reset_index()
        signals = signals.merge(long, on=KEYS, how="left", validate="one_to_one")
    signals["risk_history_months"] = signals.risk_history_months.astype(int)
    signals["risk_eligible"] = signals.risk_history_months.eq(months)
    return signals.sort_values(KEYS).reset_index(drop=True)


def downside_statistics(returns, target_monthly=0.0):
    """Full evaluation-window target downside deviation and Sortino.

    Zeros for returns at/above target remain in the denominator's observation
    count. Annualized Sortino uses sqrt(12), not CAGR divided by deviation.
    An undefined zero-denominator ratio is NaN rather than an infinite score.
    """
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite monthly returns.")
    if not np.isfinite(target_monthly) or target_monthly <= -1 or (values <= -1).any():
        raise ValueError("Target and portfolio simple returns must be above -100%.")
    excess = values - target_monthly
    downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2))
    return {
        "annualized_downside_deviation": downside * np.sqrt(12),
        "sortino": excess.mean() / downside * np.sqrt(12) if downside > 0 else np.nan,
    }


def risk_adjust_weights(base_weights, risk_signals, risk_column, monthly_floor=MONTHLY_RISK_FLOOR, fallback_dates=()):
    """Reweight the SAME momentum names by inverse trailing risk.

    Base momentum weights divided by risk, then normalized, equal positive
    momentum scores divided by risk and normalized. Risk is annualized, so the
    monthly 1% floor is multiplied by sqrt(12) before use. Parent fallback months
    retain their equal-weight allocation instead of inventing a new fallback.
    """
    if risk_column not in {"ordinary_volatility", "downside_weighted_risk"}:
        raise ValueError("Choose ordinary or downside-weighted risk.")
    if not np.isfinite(monthly_floor) or monthly_floor <= 0:
        raise ValueError("A positive finite monthly risk floor is required.")
    if base_weights.empty or not set(KEYS + ["weight"]).issubset(base_weights.columns):
        raise ValueError("Need keyed momentum weights.")
    if base_weights.duplicated(KEYS).any() or risk_signals.duplicated(KEYS).any():
        raise ValueError("Weights and risk signals must have unique keys.")
    if not np.isfinite(base_weights.weight).all() or base_weights.weight.lt(0).any():
        raise ValueError("Base weights must be finite and nonnegative.")
    if not np.allclose(base_weights.groupby("date").weight.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError("Base weights must sum to one each month.")
    merged = base_weights[KEYS + ["weight"]].merge(
        risk_signals[KEYS + [risk_column]], on=KEYS, how="left", validate="one_to_one",
    )
    floor = monthly_floor * np.sqrt(12)
    fallback_dates = set(pd.to_datetime(list(fallback_dates)))
    allocations, audit = [], []
    for date, block in merged.sort_values(KEYS).groupby("date", sort=True):
        held = block.weight.gt(0)
        preserve_fallback = date in fallback_dates
        risks = block.loc[held, risk_column]
        if not preserve_fallback and (not np.isfinite(risks).all() or risks.lt(0).any()):
            raise ValueError("Every selected stock needs finite nonnegative trailing risk.")
        allocation = block[KEYS + ["weight"]].copy()
        if not preserve_fallback:
            scores = block.loc[held, "weight"] / risks.clip(lower=floor)
            scores = scores / scores.max()
            allocation.loc[held, "weight"] = scores / scores.sum()
        positive = allocation.loc[held, "weight"]
        allocations.append(allocation)
        audit.append({
            "date": date, "n_held": int(held.sum()), "effective_n": 1 / positive.pow(2).sum(),
            "max_weight": positive.max(), "risk_floor_hits": int(risks.lt(floor).sum()) if not preserve_fallback else 0,
            "parent_fallback_preserved": preserve_fallback, "risk_column": risk_column,
            "annualized_risk_floor": floor,
        })
    return pd.concat(allocations, ignore_index=True), pd.DataFrame(audit)


def compare_downside(root=None):
    """Add one downside strategy and an ordinary-risk sizing control to the hybrid."""
    from strategies import HERE, load_panel, performance, portfolio_returns
    from hybrid import compare_hybrid

    root = root or HERE.parent
    parent = compare_hybrid(root)
    train, test = load_panel("train", root), load_panel("test", root)
    risk = build_risk_signals(pd.concat([train, test], ignore_index=True))
    risk = risk.loc[risk.date.isin(test.date.unique())].reset_index(drop=True)
    base_name = "training_high_vol_momentum_weighted"
    base = parent["weights"][KEYS + [base_name]].rename(columns={base_name: "weight"})
    parent_audit = parent["diagnostics"].loc[parent["diagnostics"].strategy.eq(base_name)]
    fallback_dates = parent_audit.loc[parent_audit.fallback, "date"]
    weights = base[KEYS].copy()
    books, metric_rows, audit_rows, monthly_rows = {}, [], [], []
    universe = parent["baseline_books"]["equal_weight_universe"].simple_return
    for name, risk_column in zip(SIZING_LABELS, ["ordinary_volatility", "downside_weighted_risk"]):
        allocation, audit = risk_adjust_weights(base, risk, risk_column, fallback_dates=fallback_dates)
        book = portfolio_returns(test, allocation)
        books[name] = book
        weights = weights.merge(allocation.rename(columns={"weight": name}), on=KEYS, validate="one_to_one")
        row = performance(book.simple_return).to_dict()
        active = book.simple_return - universe
        row.update({
            "split": "test", "strategy": name,
            "first_return_month": book.index.min(), "last_return_month": book.index.max(),
            "avg_one_way_turnover": book.turnover_one_way.iloc[1:].mean(),
            "avg_names": book.n_held.mean(), "avg_effective_n": audit.effective_n.mean(),
            "largest_single_weight": audit.max_weight.max(), "fallback_months": len(fallback_dates),
            "annualized_mean_active_return": active.mean() * 12,
            "tracking_error": active.std(ddof=1) * np.sqrt(12),
            "months_beating_universe": active.gt(0).mean(),
        })
        metric_rows.append(row)
        audit_rows.append(audit.assign(strategy=name))
        monthly_rows.append(book.reset_index().assign(strategy=name, split="test"))
    all_books = {**parent["baseline_books"], **parent["books"], **books}
    comparison = pd.concat([parent["comparison"], pd.DataFrame(metric_rows)], ignore_index=True)
    downside_metrics = {name: downside_statistics(book.simple_return) for name, book in all_books.items()}
    for column in ["sortino", "annualized_downside_deviation"]:
        comparison[column] = comparison.strategy.map({name: values[column] for name, values in downside_metrics.items()})
    return {
        "parent": parent, "risk_signals": risk, "weights": weights, "books": books,
        "all_books": all_books, "comparison": comparison,
        "diagnostics": pd.concat(audit_rows, ignore_index=True),
        "monthly": pd.concat(monthly_rows, ignore_index=True),
    }
