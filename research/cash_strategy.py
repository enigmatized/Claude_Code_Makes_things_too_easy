"""Annual consistency screen and positive-momentum allocation with cash."""
import numpy as np
import pandas as pd

from .legacy import KEYS, TARGET, return_month

STRATEGY_KEY = "annual_consistency_momentum_cash"


def annual_consistency_screen(training, asof, min_years=3):
    """Freeze a pool with >50% positive complete calendar years, using known labels.

    Annual returns compound January--December earned returns. Incomplete years
    do not count; flat years count as nonpositive. An empty pool is valid.
    """
    asof = pd.Timestamp(asof)
    if pd.isna(asof) or not asof.is_month_end or asof != asof.normalize():
        raise ValueError("Screen cutoff must be a normalized month-end.")
    if not isinstance(min_years, int) or min_years < 3:
        raise ValueError("Require at least three complete calendar years.")
    if training.empty or not set(KEYS + [TARGET]).issubset(training):
        raise ValueError("Need training keys and monthly returns.")
    if training[KEYS].isna().any().any() or training.duplicated(KEYS).any():
        raise ValueError("Training keys must be unique and nonmissing.")
    if not pd.api.types.is_datetime64_any_dtype(training.date) or not training.date.dt.is_month_end.all():
        raise ValueError("Training dates must be parsed month-ends.")
    known = training[KEYS + [TARGET]].assign(earned_month=return_month(training.date))
    known = known.loc[known.earned_month.le(asof)].copy()
    if not np.isfinite(known[TARGET]).all() or known[TARGET].lt(-1).any():
        raise ValueError("Known returns must be finite and at least -100%.")
    known["year"] = known.earned_month.dt.year
    known["gross"] = 1 + known[TARGET]
    years = known.groupby(["stock_id", "year"]).agg(
        months=(TARGET, "size"), gross=("gross", "prod"),
        last_return=("earned_month", "max")).reset_index()
    years = years.loc[years.months.eq(12)].copy()
    years["positive"] = years.gross.gt(1)
    years["negative"] = years.gross.lt(1)
    years["flat"] = years.gross.eq(1)
    summary = years.groupby("stock_id").agg(
        complete_years=("year", "size"), positive_years=("positive", "sum"),
        negative_years=("negative", "sum"), flat_years=("flat", "sum"),
        last_complete_year=("year", "max")).reset_index()
    # Do not expose stocks first observed after the screen cutoff.
    screen = known.groupby("stock_id").agg(
        training_months=(TARGET, "size"), last_training_return=("earned_month", "max")
    ).reset_index().merge(summary, on="stock_id", how="left", validate="one_to_one")
    counts = ["complete_years", "positive_years", "negative_years", "flat_years"]
    screen[counts] = screen[counts].fillna(0).astype(int)
    screen["positive_year_fraction"] = screen.positive_years / screen.complete_years.replace(0, np.nan)
    screen["eligible_for_screen"] = screen.complete_years.ge(min_years)
    screen["selected"] = screen.eligible_for_screen & screen.positive_year_fraction.gt(.5)
    screen["exclusion_reason"] = ""
    screen.loc[~screen.selected, "exclusion_reason"] = "positive years are not a strict majority"
    screen.loc[~screen.eligible_for_screen, "exclusion_reason"] = "fewer than three complete calendar years"
    screen["screen_metric"] = "annual_consistency"
    screen["screen_asof"] = asof
    return screen.sort_values("stock_id").reset_index(drop=True)


def consistency_momentum_weights(signals, pool):
    """Invest positive-stock budget slots; leave all other slots in cash.

    With P positive signals among N currently available pool stocks, invest P/N.
    Within that equity budget, weights are proportional to positive momentum.
    No pool members or no positive signals means 100% cash. All weights are
    fractions of total capital, with an implicit residual cash balance.
    """
    required = KEYS + ["eligible", "trailing_momentum"]
    if signals.empty or not set(required).issubset(signals):
        raise ValueError("Need keyed momentum and eligibility signals.")
    if signals[KEYS].isna().any().any() or signals.duplicated(KEYS).any():
        raise ValueError("Signal keys must be unique and nonmissing.")
    weights, audits = [], []
    for date, block in signals.sort_values(KEYS).groupby("date", sort=True):
        in_pool = block.stock_id.isin(pool)
        positive = in_pool & block.eligible & np.isfinite(block.trailing_momentum) & block.trailing_momentum.gt(0)
        available, count = int(in_pool.sum()), int(positive.sum())
        equity = count / available if available else 0.0
        allocation = block[KEYS].copy().assign(weight=0.0)
        if count:
            momentum = block.loc[positive, "trailing_momentum"]
            allocation.loc[positive, "weight"] = equity * momentum / momentum.sum()
        weights.append(allocation)
        audits.append({"date": date, "available_pool_stocks": available,
            "positive_momentum_stocks": count, "fallback": False,
            "allocation_reason": "positive momentum" if count else "no positive eligible pool stocks; cash"})
    return pd.concat(weights, ignore_index=True), pd.DataFrame(audits)
