"""Fit frozen stock pools using returns realized before evaluation starts."""

import numpy as np
import pandas as pd

from .legacy import KEYS, TARGET, return_month


def rank_training_ratio(training, metric, asof, fraction=0.20, min_months=12):
    """Rank stocks by historical Sharpe or Sortino with zero reference return.

    Sharpe = mean(r) / sample_std(r) * sqrt(12).
    Sortino = mean(r) / sqrt(mean(min(r, 0)**2)) * sqrt(12).
    Every observed return remains in Sortino's observation count. Zero risk
    produces an undefined ratio and an explicit exclusion, never infinity.

    The pool is the top ceil(fraction * eligible stocks), even if some eligible
    scores are negative. Fit only once at the first evaluation snapshot. There
    is no requirement that training returns be consecutive, but subsequent
    momentum eligibility still requires 12 consecutive known monthly returns.
    """
    if metric not in {"sharpe", "sortino"}:
        raise ValueError("Choose sharpe or sortino for the training screen.")
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Screen fraction must lie in (0, 1].")
    if not isinstance(min_months, int) or min_months < 2:
        raise ValueError("Need at least two observed training months.")
    if training.empty or not set(KEYS + [TARGET]).issubset(training.columns):
        raise ValueError("Need training stock-month keys and return labels.")
    if training[KEYS].isna().any().any() or training.duplicated(KEYS).any():
        raise ValueError("Training stock-month keys must be unique and nonmissing.")
    if not pd.api.types.is_datetime64_any_dtype(training.date) or not training.date.dt.is_month_end.all():
        raise ValueError("Training snapshots must be parsed month-ends.")
    asof = pd.Timestamp(asof)
    if pd.isna(asof) or not asof.is_month_end or asof != asof.normalize():
        raise ValueError("Screen cutoff must be a normalized month-end.")
    known = training[KEYS + [TARGET]].copy()
    known["earned_month"] = return_month(known.date)
    known = known.loc[known.earned_month.le(asof)].copy()
    if known.empty:
        raise ValueError("No training returns were realized by this screen cutoff.")
    if not np.isfinite(known[TARGET]).all() or known[TARGET].lt(-1).any():
        raise ValueError("Known training returns must be finite and at least -100%.")
    known["downside_square"] = known[TARGET].clip(upper=0).pow(2)
    ranked = known.groupby("stock_id").agg(
        training_months=(TARGET, "size"), training_mean_monthly_return=(TARGET, "mean"),
        training_monthly_volatility=(TARGET, "std"),
        training_downside_square=("downside_square", "mean"),
        first_training_return=("earned_month", "min"), last_training_return=("earned_month", "max"),
    ).reset_index()
    sigma = ranked.training_monthly_volatility
    downside = np.sqrt(ranked.training_downside_square)
    ranked["training_volatility"] = sigma * np.sqrt(12)
    ranked["training_downside_deviation"] = downside * np.sqrt(12)
    ranked["training_sharpe"] = ranked.training_mean_monthly_return.div(sigma.where(sigma.gt(0))) * np.sqrt(12)
    ranked["training_sortino"] = ranked.training_mean_monthly_return.div(downside.where(downside.gt(0))) * np.sqrt(12)
    ranked["screen_score"] = ranked[f"training_{metric}"]
    enough = ranked.training_months.ge(min_months)
    finite = np.isfinite(ranked.screen_score)
    ranked["eligible_for_screen"] = enough & finite
    ranked["exclusion_reason"] = ""
    ranked.loc[~finite, "exclusion_reason"] = "undefined or nonfinite ratio (zero risk denominator)"
    ranked.loc[~enough, "exclusion_reason"] = "insufficient observed training months"
    ranked = ranked.sort_values(
        ["eligible_for_screen", "screen_score", "stock_id"], ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    count = int(ranked.eligible_for_screen.sum())
    if count == 0:
        raise ValueError(f"No stock has enough history and a finite training {metric} ratio.")
    ranked["training_rank"] = pd.Series(pd.NA, index=ranked.index, dtype="Int64")
    ranked.loc[ranked.eligible_for_screen, "training_rank"] = np.arange(1, count + 1)
    ranked["selected"] = ranked.training_rank.le(int(np.ceil(fraction * count))).fillna(False).astype(bool)
    ranked["screen_asof"] = asof
    ranked["screen_metric"] = metric
    return ranked.drop(columns=["training_monthly_volatility", "training_downside_square"])
