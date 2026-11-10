"""Independent, keyed transformations of a strategy's baseline weights."""

import numpy as np
import pandas as pd

from .legacy import KEYS


def identity(weights, predictions=None, strength=0.0):
    return weights.copy()


def loss_probability(weights, predictions, strength=0.5, allow_cash=False):
    """w_new ∝ w_base * (1 - strength * P(loss)); remain fully invested.

    Strength must be below one: all previously held names retain positive weight
    even for P(loss)=1. Predictions must cover every input key exactly once.
    The default has no cash. With allow_cash=True, do not renormalize: preserve
    existing cash and transfer the probability haircut into additional cash.
    """
    if not np.isfinite(strength) or not 0 <= strength < 1:
        raise ValueError("Adjustment strength must lie in [0, 1).")
    if set(weights.columns) != set(KEYS + ["weight"]) or weights.empty:
        raise ValueError("Need nonempty date, stock_id, weight inputs.")
    if weights[KEYS].isna().any().any() or weights.duplicated(KEYS).any():
        raise ValueError("Weight keys must be unique and nonmissing.")
    if not np.isfinite(weights.weight).all() or weights.weight.lt(0).any():
        raise ValueError("Weights must be finite and nonnegative.")
    totals = weights.groupby("date").weight.sum()
    if allow_cash and totals.gt(1 + 1e-10).any():
        raise ValueError("Stock exposure cannot exceed one.")
    if not allow_cash and not np.allclose(totals, 1, atol=1e-10, rtol=0):
        raise ValueError("Baseline weights must sum to one per month.")
    if predictions is None or not set(KEYS + ["probability_of_loss"]).issubset(predictions):
        raise ValueError("Need keyed loss probabilities.")
    if predictions[KEYS].isna().any().any() or predictions.duplicated(KEYS).any():
        raise ValueError("Prediction keys must be unique and nonmissing.")
    merged = weights.merge(
        predictions[KEYS + ["probability_of_loss"]], on=KEYS,
        how="outer", validate="one_to_one", indicator=True,
    )
    if not merged._merge.eq("both").all():
        raise ValueError("Prediction keys must match baseline weight keys exactly.")
    p = merged.probability_of_loss
    if not np.isfinite(p).all() or not p.between(0, 1).all():
        raise ValueError("Loss probabilities must be finite and between zero and one.")
    if strength == 0:
        return weights.copy()
    merged["weight"] *= 1 - strength * p
    if not allow_cash:
        merged["weight"] /= merged.groupby("date").weight.transform("sum")
    return merged[KEYS + ["weight"]].sort_values(KEYS).reset_index(drop=True)


def upside_probability(weights, predictions, strength=.5, threshold=.6, allow_cash=False):
    """Boost qualifying held names, then preserve the original equity budget.

    raw = base * (1 + strength * I(P(next return > 0) >= threshold)).
    Cash-capable rules keep their baseline cash; zero-weight names remain zero.
    """
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Upside probability threshold must lie in [0, 1].")
    if not np.isfinite(strength) or not 0 <= strength < 1:
        raise ValueError("Adjustment strength must lie in [0, 1).")
    if predictions is None or "probability_of_up" not in predictions:
        raise ValueError("Need keyed upside probabilities.")
    # Reuse the existing strict key/probability/budget validation, without a haircut.
    loss_probability(weights, predictions.rename(columns={"probability_of_up": "probability_of_loss"}),
                     strength=0, allow_cash=allow_cash)
    if strength == 0:
        return weights.copy()
    merged = weights.merge(predictions[KEYS + ["probability_of_up"]], on=KEYS, validate="one_to_one")
    budget = merged.groupby("date").weight.transform("sum")
    merged["weight"] *= 1 + strength * merged.probability_of_up.ge(threshold)
    total = merged.groupby("date").weight.transform("sum")
    merged["weight"] = merged.weight * budget / total.where(total.gt(0), 1)
    return merged[KEYS + ["weight"]].sort_values(KEYS).reset_index(drop=True)
