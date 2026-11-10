"""Cash adapter around the preserved Part 0 portfolio accounting."""
import numpy as np
import pandas as pd

from .legacy import KEYS, TARGET, portfolio_returns


def portfolio_with_cash(frame, weights):
    """Account for residual cash at 0%, including it in drift-aware turnover.

    Stored weights contain real stocks only. Add a synthetic cash leg just for
    Part 0 accounting, using temporary integer IDs to avoid stock-ID collisions.
    """
    if set(weights.columns) != set(KEYS + ["weight"]):
        raise ValueError("Need date, stock_id and weight only.")
    if weights[KEYS].isna().any().any() or weights.duplicated(KEYS).any():
        raise ValueError("Weights need unique, nonmissing keys.")
    matched = frame[KEYS].merge(weights, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not matched._merge.eq("both").all():
        raise ValueError("Stock weight keys must match return keys exactly.")
    if not np.isfinite(weights.weight).all() or weights.weight.lt(0).any():
        raise ValueError("Weights must be finite and nonnegative.")
    equity = weights.groupby("date").weight.sum()
    if equity.gt(1 + 1e-10).any():
        raise ValueError("Stock exposure cannot exceed total capital.")
    cash = (1 - equity).clip(lower=0)
    mapping = {asset: index for index, asset in enumerate(frame.stock_id.unique())}
    returns = frame[KEYS + [TARGET]].copy()
    returns["stock_id"] = returns.stock_id.map(mapping)
    allocation = weights.copy()
    allocation["stock_id"] = allocation.stock_id.map(mapping)
    cash_returns = cash.rename_axis("date").reset_index(name="weight").assign(stock_id=-1)
    book = portfolio_returns(
        pd.concat([returns, cash_returns[KEYS].assign(**{TARGET: 0.0})], ignore_index=True),
        pd.concat([allocation, cash_returns[KEYS + ["weight"]]], ignore_index=True))
    by_date = weights.assign(held=weights.weight.gt(0)).groupby("date")
    book["n_assets"] = by_date.size().to_numpy()
    book["n_held"] = by_date.held.sum().to_numpy()
    book["equity_weight"] = equity.to_numpy()
    book["cash_weight"] = cash.to_numpy()
    return book
