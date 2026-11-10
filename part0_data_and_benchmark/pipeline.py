"""Small, shared foundations for the MLFactor stock-month project.

All portfolio inputs are decimal SIMPLE returns. No downloading, imputation,
clipping, feature selection, model fitting, or file writing happens on import.
"""

from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "R1M_Usd"
KEYS = ["date", "stock_id"]
# Meanings follow https://www.mlfactor.com/data-description.html.
FEATURE_INFO = [
    ("Ocf_Toa", "Cash flow", "Operating cash flow / total operating assets"),
    ("Tot_Debt_Rev", "Debt", "Total debt / revenue"),
    ("Net_Debt", "Debt", "Net debt"),
    ("Total_Liabilities_Total_Assets", "Debt", "Total liabilities / total assets"),
    ("Share_Turn_3M", "Trading activity", "Average share turnover over 3 months"),
    ("Fcf_Tbv", "Cash flow", "Free cash flow / tangible book value"),
    ("Debtequity", "Debt", "Debt / equity"),
    ("Ocf_Ce", "Cash flow", "Operating cash flow / capital employed"),
    ("Fcf_Yld", "Valuation / payout", "Free cash flow yield"),
    ("Ocf_Tbv", "Cash flow", "Operating cash flow / tangible book value"),
    ("Div_Yld", "Valuation / payout", "Dividend yield"),
    ("Netdebtyield", "Debt", "Net debt yield; detailed construction unspecified"),
    ("Share_Turn_12M", "Trading activity", "Average share turnover over 12 months"),
    ("Tev_Less_Mktcap", "Debt", "Total enterprise value minus market capitalization"),
    ("Bb_Yld", "Valuation / payout", "Buyback yield"),
    ("Ev_Ebitda", "Valuation / payout", "Enterprise value / EBITDA"),
    ("Total_Debt_Capital", "Debt", "Total debt / capital"),
    ("Share_Turn_6M", "Trading activity", "Average share turnover over 6 months"),
    ("Capex_Sales", "Investment", "Capital expenditure / sales"),
    ("Ocf_Ta", "Cash flow", "Operating cash flow / total assets"),
]
FEATURES = [row[0] for row in FEATURE_INFO]


def project_root(start=None):
    """Find the project from a notebook, the course directory, or this module."""
    start = Path(start or Path.cwd()).resolve()
    if start.is_file():
        start = start.parent
    for candidate in [start, *start.parents]:
        for root in [candidate, candidate / "11_mlfactor_long_only_project_2026"]:
            if (root / "mlfactor_long_only_train.csv").is_file():
                return root
    raise FileNotFoundError("Start inside the project or its course directory.")


def return_month(snapshot_dates):
    """The label at a snapshot is earned in the FOLLOWING calendar month."""
    dates = pd.DatetimeIndex(pd.to_datetime(snapshot_dates))
    return (dates.to_period("M") + 1).to_timestamp("M")


def _check_keys(frame):
    if not set(KEYS).issubset(frame.columns):
        raise ValueError("A panel needs date and stock_id columns.")
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("Missing or duplicate stock-month keys.")
    dates = pd.DatetimeIndex(frame["date"])
    if not pd.api.types.is_datetime64_any_dtype(frame["date"]):
        raise ValueError("Parse date to datetime before using the panel.")
    if not dates.is_month_end.all() or not dates.normalize().equals(dates):
        raise ValueError("Snapshot dates must be normalized calendar month-ends.")
    months = dates.unique().sort_values().to_period("M")
    if len(months) == 0 or not months.equals(pd.period_range(months.min(), months.max(), freq="M")):
        raise ValueError("The panel must contain consecutive snapshot months.")


def load_panel(split="train", root=None):
    """Read and validate one source CSV. Default to training; never combine splits."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    root = Path(root) if root is not None else project_root()
    frame = pd.read_csv(root / f"mlfactor_long_only_{split}.csv", parse_dates=["date"])
    if len(frame.columns) != 23 or set(frame.columns) != set(KEYS + FEATURES + [TARGET]):
        raise ValueError("Expected identifiers, the twenty documented features, and R1M_Usd.")
    _check_keys(frame)
    numeric = frame[FEATURES + [TARGET]]
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in numeric.dtypes):
        raise ValueError("Features and labels must be numeric.")
    if frame.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Missing or nonfinite source values; investigate rather than fill.")
    if not frame[FEATURES].ge(0.01 - 1e-12).all().all() or not frame[FEATURES].le(1).all().all():
        raise ValueError("Feature values outside the observed percentile-rank range.")
    if (frame[TARGET] < -1).any():
        raise ValueError("Simple stock returns cannot be less than -100%.")
    return frame.sort_values(KEYS).reset_index(drop=True)


def panel_audit(frame):
    """Basic facts, without using returns to select securities."""
    _check_keys(frame)
    counts = frame.groupby("date").size()
    return pd.Series({
        "rows": len(frame), "columns": frame.shape[1], "months": len(counts),
        "stocks": frame.stock_id.nunique(), "first_snapshot": counts.index.min(),
        "last_snapshot": counts.index.max(),
        "first_return_month": return_month([counts.index.min()])[0],
        "last_return_month": return_month([counts.index.max()])[0],
        "min_stocks_per_month": counts.min(), "max_stocks_per_month": counts.max(),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_stock_months": int(frame.duplicated(KEYS).sum()),
    })


def feature_frame(frame):
    """Keep identifiers and future returns out of model inputs."""
    return frame.loc[:, FEATURES].copy()


def equal_weights(frame):
    """Equal capital on every stock present at each snapshot; no future screening."""
    _check_keys(frame)
    weights = frame[KEYS].copy()
    weights["weight"] = 1.0 / frame.groupby("date")["stock_id"].transform("size")
    return weights


def portfolio_returns(frame, weights=None):
    """Fully invested long-only portfolio, rebalanced at each snapshot.

    Supply one keyed weight per panel row, including explicit zeros for unheld
    stocks. Rows may be shuffled: alignment is by date and stock_id, never position.
    Decisions must be made from snapshot information; this function then earns
    the existing forward return. No additional shift or pct_change is needed.
    Turnover uses previous weights AFTER their returns. First turnover is unknown
    (NaN), since initial funding is a separate convention. No costs are deducted.
    """
    _check_keys(frame)
    if TARGET not in frame or not np.isfinite(frame[TARGET]).all() or (frame[TARGET] < -1).any():
        raise ValueError("A finite simple-return label >= -1 is required for every stock.")
    weights = equal_weights(frame) if weights is None else weights.copy()
    if set(weights.columns) != set(KEYS + ["weight"]):
        raise ValueError("Weights must contain exactly date, stock_id, weight.")
    _check_keys(weights)
    if len(weights) != len(frame):
        raise ValueError("Supply one weight per panel row; use explicit zero weights.")
    data = frame[KEYS + [TARGET]].merge(weights, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not data["_merge"].eq("both").all():
        raise ValueError("Weight keys and return keys must match exactly.")
    if not np.isfinite(data.weight).all() or (data.weight < 0).any():
        raise ValueError("Weights must be finite and nonnegative.")
    if not np.allclose(data.groupby("date").weight.sum(), 1, atol=1e-10, rtol=0):
        raise ValueError("Weights must sum to one within every snapshot.")

    records = []
    previous_drift = None
    for date, block in data.sort_values(KEYS).groupby("date", sort=True):
        current = block.set_index("stock_id").weight
        stock_returns = block.set_index("stock_id")[TARGET]
        ret = float(current.dot(stock_returns))
        if ret <= -1:
            raise ValueError("A portfolio that loses 100% cannot be continued or logged.")
        turnover = np.nan
        if previous_drift is not None:
            union = current.index.union(previous_drift.index)
            turnover = 0.5 * (current.reindex(union, fill_value=0) - previous_drift.reindex(union, fill_value=0)).abs().sum()
        records.append({
            "snapshot_date": date, "return_month": return_month([date])[0],
            "n_assets": len(block), "n_held": int((current > 0).sum()),
            "simple_return": ret, "turnover_one_way": turnover,
        })
        previous_drift = current * (1 + stock_returns) / (1 + ret)
    result = pd.DataFrame(records).set_index("return_month")
    result["log_return"] = np.log1p(result.simple_return)
    result["wealth"] = (1 + result.simple_return).cumprod()
    # Include the original $1 peak, even when the first month loses money.
    result["drawdown"] = result.wealth / result.wealth.cummax().clip(lower=1) - 1
    return result


def _monthly_series(values):
    if not isinstance(values, pd.Series) or not isinstance(values.index, pd.DatetimeIndex):
        raise ValueError("Supply a Series indexed by earned month-end dates.")
    if len(values) < 2 or not values.index.is_unique or not values.index.is_monotonic_increasing:
        raise ValueError("Need at least two unique, ordered monthly returns.")
    periods = values.index.to_period("M")
    if not values.index.is_month_end.all() or not periods.equals(pd.period_range(periods.min(), periods.max(), freq="M")):
        raise ValueError("Return months must be consecutive, with one observation per month.")
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Returns must be finite and strictly above -100%.")
    return values.astype(float)


def performance(returns, risk_free_monthly=0.0):
    """Monthly data; sample volatility; simple excess-return Sharpe.

    risk_free_monthly is a decimal monthly return (scalar or exactly aligned
    Series). Zero is the course convention, not an observed Treasury-bill series.
    The sqrt(12) convention assumes sufficiently weak serial dependence.
    """
    r = _monthly_series(returns)
    if isinstance(risk_free_monthly, pd.Series):
        if not risk_free_monthly.index.equals(r.index):
            raise ValueError("Risk-free returns must align exactly with return months.")
        rf = risk_free_monthly.to_numpy(dtype=float)
    else:
        rf = float(risk_free_monthly)
    if not np.isfinite(rf).all() or np.any(np.asarray(rf) <= -1):
        raise ValueError("Risk-free monthly returns must be finite and above -100%.")
    excess = r - rf
    logs = np.log1p(r)
    wealth = (1 + r).cumprod()
    drawdown = wealth / wealth.cummax().clip(lower=1) - 1
    vol = r.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    return pd.Series({
        "months": len(r), "total_return": np.expm1(logs.sum()),
        "cagr": np.expm1(logs.mean() * 12),
        "annualized_arithmetic_mean": r.mean() * 12,
        "annualized_log_growth": logs.mean() * 12,
        "annualized_volatility": vol * np.sqrt(12),
        "sharpe": excess.mean() / excess_vol * np.sqrt(12) if excess_vol > 0 else np.nan,
        "max_drawdown": drawdown.min(), "best_month": r.max(),
        "worst_month": r.min(), "positive_month_fraction": (r > 0).mean(),
    })


def expanding_month_folds(frame, min_train_months=48, validation_months=12):
    """Date lists for expanding chronological folds; keep each month together."""
    _check_keys(frame)
    if min_train_months < 1 or validation_months < 1:
        raise ValueError("Fold sizes must be positive.")
    months = pd.DatetimeIndex(frame.date.unique()).sort_values()
    for cut in range(min_train_months, len(months) - validation_months + 1, validation_months):
        yield months[:cut], months[cut:cut + validation_months]
