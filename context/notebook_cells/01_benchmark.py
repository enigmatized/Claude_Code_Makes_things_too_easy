"""Equal-weight log growth and CAGR for train, test, and the combined path.

Monthly portfolio simple returns come from the part-0 equal-weight files.
Annual log growth is 12 * mean(log(1 + r)). CAGR is exp(that) - 1.
The combined path compounds the training months and then the test months.
This module does not open the stock files and does not fit a model.
"""
from __future__ import annotations

import pandas as pd

from research.legacy import performance

HEADERS = (
    "Period", "Months", "Annual log growth", "CAGR", "Volatility", "Sharpe", "Max drawdown",
)
TRAIN_FILE = "part0_data_and_benchmark/outputs/equal_weight_train_monthly.csv"
TEST_FILE = "part0_data_and_benchmark/outputs/equal_weight_test_monthly.csv"
METRICS_FILE = "part0_data_and_benchmark/outputs/equal_weight_metrics.csv"
PERIODS = (("Training", "train"), ("Official test", "test"), ("Combined", "combined"))


def equal_weight_benchmark() -> pd.DataFrame:
    """Train, official test, and the compounded path. One row each."""
    from research.notebook_views import PROJECT_ROOT
    train = _monthly(PROJECT_ROOT / TRAIN_FILE)
    test = _monthly(PROJECT_ROOT / TEST_FILE)
    if train.return_month.max() >= test.return_month.min():
        raise ValueError("Equal-weight training months must end before the official test.")
    combined = pd.concat([train, test], ignore_index=True)
    frames = {"train": train, "test": test, "combined": combined}
    rows = []
    for label, key in PERIODS:
        stats = performance(frames[key].set_index("return_month").simple_return)
        rows.append({
            "Period": label,
            "Months": int(stats["months"]),
            "Annual log growth": float(stats["annualized_log_growth"]),
            "CAGR": float(stats["cagr"]),
            "Volatility": float(stats["annualized_volatility"]),
            "Sharpe": float(stats["sharpe"]),
            "Max drawdown": float(stats["max_drawdown"]),
        })
    frame = pd.DataFrame(rows, columns=list(HEADERS))
    _check_saved_splits(PROJECT_ROOT / METRICS_FILE, frame)
    return frame


def equal_weight_sentence() -> str:
    """State the test CAGR and the test annual log growth from the frame."""
    frame = equal_weight_benchmark()
    test = frame.loc[frame["Period"].eq("Official test")].iloc[0]
    combined = frame.loc[frame["Period"].eq("Combined")].iloc[0]
    return (
        "The graded equal-weight figure is the official-test CAGR, "
        f"{test['CAGR']:.3%}, over {int(test['Months'])} months. "
        f"Annual log growth on that same path is {test['Annual log growth']:.3%}. "
        f"The combined path, January 2000 through December 2018, "
        f"has annual log growth {combined['Annual log growth']:.3%} "
        f"and CAGR {combined['CAGR']:.3%}."
    )


def _monthly(path):
    frame = pd.read_csv(path, parse_dates=["return_month"])
    if frame.empty or "simple_return" not in frame.columns:
        raise ValueError(f"Need monthly simple returns in {path.name}.")
    return frame.sort_values("return_month")


def _check_saved_splits(path, frame):
    """The train and test rows must match the saved equal-weight metrics file."""
    saved = pd.read_csv(path)
    for period, split in (("Training", "train"), ("Official test", "test")):
        got = frame.loc[frame["Period"].eq(period)].iloc[0]
        row = saved.loc[saved["split"].eq(split)].iloc[0]
        for ours, theirs in (
            ("CAGR", "cagr"),
            ("Annual log growth", "annualized_log_growth"),
            ("Sharpe", "sharpe"),
        ):
            if abs(float(got[ours]) - float(row[theirs])) > 1e-10:
                raise ValueError(f"Equal-weight {period} {ours} does not match {path.name}.")
