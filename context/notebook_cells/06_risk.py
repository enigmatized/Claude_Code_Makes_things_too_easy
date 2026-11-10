"""Risk and reward for the story books on each saved window.

Reads the newest final-book save and, when it is present, the saved training
months. It does not fit a model and does not open a course CSV.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from research.notebook_views import (
    CHOICE,
    STORY,
    color_for,
    finish_figure,
    load_final_windows,
    prepare_axes,
    present,
    story_slice,
    window_title,
)

_WINDOW_LABELS = {
    "test": "Official test",
    "training": "Training window",
    "entire": "Entire path",
}
_COLUMNS = ["Window", "Book", "Annual log growth", "CAGR", "Volatility", "Sharpe", "Max drawdown", "Average cash"]


def risk_table() -> pd.DataFrame:
    """One row per story book per saved window, in test, training, entire order."""
    _study, _training_meta, windows = load_final_windows()
    frames = []
    missing = []
    for key, label in _WINDOW_LABELS.items():
        if key not in windows:
            continue
        sliced = story_slice(windows[key][0])
        for book in sliced.attrs.get("missing_books", []):
            if book not in missing:
                missing.append(book)
        show = present(sliced)
        show.insert(0, "Window", label)
        frames.append(show)
    if not frames:
        raise ValueError("The final-book save has no window to show.")
    table = pd.concat(frames, ignore_index=True).loc[:, _COLUMNS].reset_index(drop=True)
    table.attrs["missing_books"] = missing
    return table


def _book_path(monthly, row):
    mask = (
        monthly["version"].eq(row["version"])
        & monthly["momentum_months"].eq(row["momentum_months"])
        & monthly["model_key"].eq(row["model_key"])
    )
    return monthly.loc[mask].sort_values("return_month")


def path_figure():
    """Growth of $1 on the entire path, or the official test when that path is absent."""
    study, _training_meta, windows = load_final_windows()
    window = "entire" if "entire" in windows else "test"
    metrics, monthly = windows[window]
    sliced = story_slice(metrics)
    figure, axis = plt.subplots(figsize=(10.5, 4.6))
    prepare_axes(axis)
    for index, (_, row) in enumerate(sliced.iterrows()):
        path = _book_path(monthly, row)
        if path.empty:
            continue
        axis.plot(
            pd.to_datetime(path["return_month"]),
            path["wealth"],
            color=color_for(row["book"], index),
            label=row["book"],
            linewidth=1.6,
        )
    if window == "entire":
        start = pd.Timestamp(study["monthly"]["return_month"].min())
        axis.axvline(start, color="#9aa39a", linestyle=":", linewidth=1, zorder=0)
        axis.annotate(
            "Official test starts",
            xy=(start, 0.98),
            xycoords=("data", "axes fraction"),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
            color="#68716e",
        )
    axis.set_ylabel("Growth of $1")
    axis.set_xlabel("Earned month")
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    return finish_figure(figure, window_title(window, monthly))


def _span(monthly) -> str:
    start = pd.Timestamp(monthly["return_month"].min())
    end = pd.Timestamp(monthly["return_month"].max())
    return f"{start:%B %Y} through {end:%B %Y}"


def _cell(table, window, book):
    matched = table.loc[table["Window"].eq(window) & table["Book"].eq(book)]
    if matched.empty:
        return None
    return matched.iloc[0]


def _opening(windows) -> str:
    if "test" not in windows:
        return "This save has no official-test window."
    test = _span(windows["test"][1])
    if "training" in windows and "entire" in windows:
        return (
            f"Official test is {test}, the training window is {_span(windows['training'][1])}, "
            f"and the entire path is {_span(windows['entire'][1])}, "
            "compounding the training months and then the test months."
        )
    if "training" in windows:
        return f"Official test is {test}, and the training window is {_span(windows['training'][1])}."
    return f"Official test is {test}."


def _comparison(table) -> str:
    # An overlay choice is the story_slice row, which already uses DISPLAY_MODEL.
    choice = next((item for item in STORY if item["version"] == CHOICE["version"]), None)
    if choice is None:
        return "The written choice is not one of the story books."
    picked = _cell(table, "Official test", choice["book"])
    base = _cell(table, "Official test", "Equal weight")
    if picked is None or base is None:
        return "The written choice and equal weight are not both on the official test."
    return (
        f"On the official test, {choice['book']} CAGR is {picked['CAGR']:.2%} "
        f"and Sharpe is {picked['Sharpe']:.3f}, against Equal weight at "
        f"{base['CAGR']:.2%} and {base['Sharpe']:.3f}."
    )


def _closing(table) -> str:
    parts = []
    base = _cell(table, "Official test", "Equal weight")
    entire = _cell(table, "Entire path", "Equal weight")
    if entire is not None:
        entire_dd = float(entire["Max drawdown"])
        text = f"Equal weight's max drawdown on the entire path is {entire_dd:.2%}"
        training = _cell(table, "Training window", "Equal weight")
        if base is not None and training is not None:
            test_dd = float(base["Max drawdown"])
            training_dd = float(training["Max drawdown"])
            # Only the training crash explains a deeper path when that window is already as deep.
            if entire_dd < test_dd - 1e-4 and training_dd <= entire_dd + 1e-4:
                text += (
                    f", larger than the official-test drawdown of {test_dd:.2%}, "
                    "because that deeper drop comes from the training window"
                )
        parts.append(text)
    if table["Window"].eq("Training window").any():
        parts.append(
            "training-window overlay rows are the saved training forecasts, "
            "while official-test rows are the one frozen fit"
        )
    if not parts:
        return ""
    sentence = parts[0] if len(parts) == 1 else parts[0] + "; " + parts[1]
    return sentence[0].upper() + sentence[1:] + "."


def risk_note() -> str:
    """Two or three sentences. Numbers come from risk_table(); months come from the frames."""
    table = risk_table()
    _study, _training_meta, windows = load_final_windows()
    closing = _closing(table)
    if closing:
        return f"{_opening(windows)} {_comparison(table)} {closing}"
    return f"{_opening(windows)} {_comparison(table)}"
