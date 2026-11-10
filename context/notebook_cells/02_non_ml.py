"""Non-ML books from the newest final-book save.

Equal weight, the top-10 positive-vol momentum book, and full baskets.
A later STORY row with role "rule" is included. A book missing from the
save is skipped. This module does not fit a model and does not open a
course CSV.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from research.notebook_views import (
    STORY,
    color_for,
    finish_figure,
    load_final_windows,
    prepare_axes,
    present,
    story_slice,
    window_title,
)

_ROLES = ("benchmark", "rule")


def _window_frames(window):
    """Metrics and monthly returns for one saved window."""
    _study, _training, windows = load_final_windows()
    if window not in windows:
        found = ", ".join(repr(name) for name in windows)
        raise ValueError(f"Window {window!r} is not in this save. Windows that exist: {found}.")
    return windows[window]


def _non_ml_story():
    return [item for item in STORY if item["role"] in _ROLES]


def non_ml_table(window="test") -> pd.DataFrame:
    """Annual log growth, CAGR, volatility, Sharpe, max drawdown, and average cash."""
    metrics, _monthly = _window_frames(window)
    return present(story_slice(metrics, roles=_ROLES))


def non_ml_windows_table() -> pd.DataFrame:
    """The same non-ML books on the training window, the official test, and the combined path.

    The combined path compounds saved training months and then saved test months.
    Those strategy months start in 2005. They are not the 2000–2018 equal-weight path.
    """
    _study, _training, windows = load_final_windows()
    labels = (
        ("training", "Training window"),
        ("test", "Official test"),
        ("entire", "Combined path"),
    )
    frames = []
    for key, label in labels:
        if key not in windows:
            continue
        metrics, monthly = windows[key]
        show = present(story_slice(metrics, roles=_ROLES))
        start = pd.Timestamp(monthly.return_month.min())
        end = pd.Timestamp(monthly.return_month.max())
        show.insert(0, "Window", f"{label} · {start:%b %Y}–{end:%b %Y}")
        frames.append(show)
    if not frames:
        raise ValueError("The final-book save has no window to show.")
    return pd.concat(frames, ignore_index=True)


def non_ml_figure(window="test"):
    """Growth of $1 for each non-ML story book in the window."""
    _metrics, monthly = _window_frames(window)
    figure, axis = plt.subplots(figsize=(10.5, 4.2))
    plotted = False
    for index, item in enumerate(_non_ml_story()):
        part = monthly.loc[
            monthly.version.eq(item["version"])
            & monthly.momentum_months.eq(item["momentum"])
            & monthly.model_key.eq("none")
        ].sort_values("return_month")
        if part.empty:
            continue
        # Saved wealth starts after the first earned month. Begin at $1 on the snapshot.
        if "snapshot_date" in part.columns:
            dates = [part.snapshot_date.iloc[0], *part.return_month]
            wealth = [1.0, *part.wealth.to_numpy()]
        else:
            dates = part.return_month
            wealth = part.wealth
        axis.plot(
            dates,
            wealth,
            color=color_for(item["book"], index),
            label=item["book"],
            linewidth=1.8,
        )
        plotted = True
    if not plotted:
        plt.close(figure)
        raise ValueError("None of the non-ML story books are in this save.")
    prepare_axes(axis)
    axis.set_ylabel("Growth of $1")
    axis.set_xlabel("Earned month")
    axis.legend(frameon=False)
    return finish_figure(figure, window_title(window, monthly))
