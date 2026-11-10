"""CAGR and Sharpe for every classifier saved on the story books.

Reads one window from the newest final-book save. It does not fit a model
and does not open a course CSV.
"""
from __future__ import annotations

import pandas as pd

from research.notebook_views import STORY, load_final_windows, model_label


def model_table(window="test") -> pd.DataFrame:
    """Story books by classifier. Benchmark and rule CAGR sit in No ML."""
    return _wide(window, "cagr")


def model_sharpe_table(window="test") -> pd.DataFrame:
    """Same shape as model_table, with Sharpe in each cell."""
    return _wide(window, "sharpe")


def _wide(window, metric) -> pd.DataFrame:
    metrics = _window_metrics(window)
    model_keys = _overlay_model_keys(metrics)
    headers = ["Book", "No ML", *[model_label(key) for key in model_keys]]
    rows = []
    for item in STORY:
        block = metrics.loc[
            metrics.version.eq(item["version"]) & metrics.momentum_months.eq(item["momentum"])
        ]
        # An overlay missing from this save is not a row. A missing classifier is NA.
        if item["role"] == "overlay" and block.empty:
            continue
        row = {header: pd.NA for header in headers}
        row["Book"] = item["book"]
        keys = block.model_key.map(_model_text)
        none = block.loc[keys.eq("none")]
        if not none.empty:
            row["No ML"] = none.iloc[0][metric]
        if item["role"] == "overlay":
            for key in model_keys:
                match = block.loc[keys.eq(key)]
                if not match.empty:
                    row[model_label(key)] = match.iloc[0][metric]
        rows.append(row)
    frame = pd.DataFrame(rows, columns=headers)
    value_columns = [column for column in headers if column != "Book"]
    frame[value_columns] = frame[value_columns].astype("Float64")
    return frame.reset_index(drop=True)


def _window_metrics(window) -> pd.DataFrame:
    _study, _training, windows = load_final_windows()
    if window not in windows:
        available = ", ".join(windows)
        raise ValueError(f"No saved window {window!r}. Windows in this save: {available}.")
    metrics, _monthly = windows[window]
    return metrics


def _overlay_model_keys(metrics) -> list[str]:
    """Classifier keys on the story overlays. none is the No ML column, not a model."""
    from research.models import default_model_registry

    found = []
    for item in STORY:
        if item["role"] != "overlay":
            continue
        part = metrics.loc[
            metrics.version.eq(item["version"]) & metrics.momentum_months.eq(item["momentum"]),
            "model_key",
        ]
        for key in part.map(_model_text):
            if key in {"none", ""} or key in found:
                continue
            found.append(key)
    registry = list(default_model_registry())
    ordered = [key for key in registry if key in found]
    ordered.extend(sorted(key for key in found if key not in registry))
    return ordered


def _model_text(model_key) -> str:
    if model_key is None or (isinstance(model_key, float) and pd.isna(model_key)):
        return ""
    return str(model_key)


_WINDOW_LABELS = (
    ("training", "Training window"),
    ("test", "Official test"),
    ("entire", "Combined path"),
)
_GROWTH = (
    ("annualized_log_growth", "Annual log growth"),
    ("cagr", "CAGR"),
    ("annualized_volatility", "Volatility"),
    ("sharpe", "Sharpe"),
    ("max_drawdown", "Max drawdown"),
)


def classifier_growth_table() -> pd.DataFrame:
    """Log growth, CAGR, volatility, Sharpe, and drawdown for every saved classifier.

    Non-ML books keep a single No ML row. Each overlay book is repeated once
    per classifier in the model registry, so a new registered model appears
    on its own. Training, the official test, and the combined path are
    separate row groups. The combined path compounds saved training months
    and then saved test months.
    """
    from research.models import default_model_registry
    from research.notebook_views import STORY, load_final_windows, model_label

    _study, _training, windows = load_final_windows()
    registry = list(default_model_registry())
    rows = []
    for key, label in _WINDOW_LABELS:
        if key not in windows:
            continue
        metrics, monthly = windows[key]
        start = pd.Timestamp(monthly.return_month.min())
        end = pd.Timestamp(monthly.return_month.max())
        window_name = f"{label} · {start:%b %Y}–{end:%b %Y}"
        for item in STORY:
            model_keys = registry if item["role"] == "overlay" else ["none"]
            for model_key in model_keys:
                part = metrics.loc[
                    metrics.version.eq(item["version"])
                    & metrics.momentum_months.eq(item["momentum"])
                    & metrics.model_key.eq(model_key)
                ]
                if part.empty:
                    continue
                source = part.iloc[0]
                row = {
                    "Window": window_name,
                    "Book": item["book"],
                    "Model": "No ML" if model_key == "none" else model_label(model_key),
                }
                for field, header in _GROWTH:
                    row[header] = source[field]
                rows.append(row)
    if not rows:
        raise ValueError("None of the story books are in the saved windows.")
    return pd.DataFrame(rows)
