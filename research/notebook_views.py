"""Read-only tables and charts for the graded notebook.

The Streamlit app, the strategy registry, and the saved studies can change.
This module does not import the app, does not fit a model, and does not open
the course stock files. It reads the newest saved study of each kind.

When the books you want to lead with change, edit STORY, DISPLAY_MODEL, and
CHOICE. A book that is not in the latest save is skipped. A new model key in
that save is shown.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# version is the string stored on the monthly file. role "overlay" uses DISPLAY_MODEL.
# momentum 0 is a book that does not use a holding signal.
STORY = (
    {"book": "Equal weight", "version": "Equal weight", "role": "benchmark", "momentum": 0},
    {"book": "Top 10 positive-vol", "version": "Top 10 positive-vol baskets", "role": "rule", "momentum": 1},
    {"book": "Top 10 + downside to cash", "version": "Top 10 + downside to cash", "role": "overlay", "momentum": 1},
    {"book": "Top 10 + upside tilt", "version": "Top 10 + upside tilt", "role": "overlay", "momentum": 1},
    {"book": "Full baskets", "version": "Basket momentum + cash", "role": "rule", "momentum": 1},
    {"book": "Baskets + downside to cash", "version": "Baskets + downside to cash", "role": "overlay", "momentum": 1},
    {"book": "Baskets + upside tilt", "version": "Baskets + upside tilt", "role": "overlay", "momentum": 1},
)

DISPLAY_MODEL = "logistic"

# overlay_version None means the written choice is the unadjusted rule.
CHOICE = {
    "version": "Top 10 positive-vol baskets",
    "overlay_version": None,
    "disclosure": (
        "The 2010–2018 score was run once, after the comparison was written down. "
        "Those official books use factory classifier settings."
    ),
}

BOOK_COLORS = {
    "Equal weight": "#68716e",
    "Top 10 positive-vol": "#b36a2e",
    "Top 10 + downside to cash": "#6b3f69",
    "Top 10 + upside tilt": "#1d6a6a",
    "Full baskets": "#287d68",
    "Baskets + downside to cash": "#536da6",
    "Baskets + upside tilt": "#97619c",
}
FALLBACK_COLORS = ("#273c38", "#bd8235", "#536da6", "#97619c", "#68716e", "#1d6a6a")

# Annual log growth is 12 * mean(log(1 + portfolio simple return)).
# CAGR is exp(that) - 1. Volatility and Sharpe stay on the simple returns.
METRIC_COLUMNS = (
    ("annualized_log_growth", "Annual log growth"),
    ("cagr", "CAGR"),
    ("annualized_volatility", "Volatility"),
    ("sharpe", "Sharpe"),
    ("max_drawdown", "Max drawdown"),
    ("avg_cash_weight", "Average cash"),
)

PERCENT_HEADERS = {"Annual log growth", "CAGR", "Volatility", "Max drawdown", "Average cash"}
GROWTH_HEADERS = ("Annual log growth", "CAGR", "Volatility", "Sharpe", "Max drawdown")


def artifact_root() -> Path:
    """Saved studies. MLFACTOR_ARTIFACT_ROOT overrides the project artifacts folder."""
    override = os.environ.get("MLFACTOR_ARTIFACT_ROOT")
    return Path(override) if override else PROJECT_ROOT / "artifacts"


def model_label(model_key) -> str:
    """Display name from the model registry, so a new registered model keeps its label."""
    if model_key in (None, "none", ""):
        return "—"
    from research.models import default_model_registry
    spec = default_model_registry().get(str(model_key))
    return spec.label if spec is not None else str(model_key).replace("_", " ")


def color_for(book, index=0) -> str:
    return BOOK_COLORS.get(book, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def load_final_windows():
    """Newest final-book study, plus training and the stitched path when that save exists.

    Returns (study, training_metadata, windows). windows maps
    "test", "training", and "entire" to (metrics, monthly). Reading this
    does not open a course CSV and does not refit.
    """
    from research.final_books import list_final_books, load_final_books, matching_training_monthly, window_books
    studies = list_final_books(artifact_root())
    if not studies:
        raise FileNotFoundError("No saved final-book study. The notebook reads that save and does not fit a model.")
    study = load_final_books(studies[0]["directory"])
    try:
        training_monthly, training_meta = matching_training_monthly(artifact_root())
    except ValueError:
        training_monthly, training_meta = None, None
    return study, training_meta, window_books(study["monthly"], training_monthly)


def latest_feature_scores():
    """Newest four-book feature comparison. Newest directory first."""
    from research.basket_features import list_feature_book_scores, load_feature_book_scores
    found = list_feature_book_scores(artifact_root())
    if not found:
        raise FileNotFoundError("No saved feature-score study.")
    return load_feature_book_scores(found[0]["directory"])


def latest_holdout():
    """Newest training-holdout study that recorded a GridSearchCV choice.

    list_tuning_holdouts walks directories in name order, so the last match
    is the newest timestamped save.
    """
    from research.tuning_holdout import list_tuning_holdouts, load_tuning_holdout
    found = [item for item in list_tuning_holdouts(artifact_root()) if item["metadata"].get("chosen")]
    if not found:
        raise FileNotFoundError("No saved GridSearchCV holdout.")
    return load_tuning_holdout(found[-1]["directory"])


def story_slice(metrics, *, roles=None, model_key=None) -> pd.DataFrame:
    """STORY rows that exist in metrics. Missing books are listed on the frame and skipped."""
    chosen_model = DISPLAY_MODEL if model_key is None else model_key
    rows, missing = [], []
    for item in STORY:
        if roles is not None and item["role"] not in roles:
            continue
        key = chosen_model if item["role"] == "overlay" else "none"
        part = metrics.loc[
            metrics.version.eq(item["version"])
            & metrics.momentum_months.eq(item["momentum"])
            & metrics.model_key.eq(key)
        ]
        if part.empty:
            missing.append(item["book"])
            continue
        row = part.iloc[0].copy()
        row["book"] = item["book"]
        row["role"] = item["role"]
        rows.append(row)
    if not rows:
        detail = ", ".join(missing) or "the requested books"
        raise ValueError(f"None of the story books are in this save: {detail}.")
    frame = pd.DataFrame(rows).reset_index(drop=True)
    frame.attrs["missing_books"] = missing
    return frame


def present(frame, extra=()) -> pd.DataFrame:
    """Small display frame. extra is (source_column, header) pairs placed after Book."""
    columns, headers = [], []
    if "book" in frame.columns:
        columns.append("book")
        headers.append("Book")
    for source, header in extra:
        if source in frame.columns:
            columns.append(source)
            headers.append(header)
    for source, header in METRIC_COLUMNS:
        if source in frame.columns:
            columns.append(source)
            headers.append(header)
    show = frame.loc[:, columns].copy()
    show.columns = headers
    show.attrs["missing_books"] = list(frame.attrs.get("missing_books", []))
    return show.reset_index(drop=True)


def style_table(frame):
    """Percent format for the graded columns. Sharpe stays a ratio."""
    formats = {column: "{:.2%}" for column in frame.columns if column in PERCENT_HEADERS}
    if "Sharpe" in frame.columns:
        formats["Sharpe"] = "{:.3f}"
    return frame.style.format(formats).hide(axis="index")


def prepare_axes(axis):
    axis.grid(alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)


def source_note(kind) -> str:
    """One line naming the save a cell is reading, including its months."""
    if kind == "benchmark":
        return (
            "Independent equal-weight files. Training is January 2000–December 2009, 120 months. "
            "The official test is January 2010–December 2018, 108 months. "
            "Combined compounds those monthly simple returns, January 2000–December 2018. "
            "Annual log growth is 12 * mean(log(1 + r)). CAGR is exp(that) - 1."
        )
    if kind == "final":
        study, training_meta, windows = load_final_windows()
        meta = study["metadata"]
        directory = Path(meta["directory"]).name
        training = ""
        if training_meta is not None:
            training = (
                f" Training window {training_meta.get('first_return_month')} through "
                f"{training_meta.get('last_return_month')}."
            )
        return (
            f"Newest final-book save {directory}: {meta.get('first_return_month')} through "
            f"{meta.get('last_return_month')}, {meta.get('fit_policy')} at {meta.get('fit_snapshot')}."
            f"{training}"
        )
    if kind == "features":
        loaded = latest_feature_scores()
        meta = loaded["metadata"]
        window = meta.get("window") or {}
        directory = Path(loaded.get("directory") or meta.get("directory") or "").name
        # load_feature_book_scores does not return the directory. Recover it from the list.
        from research.basket_features import list_feature_book_scores
        found = list_feature_book_scores(artifact_root())
        directory = Path(found[0]["directory"]).name if found else directory
        return (
            f"Newest feature-score save {directory}: {window.get('first_return_month')} through "
            f"{window.get('last_return_month')}. This window is the training years."
        )
    if kind == "tuning":
        loaded = latest_holdout()
        meta = loaded["metadata"]
        directory = Path(meta.get("directory") or "").name
        target = meta.get("prediction_target") or meta.get("scoring")
        return (
            f"Newest GridSearchCV holdout {directory}: earned returns "
            f"{meta.get('holdout_first_return_month')} through {meta.get('holdout_last_return_month')}. "
            f"Search score {meta.get('scoring')}. Prediction target {target}."
        )
    if kind == "engineering":
        from research.engineering_study import latest_engineering
        directory = latest_engineering(artifact_root())
        if directory is None:
            raise FileNotFoundError("No completed engineering training study.")
        return (
            f"Newest completed engineering training study {directory.name}. "
            "January 2007–December 2009 holdout. The test file is not part of this study's training scope. "
            "The 0_7 comparison file is a separate, previously viewed 2010–2018 notebook result."
        )
    if kind == "grouping":
        from research.grouping import load_latest_grouping
        meta = load_latest_grouping(artifact_root())["metadata"]
        directory = Path(meta.get("directory") or "").name
        return (
            f"Newest grouping study {directory}: search labels through {meta.get('asof')}, "
            f"holdout earned returns {meta.get('holdout_first_return_month')} through "
            f"{meta.get('holdout_last_return_month')}. "
            f"GridSearchCV score {meta.get('scoring')}. Training file only."
        )
    raise ValueError(f"Unknown source {kind}.")


def window_title(window, monthly) -> str:
    start = pd.Timestamp(monthly.return_month.min())
    end = pd.Timestamp(monthly.return_month.max())
    count = int(monthly.return_month.nunique())
    names = {"test": "Official test", "training": "Training window", "entire": "Entire path"}
    return f"{names.get(window, window)} · {start:%b %Y}–{end:%b %Y} · {count} months"


def read_json(path) -> dict:
    return json.loads(Path(path).read_text())


def finish_figure(figure, title):
    figure.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def _export_cell(filename, names):
    """Publish one cell module. The cell files stay separate so a new section can be replaced."""
    import importlib.util
    path = PROJECT_ROOT / "context" / "notebook_cells" / filename
    spec = importlib.util.spec_from_file_location(f"mlfactor_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in names:
        globals()[name] = getattr(module, name)


_CELL_EXPORTS = (
    ("01_benchmark.py", ("equal_weight_benchmark", "equal_weight_sentence")),
    ("02_non_ml.py", ("non_ml_table", "non_ml_windows_table", "non_ml_figure")),
    ("03_models.py", ("model_table", "model_sharpe_table", "classifier_growth_table")),
    ("04_features.py", ("feature_cut_table", "feature_name_note")),
    ("05_tuning.py", ("tuning_search_table", "tuning_portfolio_table", "tuning_note")),
    ("06_risk.py", ("risk_table", "path_figure", "risk_note")),
    ("07_choice.py", ("choice_text",)),
    ("08_grouping.py", ("grouping_search_table", "grouping_class_table", "grouping_book_table", "grouping_note")),
    ("09_engineering.py", ("engineering_holdout_table", "engineering_comparison_table", "engineering_note")),
)
for _filename, _names in _CELL_EXPORTS:
    _export_cell(_filename, _names)

__all__ = [
    "CHOICE", "DISPLAY_MODEL", "STORY", "artifact_root", "choice_text", "color_for",
    "equal_weight_benchmark", "equal_weight_sentence", "feature_cut_table", "feature_name_note",
    "finish_figure", "latest_feature_scores", "latest_holdout", "load_final_windows",
    "model_label", "model_sharpe_table", "model_table", "non_ml_figure", "non_ml_table",
    "path_figure", "prepare_axes", "present", "risk_note", "risk_table", "source_note",
    "story_slice", "style_table", "tuning_note", "tuning_portfolio_table", "tuning_search_table",
    "window_title",
]
