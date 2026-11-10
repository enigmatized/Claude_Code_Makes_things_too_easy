"""Within-month return categories, searched on the training file only.

The category is the quintile of the forward simple return inside that month.
K nearest neighbors and logistic regression each go through sklearn GridSearchCV.
Labels known after the origin are a holdout. They do not choose the parameters.
This module does not open the official test file and does not register a catalog run.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .legacy import FEATURES, PROJECT_ROOT, TARGET, load_panel, performance, return_month
from .tuning import MonthForwardSplit


N_CLASSES = 5
DEFAULT_ASOF = "2006-12-31"
SCORING = "neg_log_loss"
NEIGHBOR_GRID = (5, 15, 31, 51)
LOGISTIC_GRID = (0.1, 1.0, 10.0)


def quintile_labels(dates, returns, n_classes=N_CLASSES) -> pd.Series:
    """Equal-count categories inside each month. 0 is the lowest forward return."""
    work = pd.DataFrame({
        "date": pd.to_datetime(np.asarray(dates)),
        "ret": np.asarray(returns, dtype=float),
    })
    if len(work) == 0 or not np.isfinite(work["ret"]).all():
        raise ValueError("Category labels need finite returns.")
    counts = work.groupby("date").size()
    if counts.min() < n_classes:
        raise ValueError("Every month needs at least as many names as categories.")
    labels = pd.Series(index=work.index, dtype="int64")
    for _, part in work.groupby("date", sort=False):
        ranks = part["ret"].rank(method="first")
        labels.loc[part.index] = pd.qcut(ranks, n_classes, labels=False).astype(int).to_numpy()
    if not set(labels.unique()).issuperset(range(n_classes)):
        raise ValueError("The category labels did not produce every class.")
    return labels


def search_categories(frame, columns=None, *, asof=DEFAULT_ASOF, n_splits=3, min_train_months=24,
                      neighbor_grid=NEIGHBOR_GRID, logistic_grid=LOGISTIC_GRID,
                      scoring=SCORING, n_classes=N_CLASSES, progress=None):
    """Fit both models on labels known by `asof`, then score the later months in `frame`.

    `frame` is one training panel. The holdout rows stay out of GridSearchCV.
    """
    if os.environ.get("MLFACTOR_READ_ONLY") == "1":
        raise RuntimeError("Model fitting is disabled during artifact-only rendering.")
    if scoring != SCORING:
        raise ValueError("Score the category search with neg_log_loss.")
    columns = list(FEATURES if columns is None else columns)
    work = _prepare(frame, columns)
    asof = pd.Timestamp(asof)
    search = work.loc[work.label_available_at.le(asof)].reset_index(drop=True)
    holdout = work.loc[work.label_available_at.gt(asof)].reset_index(drop=True)
    if search.label_available_at.max() > asof or holdout.empty:
        raise ValueError("The holdout must start after the tuning origin, on rows already in this frame.")
    if holdout.label_available_at.min() <= asof:
        raise ValueError("Holdout labels leaked into the search window.")
    y_search = quintile_labels(search.date, search[TARGET], n_classes)
    splitter = MonthForwardSplit(search.date, search.label_available_at, n_splits, min_train_months)
    folds = splitter.fold_table(search)
    specs = _specs(neighbor_grid, logistic_grid)
    fitted = []
    search_rows = []
    for key, label, pipeline, grid in specs:
        if progress:
            progress(f"GridSearchCV {label}: {grid}")
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            result = GridSearchCV(
                pipeline, grid, scoring=scoring, cv=splitter, refit=True,
                n_jobs=1, error_score="raise", return_train_score=False, verbose=1,
            )
            result.fit(search[columns], y_search)
        seconds = time.perf_counter() - started
        n_fits = _candidate_count(grid) * n_splits + 1
        search_rows.append({
            "model_key": key,
            "model": label,
            "searched": _grid_text(grid),
            "chosen": _param_text(result.best_params_),
            "chosen_params": {name: _jsonable(value) for name, value in result.best_params_.items()},
            "n_fits": int(n_fits),
            "best_score": float(result.best_score_),
            "seconds": float(seconds),
        })
        fitted.append((key, label, result))
        if progress:
            progress(f"{label} chose {search_rows[-1]['chosen']} in {seconds:.1f}s, score {result.best_score_:.4f}.")
    y_holdout = quintile_labels(holdout.date, holdout[TARGET], n_classes)
    classification, portfolio = _score_holdout(fitted, holdout, columns, y_holdout, n_classes)
    return {
        "metadata": {
            "engine": "GridSearchCV",
            "scoring": scoring,
            "scoring_direction": "higher_is_better",
            "task": "within-month forward-return quintile",
            "n_classes": int(n_classes),
            "asof": str(asof.date()),
            "columns": columns,
            "search_rows": int(len(search)),
            "search_months": int(search.date.nunique()),
            "holdout_rows": int(len(holdout)),
            "holdout_first_return_month": str(pd.Timestamp(holdout.label_available_at.min()).date()),
            "holdout_last_return_month": str(pd.Timestamp(holdout.label_available_at.max()).date()),
            "holdout_months": int(holdout.label_available_at.nunique()),
            "n_splits": int(n_splits),
            "min_train_months": int(min_train_months),
            "folds": json.loads(folds.to_json(orient="records", date_format="iso")),
            "search": search_rows,
            "classification": classification,
            "portfolio": portfolio,
            "official_test_opened": False,
            "in_frozen_books": False,
        }
    }


def run_grouping_study(root=None, progress=print):
    """Load the training CSV, search, and save. Never reads the test CSV."""
    if os.environ.get("MLFACTOR_READ_ONLY") == "1":
        raise RuntimeError("Model fitting is disabled during artifact-only rendering.")
    root = Path(root) if root is not None else PROJECT_ROOT
    panel = load_panel("train", root)
    panel = panel.copy()
    panel["label_available_at"] = return_month(panel["date"])
    if progress:
        progress(
            f"Training file only: {len(panel):,} rows, "
            f"labels {panel.label_available_at.min():%Y-%m-%d} through {panel.label_available_at.max():%Y-%m-%d}."
        )
    result = search_categories(panel, progress=progress)
    return save_grouping_study(result, root / "artifacts" / "grouping_studies")


def save_grouping_study(result, directory):
    """Write one new folder of JSON. The pipelines are not stored."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + hashlib.sha256(os.urandom(16)).hexdigest()[:8]
    target = directory / f"grouping_{name}"
    if target.exists():
        raise ValueError(f"Refusing to overwrite {target}.")
    target.mkdir(parents=True)
    metadata = dict(result["metadata"])
    metadata.update(created_utc=datetime.now(timezone.utc).isoformat(), directory=str(target))
    (target / "grouping.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return target


def list_grouping_studies(root=None):
    """Newest directory first. A folder without grouping.json is skipped."""
    base = _study_root(root)
    if not base.is_dir():
        return []
    found = []
    for path in sorted(base.glob("grouping_*"), reverse=True):
        meta_path = path / "grouping.json"
        if not meta_path.is_file():
            continue
        metadata = json.loads(meta_path.read_text())
        found.append({"directory": path, "metadata": metadata})
    return found


def load_latest_grouping(root=None):
    """The newest saved search. Reading this does not fit a model."""
    found = list_grouping_studies(root)
    if not found:
        raise FileNotFoundError("No saved grouping study. The notebook reads that save and does not fit a model.")
    return found[0]


def _study_root(root):
    if root is None:
        return PROJECT_ROOT / "artifacts" / "grouping_studies"
    root = Path(root)
    if root.name == "grouping_studies":
        return root
    return root / "grouping_studies"


def _prepare(frame, columns):
    required = {"date", "label_available_at", TARGET, *columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Grouping frame is missing {sorted(missing)}.")
    work = frame.reset_index(drop=True).copy()
    work["date"] = pd.to_datetime(work["date"])
    work["label_available_at"] = pd.to_datetime(work["label_available_at"])
    if work[columns + [TARGET]].isna().any().any() or not np.isfinite(work[columns + [TARGET]].to_numpy(dtype=float)).all():
        raise ValueError("Features and returns must be finite.")
    if work[TARGET].lt(-1).any():
        raise ValueError("Training labels must be valid simple returns.")
    expected = pd.DatetimeIndex(return_month(work.date))
    if not work.label_available_at.equals(pd.Series(expected, index=work.index)):
        raise ValueError("label_available_at must be the month after the snapshot.")
    return work


def _specs(neighbor_grid, logistic_grid):
    neighbors = [int(value) for value in neighbor_grid]
    strengths = [float(value) for value in logistic_grid]
    if len(neighbors) < 1 or any(value < 1 for value in neighbors):
        raise ValueError("n_neighbors must be positive integers.")
    if len(strengths) < 1 or any(value <= 0 for value in strengths):
        raise ValueError("Logistic C must be positive.")
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(algorithm="ball_tree", n_jobs=1)),
    ])
    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(solver="lbfgs", max_iter=500, random_state=0)),
    ])
    return (
        ("knn", "K nearest neighbors", knn, {"model__n_neighbors": neighbors}),
        ("logistic", "Logistic regression", logistic, {"model__C": strengths}),
    )


def _score_holdout(fitted, holdout, columns, y_holdout, n_classes):
    chance = 1.0 / n_classes
    uniform = float(np.log(n_classes))
    months = pd.DatetimeIndex(pd.to_datetime(holdout.label_available_at))
    equal = holdout.groupby("label_available_at", sort=True)[TARGET].mean()
    classification = []
    portfolio = [_portfolio_row("none", "Equal weight", equal)]
    top = n_classes - 1
    for key, label, result in fitted:
        estimator = result.best_estimator_
        probabilities = _class_probabilities(estimator, holdout[columns], n_classes)
        predicted = probabilities.argmax(axis=1)
        classification.append({
            "model_key": key,
            "model": label,
            "holdout_accuracy": float(accuracy_score(y_holdout, predicted)),
            "chance_accuracy": float(chance),
            "holdout_log_loss": float(log_loss(y_holdout, probabilities, labels=list(range(n_classes)))),
            "uniform_log_loss": uniform,
        })
        selected = holdout.loc[predicted == top].groupby("label_available_at")[TARGET].mean()
        counts = holdout.loc[predicted == top].groupby("label_available_at").size()
        book = selected.reindex(equal.index)
        # A month with no top-category name earns cash at zero.
        invested = book.notna()
        book = book.fillna(0.0)
        row = _portfolio_row(key, f"{label}, top category", book)
        row["avg_names"] = float(counts.reindex(equal.index).fillna(0).mean())
        row["cash_month_fraction"] = float((~invested).mean())
        portfolio.append(row)
    portfolio[0]["avg_names"] = float(holdout.groupby("label_available_at").size().mean())
    portfolio[0]["cash_month_fraction"] = 0.0
    # Confirm the index passed to performance was the holdout months, in order.
    if not pd.DatetimeIndex(equal.index).equals(pd.DatetimeIndex(months.unique()).sort_values()):
        raise ValueError("Holdout book months do not match the holdout labels.")
    return classification, portfolio


def _class_probabilities(estimator, features, n_classes):
    model = estimator.named_steps["model"]
    raw = estimator.predict_proba(features)
    classes = [int(value) for value in model.classes_]
    if classes != list(range(n_classes)):
        raise ValueError(f"The fitted model is missing a category. Classes found: {classes}.")
    return raw


def _portfolio_row(model_key, book, returns):
    stats = performance(pd.Series(returns, index=pd.DatetimeIndex(returns.index))).to_dict()
    return {
        "model_key": model_key,
        "book": book,
        "months": int(stats["months"]),
        "annualized_log_growth": float(stats["annualized_log_growth"]),
        "cagr": float(stats["cagr"]),
        "annualized_volatility": float(stats["annualized_volatility"]),
        "sharpe": float(stats["sharpe"]),
        "max_drawdown": float(stats["max_drawdown"]),
    }


def _candidate_count(grid):
    count = 1
    for values in grid.values():
        count *= len(values)
    return count


def _grid_text(grid):
    parts = []
    for key, values in grid.items():
        name = key.split("__")[-1]
        shown = ", ".join(_plain(value) for value in values)
        parts.append(f"{name}: {shown}")
    return "; ".join(parts)


def _param_text(params):
    return ", ".join(f"{key.split('__')[-1]}={_plain(value)}" for key, value in params.items())


def _plain(value):
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _jsonable(value):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value
