"""Small chronological GridSearchCV. Takes a feature recipe and returns a saved model.

This module does not open the CSVs, build a portfolio, or register an experiment
run. Call it when a feature set is ready. The search score is a probability
score. Portfolio results stay a separate later comparison.

A row dated D earns its return at the next month-end. That label may train a
fold only when the validation block starts on or after that earned month.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits

from .legacy import KEYS, PROJECT_ROOT, TARGET, return_month
from .models import default_model_registry, event_labels


# Declared before any search result is inspected. Current fixed defaults sit in
# each grid. Candidate counts stay tiny so one origin is a short job.
DECLARED_GRIDS = {
    "logistic": {"model__C": [0.1, 1.0, 10.0]},
    "naive_bayes": {"model__var_smoothing": [1e-9, 1e-5]},
    "random_forest": {"model__max_depth": [4, 6], "model__min_samples_leaf": [50, 200]},
    "hist_gradient_boosting": {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31]},
}
ALLOWED_SCORING = {"neg_log_loss", "neg_brier_score"}
DEFAULT_MAX_FITS = 40


class MonthForwardSplit:
    """Expanding month blocks. Every stock from a month stays on one side."""

    def __init__(self, dates, label_available_at, n_splits=3, min_train_months=24):
        self.dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
        self.label_available_at = pd.DatetimeIndex(pd.to_datetime(np.asarray(label_available_at)))
        self.n_splits = n_splits
        self.min_train_months = min_train_months
        if len(self.dates) != len(self.label_available_at):
            raise ValueError("Dates and label-availability times must align row for row.")
        if not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError("Use at least two chronological folds.")
        if not isinstance(min_train_months, int) or min_train_months < 12:
            raise ValueError("Keep at least 12 training months before the first validation block.")

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if len(self.dates) != len(X):
            raise ValueError("The splitter was built for a different row order than X.")
        months = pd.Index(self.dates.unique()).sort_values()
        tail = months[self.min_train_months:]
        if len(tail) < self.n_splits:
            raise ValueError(
                f"Need at least {self.min_train_months + self.n_splits} distinct months for "
                f"{self.n_splits} folds after {self.min_train_months} initial months.")
        edges = np.linspace(0, len(tail), self.n_splits + 1, dtype=int)
        for fold, (start, stop) in enumerate(zip(edges[:-1], edges[1:]), start=1):
            if stop <= start:
                raise ValueError("A chronological fold received no validation months.")
            block = tail[start:stop]
            origin = pd.Timestamp(block[0])
            # date < origin keeps out the validation snapshot and any later month.
            # label_available_at <= origin is the same rule as ModelData.training_slice.
            train = np.flatnonzero((self.dates < origin) & (self.label_available_at <= origin))
            valid = np.flatnonzero(self.dates.isin(block))
            if len(train) == 0 or len(valid) == 0:
                raise ValueError(f"Fold {fold} has an empty train or validation side.")
            yield train, valid

    def fold_table(self, X):
        rows = []
        for fold, (train, valid) in enumerate(self.split(X), start=1):
            train_months = self.dates[train].unique().sort_values()
            valid_months = self.dates[valid].unique().sort_values()
            rows.append({
                "fold": fold,
                "validation_origin": valid_months[0],
                "train_start": train_months[0], "train_end": train_months[-1],
                "train_months": int(len(train_months)), "train_rows": int(len(train)),
                "validation_start": valid_months[0], "validation_end": valid_months[-1],
                "validation_months": int(len(valid_months)), "validation_rows": int(len(valid)),
            })
        return pd.DataFrame(rows)


@dataclass
class TuningResult:
    pipeline: object
    trials: pd.DataFrame
    folds: pd.DataFrame
    metadata: dict


def declared_grid(model_key):
    if model_key not in DECLARED_GRIDS:
        raise ValueError(f"No declared grid for {model_key}. Choose one of {sorted(DECLARED_GRIDS)}.")
    return {key: list(values) for key, values in DECLARED_GRIDS[model_key].items()}


def candidate_count(grid):
    count = 1
    for values in grid.values():
        if len(values) < 1:
            raise ValueError("Every grid axis needs at least one value.")
        count *= len(values)
    return count


def _check_budget(model_key, grid, n_splits, fixed_params, max_fits, allow_slow):
    fits = candidate_count(grid) * n_splits + 1
    if fits > max_fits:
        raise ValueError(
            f"This search would fit {fits} models, above the limit of {max_fits}. "
            "Shrink the grid or raise max_fits deliberately.")
    if model_key == "random_forest" and not allow_slow:
        spec = default_model_registry()[model_key]
        trees = fixed_params.get("model__n_estimators", spec.factory().get_params()["n_estimators"])
        if trees >= 80:
            raise ValueError(
                "The random-forest factory uses 100 trees, so this grid would be slow. "
                "Pass fixed_params={'model__n_estimators': 40} and record that choice, "
                "or set allow_slow=True to keep 100 trees.")
    return fits


def _prepare_frame(frame, feature_spec, asof):
    required = set(KEYS + [TARGET, "label_available_at"]) | set(feature_spec.columns)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Tuning frame is missing {sorted(missing)}.")
    work = frame.reset_index(drop=True).copy()
    work["date"] = pd.to_datetime(work["date"])
    work["label_available_at"] = pd.to_datetime(work["label_available_at"])
    asof = pd.Timestamp(asof)
    if not work.date.dt.is_month_end.all() or work.duplicated(KEYS).any():
        raise ValueError("Tuning rows need unique month-end stock-month keys.")
    if work.label_available_at.max() > asof:
        raise ValueError("Pass only labels known at the tuning origin. Use ModelData.training_slice.")
    expected = pd.DatetimeIndex(return_month(work.date))
    if not work.label_available_at.equals(pd.Series(expected, index=work.index)):
        raise ValueError("label_available_at must be the month after the snapshot.")
    columns = list(feature_spec.columns)
    if not np.isfinite(work[columns + [TARGET]].to_numpy(dtype=float)).all():
        raise ValueError("Features and returns must be finite.")
    if work[TARGET].lt(-1).any():
        raise ValueError("Training labels must be valid simple returns.")
    return work, asof, columns


def tune_classifier(frame, feature_spec, model_key, asof, *, target="loss", n_splits=3,
                    min_train_months=24, scoring="neg_log_loss", grid=None, fixed_params=None,
                    max_fits=DEFAULT_MAX_FITS, allow_slow=False):
    """Fit one model on one already-known training slice.

    `feature_spec` chooses the columns and, when set, the selector that is fit
    inside every fold. The returned pipeline is refit on the whole slice with
    the winning parameters. Nothing is written until `save_tuned_model`.
    """
    if os.environ.get("MLFACTOR_READ_ONLY") == "1":
        raise RuntimeError("Model fitting is disabled during artifact-only rendering.")
    if scoring not in ALLOWED_SCORING:
        raise ValueError(f"Score the search with one of {sorted(ALLOWED_SCORING)}. Portfolio results are separate.")
    registry = default_model_registry()
    if model_key not in registry:
        raise ValueError(f"Unknown model {model_key}.")
    spec = registry[model_key]
    work, asof, columns = _prepare_frame(frame, feature_spec, asof)
    y = event_labels(work[TARGET], target)
    if y.nunique() < 2:
        raise ValueError("Tuning needs both event classes in the training slice.")
    search_grid = declared_grid(model_key) if grid is None else {key: list(values) for key, values in grid.items()}
    fixed_params = {} if fixed_params is None else dict(fixed_params)
    overlap = set(search_grid).intersection(fixed_params)
    if overlap:
        raise ValueError(f"A parameter cannot be both fixed and searched: {sorted(overlap)}.")
    n_fits = _check_budget(model_key, search_grid, n_splits, fixed_params, max_fits, allow_slow)
    splitter = MonthForwardSplit(work.date, work.label_available_at, n_splits, min_train_months)
    folds = splitter.fold_table(work)
    pipeline = spec.pipeline(feature_spec)
    if fixed_params:
        pipeline.set_params(**fixed_params)
    search = GridSearchCV(
        pipeline, search_grid, scoring=scoring, cv=splitter, refit=True,
        n_jobs=1, error_score="raise", return_train_score=False)
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        search.fit(work[columns], y)
    seconds = time.perf_counter() - started
    fitted = search.best_estimator_
    selector = fitted.named_steps["feature_selection"]
    if hasattr(selector, "get_feature_names_out"):
        fitted_features = [str(name) for name in selector.get_feature_names_out()]
    else:
        fitted_features = columns
    trials = _trials(search.cv_results_, search.best_params_)
    code_path = Path(__file__).resolve()
    metadata = {
        "engine": "GridSearchCV",
        "scoring": scoring,
        "scoring_direction": "higher_is_better",
        "model_key": model_key,
        "model_label": spec.label,
        "target": target,
        "asof": str(asof.date()),
        "trained_through_return_month": str(work.label_available_at.max().date()),
        "n_rows": int(len(work)),
        "n_months": int(work.date.nunique()),
        "n_splits": int(n_splits),
        "min_train_months": int(min_train_months),
        "n_candidates": int(candidate_count(search_grid)),
        "n_fits": int(n_fits),
        "grid": search_grid,
        "fixed_params": fixed_params,
        "chosen_params": search.best_params_,
        "best_score": float(search.best_score_),
        "seconds": seconds,
        "feature_spec": feature_spec.metadata(),
        "feature_columns": columns,
        "fitted_features": fitted_features,
        "n_fitted_features": len(fitted_features),
        "model_params": fitted.named_steps["model"].get_params(),
        "tuning_models": [model_key],
        "portfolio_scored": False,
        "code_sha256": hashlib.sha256(code_path.read_bytes()).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    return TuningResult(fitted, trials, folds, metadata)


def _trials(cv_results, chosen):
    frame = pd.DataFrame(cv_results)
    keep = ["mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time"]
    keep += [column for column in frame.columns if column.startswith("split") and column.endswith("_test_score")]
    tidy = frame[keep].copy()
    tidy.insert(0, "params", [json.dumps(params, sort_keys=True, default=str) for params in frame["params"]])
    tidy.insert(1, "chosen", [params == chosen for params in frame["params"]])
    tidy = tidy.rename(columns={"rank_test_score": "rank", "mean_fit_time": "mean_fit_seconds"})
    return tidy.sort_values(["rank", "params"]).reset_index(drop=True)


def save_tuned_model(result, directory):
    """Write the refit pipeline, the trial table, and the fold definitions."""
    directory = Path(directory)
    if directory.exists():
        raise ValueError(f"Refusing to overwrite {directory}.")
    directory.mkdir(parents=True)
    pipeline_path = directory / "pipeline.joblib"
    trials_path = directory / "trials.parquet"
    folds_path = directory / "folds.parquet"
    joblib.dump(result.pipeline, pipeline_path)
    result.trials.to_parquet(trials_path, index=False)
    result.folds.to_parquet(folds_path, index=False)
    metadata = dict(result.metadata)
    metadata["file_sha256"] = {
        "pipeline.joblib": _digest(pipeline_path),
        "trials.parquet": _digest(trials_path),
        "folds.parquet": _digest(folds_path),
    }
    (directory / "tuning.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    result.metadata = metadata
    return directory


def load_tuned_model(directory):
    """Reload a saved search without fitting."""
    directory = Path(directory)
    metadata = json.loads((directory / "tuning.json").read_text())
    if metadata.get("engine") != "GridSearchCV":
        raise ValueError("This directory is not a GridSearchCV tuning result.")
    for name in ["pipeline.joblib", "trials.parquet", "folds.parquet"]:
        path = directory / name
        if not path.is_file() or _digest(path) != metadata["file_sha256"][name]:
            raise ValueError(f"Saved tuning file is missing or damaged: {name}")
    pipeline = joblib.load(directory / "pipeline.joblib")
    return TuningResult(pipeline, pd.read_parquet(directory / "trials.parquet"),
                        pd.read_parquet(directory / "folds.parquet"), metadata)


def predict_event_probability(result, frame):
    """Apply a loaded pipeline. The frame needs the recipe's input columns."""
    columns = list(result.metadata["feature_columns"])
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction frame is missing {missing}.")
    model = result.pipeline.named_steps["model"]
    probabilities = result.pipeline.predict_proba(frame[columns])
    return probabilities[:, list(model.classes_).index(1)]


def new_tuning_directory(root=None):
    """A new folder under artifacts/tuning. It is not an experiment-catalog run."""
    base = Path(root) if root is not None else PROJECT_ROOT / "artifacts" / "tuning"
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + hashlib.sha256(
        os.urandom(16)).hexdigest()[:8]
    return base / name


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
