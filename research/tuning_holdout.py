"""Compare no ML, fixed ML, and tuned ML on a holdout cut from the training years.

The official 2010–2018 file is not opened. Hyperparameters and the frozen
feature list use only labels known at the holdout origin. The reported books
earn their returns after that origin.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from .data import prepare_data
from .features import FeatureSpec
from .legacy import PROJECT_ROOT, load_panel, return_month
from .models import ModelSpec, default_model_registry, event_fields
from .runner import ExperimentRunner, RunConfig
from .strategies import default_registry
from .tuning import save_tuned_model, tune_classifier


# First inner-test decision. Its own return is earned in January 2007.
# Labels known at this date run through December 2006.
INNER_TEST_ORIGIN = pd.Timestamp("2006-12-31")
VERSIONS = ("No ML", "ML, not tuned", "Tuned ML")
METRIC_COLUMNS = ["months", "cagr", "annualized_log_growth", "annualized_volatility", "sharpe",
                  "sortino", "max_drawdown", "avg_one_way_turnover"]


def history_months_before(dates, origin):
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).unique().sort_values()
    origin = pd.Timestamp(origin)
    if origin not in dates:
        raise ValueError(f"{origin.date()} is not a training snapshot.")
    count = int(dates.get_loc(origin))
    if count < 24 or len(dates) - count < 12:
        raise ValueError("The inner split needs 24 months before the holdout and 12 holdout months.")
    return count


def frozen_feature_list(study, recipe, origin, model_id="logistic"):
    """Columns selected at the last saved fit on or before the holdout origin."""
    fits = study["fits"]
    rows = fits.loc[fits.recipe.eq(recipe)].copy()
    if model_id in set(rows.model_id):
        rows = rows.loc[rows.model_id.eq(model_id)]
    rows["fit_snapshot"] = pd.to_datetime(rows.fit_snapshot)
    eligible = rows.loc[rows.fit_snapshot.le(pd.Timestamp(origin))]
    if eligible.empty:
        raise ValueError("No feature-selection fit is saved on or before the holdout origin.")
    fit = eligible.sort_values("fit_snapshot").iloc[-1]
    columns = tuple(json.loads(fit.fitted_features))
    if not columns or len(columns) > 10:
        raise ValueError("The frozen feature list must contain one to ten names.")
    return columns, {"recipe": recipe, "model_id": str(fit.model_id),
                     "fit_snapshot": str(pd.Timestamp(fit.fit_snapshot).date()),
                     "trained_through_return_month": str(pd.Timestamp(fit.trained_through_return_month).date()),
                     "n_features": len(columns), "columns": list(columns)}


def _spec_with_params(model_key, params):
    base = default_model_registry()[model_key]
    def factory(base=base, params=dict(params)):
        model = base.factory()
        model.set_params(**params)
        return model
    return ModelSpec(base.key, base.label, base.description, factory, base.scale)


def _flat_params(metadata):
    params = {}
    for source in (metadata.get("fixed_params", {}), metadata["chosen_params"]):
        for key, value in source.items():
            if not key.startswith("model__"):
                raise ValueError(f"Unexpected parameter name {key}.")
            params[key.split("__", 1)[1]] = value
    return params


def _monthly(result, version, model_key):
    if version == "No ML":
        frame = result.monthly.loc[result.monthly.variant.eq("baseline")].copy()
    else:
        frame = result.monthly.loc[result.monthly.model_id.eq(model_key)].copy()
    keep = ["strategy", "label", "return_month", "snapshot_date", "simple_return", "wealth", "drawdown"]
    return frame[keep].assign(version=version, model_key=model_key if version != "No ML" else "none")


def _metrics(result, version, model_key):
    if version == "No ML":
        frame = result.metrics.loc[result.metrics.variant.eq("baseline")].copy()
    else:
        frame = result.metrics.loc[result.metrics.model_id.eq(model_key)].copy()
    columns = ["strategy", "label", *METRIC_COLUMNS, "first_return_month", "last_return_month"]
    return frame[columns].assign(version=version, model_key=model_key if version != "No ML" else "none")


def _baseline_returns(result):
    return result.monthly.loc[result.monthly.variant.eq("baseline"),
        ["strategy", "return_month", "simple_return"]].sort_values(["strategy", "return_month"]).reset_index(drop=True)


def create_tuning_holdout(root, feature_columns, model_keys=("logistic", "hist_gradient_boosting"),
                          strategies=None, origin=INNER_TEST_ORIGIN, adjustment_strength=0.5,
                          n_splits=3, min_train_months=24, grid=None, fixed_params=None,
                          allow_slow=False, feature_source=None, progress=None,
                          prediction_target="loss", upside_threshold=0.6):
    """Fit on labels known at `origin` and score the three books after it.

    `feature_columns` are already chosen. This function does not run feature
    selection again and does not read the official test CSV.
    """
    if os.environ.get("MLFACTOR_READ_ONLY") == "1":
        raise RuntimeError("Model fitting is disabled during artifact-only rendering.")
    event_fields(prediction_target)
    notify = progress or (lambda message: None)
    root = Path(root)
    feature_columns = tuple(feature_columns)
    recipe = FeatureSpec(feature_columns)
    model_keys = tuple(model_keys)
    strategies = tuple(default_registry()) if strategies is None else tuple(strategies)
    if "equal_weight_universe" not in strategies:
        raise ValueError("Include the equal-weight portfolio in the holdout comparison.")
    dates = pd.DatetimeIndex(load_panel("train", root).date.unique())
    history_months = history_months_before(dates, origin)
    origin = pd.Timestamp(origin)
    data = prepare_data(root, "training_validation", history_months, cash_momentum_months=12)
    if pd.Timestamp(data.decision_start) != origin:
        raise AssertionError("The holdout origin does not match the prepared evaluation start.")
    if "test" in data.source_splits:
        raise AssertionError("The training holdout must not open the official test file.")
    training = data.model.training_slice(origin, feature_columns)
    if training.label_available_at.max() > origin:
        raise AssertionError("Holdout labels leaked into the tuning slice.")
    runner = ExperimentRunner(root)
    config = RunConfig(strategies, evaluation="training_validation", validation_history_months=history_months,
                       include_ml=True, adjustment_strength=adjustment_strength, feature_spec=recipe,
                       fit_policy="frozen_at_origin", prediction_target=prediction_target,
                       upside_threshold=upside_threshold, model_scope="pooled")
    monthly, metrics, predictive, searches = [], [], [], {}
    original_returns = None
    for model_key in model_keys:
        notify(f"Tuning {model_key} for {prediction_target} on labels known at {origin:%b %Y}")
        search = tune_classifier(training, recipe, model_key, origin, n_splits=n_splits,
                                 min_train_months=min_train_months, grid=grid, fixed_params=fixed_params,
                                 allow_slow=allow_slow, target=prediction_target)
        searches[model_key] = search
        tuned_registry = dict(default_model_registry())
        tuned_registry[model_key] = _spec_with_params(model_key, _flat_params(search.metadata))
        notify(f"Scoring fixed and tuned {model_key} on the training holdout")
        fixed = runner.run(RunConfig(**{**config.__dict__, "models": (model_key,)}), progress=notify)
        tuned = ExperimentRunner(root, model_registry=tuned_registry).run(
            RunConfig(**{**config.__dict__, "models": (model_key,)}), progress=notify)
        baseline = _baseline_returns(fixed)
        if original_returns is None:
            original_returns = baseline
            monthly.append(_monthly(fixed, "No ML", model_key))
            metrics.append(_metrics(fixed, "No ML", model_key))
        else:
            pd.testing.assert_frame_equal(baseline, original_returns)
        pd.testing.assert_frame_equal(_baseline_returns(tuned), original_returns)
        monthly.extend([_monthly(fixed, "ML, not tuned", model_key), _monthly(tuned, "Tuned ML", model_key)])
        metrics.extend([_metrics(fixed, "ML, not tuned", model_key), _metrics(tuned, "Tuned ML", model_key)])
        for version, result in (("ML, not tuned", fixed), ("Tuned ML", tuned)):
            batch = result.prediction_batches[model_key]
            predictive.append(batch.metrics.assign(version=version, model_key=model_key))
    holdout_months = original_returns.return_month
    if holdout_months.min() != return_month([origin])[0]:
        raise AssertionError("The holdout must start with the return earned after the origin.")
    if pd.Timestamp(data.audit["last_return_month"]) >= pd.Timestamp("2010-01-31"):
        raise AssertionError("The training holdout must end before the official test period.")
    return {
        "monthly": pd.concat(monthly, ignore_index=True),
        "metrics": pd.concat(metrics, ignore_index=True),
        "predictive": pd.concat(predictive, ignore_index=True),
        "searches": searches,
        "metadata": {
            "protocol": "training_holdout",
            "origin_snapshot": str(origin.date()),
            "trained_through_return_month": str(training.label_available_at.max().date()),
            "holdout_first_return_month": str(pd.Timestamp(holdout_months.min()).date()),
            "holdout_last_return_month": str(pd.Timestamp(holdout_months.max()).date()),
            "history_months": history_months,
            "feature_columns": list(feature_columns),
            "feature_source": feature_source or {"columns": list(feature_columns)},
            "model_keys": list(model_keys),
            "strategies": list(strategies),
            "adjustment_strength": adjustment_strength,
            "prediction_target": prediction_target,
            "upside_threshold": upside_threshold,
            "fit_policy": "frozen_at_origin",
            "test_csv_loaded": False,
            "official_test_period": "Jan 2010–Dec 2018, reserved for the final tab",
            "scoring": "neg_log_loss",
            "source_sha256": {"mlfactor_long_only_train.csv": _digest(root / "mlfactor_long_only_train.csv")},
            "code_sha256": {path.name: _digest(path) for path in
                            [Path(__file__), Path(__file__).with_name("tuning.py")]},
            "chosen": {key: {"params": value.metadata["chosen_params"], "best_score": value.metadata["best_score"],
                             "seconds": value.metadata["seconds"], "n_fits": value.metadata["n_fits"],
                             "fixed_params": value.metadata["fixed_params"],
                             "grid": value.metadata["grid"]} for key, value in searches.items()},
        },
    }


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_tuning_holdout(result, root=None):
    base = Path(root) if root is not None else PROJECT_ROOT / "artifacts" / "tuning_studies"
    directory = base / ("holdout_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    frames = {}
    for name in ["metrics", "monthly", "predictive"]:
        path = directory / f"{name}.parquet"
        result[name].to_parquet(path, index=False)
        frames[name] = _digest(path)
    for model_key, search in result["searches"].items():
        save_tuned_model(search, directory / model_key)
    metadata = dict(result["metadata"])
    metadata.update(created_utc=datetime.now(timezone.utc).isoformat(), frame_sha256=frames,
                    directory=str(directory))
    (directory / "study.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    return directory


def list_tuning_holdouts(root):
    base = Path(root) / "tuning_studies"
    if not base.is_dir():
        return []
    found = []
    for path in sorted(base.glob("*/study.json")):
        metadata = json.loads(path.read_text())
        if metadata.get("protocol") == "training_holdout":
            found.append({"directory": path.parent, "metadata": metadata})
    return found


def load_tuning_holdout(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "study.json").read_text())
    if metadata.get("protocol") != "training_holdout":
        raise ValueError("This directory is not a training-holdout tuning comparison.")
    frames = {}
    for name, expected in metadata["frame_sha256"].items():
        path = directory / f"{name}.parquet"
        if not path.is_file() or _digest(path) != expected:
            raise ValueError(f"Saved holdout file is missing or damaged: {name}.parquet")
        frames[name] = pd.read_parquet(path)
    return {"metadata": metadata, **frames}
