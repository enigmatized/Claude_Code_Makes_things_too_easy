"""Matched feature-recipe experiments and read-only comparisons of saved runs."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid

import numpy as np
import pandas as pd

from .features import FeatureSpec
from .store import CALCULATION_FILES, digest_file


RECIPES = {"all_20": FeatureSpec(), "anova_10": FeatureSpec(selection="anova", k=10),
           "pruned_10": FeatureSpec(selection="correlation_pruned", k=10, correlation_threshold=.90)}
FOREST_RECIPES = {**RECIPES, "forest_10": FeatureSpec(selection="random_forest", k=10)}


def list_feature_studies(store):
    if not store.catalog.exists():
        return []
    with sqlite3.connect(store.catalog) as connection:
        connection.row_factory = sqlite3.Row
        exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='studies'").fetchone()
        return [dict(row) for row in connection.execute("SELECT * FROM studies ORDER BY created_utc DESC")] if exists else []


def compare_feature_runs(store, runs):
    """Require identical original books/calendars; join model cases explicitly."""
    summaries = {key: store.summary(run_id) for key, run_id in runs.items()}
    base = summaries["all_20"]
    baseline_config = dict(base["manifest"]["config"])
    baseline_config.pop("feature_spec")
    rows, forecasts, fits, calibrations = [], [], [], []
    reference = base["metrics"].loc[base["metrics"].variant.ne("baseline")].set_index(["strategy", "model_id"])
    original = base["monthly"].loc[base["monthly"].variant.eq("baseline")].sort_values(["strategy", "return_month"])
    for recipe, summary in summaries.items():
        config = dict(summary["manifest"]["config"])
        config.pop("feature_spec")
        if config != baseline_config or summary["manifest"]["source_sha256"] != base["manifest"]["source_sha256"]:
            raise ValueError("Feature comparisons must share models, strategies, data, dates and fitting policy.")
        if summary["manifest"]["versions"] != base["manifest"]["versions"] or any(
                summary["manifest"]["code_sha256"].get(name) != base["manifest"]["code_sha256"].get(name)
                for name in CALCULATION_FILES):
            raise ValueError("Feature comparisons must share calculation code and numerical runtime.")
        current_original = summary["monthly"].loc[summary["monthly"].variant.eq("baseline")].sort_values(["strategy", "return_month"])
        if not current_original[["strategy", "return_month"]].reset_index(drop=True).equals(original[["strategy", "return_month"]].reset_index(drop=True)):
            raise ValueError("Original portfolio calendars differ across recipes.")
        np.testing.assert_array_equal(current_original.simple_return.to_numpy(), original.simple_return.to_numpy())
        keys = ["strategy", "model_id", "variant", "return_month"]
        pd.testing.assert_frame_equal(summary["monthly"][keys].sort_values(keys).reset_index(drop=True),
                                      base["monthly"][keys].sort_values(keys).reset_index(drop=True))
        for _, row in summary["metrics"].loc[summary["metrics"].variant.ne("baseline")].iterrows():
            record = row.to_dict()
            record.update(recipe=recipe, run_id=runs[recipe])
            for metric in ["cagr", "sharpe", "sortino", "max_drawdown", "avg_one_way_turnover"]:
                record[f"change_{metric}_vs_all20"] = row[metric] - reference.loc[(row.strategy, row.model_id), metric]
            rows.append(record)
        forecasts.append(summary["model_metrics"].assign(recipe=recipe, run_id=runs[recipe]))
        fits.append(summary["model_fits"].assign(recipe=recipe, run_id=runs[recipe]))
        calibrations.append(summary["model_calibration"].assign(recipe=recipe, run_id=runs[recipe]))
    return {"summaries": summaries, "portfolio_comparison": pd.DataFrame(rows),
            "predictive_comparison": pd.concat(forecasts, ignore_index=True),
            "fits": pd.concat(fits, ignore_index=True),
            "calibration": pd.concat(calibrations, ignore_index=True)}


def create_feature_study(store, config, progress=None, recipes=None):
    recipes = dict(RECIPES if recipes is None else recipes)
    if "all_20" not in recipes or recipes["all_20"] != FeatureSpec() or len(recipes) < 2:
        raise ValueError("A feature study needs the all-20 reference and at least one selected recipe.")
    if any(recipe.selection == "none" for key, recipe in recipes.items() if key != "all_20"):
        raise ValueError("Comparison recipes must declare learned feature selection.")
    if config.evaluation != "training_validation" or not config.include_ml:
        raise ValueError("The initial feature study must use training-period validation with ML enabled.")
    if not {"logistic", "naive_bayes", "random_forest", "hist_gradient_boosting"}.issubset(config.models):
        raise ValueError("Compare all four fixed models in this initial study.")
    store._initialize()
    study_id = "features_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    folder = store.root / "studies" / study_id
    folder.mkdir(parents=True)
    plan = {"study_id": study_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "label": "Feature selection: " + " vs ".join(recipe.label for recipe in recipes.values()),
        "config": asdict(config), "recipes": {key: value.metadata() for key, value in recipes.items()},
        "protocol": "Annual selectors fitted inside historical training windows; fixed models/alpha; no test reads or tuning."}
    (folder / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")  # Written before results.
    runs = {}
    for key, recipe in recipes.items():
        if progress:
            progress(f"Feature recipe: {recipe.label}")
        result = store.get_or_run(replace(config, feature_spec=recipe), progress=progress)
        runs[key] = result.cache_info["run_id"]
        del result
    comparison = compare_feature_runs(store, runs)
    if not comparison["fits"].groupby(["recipe", "fit_snapshot"]).fitted_features.nunique().eq(1).all():
        raise AssertionError("Matched feature recipes produced different masks across predictors.")
    for name in ["portfolio_comparison", "predictive_comparison", "fits"]:
        comparison[name].to_csv(folder / f"{name}.csv", index=False)
    record = {**plan, "runs": runs,
        "checks": {"identical_original_returns": True, "same_evaluation_calendar": True,
                   "same_masks_across_predictors": True,
                   "test_csv_loaded": False, "hyperparameter_search": False}}
    for key, summary in comparison["summaries"].items():
        if key != "all_20" and not summary["model_fits"].n_features.le(10).all():
            raise AssertionError("A selected model exceeded ten factors.")
    path = folder / "study.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    with sqlite3.connect(store.catalog) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS studies (study_id TEXT PRIMARY KEY, label TEXT NOT NULL, created_utc TEXT NOT NULL, relative_path TEXT NOT NULL, sha256 TEXT NOT NULL)")
        connection.execute("INSERT INTO studies VALUES (?, ?, ?, ?, ?)",
            (study_id, record["label"], record["created_utc"], str(path.relative_to(store.root)), digest_file(path)))
    return record


def load_feature_study(store, study_id):
    record = next((row for row in list_feature_studies(store) if row["study_id"] == study_id), None)
    if record is None:
        raise ValueError("Unknown feature-selection study.")
    path = store.root / record["relative_path"]
    if digest_file(path) != record["sha256"]:
        raise ValueError("Feature-study metadata failed its integrity check.")
    metadata = json.loads(path.read_text())
    return {"metadata": metadata, **compare_feature_runs(store, metadata["runs"])}


def feature_audit(fits):
    rows = []
    for _, fit in fits.iterrows():
        for item in json.loads(fit.get("selection_audit", "[]")):
            rows.append({**item, "model_id": fit.model_id, "fit_snapshot": fit.fit_snapshot,
                         "recipe": fit.get("recipe", ""), "training_rows": fit.training_rows,
                         "trained_through_return_month": fit.trained_through_return_month})
    return pd.DataFrame(rows)
