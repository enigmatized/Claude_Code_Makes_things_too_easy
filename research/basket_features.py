"""Score the four presentation books under feature recipes that are already fitted.

Equal weight and the momentum books do not use the factors, so they are stored
once. The up-or-down overlays are stored once per recipe. This module does not
fit a model and does not read the official test file.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from .basket_adjustments import (
    ADJUSTED_VERSIONS, BASKET_ADJUSTED_VERSIONS, BASKET_VERSION, create_adjusted_basket_study,
)
from .basket_momentum import TOP_VOL_VERSION


BASELINE_VERSIONS = ("Equal weight", TOP_VOL_VERSION, BASKET_VERSION)
OVERLAY_VERSIONS = (*ADJUSTED_VERSIONS, *BASKET_ADJUSTED_VERSIONS)


def score_feature_books(store, basket_directory, recipes):
    """Replay saved forecasts for each recipe and keep portfolio results only.

    ``recipes`` is an ordered list of dicts with ``key``, ``label``,
    ``loss_run_id`` and ``upside_run_id``. The first recipe is the all-20
    reference used for the unchanged books.
    """
    if len(recipes) < 2 or recipes[0]["key"] != "all_20":
        raise ValueError("The feature-book score needs the all-20 recipe first, then at least one cut.")
    metrics_parts, monthly_parts, sources = [], [], []
    reference = None
    for recipe in recipes:
        result = create_adjusted_basket_study(
            store, basket_directory, recipe["loss_run_id"], recipe["upside_run_id"],
            keep_allocations=False)
        if result["metadata"].get("test_csv_loaded") or result["metadata"].get("new_prediction_fits"):
            raise AssertionError("Feature-book scores must reuse training forecasts.")
        metrics = result["metrics"]
        monthly = result["monthly"]
        baseline = metrics.loc[metrics.model_key.eq("none") & metrics.version.isin(BASELINE_VERSIONS)].copy()
        ordered = baseline.sort_values(["version", "momentum_months"]).reset_index(drop=True)
        if reference is None:
            reference = ordered
            metrics_parts.append(ordered.assign(recipe=""))
            base_monthly = monthly.loc[monthly.model_key.eq("none") & monthly.version.isin(BASELINE_VERSIONS)]
            monthly_parts.append(base_monthly.assign(recipe=""))
        else:
            check = ordered[["version", "momentum_months", "cagr", "sharpe"]].reset_index(drop=True)
            prior = reference[["version", "momentum_months", "cagr", "sharpe"]].reset_index(drop=True)
            if not check[["version", "momentum_months"]].equals(prior[["version", "momentum_months"]]):
                raise AssertionError("Feature recipes do not share the non-ML books.")
            if not np.allclose(check.cagr, prior.cagr) or not np.allclose(check.sharpe, prior.sharpe):
                raise AssertionError("A feature recipe changed a book that does not use factors.")
        overlay = metrics.loc[metrics.version.isin(OVERLAY_VERSIONS)].copy()
        metrics_parts.append(overlay.assign(recipe=recipe["key"]))
        overlay_monthly = monthly.loc[monthly.version.isin(OVERLAY_VERSIONS)].copy()
        monthly_parts.append(overlay_monthly.assign(recipe=recipe["key"]))
        sources.append({
            "key": recipe["key"], "label": recipe["label"],
            "loss_run_id": recipe["loss_run_id"], "upside_run_id": recipe["upside_run_id"],
            "feature_selection": result["metadata"]["feature_selection"],
        })
    metadata = {
        "protocol": "basket_feature_books",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "basket_directory": str(basket_directory),
        "recipes": sources,
        "new_prediction_fits": 0,
        "test_csv_loaded": False,
        "window": {
            "first_return_month": result["metadata"]["first_return_month"],
            "last_return_month": result["metadata"]["last_return_month"],
            "months": result["metadata"]["months"],
        },
    }
    return {
        "metrics": pd.concat(metrics_parts, ignore_index=True),
        "monthly": pd.concat(monthly_parts, ignore_index=True),
        "metadata": metadata,
    }


def save_feature_book_scores(result, root):
    directory = Path(root) / "basket_feature_scores" / (
        "features_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    result["metrics"].to_parquet(directory / "metrics.parquet", index=False)
    result["monthly"].to_parquet(directory / "monthly.parquet", index=False)
    (directory / "study.json").write_text(json.dumps(result["metadata"], indent=2) + "\n")
    return directory


def list_feature_book_scores(root):
    base = Path(root) / "basket_feature_scores"
    if not base.is_dir():
        return []
    found = []
    for path in sorted(base.glob("*/study.json"), reverse=True):
        metadata = json.loads(path.read_text())
        if metadata.get("protocol") == "basket_feature_books":
            found.append({"directory": path.parent, "metadata": metadata})
    return found


def load_feature_book_scores(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "study.json").read_text())
    if metadata.get("protocol") != "basket_feature_books":
        raise ValueError("This directory is not a four-book feature score.")
    return {
        "metadata": metadata,
        "metrics": pd.read_parquet(directory / "metrics.parquet"),
        "monthly": pd.read_parquet(directory / "monthly.parquet"),
    }
