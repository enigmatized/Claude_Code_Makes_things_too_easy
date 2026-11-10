"""Replay saved directional forecasts on the basket books.

No estimators are fitted here. Both targets were fitted separately at historical
origins in the Basic ML research. The original momentum weights are the common
starting point; a downside haircut releases capital to cash, while an upside
threshold boost redistributes only the original stock budget. The same two
adjustments are applied to the top-10 positive-vol book and to the full
correlation-basket book. They stay separate versions.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .accounting import portfolio_with_cash
from .adjustments import loss_probability, upside_probability
from .basket_momentum import (
    TOP_VOL_VERSION, _basket_table, _digest, _metrics, _monthly, load_basket_study,
)
from .legacy import KEYS, TARGET, load_panel
from .models import default_model_registry


DOWNSIDE_VERSION = "Top 10 + downside to cash"
UPSIDE_VERSION = "Top 10 + upside tilt"
ADJUSTED_VERSIONS = (DOWNSIDE_VERSION, UPSIDE_VERSION)
BASKET_VERSION = "Basket momentum + cash"
BASKET_DOWNSIDE_VERSION = "Baskets + downside to cash"
BASKET_UPSIDE_VERSION = "Baskets + upside tilt"
BASKET_ADJUSTED_VERSIONS = (BASKET_DOWNSIDE_VERSION, BASKET_UPSIDE_VERSION)


def _read_forecasts(store, run_id, target, panel, source_hash):
    record, directory, metadata = store._record(run_id)
    config = metadata["config"]
    if (config.get("evaluation") != "training_validation"
            or config.get("prediction_target", "loss") != target
            or config.get("model_scope", "pooled") != "pooled"):
        raise ValueError("Use a training-validation pooled run for the requested target.")
    if metadata["manifest"]["source_sha256"] != {"mlfactor_long_only_train.csv": source_hash}:
        raise ValueError("Saved forecasts and basket replay must use the same training file.")
    batches = {}
    for key, paths in metadata["frame_paths"]["prediction_batches"].items():
        predictions = pd.read_parquet(directory / paths["predictions"])
        matched = panel[KEYS].merge(predictions, on=KEYS, how="outer", validate="one_to_one", indicator=True)
        if not matched._merge.eq("both").all():
            raise ValueError("Forecasts must cover exactly the basket evaluation stock-months.")
        if (predictions[["fit_snapshot", "trained_through_return_month"]].isna().any().any()
                or pd.to_datetime(predictions.trained_through_return_month).gt(predictions.fit_snapshot).any()
                or pd.to_datetime(predictions.fit_snapshot).gt(predictions.date).any()):
            raise ValueError("Forecast provenance contains future training outcomes.")
        batches[key] = {name: pd.read_parquet(directory / paths[name])
                        for name in ("fits", "metrics")}
        batches[key]["predictions"] = predictions
    reference = {"run_id": run_id, "artifact_sha256": record["artifact_sha256"],
                 "feature_spec": config["feature_spec"], "model_scope": config.get("model_scope", "pooled"),
                 "fit_policy": config["fit_policy"], "refit_months": config["refit_months"],
                 "models": metadata["manifest"]["models"]}
    return batches, reference


def create_adjusted_basket_study(store, basket_directory, loss_run_id, upside_run_id,
                                 strength=.5, upside_threshold=.6, keep_allocations=True):
    """Create a new matched study, preserving the source basket study verbatim.

    All baseline frames gain model_key='none'. New rows identify the predictor,
    horizon and adjustment. Both the top-10 book and the full basket book keep
    their original weights, event probabilities and fitting cutoffs.
    """
    if os.environ.get("MLFACTOR_READ_ONLY") == "1":
        raise RuntimeError("Portfolio replay is disabled during artifact-only rendering.")
    basket_directory = Path(basket_directory)
    base = load_basket_study(basket_directory)
    if base["metadata"].get("directional_adjustment"):
        raise ValueError("Start from an unadjusted basket study.")
    selected = base["assignments"].loc[base["assignments"].version.eq(TOP_VOL_VERSION)].copy()
    if selected.empty:
        raise ValueError("The source study has no top-positive-volatility basket book.")
    train_path = store.project_root / "mlfactor_long_only_train.csv"
    source_hash = _digest(train_path)
    train = load_panel("train", store.project_root)
    panel = train.loc[train.date.between(base["metadata"]["first_snapshot"],
                                         base["metadata"]["last_snapshot"]), KEYS + [TARGET]].copy()
    if panel.empty or pd.Timestamp(base["metadata"]["last_return_month"]) >= pd.Timestamp("2010-01-31"):
        raise ValueError("Basket overlays use only training-validation months before 2010.")
    losses, loss_source = _read_forecasts(store, loss_run_id, "loss", panel, source_hash)
    upsides, up_source = _read_forecasts(store, upside_run_id, "upside", panel, source_hash)
    if (loss_source["feature_spec"] != up_source["feature_spec"]
            or loss_source["fit_policy"] != up_source["fit_policy"]
            or loss_source["refit_months"] != up_source["refit_months"]
            or [{k: v for k, v in model.items() if k != "description"} for model in loss_source["models"]]
               != [{k: v for k, v in model.items() if k != "description"} for model in up_source["models"]]):
        raise ValueError("Both directional sources must share features, model settings and fitting schedule.")
    models = [key for key in default_model_registry() if key in losses and key in upsides]
    if set(models) != set(losses) or set(models) != set(upsides):
        raise ValueError("Both directional sources must contain the same supported models.")
    for key in models:
        provenance = KEYS + ["fit_snapshot", "trained_through_return_month"]
        if not losses[key]["predictions"][provenance].sort_values(KEYS).reset_index(drop=True).equals(
                upsides[key]["predictions"][provenance].sort_values(KEYS).reset_index(drop=True)):
            raise ValueError("Both directional forecasts must use the same historical fitting origins.")
    frame_names = ("monthly", "metrics", "baskets", "assignments") if keep_allocations else ("monthly", "metrics")
    frames = {name: [base[name].assign(model_key="none")] for name in frame_names}
    books = (
        (TOP_VOL_VERSION, DOWNSIDE_VERSION, UPSIDE_VERSION),
        (BASKET_VERSION, BASKET_DOWNSIDE_VERSION, BASKET_UPSIDE_VERSION),
    )
    for source_version, downside_version, upside_version in books:
        selected = base["assignments"].loc[base["assignments"].version.eq(source_version)].copy()
        if selected.empty:
            if source_version == TOP_VOL_VERSION:
                raise ValueError("The source study has no top-positive-volatility basket book.")
            raise ValueError(f"The source study has no {source_version} book.")
        for months, members in selected.groupby("momentum_months", sort=True):
            weights = panel[KEYS].merge(members[KEYS + ["weight"]], on=KEYS, how="left", validate="one_to_one")
            weights["weight"] = weights.weight.fillna(0.)
            original = portfolio_with_cash(panel, weights)
            saved = base["monthly"].loc[base["monthly"].version.eq(source_version)
                                        & base["monthly"].momentum_months.eq(months)].sort_values("return_month")
            if not np.array_equal(original.simple_return.to_numpy(), saved.simple_return.to_numpy()):
                raise ValueError("Source basket returns do not reproduce from its saved allocations.")
            for key in models:
                for version, batch, target in ((downside_version, losses[key], "loss"),
                                               (upside_version, upsides[key], "upside")):
                    if target == "loss":
                        adjusted = loss_probability(weights, batch["predictions"], strength, allow_cash=True)
                    else:
                        adjusted = upside_probability(weights, batch["predictions"], strength,
                                                       upside_threshold, allow_cash=True)
                    book = portfolio_with_cash(panel, adjusted)
                    if not book.index.equals(original.index):
                        raise AssertionError("Adjusted and original calendars differ.")
                    frames["monthly"].append(_monthly(book, version, months).assign(model_key=key))
                    frames["metrics"].append(pd.DataFrame([_metrics(book, version, months)]).assign(model_key=key))
                    if not keep_allocations:
                        continue
                    audit = members.drop(columns=["version", "weight"]).merge(
                        members[KEYS + ["weight"]].rename(columns={"weight": "baseline_weight"}),
                        on=KEYS, validate="one_to_one").merge(adjusted, on=KEYS, validate="one_to_one").merge(
                        batch["predictions"], on=KEYS, validate="one_to_one")
                    frames["assignments"].append(audit.assign(version=version, model_key=key))
                    frames["baskets"].append(_basket_table(members.drop(columns="weight"),
                        audit[KEYS + ["weight"]]).assign(version=version, model_key=key, momentum_months=months))
    result = {name: pd.concat(parts, ignore_index=True) for name, parts in frames.items()}
    if keep_allocations:
        # Scores are for the full forecast universe, not just screened holdings.
        predictive, fits = [], []
        for target, batches in (("loss", losses), ("upside", upsides)):
            for key in models:
                predictive.append(batches[key]["metrics"].assign(model_key=key, prediction_target=target))
                fits.append(batches[key]["fits"].assign(model_key=key, prediction_target=target))
        result["predictive"] = pd.concat(predictive, ignore_index=True)
        result["fits"] = pd.concat(fits, ignore_index=True)
    result["metadata"] = {**base["metadata"],
        "directional_adjustment": True, "model_keys": models,
        "adjustment_strength": strength, "upside_threshold": upside_threshold,
        "versions": [*base["metadata"]["versions"], *ADJUSTED_VERSIONS, *BASKET_ADJUSTED_VERSIONS],
        "basket_source": {"directory": str(basket_directory),
                          "study_sha256": _digest(basket_directory / "study.json")},
        "forecast_sources": {"loss": loss_source, "upside": up_source},
        "source_sha256": {train_path.name: source_hash},
        "adjustment_code_sha256": {name: _digest(Path(__file__).parent / name) for name in
                                   ("basket_adjustments.py", "adjustments.py", "accounting.py")},
        "adjustment_rule": "Downside: w*(1-strength*p_loss), haircut to cash. Upside: "
                           "w*(1+strength*I(p_up>=threshold)), rescaled to original equity budget. "
                           "Separate fitted targets, same momentum baseline; no combined overlay.",
        "prediction_fits_reused": True, "new_prediction_fits": 0,
        "feature_selection": loss_source["feature_spec"].get("selection", "none"),
        "tuning": "none; fixed Basic ML classifiers", "test_csv_loaded": False}
    return result
