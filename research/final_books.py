"""One frozen 2010–2018 score of the four presentation books.

This is the basket path that opens the official test file. The training study
still refuses a book that reaches January 2010. Browsing a saved result does
not call ``score_final_books``.

History for the groups is the training file followed by the test file, so the
first test snapshot has a 12-month path of returns already earned. The row at
snapshot D contributes its forward return to the next month-end, so that
return is outside the window that ends on D. The classifiers fit once, at the
first test snapshot, on labels known by that date.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from .accounting import portfolio_with_cash
from .adjustments import loss_probability, upside_probability
from .basket_adjustments import (
    BASKET_DOWNSIDE_VERSION, BASKET_UPSIDE_VERSION, BASKET_VERSION,
    DOWNSIDE_VERSION, UPSIDE_VERSION,
)
from .basket_momentum import (
    FEATURE, TOP_VOL_MOMENTUM, TOP_VOL_VERSION, _digest, _metrics, _monthly, _prepare_signals,
    assign_correlation_baskets, attach_short_momentum, concentrated_basket_weights,
    positive_momentum_cash_weights, positive_volatility, select_top_positive_vol,
)
from .data import model_data
from .features import FeatureSpec
from .legacy import FEATURES, KEYS, TARGET, load_panel, performance, portfolio_returns, return_month
from .models import default_model_registry, predict_model


PROTOCOL = "final_four_books"
# Graded equal-weight row in part0_data_and_benchmark/outputs/equal_weight_metrics.csv.
OFFICIAL_EQUAL_WEIGHT = {
    "cagr": 0.13096352989640528,
    "annualized_volatility": 0.15633245339181737,
    "sharpe": 0.8680950783847605,
}
RECIPE = (
    "Equal weight of the test cross-section. Cash earns nothing.",
    "Five K-means baskets on the trailing 12-month earned path. "
    "The top-10 book keeps the highest positive-vol names, then 1- and 2-month positive momentum. "
    "The full basket book uses the same cash rule at 1, 2, and 3 months.",
    "ANOVA top 10, chosen inside the single fit at the first test snapshot from labels known that day.",
    "Factory classifiers fit once at that snapshot, pooled, for a loss event and for an upside event.",
    "Downside haircut strength 0.5 stays in cash. Upside tilt strength 0.5 starts at probability 0.6 "
    "and keeps the original equity budget. The directions stay separate.",
    "If the top-10 book does not keep its 2005–2009 return lift, that lift was a training-window result. "
    "The recipe stays as written after the decade is scored.",
)


def _official_window(book):
    stats = performance(book.simple_return)
    first = pd.Timestamp(book.index.min())
    last = pd.Timestamp(book.index.max())
    official = first == pd.Timestamp("2010-01-31") and last == pd.Timestamp("2018-12-31") and len(book) == 108
    if official:
        for name, expected in OFFICIAL_EQUAL_WEIGHT.items():
            if not np.isclose(stats[name], expected, rtol=0, atol=1e-10):
                raise AssertionError(
                    f"Equal-weight test {name} is {stats[name]}. The graded benchmark is {expected}.")
    return stats, official


def _check_window(book, test):
    expected_first = pd.Timestamp(return_month([test.date.min()])[0])
    expected_last = pd.Timestamp(return_month([test.date.max()])[0])
    if (pd.Timestamp(book.index.min()) != expected_first or pd.Timestamp(book.index.max()) != expected_last
            or len(book) != test.date.nunique()):
        raise AssertionError("The scored months must be the test file's earned months.")


def _check_freeze(predictions, fits, panel, train_rows, origin, feature_spec):
    origin = pd.Timestamp(origin)
    if len(fits) != 1:
        raise AssertionError("A frozen test fit is one row.")
    snapshots = pd.to_datetime(predictions.fit_snapshot)
    through = pd.to_datetime(predictions.trained_through_return_month)
    if snapshots.nunique() != 1 or pd.Timestamp(snapshots.iloc[0]) != origin or through.max() > origin:
        raise AssertionError("Forecasts must come from the first test snapshot, with labels known by that date.")
    if int(fits.training_rows.iloc[0]) != int(train_rows):
        raise AssertionError("The frozen fit must use every training row and no test row.")
    expected = feature_spec.k if feature_spec.selection != "none" else len(feature_spec.columns)
    if not fits.n_features.eq(expected).all() or not fits.fit_policy.eq("frozen_at_origin").all():
        raise AssertionError("The frozen fit kept an unexpected feature count or policy.")
    matched = panel[KEYS].merge(predictions[KEYS], on=KEYS, how="outer", indicator=True)
    if not matched._merge.eq("both").all() or pd.to_datetime(predictions.fit_snapshot).gt(predictions.date).any():
        raise AssertionError("Forecasts must cover each test decision exactly once.")


def _baseline_weights(history, panel, decision_dates, progress):
    """Unadjusted books. Names outside a screen stay at weight zero."""
    assigned = assign_correlation_baskets(history, decision_dates, progress=progress)
    groups = panel[KEYS].merge(assigned[KEYS + [FEATURE]], on=KEYS, how="left", validate="one_to_one")
    volatility = positive_volatility(history, decision_dates, progress=progress)
    selected = select_top_positive_vol(groups, volatility)
    if not selected.empty and selected.groupby(["date", FEATURE]).size().gt(10).any():
        raise AssertionError("The positive-vol screen kept more than ten names.")
    weights = {}
    for months in (1, 2, 3):
        momentum = attach_short_momentum(history, groups, months)
        plain, grouped = _prepare_signals(momentum)
        weights[("Momentum + cash", months)] = positive_momentum_cash_weights(plain, by_basket=False)
        weights[(BASKET_VERSION, months)] = positive_momentum_cash_weights(grouped, by_basket=True)
        if months in TOP_VOL_MOMENTUM:
            full, _, _ = concentrated_basket_weights(history, panel, selected, months)
            weights[(TOP_VOL_VERSION, months)] = full[KEYS + ["weight"]]
    return weights


def _record(book, version, months, model_key, calendar):
    if not book.index.equals(calendar):
        raise AssertionError(f"{version} does not share the test months.")
    monthly = _monthly(book, version, months).assign(model_key=model_key)
    metrics = pd.DataFrame([_metrics(book, version, months)]).assign(model_key=model_key)
    return monthly, metrics


def _feature_map(fits):
    found = {}
    for row in fits.itertuples(index=False):
        found.setdefault(row.model_key, {})[row.prediction_target] = json.loads(row.fitted_features)
    return found


def score_final_books(root, models=None, feature_spec=None, strength=0.5, upside_threshold=0.6, progress=None):
    """Score the four books on the official test file. One call is the whole pass."""
    if os.environ.get("MLFACTOR_READ_ONLY") == "1":
        raise RuntimeError("Model fitting is disabled during artifact-only rendering.")
    root = Path(root)
    feature_spec = feature_spec or FeatureSpec(selection="anova", k=10)
    registry = default_model_registry()
    chosen = tuple(registry) if models is None else tuple(models)
    if not chosen or any(key not in registry for key in chosen):
        raise ValueError("Choose classifiers from the fixed four-model registry.")
    if progress:
        progress("Loading the training file and the official test file")
    train = load_panel("train", root)
    test = load_panel("test", root)
    if train.date.max() >= test.date.min():
        raise ValueError("Training snapshots must precede test snapshots.")
    decision_dates = pd.DatetimeIndex(test.date.unique()).sort_values()
    panel = test[KEYS + [TARGET]].copy()
    equal = portfolio_returns(panel)
    _check_window(equal, test)
    stats, official = _official_window(equal)
    if progress:
        progress(f"Equal weight CAGR {stats.cagr:.6f} over {len(equal)} months")
    history = pd.concat([train, test], ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    baseline = _baseline_weights(history, panel, decision_dates, progress)
    data = model_data(history, test[KEYS + list(FEATURES)])
    origin = decision_dates[0]
    fitted, fit_rows = {}, []
    for target in ("loss", "upside"):
        for key in chosen:
            if progress:
                progress(f"Frozen {target} fit: {registry[key].label}")
            predictions, fits = predict_model(
                data, feature_spec, registry[key], refit_months=12, progress=progress,
                fit_policy="frozen_at_origin", target=target)
            _check_freeze(predictions, fits, panel, len(train), origin, feature_spec)
            fitted.setdefault(key, {})[target] = predictions
            fit_rows.append(fits.assign(model_key=key, prediction_target=target))
    monthly = [_monthly(equal, "Equal weight", 0).assign(model_key="none")]
    metrics = [pd.DataFrame([_metrics(equal, "Equal weight", 0)]).assign(model_key="none")]
    calendar = equal.index
    for (version, months), weights in baseline.items():
        matched = panel[KEYS].merge(weights[KEYS + ["weight"]], on=KEYS, how="outer", indicator=True)
        if not matched._merge.eq("both").all():
            raise AssertionError(f"{version} weights do not cover the test panel.")
        book = portfolio_with_cash(panel, weights[KEYS + ["weight"]])
        month_frame, metric_frame = _record(book, version, months, "none", calendar)
        monthly.append(month_frame)
        metrics.append(metric_frame)
        if version not in (TOP_VOL_VERSION, BASKET_VERSION):
            continue
        downside_version = DOWNSIDE_VERSION if version == TOP_VOL_VERSION else BASKET_DOWNSIDE_VERSION
        upside_version = UPSIDE_VERSION if version == TOP_VOL_VERSION else BASKET_UPSIDE_VERSION
        for key in chosen:
            adjusted = {
                downside_version: loss_probability(
                    weights[KEYS + ["weight"]], fitted[key]["loss"], strength, allow_cash=True),
                upside_version: upside_probability(
                    weights[KEYS + ["weight"]], fitted[key]["upside"], strength, upside_threshold, allow_cash=True),
            }
            for overlay, allocation in adjusted.items():
                overlay_book = portfolio_with_cash(panel, allocation)
                month_frame, metric_frame = _record(overlay_book, overlay, months, key, calendar)
                monthly.append(month_frame)
                metrics.append(metric_frame)
    fits = pd.concat(fit_rows, ignore_index=True)
    metadata = {
        "protocol": PROTOCOL,
        "recipe": list(RECIPE),
        "fit_policy": "frozen_at_origin",
        "fit_snapshot": str(pd.Timestamp(origin).date()),
        "trained_through_return_month": str(pd.to_datetime(fits.trained_through_return_month).max().date()),
        "feature_spec": feature_spec.metadata(),
        "fitted_features": _feature_map(fits),
        "n_features": int(fits.n_features.iloc[0]),
        "models": list(chosen),
        "adjustment_strength": strength,
        "upside_threshold": upside_threshold,
        "new_prediction_fits": len(chosen) * 2,
        "tuning": "none on this window; factory classifier settings",
        "test_csv_loaded": True,
        "official_equal_weight_check": bool(official),
        "first_snapshot": str(pd.Timestamp(decision_dates.min()).date()),
        "last_snapshot": str(pd.Timestamp(decision_dates.max()).date()),
        "first_return_month": str(pd.Timestamp(equal.index.min()).date()),
        "last_return_month": str(pd.Timestamp(equal.index.max()).date()),
        "months": int(len(equal)),
        "training_rows": int(len(train)),
        "versions": [
            "Equal weight", "Momentum + cash", BASKET_VERSION, TOP_VOL_VERSION,
            DOWNSIDE_VERSION, UPSIDE_VERSION, BASKET_DOWNSIDE_VERSION, BASKET_UPSIDE_VERSION,
        ],
        "source_sha256": {
            "mlfactor_long_only_train.csv": _digest(root / "mlfactor_long_only_train.csv"),
            "mlfactor_long_only_test.csv": _digest(root / "mlfactor_long_only_test.csv"),
        },
        "cash_return": 0.0,
        "gross_of_costs": True,
    }
    return {
        "monthly": pd.concat(monthly, ignore_index=True),
        "metrics": pd.concat(metrics, ignore_index=True),
        "fits": fits,
        "metadata": metadata,
        "predictions": fitted,
    }


def save_final_books(result, root):
    """Write metrics, monthly returns, and the fit record. Not a catalog run."""
    if result["metadata"].get("protocol") != PROTOCOL or not result["metadata"].get("test_csv_loaded"):
        raise ValueError("Save only a scored final-book result.")
    base = Path(root) / "final_books"
    directory = base / ("final_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    frames = {}
    for name in ("metrics", "monthly", "fits"):
        path = directory / f"{name}.parquet"
        result[name].to_parquet(path, index=False)
        frames[name] = _digest(path)
    metadata = dict(result["metadata"])
    metadata.update(created_utc=datetime.now(timezone.utc).isoformat(), frame_sha256=frames, directory=str(directory))
    (directory / "study.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    return directory


def _rescore_monthly(monthly):
    """Recompute wealth, drawdown, and performance from saved simple returns."""
    required = {"version", "model_key", "momentum_months", "return_month", "simple_return",
                "snapshot_date", "cash_weight", "n_held", "turnover_one_way"}
    if monthly.empty or not required.issubset(monthly.columns):
        raise ValueError("Need saved monthly returns for each book.")
    metrics, curves = [], []
    grouping = ["version", "model_key", "momentum_months"]
    for key, block in monthly.groupby(grouping, sort=False):
        block = block.sort_values("return_month")
        if block.return_month.duplicated().any():
            raise ValueError("A book has two rows for one earned month.")
        stats = performance(block.set_index("return_month").simple_return).to_dict()
        wealth = (1 + block.simple_return).cumprod()
        curve = block.copy()
        curve["wealth"] = wealth.to_numpy()
        curve["drawdown"] = (wealth / wealth.cummax().clip(lower=1) - 1).to_numpy()
        stats.update(version=key[0], model_key=key[1], momentum_months=int(key[2]),
                     avg_one_way_turnover=float(block.turnover_one_way.iloc[1:].mean()),
                     avg_cash_weight=float(block.cash_weight.mean()),
                     avg_names=float(block.n_held.mean()))
        metrics.append(stats)
        curves.append(curve)
    return pd.DataFrame(metrics), pd.concat(curves, ignore_index=True)


def _stitch_months(training, test):
    """Place each saved training book immediately before its test book."""
    keys = ["version", "model_key", "momentum_months"]
    identity = lambda frame: {
        (version, model, int(horizon))
        for version, model, horizon in frame[keys].drop_duplicates().itertuples(index=False, name=None)
    }
    shared = identity(training) & identity(test)
    if not shared:
        raise ValueError("Training and test saves have no book in common.")
    frames = []
    for version, model, horizon in sorted(shared):
        earlier = training.loc[training.version.eq(version) & training.model_key.eq(model)
                               & training.momentum_months.eq(horizon)].sort_values("return_month")
        later = test.loc[test.version.eq(version) & test.model_key.eq(model)
                         & test.momentum_months.eq(horizon)].sort_values("return_month")
        junction = pd.Timestamp(earlier.return_month.max()) + pd.offsets.MonthEnd(1)
        if pd.Timestamp(later.return_month.min()) != junction:
            raise ValueError("Training and test months must meet, with no gap and no overlap.")
        frames.append(pd.concat([earlier, later], ignore_index=True))
    return pd.concat(frames, ignore_index=True)


def window_books(test_monthly, training_monthly=None):
    """Score the official test, and the training window and full path when that save exists.

    The full path compounds the saved training months and then the saved test months.
    It does not fit a model and it does not read a CSV.
    """
    windows = {"test": _rescore_monthly(test_monthly)}
    if training_monthly is None or training_monthly.empty:
        return windows
    windows["training"] = _rescore_monthly(training_monthly)
    windows["entire"] = _rescore_monthly(_stitch_months(training_monthly, test_monthly))
    return windows


def matching_training_monthly(root):
    """Monthly returns from the newest training study that has both overlay books."""
    from .basket_momentum import list_basket_studies

    for item in list_basket_studies(root):
        metadata = item["metadata"]
        versions = metadata.get("versions") or []
        if not (metadata.get("directional_adjustment") and "Baskets + downside to cash" in versions):
            continue
        directory = Path(item["directory"])
        path = directory / "monthly.parquet"
        expected = metadata.get("frame_sha256", {}).get("monthly")
        if not path.is_file() or expected is None or _digest(path) != expected:
            raise ValueError("Saved training monthly file is missing or damaged.")
        return pd.read_parquet(path), metadata
    return None, None


def list_final_books(root):
    base = Path(root) / "final_books"
    if not base.is_dir():
        return []
    found = []
    for path in sorted(base.glob("*/study.json"), reverse=True):
        metadata = json.loads(path.read_text())
        if metadata.get("protocol") == PROTOCOL:
            found.append({"directory": path.parent, "metadata": metadata})
    return found


def load_final_books(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "study.json").read_text())
    if metadata.get("protocol") != PROTOCOL:
        raise ValueError("This directory is not a final four-book study.")
    frames = {}
    for name, expected in metadata["frame_sha256"].items():
        path = directory / f"{name}.parquet"
        if not path.is_file() or _digest(path) != expected:
            raise ValueError(f"Saved final-book file is missing or damaged: {name}.parquet")
        frames[name] = pd.read_parquet(path)
    return {"metadata": metadata, **frames}
