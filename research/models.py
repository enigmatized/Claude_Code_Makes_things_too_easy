"""Fixed loss predictors sharing one input contract and chronological schedule."""
from dataclasses import dataclass
import json
import os
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .data import model_data
from .features import FeatureSpec
from .legacy import FEATURES, KEYS, TARGET


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    description: str
    factory: Callable
    scale: bool = False

    def pipeline(self, feature_spec=None):
        recipe = feature_spec or FeatureSpec()
        if recipe.selection != "none":
            return Pipeline([("feature_selection", recipe.transformer()),
                             ("scaler", StandardScaler() if self.scale else "passthrough"), ("model", self.factory())])
        # Preserve the original no-selection pipeline, including operation order.
        return Pipeline([("scaler", StandardScaler() if self.scale else "passthrough"),
                         ("feature_selection", "passthrough"), ("model", self.factory())])

    def metadata(self, feature_spec=None, target="loss"):
        description = self.description if target == "loss" else self.description.replace("loss", "positive-return")
        return {"model_id": self.key, "label": self.label, "description": description,
                "estimator": type(self.factory()).__name__, "parameters": self.factory().get_params(),
                "preprocessing": "training-fitted StandardScaler" if self.scale else "none",
                "feature_selection": feature_spec.selection if feature_spec else "passthrough", "tuning": "none"}


def default_model_registry():
    return {
        "logistic": ModelSpec("logistic", "Logistic regression",
            "A linear relationship between factors and the log-odds of a loss.",
            lambda: LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=0), True),
        "naive_bayes": ModelSpec("naive_bayes", "Gaussian naive Bayes",
            "A simple probabilistic model treating factors as conditionally independent within each class.",
            lambda: GaussianNB(var_smoothing=1e-9)),
        "random_forest": ModelSpec("random_forest", "Random forest",
            "Average loss predictions from 100 shallow trees with fixed leaf-size limits.",
            lambda: RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=50,
                                           max_features="sqrt", n_jobs=1, random_state=0)),
        "hist_gradient_boosting": ModelSpec("hist_gradient_boosting", "Histogram gradient boosting",
            "Build 100 small trees sequentially. Random early stopping is disabled.",
            lambda: HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, learning_rate=.05,
                min_samples_leaf=50, l2_regularization=1.0, early_stopping=False, random_state=0)),
    }


@dataclass
class PredictionBatch:
    predictions: pd.DataFrame
    fits: pd.DataFrame
    metrics: pd.DataFrame
    calibration: pd.DataFrame


def event_fields(target="loss"):
    """Keep the historical loss schema; upside is a separate strictly-positive event."""
    if target not in {"loss", "upside"}:
        raise ValueError("Choose loss or upside prediction.")
    event = "loss" if target == "loss" else "up"
    return {"probability": f"probability_of_{event}", "historical": f"historical_{event}_rate",
            "observed": f"observed_{event}_rate", "mean_predicted": f"mean_predicted_{event}",
            "control_id": f"historical_{event}_rate", "event": event}


def event_labels(returns, target="loss"):
    event_fields(target)
    return (returns.lt(0) if target == "loss" else returns.gt(0)).astype(int)


def _fit_slice(training, current, feature_spec, model_spec, decision, fit_policy, target):
    """The same training-fitted preprocessing contract for pooled and local fits."""
    columns = list(feature_spec.columns)
    fields = event_fields(target)
    y = event_labels(training[TARGET], target)
    prevalence = float(y.mean())
    fallback = y.nunique() == 1
    selector = None
    if fallback:
        probabilities = np.full(len(current), prevalence)
        if feature_spec.selection != "none":
            selector = feature_spec.transformer().fit(training[columns], y)
    else:
        model = model_spec.pipeline(feature_spec)
        with threadpool_limits(limits=1):
            model.fit(training[columns], y)
            probabilities = model.predict_proba(current[columns])[:, list(model.classes_).index(1)]
        if feature_spec.selection != "none":
            selector = model.named_steps["feature_selection"]
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError(f"{model_spec.label} returned invalid probabilities.")
    through = training.earned_month.max()
    fitted_features = list(selector.get_feature_names_out()) if selector is not None else columns
    pred = current[KEYS].copy()
    pred[fields["probability"]] = probabilities
    pred["fit_snapshot"] = decision
    pred["trained_through_return_month"] = through
    pred[fields["historical"]] = prevalence
    fit = {
        "fit_snapshot": decision, "trained_through_return_month": through,
        "training_rows": len(training), "training_months": training.date.nunique(),
        fields["historical"]: prevalence, "prediction_rows": len(current),
        "constant_class_fallback": fallback, "model_id": model_spec.key,
        "feature_spec_id": feature_spec.feature_id, "input_features": json.dumps(columns),
        "fitted_features": json.dumps(fitted_features), "n_features": len(fitted_features),
        "n_input_features": len(columns), "selection_method": feature_spec.selection,
        "selection_audit": json.dumps(selector.audit_.to_dict("records")) if selector is not None else "[]",
        "selector_parameters": json.dumps(feature_spec.selector_parameters, sort_keys=True),
        "training_correlations": json.dumps({"columns": columns, "values": selector.correlations_.tolist()}) if selector is not None else "null",
        "fit_policy": fit_policy,
    }
    return pred, fit


def predict_model(data, feature_spec, model_spec, refit_months=12, progress=None, fit_policy="expanding",
                  target="loss", model_scope="pooled", min_stock_history=36, min_stock_class=6):
    if os.environ.get("MLFACTOR_READ_ONLY") == "1":
        raise RuntimeError("Model fitting is disabled during artifact-only rendering.")
    fields = event_fields(target)
    if model_scope not in {"pooled", "per_stock"}:
        raise ValueError("Choose pooled or per_stock model scope.")
    if type(min_stock_history) is not int or min_stock_history < 24 or type(min_stock_class) is not int or min_stock_class < 1:
        raise ValueError("Per-stock fits need at least 24 history months and a positive minimum class count.")
    columns = list(feature_spec.columns)
    for frame in [data.inputs, data.evaluation_inputs]:
        if not set(columns).issubset(frame) or not np.isfinite(frame[columns].to_numpy()).all():
            raise ValueError("Model inputs must contain the requested finite features.")
    if fit_policy not in {"expanding", "frozen_at_origin"}:
        raise ValueError("Choose expanding or frozen_at_origin model fits.")
    schedule = list(data.fit_schedule(refit_months))
    if fit_policy == "frozen_at_origin":
        schedule = [(schedule[0][0], pd.DatetimeIndex(data.evaluation_inputs.date.unique()).sort_values())]
    predictions, fits = [], []
    for decision, months in schedule:
        if progress:
            progress(f"{model_spec.label}: {model_scope} {target} fit at {decision:%b %Y}")
        training = data.training_slice(decision, columns)
        if not np.isfinite(training[TARGET]).all() or training[TARGET].lt(-1).any():
            raise ValueError("Training labels must be finite valid simple returns.")
        current = data.evaluation_inputs.loc[data.evaluation_inputs.date.isin(months)]
        pooled_pred, pooled_fit = _fit_slice(training, current, feature_spec, model_spec, decision, fit_policy, target)
        if model_scope == "pooled":
            predictions.append(pooled_pred)
            fits.append(pooled_fit)
            continue
        # Always keep a same-origin pooled model for short-history/unseen stocks.
        # Selection, scaling and estimation for eligible stocks use only that stock.
        pooled_pred = pooled_pred.set_index(KEYS)
        stock_history = dict(tuple(training.groupby("stock_id", sort=False)))
        fits.append({**pooled_fit, "stock_id": None, "model_scope": "pooled_fallback_model",
                     "prediction_source": "pooled", "fallback_reason": "", "stock_history_months": 0})
        for number, (stock_id, stock_current) in enumerate(current.groupby("stock_id", sort=True)):
            local = stock_history.get(stock_id, training.iloc[:0])
            history_months = local.date.nunique()
            counts = event_labels(local[TARGET], target).value_counts().reindex([0, 1], fill_value=0)
            reason = "short_history" if history_months < min_stock_history else (
                "insufficient_class_history" if counts.min() < min_stock_class else "")
            if reason:
                pred = pooled_pred.loc[pd.MultiIndex.from_frame(stock_current[KEYS])].reset_index()
                fit = {**pooled_fit, "prediction_rows": len(stock_current)}
            else:
                pred, fit = _fit_slice(local, stock_current, feature_spec, model_spec, decision, fit_policy, target)
                # The benchmark stays the same pooled past-frequency control in both scopes.
                pred[fields["historical"]] = pooled_fit[fields["historical"]]
            source = "pooled_fallback" if reason else "per_stock"
            pred["prediction_source"] = source
            pred["stock_history_months"] = history_months
            pred["fallback_reason"] = reason
            predictions.append(pred)
            fits.append({**fit, "stock_id": stock_id, "model_scope": "per_stock",
                         "prediction_source": source, "fallback_reason": reason,
                         "stock_history_months": history_months})
            if progress and number and number % 250 == 0:
                progress(f"{model_spec.label}: {number} stocks processed at {decision:%b %Y}")
    return pd.concat(predictions, ignore_index=True).sort_values(KEYS).reset_index(drop=True), pd.DataFrame(fits)


def predict_losses(history, evaluation_inputs, features=FEATURES, refit_months=12):
    """Compatibility entry point for the original logistic predictor."""
    return predict_model(model_data(history, evaluation_inputs), FeatureSpec(tuple(features)),
                         default_model_registry()["logistic"], refit_months)


def evaluate_predictions(predictions, evaluation_panel, fits, model_label="Logistic regression", target="loss"):
    fields = event_fields(target)
    scored = predictions.merge(evaluation_panel[KEYS + [TARGET]], on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not scored._merge.eq("both").all():
        raise ValueError("Predictions and evaluation rows must match.")
    y = event_labels(scored[TARGET], target)
    rows = []
    control_label = "Historical loss-rate control" if target == "loss" else "Historical up-rate control"
    for name, probabilities in [(model_label, scored[fields["probability"]]),
                                 (control_label, scored[fields["historical"]])]:
        rows.append({"model": name, "rows": len(y), fields["observed"]: y.mean(),
            "brier_score": brier_score_loss(y, probabilities), "log_loss": log_loss(y, probabilities, labels=[0, 1]),
            "roc_auc": roc_auc_score(y, probabilities) if y.nunique() == 2 else np.nan,
            "accuracy_at_half": ((probabilities >= .5) == y).mean()})
    bins = pd.cut(scored[fields["probability"]], np.linspace(0, 1, 11), include_lowest=True)
    calibration = scored.assign(event=y, probability_bin=bins).groupby("probability_bin", observed=True).agg(**{
        "predictions": ("event", "size"), fields["mean_predicted"]: (fields["probability"], "mean"),
        fields["observed"]: ("event", "mean"),
    }).reset_index()
    calibration["probability_bin"] = calibration.probability_bin.astype(str)
    return PredictionBatch(predictions, fits, pd.DataFrame(rows), calibration)
