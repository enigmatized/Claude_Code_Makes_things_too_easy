"""Training-fitted, auditable feature filters compatible with sklearn Pipeline."""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted


class TrainingFeatureSelector(BaseEstimator, TransformerMixin):
    """Rank by ANOVA F or a training-only forest; optionally prune correlations.

    Scores are ranking statistics, not claims of independent statistical
    significance in this stock-month panel. Ties use original input order.
    Correlation pruning may return fewer than k features; it never backfills
    the basket with factors that fail the declared redundancy threshold.
    """
    def __init__(self, method="anova", k=10, correlation_threshold=.90, forest_parameters=None):
        self.method = method
        self.k = k
        self.correlation_threshold = correlation_threshold
        self.forest_parameters = forest_parameters

    def fit(self, X, y):
        values = np.asarray(X, dtype=float)
        target = np.asarray(y)
        if values.ndim != 2 or len(values) != len(target) or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("Selection needs finite two-dimensional training inputs and aligned labels.")
        if self.method not in {"anova", "correlation_pruned", "random_forest"}:
            raise ValueError("Unknown feature-selection method.")
        if not isinstance(self.k, int) or not 1 <= self.k <= values.shape[1]:
            raise ValueError("k must lie between one and the number of candidate factors.")
        if not 0 < self.correlation_threshold <= 1:
            raise ValueError("Correlation threshold must lie in (0, 1].")
        self.n_features_in_ = values.shape[1]
        self.feature_names_in_ = np.asarray(list(X.columns) if isinstance(X, pd.DataFrame) else
                                           [f"x{index}" for index in range(values.shape[1])], dtype=object)
        variable = np.ptp(values, axis=0) > 0
        if not variable.any():
            raise ValueError("No varying candidate factors in this training window.")
        self.scores_ = np.zeros(values.shape[1])
        self.importance_std_ = np.zeros(values.shape[1])
        self.constant_target_ = np.unique(target).size < 2
        self.forest_ = None
        if self.method == "random_forest" and not self.constant_target_:
            defaults = dict(n_estimators=100, max_depth=6, min_samples_leaf=50,
                            max_features="sqrt", random_state=0, n_jobs=1,
                            criterion="gini", bootstrap=True, class_weight=None)
            self.forest_ = RandomForestClassifier(**(defaults | (self.forest_parameters or {})))
            self.forest_.fit(values, target)
            self.scores_ = self.forest_.feature_importances_.copy()
            self.importance_std_ = np.std([tree.feature_importances_ for tree in self.forest_.estimators_], axis=0)
        elif not self.constant_target_:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                raw, _ = f_classif(values[:, variable], target)
            self.scores_[variable] = np.nan_to_num(raw, nan=0, posinf=np.finfo(float).max, neginf=0)
        self.uninformative_ranking_ = not np.any(self.scores_ > 0)
        frame = pd.DataFrame(values)
        self.correlations_ = frame.corr().fillna(0).to_numpy()
        order = np.argsort(-self.scores_, kind="stable")
        selected, reasons, blockers = [], {}, {}
        for index in order:
            if not variable[index]:
                reasons[index] = "constant in training"
                continue
            if len(selected) >= self.k:
                reasons[index] = "outside selected count"
                continue
            redundant = [other for other in selected if
                         abs(self.correlations_[index, other]) >= self.correlation_threshold - 1e-12]
            if self.method == "correlation_pruned" and redundant:
                blocker = max(redundant, key=lambda other: abs(self.correlations_[index, other]))
                blockers[index] = blocker
                reasons[index] = "correlated with a higher-ranked selected factor"
                continue
            selected.append(index)
            reasons[index] = ("selected by input-order fallback (uninformative forest ranking)"
                              if self.method == "random_forest" and self.uninformative_ranking_ else "selected")
        self.support_ = np.zeros(values.shape[1], dtype=bool)
        self.support_[selected] = True
        self.audit_ = pd.DataFrame([{
            "feature": str(self.feature_names_in_[index]), "score": float(self.scores_[index]),
            "score_type": "mean_decrease_impurity" if self.method == "random_forest" else "anova_f",
            "importance_std": float(self.importance_std_[index]) if self.method == "random_forest" else None,
            "ranking_informative": not self.uninformative_ranking_,
            "rank": rank, "selected": bool(self.support_[index]), "reason": reasons[index],
            "correlated_with": str(self.feature_names_in_[blockers[index]]) if index in blockers else "",
            "absolute_correlation": float(abs(self.correlations_[index, blockers[index]])) if index in blockers else 0.0,
        } for rank, index in enumerate(order, 1)])
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        if np.asarray(X).shape[1] != self.n_features_in_:
            raise ValueError("Feature columns differ from the fitted selector.")
        if isinstance(X, pd.DataFrame):
            if list(X.columns) != list(self.feature_names_in_):
                raise ValueError("Feature names/order differ from the fitted selector.")
            return X.loc[:, self.get_feature_names_out()].copy()
        return np.asarray(X)[:, self.support_]

    def get_support(self):
        check_is_fitted(self, "support_")
        return self.support_.copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "support_")
        return self.feature_names_in_[self.support_].copy()
