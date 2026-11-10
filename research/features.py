"""Explicit model-input recipes, independent of portfolio allocation signals."""
from dataclasses import dataclass
import hashlib
import json
from .legacy import FEATURES


@dataclass(frozen=True)
class FeatureSpec:
    columns: tuple[str, ...] = tuple(FEATURES)
    selection: str = "none"
    k: int = 10
    correlation_threshold: float = .90
    forest_trees: int = 100
    forest_depth: int = 6
    forest_leaf: int = 50
    forest_seed: int = 0

    def __post_init__(self):
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError("Choose at least one distinct model input.")
        if not set(self.columns).issubset(FEATURES):
            raise ValueError("Model inputs must be original factors; IDs and future returns are excluded.")
        if self.selection not in {"none", "anova", "correlation_pruned", "random_forest"}:
            raise ValueError("Choose none, anova, correlation_pruned or random_forest feature selection.")
        if self.selection != "none" and (not isinstance(self.k, int) or not 1 <= self.k <= min(10, len(self.columns))):
            raise ValueError("Learned selection must retain between one and ten candidate factors.")
        if not 0 < self.correlation_threshold <= 1:
            raise ValueError("Correlation threshold must lie in (0, 1].")
        if any(type(value) is not int or value < 1 for value in
               (self.forest_trees, self.forest_depth, self.forest_leaf)):
            raise ValueError("Forest tree count, depth and minimum leaf size must be positive integers.")
        if type(self.forest_seed) is not int or not 0 <= self.forest_seed < 2**32:
            raise ValueError("Forest seed must be an integer in [0, 2**32).")

    @property
    def selector_parameters(self):
        """Independent selector settings, separate from the downstream predictor."""
        if self.selection != "random_forest":
            return {}
        return dict(n_estimators=self.forest_trees, max_depth=self.forest_depth,
                    min_samples_leaf=self.forest_leaf, random_state=self.forest_seed,
                    max_features="sqrt", criterion="gini", bootstrap=True,
                    class_weight=None, n_jobs=1)

    @property
    def feature_id(self):
        settings = {"columns": self.columns, "selection": self.selection}
        if self.selection != "none":
            settings.update(k=self.k)
            if self.selection == "correlation_pruned":
                settings.update(correlation_threshold=self.correlation_threshold)
            if self.selection == "random_forest":
                settings.update(selector_parameters=self.selector_parameters)
        payload = json.dumps(settings, sort_keys=True)
        return "factors_" + hashlib.sha256(payload.encode()).hexdigest()[:12]

    @property
    def label(self):
        if self.selection == "anova":
            return f"ANOVA top {self.k}"
        if self.selection == "correlation_pruned":
            return f"Correlation-pruned up to {self.k} (|r| < {self.correlation_threshold:.2f})"
        if self.selection == "random_forest":
            return f"Random forest top {self.k}"
        return "All 20 original factors" if self.columns == tuple(FEATURES) else f"Original-factor subset ({len(self.columns)})"

    def transformer(self):
        if self.selection == "none":
            return "passthrough"
        from .selection import TrainingFeatureSelector
        return TrainingFeatureSelector(self.selection, self.k, self.correlation_threshold,
                                       forest_parameters=self.selector_parameters or None)

    def metadata(self):
        return {"feature_spec_id": self.feature_id, "label": self.label, "columns": list(self.columns),
                "selection": self.selection, "maximum_selected": self.k if self.selection != "none" else len(self.columns),
                "correlation_threshold": self.correlation_threshold if self.selection == "correlation_pruned" else None,
                "selector_parameters": self.selector_parameters,
                "lineage": {c: f"source CSV: {c}" for c in self.columns}}
