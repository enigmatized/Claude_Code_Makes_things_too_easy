"""Forest selection scores, chronological isolation and configuration identity."""
from dataclasses import asdict, replace
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import FeatureSpec, RunConfig, default_model_registry
from research.data import model_data
from research.models import predict_model
from research.store import config_from_dict
from test_research import toy_panel


class ForestSelectorTests(unittest.TestCase):
    def test_importance_and_tree_variation_match_sklearn_and_transform_keeps_order(self):
        rng = np.random.default_rng(42)
        x = pd.DataFrame(rng.normal(size=(400, 5)), columns=list(FeatureSpec().columns[:5]))
        y = (x.iloc[:, 2] + .3 * x.iloc[:, 4] > 0).astype(int)
        recipe = FeatureSpec(tuple(x), selection="random_forest", k=2, forest_trees=20, forest_leaf=2)
        selector = clone(recipe.transformer()).fit(x, y)
        forest = RandomForestClassifier(**recipe.selector_parameters).fit(x.to_numpy(), y)
        np.testing.assert_array_equal(selector.scores_, forest.feature_importances_)
        np.testing.assert_array_equal(selector.importance_std_, np.std(
            [tree.feature_importances_ for tree in forest.estimators_], axis=0))
        self.assertEqual(selector.get_support().sum(), 2)
        self.assertEqual(list(selector.transform(x)), [c for c, keep in zip(x, selector.get_support()) if keep])
        self.assertTrue(selector.audit_.score_type.eq("mean_decrease_impurity").all())
        with self.assertRaises(ValueError):
            selector.transform(x[list(x)[::-1]])
        np.testing.assert_array_equal(selector.get_support(), recipe.transformer().fit(x, y).get_support())

    def test_constant_and_zero_importance_fallback_is_explicit_and_excludes_constants(self):
        x = pd.DataFrame({"a": np.arange(40), "constant": 1, "b": np.arange(40)[::-1]})
        recipe = FeatureSpec(selection="random_forest", k=2, forest_trees=5, forest_leaf=50)
        for target in (np.zeros(40), np.tile([0, 1], 20)):
            selector = recipe.transformer().fit(x, target)
            self.assertTrue(selector.uninformative_ranking_)
            self.assertEqual(list(selector.get_feature_names_out()), ["a", "b"])
            self.assertTrue(selector.audit_.loc[selector.audit_.selected, "reason"].str.contains("fallback").all())
            self.assertEqual(selector.scores_.sum(), 0)
        with self.assertRaises(ValueError):
            recipe.transformer().fit(x * 0, np.zeros(40))

    def test_identity_roundtrip_and_legacy_defaults(self):
        recipe = FeatureSpec(selection="random_forest", forest_trees=17, forest_seed=9)
        config = RunConfig(feature_spec=recipe)
        self.assertEqual(config_from_dict(asdict(config)), config)
        self.assertNotEqual(recipe.feature_id, replace(recipe, forest_seed=10).feature_id)
        self.assertNotEqual(recipe.feature_id, replace(recipe, forest_leaf=10).feature_id)
        legacy = asdict(RunConfig())
        for field in ("forest_trees", "forest_depth", "forest_leaf", "forest_seed"):
            legacy["feature_spec"].pop(field)
        self.assertEqual(config_from_dict(legacy).feature_spec.feature_id, "factors_8d5fa4272617")

    def test_future_labels_cannot_change_first_mask_importance_or_prediction(self):
        history = toy_panel(48, 8)
        cutoff = history.date.drop_duplicates().iloc[30]
        changed = history.copy()
        changed.loc[changed.date.ge(cutoff), "R1M_Usd"] = -.9
        changed.loc[changed.date.gt(cutoff), list(FeatureSpec().columns)] = .99
        recipe = FeatureSpec(selection="random_forest", forest_trees=10, forest_leaf=2)
        model = default_model_registry()["logistic"]
        p1, f1 = predict_model(model_data(history, history.loc[history.date.ge(cutoff)]), recipe, model)
        p2, f2 = predict_model(model_data(changed, changed.loc[changed.date.ge(cutoff)]), recipe, model)
        self.assertEqual(f1.selection_audit.iloc[0], f2.selection_audit.iloc[0])
        self.assertEqual(f1.fitted_features.iloc[0], f2.fitted_features.iloc[0])
        np.testing.assert_array_equal(p1.loc[p1.date.eq(cutoff), "probability_of_loss"],
                                      p2.loc[p2.date.eq(cutoff), "probability_of_loss"])


if __name__ == "__main__":
    unittest.main()
