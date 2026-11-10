"""Training-only feature fitting, redundancy rules, saved studies and UI review."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import f_classif
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import ExperimentRunner, FeatureSpec, RunConfig, default_model_registry
from research.data import model_data
from research.feature_studies import create_feature_study, feature_audit, list_feature_studies, load_feature_study
from research.models import predict_model
from research.selection import TrainingFeatureSelector
from research.store import ExperimentStore
from test_research import toy_panel


class SelectorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.y = np.tile([0, 1], 100)
        signal = self.y + rng.normal(0, .2, len(self.y))
        self.x = pd.DataFrame({"signal": signal, "duplicate": signal, "noise": rng.normal(size=len(signal)), "constant": 1})

    def test_scores_match_sklearn_and_ties_follow_input_order(self):
        selector = TrainingFeatureSelector("anova", 2).fit(self.x, self.y)
        expected, _ = f_classif(self.x[["signal", "duplicate", "noise"]], self.y)
        np.testing.assert_allclose(selector.scores_[:3], expected)
        self.assertEqual(list(selector.get_feature_names_out()), ["signal", "duplicate"])

    def test_correlated_duplicates_are_skipped_without_backfilling(self):
        selector = TrainingFeatureSelector("correlation_pruned", 3, .9).fit(self.x, self.y)
        self.assertEqual(list(selector.get_feature_names_out()), ["signal", "noise"])
        duplicate = selector.audit_.set_index("feature").loc["duplicate"]
        self.assertEqual(duplicate.correlated_with, "signal")
        self.assertAlmostEqual(duplicate.absolute_correlation, 1)
        self.assertFalse(duplicate.selected)

    def test_selected_pairwise_correlations_respect_declared_threshold(self):
        selector = TrainingFeatureSelector("correlation_pruned", 4, .9).fit(self.x, self.y)
        correlations = selector.transform(self.x).corr().abs().to_numpy()
        np.fill_diagonal(correlations, 0)
        self.assertTrue((correlations < .9).all())

    def test_transform_uses_fitted_mask_and_checks_column_order(self):
        selector = TrainingFeatureSelector("anova", 2).fit(self.x, self.y)
        future = self.x.assign(noise=100)
        self.assertEqual(list(selector.transform(future)), ["signal", "duplicate"])
        with self.assertRaises(ValueError):
            selector.transform(future[future.columns[::-1]])
        self.assertEqual(clone(selector).k, 2)

    def test_recipes_have_stable_distinct_ids_and_cap_learned_inputs(self):
        self.assertEqual(FeatureSpec().feature_id, "factors_8d5fa4272617")
        self.assertNotEqual(FeatureSpec(selection="anova").feature_id, FeatureSpec(selection="correlation_pruned").feature_id)
        self.assertNotEqual(FeatureSpec(selection="anova", k=8).feature_id, FeatureSpec(selection="anova", k=10).feature_id)
        with self.assertRaises(ValueError):
            FeatureSpec(selection="anova", k=11)

    def test_selector_is_cloned_and_fitted_inside_each_model_pipeline(self):
        recipe = FeatureSpec(selection="anova")
        for model in default_model_registry().values():
            pipeline = clone(model.pipeline(recipe))
            self.assertIsInstance(pipeline.named_steps["feature_selection"], TrainingFeatureSelector)


class SelectionTimingTests(unittest.TestCase):
    def test_future_labels_and_features_cannot_change_earlier_masks_or_predictions(self):
        history = toy_panel(48, 6)
        cutoff = history.date.drop_duplicates().iloc[30]
        evaluation = history.loc[history.date.ge(cutoff)]
        changed = history.copy()
        changed.loc[changed.date.ge(cutoff), "R1M_Usd"] = -.9
        changed.loc[changed.date.gt(cutoff), list(FeatureSpec().columns)] = .99
        for method in ["anova", "correlation_pruned"]:
            for model in default_model_registry().values():
                with self.subTest(method=method, model=model.key):
                    recipe = FeatureSpec(selection=method)
                    first, fits1 = predict_model(model_data(history, evaluation), recipe, model)
                    second, fits2 = predict_model(model_data(changed, changed.loc[changed.date.ge(cutoff)]), recipe, model)
                    self.assertEqual(fits1.fitted_features.iloc[0], fits2.fitted_features.iloc[0])
                    self.assertEqual(fits1.selection_audit.iloc[0], fits2.selection_audit.iloc[0])
                    np.testing.assert_array_equal(first.loc[first.date.eq(cutoff), "probability_of_loss"], second.loc[second.date.eq(cutoff), "probability_of_loss"])


class SavedFeatureStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        project = cls.root / "project"
        project.mkdir()
        panel = toy_panel(60, 8)
        dates = panel.date.drop_duplicates()
        panel.loc[panel.date.isin(dates.iloc[:48])].to_csv(project / "mlfactor_long_only_train.csv", index=False)
        panel.loc[panel.date.isin(dates.iloc[48:])].to_csv(project / "mlfactor_long_only_test.csv", index=False)
        cls.store = ExperimentStore(cls.root / "artifacts", project_root=project)
        cls.config = RunConfig(("equal_weight_universe", "momentum_weighted"), validation_history_months=24,
                               include_ml=True, models=tuple(default_model_registry()))
        cls.study = create_feature_study(cls.store, cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_study_contains_three_matched_recipes_and_selected_counts(self):
        loaded = load_feature_study(self.store, self.study["study_id"])
        self.assertEqual(len(loaded["portfolio_comparison"]), 24)
        self.assertTrue(loaded["fits"].loc[loaded["fits"].recipe.ne("all_20"), "n_features"].le(10).all())
        self.assertEqual(len(list_feature_studies(self.store)), 1)
        audit = feature_audit(loaded["fits"])
        self.assertFalse(audit.empty)
        self.assertTrue(audit.trained_through_return_month.le(audit.fit_snapshot).all())

    def test_all_models_use_same_mask_for_same_recipe_and_training_origin(self):
        loaded = load_feature_study(self.store, self.study["study_id"])
        for _, rows in loaded["fits"].groupby(["recipe", "fit_snapshot"]):
            self.assertEqual(rows.fitted_features.nunique(), 1)

    def test_reloading_study_and_full_allocations_never_fits(self):
        with patch("research.models.predict_model", side_effect=AssertionError("review must not fit")):
            loaded = load_feature_study(self.store, self.study["study_id"])
            run = self.store.load(self.study["runs"]["anova_10"])
            allocation = run.allocation("momentum_weighted", "loss_adjusted", run.signals.date.min())
        self.assertAlmostEqual(allocation.weight.sum(), 1)
        self.assertTrue(loaded["metadata"]["checks"]["identical_original_returns"])

    def test_feature_study_rejects_test_period_search(self):
        with self.assertRaises(ValueError):
            create_feature_study(self.store, replace(self.config, evaluation="test"))

    def test_app_reviews_selection_results_masks_and_curves_without_fitting(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        with patch.dict("os.environ", {"MLFACTOR_ARTIFACT_ROOT": str(self.store.root)}), patch(
                "research.store.ExperimentRunner.run", side_effect=AssertionError("saved review must not compute")):
            app = AppTest.from_file(str(app_path), default_timeout=90).run()
            self.assertFalse(app.exception)
            self.assertIn("Feature selection", [tab.label for tab in app.tabs])
            subtabs = [tab for tab in app.tabs if tab.label in {"Selected features & method", "Effect on ML models"}]
            self.assertEqual([tab.label for tab in subtabs], ["Selected features & method", "Effect on ML models"])
            self.assertFalse(any("Brier change vs all 20" in frame.value for frame in subtabs[0].dataframe))
            self.assertFalse(subtabs[0].multiselect)
            selected = next(frame.value for frame in subtabs[0].dataframe if "feature" in frame.value and "selected" not in frame.value)
            loaded = load_feature_study(self.store, self.study["study_id"])
            fit = loaded["fits"].loc[lambda rows: rows.recipe.eq("anova_10") & rows.model_id.eq("logistic")].iloc[0]
            self.assertEqual(selected.feature.tolist(), feature_audit(pd.DataFrame([fit])).loc[lambda rows: rows.selected].feature.tolist())
            self.assertTrue(any("Saved run" in frame.value for frame in subtabs[1].dataframe))
            self.assertTrue(any("Brier change vs all 20" in frame.value for frame in subtabs[1].dataframe))
            self.assertTrue(any("Brier change vs all 20" in frame.value for frame in app.dataframe))
            self.assertTrue(any({"feature", "selected", "reason", "correlated_with"}.issubset(frame.value.columns) for frame in app.dataframe))
            app.selectbox(key="features_study_recipe").select("Correlation-pruned up to 10 (|r| < 0.90)").run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Selected inputs:" in caption.value for caption in app.caption))

    def test_feature_review_and_prediction_scores_do_not_require_a_supported_portfolio(self):
        def panel(strategies):
            from feature_study_ui import render_feature_studies
            render_feature_studies(strategies)

        with patch.dict("os.environ", {"MLFACTOR_ARTIFACT_ROOT": str(self.store.root), "MLFACTOR_READ_ONLY": "1"}), patch(
                "research.store.ExperimentRunner.run", side_effect=AssertionError("saved review must not compute")), patch(
                "research.selection.TrainingFeatureSelector.fit", side_effect=AssertionError("saved selection must not fit")):
            for strategies in [[], ["annual_consistency_momentum_cash"]]:
                app = AppTest.from_function(panel, args=(strategies,), default_timeout=90).run()
                self.assertFalse(app.exception)
                self.assertEqual([tab.label for tab in app.tabs], ["Selected features & method", "Effect on ML models"])
                self.assertTrue(any("feature" in frame.value for frame in app.tabs[0].dataframe))
                self.assertTrue(any("Brier change vs all 20" in frame.value for frame in app.tabs[1].dataframe))
                self.assertFalse(any("cagr" in frame.value for frame in app.tabs[1].dataframe))
                if strategies:
                    self.assertTrue(any("No saved feature comparison yet" in info.value for info in app.tabs[1].info))


if __name__ == "__main__":
    unittest.main()
