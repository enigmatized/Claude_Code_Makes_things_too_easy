"""Four-recipe charts, cross-output readers and checksummed release evidence."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import RunConfig, default_model_registry
from research.feature_studies import FOREST_RECIPES, create_feature_study, load_feature_study
from research.feature_reporting import importance_table, prediction_table, selection_overlap, feature_curves
from research.reporting import feature_study_reference, load_release
from research.store import ExperimentStore
from research.data import model_data
from research.selection import TrainingFeatureSelector
from research.tuning import tune_classifier
from feature_study_ui import study_figure
from test_research import toy_panel


class ForestStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        project = cls.root / "project"
        project.mkdir()
        toy_panel(54, 8).to_csv(project / "mlfactor_long_only_train.csv", index=False)
        cls.store = ExperimentStore(cls.root / "artifacts", project_root=project)
        config = RunConfig(("equal_weight_universe",), validation_history_months=24,
                           include_ml=True, models=tuple(default_model_registry()))
        recipes = {**FOREST_RECIPES, "forest_10": replace(FOREST_RECIPES["forest_10"], forest_trees=12, forest_leaf=2)}
        cls.record = create_feature_study(cls.store, config, recipes=recipes)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_every_recipe_reaches_shared_tables_and_both_chart_renderers(self):
        study = load_feature_study(self.store, self.record["study_id"])
        self.assertEqual(len(prediction_table(study, "logistic")), 4)
        self.assertEqual(len(feature_curves(study, "equal_weight_universe", "logistic")), 5)
        for cumulative_log in (False, True):
            chart = study_figure(study, "equal_weight_universe", "logistic", cumulative_log)
            visible = [t for t in chart.data if t.showlegend is not False]
            self.assertEqual(len(visible), 5)
            self.assertTrue(any(t.name == "Random forest top 10" for t in visible))
            self.assertEqual(len({t.line.color for t in visible}), 5)
        table = importance_table(study)
        self.assertEqual(len(table), 20)
        self.assertEqual(table.selected.sum(), 10)
        overlap = selection_overlap(study)
        self.assertEqual(overlap.shape, (3, 3))
        np.testing.assert_array_equal(np.diag(overlap), [10, 10, 10])

    def test_pinned_study_is_read_only_and_checksum_changes_are_rejected(self):
        release_path = self.root / "release.json"
        reference = feature_study_reference(self.store, self.record["study_id"])
        release = {"artifact_root": "artifacts", "run_id": self.record["runs"]["all_20"],
                   "feature_study": reference}
        release_path.write_text(json.dumps(release))
        with patch("research.selection.TrainingFeatureSelector.fit", side_effect=AssertionError("no fits")), patch(
                "research.store.ExperimentRunner.run", side_effect=AssertionError("no runs")):
            self.assertEqual(load_release(release_path)["feature_study"]["metadata"]["study_id"], self.record["study_id"])
            release["feature_study"] = {**reference, "sha256": "0" * 64}
            release_path.write_text(json.dumps(release))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_release(release_path)
            release.pop("feature_study")
            release_path.write_text(json.dumps(release))
            self.assertNotIn("feature_study", load_release(release_path))

    def test_app_shows_saved_forest_chart_and_scores_without_portfolio_or_fitting(self):
        def panel():
            from feature_study_ui import render_feature_studies
            render_feature_studies([])
        with patch.dict("os.environ", {"MLFACTOR_ARTIFACT_ROOT": str(self.store.root), "MLFACTOR_READ_ONLY": "1"}), patch(
                "research.selection.TrainingFeatureSelector.fit", side_effect=AssertionError("no fits")):
            app = AppTest.from_function(panel, default_timeout=90).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key="features_study_recipe").value, "Random forest top 10")
            self.assertTrue(any("standard deviation across trees" in c.value for c in app.caption))
            self.assertTrue(any("Brier change vs all 20" in d.value for d in app.dataframe))
            chart = next(c for c in app.get("plotly_chart")
                         if json.loads(c.proto.figure.spec)["data"][0].get("type") == "bar")
            spec = json.loads(chart.proto.figure.spec)
            np.testing.assert_allclose(spec["data"][0]["x"], importance_table(load_feature_study(self.store, self.record["study_id"])).score)


class ForestTuningPipelineTests(unittest.TestCase):
    def test_selector_refits_on_each_chronological_fold(self):
        history = toy_panel(48, 8)
        data = model_data(history, history.tail(8))
        origin = data.labels.label_available_at.max()
        frame = data.training_slice(origin, FOREST_RECIPES["forest_10"].columns)
        sizes = []
        original = TrainingFeatureSelector.fit
        def fit(selector, x, y):
            sizes.append(len(x))
            return original(selector, x, y)
        recipe = replace(FOREST_RECIPES["forest_10"], forest_trees=5, forest_leaf=2)
        with patch.object(TrainingFeatureSelector, "fit", fit):
            tuned = tune_classifier(frame, recipe, "logistic", origin, n_splits=2,
                min_train_months=12, grid={"model__C": [1.]})
        self.assertEqual(len(sizes), 3)
        self.assertLess(sizes[0], sizes[1])
        self.assertLess(sizes[1], sizes[2])
        self.assertEqual(sizes[-1], len(frame))


if __name__ == "__main__":
    unittest.main()
