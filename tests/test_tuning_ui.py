"""Saved-only tuning controls, actual event results, references and metric help."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from research.legacy import FEATURES
from research.store import ExperimentStore
from research.tuning_holdout import create_tuning_holdout, save_tuning_holdout
from test_tuning import labeled_panel
from tuning_ui import METRIC_HELP, TARGETS, _parameter_table, _predictive_table, _version_table


class TuningUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name)
        cls.store = ExperimentStore(project_root=cls.root)
        panel = labeled_panel(months=60, stocks=6).drop(columns="label_available_at")
        panel[FEATURES] = panel[FEATURES].clip(.01, 1)
        panel.to_csv(cls.root / "mlfactor_long_only_train.csv", index=False)
        cls.studies = {}
        for target in ("loss", "upside"):
            result = create_tuning_holdout(
                cls.root, tuple(FEATURES[:4]), model_keys=("logistic",),
                strategies=("equal_weight_universe", "momentum_weighted"),
                origin=panel.date.drop_duplicates().iloc[36], n_splits=2,
                min_train_months=12, prediction_target=target)
            # Historical downside studies did not explicitly store their target.
            if target == "loss":
                result["metadata"].pop("prediction_target")
            directory = save_tuning_holdout(result, cls.store.root / "tuning_studies")
            cls.studies[target] = result, directory

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_controls_switch_real_saved_targets_and_strategies_without_fitting(self):
        script = ("from tuning_ui import render_tuning_holdout\n"
                  "from research.store import ExperimentStore\n"
                  f"render_tuning_holdout(ExperimentStore(project_root={str(self.root)!r}))")
        with patch.dict(os.environ, {"MLFACTOR_READ_ONLY": "1"}), \
             patch("research.tuning_holdout.create_tuning_holdout", side_effect=AssertionError("No training")), \
             patch("research.models._fit_slice", side_effect=AssertionError("No fitting")):
            app = AppTest.from_string(script).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual([box.label for box in app.selectbox[:3]],
                             ["1. Model", "2. Basic strategy", "3. Prediction target"])
            app.checkbox(key='tuning_main_families_only').uncheck().run()
            for label, target in TARGETS.items():
                app.selectbox(key="tuning_holdout_target").select(label).run()
                self.assertFalse(app.exception)
                expected = _predictive_table(self.studies[target][0]["predictive"], "logistic")
                score = next(table for table in app.dataframe if "Brier score" in table.value)
                pd.testing.assert_frame_equal(score.value, expected)
                self.assertEqual(len(score.value), 3)
                config = json.loads(score.proto.columns)
                for metric in METRIC_HELP:
                    self.assertEqual(config[metric]["help"], METRIC_HELP[metric])
                app.selectbox(key="tuning_holdout_strategy").select("Positive momentum, score weights").run()
                self.assertFalse(app.exception)
                shown = next(table.value for table in app.dataframe if "CAGR" in table.value)
                expected = _version_table(self.studies[target][0]["metrics"], "momentum_weighted", "logistic")
                pd.testing.assert_frame_equal(shown, expected)
                # Prediction scores describe the model, not a portfolio screen.
                pd.testing.assert_frame_equal(next(t.value for t in app.dataframe if "Brier score" in t.value),
                                              _predictive_table(self.studies[target][0]["predictive"], "logistic"))
            app.selectbox(key="tuning_holdout_model").select("Random forest").run()
            self.assertFalse(app.exception)
            self.assertFalse(app.dataframe)
            self.assertTrue(any("No saved tuning comparison" in item.value for item in app.info))
            app.selectbox(key="tuning_holdout_model").select("Logistic regression").run()
            self.assertFalse(app.exception)
            self.assertTrue(app.dataframe)

    def test_control_collapsing_checks_equality_and_default_setting_explanation(self):
        predictive = self.studies["loss"][0]["predictive"]
        broken = predictive.copy()
        index = broken.index[broken.model.str.startswith("Historical ")][-1]
        broken.loc[index, "log_loss"] += .01
        with self.assertRaises(AssertionError):
            _predictive_table(broken, "logistic")
        settings = _parameter_table("hist_gradient_boosting", {"params": {
            "model__learning_rate": .05, "model__max_leaf_nodes": 15}})
        self.assertTrue(settings.Changed.eq("No").all())
        settings = _parameter_table("logistic", {"params": {"model__C": .1}})
        self.assertEqual(settings.Changed.tolist(), ["Yes"])


if __name__ == "__main__":
    unittest.main()
