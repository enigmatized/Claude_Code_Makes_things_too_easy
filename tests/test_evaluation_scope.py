"""Evaluation periods, historical test evidence, stale settings and no-fit views."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import ExperimentRunner, FeatureSpec, RunConfig
from research.cash_strategy import STRATEGY_KEY
from research.evaluation import ARCHIVE_FILES, evaluation_scope, load_archived_test_results
from research.legacy import FEATURES, PROJECT_ROOT
from research.momentum_studies import create_momentum_study
from research.store import ExperimentStore
from test_research import toy_panel


class EvaluationLabelsTests(unittest.TestCase):
    def test_validation_is_separate_from_fit_rows_and_uses_recorded_dates(self):
        config = RunConfig(include_ml=True)
        scope = evaluation_scope(config, {"first_return_month": "2003-01-31", "last_return_month": "2009-12-31"})
        self.assertEqual(scope["title"], "TRAINING VALIDATION")
        self.assertIn("Jan 2003", scope["period"])
        self.assertIn("not predictions evaluated on their own fit rows", scope["protocol"])
        self.assertIn("Test-period returns are not included", scope["purpose"])

    def test_frozen_and_expanding_test_protocols_are_distinct(self):
        manifest = {"first_return_month": "2010-01-31", "last_return_month": "2018-12-31"}
        config = RunConfig(evaluation="test", include_ml=True)
        expanding = evaluation_scope(config, manifest)
        frozen = evaluation_scope(replace(config, fit_policy="frozen_at_origin"), manifest)
        self.assertEqual(expanding["title"], "TEST RESULTS")
        self.assertIn("Earlier evaluation outcomes", expanding["protocol"])
        self.assertIn("exploratory test", expanding["protocol"])
        self.assertIn("fit once", frozen["protocol"])
        self.assertNotIn("refit every", frozen["protocol"])
        self.assertIn("viewed earlier", frozen["purpose"])


class HistoricalTestArchiveTests(unittest.TestCase):
    def test_only_test_months_are_shown_and_overlapping_strategies_reconcile(self):
        with patch("research.runner.ExperimentRunner.run", side_effect=AssertionError("no new experiment")), \
                patch("research.legacy.load_panel", side_effect=AssertionError("no input CSV needed")):
            archive = load_archived_test_results()
        self.assertEqual(len(archive["metrics"]), 13)
        self.assertEqual(len(archive["monthly"]), 13 * 108)
        self.assertEqual(archive["monthly"].return_month.min(), pd.Timestamp("2010-01-31"))
        self.assertEqual(archive["monthly"].return_month.max(), pd.Timestamp("2018-12-31"))
        self.assertNotIn(STRATEGY_KEY, set(archive["metrics"].strategy))
        self.assertFalse(archive["monthly"].duplicated(["strategy", "return_month"]).any())
        saved = pd.read_csv(PROJECT_ROOT / ARCHIVE_FILES[0], float_precision="round_trip").set_index("strategy")
        actual = archive["metrics"].set_index("strategy")
        self.assertEqual(actual.loc["equal_weight_universe", "cagr"], saved.loc["equal_weight_universe", "cagr"])

    def test_missing_files_and_wrong_period_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                load_archived_test_results(root)
            for name in ARCHIVE_FILES:
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(PROJECT_ROOT / name, root / name)
            path = root / ARCHIVE_FILES[3]
            frame = pd.read_csv(path)
            frame.loc[0, "return_month"] = "2009-12-31"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "2010–2018"):
                load_archived_test_results(root)


class EvaluationAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.project = Path(cls.temp.name)
        panel = toy_panel(84, 8)
        dates = panel.date.drop_duplicates()
        panel.loc[panel.date.isin(dates.iloc[:72])].to_csv(cls.project / "mlfactor_long_only_train.csv", index=False)
        panel.loc[panel.date.isin(dates.iloc[72:])].to_csv(cls.project / "mlfactor_long_only_test.csv", index=False)
        cls.store = ExperimentStore(cls.project / "artifacts", project_root=cls.project)
        config = RunConfig(("equal_weight_universe", STRATEGY_KEY), include_ml=True,
            validation_history_months=48, feature_spec=FeatureSpec(tuple(FEATURES[:2])))
        cls.train = ExperimentRunner(cls.project).run(config)
        cls.store.save(cls.train)
        cls.study = create_momentum_study(cls.store, config)
        cls.test = ExperimentRunner(cls.project).run(replace(config, evaluation="test", fit_policy="frozen_at_origin"))
        cls.test_id = cls.store.save(cls.test)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def app(self, result=None):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=90)
        app.session_state["training_result"] = self.train if result is None else result
        return app.run()

    def test_training_view_is_baseline_only_and_uses_saved_period(self):
        with patch.dict("os.environ", {"MLFACTOR_ARTIFACT_ROOT": str(self.store.root)}), patch(
                "research.store.ExperimentRunner.run", side_effect=AssertionError("browsing cannot fit")):
            app = self.app()
            self.assertFalse(app.exception)
            loaded = next(item.value for item in app.tabs[0].info if item.value.startswith("Strategy evaluation:"))
            self.assertIn("TRAINING VALIDATION", loaded)
            self.assertIn("Non-ML rules", loaded)
            spec = json.loads(app.tabs[0].get("plotly_chart")[0].proto.figure.spec)
            self.assertIn("TRAINING VALIDATION", spec["layout"]["title"]["text"])
            self.assertFalse(any("Version / model" in frame.value for frame in app.tabs[0].dataframe))

    def test_test_view_does_not_change_training_and_is_not_a_final_release(self):
        with patch.dict("os.environ", {"MLFACTOR_ARTIFACT_ROOT": str(self.store.root)}), patch(
                "research.store.ExperimentRunner.run", side_effect=AssertionError("browsing cannot fit")):
            app = self.app()
            self.assertEqual(app.tabs[-1].label, "Final test")
            self.assertFalse(app.tabs[-1].dataframe)
            app.checkbox(key="show_test_history").check().run()
            self.assertTrue(any(len(frame.value) == 13 and "Strategy" in frame.value for frame in app.tabs[-1].dataframe))
            app.button(key="load_test_run").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["test_result"].cache_info["run_id"], self.test_id)
            self.assertEqual(app.session_state["training_result"].config.evaluation, "training_validation")
            self.assertTrue(any("Final frozen-model test evaluation is pending" in item.value for item in app.tabs[-1].info))
            self.assertTrue(any("Frozen models" in item.value for item in app.tabs[-1].info))
            self.assertTrue(any(item.value.startswith("Momentum study") and "TRAINING VALIDATION" in item.value for item in app.tabs[0].info))


if __name__ == "__main__":
    unittest.main()
