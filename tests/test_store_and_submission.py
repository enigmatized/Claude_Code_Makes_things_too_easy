"""Durable cache integrity, CSV migration, frozen fits and report-only reads."""
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import ExperimentRunner, FeatureSpec, RunConfig, default_model_registry
from research.imports import import_csv_report
from research.reporting import load_release, readiness
from research.store import ExperimentStore, digest_file
from test_research import toy_panel


class PersistentStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_directory = tempfile.TemporaryDirectory()
        cls.project = Path(cls.data_directory.name)
        panel = toy_panel(76, 6)
        dates = panel.date.drop_duplicates()
        panel.loc[panel.date.isin(dates.iloc[:60])].to_csv(cls.project / "mlfactor_long_only_train.csv", index=False)
        panel.loc[panel.date.isin(dates.iloc[60:])].to_csv(cls.project / "mlfactor_long_only_test.csv", index=False)
        cls.config = RunConfig(("equal_weight_universe", "momentum_weighted"), validation_history_months=24,
                              include_ml=True, models=("logistic", "naive_bayes"), feature_spec=FeatureSpec(tuple(panel.columns[2:5])))
        cls.result = ExperimentRunner(cls.project).run(cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.data_directory.cleanup()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = ExperimentStore(Path(self.directory.name) / "artifacts", project_root=self.project)

    def tearDown(self):
        self.directory.cleanup()

    def test_complete_roundtrip_preserves_frames_and_inspection(self):
        run_id = self.store.save(self.result)
        with patch("research.models.predict_model", side_effect=AssertionError("load must not fit")):
            loaded = self.store.load(run_id)
        self.assertEqual(loaded.config, self.result.config)
        for name in ["metrics", "paired", "monthly", "diagnostics", "signals", "cases"]:
            pd.testing.assert_frame_equal(getattr(loaded, name), getattr(self.result, name))
        for case in self.result.weights:
            pd.testing.assert_frame_equal(loaded.weights[case], self.result.weights[case])
            pd.testing.assert_frame_equal(loaded.books[case], self.result.books[case])
        for model in self.result.prediction_batches:
            pd.testing.assert_frame_equal(loaded.prediction_batches[model].predictions,
                                          self.result.prediction_batches[model].predictions)
        inspected = loaded.allocation("momentum_weighted", "loss_adjusted", loaded.signals.date.min(), "naive_bayes")
        self.assertIn("probability_of_loss", inspected)
        self.assertAlmostEqual(inspected.weight.sum(), 1)

    def test_exact_cache_hit_never_calls_runner(self):
        run_id = self.store.save(self.result)
        with patch("research.store.ExperimentRunner.run", side_effect=AssertionError("unexpected computation")):
            loaded = self.store.get_or_run(self.config)
        self.assertEqual(loaded.cache_info["run_id"], run_id)
        self.assertTrue(loaded.cache_info["cache_hit"])

    def test_new_configuration_and_force_are_explicit_computations(self):
        self.store.save(self.result)
        with patch("research.store.ExperimentRunner.run", return_value=self.result) as compute:
            self.store.get_or_run(self.config, force=True)
            self.assertEqual(compute.call_count, 1)
        for config in [replace(self.config, adjustment_strength=.3),
                       replace(self.config, fit_policy="frozen_at_origin"),
                       replace(self.config, models=("logistic",)),
                       replace(self.config, feature_spec=FeatureSpec(tuple(self.config.feature_spec.columns[:2])))]:
            self.assertIsNone(self.store.find(config))

    def test_source_content_changes_invalidate_exact_reuse(self):
        self.store.save(self.result)
        path = self.project / "mlfactor_long_only_train.csv"
        content = path.read_bytes()
        try:
            path.write_bytes(content + b"\n")
            self.assertIsNone(self.store.find(self.config))
        finally:
            path.write_bytes(content)

    def test_calculation_code_and_runtime_changes_invalidate_exact_reuse(self):
        self.store.save(self.result)
        def modified(path):
            return "0" * 64 if Path(path).name == "models.py" else digest_file(path)
        with patch("research.store.digest_file", side_effect=modified):
            self.assertIsNone(self.store.find(self.config))
        with patch("sklearn.__version__", "changed-version"):
            self.assertIsNone(self.store.find(self.config))

    def test_corrupt_artifacts_are_rejected_without_retraining(self):
        run_id = self.store.save(self.result)
        folder = self.store.root / run_id
        metadata = json.loads((folder / "artifact.json").read_text())
        (folder / next(iter(metadata["frame_paths"]["weights"].values()))).write_bytes(b"broken")
        with patch("research.store.ExperimentRunner.run", side_effect=AssertionError("must not silently refit")):
            with self.assertRaisesRegex(ValueError, "damaged"):
                self.store.get_or_run(self.config)

    def test_failed_write_does_not_publish_a_partial_run(self):
        with patch.object(pd.DataFrame, "to_parquet", side_effect=RuntimeError("interrupted write")):
            with self.assertRaises(RuntimeError):
                self.store.save(self.result)
        self.assertEqual(self.store.list_runs(), [])
        self.assertFalse(any(self.store.root.glob(".writing_*")))

    def test_catalog_can_be_queried_with_standard_sql(self):
        run_id = self.store.save(self.result)
        with sqlite3.connect(self.store.catalog) as connection:
            count = connection.execute("SELECT count(*) FROM case_metrics WHERE run_id=?", (run_id,)).fetchone()[0]
            labels = connection.execute("SELECT DISTINCT model_id FROM case_metrics WHERE run_id=?", (run_id,)).fetchall()
        self.assertEqual(count, len(self.result.cases))
        self.assertEqual({row[0] for row in labels}, {"none", "logistic", "naive_bayes"})

    def test_report_read_uses_pinned_run_without_deserializing_weights_or_fitting(self):
        run_id = self.store.save(self.result)
        release_path = Path(self.directory.name) / "release.json"
        release_path.write_text(json.dumps({"artifact_root": "artifacts", "run_id": run_id,
            "stage": "research_draft", "focus_strategies": list(self.config.strategies), "final_choice_case_id": None}))
        reads = []
        read_parquet = pd.read_parquet
        def reader(path, *args, **kwargs):
            reads.append(str(path))
            return read_parquet(path, *args, **kwargs)
        with patch("pandas.read_parquet", side_effect=reader), patch("research.models.predict_model", side_effect=AssertionError("report must not fit")):
            summary = load_release(release_path)
        self.assertEqual(summary["run_id"], run_id)
        self.assertFalse(any("/weights/" in path for path in reads))
        self.assertFalse(readiness(summary)["Official test models frozen before evaluation"])
        self.assertFalse(readiness(summary)["GridSearchCV evidence for at least two models"])

    def test_rendering_guard_allows_saved_reads_and_rejects_computation(self):
        run_id = self.store.save(self.result)
        with patch.dict("os.environ", {"MLFACTOR_READ_ONLY": "1"}):
            loaded = self.store.load(run_id)
            self.assertEqual(len(loaded.cases), len(self.result.cases))
            with self.assertRaisesRegex(RuntimeError, "rendering"):
                ExperimentRunner(self.project).run(self.config)

    def test_csv_migration_reconstructs_allocations_without_model_fitting(self):
        report = self.result.save(Path(self.directory.name) / "legacy_report")
        with patch("research.models.predict_model", side_effect=AssertionError("import must not fit")):
            run_id = import_csv_report(self.store, report)
        loaded = self.store.load(run_id)
        expected = self.result.model_metrics.set_index("model")
        actual = loaded.model_metrics.set_index("model")
        np.testing.assert_allclose(actual.loc[expected.index, "brier_score"], expected.brier_score, atol=1e-15)
        for case in self.result.weights:
            np.testing.assert_allclose(loaded.weights[case].weight, self.result.weights[case].weight, atol=1e-15)
        self.assertEqual(loaded.cache_info["origin"], "imported_csv")


class FrozenSubmissionPolicyTests(unittest.TestCase):
    def test_all_models_fit_once_before_test_and_ignore_all_test_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel = toy_panel(76, 6)
            dates = panel.date.drop_duplicates()
            train = panel.loc[panel.date.isin(dates.iloc[:60])]
            test = panel.loc[panel.date.isin(dates.iloc[60:])]
            train.to_csv(root / "mlfactor_long_only_train.csv", index=False)
            test.to_csv(root / "mlfactor_long_only_test.csv", index=False)
            config = RunConfig(("equal_weight_universe",), evaluation="test", include_ml=True,
                models=tuple(default_model_registry()), fit_policy="frozen_at_origin")
            first = ExperimentRunner(root).run(config)
            test.assign(R1M_Usd=-.9).to_csv(root / "mlfactor_long_only_test.csv", index=False)
            second = ExperimentRunner(root).run(config)
            for key, batch in first.prediction_batches.items():
                self.assertEqual(len(batch.fits), 1)
                self.assertEqual(batch.fits.fit_snapshot.iloc[0], test.date.min())
                self.assertTrue(batch.predictions.trained_through_return_month.le(test.date.min()).all())
                np.testing.assert_array_equal(batch.predictions.probability_of_loss,
                                              second.prediction_batches[key].predictions.probability_of_loss)


if __name__ == "__main__":
    unittest.main()
