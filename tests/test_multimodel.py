"""Shared data timing, configurable predictors/features, and independent parity."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import ExperimentRunner, FeatureSpec, ModelSpec, RunConfig, default_model_registry, default_registry, prepare_data
from research.data import model_data
from research.legacy import FEATURES, KEYS, PROJECT_ROOT, TARGET, load_panel
from research.models import predict_model
from test_research import toy_panel


class SharedDataAndModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        panel = toy_panel(96, 8)
        months = panel.date.drop_duplicates()
        panel.loc[panel.date.isin(months.iloc[:84])].to_csv(cls.root / "mlfactor_long_only_train.csv", index=False)
        panel.loc[panel.date.isin(months.iloc[84:])].to_csv(cls.root / "mlfactor_long_only_test.csv", index=False)
        cls.models = tuple(default_model_registry())
        cls.strategies = ("equal_weight_universe", "momentum_weighted", "training_sharpe_momentum_weighted")
        cls.config = RunConfig(cls.strategies, include_ml=True, models=cls.models)
        with patch("research.runner.predict_model", wraps=predict_model) as predictor, patch("research.runner.prepare_data", wraps=prepare_data) as prepare:
            cls.result = ExperimentRunner(cls.root).run(cls.config)
            cls.predictor_calls = predictor.call_count
            cls.prepare_calls = prepare.call_count
        cls.data = prepare_data(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_preparation_and_prediction_are_shared_not_repeated_per_strategy(self):
        self.assertEqual(self.prepare_calls, 1)
        self.assertEqual(self.predictor_calls, 4)
        self.assertEqual(len(self.result.cases), 15)
        self.assertEqual(self.result.cases.variant.eq("baseline").sum(), 3)
        self.assertEqual(len(self.result.paired), 12)
        self.assertTrue(self.result.cases.case_id.is_unique)

    def test_inputs_exclude_labels_and_known_labels_respect_cutoff(self):
        self.assertNotIn(TARGET, self.data.model.inputs)
        self.assertNotIn(TARGET, self.data.model.evaluation_inputs)
        cutoff = self.data.decision_start
        training = self.data.model.training_slice(cutoff, FEATURES)
        self.assertTrue(training.label_available_at.le(cutoff).all())
        self.assertTrue(training.date.lt(cutoff).all())
        self.assertTrue(self.data.model.labels.label_available_at.equals(self.data.model.labels.earned_month))

    def test_default_preparation_never_reads_test(self):
        reads = []
        def loader(split, root):
            reads.append(split)
            return load_panel(split, root)
        data = prepare_data(self.root, loader=loader)
        self.assertEqual(reads, ["train"])
        self.assertEqual(data.audit["source_splits"], ["train"])

    def test_models_share_keys_fit_dates_training_rows_and_features(self):
        reference = self.result.prediction_batches["logistic"]
        for batch in self.result.prediction_batches.values():
            pd.testing.assert_frame_equal(reference.predictions[KEYS], batch.predictions[KEYS])
            pd.testing.assert_frame_equal(reference.fits[["fit_snapshot", "training_rows", "input_features"]],
                                          batch.fits[["fit_snapshot", "training_rows", "input_features"]])
            self.assertTrue(batch.predictions.probability_of_loss.between(0, 1).all())

    def test_all_cases_use_identical_earned_calendars_and_valid_weights(self):
        calendar = next(iter(self.result.books.values())).index
        for key, book in self.result.books.items():
            self.assertTrue(book.index.equals(calendar))
            weights = self.result.weights[key]
            np.testing.assert_allclose(weights.groupby("date").weight.sum(), 1)
            self.assertTrue(weights.weight.ge(0).all())

    def test_model_changes_do_not_change_original_weights(self):
        single = ExperimentRunner(self.root).run(RunConfig(self.strategies, include_ml=True))
        for strategy in self.strategies:
            pd.testing.assert_frame_equal(single.weights[single.case_id(strategy)], self.result.weights[self.result.case_id(strategy)])
        self.assertFalse(np.array_equal(self.result.prediction_batches["logistic"].predictions.probability_of_loss,
                                       self.result.prediction_batches["naive_bayes"].predictions.probability_of_loss))

    def test_explicit_subset_is_used_by_all_models_without_changing_baseline(self):
        features = FeatureSpec(tuple(FEATURES[:3]))
        subset = ExperimentRunner(self.root).run(RunConfig(self.strategies[:1], include_ml=True, models=self.models, feature_spec=features))
        for batch in subset.prediction_batches.values():
            self.assertTrue(batch.fits.n_features.eq(3).all())
            self.assertTrue(batch.fits.fitted_features.eq(json.dumps(FEATURES[:3])).all())
        pd.testing.assert_frame_equal(subset.weights[subset.case_id(self.strategies[0])], self.result.weights[self.result.case_id(self.strategies[0])])
        self.assertNotEqual(features.feature_id, FeatureSpec().feature_id)
        adjusted = subset.cases.loc[subset.cases.variant.ne("baseline")]
        self.assertTrue(adjusted.feature_spec_id.eq(features.feature_id).all())

    def test_inspector_selects_the_requested_models_predictions(self):
        strategy = self.strategies[0]
        date = self.result.signals.date.min()
        for model_id, batch in self.result.prediction_batches.items():
            allocation = self.result.allocation(strategy, "loss_adjusted", date, model_id).set_index(KEYS)
            expected = batch.predictions.loc[batch.predictions.date.eq(date)].set_index(KEYS).probability_of_loss
            np.testing.assert_array_equal(allocation.loc[expected.index, "probability_of_loss"], expected)

    def test_export_has_case_metadata_data_audit_and_every_models_predictions(self):
        with zipfile.ZipFile(io.BytesIO(self.result.to_zip())) as archive:
            self.assertIn("cases.csv", archive.namelist())
            self.assertIn("data_audit.json", archive.namelist())
            for model in self.models:
                self.assertIn(f"models/{model}/predictions.csv", archive.namelist())
                self.assertIn(f"models/{model}/fits.csv", archive.namelist())
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(len(manifest["models"]), 4)
            self.assertEqual(manifest["feature_spec"]["columns"], FEATURES)
        self.assertEqual(len(self.result.model_metrics), 5)

    def test_registry_explanations_are_complete_and_exported(self):
        for spec in default_registry().values():
            self.assertIsNotNone(spec.explanation)
            for field in ["purpose", "universe", "selection", "weighting", "history", "fallback", "example"]:
                self.assertTrue(getattr(spec.explanation, field))
        self.assertTrue(all(s["explanation"] for s in self.result.manifest["strategies"]))

    def test_feature_and_model_configuration_reject_ambiguous_inputs(self):
        for columns in [(), (TARGET,), ("stock_id",), (FEATURES[0], FEATURES[0])]:
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                FeatureSpec(columns)
        with self.assertRaises(ValueError):
            RunConfig(include_ml=True, models=())
        with self.assertRaises(ValueError):
            ExperimentRunner(self.root).run(RunConfig(include_ml=True, models=("unknown",)))

    def test_custom_estimator_can_be_registered_without_strategy_changes(self):
        original = default_model_registry()["naive_bayes"]
        custom = ModelSpec("custom_nb", "Custom Bayes", "Extension example", original.factory)
        result = ExperimentRunner(self.root, model_registry={"custom_nb": custom}).run(
            RunConfig(self.strategies[:1], include_ml=True, models=("custom_nb",)))
        self.assertEqual(len(result.cases), 2)
        self.assertEqual(set(result.prediction_batches), {"custom_nb"})


class PredictorCausalityTests(unittest.TestCase):
    def test_all_four_models_ignore_future_labels_and_future_feature_values(self):
        history = toy_panel(48, 6)
        dates = history.date.drop_duplicates()
        cutoff = dates.iloc[30]
        evaluation = history.loc[history.date.ge(cutoff)]
        changed = history.copy()
        changed.loc[changed.date.ge(cutoff), TARGET] = -.99
        changed.loc[changed.date.gt(cutoff), FEATURES[:2]] = .99
        changed_evaluation = changed.loc[changed.date.ge(cutoff)]
        for model in default_model_registry().values():
            with self.subTest(model=model.key):
                first, _ = predict_model(model_data(history, evaluation), FeatureSpec(tuple(FEATURES[:2])), model)
                second, _ = predict_model(model_data(changed, changed_evaluation), FeatureSpec(tuple(FEATURES[:2])), model)
                mask = first.date.eq(cutoff)
                np.testing.assert_array_equal(first.loc[mask, "probability_of_loss"], second.loc[mask, "probability_of_loss"])


class SavedLogisticParityTests(unittest.TestCase):
    def test_default_logistic_probabilities_match_the_original_saved_experiment(self):
        saved = pd.read_csv(PROJECT_ROOT / "0_2_Strategy_Runner/outputs/example_training_validation/loss_predictions.csv", parse_dates=["date"])
        result = ExperimentRunner().run(RunConfig(include_ml=True))
        actual = result.predictions.predictions
        pd.testing.assert_frame_equal(actual[KEYS], saved[KEYS])
        np.testing.assert_allclose(actual.probability_of_loss, saved.probability_of_loss, rtol=0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
