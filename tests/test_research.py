"""Timing, accounting parity, adjustment contracts, and runner integration."""

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
from research import ExperimentRunner, RunConfig, StrategySpec, default_registry
from research.adjustments import identity, loss_probability
from research.legacy import FEATURES, KEYS, PROJECT_ROOT, TARGET, load_panel
from research.models import evaluate_predictions, predict_losses


def toy_panel(months=96, stocks=8):
    rng = np.random.default_rng(731)
    dates = pd.date_range("2000-12-31", periods=months, freq="ME")
    frame = pd.MultiIndex.from_product([dates, range(stocks)], names=KEYS).to_frame(index=False)
    for feature in FEATURES:
        frame[feature] = rng.integers(1, 101, size=len(frame)) / 100
    frame[TARGET] = rng.normal(0.008, 0.10, len(frame))
    return frame


class AdjustmentTests(unittest.TestCase):
    def setUp(self):
        self.weights = pd.DataFrame({"date": [pd.Timestamp("2020-01-31")] * 3, "stock_id": [1, 2, 3], "weight": [.6, .4, 0]})
        self.predictions = self.weights[KEYS].assign(probability_of_loss=[.8, .2, 1])

    def test_identity_is_exact_and_independent(self):
        copy = identity(self.weights)
        pd.testing.assert_frame_equal(copy, self.weights)
        copy.loc[0, "weight"] = 0
        self.assertEqual(self.weights.weight.iloc[0], .6)

    def test_soft_adjustment_uses_keys_and_preserves_basket(self):
        adjusted = loss_probability(self.weights, self.predictions.iloc[::-1])
        expected = np.array([.6 * .6, .4 * .9, 0])
        np.testing.assert_allclose(adjusted.weight, expected / expected.sum())
        self.assertTrue(adjusted.weight.gt(0).equals(self.weights.weight.gt(0)))
        self.assertAlmostEqual(adjusted.weight.sum(), 1)

    def test_zero_strength_exact(self):
        pd.testing.assert_frame_equal(loss_probability(self.weights, self.predictions, 0), self.weights)

    def test_uniform_probability_does_not_change_weights(self):
        pred = self.predictions.assign(probability_of_loss=.7)
        np.testing.assert_allclose(loss_probability(self.weights, pred).weight, self.weights.weight)

    def test_invalid_probability_keys_and_values_rejected(self):
        invalid = [self.predictions.iloc[:2], pd.concat([self.predictions, self.predictions.iloc[:1]]),
                   self.predictions.assign(probability_of_loss=1.1), self.predictions.assign(probability_of_loss=np.nan)]
        for pred in invalid:
            with self.subTest(pred=pred), self.assertRaises(ValueError):
                loss_probability(self.weights, pred)

    def test_invalid_strength_or_weights_rejected(self):
        for strength in [-.1, 1, np.nan]:
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                loss_probability(self.weights, self.predictions, strength)
        with self.assertRaises(ValueError):
            loss_probability(self.weights.assign(weight=[-.1, 1.1, 0]), self.predictions)


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.history = toy_panel(60, 4)
        self.evaluation = self.history.loc[self.history.date.ge("2003-12-31")].copy()

    def predict(self, history=None, evaluation=None):
        return predict_losses(self.history if history is None else history,
                              self.evaluation if evaluation is None else evaluation,
                              features=FEATURES[:2])

    def test_future_outcomes_cannot_change_first_fit_predictions(self):
        first, _ = self.predict()
        changed = self.history.copy()
        start = self.evaluation.date.min()
        changed.loc[changed.date.ge(start), TARGET] = -0.9
        second, _ = self.predict(history=changed)
        # The label ON the first decision row is a future label too.
        first_fit = first.fit_snapshot.eq(start)
        np.testing.assert_array_equal(first.loc[first_fit, "probability_of_loss"], second.loc[first_fit, "probability_of_loss"])

    def test_future_feature_values_cannot_change_earlier_predictions(self):
        first, _ = self.predict()
        changed = self.history.copy()
        first_decision = self.evaluation.date.min()
        changed.loc[changed.date.gt(first_decision), FEATURES[:2]] = .99
        changed_eval = changed.loc[changed.date.isin(self.evaluation.date.unique())]
        second, _ = self.predict(history=changed, evaluation=changed_eval)
        mask = first.date.eq(first_decision)
        np.testing.assert_array_equal(first.loc[mask, "probability_of_loss"], second.loc[mask, "probability_of_loss"])

    def test_refits_only_use_realized_labels(self):
        predictions, fits = self.predict()
        self.assertTrue(fits.trained_through_return_month.le(fits.fit_snapshot).all())
        self.assertTrue(predictions.trained_through_return_month.le(predictions.date).all())
        self.assertEqual(len(predictions), len(self.evaluation))
        self.assertTrue(predictions.probability_of_loss.between(0, 1).all())
        self.assertEqual(len(fits), 2)
        self.assertTrue(fits.training_rows.is_monotonic_increasing)

    def test_forward_target_never_becomes_a_feature(self):
        with self.assertRaises(ValueError):
            predict_losses(self.history, self.evaluation, features=[TARGET])

    def test_single_class_fallback_is_reported(self):
        predictions, fits = self.predict(history=self.history.assign(R1M_Usd=-.01))
        self.assertTrue(fits.constant_class_fallback.all())
        self.assertTrue(predictions.probability_of_loss.eq(1).all())

    def test_metrics_have_historical_prevalence_control(self):
        predictions, fits = self.predict()
        result = evaluate_predictions(predictions, self.evaluation, fits)
        self.assertEqual(len(result.metrics), 2)
        self.assertEqual(result.calibration.predictions.sum(), len(self.evaluation))
        self.assertTrue(result.metrics.brier_score.between(0, 1).all())


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        panel = toy_panel()
        months = panel.date.drop_duplicates()
        panel.loc[panel.date.isin(months.iloc[:84])].to_csv(cls.root / "mlfactor_long_only_train.csv", index=False)
        panel.loc[panel.date.isin(months.iloc[84:])].to_csv(cls.root / "mlfactor_long_only_test.csv", index=False)
        cls.runner = ExperimentRunner(cls.root)
        cls.config = RunConfig(tuple(default_registry()), include_ml=True)
        cls.result = cls.runner.run(cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_strategies_and_variants_share_calendar(self):
        self.assertEqual(len(self.result.books), 2 * len(self.config.strategies))
        self.assertEqual(len(self.result.paired), len(self.config.strategies))
        calendars = [book.index for book in self.result.books.values()]
        self.assertTrue(all(calendar.equals(calendars[0]) for calendar in calendars))
        self.assertEqual(len(calendars[0]), 24)

    def test_weights_are_fully_invested_and_original_baskets_preserved(self):
        for strategy in self.config.strategies:
            baseline = self.result.weights[f"{strategy}::baseline"].set_index(KEYS).weight
            adjusted = self.result.weights[f"{strategy}::loss_adjusted"].set_index(KEYS).weight
            if self.runner.registry[strategy].allows_cash:
                self.assertTrue(adjusted.groupby("date").sum().le(1 + 1e-12).all())
                self.assertTrue(adjusted.le(baseline + 1e-12).all())
            else:
                np.testing.assert_allclose(adjusted.groupby("date").sum(), 1, atol=1e-12)
            self.assertTrue(baseline.gt(0).equals(adjusted.gt(0)))

    def test_training_validation_does_not_read_test(self):
        def guarded_load(split, root):
            self.assertEqual(split, "train")
            return load_panel(split, root)
        with patch("research.runner.load_panel", side_effect=guarded_load):
            self.runner.run(RunConfig())

    def test_screen_is_fitted_before_validation_and_does_not_see_future_labels(self):
        train = load_panel("train", self.root)
        cutoff = self.result.signals.date.min()
        self.assertEqual(self.result.screen.screen_asof.min(), cutoff)
        self.assertLessEqual(self.result.screen.last_training_return.max(), cutoff)
        changed = train.copy()
        changed.loc[changed.date.ge(cutoff), TARGET] = .99
        with patch("research.runner.load_panel", return_value=changed):
            later = self.runner.run(RunConfig(("training_high_vol_equal_weight",)))
        pd.testing.assert_frame_equal(self.result.screen, later.screen)

    def test_zero_strength_produces_identical_books(self):
        result = self.runner.run(RunConfig(include_ml=True, adjustment_strength=0))
        for key in result.config.strategies:
            pd.testing.assert_frame_equal(result.books[f"{key}::baseline"], result.books[f"{key}::loss_adjusted"])

    def test_allocations_show_actual_baseline_and_adjusted_weights(self):
        date = self.result.signals.date.min()
        allocation = self.result.allocation("momentum_weighted", "loss_adjusted", date)
        self.assertEqual(len(allocation), 8)
        self.assertIn("probability_of_loss", allocation)
        np.testing.assert_allclose(allocation.weight_change, allocation.weight - allocation.baseline_weight)

    def test_export_is_self_describing_and_cannot_overwrite(self):
        with zipfile.ZipFile(io.BytesIO(self.result.to_zip())) as archive:
            self.assertIn("loss_predictions.csv", archive.namelist())
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(set(manifest["source_sha256"]), {"mlfactor_long_only_train.csv"})
        output = self.root / "new_run"
        self.result.save(output)
        with self.assertRaises(FileExistsError):
            self.result.save(output)

    def test_new_function_can_be_registered_without_changing_runner(self):
        def select_one(context):
            weights = context.signals[KEYS].copy()
            weights["weight"] = weights.stock_id.eq(0).astype(float)
            audit = weights[["date"]].drop_duplicates().assign(fallback=False)
            return weights, audit
        registry = {"first_stock": StrategySpec("first_stock", "First stock", "Toy extension", select_one)}
        result = ExperimentRunner(self.root, registry).run(RunConfig(("first_stock",)))
        self.assertTrue(result.weights["first_stock::baseline"].loc[lambda w: w.stock_id.ne(0), "weight"].eq(0).all())

    def test_invalid_config_and_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            RunConfig(strategies=())
        with self.assertRaises(ValueError):
            self.runner.run(RunConfig(("nonexistent",)))


class ExistingNotebookParityTests(unittest.TestCase):
    def test_all_eleven_test_metrics_match_saved_notebook_results(self):
        """Compare independent saved results, rather than calling old comparisons."""
        saved = pd.read_csv(PROJECT_ROOT / "0_1_Basic_Strategies/outputs/downside/strategy_comparison.csv").set_index("strategy").sort_index()
        result = ExperimentRunner().run(RunConfig(tuple(saved.index), evaluation="test"))
        actual = result.metrics.set_index("strategy").sort_index()
        self.assertTrue(actual.index.equals(saved.index))
        for metric in ["cagr", "annualized_volatility", "sharpe", "sortino", "max_drawdown", "avg_one_way_turnover", "largest_single_weight"]:
            np.testing.assert_allclose(actual[metric], saved[metric], atol=1e-12, rtol=1e-12, err_msg=metric)


if __name__ == "__main__":
    unittest.main()
