"""Chronological GridSearchCV: month integrity, small grids, save and reload."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.features import FeatureSpec
from research.legacy import FEATURES, KEYS, TARGET, return_month
from research.models import default_model_registry
from research.runner import ExperimentRunner
from research.tuning import (
    DECLARED_GRIDS, MonthForwardSplit, candidate_count, declared_grid,
    load_tuned_model, predict_event_probability, save_tuned_model, tune_classifier,
)
from research.tuning_holdout import create_tuning_holdout, load_tuning_holdout, save_tuning_holdout


def labeled_panel(months=36, stocks=6, seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-31", periods=months, freq="ME")
    frame = pd.MultiIndex.from_product([dates, range(stocks)], names=KEYS).to_frame(index=False)
    loss = rng.integers(0, 2, size=len(frame))
    frame[TARGET] = np.where(loss == 1, -0.03, 0.02)
    frame[FEATURES[0]] = np.where(loss == 1, 0.85, 0.15) + rng.normal(0, 0.01, len(frame))
    for feature in FEATURES[1:]:
        frame[feature] = rng.uniform(0.01, 1, size=len(frame))
    frame["label_available_at"] = return_month(frame.date)
    return frame


class SplitTests(unittest.TestCase):
    def test_months_stay_together_and_training_labels_are_known_at_the_origin(self):
        frame = labeled_panel()
        splitter = MonthForwardSplit(frame.date, frame.label_available_at, n_splits=3, min_train_months=12)
        previous = -1
        for train, valid in splitter.split(frame):
            train_dates = set(frame.date.iloc[train])
            valid_dates = set(frame.date.iloc[valid])
            self.assertTrue(train_dates.isdisjoint(valid_dates))
            origin = frame.date.iloc[valid].min()
            self.assertTrue((frame.date.iloc[train] < origin).all())
            self.assertTrue((frame.label_available_at.iloc[train] <= origin).all())
            self.assertGreater(len(train), previous)
            previous = len(train)
            for date in frame.date.unique():
                rows = frame.index[frame.date.eq(date)]
                in_train = set(rows).issubset(train)
                in_valid = set(rows).issubset(valid)
                self.assertFalse(in_train and in_valid)
                self.assertTrue(in_train or in_valid or (date >= origin and date not in valid_dates))


class GridTests(unittest.TestCase):
    def test_declared_grids_include_the_fixed_defaults_and_stay_small(self):
        registry = default_model_registry()
        self.assertIn(registry["logistic"].factory().get_params()["C"], DECLARED_GRIDS["logistic"]["model__C"])
        self.assertIn(registry["random_forest"].factory().get_params()["max_depth"],
                      DECLARED_GRIDS["random_forest"]["model__max_depth"])
        self.assertIn(registry["hist_gradient_boosting"].factory().get_params()["learning_rate"],
                      DECLARED_GRIDS["hist_gradient_boosting"]["model__learning_rate"])
        self.assertLessEqual(candidate_count(declared_grid("logistic")) * 3 + 1, 40)
        self.assertLessEqual(candidate_count(declared_grid("hist_gradient_boosting")) * 3 + 1, 40)
        self.assertLessEqual(candidate_count(declared_grid("random_forest")) * 3 + 1, 40)

    def test_too_many_fits_and_a_large_forest_are_refused_before_fitting(self):
        frame = labeled_panel()
        asof = frame.label_available_at.max()
        recipe = FeatureSpec(tuple(FEATURES[:3]))
        with self.assertRaises(ValueError):
            tune_classifier(frame, recipe, "logistic", asof, min_train_months=12,
                            grid={"model__C": [0.1] * 30})
        with self.assertRaises(ValueError):
            tune_classifier(frame, recipe, "random_forest", asof, min_train_months=12, n_splits=2)


class SearchTests(unittest.TestCase):
    def test_search_refits_on_the_slice_and_reloads_the_same_probabilities(self):
        frame = labeled_panel()
        asof = frame.label_available_at.max()
        recipe = FeatureSpec(tuple(FEATURES[:4]), selection="anova", k=2)
        result = tune_classifier(frame, recipe, "logistic", asof, min_train_months=12, n_splits=2)
        self.assertEqual(result.metadata["engine"], "GridSearchCV")
        self.assertLessEqual(result.metadata["n_fitted_features"], 2)
        self.assertEqual(int(result.trials.chosen.sum()), 1)
        self.assertEqual(result.metadata["n_fits"], 3 * 2 + 1)
        self.assertFalse(result.metadata["portfolio_scored"])
        winner = json.loads(result.trials.loc[result.trials.chosen].params.iloc[0])
        self.assertEqual(winner, result.metadata["chosen_params"])
        self.assertIn(result.metadata["chosen_params"]["model__C"], DECLARED_GRIDS["logistic"]["model__C"])
        fresh = predict_event_probability(result, frame)
        with tempfile.TemporaryDirectory() as directory:
            saved = save_tuned_model(result, Path(directory) / "logistic")
            loaded = load_tuned_model(saved)
            reloaded = predict_event_probability(loaded, frame)
        np.testing.assert_allclose(fresh, reloaded)
        self.assertTrue(np.isfinite(reloaded).all())
        self.assertTrue(((reloaded >= 0) & (reloaded <= 1)).all())

    def test_a_label_from_after_the_origin_is_rejected(self):
        frame = labeled_panel()
        asof = frame.label_available_at.max()
        frame.loc[0, "label_available_at"] = asof + pd.offsets.MonthEnd(1)
        with self.assertRaises(ValueError):
            tune_classifier(frame, FeatureSpec(tuple(FEATURES[:3])), "logistic", asof, min_train_months=12)

    def test_read_only_rendering_cannot_search(self):
        frame = labeled_panel()
        previous = os.environ.get("MLFACTOR_READ_ONLY")
        os.environ["MLFACTOR_READ_ONLY"] = "1"
        try:
            with self.assertRaises(RuntimeError):
                tune_classifier(frame, FeatureSpec(tuple(FEATURES[:3])), "logistic",
                                frame.label_available_at.max(), min_train_months=12)
        finally:
            if previous is None:
                os.environ.pop("MLFACTOR_READ_ONLY")
            else:
                os.environ["MLFACTOR_READ_ONLY"] = previous


class HoldoutTests(unittest.TestCase):
    def test_upside_holdout_fits_up_labels_and_applies_the_fixed_boost(self):
        frame = labeled_panel(months=60, stocks=6).drop(columns=["label_available_at"])
        frame[FEATURES] = frame[FEATURES].clip(0.01, 1)
        # Zero returns must be neither a loss nor an up event.
        frame.loc[frame.index[::17], TARGET] = 0
        origin = frame.date.drop_duplicates().iloc[36]
        runs = []
        original_run = ExperimentRunner.run
        def capture(runner, config, **kwargs):
            result = original_run(runner, config, **kwargs)
            runs.append(result)
            return result
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame.to_csv(root / "mlfactor_long_only_train.csv", index=False)
            with patch.object(ExperimentRunner, "run", autospec=True, side_effect=capture):
                result = create_tuning_holdout(
                    root, tuple(FEATURES[:4]), model_keys=("logistic",),
                    strategies=("equal_weight_universe",), origin=origin,
                    n_splits=2, min_train_months=12, prediction_target="upside")
            self.assertEqual(result["metadata"]["prediction_target"], "upside")
            self.assertEqual(result["searches"]["logistic"].metadata["target"], "upside")
            self.assertEqual(set(result["metadata"]["source_sha256"]), {"mlfactor_long_only_train.csv"})
            for run in runs:
                self.assertEqual(run.config.prediction_target, "upside")
                predictions = run.prediction_batches["logistic"].predictions
                scored = predictions.merge(frame[KEYS + [TARGET]], on=KEYS, validate="one_to_one")
                reported = run.prediction_batches["logistic"].metrics.iloc[0]
                self.assertAlmostEqual(reported.observed_up_rate, scored[TARGET].gt(0).mean())
                self.assertAlmostEqual(reported.brier_score,
                    ((scored.probability_of_up - scored[TARGET].gt(0).astype(int)) ** 2).mean())
                self.assertGreater(scored.loc[scored[TARGET].gt(0), "probability_of_up"].mean(),
                                   scored.loc[scored[TARGET].lt(0), "probability_of_up"].mean())
                original = run.weights[run.case_id("equal_weight_universe")]
                actual = run.weights[run.cases.loc[run.cases.variant.ne("baseline"), "case_id"].iloc[0]]
                expected = original.merge(predictions, on=KEYS, validate="one_to_one")
                expected["weight"] *= 1 + .5 * expected.probability_of_up.ge(.6)
                expected["weight"] /= expected.groupby("date").weight.transform("sum")
                pd.testing.assert_frame_equal(actual, expected[KEYS + ["weight"]].sort_values(KEYS).reset_index(drop=True))
            loaded = load_tuning_holdout(save_tuning_holdout(result, root / "tuning_studies"))
            self.assertEqual(loaded["metadata"]["prediction_target"], "upside")
            pd.testing.assert_frame_equal(loaded["predictive"], result["predictive"])

    def test_three_books_are_scored_only_after_the_split(self):
        frame = labeled_panel(months=60, stocks=6).drop(columns=["label_available_at"])
        frame[FEATURES] = frame[FEATURES].clip(0.01, 1)
        origin = frame.date.drop_duplicates().iloc[36]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame.to_csv(root / "mlfactor_long_only_train.csv", index=False)
            result = create_tuning_holdout(
                root, tuple(FEATURES[:4]), model_keys=("logistic",), strategies=("equal_weight_universe",),
                origin=origin, n_splits=2, min_train_months=12)
            self.assertFalse((root / "mlfactor_long_only_test.csv").exists())
            self.assertFalse(result["metadata"]["test_csv_loaded"])
            self.assertEqual(result["metadata"]["prediction_target"], "loss")
            self.assertLessEqual(pd.Timestamp(result["metadata"]["trained_through_return_month"]), origin)
            self.assertEqual(pd.Timestamp(result["metadata"]["holdout_first_return_month"]),
                             pd.Timestamp(return_month([origin])[0]))
            self.assertLess(pd.Timestamp(result["metadata"]["holdout_last_return_month"]), pd.Timestamp("2010-01-31"))
            versions = set(result["metrics"].version)
            self.assertEqual(versions, {"No ML", "ML, not tuned", "Tuned ML"})
            months = result["monthly"].groupby("version").return_month.nunique()
            self.assertTrue(months.eq(months.iloc[0]).all())
            saved = save_tuning_holdout(result, root / "tuning_studies")
            loaded = load_tuning_holdout(saved)
            self.assertEqual(len(loaded["metrics"]), len(result["metrics"]))

    def test_read_only_rendering_cannot_build_the_holdout(self):
        frame = labeled_panel(months=48, stocks=4).drop(columns=["label_available_at"])
        frame[FEATURES] = frame[FEATURES].clip(0.01, 1)
        origin = frame.date.drop_duplicates().iloc[30]
        previous = os.environ.get("MLFACTOR_READ_ONLY")
        os.environ["MLFACTOR_READ_ONLY"] = "1"
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                frame.to_csv(root / "mlfactor_long_only_train.csv", index=False)
                with self.assertRaises(RuntimeError):
                    create_tuning_holdout(root, tuple(FEATURES[:3]), model_keys=("logistic",),
                                          strategies=("equal_weight_universe",), origin=origin, n_splits=2, min_train_months=12)
        finally:
            if previous is None:
                os.environ.pop("MLFACTOR_READ_ONLY")
            else:
                os.environ["MLFACTOR_READ_ONLY"] = previous


if __name__ == "__main__":
    unittest.main()
