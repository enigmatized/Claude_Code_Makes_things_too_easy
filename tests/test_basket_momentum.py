"""Correlation baskets stay causal, positive-only, and off the official test file."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.basket_momentum import (
    FEATURE, MOMENTUM_MONTHS, TOP_VOL_VERSION, VERSIONS, assign_correlation_baskets,
    attach_short_momentum, concentrated_basket_weights, create_basket_study, load_basket_study,
    positive_momentum_cash_weights, positive_volatility, save_basket_study, select_top_positive_vol,
)
from research.legacy import FEATURES, KEYS, TARGET, build_signals, load_panel, return_month
from research.strategies import default_registry


def panel(months=18, stocks=4, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2001-01-31", periods=months, freq="ME")
    frame = pd.MultiIndex.from_product([dates, range(stocks)], names=KEYS).to_frame(index=False)
    for feature in FEATURES:
        frame[feature] = 0.5
    earned_index = {date: index for index, date in enumerate(return_month(dates))}
    frame["earned_index"] = return_month(frame.date).map(earned_index)
    wave = np.where(frame.earned_index % 2 == 0, 0.04, -0.01)
    frame[TARGET] = np.where(frame.stock_id < stocks // 2, wave, -wave)
    frame[TARGET] += rng.normal(0, 0.001, len(frame))
    return frame.drop(columns="earned_index")


class WeightTests(unittest.TestCase):
    def test_baskets_drop_negative_groups_and_negative_members(self):
        date = pd.Timestamp("2004-01-31")
        signals = pd.DataFrame({
            "date": [date] * 4,
            "stock_id": [1, 2, 3, 4],
            "eligible": [True, True, True, False],
            "trailing_momentum": [0.20, 0.10, 0.30, 0.50],
            FEATURE: pd.array([0, 0, 1, pd.NA], dtype="Int64"),
            "basket_momentum": [0.10, 0.10, -0.05, np.nan],
        })
        basket = positive_momentum_cash_weights(signals, by_basket=True).set_index("stock_id").weight
        plain = positive_momentum_cash_weights(signals, by_basket=False).set_index("stock_id").weight
        np.testing.assert_allclose(basket.loc[[1, 2, 3, 4]], [0.5 * 0.20 / 0.30, 0.5 * 0.10 / 0.30, 0, 0])
        np.testing.assert_allclose(plain.loc[[1, 2, 3, 4]], [0.75 * 0.20 / 0.60, 0.75 * 0.10 / 0.60, 0.75 * 0.30 / 0.60, 0])
        self.assertAlmostEqual(basket.sum(), 0.5)
        self.assertAlmostEqual(plain.sum(), 0.75)

    def test_one_positive_basket_matches_the_ungated_rule(self):
        date = pd.Timestamp("2004-01-31")
        signals = pd.DataFrame({
            "date": [date] * 3,
            "stock_id": [1, 2, 3],
            "eligible": [True, True, True],
            "trailing_momentum": [0.4, -0.2, 0.1],
            FEATURE: pd.array([0, 0, 0], dtype="Int64"),
            "basket_momentum": [0.05, 0.05, 0.05],
        })
        gated = positive_momentum_cash_weights(signals, by_basket=True)
        plain = positive_momentum_cash_weights(signals, by_basket=False)
        pd.testing.assert_frame_equal(gated, plain)
        self.assertAlmostEqual(gated.weight.sum(), 2 / 3)

    def test_shuffle_and_bad_inputs(self):
        date = pd.Timestamp("2004-01-31")
        signals = pd.DataFrame({
            "date": [date] * 2, "stock_id": [1, 2], "eligible": [True, True],
            "trailing_momentum": [0.2, 0.1], FEATURE: pd.array([0, 0], dtype="Int64"),
            "basket_momentum": [0.1, 0.1],
        })
        shuffled = positive_momentum_cash_weights(signals.sample(frac=1, random_state=1), by_basket=True)
        pd.testing.assert_frame_equal(shuffled, positive_momentum_cash_weights(signals, by_basket=True))
        with self.assertRaises(ValueError):
            positive_momentum_cash_weights(pd.concat([signals, signals.iloc[:1]]), by_basket=True)


class ClusterTests(unittest.TestCase):
    def test_paths_that_move_together_share_a_group(self):
        history = panel()
        decision = history.date.drop_duplicates().sort_values().iloc[14]
        assigned = assign_correlation_baskets(history, [decision], n_clusters=2, lookback=12)
        groups = assigned.set_index("stock_id")[FEATURE]
        self.assertTrue(assigned.eligible.all())
        self.assertEqual(groups.loc[0], groups.loc[1])
        self.assertEqual(groups.loc[2], groups.loc[3])
        self.assertNotEqual(groups.loc[0], groups.loc[2])

    def test_future_returns_do_not_change_the_current_group(self):
        history = panel()
        dates = history.date.drop_duplicates().sort_values()
        decision = dates.iloc[14]
        before = assign_correlation_baskets(history, dates.iloc[12:], n_clusters=2, lookback=12)
        leaked = history.copy()
        leaked.loc[leaked.date.ge(decision), TARGET] = leaked.loc[leaked.date.ge(decision), TARGET] + 1
        after = assign_correlation_baskets(leaked, dates.iloc[12:], n_clusters=2, lookback=12)
        current = before.date.le(decision)
        pd.testing.assert_frame_equal(before.loc[current].reset_index(drop=True), after.loc[current].reset_index(drop=True))
        changed = history.copy()
        previous = dates.iloc[13]
        changed.loc[changed.date.eq(previous) & changed.stock_id.eq(0), TARGET] = 0.8
        moved = assign_correlation_baskets(changed, [decision], n_clusters=2, lookback=12)
        original = before.loc[before.date.eq(decision) & before.stock_id.eq(0), "trailing_momentum"].iloc[0]
        self.assertNotAlmostEqual(moved.loc[moved.stock_id.eq(0), "trailing_momentum"].iloc[0], original)

    def test_feature_ranks_are_not_clustering_inputs(self):
        history = panel()
        decision = [history.date.max()]
        original = assign_correlation_baskets(history, decision, n_clusters=2, lookback=12)
        history[FEATURES[0]] = 0.9
        changed = assign_correlation_baskets(history, decision, n_clusters=2, lookback=12)
        pd.testing.assert_frame_equal(original, changed)

    def test_trailing_return_matches_the_existing_signal(self):
        history = panel()
        dates = history.date.drop_duplicates().sort_values().iloc[12:]
        assigned = assign_correlation_baskets(history, dates, n_clusters=2, lookback=12)
        signals = build_signals(history, lookback=12)
        merged = assigned.merge(signals[KEYS + ["trailing_momentum"]], on=KEYS, suffixes=("", "_signal"))
        np.testing.assert_allclose(merged.trailing_momentum, merged.trailing_momentum_signal)


class ShortMomentumTests(unittest.TestCase):
    def test_one_month_is_the_last_earned_return_and_ignores_older_months(self):
        history = panel(months=20, stocks=4)
        dates = history.date.drop_duplicates().sort_values()
        decision = dates.iloc[16]
        groups = assign_correlation_baskets(history, [decision], n_clusters=2, lookback=12)
        groups = groups[KEYS + [FEATURE]]
        one = attach_short_momentum(history, groups, 1)
        expected = history.loc[history.date.eq(dates.iloc[15]) & history.stock_id.eq(0), TARGET].iloc[0]
        self.assertAlmostEqual(one.loc[one.stock_id.eq(0), "trailing_momentum"].iloc[0], expected)
        three = attach_short_momentum(history, groups, 3)
        long_momentum = assign_correlation_baskets(history, [decision], n_clusters=2, lookback=12)
        self.assertNotAlmostEqual(
            three.loc[three.stock_id.eq(0), "trailing_momentum"].iloc[0],
            long_momentum.loc[long_momentum.stock_id.eq(0), "trailing_momentum"].iloc[0])
        older = history.copy()
        older.loc[older.date.eq(dates.iloc[8]) & older.stock_id.eq(0), TARGET] = 0.8
        moved = attach_short_momentum(older, groups, 1)
        pd.testing.assert_series_equal(
            one.trailing_momentum.reset_index(drop=True),
            moved.trailing_momentum.reset_index(drop=True))
        for months in (1, 2, 3):
            original = attach_short_momentum(history, groups, months)
            changed = attach_short_momentum(older, groups, months)
            pd.testing.assert_series_equal(
                original.trailing_momentum.reset_index(drop=True),
                changed.trailing_momentum.reset_index(drop=True))


class StudyTests(unittest.TestCase):
    def test_training_study_keeps_three_books_and_the_test_file_closed(self):
        self.assertNotIn("correlation_basket_momentum_cash", default_registry())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            history = panel(months=30, stocks=6)
            history.to_csv(root / "mlfactor_long_only_train.csv", index=False)
            test = history.iloc[:0].copy()
            test.to_csv(root / "mlfactor_long_only_test.csv", index=False)
            opened = []

            def guard(split, path):
                opened.append(split)
                return load_panel(split, path)

            with patch("research.basket_momentum.load_panel", side_effect=guard):
                result = create_basket_study(root, n_clusters=2, lookback=12, validation_history_months=24)
            self.assertEqual(opened, ["train"])
            self.assertFalse(result["metadata"]["test_csv_loaded"])
            self.assertLess(pd.Timestamp(result["metadata"]["last_return_month"]), pd.Timestamp("2010-01-31"))
            self.assertEqual(result["metadata"]["momentum_months"], list(MOMENTUM_MONTHS))
            self.assertEqual(result["metadata"]["versions"], list(VERSIONS))
            held = result["assignments"]
            self.assertTrue(held.loc[held.weight.gt(0), "trailing_momentum"].gt(0).all())
            self.assertTrue(held.loc[held.weight.gt(0), "basket_momentum"].gt(0).all())
            self.assertTrue(held.groupby(["date", "momentum_months", "version"]).weight.sum().le(1 + 1e-8).all())
            metrics = result["metrics"]
            self.assertEqual(metrics.loc[metrics.version.eq("Equal weight"), "avg_cash_weight"].iloc[0], 0)
            self.assertTrue(metrics.loc[metrics.version.eq("Basket momentum + cash"), "avg_cash_weight"].gt(0).all())
            self.assertEqual(result["metadata"]["top_positive_vol_months"], [1, 2])
            self.assertEqual(result["metadata"]["top_positive_vol_count"], 10)
            top = metrics.loc[metrics.version.eq(TOP_VOL_VERSION)]
            self.assertEqual(sorted(int(value) for value in top.momentum_months), [1, 2])
            screened = held.loc[held.version.eq(TOP_VOL_VERSION)]
            sizes = screened.groupby(["date", "momentum_months", FEATURE]).size()
            self.assertLessEqual(int(sizes.max()), 10)
            self.assertTrue(screened.loc[screened.weight.gt(0), "trailing_momentum"].gt(0).all())
            directory = save_basket_study(result, root / "basket_studies")
            loaded = load_basket_study(directory)
            self.assertEqual(loaded["metadata"]["model"], "KMeans")
            self.assertEqual(loaded["metadata"]["feature_name"], FEATURE)
            with patch.dict(os.environ, {"MLFACTOR_READ_ONLY": "1"}):
                with self.assertRaises(RuntimeError):
                    create_basket_study(root, n_clusters=2, lookback=12, validation_history_months=24)


def write_earned(frame, earned_month, stock_id, value):
    snapshot = (pd.Timestamp(earned_month).to_period("M") - 1).to_timestamp("M")
    mask = frame.date.eq(snapshot) & frame.stock_id.eq(stock_id)
    if int(mask.sum()) != 1:
        raise AssertionError("Expected one earned-return row.")
    frame.loc[mask, TARGET] = value
    return frame


class PositiveVolTests(unittest.TestCase):
    def test_only_up_months_count_and_future_returns_do_not_change_the_screen(self):
        history = panel(months=20, stocks=3)
        dates = pd.DatetimeIndex(history.date.drop_duplicates().sort_values())
        decision = dates[16]
        window = pd.date_range(end=decision, periods=12, freq="ME")
        history = history.copy()
        for earned in window:
            write_earned(history, earned, 0, 0.20)
            write_earned(history, earned, 1, -0.50)
        for earned in window[:6]:
            write_earned(history, earned, 2, 0.10)
        for earned in window[6:]:
            write_earned(history, earned, 2, -0.40)
        vol = positive_volatility(history, [decision], 12).set_index("stock_id").positive_volatility
        self.assertAlmostEqual(vol.loc[0], 0.20)
        self.assertAlmostEqual(vol.loc[1], 0.0)
        self.assertAlmostEqual(vol.loc[2], np.sqrt(np.mean([0.10 ** 2] * 6 + [0.0] * 6)))
        future = history.copy()
        future.loc[future.date.ge(decision), TARGET] = 0.95
        again = positive_volatility(future, [decision], 12).set_index("stock_id").positive_volatility
        pd.testing.assert_series_equal(vol, again)
        signals = history.loc[history.date.eq(decision), KEYS].copy()
        signals[FEATURE] = pd.array([0, 0, 0], dtype="Int64")
        first = select_top_positive_vol(signals, positive_volatility(history, [decision], 12), count=2)
        second = select_top_positive_vol(signals, positive_volatility(future, [decision], 12), count=2)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first.stock_id), {0, 2})
        nudged = history.copy()
        write_earned(nudged, window[0], 0, 0.80)
        moved = positive_volatility(nudged, [decision], 12).set_index("stock_id").positive_volatility
        self.assertGreater(moved.loc[0], vol.loc[0])

    def test_keeps_ten_highest_and_breaks_ties_toward_the_lower_stock_id(self):
        date = pd.Timestamp("2005-06-30")
        signals = pd.DataFrame({
            "date": [date] * 13,
            "stock_id": list(range(1, 13)) + [99],
            FEATURE: pd.array([0] * 12 + [pd.NA], dtype="Int64"),
        })
        vols = [0.0 if stock == 1 else stock / 100 for stock in range(1, 13)] + [0.9]
        volatility = pd.DataFrame({
            "date": [date] * 13,
            "stock_id": list(signals.stock_id),
            "positive_volatility": vols,
        })
        chosen = select_top_positive_vol(signals, volatility, count=10)
        self.assertEqual(chosen.stock_id.tolist(), list(range(3, 13)))
        self.assertNotIn(99, set(chosen.stock_id))
        tied = pd.DataFrame({
            "date": [date] * 3,
            "stock_id": [30, 10, 20],
            FEATURE: pd.array([0, 0, 0], dtype="Int64"),
        })
        tied_vol = tied[KEYS].assign(positive_volatility=0.2)
        kept = select_top_positive_vol(tied, tied_vol, count=2)
        self.assertEqual(kept.stock_id.tolist(), [10, 20])

    def test_two_selected_baskets_are_scaled_to_the_reduced_book(self):
        history = panel(months=18, stocks=4)
        dates = pd.DatetimeIndex(history.date.drop_duplicates().sort_values())
        decision = dates[16]
        history = history.copy()
        history.loc[history.date.eq(dates[15]), TARGET] = 0.2
        selected = pd.DataFrame({
            "date": [decision, decision],
            "stock_id": [0, 2],
            FEATURE: pd.array([0, 1], dtype="Int64"),
            "positive_volatility": [0.30, 0.20],
        })
        rows = history.loc[history.date.eq(decision), KEYS + [TARGET]].copy()
        full, _, _ = concentrated_basket_weights(history, rows, selected, 1)
        self.assertAlmostEqual(full.weight.sum(), 1.0)
        self.assertEqual(set(full.loc[full.weight.gt(0), "stock_id"]), {0, 2})
        np.testing.assert_allclose(sorted(full.loc[full.weight.gt(0), "weight"]), [0.5, 0.5])

    def test_group_momentum_follows_only_the_names_it_is_given(self):
        history = panel(months=18, stocks=4)
        dates = pd.DatetimeIndex(history.date.drop_duplicates().sort_values())
        decision = dates[16]
        everyone = pd.DataFrame({
            "date": [decision] * 4,
            "stock_id": [0, 1, 2, 3],
            FEATURE: pd.array([0, 0, 0, 0], dtype="Int64"),
        })
        half = everyone.loc[everyone.stock_id.isin([0, 1])].copy()
        wide = attach_short_momentum(history, everyone, 1)
        narrow = attach_short_momentum(history, half, 1)
        self.assertNotAlmostEqual(wide.basket_momentum.iloc[0], narrow.basket_momentum.iloc[0])


if __name__ == "__main__":
    unittest.main()
