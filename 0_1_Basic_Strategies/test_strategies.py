"""Causality, calendar continuity, and simple allocation examples."""

import unittest

import numpy as np
import pandas as pd

from strategies import LABELS, build_signals, strategy_weights


def history_example():
    dates = pd.date_range("2008-12-31", periods=15, freq="ME")
    frames = []
    for sid, returns in {
        1: np.arange(15) * 0.01,
        2: np.tile([0.1, -0.1, 0.05], 5),
        3: np.repeat(0.02, 15),
    }.items():
        frames.append(pd.DataFrame({"date": dates, "stock_id": sid, "R1M_Usd": returns}))
    return pd.concat(frames, ignore_index=True).sort_values(["date", "stock_id"]).reset_index(drop=True)


def signal_example():
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-12-31"] * 5), "stock_id": [1, 2, 3, 4, 5],
        "eligible": [True] * 5,
        "trailing_volatility": [1.0, 0.5, 0.2, 0.1, 0.3],
        "trailing_momentum": [0.1, 0.4, -0.1, 0.0, 0.2],
    })


class SignalTimingTests(unittest.TestCase):
    def test_signal_uses_previous_labels_not_own_forward_return(self):
        history = history_example()
        signals = build_signals(history, lookback=3)
        decision = pd.Timestamp("2009-03-31")
        row = signals.loc[signals.date.eq(decision) & signals.stock_id.eq(1)].iloc[0]
        # At March end, realized January–March returns come from Dec–Feb rows.
        self.assertAlmostEqual(row.trailing_momentum, 1.00 * 1.01 * 1.02 - 1)
        self.assertAlmostEqual(row.trailing_volatility, np.std([0, 0.01, 0.02], ddof=1) * np.sqrt(12))
        self.assertTrue(row.eligible)
        self.assertNotIn("R1M_Usd", signals.columns)

    def test_current_and_future_returns_cannot_change_current_signals_or_weights(self):
        history = history_example()
        decision = pd.Timestamp("2009-08-31")
        original = build_signals(history, lookback=3)
        changed = history.copy()
        changed.loc[changed.date.ge(decision), "R1M_Usd"] = 15.0
        revised = build_signals(changed, lookback=3)
        left = original.loc[original.date.le(decision)].reset_index(drop=True)
        right = revised.loc[revised.date.le(decision)].reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)
        for strategy in LABELS:
            a, _ = strategy_weights(left, strategy)
            b, _ = strategy_weights(right, strategy)
            pd.testing.assert_frame_equal(a, b)

    def test_changes_to_realized_history_can_update_next_allocation(self):
        history = history_example()
        decision = pd.Timestamp("2009-03-31")
        original = build_signals(history, lookback=3)
        changed = history.copy()
        changed.loc[changed.date.eq("2009-02-28") & changed.stock_id.eq(1), "R1M_Usd"] = 0.5
        revised = build_signals(changed, lookback=3)
        mask = original.date.eq(decision) & original.stock_id.eq(1)
        self.assertNotEqual(original.loc[mask, "trailing_momentum"].iloc[0], revised.loc[mask, "trailing_momentum"].iloc[0])

    def test_missing_calendar_month_breaks_eligibility(self):
        history = history_example()
        history = history.loc[~(history.stock_id.eq(1) & history.date.eq("2009-01-31"))]
        signals = build_signals(history, lookback=3)
        gap = signals.loc[signals.stock_id.eq(1) & signals.date.eq("2009-03-31")].iloc[0]
        self.assertFalse(gap.eligible)
        self.assertEqual(gap.history_months, 2)
        self.assertTrue(np.isnan(gap.trailing_momentum))

    def test_missing_entire_month_is_not_compressed_away(self):
        history = history_example()
        history = history.loc[~history.date.eq("2009-01-31")]
        signals = build_signals(history, lookback=3)
        rows = signals.loc[signals.date.eq("2009-03-31")]
        self.assertFalse(rows.eligible.any())

    def test_training_history_seeds_first_test_month(self):
        history = history_example()
        first_test = pd.Timestamp("2009-12-31")
        before = history.loc[history.date.lt(first_test)]
        after = history.loc[history.date.ge(first_test)]
        signals = build_signals(pd.concat([before, after]), lookback=12)
        first = signals.loc[signals.date.eq(first_test)]
        self.assertTrue(first.eligible.all())
        self.assertTrue(first.history_months.eq(12).all())
        isolated = build_signals(after, lookback=12)
        self.assertFalse(isolated.loc[isolated.date.eq(first_test), "eligible"].any())

    def test_future_new_stock_does_not_change_earlier_signals(self):
        history = history_example()
        original = build_signals(history, lookback=3)
        extra = pd.DataFrame({"date": [pd.Timestamp("2010-03-31")], "stock_id": [99], "R1M_Usd": [0.5]})
        revised = build_signals(pd.concat([history, extra]), lookback=3)
        revised = revised.loc[revised.date.le(history.date.max())].reset_index(drop=True)
        pd.testing.assert_frame_equal(original, revised)

    def test_total_loss_compounds_to_minus_one(self):
        history = history_example()
        history.loc[history.stock_id.eq(1) & history.date.eq("2009-01-31"), "R1M_Usd"] = -1
        signals = build_signals(history, lookback=3)
        value = signals.loc[signals.stock_id.eq(1) & signals.date.eq("2009-03-31"), "trailing_momentum"].iloc[0]
        self.assertEqual(value, -1)

    def test_duplicate_and_missing_returns_rejected(self):
        history = history_example()
        with self.assertRaises(ValueError):
            build_signals(pd.concat([history, history.iloc[:1]]))
        history.loc[0, "R1M_Usd"] = np.nan
        with self.assertRaises(ValueError):
            build_signals(history)


class AllocationTests(unittest.TestCase):
    def test_high_and_low_baskets_select_opposite_ends(self):
        signals = signal_example()
        high, _ = strategy_weights(signals, "high_volatility")
        low, _ = strategy_weights(signals, "low_volatility")
        self.assertEqual(high.loc[high.weight.gt(0), "stock_id"].tolist(), [1])
        self.assertEqual(low.loc[low.weight.gt(0), "stock_id"].tolist(), [4])

    def test_positive_momentum_selection_and_score_proportions(self):
        signals = signal_example()
        equal, _ = strategy_weights(signals, "momentum_equal_weight", basket_fraction=0.4)
        weighted, audit = strategy_weights(signals, "momentum_weighted", basket_fraction=0.4)
        self.assertEqual(equal.loc[equal.weight.gt(0), "stock_id"].tolist(), [2, 5])
        np.testing.assert_allclose(equal.loc[equal.weight.gt(0), "weight"], [0.5, 0.5])
        np.testing.assert_allclose(weighted.loc[weighted.weight.gt(0), "weight"], [2 / 3, 1 / 3])
        self.assertAlmostEqual(audit.effective_n.iloc[0], 1.8)
        self.assertFalse(audit.fallback.iloc[0])

    def test_fewer_positive_names_are_not_padded_with_negative_momentum(self):
        signals = signal_example()
        signals["trailing_momentum"] = [-0.1, 0.4, -0.1, 0, -0.2]
        weights, _ = strategy_weights(signals, "momentum_weighted", basket_fraction=0.6)
        self.assertEqual(weights.loc[weights.weight.gt(0), "stock_id"].tolist(), [2])
        self.assertEqual(weights.weight.max(), 1)

    def test_no_positive_momentum_falls_back_to_eligible_equal_weight(self):
        signals = signal_example()
        signals["trailing_momentum"] = -0.1
        signals.loc[0, "eligible"] = False
        for strategy in ["momentum_equal_weight", "momentum_weighted"]:
            weights, audit = strategy_weights(signals, strategy)
            np.testing.assert_allclose(weights.weight, [0, 0.25, 0.25, 0.25, 0.25])
            self.assertTrue(audit.fallback.iloc[0])

    def test_no_history_falls_back_to_current_universe(self):
        signals = signal_example()
        signals["eligible"] = False
        for strategy in LABELS:
            weights, audit = strategy_weights(signals, strategy)
            np.testing.assert_allclose(weights.weight, np.repeat(0.2, 5))
            self.assertEqual(bool(audit.fallback.iloc[0]), strategy != "equal_weight_universe")

    def test_ties_use_id_and_are_independent_of_row_order(self):
        signals = signal_example()
        signals["trailing_volatility"] = 0.1
        signals["trailing_momentum"] = 0.1
        for strategy in ["high_volatility", "low_volatility", "momentum_equal_weight", "momentum_weighted"]:
            weights, _ = strategy_weights(signals.sample(frac=1, random_state=2), strategy)
            self.assertEqual(weights.loc[weights.weight.gt(0), "stock_id"].tolist(), [1])

    def test_weights_are_long_only_fully_invested_and_keep_zero_rows(self):
        signals = signal_example()
        signals.loc[0, "eligible"] = False
        for strategy in LABELS:
            weights, _ = strategy_weights(signals, strategy)
            self.assertEqual(len(weights), len(signals))
            self.assertTrue(np.isfinite(weights.weight).all())
            self.assertTrue(weights.weight.ge(0).all())
            self.assertAlmostEqual(weights.weight.sum(), 1)
            if strategy != "equal_weight_universe":
                self.assertEqual(weights.loc[weights.stock_id.eq(1), "weight"].iloc[0], 0)


if __name__ == "__main__":
    unittest.main()
