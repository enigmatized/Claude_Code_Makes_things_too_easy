"""Check downside definitions, the three-month calendar, and causality."""

import unittest

import numpy as np
import pandas as pd

from downside import build_risk_signals, downside_statistics, risk_adjust_weights


def example_history():
    dates = pd.date_range("2009-08-31", periods=6, freq="ME")
    return pd.concat([
        pd.DataFrame({"date": dates, "stock_id": 1, "R1M_Usd": [0.1, 0.1, -0.05, 0.03, 0.04, 0.02]}),
        pd.DataFrame({"date": dates, "stock_id": 2, "R1M_Usd": [-0.1, -0.1, 0.05, 0.03, 0.04, 0.02]}),
    ], ignore_index=True)


class DownsideDefinitionTests(unittest.TestCase):
    def test_exact_three_month_definition(self):
        signals = build_risk_signals(example_history())
        row = signals.loc[signals.date.eq("2009-11-30") & signals.stock_id.eq(1)].iloc[0]
        r = np.array([0.1, 0.1, -0.05])
        ordinary = np.std(r, ddof=1) * np.sqrt(12)
        downside = np.sqrt((0 + 0 + 0.05**2) / 3) * np.sqrt(12)
        self.assertAlmostEqual(row.ordinary_volatility, ordinary)
        self.assertAlmostEqual(row.downside_deviation, downside)
        self.assertAlmostEqual(row.downside_weighted_risk, np.sqrt(ordinary**2 + downside**2))

    def test_mirror_image_has_same_volatility_but_more_downside_risk(self):
        signals = build_risk_signals(example_history())
        rows = signals.loc[signals.date.eq(pd.Timestamp("2009-11-30"))].set_index("stock_id")
        self.assertAlmostEqual(rows.loc[1, "ordinary_volatility"], rows.loc[2, "ordinary_volatility"])
        self.assertGreater(rows.loc[2, "downside_weighted_risk"], rows.loc[1, "downside_weighted_risk"])

    def test_penalty_zero_recovers_ordinary_volatility(self):
        rows = build_risk_signals(example_history(), downside_penalty=0)
        np.testing.assert_allclose(rows.ordinary_volatility, rows.downside_weighted_risk, equal_nan=True)

    def test_constant_losses_are_risky_even_when_standard_deviation_is_zero(self):
        history = example_history()
        history["R1M_Usd"] = -0.1
        signals = build_risk_signals(history)
        rows = signals.loc[signals.date.eq(pd.Timestamp("2009-11-30"))]
        np.testing.assert_allclose(rows.ordinary_volatility, 0)
        np.testing.assert_allclose(rows.downside_weighted_risk, 0.1 * np.sqrt(12))

    def test_sortino_keeps_zero_shortfalls_in_total_count(self):
        returns = np.array([0.1, 0.1, -0.05])
        result = downside_statistics(returns)
        expected = np.sqrt(0.05**2 / 3)
        self.assertAlmostEqual(result["annualized_downside_deviation"], expected * np.sqrt(12))
        self.assertAlmostEqual(result["sortino"], returns.mean() / expected * np.sqrt(12))

    def test_no_losses_has_undefined_sortino_not_infinite_reward(self):
        result = downside_statistics([0.01, 0.02, 0.03])
        self.assertEqual(result["annualized_downside_deviation"], 0)
        self.assertTrue(np.isnan(result["sortino"]))

    def test_target_return_is_monthly_and_explicit(self):
        result = downside_statistics([0.01, 0.02, 0.03], target_monthly=0.02)
        self.assertAlmostEqual(result["annualized_downside_deviation"], np.sqrt(0.01**2 / 3) * np.sqrt(12))
        self.assertAlmostEqual(result["sortino"], 0)


class RiskTimingTests(unittest.TestCase):
    def test_current_forward_return_and_future_returns_do_not_change_current_risk(self):
        history = example_history()
        original = build_risk_signals(history)
        history.loc[history.date.ge("2009-11-30"), "R1M_Usd"] = 20
        revised = build_risk_signals(history)
        pd.testing.assert_frame_equal(
            original.loc[original.date.le("2009-11-30")].reset_index(drop=True),
            revised.loc[revised.date.le("2009-11-30")].reset_index(drop=True),
        )

    def test_missing_month_breaks_the_three_observation_window(self):
        history = example_history()
        history = history.loc[~(history.stock_id.eq(1) & history.date.eq("2009-09-30"))]
        signals = build_risk_signals(history)
        row = signals.loc[signals.date.eq(pd.Timestamp("2009-11-30")) & signals.stock_id.eq(1)].iloc[0]
        self.assertFalse(row.risk_eligible)
        self.assertEqual(row.risk_history_months, 2)
        self.assertTrue(np.isnan(row.downside_weighted_risk))

    def test_duplicates_missing_values_and_invalid_penalty_rejected(self):
        history = example_history()
        with self.assertRaises(ValueError):
            build_risk_signals(pd.concat([history, history.iloc[:1]]))
        with self.assertRaises(ValueError):
            build_risk_signals(history, downside_penalty=-1)
        history.loc[0, "R1M_Usd"] = np.nan
        with self.assertRaises(ValueError):
            build_risk_signals(history)


class RiskSizingTests(unittest.TestCase):
    def setUp(self):
        self.base = pd.DataFrame({"date": pd.to_datetime(["2009-11-30"] * 3),
                                  "stock_id": [1, 2, 3], "weight": [0.5, 0.5, 0.0]})
        self.risk = self.base[["date", "stock_id"]].copy()
        self.risk["ordinary_volatility"] = [0.2, 0.4, np.nan]
        self.risk["downside_weighted_risk"] = [0.2, 0.8, np.nan]

    def test_inverse_risk_weights_and_unchanged_membership(self):
        weights, audit = risk_adjust_weights(self.base, self.risk, "ordinary_volatility")
        np.testing.assert_allclose(weights.weight, [2 / 3, 1 / 3, 0])
        self.assertEqual(audit.n_held.iloc[0], 2)

    def test_extra_downside_reduces_relative_allocation(self):
        ordinary, _ = risk_adjust_weights(self.base, self.risk, "ordinary_volatility")
        downside, _ = risk_adjust_weights(self.base, self.risk, "downside_weighted_risk")
        np.testing.assert_allclose(downside.weight, [0.8, 0.2, 0])
        self.assertLess(downside.weight.iloc[1], ordinary.weight.iloc[1])

    def test_zero_risk_uses_explicit_floor_and_finite_weights(self):
        self.risk.loc[0, "ordinary_volatility"] = 0
        weights, audit = risk_adjust_weights(self.base, self.risk, "ordinary_volatility")
        expected = np.array([0.5 / (0.01 * np.sqrt(12)), 0.5 / 0.4])
        np.testing.assert_allclose(weights.weight.iloc[:2], expected / expected.sum())
        self.assertEqual(audit.risk_floor_hits.iloc[0], 1)
        self.assertTrue(np.isfinite(weights.weight).all())

    def test_missing_risk_of_held_stock_raises(self):
        self.risk.loc[0, "ordinary_volatility"] = np.nan
        with self.assertRaises(ValueError):
            risk_adjust_weights(self.base, self.risk, "ordinary_volatility")

    def test_fallback_keeps_original_allocation_even_without_risk(self):
        self.risk["ordinary_volatility"] = np.nan
        weights, audit = risk_adjust_weights(self.base, self.risk, "ordinary_volatility", fallback_dates=["2009-11-30"])
        pd.testing.assert_frame_equal(weights, self.base)
        self.assertTrue(audit.parent_fallback_preserved.iloc[0])

    def test_key_alignment_is_independent_of_row_order(self):
        expected, _ = risk_adjust_weights(self.base, self.risk, "ordinary_volatility")
        shuffled, _ = risk_adjust_weights(self.base.sample(frac=1, random_state=1), self.risk.sample(frac=1, random_state=2), "ordinary_volatility")
        pd.testing.assert_frame_equal(expected.reset_index(drop=True), shuffled.reset_index(drop=True))


if __name__ == "__main__":
    unittest.main()
