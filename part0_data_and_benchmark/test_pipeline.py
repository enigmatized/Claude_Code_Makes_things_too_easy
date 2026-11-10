"""Independent accounting examples and failure cases; no model-profitability tests."""

import unittest

import numpy as np
import pandas as pd

from pipeline import (
    equal_weights, expanding_month_folds, performance, portfolio_returns,
    return_month,
)


def two_stock_panel():
    # Each month one stock doubles. Rebalancing should grow $1 to $2.25.
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-12-31"] * 2 + ["2021-01-31"] * 2),
        "stock_id": [1, 2, 1, 2], "R1M_Usd": [1.0, 0.0, 0.0, 1.0],
    })


class PortfolioAccountingTests(unittest.TestCase):
    def test_equal_weight_is_capital_average_then_time_compounding(self):
        panel = two_stock_panel()
        book = portfolio_returns(panel)
        np.testing.assert_allclose(book.simple_return, [0.5, 0.5])
        self.assertAlmostEqual(book.wealth.iloc[-1], 2.25)
        # Starting with $0.50 in each and never rebalancing instead ends at $2.
        self.assertNotAlmostEqual(book.wealth.iloc[-1], 0.5 * 2 + 0.5 * 2)

    def test_logs_follow_aggregation_across_assets(self):
        panel = two_stock_panel()
        book = portfolio_returns(panel)
        self.assertAlmostEqual(book.log_return.sum(), np.log(2.25))
        wrong = np.expm1(np.log1p(panel.R1M_Usd.iloc[:2]).mean())
        self.assertNotAlmostEqual(wrong, book.simple_return.iloc[0])

    def test_calendar_is_forward_exactly_once(self):
        actual = return_month(pd.to_datetime(["2009-12-31", "2018-11-30"]))
        self.assertEqual(list(actual), list(pd.to_datetime(["2010-01-31", "2018-12-31"])))

    def test_weights_join_by_keys_not_row_order(self):
        panel = two_stock_panel()
        weights = equal_weights(panel)
        weights["weight"] = [1.0, 0.0, 0.0, 1.0]
        book = portfolio_returns(panel.sample(frac=1, random_state=3), weights.sample(frac=1, random_state=4))
        np.testing.assert_allclose(book.simple_return, [1.0, 1.0])

    def test_turnover_uses_drifted_old_weights(self):
        book = portfolio_returns(two_stock_panel())
        self.assertTrue(np.isnan(book.turnover_one_way.iloc[0]))
        # Old weights drift to 2/3 and 1/3. Returning to halves trades 1/6.
        self.assertAlmostEqual(book.turnover_one_way.iloc[1], 1 / 6)

    def test_changing_universe_does_not_use_survivors_only(self):
        panel = two_stock_panel()
        panel.loc[3, "stock_id"] = 3
        book = portfolio_returns(panel)
        np.testing.assert_allclose(book.simple_return, [0.5, 0.5])
        self.assertAlmostEqual(book.turnover_one_way.iloc[1], 0.5)

    def test_first_losing_month_counts_in_drawdown(self):
        panel = two_stock_panel()
        panel["R1M_Usd"] = [-0.2, -0.2, 0, 0]
        book = portfolio_returns(panel)
        np.testing.assert_allclose(book.drawdown, [-0.2, -0.2])
        self.assertAlmostEqual(performance(book.simple_return).max_drawdown, -0.2)

    def test_duplicate_missing_and_wrong_weight_keys_rejected(self):
        panel = two_stock_panel()
        with self.assertRaises(ValueError):
            portfolio_returns(pd.concat([panel, panel.iloc[:1]], ignore_index=True))
        weights = equal_weights(panel)
        with self.assertRaises(ValueError):
            portfolio_returns(panel, weights.iloc[:-1])
        weights.loc[0, "stock_id"] = 999
        with self.assertRaises(ValueError):
            portfolio_returns(panel, weights)

    def test_missing_labels_and_invalid_weights_rejected(self):
        panel = two_stock_panel()
        missing = panel.copy()
        missing.loc[0, "R1M_Usd"] = np.nan
        with self.assertRaises(ValueError):
            portfolio_returns(missing)
        for values in [[0.4] * 4, [-0.1, 1.1, 0.5, 0.5], [np.inf, 0, 0.5, 0.5]]:
            weights = equal_weights(panel)
            weights["weight"] = values
            with self.assertRaises(ValueError):
                portfolio_returns(panel, weights)

    def test_one_month_total_loss_in_one_stock_is_supported(self):
        panel = two_stock_panel()
        panel.loc[0, "R1M_Usd"] = -1
        self.assertAlmostEqual(portfolio_returns(panel).simple_return.iloc[0], -0.5)


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2020-01-31", periods=12, freq="ME")

    def test_cagr_matches_product_and_log_identity(self):
        r = pd.Series([0.1, -0.1] * 6, index=self.dates)
        result = performance(r)
        self.assertAlmostEqual(result.cagr, (1.1 * 0.9) ** 6 - 1)
        self.assertAlmostEqual(result.cagr, np.expm1(result.annualized_log_growth))
        self.assertAlmostEqual(result.annualized_arithmetic_mean, 0)
        self.assertLess(result.cagr, 0)

    def test_sharpe_uses_arithmetic_excess_and_sample_volatility(self):
        r = pd.Series(np.linspace(-0.02, 0.04, 12), index=self.dates)
        rf = pd.Series(np.linspace(0.001, 0.003, 12), index=self.dates)
        excess = r - rf
        self.assertAlmostEqual(performance(r, rf).sharpe, excess.mean() / excess.std(ddof=1) * np.sqrt(12))
        with self.assertRaises(ValueError):
            performance(r, rf.iloc[::-1])

    def test_missing_duplicate_or_skipped_months_rejected(self):
        r = pd.Series(np.linspace(-0.02, 0.04, 12), index=self.dates)
        for bad in [r.drop(r.index[2]), pd.concat([r, r.iloc[:1]]), r * np.nan]:
            with self.assertRaises(ValueError):
                performance(bad)

    def test_zero_volatility_sharpe_is_undefined(self):
        result = performance(pd.Series(0.0, index=self.dates))
        self.assertTrue(np.isnan(result.sharpe))
        self.assertEqual(result.cagr, 0)

    def test_expanding_folds_keep_months_whole_and_ordered(self):
        panel = two_stock_panel()
        folds = list(expanding_month_folds(panel, min_train_months=1, validation_months=1))
        self.assertEqual(len(folds), 1)
        historical, validation = folds[0]
        self.assertLess(historical.max(), validation.min())
        self.assertEqual(len(panel[panel.date.isin(historical)]), 2)
        self.assertEqual(len(panel[panel.date.isin(validation)]), 2)


if __name__ == "__main__":
    unittest.main()
