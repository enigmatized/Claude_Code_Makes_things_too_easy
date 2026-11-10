"""Check chart data against independent small allocation histories."""

import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_charts import (
    allocation_composition, allocation_heatmap, allocation_history_csv,
    allocation_matrices, asset_weight_lines, composition_frames, paired_portfolio_keys,
    return_curve,
)
from research.legacy import performance


class ReturnChartTests(unittest.TestCase):
    def setUp(self):
        # +25%, then -20% returns to the original dollar; cash then stays flat.
        self.dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        self.book = pd.DataFrame({"snapshot_date": self.dates - pd.offsets.MonthEnd(1),
            "simple_return": [.25, -.20, 0.0], "log_return": np.log([1.25, .8, 1]),
            "wealth": [1.25, 1.0, 1.0]}, index=self.dates)

    def test_log_curves_keep_earned_dates_and_only_cumulative_has_initial_zero(self):
        monthly = return_curve(self.book, "Monthly log return")
        pd.testing.assert_series_equal(monthly, self.book.log_return)
        cumulative = return_curve(self.book, "Cumulative log return")
        self.assertEqual(cumulative.index[0], pd.Timestamp("2019-12-31"))
        np.testing.assert_allclose(cumulative, [0, np.log(1.25), 0, 0], atol=1e-15)
        wealth = return_curve(self.book)
        np.testing.assert_allclose(wealth, [1, 1.25, 1, 1])
        np.testing.assert_allclose(np.exp(cumulative), wealth)

    def test_cagr_and_log_growth_use_portfolio_compounding(self):
        returns = pd.Series([.10, -.05, 0.0] * 4,
            index=pd.date_range("2020-01-31", periods=12, freq="ME"))
        metrics = performance(returns)
        expected_annual_growth = (1.10 * .95) ** 4
        self.assertAlmostEqual(metrics.cagr, expected_annual_growth - 1)
        self.assertAlmostEqual(metrics.annualized_log_growth, np.log(expected_annual_growth))
        self.assertAlmostEqual(np.expm1(metrics.annualized_log_growth), metrics.cagr)
        self.assertAlmostEqual(performance(self.book.simple_return).cagr, 0)

    def test_unknown_measure_is_rejected(self):
        with self.assertRaises(ValueError):
            return_curve(self.book, "unsupported")


class AllocationChartTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        base = pd.DataFrame({
            "date": np.repeat(self.dates, 2), "stock_id": [1, 2, 2, 3, 1, 3],
            "weight": [.6, .4, .25, .75, .2, .8],
        })
        adjusted = base.assign(weight=[.4, .6, .5, .5, .1, .9])
        self.result = SimpleNamespace(
            weights={"toy::baseline": base, "toy::loss_adjusted": adjusted},
            books={"toy::baseline": None, "toy::loss_adjusted": None},
        )
        self.matrices = allocation_matrices(self.result, "toy")

    def test_histories_keep_dates_and_fill_exited_assets_with_zero(self):
        base, adjusted = self.matrices.values()
        self.assertTrue(base.index.equals(self.dates.rename("date")))
        self.assertTrue(base.columns.equals(adjusted.columns))
        np.testing.assert_allclose(base.sum(axis=1), 1)
        np.testing.assert_allclose(adjusted.sum(axis=1), 1)
        self.assertEqual(base.loc[self.dates[1], 1], 0)
        self.assertEqual(base.loc[self.dates[0], 3], 0)
        self.assertEqual(adjusted.loc[self.dates[2], 3], .9)

    def test_strategy_selection_always_includes_both_variants(self):
        self.assertEqual(paired_portfolio_keys(self.result, ["toy"]), ["toy::baseline", "toy::loss_adjusted"])
        self.result.books.pop("toy::loss_adjusted")
        self.assertEqual(paired_portfolio_keys(self.result, ["toy"]), ["toy::baseline"])

    def test_heatmaps_share_asset_order_and_color_scale(self):
        figure = allocation_heatmap(self.matrices)
        self.assertEqual(len(figure.data), 2)
        self.assertEqual(list(figure.data[0].y), list(figure.data[1].y))
        self.assertEqual(figure.data[0].coloraxis, figure.data[1].coloraxis)
        self.assertEqual(figure.layout.coloraxis.cmax, .9)
        np.testing.assert_allclose(figure.data[0].z, self.matrices["baseline"].to_numpy().T)

    def test_composition_retains_actual_weights_and_other_assets(self):
        frames = composition_frames(self.matrices, assets=[1])
        for variant, frame in frames.items():
            np.testing.assert_allclose(frame.sum(axis=1), 1)
            np.testing.assert_allclose(frame["Stock 1"], self.matrices[variant][1])
            np.testing.assert_allclose(frame["Other assets (2)"], self.matrices[variant][[2, 3]].sum(axis=1))
        figure = allocation_composition(self.matrices, assets=[1])
        self.assertEqual(len(figure.data), 4)
        self.assertEqual(figure.data[0].line.color, figure.data[2].line.color)

    def test_individual_asset_lines_keep_zero_and_distinguish_variants(self):
        figure = asset_weight_lines(self.matrices, [1])
        self.assertEqual(len(figure.data), 2)
        self.assertEqual(figure.data[0].line.dash, "solid")
        self.assertEqual(figure.data[1].line.dash, "dash")
        np.testing.assert_allclose(figure.data[0].y, [.6, 0, .2])

    def test_history_export_includes_every_month_asset_and_variant(self):
        frame = pd.read_csv(io.StringIO(allocation_history_csv(self.matrices)))
        self.assertEqual(len(frame), 18)
        self.assertEqual(set(frame.variant), {"baseline", "loss_adjusted"})
        np.testing.assert_allclose(frame.groupby(["date", "variant"]).weight.sum(), 1)

    def test_original_only_result_still_supports_timeline(self):
        self.result.weights.pop("toy::loss_adjusted")
        matrices = allocation_matrices(self.result, "toy")
        self.assertEqual(list(matrices), ["baseline"])
        self.assertEqual(len(allocation_heatmap(matrices).data), 1)


if __name__ == "__main__":
    unittest.main()
