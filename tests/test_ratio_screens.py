"""Independent ratio arithmetic, historical cutoffs, and pool allocation checks."""

import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import ExperimentRunner, RunConfig
from research.legacy import KEYS, TARGET, load_panel
from research.screens import rank_training_ratio
from test_research import toy_panel


class RatioArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2000-12-31", periods=24, freq="ME")
        paths = {
            1: [.03, -.01, .02, -.005] * 6,
            2: [.04, -.04, .02, -.02] * 6,
            3: [.01, .02, .03, .04] * 6,
            4: [0] * 24,
            5: [-.01] * 24,
            6: [.03, -.01, .02, -.005] * 6,
            7: [.1, -.01] * 3,
        }
        self.panel = pd.concat([
            pd.DataFrame({"date": self.dates[:len(values)], "stock_id": stock, TARGET: values})
            for stock, values in paths.items()
        ], ignore_index=True)
        self.asof = self.dates.max() + pd.offsets.MonthEnd(1)

    def test_ratios_match_direct_arithmetic(self):
        ranked = rank_training_ratio(self.panel, "sharpe", self.asof).set_index("stock_id")
        r = np.array([.03, -.01, .02, -.005] * 6)
        self.assertAlmostEqual(ranked.loc[1, "training_sharpe"], r.mean() / r.std(ddof=1) * np.sqrt(12))
        self.assertAlmostEqual(ranked.loc[1, "training_sortino"], r.mean() / np.sqrt(np.minimum(r, 0).dot(np.minimum(r, 0)) / len(r)) * np.sqrt(12))

    def test_sortino_counts_positive_months_as_zero_shortfalls(self):
        ranked = rank_training_ratio(self.panel, "sortino", self.asof).set_index("stock_id")
        r = np.array([.03, -.01, .02, -.005])
        expected = r.mean() / np.sqrt((.01 ** 2 + .005 ** 2) / 4) * np.sqrt(12)
        self.assertAlmostEqual(ranked.loc[1, "training_sortino"], expected)

    def test_zero_risk_ratios_are_excluded_not_infinite(self):
        sharpe = rank_training_ratio(self.panel, "sharpe", self.asof).set_index("stock_id")
        sortino = rank_training_ratio(self.panel, "sortino", self.asof).set_index("stock_id")
        self.assertFalse(sharpe.loc[4, "eligible_for_screen"])
        self.assertFalse(sortino.loc[3, "eligible_for_screen"])
        self.assertFalse(sortino.loc[3, "selected"])
        self.assertTrue(pd.isna(sortino.loc[3, "training_sortino"]))
        self.assertFalse(np.isinf(sortino.screen_score).any())

    def test_short_history_is_explicitly_excluded(self):
        ranked = rank_training_ratio(self.panel, "sharpe", self.asof).set_index("stock_id")
        self.assertFalse(ranked.loc[7, "selected"])
        self.assertIn("insufficient", ranked.loc[7, "exclusion_reason"])

    def test_fraction_rounding_and_deterministic_ties(self):
        ranked = rank_training_ratio(self.panel, "sharpe", self.asof, fraction=.5)
        self.assertEqual(ranked.selected.sum(), int(np.ceil(.5 * ranked.eligible_for_screen.sum())))
        ranks = ranked.set_index("stock_id").training_rank
        self.assertLess(ranks.loc[1], ranks.loc[6])
        shuffled = rank_training_ratio(self.panel.sample(frac=1, random_state=3), "sharpe", self.asof, fraction=.5)
        pd.testing.assert_frame_equal(ranked, shuffled)

    def test_current_and_future_labels_and_future_stock_do_not_change_pool(self):
        cutoff = self.dates[14]
        first = rank_training_ratio(self.panel, "sortino", cutoff)
        changed = self.panel.copy()
        changed.loc[changed.date.ge(cutoff), TARGET] = 9
        future_stock = changed.loc[changed.stock_id.eq(1) & changed.date.ge(cutoff)].assign(stock_id=999)
        changed = pd.concat([changed, future_stock], ignore_index=True)
        second = rank_training_ratio(changed, "sortino", cutoff)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first.last_training_return.le(cutoff).all())

    def test_negative_finite_ratios_remain_rankable(self):
        panel = self.panel.assign(R1M_Usd=-.03 - self.panel.stock_id * .001 + np.arange(len(self.panel)) % 2 * .001)
        ranking = rank_training_ratio(panel, "sortino", self.asof)
        self.assertTrue(ranking.loc[ranking.selected, "screen_score"].lt(0).all())
        self.assertGreater(ranking.selected.sum(), 0)

    def test_invalid_inputs_and_no_finite_candidates_fail_clearly(self):
        for metric, fraction, minimum in [("unknown", .2, 12), ("sharpe", 0, 12), ("sharpe", .2, 1)]:
            with self.subTest(metric=metric, fraction=fraction, minimum=minimum), self.assertRaises(ValueError):
                rank_training_ratio(self.panel, metric, self.asof, fraction=fraction, min_months=minimum)
        with self.assertRaisesRegex(ValueError, "finite"):
            rank_training_ratio(self.panel.assign(R1M_Usd=0), "sortino", self.asof)


class RatioRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        panel = toy_panel(96, 12)
        dates = panel.date.drop_duplicates()
        panel.loc[panel.date.isin(dates.iloc[:84])].to_csv(cls.root / "mlfactor_long_only_train.csv", index=False)
        panel.loc[panel.date.isin(dates.iloc[84:])].to_csv(cls.root / "mlfactor_long_only_test.csv", index=False)
        cls.runner = ExperimentRunner(cls.root)
        cls.strategies = ("training_sharpe_momentum_weighted", "training_sortino_momentum_weighted")
        cls.result = cls.runner.run(RunConfig(cls.strategies, include_ml=True))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_separate_pools_remain_frozen_and_allocations_stay_inside_them(self):
        self.assertEqual(set(self.result.screens), {"sharpe", "sortino"})
        for strategy in self.strategies:
            metric = self.result.strategy_screen_keys[strategy]
            ranking = self.result.screens[metric]
            pool = set(ranking.loc[ranking.selected, "stock_id"])
            self.assertEqual(ranking.screen_asof.nunique(), 1)
            self.assertEqual(ranking.screen_asof.iloc[0], self.result.signals.date.min())
            for variant in ["baseline", "loss_adjusted"]:
                weights = self.result.weights[f"{strategy}::{variant}"]
                self.assertTrue(weights.loc[~weights.stock_id.isin(pool), "weight"].eq(0).all())
                np.testing.assert_allclose(weights.groupby("date").weight.sum(), 1)

    def test_momentum_weighting_matches_independent_first_month_calculation(self):
        for strategy in self.strategies:
            screen = self.result.screens[self.result.strategy_screen_keys[strategy]]
            pool = set(screen.loc[screen.selected, "stock_id"])
            date = self.result.signals.date.min()
            eligible = self.result.signals.loc[lambda s: s.date.eq(date) & s.stock_id.isin(pool) & s.eligible]
            k = max(1, int(np.ceil(.2 * len(eligible))))
            selected = eligible.loc[eligible.trailing_momentum.gt(0)].sort_values(["trailing_momentum", "stock_id"], ascending=[False, True]).head(k)
            self.assertFalse(selected.empty)
            expected = (selected.trailing_momentum / selected.trailing_momentum.sum()).to_numpy()
            actual = self.result.weights[f"{strategy}::baseline"].set_index(KEYS).loc[[(date, stock) for stock in selected.stock_id], "weight"]
            np.testing.assert_allclose(actual, expected)

    def test_validation_pool_is_not_refitted_using_later_training_returns(self):
        train = load_panel("train", self.root)
        start = self.result.signals.date.min()
        changed = train.copy()
        changed.loc[changed.date.ge(start), TARGET] = .99
        with patch("research.runner.load_panel", return_value=changed):
            result = self.runner.run(RunConfig(self.strategies))
        for metric in self.result.screens:
            pd.testing.assert_frame_equal(self.result.screens[metric], result.screens[metric])
        for strategy in self.strategies:
            first = self.result.weights[f"{strategy}::baseline"].loc[lambda w: w.date.eq(start)]
            second = result.weights[f"{strategy}::baseline"].loc[lambda w: w.date.eq(start)]
            pd.testing.assert_frame_equal(first, second)

    def test_allocation_inspector_and_exports_use_correct_ratio_screen(self):
        for strategy in self.strategies:
            metric = self.result.strategy_screen_keys[strategy]
            allocation = self.result.allocation(strategy, "baseline", self.result.signals.date.min())
            self.assertTrue(allocation.loc[allocation.weight.gt(0), "screen_metric"].eq(metric).all())
            self.assertIn(f"training_{metric}", allocation)
        with zipfile.ZipFile(io.BytesIO(self.result.to_zip())) as archive:
            self.assertIn("frozen_sharpe_screen.csv", archive.namelist())
            self.assertIn("frozen_sortino_screen.csv", archive.namelist())


if __name__ == "__main__":
    unittest.main()
