"""Training-screen causality and strict hybrid-universe checks."""

import unittest

import numpy as np
import pandas as pd

from hybrid import HYBRID_LABELS, build_signals, evaluate_hybrid, hybrid_weights, rank_training_volatility


def training_example():
    dates = pd.date_range("2009-01-31", periods=12, freq="ME")
    rows = []
    for sid, returns in {
        1: np.tile([-0.3, 0.4], 6),
        2: np.tile([-0.1, 0.2], 6),
        3: np.tile([0.01, 0.02], 6),
    }.items():
        rows.append(pd.DataFrame({"date": dates, "stock_id": sid, "R1M_Usd": returns}))
    return pd.concat(rows, ignore_index=True)


def signals_example():
    return pd.DataFrame({
        "date": pd.to_datetime(["2010-01-31"] * 4), "stock_id": [1, 2, 3, 4],
        "eligible": [True] * 4,
        "trailing_volatility": [0.8, 0.5, 2.0, 3.0],
        "trailing_momentum": [0.1, 0.4, 10.0, 20.0],
    })


class TrainingScreenTests(unittest.TestCase):
    def test_training_volatility_and_selected_stock(self):
        ranking = rank_training_volatility(training_example(), fraction=0.2)
        self.assertEqual(ranking.loc[ranking.selected, "stock_id"].tolist(), [1])
        expected = np.std(np.tile([-0.3, 0.4], 6), ddof=1) * np.sqrt(12)
        self.assertAlmostEqual(ranking.loc[ranking.stock_id.eq(1), "training_volatility"].iloc[0], expected)
        self.assertTrue(ranking.screen_asof.eq(pd.Timestamp("2010-01-31")).all())

    def test_asof_excludes_future_returns_and_future_new_names(self):
        training = training_example()
        ranking = rank_training_volatility(training, asof="2010-01-31")
        future = pd.DataFrame({
            "date": pd.to_datetime(["2010-01-31", "2010-02-28"]),
            "stock_id": [4, 4], "R1M_Usd": [100.0, -0.9],
        })
        revised = rank_training_volatility(pd.concat([training, future]), asof="2010-01-31")
        pd.testing.assert_frame_equal(ranking, revised)

    def test_minimum_training_observations_and_stable_ties(self):
        training = training_example()
        training.loc[training.stock_id.eq(2), "R1M_Usd"] = np.tile([-0.3, 0.4], 6)
        short = pd.DataFrame({"date": [pd.Timestamp("2009-01-31")], "stock_id": [0], "R1M_Usd": [100]})
        ranking = rank_training_volatility(pd.concat([training, short]).sample(frac=1, random_state=2))
        self.assertNotIn(0, ranking.stock_id.tolist())
        self.assertEqual(ranking.loc[ranking.selected, "stock_id"].tolist(), [1])

    def test_final_training_screen_cannot_backtest_earlier_training_dates(self):
        training = training_example()
        ranking = rank_training_volatility(training)
        signals = build_signals(training, lookback=2)
        with self.assertRaises(ValueError):
            evaluate_hybrid(training, signals, ranking)


class HybridAllocationTests(unittest.TestCase):
    def test_outside_screen_winners_get_no_weight(self):
        signals = signals_example()
        for name in HYBRID_LABELS:
            weights, _ = hybrid_weights(signals, [1, 2], name)
            self.assertTrue(weights.loc[weights.stock_id.isin([3, 4]), "weight"].eq(0).all())
            self.assertAlmostEqual(weights.weight.sum(), 1)

    def test_inner_basket_ranks_momentum_only_within_training_screen(self):
        weights, _ = hybrid_weights(signals_example(), [1, 2], "training_high_vol_momentum_weighted")
        # ceil(20% * 2) = 1: stock 2 beats stock 1; outside winners are excluded.
        self.assertEqual(weights.loc[weights.weight.gt(0), "stock_id"].tolist(), [2])

    def test_equal_and_score_weights_use_same_names(self):
        signals = signals_example()
        equal, _ = hybrid_weights(signals, [1, 2], "training_high_vol_momentum_equal", basket_fraction=1)
        weighted, audit = hybrid_weights(signals, [1, 2], "training_high_vol_momentum_weighted", basket_fraction=1)
        np.testing.assert_allclose(equal.weight, [0.5, 0.5, 0, 0])
        np.testing.assert_allclose(weighted.weight, [0.2, 0.8, 0, 0])
        self.assertEqual(audit.n_available_frozen.iloc[0], 2)

    def test_no_positive_fallback_stays_inside_frozen_screen(self):
        signals = signals_example()
        signals.loc[signals.stock_id.isin([1, 2]), "trailing_momentum"] = -0.1
        for name in ["training_high_vol_momentum_equal", "training_high_vol_momentum_weighted"]:
            weights, audit = hybrid_weights(signals, [1, 2], name)
            np.testing.assert_allclose(weights.weight, [0.5, 0.5, 0, 0])
            self.assertTrue(audit.fallback.iloc[0])

    def test_no_eligible_fallback_never_broadens_universe(self):
        signals = signals_example()
        signals.loc[signals.stock_id.isin([1, 2]), "eligible"] = False
        for name in HYBRID_LABELS:
            weights, audit = hybrid_weights(signals, [1, 2], name)
            np.testing.assert_allclose(weights.weight, [0.5, 0.5, 0, 0])
            self.assertTrue(audit.fallback.iloc[0])

    def test_absent_frozen_universe_raises_instead_of_buying_outside_names(self):
        for name in HYBRID_LABELS:
            with self.assertRaises(ValueError):
                hybrid_weights(signals_example(), [99], name)

    def test_future_returns_cannot_change_current_hybrid_weights(self):
        training = training_example()
        test = pd.concat([
            pd.DataFrame({"date": pd.to_datetime(["2010-01-31", "2010-02-28"]),
                          "stock_id": sid, "R1M_Usd": [0.02, 0.04]}) for sid in [1, 2, 3]
        ], ignore_index=True)
        ranking = rank_training_volatility(training, fraction=0.5)
        frozen = ranking.loc[ranking.selected, "stock_id"].tolist()
        original = build_signals(pd.concat([training, test]))
        test["R1M_Usd"] = 30.0
        revised = build_signals(pd.concat([training, test]))
        date = pd.Timestamp("2010-01-31")
        for name in HYBRID_LABELS:
            a, _ = hybrid_weights(original.loc[original.date.eq(date)], frozen, name)
            b, _ = hybrid_weights(revised.loc[revised.date.eq(date)], frozen, name)
            pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":
    unittest.main()
