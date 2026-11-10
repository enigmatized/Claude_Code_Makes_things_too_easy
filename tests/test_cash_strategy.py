"""Independent annual-screen, cash-budget, accounting and artifact checks."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_charts import allocation_matrices, allocation_history_csv, composition_frames
from research import ExperimentRunner, RunConfig
from research.accounting import portfolio_with_cash
from research.adjustments import loss_probability
from research.cash_strategy import STRATEGY_KEY, annual_consistency_screen, consistency_momentum_weights
from research.imports import import_csv_report
from research.legacy import KEYS, TARGET, portfolio_returns
from research.store import ExperimentStore
from test_research import toy_panel


class AnnualScreenTests(unittest.TestCase):
    def setUp(self):
        earned = pd.date_range("2000-01-31", "2003-12-31", freq="ME")
        paths = {1: [.02, -.01, .03, .01], 2: [-.01, .02, -.01, -.01],
                 3: [.01, .01, 0, 0], 4: [.01, .01, .01, .01]}
        self.history = pd.concat([pd.DataFrame({
            "date": earned - pd.offsets.MonthEnd(1), "stock_id": stock,
            TARGET: np.repeat(annual_monthly, 12)}) for stock, annual_monthly in paths.items()], ignore_index=True)
        # Stock 4 has only two complete years despite positive partial years.
        self.history = self.history.loc[~(self.history.stock_id.eq(4) & self.history.date.isin(
            [pd.Timestamp("2000-05-31"), pd.Timestamp("2001-05-31")]))]

    def test_compounded_years_strict_majority_and_incomplete_years(self):
        result = annual_consistency_screen(self.history, "2003-12-31").set_index("stock_id")
        self.assertTrue(result.loc[1, "selected"])
        self.assertEqual(result.loc[1, "positive_years"], 3)
        self.assertEqual(result.loc[1, "negative_years"], 1)
        self.assertFalse(result.loc[2, "selected"])
        self.assertFalse(result.loc[3, "selected"])  # two positive, two flat
        self.assertEqual(result.loc[3, "flat_years"], 2)
        self.assertEqual(result.loc[4, "complete_years"], 2)
        self.assertFalse(result.loc[4, "selected"])

    def test_year_sign_uses_compounding_rather_than_mean_monthly_return(self):
        panel = self.history.loc[self.history.stock_id.eq(1)].copy()
        panel[TARGET] = np.tile([1, -.6] + [0] * 10, 4)  # positive mean, negative annual product
        result = annual_consistency_screen(panel, "2003-12-31").iloc[0]
        self.assertEqual(result.negative_years, 4)
        self.assertFalse(result.selected)

    def test_future_labels_future_stock_and_unfinished_year_cannot_change_screen(self):
        cutoff = pd.Timestamp("2002-06-30")
        original = annual_consistency_screen(self.history, cutoff)
        changed = self.history.copy()
        changed.loc[changed.date.ge(cutoff), TARGET] = 10
        added = changed.loc[changed.date.ge(cutoff) & changed.stock_id.eq(1)].assign(stock_id=999)
        later = annual_consistency_screen(pd.concat([changed, added]), cutoff)
        pd.testing.assert_frame_equal(original, later)
        self.assertTrue(original.complete_years.le(2).all())
        self.assertFalse(original.selected.any())


class CashWeightsTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        self.signals = pd.MultiIndex.from_product([dates, [1, 2, 3, 4, 5]], names=KEYS).to_frame(index=False)
        self.signals["eligible"] = True
        self.signals["trailing_momentum"] = [.1, .3, -.2, 0, 99, -.1, -.2, 0, -.3, 99, .1, np.nan, -.1, 0, 99]
        self.signals.loc[11, "eligible"] = False

    def test_stock_budget_and_no_negative_zero_missing_or_outside_pool_purchase(self):
        weights, audit = consistency_momentum_weights(self.signals, (1, 2, 3, 4))
        np.testing.assert_allclose(weights.weight.iloc[:5], [.125, .375, 0, 0, 0])
        np.testing.assert_allclose(weights.groupby("date").weight.sum(), [.5, 0, .25])
        self.assertEqual(audit.positive_momentum_stocks.tolist(), [2, 0, 1])
        self.assertFalse(audit.fallback.any())
        shuffled, _ = consistency_momentum_weights(self.signals.sample(frac=1, random_state=1), (1, 2, 3, 4))
        pd.testing.assert_frame_equal(weights, shuffled)

    def test_empty_pool_and_future_signals_leave_past_weights_unchanged(self):
        empty, _ = consistency_momentum_weights(self.signals, ())
        self.assertTrue(empty.weight.eq(0).all())
        first, _ = consistency_momentum_weights(self.signals, (1, 2))
        changed = self.signals.copy()
        changed.loc[changed.date.gt(changed.date.min()), "trailing_momentum"] = 100
        second, _ = consistency_momentum_weights(changed, (1, 2))
        pd.testing.assert_frame_equal(first.iloc[:5], second.iloc[:5])

    def test_ml_haircut_moves_to_cash_and_handles_all_cash_without_nan(self):
        weights, _ = consistency_momentum_weights(self.signals, (1, 2, 3, 4))
        pred = weights[KEYS].assign(probability_of_loss=.8)
        adjusted = loss_probability(weights, pred.iloc[::-1], .5, allow_cash=True)
        np.testing.assert_allclose(adjusted.weight, weights.weight * .6)
        np.testing.assert_allclose(adjusted.groupby("date").weight.sum(), [.3, 0, .15])
        self.assertTrue(np.isfinite(adjusted.weight).all())
        pd.testing.assert_frame_equal(loss_probability(weights, pred, 0, allow_cash=True), weights)
        for bad in [pred.iloc[:-1], pred.assign(probability_of_loss=np.nan)]:
            with self.assertRaises(ValueError):
                loss_probability(weights, bad, allow_cash=True)


class CashAccountingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"date": pd.date_range("2020-01-31", periods=3, freq="ME"),
            "stock_id": [77] * 3, TARGET: [.1, -.5, .2]})

    def test_returns_cash_drift_turnover_and_drawdown_match_hand_calculation(self):
        weights = self.frame[KEYS].assign(weight=[.5, 0, 1])
        book = portfolio_with_cash(self.frame, weights)
        np.testing.assert_allclose(book.simple_return, [.05, 0, .2])
        np.testing.assert_allclose(book.wealth, [1.05, 1.05, 1.26])
        np.testing.assert_allclose(book.cash_weight, [.5, 1, 0])
        np.testing.assert_allclose(book.turnover_one_way.iloc[1:], [.55 / 1.05, 1])
        self.assertTrue(np.isnan(book.turnover_one_way.iloc[0]))
        self.assertEqual(book.n_held.tolist(), [1, 0, 1])
        np.testing.assert_allclose(book.drawdown, 0)

    def test_cash_only_returns_zero_and_no_turnover_and_initial_loss_drawdown(self):
        book = portfolio_with_cash(self.frame, self.frame[KEYS].assign(weight=0.0))
        np.testing.assert_allclose(book.simple_return, 0)
        np.testing.assert_allclose(book.turnover_one_way.iloc[1:], 0)
        losing = self.frame.assign(R1M_Usd=-.2)
        book = portfolio_with_cash(losing, losing[KEYS].assign(weight=.5))
        self.assertAlmostEqual(book.drawdown.iloc[0], -.1)

    def test_invalid_weights_rejected_and_full_stock_book_matches_legacy(self):
        weights = self.frame[KEYS].assign(weight=1.0)
        actual = portfolio_with_cash(self.frame, weights).drop(columns=["cash_weight", "equity_weight"])
        pd.testing.assert_frame_equal(actual, portfolio_returns(self.frame, weights))
        for bad in [weights.assign(weight=1.1), weights.assign(weight=-.1), weights.iloc[:-1],
                    weights.assign(weight=np.nan), pd.concat([weights, weights.iloc[:1]])]:
            with self.assertRaises(ValueError):
                portfolio_with_cash(self.frame, bad)


class CashRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        panel = toy_panel(84, 8)
        # Last two evaluation years all decline, ensuring all-cash months.
        panel.loc[panel.date.ge("2005-12-31"), TARGET] = -.03
        panel.to_csv(cls.root / "mlfactor_long_only_train.csv", index=False)
        cls.result = ExperimentRunner(cls.root).run(RunConfig(
            ("equal_weight_universe", STRATEGY_KEY), include_ml=True))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_pool_holdings_cash_diagnostics_and_all_cash_months(self):
        result = self.result
        pool = set(result.screens["annual_consistency"].loc[lambda f: f.selected, "stock_id"])
        self.assertTrue(pool)
        for variant in ["baseline", "loss_adjusted"]:
            key = result.case_id(STRATEGY_KEY, variant)
            held = result.weights[key].merge(result.signals, on=KEYS)
            held = held.loc[held.weight.gt(0)]
            self.assertTrue(held.stock_id.isin(pool).all())
            self.assertTrue(held.trailing_momentum.gt(0).all())
            book = result.books[key]
            self.assertTrue(book.cash_weight.ge(1 - 1e-10).any())
            diagnostic = result.diagnostics.loc[result.diagnostics.case_id.eq(key)]
            self.assertEqual(len(diagnostic), len(book))
            self.assertTrue(diagnostic.loc[diagnostic.cash_weight.eq(1), "effective_n"].eq(0).all())
            inspected = result.allocation_case(key, result.signals.date.min())
            self.assertIn("positive_year_fraction", inspected)
        base = result.books[result.case_id(STRATEGY_KEY)]
        adjusted = result.books[result.case_id(STRATEGY_KEY, "loss_adjusted")]
        self.assertTrue(adjusted.cash_weight.ge(base.cash_weight - 1e-12).all())

    def test_saved_roundtrip_csv_import_and_allocation_charts_do_not_fit(self):
        store = ExperimentStore(self.root / "artifacts", project_root=self.root)
        run_id = store.save(self.result)
        with patch("research.runner.ExperimentRunner.run", side_effect=AssertionError("no fit on load")):
            loaded = store.load(run_id)
            matrices = allocation_matrices(loaded, STRATEGY_KEY)
            self.assertIn("Cash", matrices["baseline"])
            for matrix in matrices.values():
                np.testing.assert_allclose(matrix.sum(axis=1), 1)
            for frame in composition_frames(matrices, assets=[]).values():
                self.assertIn("Cash", frame)
                np.testing.assert_allclose(frame.sum(axis=1), 1)
            csv = pd.read_csv(io.StringIO(allocation_history_csv(matrices)))
            self.assertIn("Cash", csv.stock_id.values)
            report = self.root / "exported"
            self.result.save(report)
            imported = store.load(import_csv_report(store, report))
        for case in self.result.books:
            pd.testing.assert_frame_equal(loaded.books[case], self.result.books[case])
            np.testing.assert_allclose(imported.books[case].simple_return, self.result.books[case].simple_return)
            if STRATEGY_KEY in case:
                np.testing.assert_allclose(imported.books[case].cash_weight, self.result.books[case].cash_weight)

    def test_app_displays_cash_and_annual_screen_without_computation(self):
        with tempfile.TemporaryDirectory() as storage, patch.dict("os.environ", {"MLFACTOR_ARTIFACT_ROOT": storage}), \
                patch("research.runner.ExperimentRunner.run", side_effect=AssertionError("browsing must not compute")):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
            app.session_state["training_result"] = self.result
            app.run()
            self.assertFalse(app.exception)
            app.checkbox(key="basic_show_allocations").check().run()
            app.selectbox(key="basic_allocations_strategy").select("Annual consistency + positive momentum + cash").run()
            self.assertFalse(app.exception)
            self.assertTrue(any(item.label == "Cash at this rebalance" for item in app.metric))
            self.assertTrue(any("Average cash" in frame.value for frame in app.dataframe))
            self.assertTrue(any("positive_year_fraction" in frame.value for frame in app.dataframe))
            app.multiselect(key="basic_allocations_assets").set_value(["Cash"])
            app.radio(key="basic_allocations_view").set_value("Portfolio composition").run()
            self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
