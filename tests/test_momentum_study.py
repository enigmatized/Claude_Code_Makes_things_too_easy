"""Lookback arithmetic, history gaps, controlled studies and read-only app review."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import ExperimentRunner, RunConfig
from research.cash_strategy import STRATEGY_KEY
from research.data import prepare_data
from research.legacy import KEYS, TARGET, build_signals, load_panel
from research.momentum_studies import compare_momentum_runs, create_momentum_study, load_momentum_study
from research.store import ExperimentStore, config_from_dict
from test_research import toy_panel


class MomentumTimingTests(unittest.TestCase):
    def test_two_month_product_history_requirement_and_gap(self):
        panel = toy_panel(36, 2)
        dates = panel.date.drop_duplicates().tolist()
        panel[TARGET] = .01
        panel.loc[panel.date.eq(dates[1]), TARGET] = .2
        panel.loc[panel.date.eq(dates[2]), TARGET] = -.05
        # A missing March snapshot removes April's earned return for stock 1.
        panel = panel.loc[~(panel.stock_id.eq(1) & panel.date.eq(dates[3]))]
        short = build_signals(panel, lookback=2).set_index(KEYS)
        long = build_signals(panel, lookback=12).set_index(KEYS)
        self.assertAlmostEqual(short.loc[(dates[3], 0), "trailing_momentum"], 1.2 * .95 - 1)
        self.assertTrue(short.loc[(dates[3], 0), "eligible"])
        self.assertFalse(long.loc[(dates[3], 0), "eligible"])
        self.assertFalse(short.loc[(dates[4], 1), "eligible"])
        self.assertFalse(short.loc[(dates[5], 1), "eligible"])
        self.assertTrue(short.loc[(dates[6], 1), "eligible"])
        changed = panel.copy()
        changed.loc[changed.date.ge(dates[3]), TARGET] = 9
        later = build_signals(changed, lookback=2).set_index(KEYS)
        pd.testing.assert_frame_equal(short.loc[[dates[3]], :], later.loc[[dates[3]], :])

    def test_config_roundtrip_defaults_and_invalid_values(self):
        config = RunConfig(cash_momentum_months=2)
        self.assertEqual(config_from_dict(asdict(config)), config)
        legacy = asdict(RunConfig())
        legacy.pop("cash_momentum_months")
        self.assertEqual(config_from_dict(legacy).cash_momentum_months, 12)
        for value in [True, 0, 1, 13, 2.5, "2"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                RunConfig(cash_momentum_months=value)


class SavedMomentumStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        panel = toy_panel(84, 8)
        panel.to_csv(cls.root / "mlfactor_long_only_train.csv", index=False)
        cls.store = ExperimentStore(cls.root / "artifacts", project_root=cls.root)
        cls.config = RunConfig(("equal_weight_universe", "momentum_weighted", STRATEGY_KEY),
                               include_ml=True, models=("logistic",))
        cls.study = create_momentum_study(cls.store, cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_four_windows_share_pool_predictions_and_other_portfolios(self):
        previous = None
        for months, run_id in self.study["runs"].items():
            result = self.store.load(run_id)
            if previous is not None:
                pd.testing.assert_frame_equal(result.screens["annual_consistency"], previous.screens["annual_consistency"])
                pd.testing.assert_frame_equal(result.predictions.predictions, previous.predictions.predictions)
                for strategy in ["equal_weight_universe", "momentum_weighted"]:
                    pd.testing.assert_frame_equal(result.books[result.case_id(strategy)], previous.books[previous.case_id(strategy)])
            previous = result
            inspection = result.allocation_case(result.case_id(STRATEGY_KEY), result.signals.date.min())
            self.assertTrue(inspection.history_months.le(int(months)).all())
            self.assertTrue(inspection.loc[inspection.weight.gt(0), "trailing_momentum"].gt(0).all())

    def test_history_uses_actual_short_window_and_future_labels_cannot_change_weights(self):
        original = self.store.load(self.study["runs"]["2"])
        start = original.signals.date.min()
        panel = load_panel("train", self.root)
        panel.loc[panel.date.ge(start), TARGET] = -.99
        with patch("research.runner.load_panel", return_value=panel):
            changed = ExperimentRunner(self.root).run(replace(self.config, include_ml=False, cash_momentum_months=2))
        first = original.weights[original.case_id(STRATEGY_KEY)].loc[lambda f: f.date.eq(start)]
        second = changed.weights[changed.case_id(STRATEGY_KEY)].loc[lambda f: f.date.eq(start)]
        pd.testing.assert_frame_equal(first, second)
        pd.testing.assert_frame_equal(original.screens["annual_consistency"], changed.screens["annual_consistency"])

    def test_saved_study_and_exact_cache_never_compute(self):
        with patch("research.store.ExperimentRunner.run", side_effect=AssertionError("no fitting")):
            loaded = load_momentum_study(self.store, self.study["study_id"])
            result = self.store.get_or_run(replace(self.config, cash_momentum_months=2))
        self.assertEqual(len(loaded["portfolio_comparison"]), 8)
        self.assertTrue(result.cache_info["cache_hit"])
        self.assertNotEqual(self.store.fingerprint(self.config), self.store.fingerprint(replace(self.config, cash_momentum_months=2)))

    def test_mismatched_protocol_and_test_period_rejected(self):
        with self.assertRaises(ValueError):
            create_momentum_study(self.store, replace(self.config, evaluation="test"))
        with self.assertRaises(ValueError):
            compare_momentum_runs(self.store, {**self.study["runs"], "2": self.study["runs"]["3"]})
        with patch.dict("os.environ", {"MLFACTOR_READ_ONLY": "1"}), self.assertRaises(RuntimeError):
            create_momentum_study(self.store, self.config)

    def test_app_reviews_windows_models_and_loads_allocations_without_fitting(self):
        with patch.dict("os.environ", {"MLFACTOR_ARTIFACT_ROOT": str(self.store.root)}), patch(
                "research.store.ExperimentRunner.run", side_effect=AssertionError("saved review must not compute")):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=90).run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Momentum months" in frame.value for frame in app.dataframe))
            app.selectbox(key="ml_momentum_model").select("Logistic regression").run()
            self.assertFalse(app.exception)
            self.assertTrue(any(len(frame.value) == 8 and "Momentum months" in frame.value for frame in app.dataframe))
            app.selectbox(key="ml_momentum_allocations").set_value(2)
            app.button(key="ml_momentum_load_allocations").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("2-month momentum" in item.value for item in app.caption))
            self.assertEqual(app.session_state["training_result"].config.cash_momentum_months, 2)


if __name__ == "__main__":
    unittest.main()
