"""Grouping preserves every strategy/model value and needs no computation."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import ExperimentRunner, RunConfig, default_registry
from research.store import ExperimentStore
from strategy_groups import FAMILIES, change_format, change_matrix, family_for, grouped_strategies
from test_research import toy_panel


class FamilyAndMatrixTests(unittest.TestCase):
    def test_every_registered_strategy_has_exactly_one_family(self):
        registry = default_registry()
        declared = [strategy for family in FAMILIES for strategy in family.strategies]
        self.assertEqual(set(declared), set(registry))
        self.assertEqual(len(declared), len(set(declared)))
        self.assertEqual(len(grouped_strategies(registry)), 6)

    def test_grouping_keeps_input_order_and_unknown_extensions(self):
        keys = ["momentum_weighted", "custom", "momentum_equal_weight", "equal_weight_universe"]
        groups = grouped_strategies(keys)
        self.assertEqual(groups[1][1], ["momentum_weighted", "momentum_equal_weight"])
        self.assertEqual(groups[-1][1], ["custom"])
        self.assertEqual(family_for("custom").title, "Other strategies")

    def test_matrix_has_exact_cells_and_missing_pairs_stay_missing(self):
        rows = pd.DataFrame({"strategy": ["a", "a", "b"], "label": ["A", "A", "B"],
            "model_id": ["logit", "forest", "logit"], "model_label": ["Logistic", "Forest", "Logistic"],
            "change_cagr": [.001, -.002, 0]})
        matrix = change_matrix(rows, "change_cagr")
        self.assertEqual(matrix.loc["A", "Logistic"], .001)
        self.assertEqual(matrix.loc["A", "Forest"], -.002)
        self.assertEqual(matrix.loc["B", "Logistic"], 0)
        self.assertTrue(pd.isna(matrix.loc["B", "Forest"]))
        with self.assertRaises(ValueError):
            change_matrix(pd.concat([rows, rows.iloc[:1]]), "change_cagr")

    def test_changes_use_percentage_points_and_ratio_units(self):
        self.assertEqual(change_format(.001, True), "+0.10 pp")
        self.assertEqual(change_format(-.002, False), "-0.002")
        self.assertEqual(change_format(np.nan, True), "—")


class GroupedResultsAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        toy_panel(84, 8).to_csv(cls.root / "mlfactor_long_only_train.csv", index=False)
        cls.store = ExperimentStore(cls.root / "artifacts", project_root=cls.root)
        cls.result = ExperimentRunner(cls.root).run(RunConfig(tuple(default_registry()),
            include_ml=True, models=("logistic", "naive_bayes")))
        cls.store.save(cls.result)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def app(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=90)
        app.session_state["training_result"] = self.result
        return app.run()

    def test_grouped_overview_covers_all_strategies_and_exact_model_changes(self):
        with patch.dict("os.environ", {"MLFACTOR_ARTIFACT_ROOT": str(self.store.root)}), patch(
                "research.store.ExperimentRunner.run", side_effect=AssertionError("display cannot compute")):
            app = self.app()
            self.assertFalse(app.exception)
            self.assertEqual(app.radio(key="basic_ml_layout").value, "Grouped strategies")
            originals = [frame.value for frame in app.dataframe if "Version / model" in frame.value]
            self.assertEqual(len(originals), 14)
            self.assertTrue(all(len(frame) == 3 for frame in originals))
            models = {"Logistic regression", "Gaussian naive Bayes"}
            matrices = [frame.value for frame in app.dataframe if set(frame.value.columns) == models]
            self.assertEqual(len(matrices), 6)
            self.assertEqual(sum(len(frame) for frame in matrices), 14)
            combined = pd.concat(matrices)
            for _, row in self.result.paired.iterrows():
                self.assertEqual(combined.loc[row.label, row.model_label], row.change_cagr)

    def test_filter_metric_selection_full_view_and_select_all_do_not_fit(self):
        with patch.dict("os.environ", {"MLFACTOR_ARTIFACT_ROOT": str(self.store.root)}), patch(
                "research.store.ExperimentRunner.run", side_effect=AssertionError("display cannot compute")):
            app = self.app()
            app.selectbox(key="basic_ml_family").select("Momentum baskets").run()
            self.assertFalse(app.exception)
            app.button(key="basic_ml_select_all_chart_strategies").click().run()
            self.assertEqual(len(app.multiselect(key="basic_ml_chart_strategies").value), 2)
            app.selectbox(key="basic_ml_paired_metric").select("Maximum drawdown").run()
            self.assertFalse(app.exception)
            matrix = next(frame.value for frame in app.dataframe
                if set(frame.value.columns) == {"Logistic regression", "Gaussian naive Bayes"})
            self.assertEqual(len(matrix), 2)
            expected = self.result.paired.loc[self.result.paired.strategy.eq("momentum_weighted") & self.result.paired.model_id.eq("logistic")].iloc[0]
            self.assertEqual(matrix.loc[expected.label, expected.model_label], expected.change_max_drawdown)
            app.radio(key="basic_ml_layout").set_value("Full tables").run()
            self.assertFalse(app.exception)
            table = next(frame.value for frame in app.tabs[1].dataframe if "Version / model" in frame.value and "Strategy" in frame.value)
            self.assertEqual(len(table), 6)
            self.assertEqual(table["Version / model"].eq("Original").sum(), 2)
            app.selectbox(key="basic_ml_family").select("All strategy families").run()
            self.assertFalse(app.exception)
            app.button(key="basic_ml_select_all_chart_strategies").click().run()
            self.assertEqual(len(app.multiselect(key="basic_ml_chart_strategies").value), 14)
            app.selectbox(key="build_family").select("Volatility baskets").run()
            self.assertFalse(app.exception)
            app.button(key="select_all_build_family").click().run()
            self.assertEqual(set(app.multiselect(key="selected_strategies").value),
                {"High volatility, equal weight", "Low volatility, equal weight"})
            self.assertEqual(app.session_state["training_result"].config, self.result.config)


if __name__ == "__main__":
    unittest.main()
