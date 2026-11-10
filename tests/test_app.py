"""Five-stage presentation and saved-only interactions, scoped by tab."""
from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np
from streamlit.testing.v1 import AppTest

from research import ExperimentRunner, RunConfig, default_model_registry, default_registry
from research.store import ExperimentStore
from test_research import toy_panel

APP = Path(__file__).resolve().parents[1] / "app.py"
TABS = ["Strategies", "Basic ML", "Grouping With ML", "Feature selection", "Tuned model", "Feature engineering", "Final test"]


def plot_spec(element):
    return json.loads(element.proto.figure.spec)


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage = tempfile.TemporaryDirectory()
        cls.root = Path(cls.storage.name)
        panel = toy_panel(84, 8)
        dates = panel.date.drop_duplicates()
        panel.loc[panel.date.isin(dates.iloc[:72])].to_csv(cls.root / "mlfactor_long_only_train.csv", index=False)
        panel.loc[panel.date.isin(dates.iloc[72:])].to_csv(cls.root / "mlfactor_long_only_test.csv", index=False)
        cls.config = RunConfig(tuple(default_registry()), include_ml=True, models=tuple(default_model_registry()))
        cls.result = ExperimentRunner(cls.root).run(cls.config)
        cls.baseline = ExperimentRunner(cls.root).run(replace(cls.config, include_ml=False))
        cls.store = ExperimentStore(cls.root / "artifacts", project_root=cls.root)
        cls.store.save(cls.result)

    @classmethod
    def tearDownClass(cls):
        cls.storage.cleanup()

    def setUp(self):
        self.env = patch.dict("os.environ", {"MLFACTOR_ARTIFACT_ROOT": str(self.store.root)})
        self.guard = patch("research.store.ExperimentRunner.run", side_effect=AssertionError("UI browsing must not fit"))
        self.env.start(); self.guard.start()
        self.addCleanup(self.guard.stop); self.addCleanup(self.env.stop)

    def app(self, result=None):
        app = AppTest.from_file(str(APP), default_timeout=90)
        app.session_state["training_result"] = self.result if result is None else result
        app.run()
        self.assertFalse(app.exception)
        return app

    def test_five_tabs_and_first_tab_has_no_prediction_content(self):
        app = self.app()
        self.assertEqual([tab.label for tab in app.tabs], TABS)
        first = app.tabs[0]
        self.assertFalse(any("Version / model" in frame.value or "Brier score" in frame.value for frame in first.dataframe))
        self.assertFalse(any("model" in item.label.lower() for item in first.multiselect))
        names = [trace["name"] for trace in plot_spec(first.get("plotly_chart")[0])["data"]]
        self.assertTrue(all("Original" in name for name in names))
        self.assertFalse(any("advantage" in name for name in names))
        self.assertFalse(app.tabs[2].dataframe)
        self.assertFalse(app.tabs[2].button)
        self.assertEqual(app.tabs[4].label, "Tuned model")
        self.assertTrue(any("December 2006" in item.value for item in app.tabs[4].info))
        self.assertFalse(app.tabs[4].dataframe)
        self.assertFalse(app.tabs[-1].dataframe)
        self.assertTrue(any("pending" in item.value for item in app.tabs[-1].info))
        self.assertEqual(app.session_state["training_result"].config, self.config)

    def test_basic_models_have_prediction_scores_control_and_paired_results(self):
        app = self.app()
        basic = app.tabs[1]
        scores = next(frame.value for frame in basic.dataframe if "Brier score" in frame.value)
        self.assertEqual(len(scores), 5)
        self.assertTrue({"Log loss", "ROC AUC", "Accuracy (p ≥ 0.5)"}.issubset(scores))
        self.assertTrue(scores.Predictor.str.contains("Historical", case=False).any())
        tables = [frame.value for frame in basic.dataframe if "Version / model" in frame.value]
        self.assertEqual(sum(frame["Version / model"].eq("Original").sum() for frame in tables), 14)
        app.multiselect(key="basic_models").set_value(["Logistic regression"]).run()
        self.assertFalse(app.exception)
        scores = next(frame.value for frame in app.tabs[1].dataframe if "Brier score" in frame.value)
        self.assertEqual(len(scores), 2)
        self.assertEqual(app.session_state["training_result"].config, self.config)

    def test_baseline_and_paired_allocations_use_separate_widgets(self):
        app = self.app()
        app.checkbox(key="strategies_show_allocations").check().run()
        first = app.tabs[0]
        self.assertFalse(any("probability_of_loss" in frame.value for frame in first.dataframe))
        app.checkbox(key="basic_show_allocations").check().run()
        app.selectbox(key="basic_allocations_strategy").select("Positive momentum, score weights").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("probability_of_loss" in frame.value for frame in app.tabs[1].dataframe))
        assets = app.multiselect(key="basic_allocations_assets").options[:2]
        app.multiselect(key="basic_allocations_assets").set_value(assets)
        app.radio(key="basic_allocations_view").set_value("Portfolio composition").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio(key="strategies_allocations_view").value, "Asset-by-month heatmap")
        self.assertTrue(any(len(spec.get("data", [])) == 10 for spec in [plot_spec(p) for p in app.tabs[1].get("plotly_chart")]))

    def test_group_filters_and_select_all_stay_within_their_stage(self):
        app = self.app()
        app.selectbox(key="strategies_family").select("Momentum baskets").run()
        app.button(key="strategies_select_all_chart_strategies").click().run()
        self.assertEqual(len(app.multiselect(key="strategies_chart_strategies").value), 2)
        self.assertEqual(app.selectbox(key="basic_ml_family").value, "All strategy families")
        app.button(key="basic_ml_select_all_chart_strategies").click().run()
        self.assertEqual(len(app.multiselect(key="basic_ml_chart_strategies").value), 14)
        app.checkbox(key="strategies_show_guide").check().run()
        app.selectbox(key="strategies_guide_strategy").select("Annual consistency + positive momentum + cash").run()
        self.assertFalse(app.exception)
        self.assertFalse(any("The ML version" in item.value for item in app.tabs[0].markdown))

    def test_empty_strategies_or_models_are_honest_empty_states(self):
        app = self.app()
        app.multiselect(key="selected_strategies").set_value([]).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.tabs[0].dataframe)
        self.assertTrue(any("Select at least one" in item.value for item in app.tabs[0].info))
        app.multiselect(key="selected_strategies").set_value(["Equal-weight universe"]).run()
        app.multiselect(key="basic_models").set_value([]).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Choose at least one saved model" in item.value for item in app.tabs[1].info))

    def test_non_ml_source_and_saved_basic_models_can_be_reviewed_independently(self):
        app = self.app(self.baseline)
        self.assertTrue(any("strategy-table" in item.value for item in app.tabs[0].markdown))
        self.assertTrue(any("Brier score" in frame.value for frame in app.tabs[1].dataframe))
        self.assertFalse(app.session_state["training_result"].config.include_ml)

    def test_missing_basic_library_has_an_honest_empty_state(self):
        with patch("research.store.ExperimentStore.list_runs", return_value=[]):
            app = self.app(self.baseline)
        self.assertFalse(app.tabs[1].dataframe)
        self.assertTrue(any("No downside-model predictions" in item.value for item in app.tabs[1].info))

    def test_log_return_views_and_growth_metrics_use_saved_portfolios(self):
        from app import wealth_chart
        app = self.app()
        original = self.result.case_id("equal_weight_universe")
        book = self.result.books[original]
        for index in (0, 1):
            if index == 0:
                from strategy_results_ui import metrics_table
                tables = [metrics_table(self.result, self.result.metrics.loc[
                    self.result.metrics.variant.eq("baseline")], baseline_only=True).data]
            else:
                tables = [frame.value for frame in app.tabs[index].dataframe if "CAGR" in frame.value]
            self.assertTrue(tables)
            for table in tables:
                self.assertIn("Annual log growth", table)
                np.testing.assert_allclose(np.expm1(table["Annual log growth"]), table.CAGR)
        app.checkbox(key="strategies_log_wealth").check().run()
        app.selectbox(key="strategies_return_measure").select("Monthly log return").run()
        self.assertFalse(app.exception)
        self.assertTrue(app.checkbox(key="strategies_log_wealth").disabled)
        spec = plot_spec(app.tabs[0].get("plotly_chart")[0])
        self.assertEqual(spec["layout"]["yaxis"]["title"]["text"], "Monthly log return")
        self.assertEqual(spec["layout"]["yaxis"]["type"], "linear")
        self.assertEqual(app.selectbox(key="basic_ml_return_measure").value, "Growth of $1")
        for measure in ("Monthly log return", "Cumulative log return"):
            app.selectbox(key="basic_ml_return_measure").select(measure).run()
            self.assertFalse(app.exception)
            figure = wealth_chart(self.result, [original], logarithmic=True, return_measure=measure)
            expected = book.log_return if measure == "Monthly log return" else np.r_[0, book.log_return.cumsum()]
            np.testing.assert_allclose(figure.data[0].y, expected)
            np.testing.assert_allclose(figure.data[1].y, np.r_[0, book.drawdown])
            self.assertEqual(figure.layout.yaxis.type, "linear")

    def test_explicit_compute_control_always_requests_training_only(self):
        app = self.app()
        with patch("research.store.ExperimentStore.get_or_run", return_value=self.baseline) as compute:
            app.button(key="compute_strategies_run").click().run()
        self.assertFalse(app.exception)
        requested = compute.call_args.args[0]
        self.assertEqual(requested.evaluation, "training_validation")
        self.assertFalse(requested.include_ml)
        self.assertTrue(app.session_state["training_result"].cases.variant.eq("baseline").all())


if __name__ == "__main__":
    unittest.main()
