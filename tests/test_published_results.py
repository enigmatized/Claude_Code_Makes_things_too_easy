"""Offline publication parity, lazy reads, corruption and training-only boundaries."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from app_charts import allocation_matrices, return_curve
from published_results import PublishedStrategies, export_strategies, require_training
from research import ExperimentRunner, RunConfig, default_registry
from research.store import ExperimentStore
from strategy_results_ui import baseline_table_html, metrics_table
from test_research import toy_panel

ROOT = Path(__file__).resolve().parents[1]


class PublishedResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        toy_panel(84, 8).to_csv(cls.root / "mlfactor_long_only_train.csv", index=False)
        cls.config = RunConfig(tuple(default_registry()), include_ml=True, cash_momentum_months=3)
        cls.source = ExperimentRunner(cls.root).run(cls.config)
        cls.store = ExperimentStore(cls.root / "artifacts", project_root=cls.root)
        cls.run_id = cls.store.save(cls.source)
        cls.publication = export_strategies(cls.store, cls.run_id, cls.root / "publication")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_export_retains_exact_original_results_without_raw_data_or_models(self):
        p = PublishedStrategies(self.publication)
        with patch("pandas.read_csv", side_effect=AssertionError("no source CSV needed")), patch(
                "research.store.ExperimentStore.load", side_effect=AssertionError("no archive needed")):
            result = p.result()
            expected = self.source.metrics.loc[self.source.metrics.variant.eq("baseline")]
            pd.testing.assert_frame_equal(result.metrics, expected)
            self.assertEqual(result.config, self.source.config)  # preserve original provenance
            self.assertFalse(result.prediction_batches)
            self.assertEqual(len(p._frames), 3)
            for strategy in result.cases.strategy:
                key = result.case_id(strategy)
                original = self.source.books[key]
                for measure in ("Growth of $1", "Monthly log return", "Cumulative log return"):
                    pd.testing.assert_series_equal(return_curve(result.books[key], measure), return_curve(original, measure))
                np.testing.assert_array_equal(result.books[key].drawdown, original.drawdown)
                allocated = p.allocations(result, strategy)
                pd.testing.assert_frame_equal(allocated.weights[key], self.source.weights[key])
                pd.testing.assert_frame_equal(allocation_matrices(allocated, strategy)["baseline"],
                                              allocation_matrices(self.source, strategy, [])["baseline"])
            self.assertFalse(result.weights)  # allocation reads do not mutate summary
        files = [str(path.relative_to(self.publication)) for path in self.publication.rglob("*") if path.is_file()]
        self.assertTrue(all(path.endswith((".parquet", ".json")) for path in files))
        self.assertFalse(any(word in path for path in files for word in ("prediction", "signals", "catalog", "train.csv", "test.csv")))
        np.testing.assert_allclose(np.expm1(result.metrics.annualized_log_growth), result.metrics.cagr)

    def test_publications_are_immutable_and_validate_files_on_demand(self):
        with self.assertRaises(FileExistsError):
            export_strategies(self.store, self.run_id, self.publication)
        with tempfile.TemporaryDirectory() as target:
            import shutil
            copied = Path(target) / "copy"
            shutil.copytree(self.publication, copied)
            p = PublishedStrategies(copied)
            key = next(iter(p.tab["weights"]))
            (copied / p.tab["weights"][key]).write_bytes(b"corrupt allocation")
            result = p.result()  # summary does not read/check unrelated allocations
            strategy = result.cases.set_index("case_id").loc[key].strategy
            with self.assertRaisesRegex(ValueError, "missing or damaged"):
                p.allocations(result, strategy)
            (copied / p.tab["frames"]["monthly"]).write_bytes(b"corrupt chart data")
            with self.assertRaisesRegex(ValueError, "missing or damaged"):
                PublishedStrategies(copied).result()
            with self.assertRaises(ValueError):
                p.frame("../mlfactor_long_only_train.csv")

    def test_training_guard_rejects_test_sources_periods_and_ml_cases(self):
        result = PublishedStrategies(self.publication).result()
        for config, manifest, monthly in [
            (replace(result.config, evaluation="test"), result.manifest, result.monthly),
            (result.config, {**result.manifest, "data_audit": {"source_splits": ["train", "test"]}}, result.monthly),
            (result.config, {**result.manifest, "source_sha256": {"mlfactor_long_only_test.csv": "abc"}}, result.monthly),
            (result.config, result.manifest, result.monthly.assign(return_month=pd.Timestamp("2010-01-31"))),
            (result.config, result.manifest, result.monthly.assign(variant="loss_adjusted")),
            (result.config, result.manifest, result.monthly.assign(snapshot_date=result.monthly.return_month)),
        ]:
            with self.subTest(config=config.evaluation, sources=manifest.get("source_sha256")), self.assertRaises(ValueError):
                require_training(config, manifest, monthly)

    def test_hover_text_has_real_rules_correct_window_and_escaped_content(self):
        p = PublishedStrategies(self.publication)
        result = p.result()
        html = baseline_table_html(result, result.metrics, p.tab["descriptions"])
        self.assertEqual(html.count('class="strategy-help"'), 14)
        self.assertIn("3 months in this run", html)
        self.assertIn("at least three complete calendar years", html)
        self.assertIn("Freeze the top 20%", html)
        self.assertIn("100% cash", html)
        self.assertIn('role="tooltip"', html)
        self.assertIn('tabindex="0"', html)
        altered = result.metrics.iloc[:1].assign(label='<script>alert("x")</script>')
        safe = baseline_table_html(result, altered, {altered.strategy.iloc[0]: '<img src=x onerror="alert(1)">'})
        self.assertNotIn("<script>", safe)
        self.assertNotIn("<img ", safe)
        # Same formatter/data as the old first table, despite a different renderer.
        pd.testing.assert_frame_equal(metrics_table(result, result.metrics, baseline_only=True).data,
                                      metrics_table(self.source, result.metrics, baseline_only=True).data)

    def test_deploy_browses_and_downloads_without_csv_catalog_or_computation(self):
        with patch.dict("os.environ", {"MLFACTOR_APP_MODE": "deploy", "MLFACTOR_PUBLISHED_ROOT": str(self.publication)}), patch(
                "pandas.read_csv", side_effect=AssertionError("No raw CSV reads")), patch(
                "research.store.ExperimentStore.__init__", side_effect=AssertionError("No research store")), patch(
                "research.runner.ExperimentRunner.run", side_effect=AssertionError("No strategy/model computation")):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            self.assertFalse(app.exception)
            first = app.tabs[0]
            self.assertTrue(any("Training data only" in item.value for item in first.info))
            self.assertFalse(any("compute" in button.label.lower() for button in app.button))
            self.assertTrue(any("strategy-help" in item.value for item in first.markdown))
            for tab in app.tabs[1:]:
                self.assertFalse(tab.dataframe)
                self.assertTrue(any("not been published" in info.value for info in tab.info))
            app.radio(key="strategies_layout").set_value("Full tables").run()
            html = next(item.value for item in app.tabs[0].markdown if "strategy-table" in item.value)
            self.assertEqual(html.count('class="strategy-help"'), 14)
            app.button(key="strategies_select_all_chart_strategies").click().run()
            self.assertEqual(len(app.multiselect(key="strategies_chart_strategies").value), 14)
            app.selectbox(key="strategies_return_measure").select("Monthly log return").run()
            app.checkbox(key="strategies_show_allocations").check().run()
            app.selectbox(key="strategies_allocations_strategy").select("Annual consistency + positive momentum + cash").run()
            self.assertFalse(app.exception)
            table = next(f.value for f in app.tabs[0].dataframe if "Selected weight" in f.value)
            self.assertTrue(table["Selected weight"].ge(0).all())
            self.assertTrue(app.tabs[0].get("download_button"))
            app.multiselect(key="published_strategy_selection").set_value([]).run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Select at least one" in x.value for x in app.tabs[0].info))

    def test_missing_publication_never_falls_back_to_a_research_run(self):
        with patch.dict("os.environ", {"MLFACTOR_APP_MODE": "deploy", "MLFACTOR_PUBLISHED_ROOT": str(self.root / "absent")}), patch(
                "research.store.ExperimentStore.__init__", side_effect=AssertionError("No fallback")):
            app = AppTest.from_file(str(ROOT / "app.py")).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertFalse(app.get("plotly_chart"))


if __name__ == "__main__":
    unittest.main()
