"""Saved-model publication parity and no-computation Basic ML interactions."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from app_charts import allocation_matrices
from published_results import PublishedStrategies, export_strategies
from research import ExperimentRunner, RunConfig, default_model_registry, default_registry
from research.store import ExperimentStore
from result_views import select_result_view
from test_research import toy_panel

ROOT = Path(__file__).resolve().parents[1]


class PublishedBasicMLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        toy_panel(84, 8).to_csv(cls.root / "mlfactor_long_only_train.csv", index=False)
        cls.store = ExperimentStore(cls.root / "artifacts", project_root=cls.root)
        cls.sources = {}
        cls.ids = {}
        for target, scope in (("loss", "pooled"), ("upside", "pooled"), ("loss", "per_stock"), ("upside", "per_stock")):
            models = tuple(default_model_registry()) if scope == "pooled" else ("logistic",)
            config = RunConfig(tuple(default_registry()), include_ml=True, models=models,
                cash_momentum_months=3, prediction_target=target, model_scope=scope)
            result = ExperimentRunner(cls.root).run(config)
            run_id = cls.store.save(result)
            cls.ids[target, scope] = run_id
            cls.sources[run_id] = result
        cls.destination = export_strategies(cls.store, cls.ids["loss", "pooled"], cls.root / "publication",
                                            basic_run_ids=list(cls.sources))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_published_directions_scopes_and_model_summaries_match_sources(self):
        publication = PublishedStrategies(self.destination)
        library = publication.basic_store()
        self.assertEqual(set(row["run_id"] for row in library.list_runs()), set(self.sources))
        with patch("pandas.read_csv", side_effect=AssertionError("no CSVs")), patch(
                "research.store.ExperimentStore.__init__", side_effect=AssertionError("no catalog")):
            for run_id, source in self.sources.items():
                result = library.load(run_id)
                self.assertEqual(result.config, source.config)
                self.assertEqual(result.manifest, json.loads(json.dumps(source.manifest, default=str)))
                for frame in ("cases", "metrics", "monthly", "paired", "model_metrics"):
                    pd.testing.assert_frame_equal(getattr(result, frame), getattr(source, frame))
                for model, batch in result.prediction_batches.items():
                    self.assertTrue(batch.predictions.empty)
                    self.assertTrue(batch.fits.empty)
                    pd.testing.assert_frame_equal(batch.calibration, source.prediction_batches[model].calibration)
                    predictions = source.prediction_batches[model].predictions
                    expected = predictions.prediction_source.value_counts().to_dict() if "prediction_source" in predictions else {}
                    self.assertEqual(library.prediction_sources(result, model), expected)
                self.assertFalse(result.weights)
            self.assertFalse(any('/allocations/' in name or name.endswith('/fits.parquet') for name in publication._frames))
            for run_id, source in self.sources.items():
                result = library.load(run_id)
                for model in result.prediction_batches:
                    pd.testing.assert_frame_equal(library.model_fits(result, model), source.prediction_batches[model].fits)
                for strategy in result.cases.strategy.drop_duplicates():
                    models = list(result.prediction_batches)
                    allocation = library.allocations(result, strategy, models)
                    actual = allocation_matrices(allocation, strategy, models)
                    expected = allocation_matrices(source, strategy, models)
                    for model in actual:
                        pd.testing.assert_frame_equal(actual[model], expected[model])

    def test_both_layouts_and_direction_scope_controls_use_only_published_results(self):
        with patch.dict("os.environ", {"MLFACTOR_APP_MODE": "deploy", "MLFACTOR_PUBLISHED_ROOT": str(self.destination)}), patch(
                "pandas.read_csv", side_effect=AssertionError("no raw data")), patch(
                "research.store.ExperimentStore.__init__", side_effect=AssertionError("no research store")), patch(
                "research.runner.ExperimentRunner.run", side_effect=AssertionError("no fitting/replay")):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
            self.assertFalse(app.exception)
            basic = app.tabs[1]
            for label in ["1. Does the model predict downside?", "2. What changes in the portfolios?", "Change from each original portfolio"]:
                section = next(item for item in basic.expander if item.label == label)
                self.assertFalse(section.proto.expanded)
            self.assertFalse(any("compute" in button.label.lower() for button in basic.button))
            self.assertEqual(len(app.multiselect(key="basic_ml_chart_strategies").value), 14)
            app.selectbox(key="basic_pattern").select("Upside threshold boost").run()
            self.assertFalse(app.exception)
            self.assertTrue(any(x.label == "1. Does the model predict an up move?" for x in app.tabs[1].expander))
            table = next(f.value for f in app.tabs[1].dataframe if "Brier score" in f.value and "Predictor" in f.value)
            source = self.sources[self.ids["upside", "pooled"]]
            self.assertEqual(table["Brier score"].tolist(), source.model_metrics.brier_score.tolist())
            app.radio(key="basic_ml_layout").set_value("Full tables").run()
            table = next(f.value for f in app.tabs[1].dataframe if "Version / model" in f.value)
            self.assertEqual(len(table), 70)
            app.selectbox(key="basic_scope").select("Separate model per stock").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.multiselect(key="basic_models").options, ["Logistic regression"])
            expected = self.sources[self.ids["upside", "per_stock"]]
            scores = next(f.value for f in app.tabs[1].dataframe if "Brier score" in f.value and "Predictor" in f.value)
            self.assertEqual(scores["Brier score"].tolist(), expected.model_metrics.brier_score.tolist())
            self.assertTrue(any("Pooled fallback predictions" in f.value for f in app.tabs[1].dataframe))
            app.checkbox(key="basic_models_logistic_details").check().run()
            self.assertFalse(app.exception)
            pd.testing.assert_frame_equal(next(f.value for f in app.tabs[1].dataframe if "fit_snapshot" in f.value), expected.predictions.fits)
            app.checkbox(key="basic_show_allocations").check().run()
            app.selectbox(key="basic_allocations_strategy").select("Annual consistency + positive momentum + cash").run()
            self.assertFalse(app.exception)
            selected = next(f.value for f in app.tabs[1].dataframe if "Selected weight" in f.value)
            self.assertTrue(selected["Selected weight"].ge(0).all())
            self.assertFalse(any("probability_of_up" in f.value for f in app.tabs[1].dataframe))
            app.selectbox(key="basic_pattern").select("Downside protection").run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Training data only' in x.value for x in app.tabs[1].info))
            self.assertTrue(any('Training data only' in x.value for x in app.tabs[0].info))
            for tab in app.tabs[2:]:
                self.assertTrue(any("not been published" in x.value for x in tab.info))

    def test_missing_published_combination_cannot_reuse_a_local_browser_result(self):
        with tempfile.TemporaryDirectory() as target:
            output = export_strategies(self.store, self.ids["loss", "pooled"], Path(target) / "only_loss",
                                        basic_run_ids=[self.ids["loss", "pooled"]])
            with patch.dict("os.environ", {"MLFACTOR_APP_MODE": "deploy", "MLFACTOR_PUBLISHED_ROOT": str(output)}), patch(
                    "research.store.ExperimentStore.__init__", side_effect=AssertionError("no fallback")):
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45)
                app.session_state['basic_result'] = self.sources[self.ids['upside', 'pooled']]
                app.run()
                app.selectbox(key="basic_pattern").select("Upside threshold boost").run()
                self.assertFalse(app.exception)
                self.assertFalse(app.tabs[1].dataframe)
                self.assertTrue(any("No published Basic ML" in x.value for x in app.tabs[1].info))

    def test_publication_rejects_a_test_run_without_writing_a_partial_release(self):
        source = self.sources[self.ids["loss", "pooled"]]
        invalid = replace(source, config=replace(source.config, evaluation="test"))
        destination = self.root / "invalid"
        real_load = self.store.load
        with patch.object(self.store, "load", side_effect=lambda key: invalid if key == 'test' else real_load(key)):
            with self.assertRaises(ValueError):
                export_strategies(self.store, self.ids["loss", "pooled"], destination, basic_run_ids=['test'])
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
