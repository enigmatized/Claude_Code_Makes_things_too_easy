"""Portable feature/tuning/final readers, focus controls and descriptive hovers."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from published_results import PublishedStrategies, export_strategies
from published_analysis import check_features, check_tuning, check_final
from research import RunConfig, default_model_registry
from research.feature_studies import FOREST_RECIPES, create_feature_study, load_feature_study
from research.feature_reporting import importance_table
from research.store import ExperimentStore
from research.tuning_holdout import create_tuning_holdout, save_tuning_holdout
from test_research import toy_panel
from test_final_books import _fake_study

ROOT = Path(__file__).resolve().parents[1]


class PublishedAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        toy_panel(60, 8).to_csv(cls.root / 'mlfactor_long_only_train.csv', index=False)
        cls.store = ExperimentStore(project_root=cls.root)
        config = RunConfig(('equal_weight_universe', 'momentum_weighted'), include_ml=True,
                           validation_history_months=24, models=tuple(default_model_registry()))
        recipes = {**FOREST_RECIPES, 'forest_10': replace(FOREST_RECIPES['forest_10'], forest_trees=5, forest_leaf=2)}
        cls.feature_record = create_feature_study(cls.store, config, recipes=recipes)
        cls.features = load_feature_study(cls.store, cls.feature_record['study_id'])
        cls.tuning = create_tuning_holdout(cls.root, tuple(config.feature_spec.columns[:4]), model_keys=('logistic',),
            strategies=config.strategies, origin='2003-12-31', min_train_months=12, n_splits=2,
            grid={'model__C':[.1,1.]})
        cls.tuning_dir = save_tuning_holdout(cls.tuning, cls.store.root / 'tuning_studies')
        cls.final_dir = _fake_study(cls.store.root)
        cls.destination = export_strategies(cls.store, cls.feature_record['runs']['all_20'], cls.root / 'publication',
            feature_study_id=cls.feature_record['study_id'], tuning_studies=[cls.tuning_dir], final_book_study=cls.final_dir)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_portable_feature_frames_and_tuning_trials_match_saved_sources(self):
        pub = PublishedStrategies(self.destination)
        with patch('pandas.read_csv', side_effect=AssertionError('no CSVs')), patch(
                'research.store.ExperimentStore.__init__', side_effect=AssertionError('no catalog')):
            study = pub.analysis_section('features').study()
            for name in ('fits', 'calibration', 'portfolio_comparison', 'predictive_comparison'):
                pd.testing.assert_frame_equal(study[name], self.features[name])
            for recipe, summary in study['summaries'].items():
                pd.testing.assert_frame_equal(summary['monthly'], self.features['summaries'][recipe]['monthly'])
            self.assertIn('Random forest top 10', study['method_notes'])
            tuning = pub.analysis_section('tuning')
            item = tuning.list_studies()[0]
            saved = tuning.study(item)
            pd.testing.assert_frame_equal(saved['monthly'], self.tuning['monthly'])
            pd.testing.assert_frame_equal(saved['predictive'], self.tuning['predictive'])
            self.assertFalse(any('/trials.parquet' in path for path in pub._frames))
            self.assertEqual(len(tuning.trials(item, 'logistic')), 2)
            final = pub.analysis_section('final')
            self.assertEqual(list(final.windows()), ['test'])
            self.assertEqual(final.study()['metadata']['fit_policy'], 'frozen_at_origin')

    def test_deployed_features_have_explanatory_feature_bar_hovers_and_optional_registry_results(self):
        with patch.dict('os.environ', {'MLFACTOR_APP_MODE':'deploy','MLFACTOR_PUBLISHED_ROOT':str(self.destination)}), patch(
                'pandas.read_csv', side_effect=AssertionError('no raw CSV')), patch(
                'research.store.ExperimentStore.__init__', side_effect=AssertionError('no store')), patch(
                'research.selection.TrainingFeatureSelector.fit', side_effect=AssertionError('no selection fits')):
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(app.checkbox(key='features_main_families_only').value)
            # With no advanced book in this small fixture the existing registry
            # evidence stays reviewable; no missing advanced results are invented.
            bars = [json.loads(chart.proto.figure.spec) for chart in app.get('plotly_chart')]
            forest = next(spec for spec in bars if spec['data'][0].get('type') == 'bar')
            rows = importance_table(self.features)
            self.assertEqual([row[0] for row in forest['data'][0]['customdata']], rows.meaning.tolist())
            self.assertIn('customdata[0]', forest['data'][0]['hovertemplate'])
            self.assertTrue(all(bool(row[1]) for row in forest['data'][0]['customdata']))
            app.checkbox(key='features_main_families_only').uncheck().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.selectbox(key='features_portfolio_model'))

    def test_tuning_focus_and_saved_final_window_work_without_computation(self):
        with patch.dict('os.environ', {'MLFACTOR_APP_MODE':'deploy','MLFACTOR_PUBLISHED_ROOT':str(self.destination)}), patch(
                'pandas.read_csv', side_effect=AssertionError('no CSVs')), patch(
                'research.store.ExperimentStore.__init__', side_effect=AssertionError('no catalog')), patch(
                'research.final_books.window_books', side_effect=AssertionError('windows must be precomputed')):
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key='tuning_holdout_strategy').options, ['Equal-weight universe'])
            app.checkbox(key='tuning_main_families_only').uncheck().run()
            app.selectbox(key='tuning_holdout_strategy').select('Positive momentum, score weights').run()
            self.assertFalse(app.exception)
            app.selectbox(key='tuning_holdout_target').select('Next up — positive next-month return').run()
            self.assertFalse(app.exception)
            self.assertTrue(any('No saved tuning comparison' in info.value for info in app.info))
            self.assertEqual(app.selectbox(key='final_books_window').value, 'Official test only')
            self.assertTrue(app.checkbox(key='final_books_show_all').disabled)
            app.checkbox(key='final_main_families_only').uncheck().run()
            app.checkbox(key='final_books_show_all').check().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.checkbox(key='final_books_show_all').disabled)

    def test_training_and_frozen_test_guards_reject_period_or_fit_leakage(self):
        pub = PublishedStrategies(self.destination)
        features = pub.analysis_section('features').study()
        features['metadata'] = {**features['metadata'], 'config':{**features['metadata']['config'], 'evaluation':'test'}}
        with self.assertRaises(ValueError): check_features(features)
        tuning = {**self.tuning, 'metadata':{**self.tuning['metadata'], 'test_csv_loaded':True}}
        with self.assertRaises(ValueError): check_tuning(tuning)
        final = pub.analysis_section('final').study()
        final['fits'] = final['fits'].assign(fit_snapshot=pd.Timestamp('2010-01-31'))
        with self.assertRaises(ValueError): check_final(final)


if __name__ == '__main__':
    unittest.main()
