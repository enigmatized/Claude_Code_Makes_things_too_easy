"""Stage isolation, exact saved values, provenance and final-test readiness."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from research import ExperimentRunner, FeatureSpec, RunConfig, default_model_registry
from research.legacy import FEATURES
from research.store import ExperimentStore
from result_views import pinned_final_summary, select_result_view
from test_research import toy_panel


class ResultViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        frame = toy_panel(84, 6)
        dates = frame.date.drop_duplicates()
        frame.loc[frame.date.isin(dates.iloc[:72])].to_csv(cls.root / 'mlfactor_long_only_train.csv', index=False)
        frame.loc[frame.date.isin(dates.iloc[72:])].to_csv(cls.root / 'mlfactor_long_only_test.csv', index=False)
        cls.config = RunConfig(('equal_weight_universe', 'momentum_weighted'), include_ml=True,
            models=tuple(default_model_registry()), feature_spec=FeatureSpec(tuple(FEATURES[:2])))
        cls.source = ExperimentRunner(cls.root).run(cls.config)
        cls.store = ExperimentStore(cls.root / 'artifacts', project_root=cls.root)
        cls.train_id = cls.store.save(cls.source)
        cls.test = ExperimentRunner(cls.root).run(replace(cls.config, evaluation='test', fit_policy='frozen_at_origin'))
        cls.test_id = cls.store.save(cls.test)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_baseline_view_preserves_source_config_and_exact_baseline_data(self):
        view = select_result_view(self.source, baseline_only=True)
        self.assertTrue(view.cases.variant.eq('baseline').all())
        self.assertFalse(view.prediction_batches)
        self.assertIsNone(view.predictions)
        self.assertTrue(view.paired.empty)
        self.assertIs(view.config, self.source.config)
        self.assertIs(view.manifest, self.source.manifest)
        self.assertEqual(view.cache_info['run_id'], self.train_id)
        for key, book in view.books.items():
            pd.testing.assert_frame_equal(book, self.source.books[key])
            pd.testing.assert_frame_equal(view.weights[key], self.source.weights[key])
        self.assertEqual(len(self.source.prediction_batches), 4)
        self.assertEqual(len(self.source.cases), 10)

    def test_model_and_strategy_filter_retains_own_original_and_matching_changes(self):
        view = select_result_view(self.source, strategies=['momentum_weighted'], models=['random_forest'])
        self.assertEqual(set(view.cases.model_id), {'none', 'random_forest'})
        self.assertEqual(len(view.paired), 1)
        self.assertEqual(set(view.prediction_batches), {'random_forest'})
        expected = self.source.paired.loc[self.source.paired.case_id.isin(view.cases.case_id)]
        pd.testing.assert_frame_equal(view.paired, expected)
        pd.testing.assert_frame_equal(view.predictions.predictions,
                                      self.source.prediction_batches['random_forest'].predictions)

    def test_training_view_refuses_test_data_and_unknown_strategies(self):
        with self.assertRaises(ValueError):
            select_result_view(self.test)
        with self.assertRaises(ValueError):
            select_result_view(self.source, strategies=['does_not_exist'])
        self.assertTrue(select_result_view(self.source, strategies=[]).cases.empty)

    def test_frozen_test_is_not_final_without_recorded_choice_and_tuning(self):
        path = self.root / 'release.json'
        release = {'artifact_root': 'artifacts', 'run_id': self.test_id, 'final_choice_case_id': None}
        path.write_text(json.dumps(release))
        with patch('research.store.ExperimentRunner.run', side_effect=AssertionError('reading cannot fit')):
            self.assertIsNone(pinned_final_summary(self.store, path))
            with patch('research.reporting.readiness', return_value={'recorded_requirements': True}):
                self.assertEqual(pinned_final_summary(self.store, path)['run_id'], self.test_id)
            release['run_id'] = self.train_id
            path.write_text(json.dumps(release))
            self.assertIsNone(pinned_final_summary(self.store, path))


if __name__ == '__main__':
    unittest.main()
