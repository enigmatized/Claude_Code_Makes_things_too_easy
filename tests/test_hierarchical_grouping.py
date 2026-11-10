"""Matched clustering, no look-ahead, and saved-only hierarchical presentation."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree
from streamlit.testing.v1 import AppTest

from research import ExperimentRunner, RunConfig
from research.basket_momentum import (FEATURE, assign_correlation_baskets, create_basket_study,
                                      load_basket_study, save_basket_study)
from research.basket_adjustments import create_adjusted_basket_study
from research.grouping_charts import hierarchy_figure
from research.store import ExperimentStore
from published_results import PublishedStrategies, export_strategies, export_grouping_comparison
from test_basket_momentum import panel

ROOT = Path(__file__).resolve().parents[1]


class HierarchicalTimingTests(unittest.TestCase):
    def test_saved_tree_cut_matches_memberships_and_ignores_future_labels(self):
        frame = panel(months=24, stocks=8)
        date = frame.date.drop_duplicates().iloc[18]
        trees = []
        grouped = assign_correlation_baskets(frame, [date], n_clusters=3, grouping_method='ward', hierarchy=trees)
        tree = trees[0][['left', 'right', 'distance', 'count']].to_numpy()
        np.testing.assert_array_equal(grouped[FEATURE].to_numpy(dtype=int), cut_tree(tree, n_clusters=[3]).ravel())
        changed = frame.copy()
        changed.loc[changed.date.ge(date), 'R1M_Usd'] = .99
        changed.loc[:, 'Div_Yld'] = -99.
        again = assign_correlation_baskets(changed, [date], n_clusters=3, grouping_method='ward')
        pd.testing.assert_frame_equal(grouped, again)
        fig = hierarchy_figure(trees[0], date, 3)
        self.assertEqual(len(fig.layout.shapes), 1)
        self.assertIn('8 stocks', fig.layout.title.text)
        with patch.dict(os.environ, {'MLFACTOR_READ_ONLY': '1'}):
            with self.assertRaises(RuntimeError):
                assign_correlation_baskets(frame, [date], grouping_method='ward')

    def test_incomplete_and_constant_histories_are_excluded(self):
        frame = panel(months=24, stocks=8)
        frame.loc[frame.stock_id.eq(0), 'R1M_Usd'] = .02
        frame = frame.drop(frame.loc[frame.stock_id.eq(1)].index[12])
        grouped = assign_correlation_baskets(frame, [pd.Timestamp('2002-07-31')], n_clusters=3, grouping_method='ward')
        self.assertFalse(grouped.loc[grouped.stock_id.isin([0, 1]), 'eligible'].any())
        self.assertTrue(grouped.loc[~grouped.stock_id.isin([0, 1]), 'eligible'].all())
        with self.assertRaises(ValueError):
            assign_correlation_baskets(frame, [frame.date.max()], grouping_method='random_forest')


class PublishedHierarchicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        panel(months=42, stocks=10).to_csv(cls.root / 'mlfactor_long_only_train.csv', index=False)
        cls.store = ExperimentStore(project_root=cls.root)
        runs = {}
        for target in ('loss', 'upside'):
            result = ExperimentRunner(cls.root).run(RunConfig(('equal_weight_universe',), include_ml=True,
                prediction_target=target, validation_history_months=24))
            runs[target] = cls.store.save(result)
        cls.studies = {}
        for method in ('kmeans', 'ward'):
            base = create_basket_study(cls.root, n_clusters=3, validation_history_months=24, grouping_method=method)
            base_dir = save_basket_study(base, cls.store.root / 'basket_studies')
            overlay = create_adjusted_basket_study(cls.store, base_dir, runs['loss'], runs['upside'])
            directory = save_basket_study(overlay, cls.store.root / 'basket_studies')
            cls.studies[method] = (overlay, directory)
        cls.old = export_strategies(cls.store, runs['loss'], cls.root / 'old', basket_study=cls.studies['kmeans'][1])
        cls.output = export_grouping_comparison(cls.old, cls.studies['ward'][1], cls.root / 'new')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_saved_frames_and_prior_evidence_preserved(self):
        old, new = PublishedStrategies(self.old), PublishedStrategies(self.output)
        for path, digest in old.metadata['file_sha256'].items():
            self.assertEqual(new.metadata['file_sha256'][path], digest)
        with patch('pandas.read_csv', side_effect=AssertionError('no CSV')):
            reader = new.baskets().for_method('ward')
            study = reader.study()
            for key in ('monthly', 'metrics', 'predictive', 'hierarchy'):
                pd.testing.assert_frame_equal(study[key], self.studies['ward'][0][key])
            held = reader.details(1, ['Top 10 + downside to cash'], 'logistic')['assignments']
            self.assertTrue(held.weight.le(held.baseline_weight + 1e-12).all())
            self.assertTrue(held.loc[held.weight.gt(0), 'trailing_momentum'].gt(0).all())
            self.assertFalse(any('/assignments/' in path and '/kmeans/' in path for path in new._frames))
        with self.assertRaises(FileExistsError):
            export_grouping_comparison(self.old, self.studies['ward'][1], self.output)

    def test_three_subtabs_and_both_directional_choices_in_read_only_mode(self):
        with patch.dict(os.environ, {'MLFACTOR_APP_MODE':'deploy', 'MLFACTOR_PUBLISHED_ROOT':str(self.output)}), patch(
                'pandas.read_csv', side_effect=AssertionError('no CSVs')), patch(
                'research.store.ExperimentStore.__init__', side_effect=AssertionError('no store')), patch(
                'research.runner.ExperimentRunner.run', side_effect=AssertionError('no replay')), patch(
                'research.basket_momentum.linkage', side_effect=AssertionError('no clustering')), patch(
                'sklearn.cluster.KMeans.fit_predict', side_effect=AssertionError('no clustering')):
            app = AppTest.from_file(str(ROOT / 'deploy_app.py'), default_timeout=60).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.tabs[2].label, 'Grouping With ML')
            for label in ('K-means', 'Hierarchical clustering', 'Adding a direction'):
                self.assertTrue(any(tab.label == label for tab in app.tabs))
            app.selectbox(key='basket_direction_grouping').select('Hierarchical (Ward)').run()
            self.assertFalse(app.exception)
            app.checkbox(key='basket_overlay_inspect').check().run()
            self.assertFalse(app.exception)
            app.selectbox(key='ward_base_horizon').select('2-month momentum').run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Ward' in text.value for text in app.markdown))
            self.assertTrue(any('Training data only' in info.value for info in app.info))
