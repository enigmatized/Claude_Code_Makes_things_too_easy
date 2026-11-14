"""Causal saved grouping illustrations and self-contained Advanced ML views."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from research.basket_reporting import grouping_profiles, selected_vol_ranks, require_basket_training
from research.basket_momentum import FEATURE, create_basket_study, save_basket_study, TOP_VOL_VERSION
from research.basket_adjustments import create_adjusted_basket_study, DOWNSIDE_VERSION, BASKET_DOWNSIDE_VERSION
from research import ExperimentRunner, RunConfig
from research.store import ExperimentStore
from published_results import PublishedStrategies, export_strategies
from test_basket_momentum import panel

ROOT = Path(__file__).resolve().parents[1]
FULL = "Basket momentum + cash"


class GroupIllustrationTests(unittest.TestCase):
    def signals(self):
        dates = pd.date_range("2004-12-31", periods=24, freq="ME")
        rows = []
        for i, date in enumerate(dates):
            for stock in range(6):
                rows.append({"date": date, "stock_id": stock, FEATURE: stock % 2,
                    "trailing_momentum": .1 * np.sin(i + stock), "version": FULL,
                    "momentum_months": 1, "model_key": "none"})
        return pd.DataFrame(rows), dates

    def test_profiles_use_current_membership_and_only_then_known_saved_returns(self):
        frame, dates = self.signals()
        decision = dates[18]
        frame.loc[frame.date.eq(decision), FEATURE] = frame.loc[frame.date.eq(decision), "stock_id"].floordiv(3)
        result, coverage = grouping_profiles(frame, decision)
        self.assertEqual(coverage['illustrated'], 6)
        self.assertEqual(result.input_month.max(), decision)
        self.assertEqual(result.input_month.min(), dates[7])
        past = frame.loc[frame.date.between(dates[7], decision)].pivot(index='date', columns='stock_id', values='trailing_momentum')
        z = (past - past.mean()) / past.std(ddof=0)
        for group, stocks in [(0, [0, 1, 2]), (1, [3, 4, 5])]:
            np.testing.assert_allclose(result.loc[result[FEATURE].eq(group), 'mean_standardized_return'], z[stocks].mean(axis=1))
        changed = frame.copy()
        changed.loc[changed.date.gt(decision), 'trailing_momentum'] = 999
        changed.loc[changed.date.lt(decision), FEATURE] = 99
        again, _ = grouping_profiles(changed, decision)
        pd.testing.assert_frame_equal(result, again)
        empty, early = grouping_profiles(frame, dates[3])
        self.assertTrue(empty.empty)
        self.assertEqual(early['illustrated'], 0)

    def test_missing_profile_coverage_is_explicit_and_selected_rank_ties_are_stable(self):
        frame, dates = self.signals()
        frame.loc[frame.date.eq(dates[8]) & frame.stock_id.eq(0), 'trailing_momentum'] = np.nan
        _, coverage = grouping_profiles(frame, dates[18])
        self.assertEqual(coverage, {'members': 6, 'illustrated': 5,
            'first_input_month': dates[7], 'last_input_month': dates[18]})
        rows = frame.loc[frame.date.eq(dates[18])].assign(positive_volatility=.2)
        ranks = selected_vol_ranks(rows, dates[18])
        self.assertEqual(ranks.loc[ranks[FEATURE].eq(0), 'stock_id'].tolist(), [0, 2, 4])
        self.assertEqual(ranks.loc[ranks[FEATURE].eq(0), 'Positive-vol rank in group'].tolist(), [1, 2, 3])


class PublishedBasketPresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        panel(months=42, stocks=8).to_csv(cls.root / 'mlfactor_long_only_train.csv', index=False)
        cls.store = ExperimentStore(project_root=cls.root)
        base = create_basket_study(cls.root, n_clusters=2, validation_history_months=24)
        base_dir = save_basket_study(base, cls.store.root / 'basket_studies')
        runs = {}
        for target in ('loss', 'upside'):
            result = ExperimentRunner(cls.root).run(RunConfig(('equal_weight_universe',), include_ml=True,
                prediction_target=target, validation_history_months=24))
            runs[target] = cls.store.save(result)
        cls.study = create_adjusted_basket_study(cls.store, base_dir, runs['loss'], runs['upside'])
        cls.study_dir = save_basket_study(cls.study, cls.store.root / 'basket_studies')
        cls.output = export_strategies(cls.store, runs['loss'], cls.root / 'published', basket_study=cls.study_dir)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_partition_reads_preserve_exact_results_and_are_lazy(self):
        publication = PublishedStrategies(self.output)
        baskets = publication.baskets()
        with patch('pandas.read_csv', side_effect=AssertionError('no raw CSVs')), patch(
                'research.store.ExperimentStore.__init__', side_effect=AssertionError('no catalog')):
            saved = baskets.study()
            for name in ('metrics', 'monthly', 'predictive'):
                pd.testing.assert_frame_equal(saved[name], self.study[name])
            self.assertFalse(any('/assignments/' in path for path in publication._frames))
            detail = baskets.details(1, [FULL], 'none')
            for name in ('baskets', 'assignments'):
                expected = self.study[name].loc[lambda f: f.momentum_months.eq(1) & f.version.eq(FULL) & f.model_key.eq('none')]
                pd.testing.assert_frame_equal(detail[name], expected.reset_index(drop=True))
            self.assertEqual(sum('/assignments/' in path for path in publication._frames), 1)

    def test_two_subtabs_and_allocation_choices_work_without_fitting_or_sources(self):
        with patch.dict('os.environ', {'MLFACTOR_APP_MODE':'deploy', 'MLFACTOR_PUBLISHED_ROOT':str(self.output)}), patch(
                'pandas.read_csv', side_effect=AssertionError('no CSVs')), patch(
                'research.store.ExperimentStore.__init__', side_effect=AssertionError('no research store')), patch(
                'sklearn.cluster.KMeans.fit_predict', side_effect=AssertionError('no refitting')), patch(
                'research.runner.ExperimentRunner.run', side_effect=AssertionError('no replay')):
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
            self.assertFalse(app.exception)
            advanced = app.tabs[2]
            subtabs = [tab for tab in advanced.tabs if tab.label != 'Grouping With ML']
            self.assertEqual([tab.label for tab in subtabs], ['K-means', 'Hierarchical clustering', 'Adding a direction', 'Momentum acceleration'])
            build, hierarchy, overlay, acceleration = subtabs
            self.assertFalse(any('Brier score' in f.value or 'Predicted loss probability' in f.value for f in build.dataframe))
            self.assertFalse(any('Prediction model' in w.label for w in build.selectbox))
            self.assertTrue(any('non-constant' in text.value for text in build.markdown))
            app.selectbox(key='basket_base_horizon').select('3-month momentum').run()
            self.assertFalse(app.exception)
            build = next(tab for tab in app.tabs if tab.label == 'K-means')
            self.assertTrue(any('No top-10 result' in x.value for x in build.info))
            app.checkbox(key='basket_overlay_inspect').check().run()
            app.selectbox(key='basket_screen_1').select(DOWNSIDE_VERSION).run()
            self.assertFalse(app.exception)
            overlay = next(tab for tab in app.tabs if tab.label == 'Adding a direction')
            self.assertTrue(any('Predicted loss probability' in f.value for f in overlay.dataframe))
            app.selectbox(key='basket_selection_basket').select('K-means full groups').run()
            app.selectbox(key='basket_screen_1').select(BASKET_DOWNSIDE_VERSION).run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Training data only' in info.value for info in app.tabs[2].info))

    def test_test_period_and_mistimed_studies_are_rejected(self):
        for mutation in ('test', 'future', 'timing'):
            study = {**self.study, 'metadata': dict(self.study['metadata']), 'monthly': self.study['monthly'].copy()}
            if mutation == 'test': study['metadata']['test_csv_loaded'] = True
            if mutation == 'future': study['monthly']['return_month'] = pd.Timestamp('2010-01-31')
            if mutation == 'timing': study['monthly']['snapshot_date'] = study['monthly']['return_month']
            with self.assertRaises(ValueError): require_basket_training(study)


if __name__ == '__main__':
    unittest.main()
