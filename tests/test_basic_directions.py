"""Threshold accounting, event definitions, local fits and saved-only UI."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research import ExperimentRunner, FeatureSpec, RunConfig
from research.adjustments import upside_probability
from research.data import model_data
from research.imports import import_csv_report
from research.legacy import FEATURES, KEYS, TARGET
from research.models import default_model_registry, event_labels, evaluate_predictions, predict_model
from research.store import ExperimentStore, config_from_dict
from test_research import toy_panel


class UpsideWeightTests(unittest.TestCase):
    def setUp(self):
        self.weights = pd.DataFrame({'date': [pd.Timestamp('2020-01-31')]*3,
                                    'stock_id': [1, 2, 3], 'weight': [.4, .4, 0]})
        self.pred = self.weights[KEYS].assign(probability_of_up=[.6, .59, .99])

    def test_threshold_keys_original_cash_and_excluded_stock(self):
        result = upside_probability(self.weights, self.pred.iloc[::-1], allow_cash=True)
        np.testing.assert_allclose(result.weight, [.48, .32, 0])
        self.assertAlmostEqual(result.weight.sum(), .8)
        # Cross the threshold in the other direction: it reverses the tilt.
        result = upside_probability(self.weights, self.pred.assign(probability_of_up=[.59, .6, 1]), allow_cash=True)
        np.testing.assert_allclose(result.weight, [.32, .48, 0])

    def test_fully_invested_all_cash_and_uniform_boost(self):
        fully = self.weights.assign(weight=[.5, .5, 0])
        np.testing.assert_allclose(upside_probability(fully, self.pred).weight, [.6, .4, 0])
        np.testing.assert_allclose(upside_probability(fully, self.pred.assign(probability_of_up=.9)).weight, fully.weight)
        np.testing.assert_allclose(upside_probability(fully, self.pred.assign(probability_of_up=.1)).weight, fully.weight)
        cash = self.weights.assign(weight=0.)
        self.assertTrue(upside_probability(cash, self.pred, allow_cash=True).weight.eq(0).all())
        pd.testing.assert_frame_equal(upside_probability(fully, self.pred, strength=0), fully)

    def test_invalid_probabilities_budgets_and_settings_rejected(self):
        fully = self.weights.assign(weight=[.5, .5, 0])
        for pred in [self.pred.iloc[:-1], pd.concat([self.pred, self.pred.iloc[:1]]),
                     self.pred.assign(probability_of_up=np.nan), self.pred.assign(probability_of_up=1.1)]:
            with self.assertRaises(ValueError):
                upside_probability(fully, pred)
        for options in [{'threshold': -1}, {'threshold': np.nan}, {'strength': 1}]:
            with self.assertRaises(ValueError):
                upside_probability(fully, self.pred, **options)
        with self.assertRaises(ValueError):
            upside_probability(self.weights, self.pred)


class DirectionPredictionTests(unittest.TestCase):
    def setUp(self):
        self.panel = toy_panel(72, 4)
        self.evaluation = self.panel.loc[self.panel.date.ge(self.panel.date.drop_duplicates().iloc[48])]
        self.recipe = FeatureSpec(tuple(FEATURES[:3]))
        self.model = default_model_registry()['logistic']

    def predict(self, panel=None, scope='per_stock', target='upside', policy='expanding'):
        history = self.panel if panel is None else panel
        evaluation = history.loc[history.date.isin(self.evaluation.date.unique())]
        return predict_model(model_data(history, evaluation), self.recipe, self.model,
            target=target, model_scope=scope, fit_policy=policy)

    def test_flat_returns_are_neither_positive_nor_negative(self):
        returns = pd.Series([-.1, 0, .1])
        self.assertEqual(event_labels(returns, 'loss').tolist(), [1, 0, 0])
        self.assertEqual(event_labels(returns, 'upside').tolist(), [0, 0, 1])
        panel = self.panel.assign(R1M_Usd=0.)
        pred, fits = self.predict(panel, scope='pooled')
        self.assertTrue(pred.probability_of_up.eq(0).all())
        scored = evaluate_predictions(pred, panel.loc[panel.date.isin(self.evaluation.date.unique())], fits, target='upside')
        self.assertTrue(scored.metrics.observed_up_rate.eq(0).all())
        self.assertEqual(scored.metrics.model.iloc[-1], 'Historical up-rate control')

    def test_local_fit_matches_independent_stock_only_scaler_and_logistic(self):
        pred, fits = self.predict()
        origin = self.evaluation.date.min()
        train = self.panel.loc[self.panel.date.lt(origin) & self.panel.stock_id.eq(0)]
        current = self.evaluation.loc[self.evaluation.date.eq(origin) & self.evaluation.stock_id.eq(0)]
        scaler = StandardScaler().fit(train[list(self.recipe.columns)])
        estimator = LogisticRegression(C=1., solver='lbfgs', max_iter=1000, random_state=0)
        estimator.fit(scaler.transform(train[list(self.recipe.columns)]), train[TARGET].gt(0).astype(int))
        expected = estimator.predict_proba(scaler.transform(current[list(self.recipe.columns)]))[0, 1]
        actual = pred.loc[pred.date.eq(origin) & pred.stock_id.eq(0)]
        self.assertAlmostEqual(actual.probability_of_up.iloc[0], expected)
        self.assertEqual(actual.prediction_source.iloc[0], 'per_stock')
        self.assertTrue(fits.trained_through_return_month.le(fits.fit_snapshot).all())

    def test_other_stock_outcomes_do_not_change_local_predictions(self):
        first, _ = self.predict()
        changed = self.panel.copy()
        changed.loc[changed.stock_id.ne(0), TARGET] = -.9
        second, _ = self.predict(changed)
        np.testing.assert_array_equal(first.loc[first.stock_id.eq(0), 'probability_of_up'],
                                      second.loc[second.stock_id.eq(0), 'probability_of_up'])

    def test_future_labels_cannot_change_first_origin_and_frozen_never_learns_them(self):
        for target in ['loss', 'upside']:
            for scope in ['pooled', 'per_stock']:
                first, _ = self.predict(scope=scope, target=target, policy='frozen_at_origin')
                changed = self.panel.copy()
                changed.loc[changed.date.ge(self.evaluation.date.min()), TARGET] = -.9
                second, _ = self.predict(changed, scope=scope, target=target, policy='frozen_at_origin')
                field = 'probability_of_loss' if target == 'loss' else 'probability_of_up'
                np.testing.assert_array_equal(first[field], second[field])

    def test_unseen_short_and_one_class_histories_have_explicit_pooled_fallback(self):
        origin = self.evaluation.date.min()
        changed = self.panel.loc[~(self.panel.stock_id.eq(3) & self.panel.date.lt(origin))].copy()
        changed = changed.loc[~(changed.stock_id.eq(2) & changed.date.lt(origin-pd.offsets.MonthEnd(12)))].copy()
        changed.loc[changed.stock_id.eq(1) & changed.date.lt(origin), TARGET] = .1
        pred, fits = self.predict(changed)
        pooled, _ = self.predict(changed, scope='pooled')
        first = pred.loc[pred.fit_snapshot.eq(origin)]
        expected = pooled.loc[pooled.fit_snapshot.eq(origin)].set_index(KEYS).probability_of_up
        fallback = first.loc[first.stock_id.ne(0)].set_index(KEYS)
        np.testing.assert_array_equal(fallback.probability_of_up, expected.loc[fallback.index])
        self.assertTrue(fallback.prediction_source.eq('pooled_fallback').all())
        reasons = first.groupby('stock_id').fallback_reason.first()
        self.assertEqual(reasons.loc[1], 'insufficient_class_history')
        self.assertEqual(reasons.loc[2], 'short_history')
        self.assertEqual(reasons.loc[3], 'short_history')
        # Same historical pooled control across training scopes.
        np.testing.assert_array_equal(pred.historical_up_rate, pooled.historical_up_rate)


class DirectionRunnerAndAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        toy_panel(84, 6).to_csv(cls.root/'mlfactor_long_only_train.csv', index=False)
        cls.store = ExperimentStore(cls.root/'artifacts', project_root=cls.root)
        cls.base = RunConfig(('equal_weight_universe', 'momentum_weighted', 'annual_consistency_momentum_cash'),
                             include_ml=True, models=('logistic',))
        cls.runner = ExperimentRunner(cls.root)
        cls.results = {}
        for target in ['loss', 'upside']:
            for scope in ['pooled', 'per_stock']:
                result = cls.runner.run(replace(cls.base, prediction_target=target, model_scope=scope))
                cls.store.save(result)
                cls.results[(target,scope)] = result

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_directions_scopes_have_exact_original_books_and_valid_adjustments(self):
        original = self.results[('loss','pooled')]
        for (target, scope), result in self.results.items():
            for strategy in self.base.strategies:
                key = result.case_id(strategy)
                pd.testing.assert_frame_equal(result.books[key], original.books[key])
                variant = 'loss_adjusted' if target == 'loss' else 'upside_adjusted'
                adjusted = result.weights[result.case_id(strategy, variant)]
                base = result.weights[key]
                if target == 'upside':
                    np.testing.assert_allclose(adjusted.groupby('date').weight.sum(), base.groupby('date').weight.sum())
                self.assertTrue((adjusted.weight.gt(0).to_numpy() == base.weight.gt(0).to_numpy()).all())
            self.assertEqual(len(result.paired), 3)
            self.assertEqual(result.manifest['model']['scope'], scope)
            loaded = self.store.load(result.cache_info['run_id'])
            self.assertEqual(loaded.config, result.config)
            pd.testing.assert_frame_equal(loaded.predictions.predictions, result.predictions.predictions)
            summary = self.store.summary(result.cache_info['run_id'])
            self.assertIn('observed_up_rate' if target == 'upside' else 'observed_loss_rate', summary['model_metrics'])

    def test_new_settings_are_separate_cache_entries_and_old_config_has_defaults(self):
        keys = {self.store.fingerprint(r.config) for r in self.results.values()}
        self.assertEqual(len(keys), 4)
        self.assertIsNone(self.store.find(replace(self.base,prediction_target='upside',upside_threshold=.7)))
        old = config_from_dict({'strategies':['equal_weight_universe']})
        self.assertEqual((old.prediction_target, old.model_scope), ('loss','pooled'))
        for kwargs in [{'prediction_target':'bad'}, {'model_scope':'bad'}, {'min_stock_history':1}, {'min_stock_class':0}, {'upside_threshold':1.1}]:
            with self.assertRaises(ValueError):
                replace(self.base,**kwargs)

    def test_upside_csv_import_replays_without_fitting(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.results[('upside','per_stock')]
            report = result.save(Path(folder)/'report')
            with patch('research.models.predict_model',side_effect=AssertionError('must not fit')):
                run_id = import_csv_report(self.store,report)
            loaded = self.store.load(run_id)
            for key in result.books:
                np.testing.assert_allclose(loaded.books[key].simple_return,result.books[key].simple_return)

    def test_saved_app_switches_patterns_and_scopes_without_computation(self):
        with patch.dict('os.environ',{'MLFACTOR_ARTIFACT_ROOT':str(self.store.root)}), patch(
                'research.store.ExperimentRunner.run',side_effect=AssertionError('browsing cannot fit')):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=90)
            app.session_state['training_result'] = self.results[('loss','pooled')]
            app.run()
            self.assertFalse(app.exception)
            source_id = app.session_state['training_result'].cache_info['run_id']
            app.selectbox(key='basic_pattern').select('Upside threshold boost').run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Observed up rate' in f.value for f in app.tabs[1].dataframe))
            app.selectbox(key='basic_scope').select('Separate model per stock').run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Pooled fallback predictions' in f.value for f in app.tabs[1].dataframe))
            self.assertTrue(any('Scope' in f.value for f in app.tabs[1].dataframe))
            app.checkbox(key='basic_show_allocations').check().run()
            self.assertFalse(app.exception)
            # Inspector defaults to the original; select the adjusted case.
            self.assertFalse(any('probability_of_up' in f.value for f in app.tabs[0].dataframe))
            self.assertEqual(app.session_state['training_result'].cache_info['run_id'],source_id)
            self.assertFalse(app.tabs[2].dataframe)


if __name__ == '__main__':
    unittest.main()
