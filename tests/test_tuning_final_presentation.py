"""Saved-result comparisons keep original weights, dates and growth conventions."""
import unittest

import numpy as np
import pandas as pd

from final_book_ui import _figure, _overlay_changes, _table
from research.basket_momentum import TOP_VOL_VERSION
from research.basket_adjustments import DOWNSIDE_VERSION
from tuning_ui import _wide_metric


class SavedPresentationTests(unittest.TestCase):
    def test_final_log_views_use_portfolio_returns_and_earned_months(self):
        earned = pd.to_datetime(['2010-01-31','2010-02-28'])
        rows = pd.DataFrame({'return_month':earned, 'snapshot_date':earned-pd.offsets.MonthEnd(1),
            'version':'Equal weight', 'momentum_months':0, 'model_key':'none',
            'simple_return':[.1,-.1], 'wealth':[1.1,.99]})
        cumulative = _figure(rows, 'logistic', 'Downside to cash', 'entire', earned[0], 'Cumulative log return')
        np.testing.assert_allclose(cumulative.data[0].y, [0,np.log(1.1),np.log(.99)])
        self.assertEqual(pd.Timestamp(cumulative.data[0].x[0]), pd.Timestamp('2009-12-31'))
        self.assertEqual(pd.Timestamp(cumulative.layout.shapes[0].x0), earned[0])
        monthly = _figure(rows, 'logistic', 'Downside to cash', 'test', earned[0], 'Monthly log return')
        self.assertEqual(len(monthly.data[0].x),2)
        self.assertEqual(pd.Timestamp(monthly.data[0].x[0]), earned[0])
        np.testing.assert_allclose(monthly.data[0].y,np.log1p([.1,-.1]))

    def test_portfolio_deltas_pair_each_overlay_with_its_own_horizon_and_original(self):
        metrics = pd.DataFrame([
            dict(version=TOP_VOL_VERSION, model_key='none', momentum_months=1, cagr=.10, sharpe=.5, max_drawdown=-.2,avg_cash_weight=.3),
            dict(version=TOP_VOL_VERSION, model_key='none', momentum_months=2, cagr=.50, sharpe=1., max_drawdown=-.4,avg_cash_weight=.1),
            dict(version=DOWNSIDE_VERSION, model_key='logistic', momentum_months=1, cagr=.08, sharpe=.6, max_drawdown=-.15,avg_cash_weight=.4),
            dict(version=DOWNSIDE_VERSION, model_key='naive_bayes', momentum_months=1, cagr=.9, sharpe=3., max_drawdown=-.1,avg_cash_weight=.5),
        ])
        result=_overlay_changes(metrics,'logistic')
        self.assertEqual(len(result),1)
        np.testing.assert_allclose(result[['CAGR change (pp)','Sharpe change','Max drawdown change (pp)','Cash change (pp)']],
                                   [[-2.,.1,5.,10.]])

    def test_final_table_adds_log_growth_to_old_saved_metrics_without_mutation(self):
        metrics=pd.DataFrame([dict(version='Equal weight',model_key='none',momentum_months=0,cagr=.1,
            annualized_volatility=.2,sharpe=.5,max_drawdown=-.2,avg_cash_weight=0.,avg_names=10)])
        displayed=_table(metrics,'logistic',False)
        self.assertAlmostEqual(displayed['Annual log growth'].iloc[0],np.log1p(.1))
        self.assertNotIn('annualized_log_growth',metrics)

    def test_focused_final_names_kmeans_top_ten_and_actual_momentum_window(self):
        common=dict(model_key='none',annualized_volatility=.2,sharpe=.5,max_drawdown=-.2,avg_cash_weight=.5,avg_names=25)
        metrics=pd.DataFrame([
            dict(common,version='Equal weight',momentum_months=0,cagr=.13),
            dict(common,version=TOP_VOL_VERSION,momentum_months=1,cagr=.196),
            dict(common,version=TOP_VOL_VERSION,momentum_months=2,cagr=.15),
        ])
        focused=_table(metrics,'logistic',False)
        top=focused.loc[focused.Book.str.contains('K-means top 10')]
        self.assertEqual(len(top),1)
        self.assertIn('1M momentum + cash',top.Book.iloc[0])
        self.assertEqual(top.CAGR.iloc[0],.196)
        expanded=_table(metrics,'logistic',True)
        self.assertTrue(expanded.Book.str.contains('2M momentum + cash',regex=False).any())

    def test_all_strategy_comparison_retains_exact_saved_metrics(self):
        metrics=pd.DataFrame([dict(label=label,version=version,model_key=model,cagr=value)
            for label,value in [('Equal weight',.1),('Momentum',.2)]
            for version,model in [('No ML','none'),('ML, not tuned','logistic'),('Tuned ML','logistic')]])
        result=_wide_metric(metrics,'logistic','cagr')
        self.assertEqual(result.index.tolist(),['Equal weight','Momentum'])
        np.testing.assert_allclose(result.loc['Momentum'],[.2,.2,.2])
