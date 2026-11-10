"""Feature timing, fold isolation, reconstruction and full train/final/publication flow."""
from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from research.legacy import FEATURES, KEYS, TARGET
from research.engineering import (FeatureMap, EngineeredClassifier, INPUT_COLUMNS, RECIPES, TECH, technical_inputs)
from research.engineering_study import create_engineering_study, run_engineering_stage, load_engineering_study, _inputs
from engineering_publication import check_engineering
from test_research import toy_panel


class FeatureEngineeringTests(unittest.TestCase):
    def test_technical_inputs_ignore_current_and_future_labels_and_respect_calendar_gaps(self):
        panel=toy_panel(42,8);dates=pd.DatetimeIndex(panel.date.unique());origin=dates[30]
        before=technical_inputs(panel)
        altered=panel.copy();altered.loc[altered.date.ge(origin),TARGET]=.9
        after=technical_inputs(altered)
        pd.testing.assert_frame_equal(before.loc[before.date.le(origin)].reset_index(drop=True),
                                      after.loc[after.date.le(origin)].reset_index(drop=True))
        stock=panel.stock_id.iloc[0]
        wanted=panel.loc[panel.stock_id.eq(stock)&panel.date.eq(dates[29]),TARGET].iloc[0]
        self.assertAlmostEqual(before.loc[before.stock_id.eq(stock)&before.date.eq(origin),'ta_ret_1m'].iloc[0],wanted)
        gap=panel.drop(panel.loc[panel.stock_id.eq(stock)&panel.date.eq(dates[25])].index)
        result=technical_inputs(gap)
        row=result.loc[result.stock_id.eq(stock)&result.date.eq(origin)].iloc[0]
        self.assertTrue(pd.isna(row.ta_vol_12m))
        self.assertEqual(row.ta_vol_12m_rk,.5)

    def test_all_recipes_keep_at_most_ten_inputs_and_save_reconstructable_pca(self):
        panel=toy_panel(60,10);X=_inputs(panel)[INPUT_COLUMNS]
        known=X.date.le(X.date.drop_duplicates().iloc[35])
        for recipe in RECIPES:
            fitted=FeatureMap(recipe).fit(X.loc[known],panel.loc[known,TARGET])
            actual=fitted.transform(X.loc[~known])
            self.assertLessEqual(actual.shape[1],10);self.assertTrue(np.isfinite(actual).all().all())
            self.assertEqual(actual.index.tolist(),X.loc[~known].index.tolist())
            if recipe=='cluster_pca':
                for group,record in fitted.audit_['pca'].items():
                    block=X.loc[~known,record['members']]
                    z=(block.to_numpy()-record['mean'])/record['scale']
                    projected=(z-np.asarray(record['pca_mean']))@np.asarray(record['components']).T
                    ranked=pd.Series(projected[:,0],index=block.index).groupby(X.loc[~known].date).rank(pct=True)
                    np.testing.assert_allclose(actual[f'group_{int(group)+1}_pc1'],ranked)
        with self.assertRaises(ValueError):FeatureMap().fit(X.assign(**{TARGET:panel[TARGET]}),panel[TARGET])

    def test_cv_learns_every_composite_from_its_own_training_fold(self):
        from sklearn.model_selection import GridSearchCV
        from research.engineering import _FEATURE_CACHE
        from research.engineering_study import _score
        from research.tuning import MonthForwardSplit
        from research.legacy import return_month
        panel=toy_panel(60,8);X=_inputs(panel)[INPUT_COLUMNS]
        splitter=MonthForwardSplit(X.date,return_month(X.date),n_splits=2,min_train_months=24)
        expected={tuple(X.iloc[indices].index) for indices,_ in splitter.split(X)}
        observed=[];original=FeatureMap.fit
        def fit(instance,frame,y):
            observed.append(tuple(frame.index))
            return original(instance,frame,y)
        _FEATURE_CACHE.clear()
        with patch.object(FeatureMap,'fit',fit):
            GridSearchCV(EngineeredClassifier('ic_blends','naive_bayes'),{'correlation_cap':[.4,.6]},
                         cv=splitter,scoring=_score,error_score='raise').fit(X,panel[TARGET])
        self.assertTrue(expected.issubset(set(observed)))
        self.assertTrue(all(indices in expected or indices==tuple(X.index) for indices in observed))

    def test_estimator_freezes_learned_features_and_roundtrips_without_refitting(self):
        panel=toy_panel(60,10);X=_inputs(panel)[INPUT_COLUMNS];dates=pd.DatetimeIndex(panel.date.unique())
        train=X.date.lt(dates[36]);test=~train
        model=EngineeredClassifier('ic_blends','logistic','upside').fit(X.loc[train],panel.loc[train,TARGET])
        audit=json.dumps(model.feature_map_.audit_,sort_keys=True)
        prob=model.predict_proba(X.loc[test])
        self.assertEqual(audit,json.dumps(model.feature_map_.audit_,sort_keys=True))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.joblib';joblib.dump(model,path)
            with patch.dict(os.environ,{'MLFACTOR_READ_ONLY':'1'}):
                np.testing.assert_array_equal(joblib.load(path).predict_proba(X.loc[test]),prob)
                with self.assertRaises(RuntimeError):EngineeredClassifier().fit(X.loc[train],panel.loc[train,TARGET])


class EngineeringFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        panel=toy_panel(144,8);panel['date']=(pd.DatetimeIndex(panel.date).to_period('M')-12).to_timestamp('M');dates=pd.DatetimeIndex(panel.date.unique())
        train=panel.loc[panel.date.isin(dates[:120])];test=panel.loc[panel.date.isin(dates[120:])]
        train.to_csv(cls.root/'mlfactor_long_only_train.csv',index=False);test.to_csv(cls.root/'mlfactor_long_only_test.csv',index=False)
        cls.directory=create_engineering_study(cls.root,recipes=['anova_10','cluster_pca','ic_blends'],models=['logistic'],origin=dates[84],progress=lambda _:None)
        run_engineering_stage(cls.root,cls.directory,'test',progress=lambda _:None)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_train_and_final_scope_rows_inputs_and_chosen_parameters(self):
        training=load_engineering_study(self.directory);final=load_engineering_study(self.directory,'test')
        for window,study in [('training',training),('test',final)]:
            check_engineering(study,window)
            self.assertTrue(study['audits'].n_features.le(10).all())
            self.assertEqual(study['metrics'].book.nunique(),6)
            for audit in study['audits'].audit_json:
                data=json.loads(audit)
                self.assertLess(pd.Timestamp(data['fit_last_snapshot']),pd.Timestamp(data['fit_origin']))
        self.assertEqual(final['scope']['fit_rows'],120*8)
        self.assertFalse(training['scope']['test_csv_loaded'])
        invalid={**training,'scope':{**training['scope'],'fit_origin':'2009-01-31'}}
        with self.assertRaises(ValueError):check_engineering(invalid,'training')
        invalid={**training,'scope':{**training['scope'],'test_csv_loaded':True}}
        with self.assertRaises(ValueError):check_engineering(invalid,'training')
        before=(self.directory/'test'/'complete.json').read_bytes()
        with patch('research.engineering_study.load_panel',side_effect=AssertionError('No recompute')):
            run_engineering_stage(self.root,self.directory,'test',progress=lambda _:None)
        self.assertEqual((self.directory/'test'/'complete.json').read_bytes(),before)

    def test_publication_keeps_old_frames_and_never_needs_the_archive(self):
        from research import ExperimentRunner, RunConfig
        from research.store import ExperimentStore
        from published_results import export_strategies, PublishedStrategies
        from engineering_publication import export_engineering
        root=self.root/'publication_fixture';root.mkdir(exist_ok=True)
        import shutil
        shutil.copy2(self.root/'mlfactor_long_only_train.csv',root/'mlfactor_long_only_train.csv')
        store=ExperimentStore(project_root=root)
        result=ExperimentRunner(root).run(RunConfig(('equal_weight_universe',),validation_history_months=84))
        run=store.save(result)
        old=export_strategies(store,run,root/'old')
        output=export_engineering(old,self.directory,root/'new')
        pub=PublishedStrategies(output)
        for name,digest in PublishedStrategies(old).metadata['file_sha256'].items():
            self.assertEqual(pub.metadata['file_sha256'][name],digest)
        with patch('pandas.read_csv',side_effect=AssertionError('No source CSV')),patch(
                'research.store.ExperimentStore.__init__',side_effect=AssertionError('No archive')),patch(
                'joblib.load',side_effect=AssertionError('No estimator object')):
            reader=pub.analysis_section('engineering')
            pd.testing.assert_frame_equal(reader.study()['metrics'],load_engineering_study(self.directory)['metrics'])
            script=("from published_results import PublishedStrategies\nfrom engineering_ui import render_engineering\n"
                    f"p=PublishedStrategies({str(output)!r}).analysis_section('engineering')\n"
                    "render_engineering(published=p)\nrender_engineering(published=p,window='test')")
            app=AppTest.from_string(script,default_timeout=60).run()
            self.assertFalse(app.exception)
            self.assertTrue(any('frozen' in x.value.lower() for x in app.info))
        with self.assertRaises(FileExistsError):export_engineering(old,self.directory,output)

    def test_local_saved_app_reads_both_windows_without_any_fitting(self):
        script=("from engineering_ui import render_engineering\nfrom types import SimpleNamespace\n"
                f"store=SimpleNamespace(root={str(self.root/'artifacts')!r})\n"
                "render_engineering(store)\nrender_engineering(store,window='test')")
        with patch('pandas.read_csv',side_effect=AssertionError('No CSV')),patch(
                'research.engineering.EngineeredClassifier.fit',side_effect=AssertionError('No fit')):
            app=AppTest.from_string(script,default_timeout=60).run()
            self.assertFalse(app.exception)
            app.selectbox(key='engineering_method').select('Cluster PCA + technical').run()
            self.assertFalse(app.exception)
            app.radio(key='engineering_final_settings').set_value('Both').run()
            app.selectbox(key='engineering_final_return').select('Cumulative log return').run()
            self.assertFalse(app.exception)
