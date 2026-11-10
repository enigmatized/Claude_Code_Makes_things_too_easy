"""Explicit train/tune/freeze runner for the feature-engineering extension.

Training (including every learned composite) is chronological. Final evaluation
is a separate explicit phase; nothing fits while the application reads results.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits

from .engineering import RECIPES, INPUT_COLUMNS, TECH, EngineeredClassifier, technical_inputs
from .legacy import FEATURES, KEYS, TARGET, load_panel, return_month
from .models import default_model_registry, event_labels
from .tuning import MonthForwardSplit
from .basket_momentum import (FEATURE, assign_correlation_baskets, attach_short_momentum,
    positive_volatility, select_top_positive_vol, concentrated_basket_weights, _prepare_signals,
    positive_momentum_cash_weights, _metrics, _monthly, _digest)
from .accounting import portfolio_with_cash
from .adjustments import loss_probability, upside_probability

ORIGIN = pd.Timestamp('2006-12-31')
BOOKS = ('Equal weight','Momentum + cash','K-means full groups','K-means top 10','Ward full groups','Ward top 10')
MODEL_GRIDS = {
    'logistic':[{'C':1.0},{'C':.1}],
    'naive_bayes':[{'var_smoothing':1e-9},{'var_smoothing':1e-5}],
    'random_forest':[{'max_depth':6,'min_samples_leaf':50},{'max_depth':4,'min_samples_leaf':200}],
    'hist_gradient_boosting':[{'learning_rate':.05},{'learning_rate':.1}],
}
FRAMES = ('metrics','monthly','predictive','audits','trials','folds')


def _guard():
    if os.environ.get('MLFACTOR_READ_ONLY')=='1':raise RuntimeError('Research computation is disabled in view mode.')


def _write_json(path,value):
    path.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n')


def _inputs(history):
    technical=technical_inputs(history)
    return history[KEYS+FEATURES].merge(
        technical[KEYS+list(TECH)+['technical_missing_count']],on=KEYS,validate='one_to_one')


def _baselines(history,panel,progress):
    dates=pd.DatetimeIndex(panel.date.unique()).sort_values()
    weights={'Equal weight':panel[KEYS].assign(weight=1/panel.groupby('date').stock_id.transform('size'))}
    volatility=positive_volatility(history,dates)
    for method,label in [('kmeans','K-means'),('ward','Ward')]:
        progress(f'Building {label} baseline weights on {len(dates)} decision months')
        assigned=assign_correlation_baskets(history,dates,grouping_method=method)
        groups=panel[KEYS].merge(assigned[KEYS+[FEATURE]],on=KEYS,validate='one_to_one')
        plain,grouped=_prepare_signals(attach_short_momentum(history,groups,1))
        weights[label+' full groups']=positive_momentum_cash_weights(grouped,by_basket=True)
        selected=select_top_positive_vol(groups,volatility)
        full,_,_=concentrated_basket_weights(history,panel,selected,1)
        weights[label+' top 10']=full[KEYS+['weight']]
        if method=='kmeans':weights['Momentum + cash']=positive_momentum_cash_weights(plain,by_basket=False)
    return weights


def _score(estimator,X,y):
    event=event_labels(pd.Series(np.asarray(y)),estimator.prediction_target)
    return -log_loss(event,estimator.predict_proba(X)[:,1],labels=[0,1])


def _predictive(y,p,model_key,target,status,recipe,window,control):
    event=event_labels(pd.Series(np.asarray(y)),target)
    rows=[]
    for label,prob in [('Classifier',p),('Historical event rate',np.repeat(control,len(event)))]:
        rows.append(dict(recipe=recipe,model_key=model_key,target=target,status=status,window=window,predictor=label,
            rows=len(event),brier_score=brier_score_loss(event,prob),log_loss=log_loss(event,prob,labels=[0,1]),
            roc_auc=roc_auc_score(event,prob) if event.nunique()>1 else .5))
    return pd.DataFrame(rows)


def _portfolio_frames(panel,weights,predictions=None,*,recipe='original',model_key='none',target='none',status='Original',window):
    metrics,monthly=[],[]
    for name,allocation in weights.items():
        if predictions is None:adjusted=allocation
        elif target=='loss':adjusted=loss_probability(allocation,predictions,.5,allow_cash=name!='Equal weight')
        else:adjusted=upside_probability(allocation,predictions,.5,.6,allow_cash=name!='Equal weight')
        book=portfolio_with_cash(panel,adjusted)
        identity=dict(recipe=recipe,model_key=model_key,target=target,status=status,window=window,book=name)
        metrics.append(pd.DataFrame([_metrics(book,name,1)]).assign(**identity))
        monthly.append(_monthly(book,name,1).assign(**identity))
    return pd.concat(metrics,ignore_index=True),pd.concat(monthly,ignore_index=True)


def _audit(estimator,identity,origin):
    feature=estimator.feature_map_
    result={**feature.audit_,'output_columns':feature.columns_, 'model_parameters':estimator.model_.get_params(),
            'classifier_scaler':None if estimator.scaler_ is None else {
                'mean':estimator.scaler_.mean_.tolist(),'scale':estimator.scaler_.scale_.tolist()},
            'groups_count':estimator.n_groups,'correlation_cap':estimator.correlation_cap,
            'fit_origin':str(pd.Timestamp(origin).date()),'frozen':True}
    if hasattr(feature,'tree_'):
        result['linkage']=feature.tree_.tolist();result['correlations']=feature.corr_.to_numpy().tolist()
        result['source_columns']=feature.corr_.columns.tolist()
    return {**identity,'n_features':len(feature.columns_),'audit_json':json.dumps(result,allow_nan=False)}


def _case_id(recipe,model,target):return f'{recipe}__{model}__{target}'


def _aggregate(directory,window):
    stage=directory/window
    frames={name:[] for name in FRAMES}
    for path in sorted((stage/'cases').glob('*/case.json')):
        record=json.loads(path.read_text())
        for name,digest in record['frames'].items():
            frame=path.parent/(name+'.parquet')
            if _digest(frame)!=digest:raise ValueError(f'Damaged case frame: {frame}')
            frames[name].append(pd.read_parquet(frame))
    for name in ('metrics','monthly'):
        baseline=stage/('original_'+name+'.parquet')
        frames[name].insert(0,pd.read_parquet(baseline))
    checksums={}
    for name,parts in frames.items():
        frame=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
        out=stage/(name+'.parquet');frame.to_parquet(out,index=False)
        checksums[name]=_digest(out)
    _write_json(stage/'complete.json',{'window':window,'frames':checksums,'completed_utc':datetime.now(timezone.utc).isoformat()})


def create_engineering_study(root,recipes=None,models=None,targets=('loss','upside'),origin=ORIGIN,progress=print):
    """Write a declared protocol before any search; return a resumable study path."""
    _guard();root=Path(root)
    recipes=list(RECIPES if recipes is None else recipes);models=list(default_model_registry() if models is None else models)
    if not recipes or not set(recipes).issubset(RECIPES) or not models or not set(models).issubset(MODEL_GRIDS):raise ValueError('Unknown recipe/model.')
    if not set(targets).issubset({'loss','upside'}) or not targets:raise ValueError('Unknown direction.')
    directory=root/'artifacts'/'engineering_studies'/('engineering_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    _write_json(directory/'study.json',{'protocol':'point_in_time_feature_engineering','created_utc':datetime.now(timezone.utc).isoformat(),
        'origin':str(pd.Timestamp(origin).date()),'recipes':recipes,'models':models,'targets':list(targets),'books':list(BOOKS),
        'model_grids':MODEL_GRIDS,'group_grid':[4,6],'ic_correlation_grid':[.4,.6],'default_groups':6,
        'cv_splits':3,'cv_min_train_months':24,'scoring':'negative binary event log loss (sklearn.metrics.log_loss)',
        'feature_limit':10,'source_factor_count':20,'technical_missing_policy':'neutral current-month rank 0.5; same rows for every recipe',
        'final_policy':'Refit fixed and chosen training-CV parameters once at the test origin; no test-label fits or retuning.',
        'source_train_sha256':_digest(root/'mlfactor_long_only_train.csv'),
        'code_sha256':{name:_digest(Path(__file__).with_name(name)) for name in ['engineering.py','engineering_study.py']},
        'runtime':{'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__},
        'test_previously_viewed':True,'formal_feature_limit_note':'Ten model inputs may aggregate twenty source factors; instructor interpretation must be checked.'})
    progress(f'Declared study {directory}')
    run_engineering_stage(root,directory,'training',progress)
    return directory


def run_engineering_stage(root,directory,window='training',progress=print):
    """Execute the training or separately authorized final phase; completed cases resume."""
    _guard();root,directory=Path(root),Path(directory)
    meta=json.loads((directory/'study.json').read_text())
    if _digest(root/'mlfactor_long_only_train.csv')!=meta['source_train_sha256']:raise ValueError('Training data changed.')
    for name,digest in meta['code_sha256'].items():
        if _digest(Path(__file__).with_name(name))!=digest:raise ValueError('Computation code changed; create a new study.')
    if window not in {'training','test'}:raise ValueError('Choose training or test.')
    stage=directory/window
    if (stage/'complete.json').exists():
        progress(f'Completed {window} results already saved; no recomputation.');return stage
    if window=='test' and not (directory/'training'/'complete.json').exists():raise ValueError('Complete training before final evaluation.')
    train=load_panel('train',root)
    if window=='test':
        test=load_panel('test',root)
        if train.date.max()>=test.date.min():raise ValueError('Train/test snapshots overlap.')
        history=pd.concat([train,test],ignore_index=True).sort_values(KEYS).reset_index(drop=True)
        panel=test[KEYS+[TARGET]];origin=pd.Timestamp(test.date.min())
    else:
        history=train;origin=pd.Timestamp(meta['origin']);panel=train.loc[train.date.ge(origin),KEYS+[TARGET]]
        if return_month(panel.date).max()>pd.Timestamp('2009-12-31'):raise ValueError('Training stage reached the test period.')
    X=_inputs(history)
    known=return_month(history.date)<=origin
    fit_X=X.loc[known,INPUT_COLUMNS].reset_index(drop=True)
    fit_y=history.loc[known,TARGET].reset_index(drop=True)
    if window=='test' and int(known.sum())!=len(train):raise AssertionError('Final classifier training must use every training row and no test row.')
    current=panel[KEYS].merge(X[INPUT_COLUMNS],on=KEYS,validate='one_to_one')
    stage.mkdir(exist_ok=True)
    _write_json(stage/'scope.json',{'window':window,'fit_origin':str(origin.date()),'trained_through_return_month':str(return_month(history.loc[known,'date']).max().date()),
        'fit_rows':len(fit_y),'first_return_month':str(return_month(panel.date).min().date()),'last_return_month':str(return_month(panel.date).max().date()),
        'months':int(panel.date.nunique()),'test_csv_loaded':window=='test',
        'technical_missing_rows':int(X.loc[known,'technical_missing_count'].gt(0).sum()),
        'test_source_sha256':_digest(root/'mlfactor_long_only_test.csv') if window=='test' else None})
    progress(f'{window}: fit {len(fit_y):,} rows known at {origin.date()}, predict {len(current):,} rows')
    weights=_baselines(history,panel,progress)
    metrics,monthly=_portfolio_frames(panel,weights,window=window)
    for name,frame in [('metrics',metrics),('monthly',monthly)]:frame.to_parquet(stage/('original_'+name+'.parquet'),index=False)
    for recipe in meta['recipes']:
        for model in meta['models']:
            for target in meta['targets']:
                token=_case_id(recipe,model,target);case=stage/'cases'/token
                if (case/'case.json').exists():
                    saved=json.loads((case/'case.json').read_text())
                    if all(_digest(case/(name+'.parquet'))==digest for name,digest in saved['frames'].items()):
                        progress(f'Reusing complete {window} {token}');continue
                    raise ValueError(f'Incomplete/corrupt saved case: {token}')
                case.mkdir(parents=True,exist_ok=True)
                identity=dict(recipe=recipe,model_key=model,target=target,window=window)
                progress(f'{window}: {token}');started=time.perf_counter()
                fixed=EngineeredClassifier(recipe,model,target)
                if window=='training':
                    splitter=MonthForwardSplit(fit_X.date,return_month(fit_X.date),meta['cv_splits'],meta['cv_min_train_months'])
                    grid={'model_params':meta['model_grids'][model]}
                    if recipe in {'cluster_average','cluster_pca'}:grid['n_groups']=meta['group_grid']
                    if recipe=='ic_blends':grid['correlation_cap']=meta['ic_correlation_grid']
                    search=GridSearchCV(fixed,grid,cv=splitter,scoring=_score,n_jobs=1,error_score='raise',refit=True)
                    with threadpool_limits(limits=1):search.fit(fit_X,fit_y)
                    tuned=search.best_estimator_;chosen=search.best_params_
                    trials=pd.DataFrame({'params':[json.dumps(p,sort_keys=True) for p in search.cv_results_['params']],
                        'mean_cv_log_loss':-search.cv_results_['mean_test_score'],'std_cv_log_loss':search.cv_results_['std_test_score'],
                        'rank':search.cv_results_['rank_test_score']}).assign(**identity)
                    folds=splitter.fold_table(pd.DataFrame({'date':fit_X.date,'label_available_at':return_month(fit_X.date)})).assign(**identity)
                else:
                    training_case=json.loads((directory/'training'/'cases'/token/'case.json').read_text())
                    chosen=training_case['chosen_params'];tuned=EngineeredClassifier(recipe,model,target).set_params(**chosen)
                    with threadpool_limits(limits=1):tuned.fit(fit_X,fit_y)
                    trials,folds=pd.DataFrame(),pd.DataFrame()
                with threadpool_limits(limits=1):fixed.fit(fit_X,fit_y)
                frames={name:[] for name in ['metrics','monthly','predictive','audits']}
                probability=panel[KEYS].copy()
                for status,estimator in [('Fixed',fixed),('Tuned',tuned)]:
                    p=estimator.predict_proba(current)[:,1]
                    probability[status.lower()+'_probability']=p
                    predictions=panel[KEYS].assign(**{('probability_of_loss' if target=='loss' else 'probability_of_up'):p})
                    metrics,monthly=_portfolio_frames(panel,weights,predictions,status=status,**identity)
                    frames['metrics'].append(metrics);frames['monthly'].append(monthly)
                    frames['predictive'].append(_predictive(panel[TARGET],p,model,target,status,recipe,window,estimator.prevalence_))
                    frames['audits'].append(pd.DataFrame([_audit(estimator,{**identity,'status':status},origin)]))
                    joblib.dump(estimator,case/(status.lower()+'.joblib'),compress=3)
                combined={name:pd.concat(parts,ignore_index=True) for name,parts in frames.items()}
                combined.update(trials=trials,folds=folds)
                checksums={}
                for name,frame in combined.items():
                    path=case/(name+'.parquet');frame.to_parquet(path,index=False);checksums[name]=_digest(path)
                probability.to_parquet(case/'probabilities.parquet',index=False)
                _write_json(case/'case.json',{'frames':checksums,'chosen_params':chosen,'seconds':time.perf_counter()-started,
                    'probabilities_sha256':_digest(case/'probabilities.parquet'),'fit_origin':str(origin.date()),**identity})
                progress(f'Completed {token} in {time.perf_counter()-started:.1f}s')
    _aggregate(directory,window)
    return stage


def load_engineering_study(directory,window='training'):
    directory=Path(directory);meta=json.loads((directory/'study.json').read_text());stage=directory/window
    complete=json.loads((stage/'complete.json').read_text());frames={}
    for name,digest in complete['frames'].items():
        path=stage/(name+'.parquet')
        if _digest(path)!=digest:raise ValueError(f'Damaged engineering frame {name}')
        frames[name]=pd.read_parquet(path)
    scope=json.loads((stage/'scope.json').read_text())
    if (window=='training' and (scope['test_csv_loaded'] or pd.Timestamp(scope['last_return_month'])>pd.Timestamp('2009-12-31'))
            or pd.Timestamp(scope['trained_through_return_month'])>pd.Timestamp(scope['fit_origin'])):raise ValueError('Invalid engineering evaluation scope.')
    if not frames['audits'].n_features.between(1,10).all():raise ValueError('Engineered model exceeds feature limit.')
    return {'metadata':meta,'scope':scope,**frames}


def latest_engineering(root):
    found=sorted((Path(root)/'engineering_studies').glob('*/training/complete.json'),reverse=True)
    return found[0].parent.parent if found else None
