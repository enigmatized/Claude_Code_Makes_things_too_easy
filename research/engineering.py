"""Point-in-time technical inputs and training-fitted factor composites.

This is separate from stock clustering: here columns (factors) are combined.
Every learned operation is fitted inside the estimator's historical training fold.
Dates/IDs identify complete cross-sections; forward returns are never X columns.
"""
from collections import OrderedDict
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .legacy import FEATURES, KEYS, TARGET
from .models import default_model_registry, event_labels
from .basket_momentum import realized_returns

RECIPES = {
    'anova_10': 'Original factors: ANOVA 10',
    'technical_10': 'Technical signals only',
    'cluster_average': 'Cluster averages + technical',
    'cluster_pca': 'Cluster PCA + technical',
    'ic_blends': 'IC blends + technical',
    'economic': 'Economic composites + technical',
}
TECH_RAW = ('ta_ret_1m','ta_ma_3m','ta_ma_6m','ta_ma_12m','ta_ma_cross',
            'ta_px_vs_ma_3m','ta_px_vs_ma_6m','ta_px_vs_ma_12m',
            'ta_mom_12_1','ta_vol_6m','ta_vol_12m')
TECH = tuple(name+'_rk' for name in TECH_RAW)
FOUR_TECH = ('ta_ret_1m_rk','ta_ma_cross_rk','ta_mom_12_1_rk','ta_vol_12m_rk')
ECONOMIC = {
    'leverage': ('Debtequity','Total_Debt_Capital','Total_Liabilities_Total_Assets','Tot_Debt_Rev','Net_Debt'),
    'cashflow': ('Ocf_Ta','Ocf_Toa','Ocf_Ce','Ocf_Tbv','Fcf_Tbv'),
    'payout': ('Div_Yld','Bb_Yld','Netdebtyield','Fcf_Yld'),
}
INPUT_COLUMNS = [*KEYS, *FEATURES, *TECH]
_FEATURE_CACHE = OrderedDict()


def technical_inputs(panel):
    """Only already-earned calendar returns; gaps invalidate rolling windows.

    Nonfinite log history (including a total loss) remains missing. Percentile
    ranks use the current stock cross-section, never future dates or labels.
    Missing histories become neutral 0.5, preserving a matched universe.
    """
    wide = realized_returns(panel)
    lr = np.log1p(wide.where(wide.gt(-1)))
    feats = {'ta_ret_1m': wide}
    for k in (3,6,12):
        feats[f'ta_ma_{k}m'] = wide.rolling(k,min_periods=k).mean()
        acc = wide.notna().astype(float).where(wide.notna()) * 0
        back = acc.copy()
        for j in range(k):
            acc = acc + np.exp(-back)
            back = back + lr.shift(j)
        feats[f'ta_px_vs_ma_{k}m'] = 1/(acc/k)-1
    feats['ta_ma_cross'] = feats['ta_ma_3m']-feats['ta_ma_12m']
    feats['ta_mom_12_1'] = np.expm1(lr.shift(1).rolling(11,min_periods=11).sum())
    for k in (6,12):
        feats[f'ta_vol_{k}m'] = lr.rolling(k,min_periods=k).std()
    long = pd.concat([feats[name].stack(future_stack=True).rename(name) for name in TECH_RAW],axis=1)
    long.index.names = KEYS
    result = panel[KEYS].merge(long.reset_index(),on=KEYS,validate='one_to_one')
    result['technical_missing_count'] = result[list(TECH_RAW)].isna().sum(axis=1)
    for name in TECH_RAW:
        result[name+'_rk'] = result.groupby('date')[name].rank(pct=True).fillna(.5)
    return result


def monthly_ic(X, returns):
    """Monthly Spearman IC; rank of simple and finite log return gives same order."""
    data = X[list(FEATURES)].copy()
    data['_return'] = np.asarray(returns)
    data['date'] = pd.to_datetime(X.date).to_numpy()
    ranked = data.groupby('date')[[*FEATURES,'_return']].rank()
    ranked['date'] = data.date
    return ranked.groupby('date').apply(lambda block: block[FEATURES].corrwith(block['_return']),
                                        include_groups=False)


def _rank(frame, values):
    return pd.Series(np.asarray(values),index=frame.index).groupby(frame.date).rank(pct=True).fillna(.5)


class FeatureMap:
    """An immutable fitted recipe, including full-precision reconstructable state."""
    def __init__(self, recipe='cluster_average', n_groups=6, correlation_cap=.6, target='loss'):
        self.recipe,self.n_groups,self.correlation_cap,self.target=recipe,n_groups,correlation_cap,target

    def fit(self,X,returns):
        if self.recipe not in RECIPES or not 2 <= self.n_groups <= 6:
            raise ValueError('Choose a declared recipe and 2–6 groups (at most ten outputs).')
        if TARGET in X or 'fwd_log_ret' in X:
            raise ValueError('Forward returns cannot be input columns.')
        if not set(INPUT_COLUMNS).issubset(X):
            raise ValueError('Missing declared source or technical inputs.')
        if not np.isfinite(X[[*FEATURES,*TECH]].to_numpy()).all():
            raise ValueError('Inputs must be finite; technical missing history must be neutralized.')
        self.audit_ = {'recipe':self.recipe,'fit_first_snapshot':str(pd.Timestamp(X.date.min()).date()),
            'fit_last_snapshot':str(pd.Timestamp(X.date.max()).date()),'rows':len(X),
            'groups':{},'signs':{},'pca':{},'weights':{},'source_factor_count':len(FEATURES)}
        self.columns_ = []
        if self.recipe=='anova_10':
            scores,_=f_classif(X[FEATURES],event_labels(pd.Series(np.asarray(returns)),self.target))
            scores=np.nan_to_num(scores,nan=-np.inf)
            self.selected_=list(np.asarray(FEATURES)[np.argsort(-scores,kind='stable')[:10]])
            self.columns_=self.selected_
            self.audit_['selected']=self.selected_
            self.audit_['output_columns']=self.columns_
            return self
        if self.recipe.startswith('cluster_'):
            corr=X[FEATURES].corr(method='spearman').fillna(0)
            np.fill_diagonal(corr.values,1.)
            dist=(1-corr.abs()).clip(0,1);np.fill_diagonal(dist.values,0.)
            self.tree_=linkage(squareform(dist.to_numpy(),checks=True),method='average')
            self.corr_=corr
            labels=cut_tree(self.tree_,n_clusters=[self.n_groups]).ravel()
            self.groups_={int(g):[f for f,label in zip(FEATURES,labels) if label==g] for g in sorted(set(labels))}
            self.signs_,self.pcas_={},{}
            for group,members in self.groups_.items():
                sub=corr.loc[members,members]
                anchor=sub.abs().mean().idxmax()
                signs={f:float(np.sign(sub.loc[anchor,f]) or 1) for f in members}
                self.signs_[group]=signs
                self.audit_['groups'][str(group)]=members
                self.audit_['signs'][str(group)]={'anchor':anchor,**signs}
                if self.recipe=='cluster_pca':
                    scaler=StandardScaler().fit(X[members])
                    values=scaler.transform(X[members])
                    pca=PCA(n_components=1,svd_solver='full').fit(values)
                    reference=sum(signs[f]*(X[f]-.5) for f in members)
                    if np.corrcoef(pca.transform(values)[:,0],reference)[0,1]<0:pca.components_*=-1
                    self.pcas_[group]=(scaler,pca)
                    self.audit_['pca'][str(group)]={'members':members,'mean':scaler.mean_.tolist(),
                        'scale':scaler.scale_.tolist(),'pca_mean':pca.mean_.tolist(),
                        'components':pca.components_.tolist(),'explained_variance_ratio':float(np.nan_to_num(pca.explained_variance_ratio_[0]))}
                self.columns_.append(f'group_{group+1}_'+('pc1' if self.recipe=='cluster_pca' else 'average'))
            self.columns_ += list(FOUR_TECH)
        elif self.recipe=='ic_blends':
            ic=monthly_ic(X,returns)
            weights=ic.mean().fillna(0.)
            order=(ic.mean()/ic.std().replace(0,np.nan)).abs().fillna(0).sort_values(ascending=False,kind='stable').index
            corr=X[FEATURES].corr(method='spearman').fillna(0.)
            chosen=[]
            for f in order:
                if all(abs(corr.loc[f,c])<self.correlation_cap for c in chosen):chosen.append(f)
                if len(chosen)==6:break
            self.weights_={'blend_ic_all':weights.to_dict(),'blend_ic_pruned':weights[chosen].to_dict()}
            self.audit_['weights']=self.weights_
            self.audit_['ic_months']=int(len(ic))
            self.columns_=['blend_ic_all','blend_ic_pruned',*TECH[:8]]
        elif self.recipe=='economic':
            self.columns_=['hb_leverage','hb_cashflow','hb_payout','hb_quality_minus_lev','hb_turnover_trend',*FOUR_TECH,'ta_px_vs_ma_6m_rk']
        else:self.columns_=list(TECH[:10])
        if len(self.columns_)>10:raise AssertionError('More than ten model inputs.')
        self.audit_['output_columns']=self.columns_
        return self

    def transform(self,X):
        if self.recipe=='anova_10':return X[self.selected_].copy()
        out=pd.DataFrame(index=X.index)
        if self.recipe.startswith('cluster_'):
            for group,members in self.groups_.items():
                if self.recipe=='cluster_pca':
                    scaler,pca=self.pcas_[group];values=pca.transform(scaler.transform(X[members]))[:,0]
                else:values=sum(self.signs_[group][f]*(X[f]-.5) for f in members)/len(members)
                out[self.columns_[group]]=_rank(X,values)
        elif self.recipe=='ic_blends':
            for name,weights in self.weights_.items():out[name]=_rank(X,sum(w*(X[f]-.5) for f,w in weights.items()))
        elif self.recipe=='economic':
            averages={name:X[list(members)].mean(axis=1) for name,members in ECONOMIC.items()}
            for name,value in averages.items():out['hb_'+name]=_rank(X,value)
            out['hb_quality_minus_lev']=_rank(X,averages['cashflow']-averages['leverage'])
            out['hb_turnover_trend']=_rank(X,X.Share_Turn_3M-X.Share_Turn_12M)
        for name in self.columns_:
            if name in TECH:out[name]=X[name]
        return out[self.columns_]


def _fit_features(X,y,recipe,n_groups,cap,target):
    # Sharing identical training-only maps between algorithms/targets saves work;
    # content hashes include every input and outcome, preventing stale reuse.
    digest=hashlib.sha256(pd.util.hash_pandas_object(X,index=True).values.tobytes()+np.asarray(y,dtype=float).tobytes()).hexdigest()
    token=(digest,recipe,n_groups,cap,target if recipe=='anova_10' else 'return_IC')
    if token not in _FEATURE_CACHE:
        fitted=FeatureMap(recipe,n_groups,cap,target).fit(X,y)
        _FEATURE_CACHE[token]=(fitted,fitted.transform(X))
        if len(_FEATURE_CACHE)>16:_FEATURE_CACHE.popitem(last=False)
    return _FEATURE_CACHE[token]


class EngineeredClassifier(ClassifierMixin,BaseEstimator):
    """GridSearchCV estimator with continuous returns used only inside fit/scorer.

    The classifier consumes a binary event. Continuous y is retained externally
    so IC composites can learn *return* ranks, as in the source notebook. A custom
    scorer computes standard sklearn binary log loss, not portfolio performance.
    """
    def __init__(self,recipe='cluster_average',model_key='logistic',prediction_target='loss',
                 n_groups=6,correlation_cap=.6,model_params=None):
        self.recipe,self.model_key,self.prediction_target=recipe,model_key,prediction_target
        self.n_groups,self.correlation_cap,self.model_params=n_groups,correlation_cap,model_params

    def fit(self,X,y):
        if os.environ.get('MLFACTOR_READ_ONLY')=='1':raise RuntimeError('Fitting disabled in view mode.')
        self.feature_map_,features=_fit_features(X,y,self.recipe,self.n_groups,self.correlation_cap,self.prediction_target)
        spec=default_model_registry()[self.model_key]
        self.scaler_=StandardScaler().fit(features) if spec.scale else None
        values=self.scaler_.transform(features) if self.scaler_ else features.to_numpy()
        event=event_labels(pd.Series(np.asarray(y)),self.prediction_target)
        self.prevalence_=float(event.mean());self.classes_=np.array([0,1])
        self.model_=spec.factory().set_params(**(self.model_params or {}))
        self.constant_=event.nunique()<2
        if not self.constant_:
            with threadpool_limits(limits=1):self.model_.fit(values,event)
        self.n_features_in_=len(self.feature_map_.columns_)
        return self

    def predict_proba(self,X):
        features=self.feature_map_.transform(X)
        values=self.scaler_.transform(features) if self.scaler_ else features.to_numpy()
        if self.constant_:p=np.repeat(self.prevalence_,len(X))
        else:
            with threadpool_limits(limits=1):p=self.model_.predict_proba(values)[:,list(self.model_.classes_).index(1)]
        return np.column_stack([1-p,p])

    def predict(self,X):return (self.predict_proba(X)[:,1]>=.5).astype(int)
