"""Shared saved-evidence feature-engineering views for local and hosted apps."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram
import streamlit as st

from app_charts import RETURN_MEASURES, return_curve
from research.engineering import RECIPES, TECH_RAW, ECONOMIC
from research.engineering_study import latest_engineering, load_engineering_study
from research.models import default_model_registry
from tuning_ui import METRIC_HELP

METHOD_NOTES = {
 'anova_10':'Reference: select ten original factors using training-only ANOVA for the chosen up/down event. No technical signals.',
 'technical_10':'Ten percentile-ranked signals from already-earned returns. These add price-history information; no original factor is passed directly to the classifier.',
 'cluster_average':'Cluster the twenty factor columns by average linkage on 1 − |Spearman correlation|. Align each member to a central anchor, average, re-rank within the current month, and add four technical signals. Fixed: six averages + four signals; tuning may use four averages.',
 'cluster_pca':'Use the same factor groups, standardize members using training-only means/scales, and keep each group’s first principal component. Align its arbitrary sign with the group average, re-rank, and add four technical signals. PCA preserves variation, not necessarily predictive information.',
 'ic_blends':'Learn each factor’s signed mean monthly Spearman IC with future returns inside the training fold. Combine centered factor ranks in an all-factor blend and a greedy low-correlation blend, then add eight technical signals. These two blends use labels and need honest later validation.',
 'economic':'Five fixed heuristics: leverage, cash flow, yield/financing, cash flow minus leverage, and 3M minus 12M turnover rank. Add five technical signals. These combine percentile ranks, not raw accounting ratios; they are hypotheses rather than established economic identities.',
}
COLORS=['#287d68','#536da6','#b36a2e','#7a4e8a','#247a9b','#946134']


def _study(store,published,window):
    if published is not None:
        return published.study(window) if published.has_window(window) else None
    if window=='test':
        saved=sorted((Path(store.root)/'engineering_studies').glob('*/test/complete.json'),reverse=True)
        directory=saved[0].parent.parent if saved else None
    else:directory=latest_engineering(store.root)
    if directory is None or not (directory/window/'complete.json').exists():return None
    token=(str(directory),window,(directory/window/'complete.json').read_text())
    key='engineering_loaded_'+window
    if st.session_state.get(key, (None,))[0]!=token:
        st.session_state[key]=(token,load_engineering_study(directory,window))
    return st.session_state[key][1]


def render_engineering(store=None,published=None,window='training'):
    study=_study(store,published,window)
    if study is None:
        if window=='training':
            st.header('Feature engineering')
            st.info('No completed feature-engineering study is saved here. Run python3 run_feature_engineering.py locally; browsing never fits models.')
        return False
    scope=study['scope'];final=window=='test';prefix='engineering_final' if final else 'engineering'
    st.header('Feature-engineered models — final evaluation' if final else 'Feature engineering')
    st.info(('**Official test — previously viewed.**' if final else '**Training data only.**')+
        f" Earned returns {scope['first_return_month']} to {scope['last_return_month']} · {scope['months']} months. "
        f"All learned features and classifiers are frozen at {scope['fit_origin']}; labels used through {scope['trained_through_return_month']}.")
    st.caption('Fits pool stock-months across stocks: one classifier per recipe/algorithm/direction/settings. '
               'These are new input-recipe × classifier variants. Factor clustering combines columns; Grouping With ML clusters stocks. '
               'The same four classifier algorithms and fixed portfolio rules are used. Cash earns 0%; gross of trading costs.')
    if final:
        st.caption('Tuned variants carry the settings chosen in the early-training CV search, then refit their features/classifier once on all training data. '
                   'No test-label fitting or retuning. Stock groups and technical signals update using returns already earned at each decision.')
        _results(study,prefix)
        return True
    methods,results,tuning=st.tabs(['Methods & feature construction','Model & portfolio results','Hyperparameter tuning'])
    with methods:_methods(study)
    with results:_results(study,prefix)
    with tuning:_tuning(study)
    if published is None:
        with st.expander('Explicit research commands — viewing never retrains'):
            st.code('python3 run_feature_engineering.py\n'
                    'python3 run_feature_engineering.py --phase final --study artifacts/engineering_studies/STUDY',language='bash')
            st.caption('The runner writes a declared, resumable experiment before computation. Keep final evaluation separate from training exploration.')
    return True


def _methods(study):
    labels={RECIPES[key]:key for key in study['metadata']['recipes']}
    recipe=labels[st.selectbox('Feature recipe to inspect',list(labels),
        index=list(labels.values()).index('cluster_average') if 'cluster_average' in labels.values() else 0,key='engineering_method')]
    st.write(METHOD_NOTES[recipe])
    st.caption('Each fitted model receives at most ten constructed inputs. Several recipes aggregate twenty source factors. '
               'That source count is disclosed: confirm the instructor’s interpretation of the ten-feature rule before formal submission.')
    frame=study['audits'].loc[lambda x:x.recipe.eq(recipe)&x.status.eq('Fixed')]
    audit=json.loads(frame.iloc[0].audit_json)
    st.markdown('**Inputs used by this saved fixed recipe:** '+', '.join(audit['output_columns']))
    st.caption(f"Construction audit: {frame.iloc[0].model_key} / {frame.iloc[0].target}, fitted at {audit['fit_origin']}. "
               'The results tab independently trains every model/target. The technical-only recipe uses no original factors as model inputs.')
    if audit.get('linkage'):
        tree=np.asarray(audit['linkage'])
        data=dendrogram(tree,no_plot=True,labels=audit['source_columns'])
        figure=go.Figure()
        for xs,ys in zip(data['dcoord'],data['icoord']):figure.add_trace(go.Scatter(x=xs,y=ys,mode='lines',
            line=dict(color='#536da6'),showlegend=False,hovertemplate='Merge distance %{x:.3f}<extra></extra>'))
        figure.update_layout(template='plotly_white',height=580,xaxis_title='1 − |Spearman correlation| merge distance',
            yaxis=dict(tickmode='array',tickvals=np.arange(20)*10+5,ticktext=data['ivl']),
            margin=dict(l=210,r=20,t=30,b=40),title='Factor redundancy tree — training fit only')
        st.plotly_chart(figure,use_container_width=True,key='engineering_factor_tree')
        st.caption('Nearby branches combine redundant factor columns, including opposite-sign relationships. '
                   'Group count, not an individual pairwise correlation threshold, determines the saved partition.')
        groups=[]
        for key,members in audit['groups'].items():groups.append({'Group':int(key)+1,'Factors':', '.join(members),'Anchor':audit['signs'][key]['anchor']})
        st.dataframe(pd.DataFrame(groups),hide_index=True,use_container_width=True)
        if audit.get('pca'):
            pca=audit['pca']
            fig=go.Figure(go.Bar(x=[f'Group {int(g)+1}' for g in pca],y=[v['explained_variance_ratio'] for v in pca.values()],
                hovertemplate='%{x}<br>Variance retained %{y:.1%}<extra></extra>'))
            fig.update_layout(template='plotly_white',title='Variation retained by the first component',yaxis=dict(range=[0,1],tickformat='.0%'))
            st.plotly_chart(fig,use_container_width=True,key='engineering_pca_variance')
            rows=[{'Group':int(g)+1,'Factor':f,'PC1 loading':w} for g,v in pca.items() for f,w in zip(v['members'],v['components'][0])]
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    if audit.get('weights'):
        weights=audit['weights']['blend_ic_all']
        fig=go.Figure(go.Bar(y=list(weights),x=list(weights.values()),orientation='h',
            marker_color=['#287d68' if value>0 else '#b36a2e' for value in weights.values()],
            hovertemplate='%{y}<br>Training mean monthly IC %{x:.4f}<extra></extra>'))
        fig.update_layout(template='plotly_white',height=560,title='Signed training IC weights',xaxis_title='Mean monthly Spearman IC',margin=dict(l=200))
        st.plotly_chart(fig,use_container_width=True,key='engineering_ic_weights')
        st.caption('These weights use future-return labels from the fitting slice. Their fit-sample IC is optimistic; '
                   'the later holdout, not this bar chart, evaluates usefulness.')
    with st.expander('Technical definitions, missing history and notebook review'):
        st.write('Technical signals are last earned return; 3/6/12M return averages; 3M−12M average; synthetic price versus '
                 '3/6/12 price averages; 12−1 momentum skipping the latest earned month; and 6/12M log-return volatility. '
                 'Calendar gaps break histories. Unknown/nonfinite histories receive neutral rank 0.5; rows are not dropped differently between recipes.')
        st.write('The original notebook is exploratory and has already viewed test results. Full-training composites cannot be reused '
                 'for earlier validation dates. Here every group, sign, PCA loading, scaler, IC weight and ANOVA mask is learned '
                 'inside the correct CV fold. Final features fit on training only. Unsupervised choices can still overfit through recipe selection.')
        st.caption('Full-precision signs, groups, PCA/scaler parameters, IC weights and model settings are saved; '
                   'the app needs no exported extended CSV or fitted estimator to display these audits.')
        st.markdown('[scikit-learn: preventing preprocessing leakage](https://scikit-learn.org/1.4/common_pitfalls.html#data-leakage)')


def _controls(study,prefix):
    registry=default_model_registry();cols=st.columns(3)
    models={registry[k].label:k for k in study['metadata']['models']}
    model=models[cols[0].selectbox('Prediction model',list(models),key=prefix+'_model')]
    target={'Next down':'loss','Next up':'upside'}[cols[1].selectbox('Prediction target',['Next down','Next up'],key=prefix+'_target')]
    books=study['metadata']['books']
    default=books.index('K-means top 10') if prefix=='engineering_final' and 'K-means top 10' in books else 0
    book=cols[2].selectbox('Starting portfolio',books,index=default,key=prefix+'_book')
    if book in {'K-means top 10','Ward top 10'}:
        st.caption(f"{book}: five stock groups → top 10 by trailing 12-month positive volatility in each group → "
                   "1-month positive-momentum weights, leaving unused capital in cash. The Original row/curve is this strategy before any directional adjustment.")
    status=st.radio('Estimator settings',['Fixed','Tuned','Both'],horizontal=True,key=prefix+'_settings')
    labels={RECIPES[k]:k for k in study['metadata']['recipes']}
    selected=st.multiselect('Feature recipes to compare',list(labels),default=list(labels),key=prefix+'_recipes')
    return model,target,book,status,[labels[label] for label in selected]


def _results(study,prefix):
    model,target,book,status,recipes=_controls(study,prefix)
    statuses=['Fixed','Tuned'] if status=='Both' else [status]
    frames={}
    for name in ('metrics','monthly'):
        data=study[name]
        chosen=data.book.eq(book)&(data.recipe.eq('original') | (data.model_key.eq(model)&data.target.eq(target)&data.status.isin(statuses)&data.recipe.isin(recipes)))
        frames[name]=data.loc[chosen].copy()
    metrics=frames['metrics']
    show=metrics[['recipe','status','cagr','annualized_log_growth','annualized_volatility','sharpe','max_drawdown','avg_cash_weight']].copy()
    show['recipe']=show.recipe.map(lambda key:RECIPES.get(key,'Original portfolio'))
    show=show.rename(columns={'recipe':'Inputs','status':'Settings','cagr':'CAGR','annualized_log_growth':'Annual log growth',
        'annualized_volatility':'Volatility','sharpe':'Sharpe','max_drawdown':'Max drawdown','avg_cash_weight':'Average cash'})
    counts=study['audits'].loc[lambda f:f.model_key.eq(model)&f.target.eq(target)].set_index(['recipe','status']).n_features
    show['Model inputs']=[int(counts.loc[(row.recipe,row.status)]) if row.recipe!='original' else 0 for row in metrics.itertuples()]
    st.dataframe(show.style.format({c:'{:.3f}' if c=='Sharpe' else '{:.2%}' for c in show.columns[2:] if c!='Model inputs'}),hide_index=True,use_container_width=True)
    st.caption('Original = unchanged starting rule. Every recipe uses the same stocks/months and portfolio rule. '
               'Downside reductions remain in cash for cash-capable books; the equal-weight overlay renormalizes to 100% stocks. '
               'Upside boosts held names at probability ≥60% and preserves the original stock budget.')
    measure=st.selectbox('Return measure',list(RETURN_MEASURES),key=prefix+'_return')
    st.plotly_chart(_curves(frames['monthly'],measure),use_container_width=True,key=prefix+'_curves')
    with st.expander('Prediction quality on the full stock-month universe'):
        data=study['predictive'];rows=data.loc[data.model_key.eq(model)&data.target.eq(target)&data.recipe.isin(recipes)&data.status.isin(statuses)]
        classifier=rows.loc[rows.predictor.eq('Classifier')].copy()
        control=rows.loc[rows.predictor.eq('Historical event rate')].head(1).copy()
        control['recipe']='reference';control['status']='Reference'
        scored=pd.concat([classifier,control],ignore_index=True)
        scored['recipe']=scored.recipe.map(lambda key:RECIPES.get(key,'No features'))
        scored=scored.rename(columns={'recipe':'Inputs','status':'Settings','predictor':'Predictor','brier_score':'Brier score','log_loss':'Log loss','roc_auc':'ROC AUC'})
        st.dataframe(scored[['Inputs','Settings','Predictor','Brier score','Log loss','ROC AUC']],hide_index=True,use_container_width=True,
            column_config={name:st.column_config.NumberColumn(name,help=text,format='%.5f') for name,text in METRIC_HELP.items()})
        st.caption('The constant historical-event-rate control is shown once. These probabilities score all stocks, '
                   'not only the selected holdings. Changing the starting portfolio does not change prediction scores.')
    with st.expander('Fitted inputs and construction details for these models'):
        audits=study['audits'].loc[lambda f:f.model_key.eq(model)&f.target.eq(target)&f.recipe.isin(recipes)&f.status.isin(statuses)]
        details=[]
        for row in audits.itertuples():
            data=json.loads(row.audit_json)
            details.append({'Inputs':RECIPES[row.recipe],'Settings':row.status,'Model inputs':row.n_features,
                'Fitted at':data['fit_origin'],'Feature names':', '.join(data['output_columns'])})
        st.dataframe(pd.DataFrame(details),use_container_width=True,hide_index=True)
        st.caption('Names here describe this evaluation’s own fitted model. Training and final-origin group/PCA/IC fits can differ.')
    st.download_button('Download displayed portfolio results',show.to_csv(index=False),'engineered_portfolio_results.csv',key=prefix+'_download')


def _curves(monthly,measure):
    figure=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.07,row_heights=[.5,.25,.25])
    for (recipe,status),rows in monthly.groupby(['recipe','status'],sort=False):
        rows=rows.sort_values('return_month');book=rows.set_index('return_month').copy();book['log_return']=np.log1p(book.simple_return)
        curve=return_curve(book,measure)
        color='#68716e' if recipe=='original' else COLORS[list(RECIPES).index(recipe)]
        name='Original' if recipe=='original' else RECIPES[recipe]+' · '+status
        line=dict(color=color,dash='dash' if status=='Tuned' else 'solid')
        figure.add_trace(go.Scatter(x=curve.index,y=curve,name=name,line=line),row=1,col=1)
        figure.add_trace(go.Scatter(x=rows.return_month,y=rows.drawdown,name=name,line=line,showlegend=False),row=2,col=1)
        figure.add_trace(go.Scatter(x=rows.return_month,y=rows.cash_weight,name=name,line=line,showlegend=False),row=3,col=1)
    figure.update_yaxes(title_text=measure,row=1,col=1)
    figure.update_yaxes(title_text='Drawdown',tickformat='.0%',row=2,col=1)
    figure.update_yaxes(title_text='Cash',tickformat='.0%',row=3,col=1)
    extra_legend_space=max(0, (monthly[['recipe','status']].drop_duplicates().shape[0]+1)//2-4)*22
    figure.update_layout(template='plotly_white',height=820+extra_legend_space,hovermode='x unified',
                         legend=dict(orientation='h',y=-.12,entrywidth=320),margin=dict(t=20,b=180+extra_legend_space))
    return figure


def _tuning(study):
    st.write('Chronological GridSearchCV minimizes binary event log loss. Each candidate refits its feature construction and '
             'classifier using only the fold’s training rows. Composite IC weights use continuous return ranks; the classifier predicts the chosen binary event.')
    registry=default_model_registry()
    labels={RECIPES[k]:k for k in study['metadata']['recipes']}
    recipe=labels[st.selectbox('Recipe to tune/review',list(labels),key='engineering_tune_recipe')]
    models={registry[k].label:k for k in study['metadata']['models']}
    model=models[st.selectbox('Tuned prediction model',list(models),key='engineering_tune_model')]
    target={'Next down':'loss','Next up':'upside'}[st.selectbox('Tuned event',['Next down','Next up'],key='engineering_tune_target')]
    trials=study['trials'].loc[lambda f:f.recipe.eq(recipe)&f.model_key.eq(model)&f.target.eq(target)].copy()
    st.caption('Classifier grid: '+json.dumps(study['metadata']['model_grids'][model])+'. '+
        ('Cluster counts: 4 or 6, giving 8 or 10 constructed model inputs.' if recipe.startswith('cluster_') else
         'IC pruning correlation cap: 0.4 or 0.6.' if recipe=='ic_blends' else 'The recipe itself stays fixed.'))
    fig=go.Figure(go.Bar(x=[f'Candidate {i+1}' for i in range(len(trials))],y=trials.mean_cv_log_loss,error_y=dict(type='data',array=trials.std_cv_log_loss),
        customdata=trials.params,hovertemplate='%{customdata}<br>CV log loss %{y:.5f}<extra></extra>'))
    fig.update_layout(template='plotly_white',height=380,yaxis_title='Mean CV log loss (lower is better)',title='Candidate settings; error bars are variation across folds')
    st.plotly_chart(fig,use_container_width=True,key='engineering_trials_graph')
    st.dataframe(trials[['params','mean_cv_log_loss','std_cv_log_loss','rank']],hide_index=True,use_container_width=True)
    st.dataframe(study['folds'].loc[lambda f:f.recipe.eq(recipe)&f.model_key.eq(model)&f.target.eq(target)],hide_index=True,use_container_width=True)
    st.caption('Fold variation is not a confidence interval. The independent later training holdout evaluates the winner; '
               'a lower CV score does not guarantee a better portfolio. Fixed and tuned variants are both carried to Final test without retuning there.')
