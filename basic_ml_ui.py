"""Saved basic-ML experiment selection; browsing never computes a model."""
from dataclasses import asdict

import pandas as pd
import streamlit as st


def saved_basic_source(store, source, strategies, target, scope, required_model=None):
    def matches(config):
        return (config.include_ml and config.evaluation == 'training_validation'
                and config.prediction_target == target and config.model_scope == scope
                and (required_model is None or required_model in config.models)
                and set(strategies).issubset(config.strategies))
    records = [row for row in store.list_runs() if row['evaluation'] == 'training_validation']
    configs = {row['run_id']: store.configuration(row['run_id']) for row in records}
    candidates = [row for row in records if matches(configs[row['run_id']])]
    # A published chooser must never reuse a research-mode browser result.
    active = None if hasattr(store, 'publication_token') else st.session_state.get('basic_result')
    if active is None or not matches(active.config):
        active = source if source is not None and matches(source.config) else None
    if not candidates:
        return active
    names = {}
    for row in candidates:
        config = configs[row['run_id']]
        names[f"{row['case_count']} cases · {config.cash_momentum_months}m cash momentum · {config.feature_spec.label} · {row['run_id']}"] = row['run_id']
    key = 'basic_saved_experiment'
    active_id = getattr(active, 'cache_info', {}).get('run_id')
    if st.session_state.get(key) not in names:
        st.session_state.pop(key, None)
    default = list(names.values()).index(active_id) if active_id in names.values() else 0
    run_id = names[st.selectbox('Saved basic ML experiment', list(names), index=default, key=key)]
    if active is not None and getattr(active, 'cache_info', {}).get('run_id') == run_id:
        return active
    cached = st.session_state.get('basic_loaded_result')
    token = (str(store.root), run_id, getattr(store, 'publication_token', None))
    if cached is None or cached[0] != token:
        cached = (token, store.load(run_id))
        st.session_state['basic_loaded_result'] = cached
    return cached[1]


def basic_case_catalog(store, source, strategies, scope):
    """Inventory configurations only; never load forecasts or fit on dropdown changes."""
    from portfolio_selection import registry_cases
    frames = []
    configs = [(row['run_id'], store.configuration(row['run_id'])) for row in store.list_runs()
               if row['evaluation'] == 'training_validation']
    if source is not None and not hasattr(store, 'publication_token'):
        configs.append(('active', source.config))
    for run_id, config in configs:
        if not config.include_ml or config.evaluation != 'training_validation' or config.model_scope != scope:
            continue
        present = [key for key in strategies if key in config.strategies]
        frames.append(registry_cases(present, config.models, (config.prediction_target,), run_id=run_id,
                                     model_scope=scope, feature_id=config.feature_spec.feature_id))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def scope_comparison(store, result, strategies):
    """Match configurations, data and calculation provenance before comparing scopes."""
    def protocol(config):
        values = asdict(config)
        for key in ['model_scope', 'models', 'strategies']:
            values.pop(key)
        return values
    other = 'per_stock' if result.config.model_scope == 'pooled' else 'pooled'
    for row in store.list_runs():
        if row['evaluation'] != 'training_validation':
            continue
        config = store.configuration(row['run_id'])
        if config.model_scope != other or protocol(config) != protocol(result.config):
            continue
        if not set(strategies).issubset(config.strategies):
            continue
        common_models = set(config.models) & set(result.prediction_batches)
        if not common_models:
            continue
        summary = store.summary(row['run_id'])
        if any(summary['manifest'][key] != result.manifest[key] for key in ['source_sha256', 'code_sha256', 'versions']):
            continue
        frames, forecasts = [], []
        for metrics, predictive, scope in [(result.metrics, result.model_metrics, result.config.model_scope),
                                           (summary['metrics'], summary['model_metrics'], other)]:
            frame = metrics.loc[metrics.variant.ne('baseline') & metrics.strategy.isin(strategies) & metrics.model_id.isin(common_models)]
            frames.append(frame.assign(Scope='Pooled' if scope == 'pooled' else 'Per stock'))
            forecasts.append(predictive.loc[predictive.model_id.isin(common_models)].assign(
                Scope='Pooled' if scope == 'pooled' else 'Per stock'))
        return pd.concat(frames, ignore_index=True), pd.concat(forecasts, ignore_index=True), row['run_id']
    return None


def render_scope_comparison(store, result, strategies):
    with st.expander('Compare pooled and per-stock portfolios'):
        matched = scope_comparison(store, result, strategies)
        if matched is None:
            st.info('No matched saved comparison yet. Run the other scope with the same target, factors, trading settings and fitting policy.')
            return
        rows, predictions, other_id = matched
        st.markdown('**Prediction quality**')
        predictive = predictions[['model', 'Scope', 'brier_score', 'log_loss', 'roc_auc']].rename(
            columns={'model': 'Model', 'brier_score': 'Brier score', 'log_loss': 'Log loss', 'roc_auc': 'ROC AUC'})
        st.dataframe(predictive.style.format({'Brier score': '{:.4f}', 'Log loss': '{:.4f}', 'ROC AUC': '{:.4f}'}),
                     use_container_width=True, hide_index=True)
        st.markdown('**Portfolio results**')
        fields = {'label': 'Strategy', 'model_label': 'Model', 'Scope': 'Scope', 'cagr': 'CAGR',
                  'annualized_log_growth': 'Annual log growth', 'sharpe': 'Sharpe', 'max_drawdown': 'Max drawdown'}
        table = rows[list(fields)].rename(columns=fields).sort_values(['Strategy', 'Model', 'Scope'])
        st.dataframe(table.style.format({'CAGR': '{:.2%}', 'Annual log growth': '{:.2%}', 'Sharpe': '{:.3f}',
                                        'Max drawdown': '{:.2%}'}), use_container_width=True, hide_index=True)
        st.caption(f'Matched target, features, trading settings, fit schedule and data/code/runtime provenance. Shared models only. Other run: {other_id}.')
        st.download_button('Download model-scope comparison', table.to_csv(index=False),
                           file_name='model_scope_comparison.csv', mime='text/csv', key='basic_scope_download')
