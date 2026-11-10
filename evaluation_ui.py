"""Explicit period boundaries and saved historical test-result presentation."""
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots
import streamlit as st

from research.evaluation import evaluation_scope, load_archived_test_results
from research.store import ExperimentStore
from research.strategies import default_registry
from strategy_groups import grouped_strategies


def render_scope(config, manifest, label="Displayed comparison", baseline_only=False):
    scope = evaluation_scope(config, manifest)
    if baseline_only:
        scope.update(fitting="Non-ML rules", protocol="Trading rules use historical information available at each rebalance.")
    st.info(f"{label}: **{scope['title']} · {scope['period']}** · {scope['fitting']}")
    st.caption(scope["protocol"] + " " + scope["purpose"])
    return scope


def _load_test_run(store, run_id):
    st.session_state["test_result"] = store.load(run_id)


def render_test_results(published=None):
    st.subheader("Test results · Jan 2010 – Dec 2018")
    st.write("This tab separates earlier test evidence from training-validation research. Loading saved results does not fit models.")
    store = ExperimentStore() if published is None else None
    available = [row for row in store.list_runs() if row["evaluation"] == "test"] if store is not None else []
    if available:
        options = {f"{row['label']} · {row['run_id']}": row["run_id"] for row in available}
        run_id = options[st.selectbox("Registered test comparison", list(options), key="test_run")]
        st.button("Load exploratory test result", key="load_test_run", on_click=_load_test_run, args=(store, run_id))
        st.caption("This loads an exploratory result inside Final test. Training results and independently saved studies keep their own selections and periods.")
    elif published is None:
        st.info("No test-period runs are registered in the experiment library yet. The older saved non-ML test results are shown below.")
    if published is None:
        st.markdown("**Latest research status:** the feature-selection and 2/3/6/12-month cash-momentum studies cover training validation only. The final tuned, selected-feature, frozen-model test evaluation is still pending.")
    else:
        st.caption("Earlier non-ML test comparisons are separate from the frozen four-family study above.")
    st.divider()
    st.subheader("Archived non-ML test results")
    try:
        archive = load_archived_test_results() if published is None else published.archive()
        if archive is None:
            st.info("No older non-ML test comparison was included in this publication.")
            return
    except (ValueError, AssertionError, FileNotFoundError) as error:
        st.info(f"Historical test archive unavailable: {error}")
        return
    render_scope(archive["manifest"]["config"], archive["manifest"], "Historical CSV archive")
    st.caption("Thirteen earlier non-ML rules, read from recorded CSV reports. These exclude the new annual-consistency cash rule and its lookback variants. No current-code replay or new test backtest was performed.")
    labels = {key: spec.label for key, spec in default_registry().items()}
    metrics = archive["metrics"].assign(label=lambda frame: frame.strategy.map(labels).fillna(frame.strategy))
    groups = {"All strategy families": metrics.strategy.tolist()}
    groups.update({family.title: keys for family, keys in grouped_strategies(metrics.strategy)})
    family = st.selectbox("Archived test strategy family", list(groups), key="archived_test_family")
    metrics = metrics.loc[metrics.strategy.isin(groups[family])]
    columns = {"label": "Strategy", "cagr": "CAGR", "annualized_log_growth": "Annual log growth", "sharpe": "Sharpe", "sortino": "Sortino",
               "max_drawdown": "Max drawdown", "avg_one_way_turnover": "Monthly turnover"}
    st.dataframe(metrics[list(columns)].rename(columns=columns).style.format({
        "CAGR": "{:.2%}", "Annual log growth": "{:.2%}", "Sharpe": "{:.3f}", "Sortino": "{:.3f}",
        "Max drawdown": "{:.2%}", "Monthly turnover": "{:.2%}"}), use_container_width=True, hide_index=True)
    options = {labels.get(key, key): key for key in metrics.strategy}
    defaults = [labels[key] for key in ["equal_weight_universe", "momentum_weighted"] if key in set(metrics.strategy)]
    selected = st.multiselect("Archived test strategies to chart", list(options), default=defaults, key="archived_test_strategies")
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.12)
    for index, name in enumerate(selected):
        data = archive["monthly"].loc[lambda frame: frame.strategy.eq(options[name])].sort_values("return_month")
        dates = [data.snapshot_date.iloc[0], *data.return_month]
        line = dict(color=qualitative.Dark24[index % len(qualitative.Dark24)])
        figure.add_trace(go.Scatter(x=dates, y=[1, *data.wealth], name=name, line=line), row=1, col=1)
        figure.add_trace(go.Scatter(x=dates, y=[0, *data.drawdown], name=name, line=line, showlegend=False), row=2, col=1)
    figure.update_layout(title="TEST RESULTS · Jan 2010 – Dec 2018 · archived non-ML rules",
        height=650, template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=-.15))
    figure.update_yaxes(title_text="Growth of $1", row=1, col=1)
    figure.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    if selected:
        st.plotly_chart(figure, use_container_width=True)
    st.download_button("Download archived test metrics", metrics.to_csv(index=False), file_name="archived_test_metrics.csv", mime="text/csv")
    with st.expander("Historical test source files and content hashes"):
        st.json(archive["file_sha256"])
