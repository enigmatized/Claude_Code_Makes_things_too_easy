"""Saved lookback comparisons, shared figure data and explicit allocation loading."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from research.cash_strategy import STRATEGY_KEY
from research.models import default_model_registry
from research.momentum_studies import list_momentum_studies, load_momentum_study
from research.store import ExperimentStore
from research.evaluation import evaluation_scope
from evaluation_ui import render_scope


def momentum_figure(study, model_id="none"):
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.07)
    for (months, summary), color in zip(study["summaries"].items(), ["#287d68", "#bd8235", "#536da6", "#97619c"]):
        models = ["none"] if model_id == "none" else ["none", model_id]
        for model in models:
            case = summary["cases"].loc[lambda f: f.strategy.eq(STRATEGY_KEY) & f.model_id.eq(model)].iloc[0]
            data = summary["monthly"].loc[lambda f: f.case_id.eq(case.case_id)].sort_values("return_month")
            name = f"{months}m · {case.model_label}"
            line = dict(color=color, dash="solid" if model == "none" else "dash")
            dates = [data.snapshot_date.iloc[0], *data.return_month]
            figure.add_trace(go.Scatter(x=dates, y=[1, *data.wealth], name=name, line=line), row=1, col=1)
            figure.add_trace(go.Scatter(x=dates, y=[0, *data.drawdown], name=name, line=line, showlegend=False), row=2, col=1)
            figure.add_trace(go.Scatter(x=data.snapshot_date, y=data.cash_weight, name=name, line=line, showlegend=False), row=3, col=1)
    summary = study["summaries"]["12"]
    control = summary["cases"].loc[lambda f: f.strategy.eq("equal_weight_universe") & f.variant.eq("baseline")].iloc[0]
    data = summary["monthly"].loc[lambda f: f.case_id.eq(control.case_id)].sort_values("return_month")
    for row, values in [(1, [1, *data.wealth]), (2, [0, *data.drawdown])]:
        figure.add_trace(go.Scatter(x=[data.snapshot_date.iloc[0], *data.return_month], y=values,
            name="Equal weight", line=dict(color="#68716e", width=2), showlegend=row == 1), row=row, col=1)
    figure.update_yaxes(title_text="Growth of $1", row=1, col=1)
    figure.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    figure.update_yaxes(title_text="Cash allocation", tickformat=".0%", range=[0, 1], row=3, col=1)
    figure.update_xaxes(title_text="Cash at snapshot; returns earned the following month", row=3, col=1)
    figure.update_layout(height=850, template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", y=-.15), margin=dict(l=10, r=10, t=10, b=10))
    scope = evaluation_scope(summary["manifest"]["config"], summary["manifest"])
    figure.update_layout(title=f"{scope['title']} · {scope['period']} · momentum study", margin_t=60)
    return figure


def _load_allocations(store, runs, key_prefix, include_ml):
    months = st.session_state[f"{key_prefix}_allocations"]
    result = store.load(runs[str(months)])
    if result.config.evaluation != "training_validation":
        raise ValueError("Momentum research must use training validation.")
    st.session_state["training_result"] = result
    target = "basic" if include_ml else "strategies"
    st.session_state[f"{target}_show_allocations"] = True
    st.session_state[f"{target}_allocations_strategy"] = "Annual consistency + positive momentum + cash"


def render_momentum_studies(include_ml=True, key_prefix="momentum", published=None):
    if published is not None:
        study = published.momentum()
        if study is None:
            return
    else:
        store = ExperimentStore()
        available = list_momentum_studies(store)
        if not available:
            return
    st.divider()
    st.subheader("Momentum lookback: 2, 3, 6 or 12 months?")
    if published is None:
        options = {f"{row['label']} · {row['study_id']}": row for row in available}
        record = options[st.selectbox("Saved momentum study", list(options), key=f"{key_prefix}_study")]
        token = (record["study_id"], record["sha256"])
        cached = st.session_state.get("loaded_momentum_study")
        if cached is None or cached[0] != token:
            cached = (token, load_momentum_study(store, record["study_id"]))
            st.session_state["loaded_momentum_study"] = cached
        study = cached[1]
    if study["metadata"]["config"]["evaluation"] != "training_validation":
        st.error("This study is not training-validation evidence.")
        return
    baseline = study["summaries"]["12"]
    render_scope(baseline["manifest"]["config"], baseline["manifest"], "Momentum study", baseline_only=not include_ml)
    st.write("The annual-winner pool stays fixed. Only the momentum window and its consecutive-history requirement change. Cash budgeting and evaluation dates stay the same.")
    st.caption("Training validation only; saved results." + (" Solid lines = original; dashed lines = selected ML version." if include_ml else " The table and curves show only the portfolio rules."))
    models = default_model_registry()
    versions = {"Original (no ML)": "none"}
    versions.update({models[key].label: key for key in study["metadata"]["config"]["models"]
                     if study["metadata"]["config"]["include_ml"]})
    model_id = versions[st.selectbox("Lookback comparison model", list(versions), key=f"{key_prefix}_model")] if include_ml else "none"
    table = study["portfolio_comparison"].loc[lambda f: f.model_id.isin(["none", model_id])]
    columns = {"lookback_months": "Momentum months", "model_label": "Version", "cagr": "CAGR",
        "annualized_log_growth": "Annual log growth",
        "sharpe": "Sharpe", "sortino": "Sortino", "max_drawdown": "Max drawdown",
        "avg_one_way_turnover": "Monthly turnover", "avg_cash_weight": "Average cash",
        "cash_only_months": "Cash-only months", "change_cagr_vs_12m": "CAGR change vs 12m"}
    if not include_ml:
        columns.pop("model_label")
    st.dataframe(table[list(columns)].rename(columns=columns).style.format({
        **{key: "{:.2%}" for key in ["CAGR", "Annual log growth", "Max drawdown", "Monthly turnover", "Average cash"]},
        "Sharpe": "{:.3f}", "Sortino": "{:.3f}", "CAGR change vs 12m": "{:+.2%}",
        "Cash-only months": "{:.0f}"}), use_container_width=True, hide_index=True)
    st.plotly_chart(momentum_figure(study, model_id), use_container_width=True, key=f"{key_prefix}_curves")
    st.caption("CAGR changes compare the same version with its 12-month version. A return gain can come with higher turnover, less cash or deeper drawdowns; no winner is selected automatically.")
    st.download_button("Download all lookback results", (study["portfolio_comparison"] if include_ml else study["portfolio_comparison"].loc[lambda f: f.variant.eq("baseline")].drop(columns=["model_id", "model_label"], errors="ignore")).to_csv(index=False),
                       file_name="momentum_lookback_comparison.csv", mime="text/csv", key=f"{key_prefix}_download")
    if published is None:
        st.selectbox("Lookback to inspect allocations", [2, 3, 6, 12], key=f"{key_prefix}_allocations")
        st.button("Load this lookback's saved allocations", key=f"{key_prefix}_load_allocations",
                  on_click=_load_allocations, args=(store, study["metadata"]["runs"], key_prefix, include_ml))
    else:
        st.caption("Allocation exploration below uses the main published experiment. Other lookbacks are shown here as a saved performance comparison.")
