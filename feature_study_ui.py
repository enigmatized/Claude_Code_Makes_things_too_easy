"""Read-only feature-study charts and explanations for the local app."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from basket_feature_ui import render_four_book_features
from research.feature_studies import feature_audit, list_feature_studies, load_feature_study
from research.models import default_model_registry
from research.models import event_fields
from research.feature_reporting import (feature_curves, importance_table, method_notes,
                                        prediction_table, portfolio_table, selection_overlap,
                                        calibration_table, RECIPE_COLORS, forest_findings)
from pipeline import FEATURE_INFO  # Made importable by research.legacy above.
from research.store import ExperimentStore
from research.strategies import default_registry
from evaluation_ui import render_scope
from research.evaluation import evaluation_scope
from portfolio_focus import main_portfolio_focus


def _sync_feature_prediction_model(source_key, target_key):
    """Keep the nearby portfolio control and shared feature-study model in sync."""
    st.session_state[target_key] = st.session_state[source_key]


def study_figure(study, strategy, model_id, cumulative_log=False):
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.07, row_heights=[.5, .25, .25])
    summaries = study["summaries"]
    base = summaries["all_20"]
    histories = feature_curves(study, strategy, model_id)
    reference = histories[1][1].wealth.to_numpy()
    for label, monthly, color in histories[2:]:
        figure.add_trace(go.Scatter(x=[monthly.snapshot_date.iloc[0], *monthly.return_month],
            y=[0, *(monthly.wealth.to_numpy() / reference - 1)], name=label + " vs all 20", showlegend=False,
            line=dict(color=color), hovertemplate="%{x|%b %Y}<br>Relative wealth %{y:+.2%}<extra>%{fullData.name}</extra>"), row=3, col=1)
    for label, monthly, color in histories:
        x = [monthly.snapshot_date.iloc[0], *monthly.return_month]
        values = [0, *monthly.log_return.cumsum()] if cumulative_log else [1, *monthly.wealth]
        figure.add_trace(go.Scatter(x=x, y=values, name=label, line=dict(color=color),
            hovertemplate="%{x|%b %Y}<br>%{y:.3f}<extra>%{fullData.name}</extra>"), row=1, col=1)
        figure.add_trace(go.Scatter(x=x, y=[0, *monthly.drawdown], name=label, showlegend=False, line=dict(color=color),
            hovertemplate="%{x|%b %Y}<br>Drawdown %{y:.1%}<extra>%{fullData.name}</extra>"), row=2, col=1)
    figure.add_hline(y=0, line_dash="dot", line_color="#999999", row=3, col=1)
    figure.update_yaxes(title_text="Cumulative log return" if cumulative_log else "Growth of $1", row=1, col=1)
    figure.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    figure.update_yaxes(title_text="vs all 20", tickformat="+.1%", row=3, col=1)
    figure.update_layout(template="plotly_white", height=780, hovermode="x unified",
        legend=dict(orientation="h", y=-.15), margin=dict(l=10, r=10, t=20, b=10))
    scope = evaluation_scope(base["manifest"]["config"], base["manifest"])
    figure.update_layout(title=f"{scope['title']} · {scope['period']} · feature study", margin_t=60)
    return figure


def render_feature_studies(selected_strategies=None, key_prefix="features", published=None):
    store = ExperimentStore() if published is None else None
    main_only = main_portfolio_focus(key_prefix)
    st.info("**Training data only.** Feature comparisons use saved training-validation results; browsing does not fit selectors or prediction models.")
    has_books = render_four_book_features(store, published=published, main_only=main_only)
    available = list_feature_studies(store) if published is None else published.list_studies()
    st.subheader("Selected features and classifier evidence" if main_only and has_books else "Every saved registry strategy")
    st.caption("The selection audit and prediction diagnostics remain available for all four classifier types. "
               "Turn off the focus filter to add the fourteen basic strategy comparisons." if main_only and has_books else
               "These fourteen rules sit below the four books. A feature cut changes only their model inputs.")
    if not available:
        st.info("No completed feature-selection study is saved yet. Run python3 run_feature_selection.py to create one.")
        return
    options = {f"{row['label']} · {row['study_id']}": row for row in available}
    record = options[st.selectbox("Saved feature study", list(options), key=f"{key_prefix}_feature_study")]
    cache_key = (str(store.root) if published is None else published.token, record["study_id"], record["sha256"])
    cached = st.session_state.get("loaded_feature_study")
    if cached is None or cached[0] != cache_key:
        cached = (cache_key, load_feature_study(store, record["study_id"]) if published is None else published.study())
        st.session_state["loaded_feature_study"] = cached
    study = cached[1]
    if study["metadata"]["config"]["evaluation"] != "training_validation":
        st.error("Feature research requires training-validation evidence.")
        return
    baseline = study["summaries"]["all_20"]
    render_scope(baseline["manifest"]["config"], baseline["manifest"], "Feature study (independent of the main comparison)")
    st.caption("Loaded saved results only. No models or selectors are fitted while reviewing this tab.")
    base_config = study["metadata"]["config"]
    st.caption("The saved study refitted each model for every feature recipe at every historical origin. Browsing reads those results; it does not retrain them.")
    target = "next-month loss" if base_config.get("prediction_target", "loss") == "loss" else "positive next-month return"
    scope = base_config.get("model_scope", "pooled").replace("_", " ")
    st.caption(f"Recorded study: {target} prediction · {scope} models. This study is separate from the target/scope currently selected in Basic ML.")
    models = default_model_registry()
    model = {spec.label: key for key, spec in models.items() if key in base_config["models"]}
    model_id = model[st.selectbox("Prediction model to review", list(model),
        help="Applies to both feature subtabs. Feature recipes choose inputs; this control chooses the model that predicts next-month direction.",
        key=f"{key_prefix}_study_model")]
    selection_tab, effects_tab = st.tabs(["Selected features & method", "Effect on ML models"])
    with selection_tab:
        render_selected_features(study, model_id, key_prefix)
    with effects_tab:
        render_feature_effects(study, model_id, selected_strategies, key_prefix,
                              include_registry=not main_only or not has_books)
        st.caption("The Tuned model tab freezes one feature list from this study and compares no ML, fixed ML, and tuned ML on a later slice of the training years.")


def render_selected_features(study, model_id, key_prefix):
    st.subheader("Which inputs were selected, and how?")
    st.write("Start with the 20 supplied factors. Every saved recipe selected inputs and refitted the model using only outcomes known at each historical fit.")
    with st.expander("How the methods work and how to read the charts"):
        st.markdown(study.get("method_notes") or method_notes())
        st.caption("Available in this saved study: " + ", ".join(
            value["label"] for value in study["metadata"]["recipes"].values()))
    st.caption("Scores are ranking heuristics, not significance or causality claims. Feature selection changes ML inputs; it does not alter the non-ML portfolio rules.")
    fits = study["fits"].loc[lambda rows: rows.model_id.eq(model_id)]
    recipes = {study["metadata"]["recipes"][key]["label"]: key for key in study["metadata"]["runs"] if key != "all_20"}
    controls = st.columns(2)
    labels = list(recipes)
    forest_default = next((i for i, label in enumerate(labels) if recipes[label] == "forest_10"), 0)
    recipe = recipes[controls[0].selectbox("Selection method to inspect", labels, index=forest_default, key=f"{key_prefix}_study_recipe")]
    settings = study["metadata"]["recipes"][recipe]
    if settings["selection"] == "correlation_pruned":
        st.caption(f"Recorded pruning rule: reject |training correlation| ≥ {settings['correlation_threshold']:.2f}; retain at most {settings['maximum_selected']} inputs without backfilling rejected duplicates.")
    else:
        st.caption(f"Recorded ranking rule: retain at most {settings['maximum_selected']} varying inputs; ties follow the original candidate order.")
    audit = feature_audit(fits.loc[fits.recipe.eq(recipe)])
    snapshots = {f"{pd.Timestamp(date):%b %Y}": pd.Timestamp(date) for date in sorted(audit.fit_snapshot.unique())}
    snapshot = snapshots[controls[1].selectbox("Historical fit to inspect", list(snapshots), key=f"{key_prefix}_study_fit")]
    selected_audit = audit.loc[audit.fit_snapshot.eq(snapshot)].sort_values("rank")
    groups = {row[0]: row[1] for row in FEATURE_INFO}
    meanings = {row[0]: row[2] for row in FEATURE_INFO}
    selected_audit = selected_audit.assign(group=selected_audit.feature.map(groups), meaning=selected_audit.feature.map(meanings))
    fit = fits.loc[fits.recipe.eq(recipe) & fits.fit_snapshot.eq(snapshot)].iloc[0]
    st.caption(f"Training rows: {fit.training_rows:,} · Outcomes known through {fit.trained_through_return_month:%b %Y} · Selected inputs: {fit.n_features}")
    st.markdown("#### Selected features at this fit")
    chosen = selected_audit.loc[selected_audit.selected, ["feature", "group", "meaning", "rank", "score"]]
    st.dataframe(chosen.style.format({"score": "{:.4g}"}), use_container_width=True, hide_index=True)
    if settings["selection"] == "random_forest":
        rows = importance_table(study, model_id, recipe, snapshot)
        st.markdown("#### Random forest feature importance")
        figure = go.Figure(go.Bar(x=rows.score, y=rows.feature, orientation="h",
            marker_color=np.where(rows.selected, "#277d68", "#c8cfcc"),
            error_x=dict(type="data", array=rows.importance_std.fillna(0), visible=True),
            customdata=np.column_stack([rows.meaning.fillna("Meaning not recorded"), rows.group.fillna("Other"),
                np.where(rows.selected, "Selected", "Not selected"), rows.importance_std.fillna(0)]),
            hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>Category: %{customdata[1]}"
                "<br>Importance %{x:.4f}<br>Across-tree SD %{customdata[3]:.4f}<br>%{customdata[2]}<extra></extra>"))
        figure.update_layout(template="plotly_white", height=650, yaxis=dict(autorange="reversed"),
            xaxis_title="Mean decrease in impurity", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(figure, use_container_width=True, key=f"{key_prefix}_forest_importance")
        st.caption("Hover over a feature's bar to read the meaning of its abbreviated name on the left. "
                   "The supplied values are rank-like inputs; for example, a yield rank of 0.8 is not an 80% cash-flow yield.")
        st.caption(f"Green: selected {int(rows.selected.sum())} inputs. Gray: excluded. Error bars: standard deviation across trees, not confidence intervals. Training split importance does not establish future predictive value or the direction of an effect.")
        if "ranking_informative" in rows and not rows.ranking_informative.all():
            st.warning("This forest ranking is uninformative. Saved selection used the deterministic input-order fallback; the bars provide no positive importance evidence.")
        with st.expander("Selector forest settings (separate from the prediction model)"):
            st.json(json.loads(fit.get("selector_parameters", "{}")))
    with st.expander("All candidates: rankings and exclusion reasons"):
        columns = ["feature", "group", "meaning", "score", "rank", "selected", "reason", "correlated_with", "absolute_correlation"]
        columns += [name for name in ["score_type", "importance_std"] if name in selected_audit]
        st.dataframe(selected_audit[columns].style.format({"score": "{:.4g}", "absolute_correlation": "{:.3f}"}), use_container_width=True, hide_index=True)
    st.markdown("#### Selected-feature overlap between methods")
    st.dataframe(selection_overlap(study, model_id, snapshot), use_container_width=True)
    st.caption("Each cell counts shared selected factors at this origin; diagonals show the actual selection count. Score magnitudes have different units across methods.")
    st.markdown("#### Selection through time")
    history = audit.pivot(index="feature", columns="fit_snapshot", values="selected").astype(int)
    history = history.loc[history.sum(axis=1).sort_values(ascending=False, kind="stable").index]
    figure = go.Figure(go.Heatmap(x=[f"{date:%b %Y}" for date in history.columns], y=history.index,
        z=history.to_numpy(), text=np.where(history.to_numpy() == 1, "Selected", "Not selected"),
        colorscale=[[0, "#f1f3f2"], [1, "#287d68"]], zmin=0, zmax=1, showscale=False,
        hovertemplate="%{y}<br>Fit %{x}<br>%{text}<extra></extra>"))
    figure.update_layout(template="plotly_white", height=540, xaxis_title="Historical fit snapshot", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(figure, use_container_width=True, key=f"{key_prefix}_selection_history")
    st.caption("At most 10 inputs are used at each fit. The chosen names can change annually; their union over the whole validation period may exceed 10. A frozen submission fit would select once before its evaluation period.")
    with st.expander("Training-only factor correlation matrix"):
        correlation = json.loads(fit.training_correlations)
        figure = go.Figure(go.Heatmap(z=correlation["values"], x=correlation["columns"], y=correlation["columns"],
            colorscale="RdBu_r", zmin=-1, zmax=1, hovertemplate="%{x}<br>%{y}<br>Correlation %{z:.3f}<extra></extra>"))
        figure.update_layout(height=670, template="plotly_white", margin=dict(l=10, r=10, t=10, b=150), xaxis_tickangle=-45)
        st.plotly_chart(figure, use_container_width=True, key=f"{key_prefix}_correlations")
    st.download_button("Download selected-feature audit", feature_audit(study["fits"]).to_csv(index=False), file_name="feature_selection_audit.csv", mime="text/csv", key=f"{key_prefix}_audit_download")


def render_feature_effects(study, model_id, selected_strategies, key_prefix, include_registry=True):
    st.subheader("What changed after retraining the ML models?")
    model_labels = {key: spec.label for key, spec in default_model_registry().items()}
    model_label = model_labels[model_id]
    st.write(f"Prediction model shown: **{model_label}**. Use the model selector above or beside the portfolio results to compare another predictor.")
    st.write("Each recipe has its own independently refitted model at each historical origin. The estimator settings, forecast dates and probability-to-weight rule stay fixed. Compare the saved predictions and ML-adjusted portfolios against the same model using all 20 factors.")
    st.caption("Feature selection applies to the ML component only. Non-ML strategies have no model inputs to select; their unchanged returns appear only as a reference curve.")
    base_config = study["metadata"]["config"]
    registry = default_registry()
    with st.expander("Saved retraining evidence"):
        fits = study["fits"].loc[study["fits"].model_id.eq(model_id)]
        rows = [{"Feature recipe": study["metadata"]["recipes"][recipe]["label"],
                 "Saved run": run_id, "Historical fits": int(fits.recipe.eq(recipe).sum()),
                 "Inputs per fit": ", ".join(str(n) for n in sorted(fits.loc[fits.recipe.eq(recipe), "n_features"].unique()))}
                for recipe, run_id in study["metadata"]["runs"].items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    predictive = prediction_table(study, model_id)
    st.markdown("**Prediction quality — lower Brier score and log loss are better.**")
    table = predictive[["model_id", "recipe", "brier_score", "log_loss", "roc_auc", "Brier change vs all 20", "Log-loss change vs all 20"]]
    recipe_names = {key: value["label"] for key, value in study["metadata"]["recipes"].items()}
    table = table.assign(model_id=table.model_id.map(model_labels), recipe=table.recipe.map(recipe_names)).rename(columns={"model_id": "Prediction model", "recipe": "Feature recipe"})
    st.dataframe(table.style.format({key: "{:.6f}" for key in table if key not in {"Prediction model", "Feature recipe"}}), use_container_width=True, hide_index=True)
    control_id = event_fields(base_config.get("prediction_target", "loss"))["control_id"]
    control = study["predictive_comparison"].loc[lambda rows: rows.model_id.eq(control_id)].iloc[0]
    st.caption(f"The common historical event-frequency control has Brier score {control.brier_score:.6f} and log loss {control.log_loss:.6f}. A gain over all 20 factors does not necessarily beat this control.")
    if "forest_10" in study["metadata"]["runs"]:
        st.info(forest_findings(study))
    with st.expander("Compare probability calibration"):
        calibration = calibration_table(study, model_id)
        fields = event_fields(base_config.get("prediction_target", "loss"))
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect calibration", line=dict(color="#999999", dash="dot")))
        for recipe in study["metadata"]["runs"]:
            bins = calibration.loc[calibration.recipe.eq(recipe)]
            figure.add_trace(go.Scatter(x=bins[fields["mean_predicted"]], y=bins[fields["observed"]],
                mode="lines+markers", name=recipe_names[recipe], line=dict(color=RECIPE_COLORS.get(recipe)),
                customdata=bins.predictions, hovertemplate="Predicted %{x:.1%}<br>Observed %{y:.1%}<br>Rows %{customdata}<extra>%{fullData.name}</extra>"))
        figure.update_layout(template="plotly_white", height=430, xaxis_title="Mean predicted event probability",
            yaxis_title="Observed event frequency", xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
        st.plotly_chart(figure, use_container_width=True, key=f"{key_prefix}_calibration")
        st.caption("Saved later-month predictions grouped into probability bins. Points near the diagonal are better calibrated; small bins can be noisy.")
    if not include_registry:
        st.info("Portfolio effects for the main families are shown in the advanced-book comparison above. "
                "Turn off 'Show only the four main portfolio families' to inspect all saved basic-strategy portfolios too.")
        return
    requested = list(base_config["strategies"] if selected_strategies is None else selected_strategies)
    choices = {registry[key].label: key for key in requested}
    if not choices:
        st.info("Select training strategies in the sidebar to review feature effects.")
        return
    selection_key = f"{key_prefix}_strategies"
    if selection_key in st.session_state:
        previous = st.session_state[selection_key]
        filtered = [name for name in previous if name in choices]
        if previous and not filtered:
            st.session_state.pop(selection_key, None)
        else:
            st.session_state[selection_key] = filtered
    selected = st.multiselect("ML-adjusted strategies in feature comparison", list(choices), default=list(choices), key=selection_key)
    wanted = [choices[name] for name in selected]
    supported = [key for key in wanted if key in base_config["strategies"]]
    unavailable = [registry[key].label for key in wanted if key not in base_config["strategies"]]
    if unavailable:
        st.info("No saved feature comparison yet for: " + ", ".join(unavailable) + ". The existing study predates these rules; no experiment will run automatically.")
    if not supported:
        st.info("Choose a strategy covered by the saved study to see matched feature results.")
        return
    strategies = {registry[key].label: key for key in supported}
    detail_key = f"{key_prefix}_study_strategy"
    if st.session_state.get(detail_key) not in strategies:
        st.session_state.pop(detail_key, None)
    strategy = strategies[st.selectbox("Portfolio curve to inspect", list(strategies), key=detail_key)]
    st.markdown("**ML-adjusted portfolio results**")
    model_options = [label for key, label in model_labels.items() if key in base_config["models"]]
    portfolio_model_key = f"{key_prefix}_portfolio_model"
    st.session_state[portfolio_model_key] = model_label
    st.selectbox("Prediction model for portfolio comparison", model_options,
        key=portfolio_model_key, on_change=_sync_feature_prediction_model,
        args=(portfolio_model_key, f"{key_prefix}_study_model"),
        help="Switch among the saved basic models. This also updates prediction scores, calibration, feature inspection and portfolio curves. Logistic regression is first.")
    st.caption(f"Within each strategy, compare {model_label} independently refitted for each feature recipe. Estimator settings and trading rules stay fixed. Feature recipe names describe input selection: Random forest top 10 selects inputs for the prediction model shown here.")
    portfolio = portfolio_table(study, model_id, supported)
    columns = ["label", "model_id", "recipe", "cagr", "annualized_log_growth", "sharpe", "sortino", "max_drawdown", "change_cagr_vs_all20", "change_sharpe_vs_all20"]
    display_portfolio = portfolio[columns].assign(model_id=portfolio.model_id.map(model_labels), recipe=portfolio.recipe.map(recipe_names)).rename(columns={"model_id": "Prediction model", "recipe": "Feature recipe", "label": "Strategy", "annualized_log_growth": "Annual log growth"})
    st.dataframe(display_portfolio.style.format({"cagr": "{:.2%}", "Annual log growth": "{:.2%}", "sharpe": "{:.3f}", "sortino": "{:.3f}",
        "max_drawdown": "{:.2%}", "change_cagr_vs_all20": "{:+.2%}", "change_sharpe_vs_all20": "{:+.3f}"}), use_container_width=True, hide_index=True)
    measure = st.radio("Feature-study return view", ["Growth of $1", "Cumulative log return"], horizontal=True, key=f"{key_prefix}_return_view")
    st.plotly_chart(study_figure(study, strategy, model_id, measure == "Cumulative log return"), use_container_width=True, key=f"{key_prefix}_curves")
    st.caption("The bottom panel compares each selected-feature ML portfolio with the same model using all 20 factors. The no-ML original is shown once.")
    st.download_button("Download feature-study portfolio comparison", study["portfolio_comparison"].to_csv(index=False), file_name="feature_selection_portfolios.csv", mime="text/csv", key=f"{key_prefix}_portfolio_download")
