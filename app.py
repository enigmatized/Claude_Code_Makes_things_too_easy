"""Research or published-results UI. Portfolio calculations live in research/."""
import os
from dataclasses import replace
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots
import streamlit as st
from research import FeatureSpec, RunConfig, default_model_registry, default_registry
from research.models import event_fields
from basic_ml_ui import saved_basic_source, render_scope_comparison
from research.legacy import FEATURES
from research.store import ExperimentStore
from feature_study_ui import render_feature_studies
from tuning_ui import render_tuning_holdout
from basket_ui import render_basket_strategy
from final_book_ui import render_final_books
from momentum_study_ui import render_momentum_studies
from evaluation_ui import render_scope, render_test_results
from research.evaluation import evaluation_scope
from result_views import select_result_view, pinned_final_summary
from strategy_groups import grouped_strategies
from strategy_results_ui import render_metric_groups, render_paired_groups
from app_mode import app_mode, published_root
from app_charts import (allocation_composition, allocation_heatmap, allocation_history_csv,
    allocation_matrices, asset_weight_lines, case_label, case_rows, line_dash, paired_portfolio_keys,
    RETURN_MEASURES, return_curve)

REGISTRY = default_registry()
MODELS = default_model_registry()
EVALUATIONS = {"training_validation": "Training validation · 2005–2009", "test": "Previously viewed test · 2010–2018"}


def run_experiment(config, artifact_root):
    return ExperimentStore(artifact_root).get_or_run(config)


def explanation(spec, include_ml=True, prediction_target="loss"):
    st.write(spec.description)
    if spec.explanation:
        for field, label in [("purpose", "Purpose"), ("universe", "Stock pool"),
                ("selection", "Monthly selection"), ("weighting", "Capital allocation"),
                ("history", "History used"), ("fallback", "Fallback"), ("example", "Example")]:
            value = getattr(spec.explanation, field)
            if not include_ml or prediction_target == "upside":
                value = value.split(" The ML version")[0]
            st.markdown(f"**{label}:** {value}")
    if include_ml:
        st.caption("The upside adjustment boosts threshold-qualified held stocks and normalizes within the original equity budget; cash is preserved."
            if prediction_target == "upside" else "The ML adjustment reduces weights using predicted next-month loss probabilities. For the annual-consistency cash strategy, reductions stay in cash. Other strategies renormalize to remain fully invested.")


def strategy_guide(strategies=None, include_ml=True, key_prefix="guide"):
    names = {REGISTRY[key].label: key for key in (strategies or REGISTRY)}
    selected = st.selectbox("Strategy to explain", list(names), key=f"{key_prefix}_strategy")
    explanation(REGISTRY[names[selected]], include_ml)
    with st.expander("Terms used in the app"):
        st.markdown("**Momentum:** compounded return over the latest 12 known months. **Volatility:** variability of monthly returns. **Eligible:** enough consecutive known history. **Frozen pool:** membership selected once before evaluation. **Sharpe:** mean excess return divided by total variability. **Sortino:** mean return above a target divided by downside deviation.")
        st.write("Sharpe's risk-free return and Sortino's target are zero. Negative/positive momentum is historical performance, not a next-month prediction.")


def wealth_chart(result, keys, logarithmic=False, return_measure="Growth of $1"):
    cases = result.cases.set_index("case_id")
    adjusted_keys = [key for key in keys if cases.loc[key, "variant"] != "baseline"]
    rows = 3 if adjusted_keys else 2
    figure = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=.08,
                          row_heights=[.5, .25, .25] if rows == 3 else [.68, .32])
    variants = ["baseline", *result.config.models]
    for key in keys:
        case = cases.loc[key]
        color = qualitative.Dark24[list(REGISTRY).index(case.strategy) % 24]
        variant = "baseline" if case.variant == "baseline" else case.model_id
        line = dict(color=color, width=2.1, dash=line_dash(variant, variants))
        book = result.books[key]
        x = [book.snapshot_date.iloc[0], *book.index]
        name = case_label(result, key)
        curve = return_curve(book, return_measure)
        value_hover = "Wealth $%{y:.3f}" if return_measure == "Growth of $1" else return_measure + " %{y:.2%}"
        figure.add_trace(go.Scatter(x=curve.index, y=curve, name=name, legendgroup=key, line=line,
            hovertemplate="%{x|%b %Y}<br>" + value_hover + "<extra>%{fullData.name}</extra>"), row=1, col=1)
        figure.add_trace(go.Scatter(x=x, y=[0, *book.drawdown], name=name, legendgroup=key, showlegend=False,
            line=line, hovertemplate="%{x|%b %Y}<br>Drawdown %{y:.1%}<extra>%{fullData.name}</extra>"), row=2, col=1)
        if case.variant != "baseline":
            original = result.books[result.case_id(case.strategy)]
            figure.add_trace(go.Scatter(x=x, y=[0, *(book.wealth / original.wealth - 1)],
                name=name + " · ML advantage", showlegend=False, line=line,
                hovertemplate="%{x|%b %Y}<br>Relative wealth %{y:+.2%}<extra>%{fullData.name}</extra>"), row=3, col=1)
    if rows == 3:
        figure.add_hline(y=0, line_color="#a0a0a0", line_dash="dot", row=3, col=1)
        figure.update_yaxes(title_text="ML advantage", tickformat="+.1%", row=3, col=1)
    figure.update_layout(height=810 if rows == 3 else 630, margin=dict(l=10, r=10, t=20, b=10),
        template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=-.14))
    scope = evaluation_scope(result.config, result.manifest)
    if not adjusted_keys:
        scope["fitting"] = "Non-ML rules"
    figure.update_layout(title=f"{scope['title']} · {scope['period']} · {scope['fitting']}", margin_t=60)
    is_wealth = return_measure == "Growth of $1"
    figure.update_yaxes(title_text=return_measure, type="log" if logarithmic and is_wealth else "linear",
        tickformat=None if is_wealth else ".1%", row=1, col=1)
    figure.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    return figure


def render_comparison(result, include_ml=True, key_prefix="basic_ml", descriptions=None,
                      include_paired=True, nested=False):
    st.subheader("Strategy results")
    if include_ml:
        st.caption(f"{result.config.feature_spec.label} · Models use the same dates and trading rules.")
    available = result.cases.strategy.drop_duplicates().tolist()
    families = {"All strategy families": available}
    families.update({family.title: keys for family, keys in grouped_strategies(available)})
    if st.session_state.get(f"{key_prefix}_family", "All strategy families") not in families:
        st.session_state.pop(f"{key_prefix}_family", None)
    controls = st.columns([2, 2])
    family = controls[0].selectbox("Strategy family", list(families), key=f"{key_prefix}_family")
    layout = controls[1].radio("Results view", ["Grouped strategies", "Full tables"], horizontal=True, key=f"{key_prefix}_layout")
    strategies = families[family]
    st.caption(f"Showing {len(strategies)} strategies in {len(grouped_strategies(strategies))} families. Display filters do not run experiments.")
    if not include_ml:
        st.caption("Hover over a strategy name (ⓘ), or focus it with the keyboard, to read its trading rule.")
    render_metric_groups(result, strategies, grouped=layout == "Grouped strategies", baseline_only=not include_ml,
                         descriptions=descriptions, nested=nested)
    st.caption("CAGR is the compounded annual return. Annual log growth = 12 × mean monthly log(1 + portfolio return); CAGR = exp(annual log growth) − 1. These describe the same growth in different units.")
    st.caption("Each original appears once. Sharpe uses risk-free return zero; Sortino uses target zero. Turnover includes price drift and excludes initial funding.")
    options = {REGISTRY[key].label: key for key in strategies}
    default = strategies if key_prefix == "basic_ml" else [
        key for key in strategies if key in {"equal_weight_universe", "momentum_weighted"}]
    if f"{key_prefix}_chart_strategies" in st.session_state:
        previous = st.session_state[f"{key_prefix}_chart_strategies"]
        filtered = [name for name in previous if name in options]
        if previous and not filtered:
            st.session_state.pop(f"{key_prefix}_chart_strategies", None)
        else:
            st.session_state[f"{key_prefix}_chart_strategies"] = filtered
    st.markdown("#### Performance through time")
    select, all_button = st.columns([4, 1])
    if all_button.button("Select all strategies", key=f"{key_prefix}_select_all_chart_strategies", use_container_width=True):
        st.session_state[f"{key_prefix}_chart_strategies"] = list(options)
    labels = select.multiselect("Strategies to chart", list(options),
        default=[REGISTRY[key].label for key in (default or strategies[:3])], key=f"{key_prefix}_chart_strategies")
    if family != "All strategy families":
        st.caption("Select all includes this family's strategies. Choose All strategy families to chart across groups.")
    selected_models = model_chart_selector(result, "Models to chart", f"{key_prefix}_chart_models") if include_ml else []
    keys = paired_portfolio_keys(result, [options[name] for name in labels], selected_models)
    return_measure = st.selectbox("Return measure", RETURN_MEASURES, key=f"{key_prefix}_return_measure")
    logarithmic = st.checkbox("Logarithmic wealth scale", value=False, key=f"{key_prefix}_log_wealth",
        disabled=return_measure != "Growth of $1")
    st.caption("Monthly log return = log(1 + the portfolio's simple return). Cumulative log return adds these across months and equals log(wealth). A logarithmic wealth axis only changes the scale of the wealth chart.")
    if keys:
        st.plotly_chart(wealth_chart(result, keys, logarithmic, return_measure), use_container_width=True, key=f"{key_prefix}_wealth")
        if include_ml:
            st.caption("Each strategy has one color; line patterns identify its models. Solid = original. The bottom panel is adjusted wealth / original wealth − 1. Nearly overlapping curves mean small portfolio effects.")
    if include_ml and include_paired and not result.paired.empty:
        st.divider()
        render_paired_groups(result, strategies, grouped=layout == "Grouped strategies", key_prefix=f"{key_prefix}_paired")
    return strategies


def model_chart_selector(result, title, key):
    options = {MODELS[mid].label: mid for mid in result.prediction_batches}
    if not options:
        return []
    if key in st.session_state:
        st.session_state[key] = [name for name in st.session_state[key] if name in options]
    return [options[name] for name in st.multiselect(title, list(options), default=list(options), key=key)]


def render_allocations(result, include_ml=True, key_prefix="basic_allocations", published=None):
    st.subheader("Allocation through time")
    options = {REGISTRY[key].label: key for key in result.cases.strategy.drop_duplicates()}
    if st.session_state.get(f"{key_prefix}_strategy") not in options:
        st.session_state.pop(f"{key_prefix}_strategy", None)
    strategy = options[st.selectbox("Strategy", list(options), key=f"{key_prefix}_strategy")]
    if REGISTRY[strategy].allows_cash:
        st.caption(f"This saved run uses {result.config.cash_momentum_months}-month momentum and requires that many consecutive known returns. The historical annual pool is separate.")
    with st.expander("How this strategy works"):
        explanation(REGISTRY[strategy], include_ml, result.config.prediction_target)
    model_ids = model_chart_selector(result, "Models in allocation charts", f"{key_prefix}_models") if include_ml else []
    if published is not None:
        result = published.allocations(result, strategy, model_ids)
    matrices = allocation_matrices(result, strategy, model_ids)
    labels = {"baseline": "Original", **{mid: MODELS[mid].label for mid in model_ids}}
    metric = result.strategy_screen_keys.get(strategy)
    if metric:
        ranking = result.screens[metric]
        selected = ranking.loc[ranking.selected]
        with st.expander(f"Frozen training {metric.title()} pool"):
            st.write(f"Selected {len(selected)} stocks using returns known at {ranking.screen_asof.iloc[0]:%b %Y}. Membership stays fixed throughout evaluation.")
            columns = [c for c in ["stock_id", "training_months", "training_rank", "training_volatility", "training_sharpe", "training_sortino",
                "complete_years", "positive_years", "negative_years", "flat_years", "positive_year_fraction", "last_complete_year"] if c in ranking]
            st.dataframe(selected[columns], use_container_width=True, hide_index=True)
            if metric == "annual_consistency":
                st.caption("Only complete calendar years count; at least three are required and more than half must be positive. Empty pools hold cash.")
    if "Cash" in matrices["baseline"].columns:
        st.subheader("Cash allocation through time")
        cash = pd.DataFrame({labels.get(key, key): matrix["Cash"] for key, matrix in matrices.items()})
        st.line_chart(cash)
        st.caption("Residual capital held in cash, with an assumed 0% monthly return." + (
            " Upside boosts preserve the original cash budget." if include_ml and result.config.prediction_target == "upside" else
            " ML reductions increase cash rather than being redistributed." if include_ml else ""))
    asset_options = {str(asset): asset for asset in matrices["baseline"].columns}
    if f"{key_prefix}_assets" in st.session_state:
        st.session_state[f"{key_prefix}_assets"] = [a for a in st.session_state[f"{key_prefix}_assets"] if a in asset_options]
    assets = [asset_options[name] for name in st.multiselect("Assets to inspect (leave empty to show all)",
        list(asset_options), default=[], key=f"{key_prefix}_assets")]
    view = st.radio("Allocation view", ["Asset-by-month heatmap", "Portfolio composition"], horizontal=True, key=f"{key_prefix}_view")
    st.caption("Target weights at month-end, held over the following month. Every panel uses matching stock IDs and scales.")
    if view == "Asset-by-month heatmap":
        st.plotly_chart(allocation_heatmap(matrices, assets or None, labels), use_container_width=True, config={"scrollZoom": True}, key=f"{key_prefix}_heatmap")
        st.caption(f"Showing {len(assets) or len(asset_options)} assets. Zero means unheld. Select IDs above or zoom into rows for individual histories.")
    else:
        st.plotly_chart(allocation_composition(matrices, assets or None, labels), use_container_width=True, key=f"{key_prefix}_composition")
        st.caption("Without a selection, show 12 largest average original allocations; group the remaining positions as Other assets. Weights retain their actual values and sum to 100%.")
    if assets:
        st.subheader("Individual asset weights")
        st.plotly_chart(asset_weight_lines(matrices, assets, labels), use_container_width=True, key=f"{key_prefix}_asset_lines")
    st.download_button("Download complete allocation history", allocation_history_csv(matrices),
        file_name=f"allocation_history_{strategy}.csv", mime="text/csv", key=f"{key_prefix}_history_download")
    with st.expander("Inspect one rebalance"):
        options = {"Original": result.case_id(strategy)}
        options.update({row.model_label: row.case_id for _, row in case_rows(result, strategy, model_ids).iterrows() if row.variant != "baseline"})
        selected_case = st.selectbox("Allocation version / model", list(options),
            index=1 if len(options) > 1 else 0, key=f"{key_prefix}_case")
        dates = {f"{date:%b %Y}": date for date in matrices["baseline"].index}
        snapshot = dates[st.selectbox("Rebalance snapshot", list(dates), key=f"{key_prefix}_snapshot")]
        inspection = replace(result, prediction_batches={}) if published is not None else result
        allocation = inspection.allocation_case(options[selected_case], snapshot)
        if published is not None and include_ml:
            st.caption("Published inspection compares target weights. Individual stock probability and signal audits remain available in local research mode.")
        if result.cases.set_index("case_id").loc[options[selected_case]].get("allows_cash", False):
            st.metric("Cash at this rebalance", f"{max(0, 1 - allocation.weight.sum()):.2%}")
        columns = ["stock_id", "baseline_weight", "weight", "weight_change"] if include_ml else ["stock_id", "weight"]
        columns += [c for c in ["probability_of_loss", "probability_of_up", "prediction_source", "fallback_reason", "stock_history_months", "trailing_momentum", "trailing_volatility", "history_months",
            "ordinary_volatility", "downside_weighted_risk", "training_rank", "training_sharpe", "training_sortino",
            "screen_asof", "fit_snapshot", "trained_through_return_month", "complete_years", "positive_years",
            "negative_years", "flat_years", "positive_year_fraction"] if c in allocation]
        table = allocation.loc[allocation.weight.gt(0), columns].rename(columns={
            "baseline_weight": "Original weight", "weight": "Selected weight", "weight_change": "Weight change"})
        ratios = {"training_sharpe", "training_sortino"}
        excluded = {"stock_id", "history_months", "training_rank", "screen_asof", "fit_snapshot", "trained_through_return_month",
                    "complete_years", "positive_years", "negative_years", "flat_years", "prediction_source", "fallback_reason", "stock_history_months"}
        formats = {c: "{:.3f}" if c in ratios else "{:.2%}" for c in table if c not in excluded}
        st.dataframe(table.style.format(formats, na_rep="—"), use_container_width=True, hide_index=True)
        st.download_button("Download this rebalance", allocation.to_csv(index=False),
            file_name=f"allocation_{strategy}_{snapshot:%Y%m}.csv", mime="text/csv", key=f"{key_prefix}_snapshot_download")


def render_models(result, key_prefix="basic_models", nested=False, published=None):
    if not result.prediction_batches:
        st.info("Enable Compare original + ML and select models to see predictive diagnostics.")
        return
    st.subheader("Prediction quality across models")
    fields = event_fields(result.config.prediction_target)
    event = "loss" if result.config.prediction_target == "loss" else "positive return"
    st.write(f"Each model estimates the same next-month {event} event. Lower Brier score and log loss are better; ROC AUC measures ranking. The historical event-rate control uses only past outcomes.")
    columns = {"model": "Predictor", "rows": "Predictions", fields["observed"]: "Observed loss rate" if event == "loss" else "Observed up rate",
        "brier_score": "Brier score", "log_loss": "Log loss", "roc_auc": "ROC AUC", "accuracy_at_half": "Accuracy (p ≥ 0.5)"}
    scores = result.model_metrics[list(columns)].rename(columns=columns)
    st.dataframe(scores.style.format({"Predictions": "{:,.0f}", "Observed loss rate": "{:.2%}", "Observed up rate": "{:.2%}",
        "Brier score": "{:.4f}", "Log loss": "{:.4f}", "ROC AUC": "{:.4f}", "Accuracy (p ≥ 0.5)": "{:.2%}"}),
        use_container_width=True, hide_index=True)
    figure = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect calibration",
                                 line=dict(color="#a0a0a0", dash="dash")))
    for mid, batch in result.prediction_batches.items():
        calibration = batch.calibration
        figure.add_trace(go.Scatter(x=calibration[fields["mean_predicted"]], y=calibration[fields["observed"]],
            mode="lines+markers", name=MODELS[mid].label, text=calibration.predictions,
            hovertemplate="Predicted %{x:.1%}<br>Observed %{y:.1%}<br>%{text} predictions<extra>%{fullData.name}</extra>"))
    figure.update_layout(template="plotly_white", height=400, xaxis=dict(title=f"Predicted {event} probability", tickformat=".0%", range=[0, 1]),
                          yaxis=dict(title=f"Observed {event} frequency", tickformat=".0%", range=[0, 1]))
    st.plotly_chart(figure, use_container_width=True, key=f"{key_prefix}_calibration")
    st.caption("Calibration and portfolio results are separate checks. A better classifier does not necessarily produce a better portfolio.")
    if result.config.prediction_target == "upside":
        st.caption(f"Accuracy uses probability ≥ 50%; the portfolio boost uses the separately recorded {result.config.upside_threshold:.0%} threshold. P(up) predicts direction, not the size of the return.")
    if result.config.model_scope == "per_stock":
        rows = []
        for mid, batch in result.prediction_batches.items():
            counts = (published.prediction_sources(result, mid) if published is not None else
                      batch.predictions.prediction_source.value_counts())
            rows.append({"Model": MODELS[mid].label, "Per-stock predictions": counts.get("per_stock", 0),
                         "Pooled fallback predictions": counts.get("pooled_fallback", 0)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"Local fits require at least {result.config.min_stock_history} known months and {result.config.min_stock_class} examples in each event class. Short histories, unseen stocks and insufficient class counts use the same-origin pooled model. The predictive control remains pooled historical prevalence.")
    for mid, batch in result.prediction_batches.items():
        title = f"{MODELS[mid].label}: settings, features and refit dates"
        if nested and not st.checkbox(title, key=f"{key_prefix}_{mid}_details"):
            continue
        with (st.container(border=True) if nested else st.expander(title)):
            st.write(f"Estimator: {MODELS[mid].metadata()['estimator']} · Target: {result.manifest['model']['target']} · Scope: {result.config.model_scope}.")
            st.json(next(m for m in result.manifest["models"] if m["model_id"] == mid))
            fits = published.model_fits(result, mid) if published is not None else batch.fits
            st.dataframe(fits, use_container_width=True, hide_index=True)


def render_data(result, key_prefix="basic_data"):
    st.subheader("The data used by this run")
    audit = result.data_audit
    st.dataframe(pd.DataFrame([{"Property": key.replace("_", " ").title(), "Value": str(audit[key])} for key in [
        "source_splits", "training_source_rows", "evaluation_rows", "evaluation_stocks", "evaluation_months",
        "first_snapshot", "last_snapshot", "first_return_month", "last_return_month",
        "initial_known_label_rows", "initial_labels_available_through", "missing_input_cells"]]),
        use_container_width=True, hide_index=True)
    st.write("One row is one stock at a month-end snapshot. Its R1M_Usd return is earned over the following month. Model inputs exclude that forward return and stock IDs. Training joins labels only after their availability date.")
    st.write(f"Selected model inputs: {', '.join(result.config.feature_spec.columns)}")
    st.caption(f"Feature recipe: {result.config.feature_spec.label}. Learned selectors fit inside each historical training window; explicit manual subsets are declarations. Source extreme returns are retained.")
    st.code("Source validation → available inputs + separate labels → common fit schedule\n"
            "Strategy → original weights\n"
            "Selected features → each fixed model → event probabilities → configured weight adjustment\n"
            "Original + adjusted weights → shared portfolio accounting → comparisons", language=None)
    if result.config.fit_policy == "frozen_at_origin":
        st.write("Models fit once using outcomes known before evaluation starts. No evaluation labels enter model fitting. Lagged portfolio signals can update after returns are realized.")
    else:
        st.write(f"Research mode: models refit every {result.config.refit_months} decision months on expanding realized history. Earlier evaluation outcomes may enter later fits. This is distinct from the assignment's frozen-training test model fits.")
    st.write("Monthly long-only portfolios; gross returns; zero Sharpe risk-free return and Sortino target. The annual-consistency cash strategy holds residual capital at an assumed 0% monthly return; other rules are fully invested.")
    st.caption("The test period was inspected earlier. Feature selectors may be fitted when the recipe requests them; this stage does not search hyperparameters, fit calibration or combine models into a cross-model ensemble.")
    with st.expander("Configuration, strategy explanations and source fingerprints"):
        st.json(result.manifest)
    if st.button("Prepare run report", key=f"{key_prefix}_prepare_report"):
        with st.spinner("Preparing report…"):
            st.session_state[f"{key_prefix}_download_report"] = (result.manifest["created_utc"], result.to_zip())
    report = st.session_state.get(f"{key_prefix}_download_report")
    if report and report[0] == result.manifest["created_utc"]:
        st.download_button("Download run report (.zip)", report[1], file_name="mlfactor_run_report.zip", mime="application/zip", key=f"{key_prefix}_report_download")
    st.caption("The report includes model-specific predictions, refits, calibration, case metadata, monthly returns, data audit and fitted pools.")



def _load_training(store, run_id):
    result = store.load(run_id)
    if result.config.evaluation != "training_validation":
        raise ValueError("Training stages require training-validation results.")
    st.session_state["training_result"] = result
    st.session_state.pop("training_error", None)


def _compute_training(store, config, force=False):
    if app_mode() == "deploy":
        raise RuntimeError("Deployment displays published results; computation is disabled.")
    if config.evaluation != "training_validation":
        raise ValueError("The training controls cannot run test experiments.")
    try:
        st.session_state["training_result"] = store.get_or_run(config, force=force)
        st.session_state.pop("training_error", None)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        st.session_state["training_error"] = str(error)


def _compute_basic(store, config, force=False):
    if app_mode() == "deploy":
        raise RuntimeError("Deployment displays published results; computation is disabled.")
    if config.evaluation != "training_validation":
        raise ValueError("Basic ML controls require training validation.")
    try:
        st.session_state["basic_result"] = store.get_or_run(config, force=force)
        st.session_state.pop("basic_saved_experiment", None)
        st.session_state.pop("basic_error", None)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        st.session_state["basic_error"] = str(error)


def training_settings(store, source, strategies, include_ml, target="loss", scope="pooled"):
    """Explicit computation stays local to its teaching stage; browsing never fits."""
    prefix = "compute_ml" if include_ml else "compute_strategies"
    title = "Model settings and explicit training run" if include_ml else "Run a non-ML training comparison"
    with st.expander(title):
        st.caption("Settings affect a new requested comparison. Existing saved results stay visible until you load or compute it.")
        current_window = source.config.cash_momentum_months if source is not None else 12
        windows = sorted({2, 3, 6, 12, current_window})
        months = st.selectbox("Cash strategy momentum months", windows,
            index=windows.index(current_window),
            key=f"{prefix}_momentum")
        models, strength, policy = ("logistic",), .5, "expanding"
        if include_ml:
            names = {spec.label: key for key, spec in MODELS.items()}
            selected = st.multiselect("Models to train", list(names), default=["Logistic regression"] if scope == "per_stock" else list(names), key=f"{prefix}_models_{scope}")
            models = tuple(names[label] for label in selected)
            strength = st.slider("Weight adjustment strength", 0.0, .9, .5, .1, key=f"{prefix}_strength")
            threshold = st.slider("Upside probability threshold", .0, 1.0, .6, .05, key=f"{prefix}_threshold") if target == "upside" else .6
            if scope == "per_stock":
                st.caption("Per-stock fits multiply fitting work by the number of eligible stocks. Start with logistic regression; the existing tree leaf-size defaults were designed for pooled data. Minimum 36 historical months and six observations in each class; otherwise use the pooled fallback.")
            policies = {"Expanding historical fits": "expanding", "Fit once before validation": "frozen_at_origin"}
            policy = policies[st.selectbox("Training-validation fitting policy", list(policies), key=f"{prefix}_policy")]
            st.caption("This introductory comparison uses all 20 original factors. Feature methods are reviewed in Feature selection.")
        force = st.checkbox("Recompute even if saved", value=False, key=f"{prefix}_force")
        if not strategies or (include_ml and not models):
            st.info("Choose at least one strategy and, for prediction, one model.")
        else:
            config = RunConfig(tuple(strategies), include_ml=include_ml, models=models,
                adjustment_strength=strength, fit_policy=policy, cash_momentum_months=months,
                validation_history_months=source.config.validation_history_months if source is not None else 60,
                prediction_target=target, model_scope=scope, upside_threshold=threshold if include_ml else .6)
            st.button("Load or compute training comparison", key=f"{prefix}_run", on_click=_compute_basic if include_ml else _compute_training,
                args=(store, config, force), type="primary", use_container_width=True)


def _source_caption(result):
    info = getattr(result, "cache_info", {})
    if info:
        st.caption(f"Saved experiment {info['run_id']} · Display filters preserve the original computation and provenance.")


def render_strategies(source, strategies, store=None, published=None):
    st.header("Strategies")
    st.write("Start with the portfolio rules: which stocks to hold, how to weight them, and when to hold cash.")
    st.info("**Training data only.** This tab compares non-ML rules within the training dataset (earned returns 2000–2009). No test-period results are used here.")
    if published is None:
        training_settings(store, source, strategies, include_ml=False)
    if source is None:
        st.info("No saved training comparison is available. Load one or explicitly compute the portfolio rules above.")
        return
    if not strategies:
        st.info("Select at least one training strategy in the sidebar.")
        return
    view = select_result_view(source, baseline_only=True, strategies=strategies)
    from published_results import require_training
    require_training(view.config, view.manifest, view.monthly)
    render_scope(view.config, view.manifest, "Strategy evaluation", baseline_only=True)
    _source_caption(view)
    render_comparison(view, include_ml=False, key_prefix="strategies",
        descriptions=published.tab["descriptions"] if published is not None else None)
    render_momentum_studies(include_ml=False, key_prefix="strategy_momentum", published=published)
    if st.checkbox("Explore allocations through time", key="strategies_show_allocations"):
        render_allocations(view, include_ml=False, key_prefix="strategies_allocations", published=published)
    if st.checkbox("Read strategy explanations", key="strategies_show_guide"):
        strategy_guide(strategies, include_ml=False, key_prefix="strategies_guide")
    with st.expander("Data and timing"):
        st.write("One row is a stock at a month-end snapshot. Its return is earned over the following month. Portfolio signals use only then-known historical returns.")
        st.write("The training dataset earns Jan 2000–Dec 2009; default validation earns Jan 2005–Dec 2009 after the initial history. Returns are gross of costs. Cash earns an assumed 0%.")


def render_basic_ml(source, strategies, store, published=None):
    st.header("Basic ML")
    controls = st.columns(2)
    patterns = {"Downside protection": "loss", "Upside threshold boost": "upside"}
    scopes = {"Pooled across stocks": "pooled", "Separate model per stock": "per_stock"}
    target = patterns[controls[0].selectbox("Basic ML pattern", list(patterns), key="basic_pattern")]
    scope = scopes[controls[1].selectbox("Model training scope", list(scopes), key="basic_scope")]
    st.info("**Training data only.** These are chronological validation results within the training years; test-period returns are excluded.")
    st.write("Pooled: one model per algorithm at each historical fit, trained on all stocks; apply it to each stock's current features. Per stock: separate fits using only that stock's past, with an explicit pooled fallback for insufficient history.")
    if target == "loss":
        st.write("Estimate each stock's probability of a negative next-month return, then reduce its portfolio weight using that probability.")
        st.code("adjusted weight = original weight × (1 − strength × probability of loss)", language=None)
        st.caption("Fully invested rules normalize these weights back to 100% stock exposure. The cash-capable rule retains the reduction in cash.")
    else:
        st.write("Estimate the probability of a strictly positive next-month return. Boost held stocks that meet the probability threshold, then normalize within the original stock budget.")
        st.code("raw weight = original weight × (1 + strength × I(probability of up ≥ threshold))", language=None)
        st.caption("Default threshold 60%, strength 50%. Excluded stocks stay excluded; the original cash allocation is preserved. If all held stocks get the same multiplier, relative weights do not change. Flat returns count as neither an up move nor a loss.")
    selected = saved_basic_source(store, source, strategies, target, scope)
    if published is None:
        training_settings(store, selected or source, strategies, include_ml=True, target=target, scope=scope)
    if st.session_state.get("basic_error"):
        st.error(st.session_state["basic_error"])
    if selected is None or not selected.prediction_batches:
        if published is not None:
            st.info("No published Basic ML comparison for this prediction pattern, scope and strategy selection. Choose an available saved combination; viewing never trains a model.")
            return
        if target == "loss" and scope == "pooled":
            st.info("No downside-model predictions are saved in this training comparison. Choose a saved model comparison or explicitly compute one above.")
        else:
            st.info("No saved training comparison for this ML pattern and scope yet. Explicitly compute one above; changing these controls never fits models.")
        return
    source = selected
    from published_results import require_training
    require_training(source.config, source.manifest, source.monthly, baseline_only=False)
    if not strategies:
        st.info("Select at least one training strategy in the sidebar.")
        return
    model_ids = model_chart_selector(source, "Models to compare", "basic_models")
    if not model_ids:
        st.info("Choose at least one saved model to review.")
        return
    view = select_result_view(source, strategies=strategies, models=model_ids)
    render_scope(view.config, view.manifest, "Basic ML evaluation")
    _source_caption(view)
    st.caption(f"Recorded pattern: {source.config.prediction_target} · Scope: {source.config.model_scope} · Strength: {source.config.adjustment_strength:.0%}" + (f" · Up threshold: {source.config.upside_threshold:.0%}" if target == "upside" else ""))
    with st.expander("1. Does the model predict downside?" if target == "loss" else "1. Does the model predict an up move?", expanded=False):
        render_models(view, nested=True, published=published)
    with st.expander("2. What changes in the portfolios?", expanded=False):
        comparison_strategies = render_comparison(view, key_prefix="basic_ml", include_paired=False, nested=True)
    with st.expander("Change from each original portfolio", expanded=False):
        render_paired_groups(view, comparison_strategies,
            grouped=st.session_state.get("basic_ml_layout") == "Grouped strategies",
            key_prefix="basic_ml_paired", nested=True)
    render_scope_comparison(store, view, strategies)
    if published is None and target == "loss" and scope == "pooled":
        render_momentum_studies(include_ml=True, key_prefix="ml_momentum")
    if st.checkbox("Inspect original and adjusted allocations", key="basic_show_allocations"):
        render_allocations(view, key_prefix="basic_allocations", published=published)
    if st.checkbox("Inspect inputs, fitting and run provenance", key="basic_show_data"):
        st.caption("Full source experiment metadata and report; display filters do not rewrite the saved experiment.")
        if published is None:
            render_data(source)
        else:
            st.json(source.manifest)
            st.caption("Recorded inputs, fitting protocol and source fingerprints. Full stock-level audits remain in local research mode.")


def render_advanced(store):
    render_basket_strategy(store)


def render_features(strategies, published=None):
    st.header("Feature selection")
    render_feature_studies(selected_strategies=strategies, key_prefix="features", published=published)


def render_final_test(store):
    st.header("Final test")
    st.caption("Test-period earned returns: Jan 2010–Dec 2018. This period was inspected in earlier research.")
    st.markdown(
        "**Four books, then everything else.** Equal weight. High positive-vol names with "
        "short-term momentum weights. That momentum book with an up-or-down model on the weights. "
        "The correlation-basket book with the same up-or-down model. "
        "The other strategies and classifiers stay one control away.")
    has_books = render_final_books(store)
    final = pinned_final_summary(store)
    if final is None:
        st.info("Final frozen-model test evaluation is pending. The Tuned model tab scores a holdout inside the training years. This tab is the 2010–2018 test, after that choice is recorded.")
        if has_books:
            st.caption(
                "The table above is the frozen 2010–2018 score of the four books. "
                "The pending note is the submission release, which is a different saved run and is still unmarked. "
                "The other saved mixtures stay behind the control below.")
        else:
            st.caption(
                "Equal weight already has an archived 2010–2018 number in the saved mixtures below. "
                "The high positive-vol momentum book, the correlation-basket book, and the "
                "up-or-down overlays on both are saved on the training window, in Advanced ML. "
                "This page has no 2010–2018 row for those books.")
    else:
        final_id = final["run_id"]
        cached = st.session_state.get("final_test_result")
        if cached is None or getattr(cached, "cache_info", {}).get("run_id") != final_id:
            cached = store.load(final_id)
            st.session_state["final_test_result"] = cached
        render_scope(cached.config, cached.manifest, "Pinned final evaluation")
        show_all = st.checkbox("Show every saved model and strategy in the pinned run", key="final_show_all_mixtures")
        view = cached
        if not show_all:
            present = set(cached.cases.strategy)
            primary = [key for key in ("equal_weight_universe",) if key in present]
            if primary:
                view = select_result_view(cached, strategies=primary, evaluation="test")
        render_comparison(view, key_prefix="final_test")
        render_models(view, key_prefix="final_models")
    if not st.session_state.get('final_main_families_only', False) and st.checkbox("Show every saved test mixture", key="show_test_history"):
        render_test_results()
        test = st.session_state.get("test_result")
        if test is not None:
            if test.config.evaluation != "test":
                st.error("The exploratory test selection is not a test-period result.")
            else:
                st.subheader("Selected exploratory saved run")
                st.caption("This saved result is not the pinned final specification.")
                render_scope(test.config, test.manifest, "Exploratory test selection")
                render_comparison(test, include_ml=bool(test.prediction_batches), key_prefix="exploratory_test")


def _training_sidebar(store):
    records = [row for row in store.list_runs() if row["evaluation"] == "training_validation"]
    # Migrate an old browser session safely; a legacy test selection cannot enter training.
    legacy = st.session_state.pop("result", None)
    if legacy is not None and legacy.config.evaluation == "test" and st.session_state.get("test_result") is None:
        st.session_state["test_result"] = legacy
    if st.session_state.get("training_result") is None and legacy is not None and legacy.config.evaluation == "training_validation":
        st.session_state["training_result"] = legacy
    source = st.session_state.get("training_result")
    if source is not None and source.config.evaluation != "training_validation":
        st.session_state.pop("training_result", None)
        source = None
    if source is None and records:
        originals = [row for row in records if store.configuration(row["run_id"]).prediction_target == "loss"
                     and store.configuration(row["run_id"]).model_scope == "pooled"]
        preferred = [row for row in originals if row["feature_id"] == FeatureSpec().feature_id] or originals or records
        chosen = max(preferred, key=lambda row: (row["case_count"], row["created_utc"]))
        _load_training(store, chosen["run_id"])
        source = st.session_state["training_result"]
    with st.sidebar:
        st.subheader("Training research")
        st.caption("These selections apply to Strategies and Basic ML. Advanced ML, Tuned model, and Final test read their own saved results.")
        if records:
            options = {f"{row['case_count'] // (row['model_count'] + 1)} strategy rules · {row['created_utc'][:16]} · {row['run_id']}": row["run_id"] for row in records}
            active_id = getattr(source, "cache_info", {}).get("run_id")
            default = list(options.values()).index(active_id) if active_id in options.values() else 0
            chosen = options[st.selectbox("Saved training experiment", list(options), index=default, key="training_experiment")]
            st.button("Load training results", key="load_training", on_click=_load_training, args=(store, chosen), use_container_width=True)
        available = source.cases.strategy.drop_duplicates().tolist() if source is not None else list(REGISTRY)
        with st.expander("Choose strategy families or individual rules"):
            families = {"All strategy families": available}
            families.update({family.title: keys for family, keys in grouped_strategies(available)})
            if st.session_state.get("build_family", "All strategy families") not in families:
                st.session_state.pop("build_family", None)
            family = st.selectbox("Family for strategy selection", list(families), key="build_family")
            names = {REGISTRY[key].label: key for key in families[family]}
            previous = st.session_state.get("selected_strategies")
            if previous is not None:
                filtered = [name for name in previous if name in names]
                if previous and not filtered:
                    st.session_state.pop("selected_strategies", None)
                else:
                    st.session_state["selected_strategies"] = filtered
            if st.button("Select all in this family", key="select_all_build_family", use_container_width=True):
                st.session_state["selected_strategies"] = list(names)
            labels = st.multiselect("Training strategies to compare", list(names), default=list(names), key="selected_strategies")
        strategies = [names[label] for label in labels]
        st.caption(f"{len(strategies)} selected strategies. Display changes do not train models.")
    return source, strategies


def main():
    st.set_page_config(page_title="MLFactor · Strategy lab", page_icon="📈", layout="wide")
    st.markdown("""<style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1550px;}
        [data-testid="stSidebar"] {background-color: #f3f5f3;}
        h1 {letter-spacing: -0.04em;}
    </style>""", unsafe_allow_html=True)
    st.caption("MLFACTOR / RESEARCH WORKBENCH")
    st.title("Strategy lab")
    try:
        mode = app_mode()
    except ValueError as error:
        st.error(str(error))
        return
    if mode == "deploy":
        os.environ["MLFACTOR_READ_ONLY"] = "1"
        render_deployment()
        return
    st.caption("Research mode · Load saved experiments or explicitly run new comparisons locally.")
    st.write("Portfolio rules → downside prediction → correlation-basket momentum → feature evidence → tuned model → final evaluation.")
    store = ExperimentStore()
    try:
        source, strategies = _training_sidebar(store)
    except (ValueError, FileNotFoundError) as error:
        st.error(f"Could not load the training evidence: {error}")
        source, strategies = None, list(REGISTRY)
    if st.session_state.get("training_error"):
        st.error(st.session_state["training_error"])
    tabs = st.tabs(["Strategies", "Basic ML", "Advanced ML strategy", "Feature selection", "Tuned model", "Final test"])
    with tabs[0]: render_strategies(source, strategies, store)
    with tabs[1]: render_basic_ml(source, strategies, store)
    with tabs[2]: render_advanced(store)
    with tabs[3]: render_features(strategies)
    with tabs[4]: render_tuning_holdout(store)
    with tabs[5]: render_final_test(store)


def render_deployment():
    """Only published pages execute; unfinished pages cannot open local archives."""
    from published_results import PublishedStrategies
    from research.store import digest_file
    st.caption("View mode · Published results")
    root = published_root().resolve()
    try:
        token = (str(root), digest_file(root / "manifest.json"))
        cached = st.session_state.get("published_strategies")
        if cached is None or cached[0] != token:
            publication = PublishedStrategies(root)
            cached = (token, publication, publication.result())
            st.session_state["published_strategies"] = cached
        _, publication, source = cached
    except (ValueError, FileNotFoundError, KeyError) as error:
        st.error(f"Published Strategies results are unavailable: {error}")
        st.info("Publish a completed training comparison before starting the deployed app. No research run will be started automatically.")
        return
    with st.sidebar:
        st.subheader("Published strategies")
        st.caption("Training data only · Saved portfolio results")
        names = dict(zip(source.cases.label, source.cases.strategy))
        selection_key = "published_strategy_selection"
        if selection_key in st.session_state:
            previous = st.session_state[selection_key]
            filtered = [label for label in previous if label in names]
            if previous and not filtered:
                st.session_state.pop(selection_key, None)
            else:
                st.session_state[selection_key] = filtered
        chosen = st.multiselect("Strategies to compare", list(names), default=list(names), key="published_strategy_selection")
        strategies = [names[label] for label in chosen]
    tabs = st.tabs(["Strategies", "Basic ML", "Advanced ML strategy", "Feature selection", "Tuned model", "Final test"])
    with tabs[0]:
        try:
            render_strategies(source, strategies, published=publication)
        except (ValueError, FileNotFoundError, KeyError) as error:
            st.error(f"Published Strategies results are unavailable: {error}")
    with tabs[1]:
        try:
            basic_store = publication.basic_store()
            if basic_store.list_runs():
                render_basic_ml(None, strategies, basic_store, published=basic_store)
            else:
                st.info("This section has not been published yet. Its research tools remain available in local research mode.")
        except (ValueError, FileNotFoundError, KeyError) as error:
            st.error(f"Published Basic ML results are unavailable: {error}")
    with tabs[2]:
        try:
            baskets = publication.baskets()
            if baskets is None:
                st.info("This section has not been published yet. Its research tools remain available in local research mode.")
            else:
                render_basket_strategy(published=baskets)
        except (ValueError, FileNotFoundError, KeyError) as error:
            st.error(f"Published Advanced ML results are unavailable: {error}")
    for tab, section in zip(tabs[3:], ['features', 'tuning', 'final']):
        with tab:
            try:
                analysis = publication.analysis_section(section)
                if analysis is None:
                    st.info("This section has not been published yet. Its research tools remain available in local research mode.")
                elif section == 'features':
                    render_features(strategies, published=analysis)
                elif section == 'tuning':
                    render_tuning_holdout(published=analysis)
                else:
                    st.header('Final test')
                    st.info('**Official test: Jan 2010–Dec 2018.** This period was inspected previously. Training-only and entire-path views are separately labeled below.')
                    render_final_books(published=analysis)
                    st.caption('This saved final study uses a frozen ANOVA-10 fit with fixed classifier settings. '
                        'It is separate from the Tuned model training holdout; formal report/release sign-off remains a local project step.')
                    if not st.session_state.get('final_main_families_only', False) and st.checkbox('Show every saved test mixture', key='show_test_history'):
                        render_test_results(published=analysis)
            except (ValueError, FileNotFoundError, KeyError) as error:
                st.error(f'Published {section} results are unavailable: {error}')


if __name__ == "__main__":
    main()
