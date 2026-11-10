"""Training-holdout comparison: no ML, fixed ML, and tuned ML."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from research.models import default_model_registry
from research.strategies import default_registry
from research.tuning_holdout import VERSIONS, list_tuning_holdouts, load_tuning_holdout
from portfolio_focus import main_portfolio_focus


COLORS = {"No ML": "#68716e", "ML, not tuned": "#536da6", "Tuned ML": "#287d68"}
TARGETS = {
    "Next down — negative next-month return": "loss",
    "Next up — positive next-month return": "upside",
}
METRIC_HELP = {
    "Brier score": (
        "Mean squared error of the event probability: average (p − y)², where y is 1 "
        "when the selected next-up/down event happens and 0 otherwise. Lower is better; "
        "0 is perfect. Predicting 50% for every observation gives 0.25. Compare with the "
        "historical-rate control to judge improvement over a simple forecast."),
    "Log loss": (
        "Average −[y × ln(p) + (1 − y) × ln(1 − p)] for the selected next-up/down event. "
        "Lower is better; 0 is perfect. Confident wrong predictions are penalized heavily. "
        "A constant 50% forecast gives about 0.693. This is the score minimized during "
        "chronological cross-validation; this table reports a separate later holdout."),
    "ROC AUC": (
        "How well probabilities rank event cases above non-event cases, across all "
        "thresholds. 0.5 is chance-level ranking; 1 is perfect; higher is better. "
        "This is neither classification accuracy nor a test of probability calibration. "
        "It does not measure portfolio return."),
}


def render_tuning_holdout(store=None, published=None):
    st.header("Tuned model")
    main_only = main_portfolio_focus('tuning')
    st.info("**Training data only.** These are frozen holdout comparisons inside the training years, separate from the official 2010–2018 test.")
    if main_only:
        st.caption("Among the four main portfolio families, tuning is saved for the equal-weight reference only. "
                   "The grouped/top-10 momentum families have not been tuned. Turn the focus filter off to inspect all saved basic-strategy holdouts.")
    st.write("Choose a prediction model, a basic trading strategy, and the next-month event "
             "to predict. Compare the model before and after hyperparameter tuning.")
    names = default_model_registry()
    rules = default_registry()
    controls = st.columns(3)
    model_labels = {spec.label: key for key, spec in names.items()}
    model_key = model_labels[controls[0].selectbox(
        "1. Model", list(model_labels), key="tuning_holdout_model",
        help="Compare this model's original settings with its chronological grid-search winner.")]
    strategies = {spec.label: key for key, spec in rules.items() if not main_only or key == 'equal_weight_universe'}
    if st.session_state.get('tuning_holdout_strategy') not in strategies:
        st.session_state.pop('tuning_holdout_strategy', None)
    strategy_label = controls[1].selectbox(
        "2. Basic strategy", list(strategies), key="tuning_holdout_strategy",
        help="The trading rule stays fixed. Its weights are adjusted using the chosen model's probabilities.")
    strategy = strategies[strategy_label]
    target = TARGETS[controls[2].selectbox(
        "3. Prediction target", list(TARGETS), key="tuning_holdout_target",
        help="Choose what the model predicts. Settings minimize log loss for this event, "
             "rather than maximizing portfolio return. Up and down use separately trained models; "
             "a zero return counts as neither event.")]
    event = "next down" if target == "loss" else "next up"
    st.caption("Tuning objective: minimize prediction log loss on chronological cross-validation folds. "
               "The strategy, feature list and trading thresholds are held fixed.")
    studies = list_tuning_holdouts(store.root) if published is None else published.list_studies()
    compatible = [item for item in studies
                  if item["metadata"].get("prediction_target", "loss") == target
                  and model_key in item["metadata"]["model_keys"]
                  and strategy in item["metadata"]["strategies"]]
    if not compatible:
        st.info("No saved tuning comparison for these choices. Fit and tune with labels known "
                "through December 2006, then score January 2007–December 2009. "
                "The 2010–2018 test file stays on the Final test tab. "
                "Browsing does not train models.")
        st.code(f"python3 run_tuning_holdout.py --target {target} --models {model_key}"
                + (" --allow-slow" if model_key == "random_forest" else ""), language="bash")
        return
    compatible.sort(key=lambda item: item["metadata"]["created_utc"], reverse=True)
    labels = {f"{item['metadata']['feature_source'].get('recipe', 'fixed list')} · "
              f"{item['metadata']['created_utc'][:16]} · {item['directory'].name}": item
              for item in compatible}
    # A target/model change must not leave an incompatible saved-study selection.
    if st.session_state.get("tuning_holdout_study") not in labels:
        st.session_state["tuning_holdout_study"] = next(iter(labels))
    details = st.expander("Feature inputs and saved comparison")
    with details:
        chosen = labels[st.selectbox("Saved tuning comparison", list(labels), key="tuning_holdout_study")]
    cache_key = str(chosen["directory"]) if published is None else (published.token, chosen['directory'].name)
    cached = st.session_state.get("loaded_tuning_holdout")
    if cached is None or cached[0] != cache_key:
        cached = (cache_key, load_tuning_holdout(chosen["directory"]) if published is None else published.study(chosen))
        st.session_state["loaded_tuning_holdout"] = cached
    study = cached[1]
    meta = study["metadata"]
    with details:
        source = meta.get("feature_source", {})
        st.markdown(f"**Feature recipe: {source.get('recipe', 'fixed list')}** · "
                    + ", ".join(meta["feature_columns"]))
        st.write("These inputs are fixed for both versions. Changing the Feature selection "
                 "tab does not retrain this saved comparison.")
        if source.get("recipe") in {"anova_10", "pruned_10"}:
            st.write("This feature list was originally selected for downside prediction "
                     "and is reused for both targets here.")
        if source.get("fit_snapshot"):
            st.caption(f"Selected at {source['fit_snapshot']}, using outcomes known through "
                       f"{source.get('trained_through_return_month', 'that date')}. "
                       "The earlier feature study also scored 2007–2009, so the choice of ANOVA "
                       "versus pruning was not blind to this holdout. The hyperparameter search was.")
        st.caption("One pooled model per version scores every stock; there is no per-stock tuning here. "
                   "The grouped positive-volatility strategy belongs to the separate Advanced ML study.")
    st.caption(f"Inputs: {source.get('recipe', 'fixed list')} ({len(meta['feature_columns'])} features). "
               f"Fit/tune: labels known through {meta['trained_through_return_month']}. "
               f"Training-period holdout: {meta['holdout_first_return_month']} to {meta['holdout_last_return_month']}. "
               "Both models stay frozen during this holdout. This study does not read the official test CSV.")
    strength = meta.get("adjustment_strength", .5)
    if target == "loss":
        adjustment = (f"Both ML versions multiply original weights by "
                      f"(1 − {strength:g} × probability of a negative return). ")
        cash_policy = ("Removed allocation stays in cash." if rules[strategy].allows_cash
                       else "Adjusted weights rescale to 100% stocks, changing relative allocations.")
    else:
        threshold = meta.get("upside_threshold", .6)
        adjustment = (f"Both ML versions boost a stock's original weight by {strength:.0%} "
                      f"when its positive-return probability is at least {threshold:.0%}, "
                      "then rescale within the original equity budget. ")
        cash_policy = "Original cash is preserved; stocks excluded by the basic rule stay excluded."
    st.info(f"{names[model_key].label} · {event} · {strategy_label}. " + adjustment + cash_policy)

    st.subheader(f"Prediction quality — {names[model_key].label}, {event}")
    _predictive(study["predictive"], model_key)
    st.caption("Scores cover all holdout stock-months, independently of the chosen trading strategy. "
               "Changing the strategy changes the portfolio below, not these prediction scores.")
    chosen_params = meta["chosen"][model_key]
    settings = _parameter_table(model_key, chosen_params)
    if settings["Changed"].eq("No").all():
        st.info("The search selected the model's existing settings. With the same inputs and training "
                "rows, tuned and untuned predictions and portfolios are identical. Tuning does not "
                "guarantee an improvement.")
    with st.expander("What tuning changed — settings and search trials"):
        st.dataframe(settings, use_container_width=True, hide_index=True)
        st.caption(f"Best cross-validation log loss: {-chosen_params['best_score']:.6f}; "
                   f"{chosen_params['n_fits']} search/refit fits. "
                   "Holdout scores above evaluate later months and can be worse.")
        trials = (pd.read_parquet(chosen["directory"] / model_key / "trials.parquet") if published is None
                  else published.trials(chosen, model_key))
        st.dataframe(trials, use_container_width=True, hide_index=True)
        st.caption("Higher mean test score (negative log loss) is better. Here 'test' means a "
                   "cross-validation fold, not the official test period or the later holdout.")

    st.subheader(f"Portfolio results — {strategy_label}")
    st.caption("No ML = original rule. ML, not tuned = original classifier settings. "
               "Tuned ML = winning classifier settings. The trading rule is identical in all three.")
    table = _version_table(study["metrics"], strategy, model_key)
    formats = {column: "{:.3f}" if column == "Sharpe" else "{:.2%}" for column in table.columns if column != "Version"}
    st.dataframe(table.style.format(formats), use_container_width=True, hide_index=True)
    title = f"Training holdout · {meta['holdout_first_return_month']} to {meta['holdout_last_return_month']}"
    st.plotly_chart(_figure(study["monthly"], strategy, model_key, title), use_container_width=True, key="tuning_holdout_chart")
    if not main_only:
        with st.expander("All basic strategies on this holdout"):
            wide = _wide_sharpe(study["metrics"], model_key)
            st.dataframe(wide.style.format("{:.3f}"), use_container_width=True)


def _parameter_table(model_key, chosen):
    defaults = default_model_registry()[model_key].factory().get_params()
    searched = {**chosen.get("fixed_params", {}), **chosen["params"]}
    return pd.DataFrame([{
        "Parameter": key.removeprefix("model__"),
        "Before tuning": str(defaults[key.removeprefix("model__")]),
        "After tuning": str(value),
        "Changed": "No" if defaults[key.removeprefix("model__")] == value else "Yes",
    } for key, value in searched.items()])


def _version_table(metrics, strategy, model_key):
    rows = metrics.loc[metrics.strategy.eq(strategy) & metrics.model_key.isin(["none", model_key])].copy()
    rows["version"] = pd.Categorical(rows.version, VERSIONS, ordered=True)
    rows = rows.sort_values("version")
    show = rows[["version", "cagr", "annualized_log_growth", "annualized_volatility", "sharpe", "max_drawdown", "avg_one_way_turnover"]]
    return show.rename(columns={
        "version": "Version", "cagr": "CAGR", "annualized_log_growth": "Annual log growth",
        "annualized_volatility": "Volatility", "sharpe": "Sharpe", "max_drawdown": "Max drawdown",
        "avg_one_way_turnover": "Turnover"})


def _figure(monthly, strategy, model_key, title):
    frame = monthly.loc[monthly.strategy.eq(strategy) & monthly.model_key.isin(["none", model_key])].copy()
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.08, row_heights=[.7, .3])
    for version in VERSIONS:
        rows = frame.loc[frame.version.eq(version)].sort_values("return_month")
        if rows.empty:
            continue
        x = [rows.snapshot_date.iloc[0], *rows.return_month]
        figure.add_trace(go.Scatter(x=x, y=[1, *rows.wealth], name=version, line=dict(color=COLORS[version])), row=1, col=1)
        figure.add_trace(go.Scatter(x=x, y=[0, *rows.drawdown], name=version, showlegend=False,
                                     line=dict(color=COLORS[version])), row=2, col=1)
    figure.update_yaxes(title_text="Growth of $1", row=1, col=1)
    figure.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    figure.update_layout(template="plotly_white", height=640, hovermode="x unified",
        legend=dict(orientation="h", y=-.12), margin=dict(l=10, r=10, t=20, b=10),
        title=title)
    return figure


def _predictive_table(predictive, model_key):
    rows = predictive.loc[predictive.model_key.eq(model_key)].copy()
    control_mask = rows.model.str.startswith("Historical ")
    controls = rows.loc[control_mask]
    # A control uses the same frozen historical event rate for both versions.
    # Collapse it only after verifying that the scored observations/results agree.
    columns = ["model", "rows", "brier_score", "log_loss", "roc_auc"]
    if not controls.empty:
        first = controls.iloc[[0]][columns].reset_index(drop=True)
        for index in range(1, len(controls)):
            pd.testing.assert_frame_equal(first, controls.iloc[[index]][columns].reset_index(drop=True))
    show = pd.concat([rows.loc[~control_mask], controls.iloc[:1]], ignore_index=True)
    show.loc[show.model.str.startswith("Historical "), "version"] = "Reference (no features)"
    return show[["version", "model", "brier_score", "log_loss", "roc_auc"]].rename(columns={
        "version": "Version", "model": "Predictor", "brier_score": "Brier score",
        "log_loss": "Log loss", "roc_auc": "ROC AUC"})


def _predictive(predictive, model_key):
    st.caption("Hover over a metric's column header for its definition. "
               "Lower Brier score and log loss are better; higher ROC AUC is better.")
    st.dataframe(_predictive_table(predictive, model_key),
                 column_config={name: st.column_config.NumberColumn(
                     name, help=help_text, format="%.4f" if name == "ROC AUC" else "%.6f")
                     for name, help_text in METRIC_HELP.items()},
                 use_container_width=True, hide_index=True)
    st.caption("Historical-rate control predicts the same training event frequency for every stock. "
               "It uses no factors and is shown once because hyperparameter tuning does not change it. "
               "These prediction scores are separate from portfolio performance.")


def _wide_sharpe(metrics, model_key):
    rows = metrics.loc[metrics.model_key.isin(["none", model_key]), ["label", "version", "sharpe"]]
    wide = rows.pivot(index="label", columns="version", values="sharpe")
    return wide.reindex(columns=list(VERSIONS)).rename_axis("Strategy")
