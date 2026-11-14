"""Training-holdout comparison: no ML, fixed ML, and tuned ML."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from research.models import default_model_registry
from research.strategies import default_registry
from research.tuning_holdout import VERSIONS, list_tuning_holdouts, load_tuning_holdout
from strategy_groups import family_for
from portfolio_selection import registry_cases
from portfolio_selection_ui import select_portfolio_case


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
    st.info("**Training data only.** These are frozen holdout comparisons inside the training years, separate from the official 2010–2018 test.")
    st.write("Tune the up/down prediction model first, then see how its probabilities change a fixed starting portfolio. "
             "The search minimizes prediction log loss; it does not optimize the portfolio's Sharpe or return.")
    studies = list_tuning_holdouts(store.root) if published is None else published.list_studies()
    saved_rules = {key for item in studies for key in item["metadata"]["strategies"]}
    st.caption(f"{len(saved_rules)} basic strategies have saved tuning comparisons. All are available below by default; "
               "the main-portfolio filter on other tabs does not restrict this chooser.")
    if not studies:
        st.info('No saved tuning comparison. Fit with labels known through December 2006, then score '
                'January 2007–December 2009. Viewing this page does not train a model.')
        return
    with st.expander("What is being tuned? And does equal weight plus ML make sense?"):
        st.write("The grid search chooses classifier settings, such as logistic regularization or tree depth, "
                 "on chronological folds of early training data. Up and down are separate prediction targets. "
                 "It does not tune the stock-selection rule, momentum window, grouping method or trading threshold.")
        st.write("Yes: equal weight is a useful starting portfolio and a clean benchmark. Its original stock weights are 1/N. "
                 "Multiplying those weights by stock-specific ML signals produces an ML-tilted portfolio, which is no longer equally weighted. "
                 "Keep the untouched equal-weight row beside it to measure what the overlay actually changes.")
        st.caption("Example: two stocks start at 50% each. Loss probabilities of 80% and 20%, at strength 0.5, "
                   "give raw weights 30% and 45%; normalizing the fully invested rule gives 40% and 60%. "
                   "The original benchmark stays 50%/50%.")
        st.write("For a fully invested strategy, the downside reductions are normalized back to 100% stocks. "
                 "They change relative weights rather than market exposure. For the explicit cash-capable strategy, reductions stay in cash. "
                 "If every held stock receives the same multiplier, the normalized portfolio stays unchanged.")
    _grouping_tuning_notes()
    names = default_model_registry()
    rules = default_registry()
    cases=pd.concat([registry_cases(item['metadata']['strategies'], item['metadata']['model_keys'],
        (item['metadata'].get('prediction_target','loss'),), study_id=str(item['directory']))
        for item in studies],ignore_index=True)
    choice=select_portfolio_case(cases,'tuning_selection',keys={
        'model':'tuning_holdout_model','allocation':'tuning_holdout_strategy','target':'tuning_holdout_target'})
    if choice.empty:
        return
    row=choice.iloc[0]
    model_key,strategy,target=row.model_key,row.strategy,row.target
    strategy_label=rules[strategy].label
    st.caption(f"{family_for(strategy).title} · {rules[strategy].description}")
    event = "next down" if target == "loss" else "next up"
    st.caption("Tuning objective: minimize prediction log loss on chronological cross-validation folds. "
               "The strategy, feature list and trading thresholds are held fixed.")
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
                   "Grouped K-means/Ward portfolios have separate untuned evidence in Grouping With ML; "
                   "they have not been replayed using these basic-study tuning forecasts. Feature engineering evaluates its own recipe-specific tuned models.")
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
    with st.expander("Compare every saved strategy on this holdout", expanded=True):
        metric = st.selectbox("Portfolio comparison metric", list(PORTFOLIO_METRICS), key="tuning_comparison_metric")
        field, number_format = PORTFOLIO_METRICS[metric]
        wide = _wide_metric(study["metrics"], model_key, field)
        st.dataframe(wide.style.format(number_format), use_container_width=True)
        st.caption("The same fixed/tuned probability forecasts are reused across these rules. This is a comparison, "
                   "not a strategy-ranking objective for the grid search. Changing a choice never reruns training.")
    st.info("Final-test status: these basic-strategy tuning winners are not automatically applied to a final study. "
            "The earlier grouped final study uses fixed classifiers; the Feature engineering extension carries its own tuning winners. "
            "Selecting a strategy here does not change either saved evaluation.")


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


def _wide_metric(metrics, model_key, metric):
    rows = metrics.loc[metrics.model_key.isin(["none", model_key]), ["label", "version", metric]]
    wide = rows.pivot(index="label", columns="version", values=metric)
    return wide.reindex(columns=list(VERSIONS)).rename_axis("Strategy")


def _wide_sharpe(metrics, model_key):
    return _wide_metric(metrics, model_key, "sharpe")


PORTFOLIO_METRICS = {
    "Sharpe": ("sharpe", "{:.3f}"),
    "CAGR": ("cagr", "{:.2%}"),
    "Annual log growth": ("annualized_log_growth", "{:.2%}"),
    "Volatility": ("annualized_volatility", "{:.2%}"),
    "Max drawdown": ("max_drawdown", "{:.2%}"),
    "Turnover": ("avg_one_way_turnover", "{:.2%}"),
}


def _grouping_tuning_notes():
    with st.expander("Can we tune K-means and hierarchical grouping too?"):
        st.write("Yes. Cluster count, trailing-history length and the clustering method are design choices "
                 "we can compare using training data. A silhouette score measures compactness/separation without future labels; "
                 "we can also check whether groups remain similar under small input changes. Neither proves the groups make a profitable portfolio.")
        st.write("For a trading application, keep the momentum/cash rule fixed and evaluate each declared grouping recipe on later "
                 "training months, including turnover and cash. Choose a recipe before examining official-test results. "
                 "Optimizing portfolio returns is a separate strategy search, not the log-loss classifier search shown here.")
        st.caption("Current coverage: K-means and Ward use five groups and a 12-month history; no clustering hyperparameter search "
                   "has been run for stock grouping. Grouped portfolios are absent from this basic-strategy tuning holdout. "
                   "Feature engineering separately tunes factor recipes/classifiers and applies them to K-means/Ward starting portfolios.")
        st.markdown("[scikit-learn: evaluating clustering without labels](https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient)")
