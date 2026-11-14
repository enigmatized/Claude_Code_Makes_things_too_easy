"""Feature-tab view of the four presentation books."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from research.basket_adjustments import (
    BASKET_DOWNSIDE_VERSION, BASKET_UPSIDE_VERSION, DOWNSIDE_VERSION, UPSIDE_VERSION,
)
from research.basket_features import list_feature_book_scores, load_feature_book_scores
from research.basket_momentum import TOP_VOL_VERSION
from research.models import default_model_registry
from portfolio_selection import grouped_cases
from portfolio_selection_ui import select_portfolio_case


FOCUS = (
    ("Equal weight", "Equal weight", ""),
    ("Top 10 positive-vol", TOP_VOL_VERSION, ""),
    ("Top 10 + downside to cash", DOWNSIDE_VERSION, "recipe"),
    ("Top 10 + upside tilt", UPSIDE_VERSION, "recipe"),
    ("Full baskets", "Basket momentum + cash", ""),
    ("Baskets + downside to cash", BASKET_DOWNSIDE_VERSION, "recipe"),
    ("Baskets + upside tilt", BASKET_UPSIDE_VERSION, "recipe"),
)
COLORS = {
    "": "#68716e",
    "all_20": "#68716e",
    "anova_10": "#287d68",
    "pruned_10": "#536da6",
    "forest_10": "#b36a2e",
}


def render_four_book_features(store=None, published=None, main_only=False):
    studies = list_feature_book_scores(store.root) if published is None else published.book_studies()
    st.subheader("Four books, by feature recipe")
    if not studies:
        st.caption(
            "The four-book feature comparison is not saved yet. "
            "Equal weight and the momentum books do not use the factors. "
            "The tables below are the fourteen registry rules.")
        return False
    chosen = studies[0]
    cache_key = str(chosen["directory"]) if published is None else published.token
    cached = st.session_state.get("loaded_feature_books")
    if cached is None or cached[0] != cache_key:
        cached = (cache_key, load_feature_book_scores(chosen["directory"]) if published is None else published.books())
        st.session_state["loaded_feature_books"] = cached
    study = cached[1]
    meta = study["metadata"]
    st.info("**The advanced strategy is included here.** The top-10 and full-group basket portfolios "
            "use the same four classifier types as Advanced ML, independently refitted for each feature recipe. "
            "Feature selection changes their up/down predictor inputs; K-means groups, the positive-volatility screen "
            "and the momentum/cash trading rules stay fixed.")
    st.caption("The detailed feature-ranking audit below describes downside fits. The advanced upside portfolios "
               "use separately fitted upside recipes, whose selected names can differ. Source run IDs record both directions.")
    labels = {item["label"]: item["key"] for item in meta["recipes"]}
    window = meta["window"]
    st.caption(
        f"Training window {window['first_return_month']} through {window['last_return_month']}, "
        f"{window['months']} months. Equal weight and the top-10 momentum book do not use the factors, "
        "so a feature cut cannot move them. The overlays replay saved forecasts. "
        "Browsing does not fit a model. The 2010–2018 test file was not read.")
    registry = default_model_registry()
    model_labels = {registry[key].label: key for key in registry if key in set(study["metrics"].model_key)}
    choices=grouped_cases(study['metrics'])
    choice=select_portfolio_case(choices.loc[choices.model_key.ne('none')], 'feature_books_selection',
        keys={'model':'feature_books_model'},preferred_basket='K-means top 10 positive-vol per group')
    if choice.empty:
        return True
    selected=choice.iloc[0]
    model_key=selected.model_key
    original=TOP_VOL_VERSION if selected.version in {DOWNSIDE_VERSION,UPSIDE_VERSION} else 'Basket momentum + cash'
    show_all=st.checkbox('Compare more saved books and horizons',key='feature_books_show_all')
    chosen_metrics=study['metrics']
    if not show_all:
        chosen_metrics=chosen_metrics.loc[chosen_metrics.version.isin([original,selected.version])
            & chosen_metrics.momentum_months.eq(selected.momentum_months)
            & chosen_metrics.model_key.isin(['none',model_key])]
    table = _table(chosen_metrics, meta, model_key, True)
    formats = {column: "{:.3f}" if column == "Sharpe" else "{:+.2%}" if column.startswith("CAGR vs") else "{:.2%}"
               for column in table.columns if column not in {"Book", "Feature recipe", "Model", "Momentum"}}
    st.dataframe(table.style.format(formats), use_container_width=True, hide_index=True)
    chart_version=selected.version
    chart_study={**study,'monthly':study['monthly'].loc[study['monthly'].momentum_months.isin([0,int(selected.momentum_months)])].copy()}
    st.plotly_chart(_figure(chart_study, chart_version, model_key, labels, months=int(selected.momentum_months)), use_container_width=True, key="feature_books_chart")
    st.caption("Turn off the focus filter to inspect additional basic strategies and every saved model/horizon." if main_only
               else "The fourteen registry rules stay below. They are a different set of books from these four.")
    return True


def _table(metrics, meta, model_key, show_all):
    names = {item["key"]: item["label"] for item in meta["recipes"]}
    rows = []
    if show_all:
        frame = metrics.copy()
    else:
        wanted = []
        for _, version, recipe_mode in FOCUS:
            part = metrics.loc[metrics.version.eq(version) & metrics.momentum_months.isin([0, 1])]
            if recipe_mode == "":
                part = part.loc[part.model_key.eq("none")]
            else:
                part = part.loc[part.model_key.eq(model_key)]
            wanted.append(part)
        frame = pd.concat(wanted, ignore_index=True)
    reference = metrics.loc[metrics.recipe.eq("all_20"), ["version", "model_key", "momentum_months", "cagr"]]
    reference = reference.rename(columns={"cagr": "all20_cagr"})
    frame = frame.merge(reference, on=["version", "model_key", "momentum_months"], how="left")
    frame["delta"] = frame.cagr - frame.all20_cagr
    frame.loc[frame.recipe.eq("") | frame.recipe.eq("all_20"), "delta"] = 0.0
    order = {version: index for index, (_, version, _) in enumerate(FOCUS)}
    recipe_order = {"": -1, **{item["key"]: index for index, item in enumerate(meta["recipes"])}}
    frame["_order"] = frame.version.map(order).fillna(len(order))
    frame["_recipe"] = frame.recipe.map(recipe_order).fillna(len(recipe_order))
    frame = frame.sort_values(["_order", "momentum_months", "_recipe", "model_key"])
    frame["book"] = frame.version.map({
        "Equal weight": "Equal weight",
        TOP_VOL_VERSION: "Top 10 positive-vol",
        "Basket momentum + cash": "Full baskets",
        DOWNSIDE_VERSION: "Top 10 + downside to cash",
        UPSIDE_VERSION: "Top 10 + upside tilt",
        BASKET_DOWNSIDE_VERSION: "Baskets + downside to cash",
        BASKET_UPSIDE_VERSION: "Baskets + upside tilt",
    })
    frame["recipe_label"] = frame.recipe.map(lambda key: "No factors" if key == "" else names.get(key, key))
    frame["horizon"] = frame.momentum_months.map(lambda value: "—" if int(value) == 0 else f"{int(value)} month")
    columns = ["book", "recipe_label", "cagr", "sharpe", "max_drawdown", "avg_cash_weight", "delta"]
    names_out = ["Book", "Feature recipe", "CAGR", "Sharpe", "Max drawdown", "Average cash", "CAGR vs all 20"]
    if show_all:
        columns = ["book", "recipe_label", "model_key", "horizon", "cagr", "sharpe", "max_drawdown", "avg_cash_weight", "delta"]
        names_out = ["Book", "Feature recipe", "Model", "Momentum", "CAGR", "Sharpe", "Max drawdown", "Average cash", "CAGR vs all 20"]
    show = frame[columns].copy()
    show.columns = names_out
    return show


def _figure(study, version, model_key, labels, months=1):
    monthly = study["monthly"]
    figure = go.Figure()
    for name, rows, color in (
        ("Equal weight", monthly.loc[monthly.version.eq("Equal weight")], "#68716e"),
        ("Top 10 positive-vol", monthly.loc[monthly.version.eq(TOP_VOL_VERSION) & monthly.momentum_months.eq(months)], "#b36a2e"),
    ):
        rows = rows.sort_values("return_month")
        figure.add_trace(go.Scatter(
            x=[rows.snapshot_date.iloc[0], *rows.return_month], y=[1, *rows.wealth],
            name=name, line=dict(color=color), hovertemplate="%{x|%b %Y}<br>Growth $%{y:.3f}<br>" +
            ("Equal-weight stock reference; no selected factors." if name == "Equal weight" else
             "Ten high positive-volatility stocks per group, weighted by positive momentum with cash; no directional predictor.") +
            "<extra>%{fullData.name}</extra>"))
    if version in {BASKET_DOWNSIDE_VERSION, BASKET_UPSIDE_VERSION}:
        original = monthly.loc[monthly.version.eq('Basket momentum + cash') & monthly.momentum_months.eq(months)]
        original = original.sort_values('return_month')
        if not original.empty:
            figure.add_trace(go.Scatter(x=[original.snapshot_date.iloc[0], *original.return_month],
                y=[1, *original.wealth], name='Full baskets — original', line=dict(color='#287d68'),
                hovertemplate='%{x|%b %Y}<br>Growth $%{y:.3f}<br>Original grouped positive-momentum and cash portfolio; no directional adjustment.<extra>%{fullData.name}</extra>'))
    overlays = monthly.loc[monthly.version.eq(version) & monthly.model_key.eq(model_key) & monthly.momentum_months.eq(months)]
    for recipe, rows in overlays.groupby("recipe", sort=False):
        rows = rows.sort_values("return_month")
        explanation = {"all_20": "Directional model fitted with all 20 supplied factors.",
            "anova_10": "Directional model refitted after selecting the top 10 ANOVA-ranked inputs.",
            "pruned_10": "Directional model refitted using up to 10 ANOVA-ranked inputs after removing correlated duplicates.",
            "forest_10": "Directional model refitted with inputs selected by a separate random forest's training importance."}.get(recipe, recipe)
        figure.add_trace(go.Scatter(
            x=[rows.snapshot_date.iloc[0], *rows.return_month], y=[1, *rows.wealth],
            name=labels.get(recipe, recipe), line=dict(color=COLORS.get(recipe, "#333333")),
            hovertemplate="%{x|%b %Y}<br>Growth $%{y:.3f}<br>" + explanation + "<extra>%{fullData.name}</extra>"))
    figure.update_layout(template="plotly_white", height=460, hovermode="x unified",
                         legend=dict(orientation="h", y=-.2), margin=dict(l=10, r=10, t=30, b=10),
                         title=f"{months}-month momentum · growth of $1", yaxis_title="Growth of $1")
    return figure
