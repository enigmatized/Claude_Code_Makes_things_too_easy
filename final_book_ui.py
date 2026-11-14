"""Final-tab view of the one frozen 2010–2018 pass."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from research.basket_adjustments import (
    BASKET_DOWNSIDE_VERSION, BASKET_UPSIDE_VERSION, BASKET_VERSION,
    DOWNSIDE_VERSION, UPSIDE_VERSION,
)
from research.basket_momentum import TOP_VOL_VERSION
from research.final_books import list_final_books, load_final_books, matching_training_monthly, window_books
from research.models import default_model_registry
from portfolio_focus import main_portfolio_focus
from app_charts import RETURN_MEASURES, return_curve
from portfolio_selection import grouped_cases
from portfolio_selection_ui import select_portfolio_case
from final_comparison_ui import offer_final_comparison


FOCUS = (
    ("Equal weight", "Equal weight"),
    ("Top 10 positive-vol", TOP_VOL_VERSION),
    ("Top 10 + downside to cash", DOWNSIDE_VERSION),
    ("Top 10 + upside tilt", UPSIDE_VERSION),
    ("Full baskets", BASKET_VERSION),
    ("Baskets + downside to cash", BASKET_DOWNSIDE_VERSION),
    ("Baskets + upside tilt", BASKET_UPSIDE_VERSION),
)
OVERLAYS = {DOWNSIDE_VERSION, UPSIDE_VERSION, BASKET_DOWNSIDE_VERSION, BASKET_UPSIDE_VERSION}
NAMES = {
    "Equal weight": "Equal weight",
    "Momentum + cash": "Momentum + cash",
    TOP_VOL_VERSION: "K-means top 10 positive-vol + momentum/cash",
    BASKET_VERSION: "K-means full groups + momentum/cash",
    DOWNSIDE_VERSION: "Top 10 + downside to cash",
    UPSIDE_VERSION: "Top 10 + upside tilt",
    BASKET_DOWNSIDE_VERSION: "Baskets + downside to cash",
    BASKET_UPSIDE_VERSION: "Baskets + upside tilt",
}
DIRECTIONS = {
    "Downside to cash": (DOWNSIDE_VERSION, BASKET_DOWNSIDE_VERSION, "#6b3f69"),
    "Upside tilt": (UPSIDE_VERSION, BASKET_UPSIDE_VERSION, "#1d6a6a"),
}
WINDOW_LABELS = (
    ("test", "Official test only"),
    ("training", "Training window only"),
    ("entire", "Entire path"),
)


def render_final_books(store=None, published=None):
    """Draw the saved four-book pass. Return whether a study was on disk."""
    studies = list_final_books(store.root) if published is None else published.list_studies()
    if not studies:
        return False
    main_only = main_portfolio_focus('final')
    chosen = studies[0]
    cache_key = str(chosen["directory"]) if published is None else published.token
    cached = st.session_state.get("loaded_final_books")
    if cached is None or cached[0] != cache_key:
        cached = (cache_key, load_final_books(chosen["directory"]) if published is None else published.study())
        st.session_state["loaded_final_books"] = cached
    study = cached[1]
    meta = study["metadata"]
    st.markdown("**K-means → top 10 positive-volatility stocks per group → 1-month positive momentum → unused capital in cash.**")
    st.caption("This original grouped momentum strategy is included below alongside equal weight and its separate ML adjustments. "
               "The top-10 screen uses trailing 12-month positive-side volatility; the holding signal uses 1-month momentum. "
               "Non-positive group/stock momentum receives no allocation. It needs no directional classifier to run.")
    st.info("**Saved fixed-model evaluation.** These official-test portfolios use factory classifier settings and K-means grouping. "
            "They are not the Tuned model tab’s winners. New feature-engineered variants, when published, have their own evaluation section below.")
    with st.expander("What is included — and how this relates to tuning"):
        st.write("The Tuned model tab measures a different question: fixed versus searched classifier settings on a 2007–2009 "
                 "holdout within training. It covers fourteen basic strategies. This page shows the existing frozen grouped-portfolio "
                 "evaluation on 2010–2018; it is not an automatic next stage of your current UI selections.")
        st.write("Grouping With ML now includes both K-means and Ward on training data. Only K-means has the saved official-test "
                 "evaluation in this earlier study. The feature-engineering extension separately evaluates K-means and Ward starting books. "
                 "Stock-group five-group / twelve-month settings stay fixed; factor grouping is a different learned step.")
        st.caption("No new tuning, clustering or portfolio replay occurs when viewing this page. The official test period was already inspected.")
    if published is None:
        try:
            training_monthly, training_meta = matching_training_monthly(store.root)
        except ValueError as error:
            training_monthly, training_meta = None, None
            st.caption(f"Training window could not be read: {error}")
        windows = window_books(study["monthly"], training_monthly)
    else:
        windows, training_meta = published.windows(), published.training_metadata()
    labels = {label: key for key, label in WINDOW_LABELS if key in windows}
    if st.session_state.get('final_books_window') not in labels:
        st.session_state.pop('final_books_window', None)
    window = labels[st.selectbox("Which years", list(labels), key="final_books_window")]
    metrics, monthly = windows[window]
    st.subheader(_window_title(window, monthly))
    st.caption(_window_caption(window, monthly, meta, training_meta))
    st.caption(meta["recipe"][-1])
    cases=grouped_cases(metrics)
    choice=select_portfolio_case(cases,'final_selection',keys={'model':'final_books_model','target':'final_books_direction'},
                                preferred_basket='K-means top 10 positive-vol per group')
    if choice.empty:
        return True
    row=choice.iloc[0]
    model_key=row.model_key
    direction='Upside tilt' if row.target=='upside' else 'Downside to cash'
    horizon=int(row.momentum_months)
    show_all=st.checkbox('Compare more saved models, baskets and horizons',key='final_books_show_all')
    if not show_all:
        same=cases.basket.eq(row.basket)&cases.allocation.eq(row.allocation)
        exact=same & ((cases.model_key.eq(model_key)&cases.target.eq(row.target))|cases.model_key.eq('none'))
        metrics=cases.loc[exact].copy()
        keys=metrics[['version','model_key','momentum_months']].drop_duplicates()
        monthly=monthly.merge(keys,on=['version','model_key','momentum_months'],validate='many_to_one')
    table = _table(metrics, model_key, True)
    formats = {column: "{:.3f}" if column == "Sharpe" else "{:.1f}" if column == "Names held" else "{:.2%}"
               for column in table.columns if column not in {"Book", "Model", "Momentum"}}
    st.dataframe(table.style.format(formats), use_container_width=True, hide_index=True)
    with st.expander("Change from each original grouped portfolio"):
        changes = _overlay_changes(metrics, model_key)
        st.dataframe(changes.style.format({column: "{:+.3f}" if column == "Sharpe change" else "{:+.2f}"
                     for column in changes.columns if column.endswith("(pp)") or column == "Sharpe change"}),
                     use_container_width=True, hide_index=True)
        st.caption("Each adjustment is paired with its own full-group or top-ten original at the same momentum window. "
                   "pp = percentage points. Positive drawdown change means a less negative drawdown. "
                   "These are observed differences, not proof of an improvement.")
    measure = st.selectbox("Return measure", list(RETURN_MEASURES), key="final_books_return_measure")
    st.plotly_chart(
        _figure(monthly, model_key, direction, window, study["monthly"].return_month.min(), measure, months=horizon if horizon else 1),
        use_container_width=True, key=f"final_books_chart_{window}")
    offer_final_comparison(metrics,monthly,protocol='K-means fixed-model study',source=cache_key,
        case_fields=['version','model_key','momentum_months'],prefix='final_books',window=window)
    st.caption("Log returns are calculated from the saved portfolio simple returns, after stock weights and cash have been accounted for. "
               "Cumulative log returns add across earned months; CAGR compounds the same path.")
    with st.expander("Official-test features kept by the frozen fit"):
        st.caption("This list describes the frozen official-test model, even when viewing training or the joined path. "
                   "The training overlays use all 20 inputs instead. Loss and upside are separate events, "
                   "so the columns can differ; both test lists were chosen at the freeze date.")
        if model_key == 'none':
            st.caption('This original portfolio uses no directional classifier or selected model inputs.')
        else:
            for target, label in (("loss", "Loss"), ("upside", "Upside")):
                columns = meta.get("fitted_features", {}).get(model_key, {}).get(target, [])
                st.write(f"{label}: {', '.join(columns) if columns else 'not recorded'}")
    return True


def _span(monthly):
    start = pd.Timestamp(monthly.return_month.min())
    end = pd.Timestamp(monthly.return_month.max())
    count = int(monthly.return_month.nunique())
    return start, end, count


def _window_title(window, monthly):
    start, end, count = _span(monthly)
    if window == "training":
        return f"Training window only · {start:%b %Y}–{end:%b %Y}"
    if window == "entire":
        return f"Entire path · {start:%b %Y}–{end:%b %Y}"
    return f"Official test only · {start:%b %Y}–{end:%b %Y}"


def _window_caption(window, monthly, meta, training_meta):
    start, end, count = _span(monthly)
    span = f"{start:%b %Y} through {end:%b %Y}, {count} months"
    if window == "training":
        return (
            f"Training window only: {span}. "
            "These are the saved K-means grouped portfolios, before the official test. "
            "Overlays there use expanding all-20 forecasts. "
            "Gross of costs. Cash earns nothing. Browsing does not fit a model.")
    if window == "entire":
        source = ""
        if training_meta is not None:
            source = (
                f" Training months are {training_meta.get('first_return_month')} through "
                f"{training_meta.get('last_return_month')}, with expanding all-20 overlays. "
                f"Test months use the one frozen ANOVA fit at {meta['fit_snapshot']}.")
        return (
            f"Entire path: {span}. The dotted line is the first official test month. "
            "Growth compounds from the training window straight into the test. This is a joined illustration across two model protocols, "
            "not one model fitted and evaluated over the whole path."
            f"{source} Nothing is refit.")
    omitted = "Choose Training window only or Entire path above to see the earlier years."
    if training_meta is not None:
        omitted = (
            f"The training window {training_meta.get('first_return_month')} through "
            f"{training_meta.get('last_return_month')} is a separate choice above.")
    return (
        f"Official test only: {span}. "
        f"One {meta['fit_policy']} fit at {meta['fit_snapshot']}, "
        f"using labels known through {meta['trained_through_return_month']}. "
        f"ANOVA kept {meta['n_features']} factors inside that fit. "
        f"Strength {meta['adjustment_strength']}, upside threshold {meta['upside_threshold']}. "
        f"{omitted} "
        "Gross of costs. Cash earns nothing. Browsing reads the saved file and does not fit a model.")


def _table(metrics, model_key, show_all):
    if show_all:
        frame = metrics.copy()
    else:
        wanted = []
        for _, version in FOCUS:
            part = metrics.loc[metrics.version.eq(version) & metrics.momentum_months.isin([0, 1])]
            key = model_key if version in OVERLAYS else "none"
            wanted.append(part.loc[part.model_key.eq(key)])
        frame = pd.concat(wanted, ignore_index=True)
    order = {version: index for index, (_, version) in enumerate(FOCUS)}
    frame["_order"] = frame.version.map(order).fillna(len(order))
    frame = frame.sort_values(["_order", "momentum_months", "model_key"])
    frame["book"] = frame.version.map(NAMES).fillna(frame.version)
    top = frame.version.eq(TOP_VOL_VERSION)
    frame.loc[top, "book"] = frame.loc[top, "momentum_months"].map(
        lambda months: f"K-means top 10 positive-vol · {int(months)}M momentum + cash")
    frame["horizon"] = frame.momentum_months.map(lambda value: "—" if int(value) == 0 else f"{int(value)} month")
    frame["model"] = frame.model_key.map(lambda key: "—" if key == "none" else key)
    if "annualized_log_growth" not in frame:
        frame["annualized_log_growth"] = np.log1p(frame.cagr)
    columns = ["book", "cagr", "annualized_log_growth", "annualized_volatility", "sharpe", "max_drawdown", "avg_cash_weight", "avg_names"]
    headers = ["Book", "CAGR", "Annual log growth", "Volatility", "Sharpe", "Max drawdown", "Average cash", "Names held"]
    if show_all:
        columns = ["book", "model", "horizon", *columns[1:]]
        headers = ["Book", "Model", "Momentum", *headers[1:]]
    show = frame[columns].copy()
    show.columns = headers
    return show


def _curve(figure, rows, name, color, measure="Growth of $1"):
    rows = rows.sort_values("return_month")
    if rows.empty:
        return
    book = rows.set_index("return_month").copy()
    book["log_return"] = np.log1p(book.simple_return)
    series = return_curve(book, measure)
    figure.add_trace(go.Scatter(x=series.index, y=series, name=name, line=dict(color=color)))


def _figure(monthly, model_key, direction, window, test_start, measure="Growth of $1", months=1):
    top_version, basket_version, color = DIRECTIONS[direction]
    figure = go.Figure()
    _curve(figure, monthly.loc[monthly.version.eq("Equal weight")], "Equal weight", "#68716e", measure)
    _curve(figure, monthly.loc[monthly.version.eq("Momentum + cash") & monthly.momentum_months.eq(months)],
           f"{months}M momentum + cash (original)", "#899594", measure)
    _curve(figure, monthly.loc[monthly.version.eq(TOP_VOL_VERSION) & monthly.momentum_months.eq(months) & monthly.model_key.eq("none")],
           f"K-means top 10 · {months}M momentum + cash (original)", "#b36a2e", measure)
    _curve(figure, monthly.loc[monthly.version.eq(top_version) & monthly.momentum_months.eq(months) & monthly.model_key.eq(model_key)],
           f"K-means top 10 + {direction.lower()}", color, measure)
    _curve(figure, monthly.loc[monthly.version.eq(BASKET_VERSION) & monthly.momentum_months.eq(months) & monthly.model_key.eq("none")],
           f"K-means full groups · {months}M momentum + cash", "#287d68", measure)
    _curve(figure, monthly.loc[monthly.version.eq(basket_version) & monthly.momentum_months.eq(months) & monthly.model_key.eq(model_key)],
           f"Baskets + {direction.lower()}", "#536da6", measure)
    start, end, _ = _span(monthly)
    titles = {
        "test": f"Official test only · {start:%b %Y}–{end:%b %Y}",
        "training": f"Training window only · {start:%b %Y}–{end:%b %Y}",
        "entire": f"Entire path · {start:%b %Y}–{end:%b %Y}",
    }
    if window == "entire":
        figure.add_vline(x=pd.Timestamp(test_start), line_dash="dot", line_color="#9aa39a")
        figure.add_annotation(x=pd.Timestamp(test_start), y=1, yref="paper", text="Official test starts",
                              showarrow=False, xanchor="left", font=dict(size=12, color="#68716e"))
    figure.update_layout(template="plotly_white", height=460, hovermode="x unified",
                         legend=dict(orientation="h", y=-.2), margin=dict(l=10, r=10, t=40, b=10),
                         title=titles[window], yaxis_title=measure, xaxis_title="Earned month")
    return figure


def _overlay_changes(metrics, model_key):
    """Pair saved metric rows; never compare different windows or starting books."""
    pairs = {DOWNSIDE_VERSION: TOP_VOL_VERSION, UPSIDE_VERSION: TOP_VOL_VERSION,
             BASKET_DOWNSIDE_VERSION: BASKET_VERSION, BASKET_UPSIDE_VERSION: BASKET_VERSION}
    rows = []
    for version, original in pairs.items():
        adjusted = metrics.loc[metrics.version.eq(version) & metrics.model_key.eq(model_key)]
        for _, row in adjusted.iterrows():
            baseline = metrics.loc[metrics.version.eq(original) & metrics.model_key.eq("none")
                                   & metrics.momentum_months.eq(row.momentum_months)]
            if len(baseline) != 1:
                continue
            base = baseline.iloc[0]
            rows.append({"Original": NAMES[original], "Adjustment": NAMES[version],
                "Momentum": f"{int(row.momentum_months)} month",
                "CAGR change (pp)": 100 * (row.cagr - base.cagr),
                "Sharpe change": row.sharpe - base.sharpe,
                "Max drawdown change (pp)": 100 * (row.max_drawdown - base.max_drawdown),
                "Cash change (pp)": 100 * (row.avg_cash_weight - base.avg_cash_weight)})
    return pd.DataFrame(rows)
