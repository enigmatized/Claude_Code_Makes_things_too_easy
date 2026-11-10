"""Final-tab view of the one frozen 2010–2018 pass."""
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
    TOP_VOL_VERSION: "Top 10 positive-vol",
    BASKET_VERSION: "Full baskets",
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
    registry = default_model_registry()
    present = set(metrics.model_key)
    model_labels = {registry[key].label: key for key in registry if key in present}
    model_key = model_labels[st.selectbox("Prediction model for the four books", list(model_labels), key="final_books_model")]
    if main_only:
        st.session_state['final_books_show_all'] = False
    show_all = st.checkbox("Show every model and horizon", key="final_books_show_all", disabled=main_only)
    table = _table(metrics, model_key, show_all)
    formats = {column: "{:.3f}" if column == "Sharpe" else "{:.1f}" if column == "Names held" else "{:.2%}"
               for column in table.columns if column not in {"Book", "Model", "Momentum"}}
    st.dataframe(table.style.format(formats), use_container_width=True, hide_index=True)
    direction = st.selectbox("Overlay direction on the chart", list(DIRECTIONS), key="final_books_direction")
    st.plotly_chart(
        _figure(monthly, model_key, direction, window, study["monthly"].return_month.min()),
        use_container_width=True, key=f"final_books_chart_{window}")
    with st.expander("Factors kept by the frozen fit"):
        st.caption("Loss and upside are separate events, so the columns can differ. Both lists were chosen at the freeze date.")
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
            "These are the saved Advanced ML books, before the official test. "
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
            "Growth compounds from the training window straight into the test."
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
    frame["horizon"] = frame.momentum_months.map(lambda value: "—" if int(value) == 0 else f"{int(value)} month")
    frame["model"] = frame.model_key.map(lambda key: "—" if key == "none" else key)
    columns = ["book", "cagr", "annualized_volatility", "sharpe", "max_drawdown", "avg_cash_weight", "avg_names"]
    headers = ["Book", "CAGR", "Volatility", "Sharpe", "Max drawdown", "Average cash", "Names held"]
    if show_all:
        columns = ["book", "model", "horizon", *columns[1:]]
        headers = ["Book", "Model", "Momentum", *headers[1:]]
    show = frame[columns].copy()
    show.columns = headers
    return show


def _curve(figure, rows, name, color):
    rows = rows.sort_values("return_month")
    if rows.empty:
        return
    figure.add_trace(go.Scatter(
        x=[rows.snapshot_date.iloc[0], *rows.return_month], y=[1, *rows.wealth],
        name=name, line=dict(color=color)))


def _figure(monthly, model_key, direction, window, test_start):
    top_version, basket_version, color = DIRECTIONS[direction]
    figure = go.Figure()
    _curve(figure, monthly.loc[monthly.version.eq("Equal weight")], "Equal weight", "#68716e")
    _curve(figure, monthly.loc[monthly.version.eq(TOP_VOL_VERSION) & monthly.momentum_months.eq(1) & monthly.model_key.eq("none")],
           "Top 10 positive-vol", "#b36a2e")
    _curve(figure, monthly.loc[monthly.version.eq(top_version) & monthly.momentum_months.eq(1) & monthly.model_key.eq(model_key)],
           f"Top 10 + {direction.lower()}", color)
    _curve(figure, monthly.loc[monthly.version.eq(BASKET_VERSION) & monthly.momentum_months.eq(1) & monthly.model_key.eq("none")],
           "Full baskets", "#287d68")
    _curve(figure, monthly.loc[monthly.version.eq(basket_version) & monthly.momentum_months.eq(1) & monthly.model_key.eq(model_key)],
           f"Baskets + {direction.lower()}", "#536da6")
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
                         title=titles[window], yaxis_title="Growth of $1", xaxis_title="Earned month")
    return figure
