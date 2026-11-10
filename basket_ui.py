"""Advanced-strategy view: correlation baskets, short momentum, and a positive-vol screen."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from research.basket_momentum import FEATURE, TOP_VOL_VERSION, VERSIONS, list_basket_studies, load_basket_study
from research.basket_adjustments import (
    ADJUSTED_VERSIONS, BASKET_ADJUSTED_VERSIONS, BASKET_DOWNSIDE_VERSION, BASKET_UPSIDE_VERSION,
    DOWNSIDE_VERSION, UPSIDE_VERSION,
)
from research.models import default_model_registry
from research.basket_reporting import (normalize_study, require_basket_training, detail_frames, grouping_profiles,
                                       selected_vol_ranks)


COLORS = {
    "Equal weight": "#68716e",
    "Momentum + cash": "#536da6",
    "Basket momentum + cash": "#287d68",
    TOP_VOL_VERSION: "#b36a2e",
    DOWNSIDE_VERSION: "#7a4e8a",
    UPSIDE_VERSION: "#247a9b",
    BASKET_DOWNSIDE_VERSION: "#6b3f69",
    BASKET_UPSIDE_VERSION: "#1d6a6a",
}
HORIZON_COLORS = {1: "#287d68", 2: "#536da6", 3: "#b36a2e", 12: "#7a4e8a"}
FULL_GROUP = "Basket momentum + cash"
DISPLAY_ORDER = (
    "Equal weight", "Momentum + cash", FULL_GROUP, *BASKET_ADJUSTED_VERSIONS,
    TOP_VOL_VERSION, *ADJUSTED_VERSIONS,
)


def _research_study(studies, method):
    candidates = [item for item in studies if item["metadata"].get("grouping_method", "kmeans") == method]
    if not candidates:
        return None
    labels = {_study_label(item): item for item in candidates}
    lead_label = next((label for label, item in labels.items()
        if item["metadata"].get("directional_adjustment")
        and BASKET_DOWNSIDE_VERSION in item["metadata"].get("versions", [])), next(iter(labels)))
    key = f"basket_study_{method}"
    chosen = labels[_select(f"Saved {METHOD_NAMES[method]} study", list(labels), key, list(labels).index(lead_label))]
    from research.basket_momentum import _digest
    cache_key = (str(chosen["directory"]), _digest(chosen["directory"] / "study.json"))
    state_key = f"loaded_basket_study_{method}"
    cached = st.session_state.get(state_key)
    if cached is None or cached[0] != cache_key:
        cached = (cache_key, _normalize(load_basket_study(chosen["directory"])))
        st.session_state[state_key] = cached
    return cached[1]


METHOD_NAMES = {"kmeans": "K-means", "ward": "Hierarchical (Ward)"}


def render_basket_strategy(store=None, published=None):
    st.header("Grouping With ML")
    st.write("Compare two ways of grouping stocks, apply the same investment rules, then add separate up/down predictions.")
    st.info("**Training data only.** These saved training-validation results use no test-period data. Test results belong in Final test.")
    st.caption("These grouping studies keep their original model inputs and evaluation period. Feature engineering separately "
               "trains and tunes new input recipes on both K-means/Ward starting portfolios, with its own Final test section.")
    available = {}
    studies = [] if published is not None else list_basket_studies(store.root)
    if published is None and not studies:
        st.info("No grouped-strategy study is saved yet. Run python3 run_basket_momentum.py explicitly in research mode.")
        return
    kmeans, ward, predictions = st.tabs(["K-means", "Hierarchical clustering", "Adding a direction"])
    for method, tab in (("kmeans", kmeans), ("ward", ward)):
        with tab:
            reader = published.for_method(method) if published is not None else None
            study = reader.study() if reader is not None else _research_study(studies, method) if published is None else None
            if study is None:
                st.info(f"No saved {METHOD_NAMES[method]} result is included. In research mode run: python3 run_basket_momentum.py --grouping {method}.")
                continue
            require_basket_training(study)
            available[method] = study
            meta = study["metadata"]
            st.caption(f"TRAINING VALIDATION · {meta['first_return_month']} to {meta['last_return_month']} · "
                       f"{meta['months']} earned months. Decisions precede returns. Gross of costs; cash earns 0%. Saved results only.")
            st.markdown("**Trailing return paths → learned groups → optional top-10 screen → positive momentum + cash**")
            _render_grouped_strategy(study, prefix="basket" if method == "kmeans" else "ward")
    with predictions:
        if not available:
            st.info("No saved grouped strategy is available yet.")
            return
        options = {METHOD_NAMES[key]: key for key in available}
        method = options[_select("Grouping method for directional adjustments", list(options), "basket_direction_grouping")]
        _render_prediction_adjustments(available[method])
        if len(available) == 2:
            with st.expander("Compare grouping methods under the same trading rules"):
                st.caption("Original portfolios only; identical earned months, momentum windows and cash rules. Group numbers are unrelated across methods.")
                rows = []
                for key, saved in available.items():
                    rows.append(_overview_table(_originals(saved)["metrics"]).assign(Grouping=METHOD_NAMES[key]))
                st.dataframe(pd.concat(rows, ignore_index=True), hide_index=True, use_container_width=True)


def _select(label, options, key, default=0):
    if st.session_state.get(key) not in options:
        st.session_state.pop(key, None)
    return st.selectbox(label, options, index=default, key=key)


def _originals(study):
    return {**study, **{name: study[name].loc[study[name].model_key.eq("none")].copy()
                       for name in ("metrics", "monthly")}}


def _grouping_view(study, prefix="basket"):
    meta = study["metadata"]
    lookback = meta.get("correlation_lookback_months", meta["lookback_months"])
    st.markdown("### 1. How the groups are learned")
    method = meta.get("grouping_method", "kmeans")
    st.write(f"Each stock becomes a row of {lookback} already-earned monthly returns. We subtract that stock's own mean "
             "and divide by its own standard deviation. Two stocks with similar up/down timing therefore look close, "
             "even if one usually has larger moves. Neither next-month returns nor the supplied factor ranks enter this fit.")
    if method == "ward":
        st.write(f"Hierarchical clustering starts with each stock in its own group. Ward's rule repeatedly merges the pair "
                 f"that adds the least within-group squared variation. Cut this merge tree when {meta['n_clusters']} groups remain. "
                 "Unlike K-means, it builds a nested hierarchy rather than repeatedly moving stocks between centroids.")
        st.caption("The tree below is the actual saved fit, not a newly fitted illustration. Larger merge heights mean more dissimilar groups were joined.")
        st.markdown("[Method reference: SciPy Ward linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)")
    else:
        st.write(f"K-means starts with {meta['n_clusters']} representative paths called centroids. Assign each stock to its closest centroid, "
                 "replace each centroid by its members' mean path, and repeat until the assignments settle. "
                 "We use ten seeded starts and keep the fit with the smallest total squared distance to the centroids.")
        st.caption("The heatmap below shows each group's mean standardized return path: it helps explain what 'similar' means. "
                   "Grouping is unsupervised: it receives no up/down labels and does not itself predict an investment return.")
        st.markdown("[Method reference: scikit-learn K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means)")
    st.write("Only complete, non-constant histories enter the grouping. Membership is fitted again each month. "
             "Group numbers are temporary labels: Group 2 next month can contain different stocks. "
             "The supplied factor ranks and next-month outcomes are not clustering inputs.")
    st.caption("Grouping produces memberships. Portfolio returns begin when we apply an allocation rule below; "
               "there is no separate saved grouping-only investment backtest.")
    if 1 not in _horizons(study):
        st.info("This older study has no saved one-month signals for the input-path illustration. Its portfolio results remain available below.")
        return
    details = detail_frames(study, 1, [FULL_GROUP])
    assigned = details["assignments"]
    dates = sorted(pd.to_datetime(assigned.date).unique())
    if not dates:
        st.info("No saved group membership is available.")
        return
    choices = {pd.Timestamp(date).strftime("%Y-%m-%d"): pd.Timestamp(date) for date in dates}
    snapshot = choices[_select("Grouping decision snapshot", list(choices), f"{prefix}_group_snapshot", len(choices) - 1)]
    if method == "ward" and "hierarchy" in study:
        from research.grouping_charts import hierarchy_figure
        st.plotly_chart(hierarchy_figure(study["hierarchy"], snapshot, meta["n_clusters"]),
                        use_container_width=True, key=f"{prefix}_tree")
        st.caption("The tree is shortened to its final 25 branches. Parenthesized numbers count stocks inside a collapsed branch; "
                   "the dashed line marks the cut used for the saved groups. Horizontal order carries no return ranking.")
    members = assigned.loc[pd.to_datetime(assigned.date).eq(snapshot)].dropna(subset=[FEATURE])
    counts = members.groupby(FEATURE).size().rename("Stocks")
    st.plotly_chart(go.Figure(go.Bar(x=[f"Group {int(group)}" for group in counts.index], y=counts,
        marker_color="#287d68", hovertemplate="%{x}<br>%{y} members<extra></extra>")).update_layout(
        template="plotly_white", height=300, margin=dict(l=10,r=10,t=15,b=10), yaxis_title="Stocks in the learned group"),
        use_container_width=True, key=f"{prefix}_group_counts")
    profiles, coverage = grouping_profiles(assigned, snapshot, lookback)
    if profiles.empty:
        st.info("The saved evaluation window does not contain a complete earlier input path for this snapshot. "
                "Choose a later month to see the pattern illustration; memberships above are still the actual saved groups.")
    else:
        table = profiles.pivot(index=FEATURE, columns="input_month", values="mean_standardized_return")
        figure = go.Figure(go.Heatmap(x=[f"{date:%b %Y}" for date in table.columns],
            y=[f"Group {int(group)}" for group in table.index], z=table.to_numpy(),
            colorscale="RdBu", zmid=0, colorbar=dict(title="Standardized return"),
            hovertemplate="%{y}<br>%{x}<br>Mean standardized return %{z:.2f}<extra></extra>"))
        figure.update_layout(template="plotly_white", height=320, margin=dict(l=10,r=10,t=15,b=10),
                             xaxis_title="Already-earned input month")
        st.plotly_chart(figure, use_container_width=True, key=f"{prefix}_group_profiles")
        st.caption(f"Return-pattern illustration: mean standardized path of {coverage['illustrated']} of "
                   f"{coverage['members']} current members with complete paths in the saved archive. "
                   "These are dimensionless scores, not percentage returns. Current membership is applied to the past paths; "
                   "older group numbers are not linked. No clustering model is fitted to draw this chart.")
    groups = {f"Group {int(group)}": group for group in counts.index}
    if groups:
        group = groups[_select("Group members to inspect", list(groups), f"{prefix}_group_members")]
        st.dataframe(members.loc[members[FEATURE].eq(group), ["stock_id", FEATURE]].rename(
            columns={"stock_id":"Stock ID", FEATURE:"Saved group"}), hide_index=True, use_container_width=True)


def _performance(study, months, versions, key):
    metrics = _slice(study["metrics"], months).loc[lambda f: f.version.isin(versions)]
    monthly = _slice(study["monthly"], months).loc[lambda f: f.version.isin(versions)]
    formats = {column: "{:.3f}" if column == "Sharpe" else "{:.2%}"
               for column in _metric_table(metrics).columns if column != "Version"}
    st.dataframe(_metric_table(metrics).style.format(formats), use_container_width=True, hide_index=True)
    title = f"{months}-month momentum · Training validation · {study['metadata']['first_return_month']} to {study['metadata']['last_return_month']}"
    st.plotly_chart(_figure(monthly, title), use_container_width=True, key=key)
    st.caption("CAGR compounds portfolio returns. Annual log growth = log(1 + CAGR). Cash earns 0%; no trading costs are deducted.")


def _render_grouped_strategy(study, prefix="basket"):
    base = _originals(study)
    _grouping_view(base, prefix)
    st.markdown("### 2. Apply positive momentum and leave the rest in cash")
    horizons = _horizons(base)
    names = {f"{month}-month momentum": month for month in horizons}
    selected = names[_select("Momentum window for the grouped strategy", list(names), f"{prefix}_base_horizon")]
    st.write(f"At each snapshot, compound the last {selected} earned monthly return{'s' if selected != 1 else ''}. "
             "Compute group momentum from an equal-weight return path of its current members. "
             "A group with zero or negative momentum gets zero stock allocation. "
             "Inside a positive group, buy only stocks with strictly positive momentum; zero and negative stock momentum receive no money.")
    st.write("If P stocks in a positive group qualify out of N available stocks in the whole book, "
             "that group invests P/N of portfolio capital, split in proportion to those stocks' positive momentum. "
             "All remaining capital stays in cash. With four available stocks, two positive signals of 10% and 30% "
             "in a positive group give weights 12.5% and 37.5%, leaving 50% cash.")
    st.caption("The 12-month grouping window describes co-movement; the separate 1/2/3-month momentum window decides holdings. "
               "The comparison below adds grouping to the same momentum-and-cash idea.")
    _performance(base, selected, ["Equal weight", "Momentum + cash", FULL_GROUP], f"{prefix}_base_curves")
    st.markdown("### 3. Keep the top 10 by positive-side volatility in each group")
    st.write("After learning the groups, rank stocks inside each group by positive volatility over the same trailing 12 months. "
             "Keep up to 10 per group, then recompute group momentum on those retained names and apply the same positive-only/cash rule. "
             "The selected names define the new book's capital budget; excluded names receive zero weight.")
    st.latex(r"\mathrm{PositiveVol}=\sqrt{\frac{1}{12}\sum_{j=1}^{12}\max(r_j,0)^2}")
    st.caption("A down month contributes zero and still counts among the 12 observations. "
               "Stocks with no up months are excluded; tied scores use lower stock ID. "
               "This measures the size of past upside moves, not a prediction of next month's return.")
    if _has_top_vol(base, selected):
        _performance(base, selected, ["Equal weight", FULL_GROUP, TOP_VOL_VERSION], f"{prefix}_top_curves")
    else:
        st.info("No top-10 result is saved for this momentum window. Choose 1 or 2 months for that comparison.")
    if st.checkbox("Inspect group cash and selected-stock weights", key=f"{prefix}_base_inspect"):
        details = detail_frames(base, selected, [FULL_GROUP, TOP_VOL_VERSION])
        _baskets({**base, **details}, selected, key_prefix=f"{prefix}_base")


def _render_prediction_adjustments(study):
    meta = study["metadata"]
    st.markdown("### Add the simple up/down models to the same starting portfolio")
    if not meta.get("directional_adjustment"):
        st.info("This study contains the grouped strategy only. Choose a saved directional-overlay study to compare prediction adjustments.")
        return
    registry = default_model_registry()
    options = {registry[key].label: key for key in registry if key in meta["model_keys"]}
    model_label = _select("Prediction model for basket adjustments", list(options), "basket_prediction_model")
    model = options[model_label]
    books = {"Top 10 positive-vol stocks per group": (TOP_VOL_VERSION, *ADJUSTED_VERSIONS),
             "Full correlation groups": (FULL_GROUP, *BASKET_ADJUSTED_VERSIONS)}
    books = {label: versions for label, versions in books.items() if any(
        study["metrics"].version.isin(versions[1:]))}
    versions = books[_select("Starting grouped portfolio", list(books), "basket_overlay_book")]
    available = study["metrics"].loc[lambda f: f.version.isin(versions[1:]) & f.model_key.eq(model)]
    horizons = sorted(int(value) for value in available.momentum_months.unique())
    names = {f"{month}-month momentum": month for month in horizons}
    selected = names[_select("Momentum window", list(names), "basket_horizon")]
    strength, threshold = meta["adjustment_strength"], meta["upside_threshold"]
    st.markdown(f"**Downside to cash:** original momentum weight × (1 − {strength:g} × P(next return < 0)). "
                "The reduction stays in cash.\n\n"
                f"**Upside tilt:** when P(next return > 0) ≥ {threshold:.0%}, multiply its weight by {1 + strength:g}, "
                "then rescale stock positions to the original equity budget. Baseline cash is preserved.")
    st.write("Both begin with the exact same grouped momentum weights. Excluded and non-positive-momentum stocks stay at zero. "
             "The two directions are independently trained, separate portfolio versions; upside probability is not 1 minus downside probability.")
    recipe = meta["forecast_sources"]["loss"]["feature_spec"]
    st.caption(f"{model_label} · {len(recipe['columns'])} supplied factors · {meta['feature_selection']} feature selection. "
               "Fixed settings, no tuning; pooled stock-month training with annual expanding historical fits. "
               f"{METHOD_NAMES[meta.get('grouping_method', 'kmeans')]} group formation and the later directional classifiers are separate ML tasks.")
    st.caption("The grouping choice changes the starting portfolio. Both grouping methods reuse the same saved directional forecasts, "
               "so full-universe prediction scores are identical; portfolio results can differ.")
    scores = study["predictive"].loc[study["predictive"].model_key.eq(model)]
    with st.expander("Does the directional model predict usefully?", expanded=False):
        st.caption("Scores cover the full training-validation forecast universe, not just grouped or screened stocks. "
                   "Lower Brier/log loss is better; higher ROC AUC ranks events better. Each direction includes its historical-frequency control.")
        st.dataframe(scores, use_container_width=True, hide_index=True)
    filtered = {**study, **{name: study[name].loc[study[name].model_key.isin(["none", model])].copy()
                           for name in ("metrics", "monthly")}}
    _performance(filtered, selected, ["Equal weight", *versions], "basket_curves")
    if st.checkbox("Inspect original weights, predictions and adjusted weights", key="basket_overlay_inspect"):
        details = detail_frames(study, selected, list(versions), model)
        _baskets({**filtered, **details}, selected)
    with st.expander("Forecast and study provenance"):
        st.json(meta["forecast_sources"])

def _study_label(item):
    meta = item["metadata"]
    horizons = meta.get("momentum_months") or [meta.get("lookback_months", 12)]
    windows = ", ".join(f"{months}m" for months in horizons)
    overlay = " · downside/upside ML" if meta.get("directional_adjustment") else ""
    return f"{windows} momentum · {meta['n_clusters']} baskets{overlay} · {meta['created_utc'][:16]} · {item['directory'].name}"


def _normalize(study):
    return normalize_study(study)


def _horizons(study):
    values = study["metrics"].momentum_months
    return sorted(int(value) for value in values.dropna().unique() if int(value) > 0)


def _slice(frame, months):
    own = frame.momentum_months.eq(months)
    benchmark = pd.Series(False, index=frame.index)
    if "version" in frame.columns and frame.version.eq("Equal weight").any():
        if frame.momentum_months.eq(0).any():
            benchmark = frame.version.eq("Equal weight") & frame.momentum_months.eq(0)
        else:
            benchmark = frame.version.eq("Equal weight")
    return frame.loc[own | benchmark].copy()


def _screen_sentence(meta):
    count = meta.get("top_positive_vol_count")
    months = meta.get("top_positive_vol_months") or []
    if not count or not months:
        return ""
    windows = " and ".join(f"{int(month)}-month" for month in months)
    return (
        f"A second book keeps the {int(count)} highest positive-vol names in each group and trades {windows} momentum. "
        "Positive volatility is the root-mean-square of up months over that 12-month path. Down months contribute zero. "
        "The smaller book is scaled to the names that remain. ")


def _has_top_vol(study, months):
    metrics = study["metrics"]
    return bool((metrics.version.eq(TOP_VOL_VERSION) & metrics.momentum_months.eq(months)).any())


def _top_sentence(months):
    return (
        f"The top-10 book keeps the 10 highest positive-vol names in each group, "
        f"then applies this {months}-month rule to that smaller set. ")


def _overview_table(metrics):
    keep = ["Equal weight", "Basket momentum + cash", *BASKET_ADJUSTED_VERSIONS, TOP_VOL_VERSION, *ADJUSTED_VERSIONS]
    rows = metrics.loc[metrics.version.isin(keep)].copy()
    order = {version: index for index, version in enumerate(keep)}
    rows["_order"] = rows.version.map(order)
    rows = rows.sort_values(["momentum_months", "_order"])

    def book_name(row):
        months = int(row.momentum_months)
        if row.version == "Equal weight":
            return "Equal weight"
        if row.version == TOP_VOL_VERSION:
            return f"{months}-month top 10"
        if row.version in (*ADJUSTED_VERSIONS, *BASKET_ADJUSTED_VERSIONS):
            return f"{months}-month {row.version}"
        return f"{months}-month baskets"

    rows["book"] = rows.apply(book_name, axis=1)
    rows["window"] = rows.momentum_months.map(lambda value: "—" if int(value) == 0 else f"{int(value)} months")
    show = rows[["book", "window", "cagr", "annualized_log_growth", "annualized_volatility", "sharpe",
                 "max_drawdown", "avg_one_way_turnover", "avg_cash_weight"]]
    return show.rename(columns={
        "book": "Book", "window": "Momentum", "cagr": "CAGR", "annualized_log_growth": "Annual log growth",
        "annualized_volatility": "Volatility", "sharpe": "Sharpe", "max_drawdown": "Max drawdown",
        "avg_one_way_turnover": "Turnover", "avg_cash_weight": "Average cash"})


def _metric_table(metrics):
    rows = metrics.copy()
    rows["version"] = pd.Categorical(rows.version, list(DISPLAY_ORDER), ordered=True)
    rows = rows.sort_values("version")
    show = rows[["version", "cagr", "annualized_log_growth", "annualized_volatility", "sharpe",
                 "max_drawdown", "avg_one_way_turnover", "avg_cash_weight"]]
    return show.rename(columns={
        "version": "Version", "cagr": "CAGR", "annualized_log_growth": "Annual log growth",
        "annualized_volatility": "Volatility", "sharpe": "Sharpe", "max_drawdown": "Max drawdown",
        "avg_one_way_turnover": "Turnover", "avg_cash_weight": "Average cash"})


def _horizon_figure(monthly, title):
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.06, row_heights=[.5, .25, .25])
    series = [("Equal weight", monthly.loc[monthly.version.eq("Equal weight")], COLORS["Equal weight"], "solid")]
    baskets = monthly.loc[monthly.version.eq("Basket momentum + cash")]
    for months in sorted(int(value) for value in baskets.momentum_months.unique()):
        series.append((f"{months}-month baskets", baskets.loc[baskets.momentum_months.eq(months)],
                       HORIZON_COLORS.get(months, "#333333"), "solid"))
    if monthly.version.eq(TOP_VOL_VERSION).any():
        top = monthly.loc[monthly.version.eq(TOP_VOL_VERSION)]
        for months in sorted(int(value) for value in top.momentum_months.unique()):
            series.append((f"{months}-month top 10", top.loc[top.momentum_months.eq(months)],
                           HORIZON_COLORS.get(months, "#333333"), "dash"))
    for version in (*BASKET_ADJUSTED_VERSIONS, *ADJUSTED_VERSIONS):
        adjusted = monthly.loc[monthly.version.eq(version)]
        for months in sorted(int(value) for value in adjusted.momentum_months.unique()):
            if version in BASKET_ADJUSTED_VERSIONS:
                dash = "dash" if version == BASKET_DOWNSIDE_VERSION else "dot"
            else:
                dash = "dot" if months == 2 else "solid"
            series.append((f"{months}-month {version}", adjusted.loc[adjusted.momentum_months.eq(months)],
                           COLORS[version], dash))
    for name, rows, color, dash in series:
        rows = rows.sort_values("return_month")
        if rows.empty:
            continue
        x = [rows.snapshot_date.iloc[0], *rows.return_month]
        line = dict(color=color, dash=dash)
        figure.add_trace(go.Scatter(x=x, y=[1, *rows.wealth], name=name, line=line), row=1, col=1)
        figure.add_trace(go.Scatter(x=x, y=[0, *rows.drawdown], name=name, showlegend=False, line=line), row=2, col=1)
        figure.add_trace(go.Scatter(x=rows.return_month, y=rows.cash_weight, name=name, showlegend=False, line=line), row=3, col=1)
    figure.update_yaxes(title_text="Growth of $1", row=1, col=1)
    figure.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    figure.update_yaxes(title_text="Cash", tickformat=".0%", row=3, col=1)
    figure.update_layout(template="plotly_white", height=780, hovermode="x unified",
                         legend=dict(orientation="h", y=-.08), margin=dict(l=10, r=10, t=48, b=10),
                         title=title)
    return figure


def _figure(monthly, title):
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.06, row_heights=[.5, .25, .25])
    for version in DISPLAY_ORDER:
        rows = monthly.loc[monthly.version.eq(version)].sort_values("return_month")
        if rows.empty:
            continue
        x = [rows.snapshot_date.iloc[0], *rows.return_month]
        if version == TOP_VOL_VERSION:
            dash = "dash"
        elif version in BASKET_ADJUSTED_VERSIONS:
            dash = "dot"
        else:
            dash = "solid"
        line = dict(color=COLORS[version], dash=dash)
        figure.add_trace(go.Scatter(x=x, y=[1, *rows.wealth], name=version, line=line), row=1, col=1)
        figure.add_trace(go.Scatter(x=x, y=[0, *rows.drawdown], name=version, showlegend=False, line=line), row=2, col=1)
        figure.add_trace(go.Scatter(x=rows.return_month, y=rows.cash_weight, name=version, showlegend=False, line=line), row=3, col=1)
    figure.update_yaxes(title_text="Growth of $1", row=1, col=1)
    figure.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    figure.update_yaxes(title_text="Cash", tickformat=".0%", row=3, col=1)
    figure.update_layout(template="plotly_white", height=780, hovermode="x unified",
                         legend=dict(orientation="h", y=-.08), margin=dict(l=10, r=10, t=48, b=10),
                         title=title)
    return figure


def _cash_note(monthly, version="Basket momentum + cash", label="This grouped book"):
    book = monthly.loc[monthly.version.eq(version)].sort_values("return_month")
    if book.empty:
        return None
    periods = list(pd.to_datetime(book.return_month).dt.to_period("M"))
    cash = book.cash_weight.ge(1 - 1e-9).to_numpy()
    runs, start = [], None
    for index, is_cash in enumerate(cash):
        if is_cash and start is None:
            start = index
        ended = start is not None and (not is_cash or index == len(cash) - 1)
        if ended:
            end = index if is_cash else index - 1
            runs.append((periods[start], periods[end], end - start + 1))
            start = None
    if not runs:
        return f"{label} kept some stock exposure in every earned month."
    longest = max(runs, key=lambda item: item[2])
    span = str(longest[0]) if longest[2] == 1 else f"{longest[0]} through {longest[1]}"
    return (f"{label} was entirely in cash for {int(cash.sum())} of {len(book)} earned months. "
            f"The longest stretch is {span}.")


def _version_rows(frame, version):
    if "version" not in frame.columns:
        return frame
    return frame.loc[frame.version.eq(version)].copy()


def _baskets(study, months, key_prefix="basket"):
    st.markdown("**The groups in one decision month**")
    baskets = _slice(study["baskets"], months)
    assignments = _slice(study["assignments"], months)
    options = []
    full = _version_rows(baskets, FULL_GROUP)
    if not full.empty:
        options.append(("Full correlation group", full, _version_rows(assignments, FULL_GROUP)))
    top_baskets = _version_rows(baskets, TOP_VOL_VERSION) if "version" in baskets.columns else baskets.iloc[:0]
    if not top_baskets.empty:
        options.append(("Top 10 positive vol", top_baskets, _version_rows(assignments, TOP_VOL_VERSION)))
    for version in (*BASKET_ADJUSTED_VERSIONS, *ADJUSTED_VERSIONS):
        rows = _version_rows(baskets, version)
        if not rows.empty:
            options.append((version, rows, _version_rows(assignments, version)))
    if not options:
        st.info("No saved group/weight inspection for this combination.")
        return
    labels = [label for label, _, _ in options]
    default = labels.index("Top 10 positive vol") if "Top 10 positive vol" in labels else 0
    if len(labels) > 1:
        choice = _select("Basket book", labels, f"{key_prefix}_screen_{months}", default)
    else:
        choice = labels[0]
    _, baskets, assignments = next(item for item in options if item[0] == choice)
    if choice == "Top 10 positive vol" or choice in ADJUSTED_VERSIONS:
        st.caption(
            "These are the 10 names in each group with the highest positive volatility. "
            "Positive volatility is the root-mean-square of up months over the trailing 12 earned months. "
            "Down months contribute zero. A row with weight zero was screened in, then left in cash by the momentum rule. "
            "Names that missed the 10 are out of this book.")
    elif choice in BASKET_ADJUSTED_VERSIONS:
        st.caption(
            "These are the full correlation groups after the up-or-down weight change. "
            "A group with negative short-window momentum stays in cash. "
            "Inside a positive group, only positive-momentum names start with weight. "
            "Downside to cash reduces those weights and leaves the reduction in cash. "
            "Upside tilt rescales qualifying names inside the original stock budget.")
    else:
        st.caption(
            f"`{FEATURE}` is the saved learned group from the 12-month path. The number is reused next month for a different set of stocks. "
            f"The bars are {months}-month group momentum. Green groups can hold positive members. Gray groups are cash.")
    baskets = baskets.copy()
    baskets["date"] = pd.to_datetime(baskets.date)
    choices = {pd.Timestamp(date).strftime("%Y-%m-%d"): pd.Timestamp(date)
               for date in baskets.date.drop_duplicates().sort_values()}
    if not choices:
        st.caption("This book has no groups in the saved window.")
        return
    snapshot = choices[st.selectbox("Decision snapshot", list(choices), index=len(choices) - 1, key=f"{key_prefix}_month")]
    month = baskets.loc[baskets.date.eq(snapshot)].sort_values("basket_momentum", ascending=False)
    if month.empty:
        st.caption("No names passed the screen on this snapshot.")
        return
    colors = ["#287d68" if invested else "#68716e" for invested in month.invested]
    figure = go.Figure(go.Bar(
        x=[f"Group {int(value)}" for value in month[FEATURE]],
        y=month.basket_momentum,
        marker_color=colors,
        customdata=np_columns(month),
        hovertemplate="%{x}<br>group momentum %{y:.1%}<br>%{customdata[0]} stocks, %{customdata[1]} positive"
                      "<br>invested %{customdata[2]:.1%}<br>left in cash %{customdata[3]:.1%}<extra></extra>",
    ))
    figure.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=10, b=10),
                         yaxis_title=f"{months}-month group return", yaxis_tickformat=".0%")
    st.plotly_chart(figure, use_container_width=True, key=f"{key_prefix}_month_chart")
    show = month[[FEATURE, "n_stocks", "n_positive", "basket_momentum", "equity_weight", "sleeve_cash"]].rename(columns={
        FEATURE: "Group", "n_stocks": "Stocks", "n_positive": "Positive momentum",
        "basket_momentum": "Group momentum", "equity_weight": "Invested", "sleeve_cash": "Left in cash"})
    st.dataframe(show.style.format({"Group momentum": "{:.2%}", "Invested": "{:.2%}", "Left in cash": "{:.2%}"}),
                 use_container_width=True, hide_index=True)
    names = assignments.loc[pd.to_datetime(assignments.date).eq(snapshot)].copy()
    if choice in BASKET_ADJUSTED_VERSIONS:
        probability = "probability_of_loss" if choice == BASKET_DOWNSIDE_VERSION else "probability_of_up"
        columns = ["stock_id", FEATURE, "trailing_momentum", "basket_momentum",
                   "baseline_weight", probability, "weight"]
        visible = names.loc[names.baseline_weight.gt(0) | names.weight.gt(0), columns]
        visible = visible.sort_values(["weight", "stock_id"], ascending=[False, True]).rename(columns={
            "stock_id": "Stock", FEATURE: "Group", "trailing_momentum": "Stock momentum",
            "basket_momentum": "Group momentum", "baseline_weight": "Original momentum weight",
            "probability_of_loss": "Predicted loss probability", "probability_of_up": "Predicted up probability",
            "weight": "Weight"})
        held = int(names.weight.gt(0).sum())
        st.caption(
            f"{held} stocks are held at this snapshot. "
            "The table keeps names that had momentum weight before the adjustment.")
        st.dataframe(visible.style.format({
            "Stock momentum": "{:.2%}", "Group momentum": "{:.2%}", "Original momentum weight": "{:.2%}",
            "Predicted loss probability": "{:.2%}", "Predicted up probability": "{:.2%}", "Weight": "{:.2%}"}),
            use_container_width=True, hide_index=True)
        return
    if choice == "Top 10 positive vol" or choice in ADJUSTED_VERSIONS:
        names = selected_vol_ranks(names, snapshot)
        columns = ["stock_id", FEATURE, "Positive-vol rank in group", "positive_volatility", "trailing_momentum", "basket_momentum", "weight"]
        if choice in ADJUSTED_VERSIONS:
            probability = "probability_of_loss" if choice == DOWNSIDE_VERSION else "probability_of_up"
            columns += ["baseline_weight", probability]
        visible = names[columns].sort_values(
            ["positive_volatility", "weight", "stock_id"], ascending=[False, False, True])
        visible = visible.rename(columns={
            "stock_id": "Stock", FEATURE: "Group", "positive_volatility": "Positive vol",
            "trailing_momentum": "Stock momentum", "basket_momentum": "Group momentum", "weight": "Weight",
            "baseline_weight": "Original momentum weight", "probability_of_loss": "Predicted loss probability",
            "probability_of_up": "Predicted up probability"})
        held = int(names.weight.gt(0).sum())
        st.caption(
            f"{len(visible)} names passed the positive-vol screen at this snapshot. "
            f"{held} of them are held. A zero weight is cash inside this smaller book.")
        st.dataframe(visible.style.format({
            "Positive vol": "{:.2%}", "Stock momentum": "{:.2%}", "Group momentum": "{:.2%}", "Weight": "{:.2%}",
            **({"Original momentum weight": "{:.2%}", "Predicted loss probability": "{:.2%}",
                "Predicted up probability": "{:.2%}"} if choice in ADJUSTED_VERSIONS else {})}),
            use_container_width=True, hide_index=True)
        return
    visible = names.loc[names.weight.gt(0), ["stock_id", FEATURE, "trailing_momentum", "basket_momentum", "weight"]]
    visible = visible.sort_values(["weight", "stock_id"], ascending=[False, True]).rename(columns={
        "stock_id": "Stock", FEATURE: "Group", "trailing_momentum": "Stock momentum",
        "basket_momentum": "Group momentum", "weight": "Weight"})
    st.caption(f"{len(visible)} stocks are held at this snapshot under {months}-month momentum. Everyone else is cash.")
    st.dataframe(visible.style.format({"Stock momentum": "{:.2%}", "Group momentum": "{:.2%}", "Weight": "{:.2%}"}),
                 use_container_width=True, hide_index=True)


def np_columns(month):
    return list(zip(month.n_stocks.astype(int), month.n_positive.astype(int),
                    month.equity_weight.astype(float), month.sleeve_cash.astype(float)))
