"""Grouped presentation of recorded portfolio metrics and ML changes."""
import pandas as pd
import streamlit as st
from html import escape

from strategy_groups import DELTA_METRICS, change_format, change_matrix, grouped_strategies
from strategy_help import strategy_descriptions


def baseline_table_html(result, rows, descriptions=None):
    """Small read-only table with real per-strategy hover/focus explanations.

    Streamlit's canvas dataframe supports header help, not cell-specific help.
    Keep the same metric formatter and escape all labels/descriptions before HTML.
    """
    descriptions = descriptions or strategy_descriptions(result.config)
    by_label = {row.label: descriptions.get(row.strategy, "No description recorded.")
                for row in rows.itertuples()}

    def name(label):
        help_text = escape(by_label[label], quote=True)
        return (f'<span class="strategy-help" tabindex="0" title="{help_text}" '
                f'aria-label="{escape(label, quote=True)}. {help_text}">'
                f'{escape(label)} <span aria-hidden="true">ⓘ</span>'
                f'<span class="strategy-tooltip" role="tooltip">{help_text}</span></span>')

    styled = metrics_table(result, rows, baseline_only=True).hide(axis="index")
    styled = styled.format(name, subset=["Strategy"]).set_table_attributes('class="strategy-metrics"')
    # Metric values keep their existing formats and precision; only the name cell
    # changes renderer. Escape arbitrary text outside the explicit help cell.
    for column in styled.data.select_dtypes(include="object"):
        if column != "Strategy":
            styled = styled.format(lambda value: escape(str(value)), subset=[column])
    return ('<style>.strategy-table {overflow-x:auto; margin-bottom:1rem;}'
        '.strategy-metrics {width:100%;border-collapse:collapse;font-size:.88rem;}'
        '.strategy-metrics th,.strategy-metrics td {padding:.65rem .8rem;text-align:right;white-space:nowrap;'
        'border-bottom:1px solid #dce2df;}'
        '.strategy-metrics th:first-child,.strategy-metrics td:first-child {text-align:left;}'
        '.strategy-metrics th {background:var(--secondary-background-color,#f3f5f3);}'
        '.strategy-help {cursor:help;text-decoration:underline dotted;text-underline-offset:4px;}'
        '.strategy-tooltip {display:none;position:fixed;bottom:24px;left:24px;z-index:1000000;'
        'max-width:min(640px,calc(100vw - 48px));padding:16px;border-radius:8px;'
        'background:#193d32;color:white;white-space:normal;line-height:1.5;'
        'box-shadow:0 4px 24px #0003;font-weight:400;text-align:left;pointer-events:none;}'
        '.strategy-help:hover .strategy-tooltip,.strategy-help:focus .strategy-tooltip {display:block;}'
        '</style><div class="strategy-table">' + styled.to_html() + '</div>')


def metrics_table(result, rows=None, hide_strategy=False, baseline_only=False):
    rows = result.metrics if rows is None else rows
    fields = {"label": "Strategy", "model_label": "Version / model", "cagr": "CAGR",
        "annualized_log_growth": "Annual log growth",
        "annualized_volatility": "Volatility", "sharpe": "Sharpe", "sortino": "Sortino",
        "max_drawdown": "Max drawdown", "avg_one_way_turnover": "Monthly turnover",
        "avg_names": "Average stocks", "largest_single_weight": "Largest weight"}
    if hide_strategy:
        fields.pop("label")
        fields.pop("avg_names")
        fields.pop("largest_single_weight")
    if baseline_only:
        fields.pop("model_label")
    table = rows[list(fields)].rename(columns=fields)
    if "avg_cash_weight" in result.metrics:
        table["Average cash"] = rows.avg_cash_weight.fillna(0)
    return table.style.format({**{k: "{:.2%}" for k in ["CAGR", "Annual log growth", "Volatility", "Max drawdown", "Monthly turnover", "Largest weight", "Average cash"]},
        "Sharpe": "{:.3f}", "Sortino": "{:.3f}", "Average stocks": "{:.1f}"}, na_rep="—")


def render_metric_groups(result, strategies, grouped=True, baseline_only=False, descriptions=None, nested=False):
    rows = result.metrics.loc[result.metrics.strategy.isin(strategies)]
    if not grouped:
        if baseline_only:
            st.markdown(baseline_table_html(result, rows, descriptions), unsafe_allow_html=True)
        else:
            st.dataframe(metrics_table(result, rows), use_container_width=True, hide_index=True)
        return
    groups = grouped_strategies(strategies)
    for index, (family, keys) in enumerate(groups):
        noun = "strategy" if len(keys) == 1 else "strategies"
        # Streamlit 1.32 prohibits an expander inside another expander.
        with (st.container(border=True) if nested else
              st.expander(f"{family.title} · {len(keys)} {noun}", expanded=index == 0)):
            if nested:
                st.markdown(f"#### {family.title} · {len(keys)} {noun}")
            st.caption(family.description)
            if baseline_only:
                st.markdown(baseline_table_html(result, rows.loc[rows.strategy.isin(keys)], descriptions), unsafe_allow_html=True)
                continue
            for key in keys:
                subset = rows.loc[rows.strategy.eq(key)]
                st.markdown(f"**{subset.label.iloc[0]}**")
                st.dataframe(metrics_table(result, subset, hide_strategy=True),
                    height=35 * (len(subset) + 1) + 3, use_container_width=True, hide_index=True)


def _cell_style(value, limit, higher_better):
    if pd.isna(value) or abs(value) < 1e-14 or not limit:
        return "color: #66756e"
    favorable = value > 0 if higher_better else value < 0
    rgb = "35, 122, 88" if favorable else "180, 78, 68"
    alpha = .06 + .15 * min(abs(value) / limit, 1)
    color = "#176347" if favorable else "#984537"
    return f"background-color: rgba({rgb}, {alpha:.3f}); color: {color}; font-weight: 600"


def render_paired_groups(result, strategies, grouped=True, key_prefix="paired", nested=False):
    rows = result.paired.loc[result.paired.strategy.isin(strategies)]
    if rows.empty:
        return
    if not nested:
        st.subheader("Change from each original portfolio")
    st.caption("Each ML version is compared with its own non-ML original over the same months.")
    if not grouped:
        _detail_table(rows)
    else:
        selected = st.selectbox("Compare the ML change in", list(DELTA_METRICS), key=f"{key_prefix}_metric")
        metric, points, higher_better = DELTA_METRICS[selected]
        limit = rows[metric].abs().max()
        st.caption("One row per strategy · one column per model · pp = percentage points. Green indicates the preferred direction for this metric; red indicates the opposite.")
        if metric == "change_max_drawdown":
            st.caption("A positive change means a shallower worst drawdown.")
        elif metric == "change_avg_one_way_turnover":
            st.caption("Lower turnover is shaded green. It is a trading-cost consideration, not a return improvement.")
        for family, keys in grouped_strategies(strategies):
            subset = rows.loc[rows.strategy.isin(keys)]
            if subset.empty:
                continue
            with st.container(border=True):
                st.markdown(f"#### {family.title}")
                st.caption(family.description)
                matrix = change_matrix(subset, metric)
                styled = matrix.style.format(lambda value: change_format(value, points), na_rep="—")
                styled = styled.map(lambda value: _cell_style(value, limit, higher_better))
                st.dataframe(styled, height=35 * (len(matrix) + 1) + 3,
                    column_config={"_index": st.column_config.TextColumn("Strategy", width="large")},
                    use_container_width=True)
        inspect = not nested or st.checkbox("Inspect one strategy · every ML change", key=f"{key_prefix}_show_detail")
        if inspect:
            with (st.container() if nested else st.expander("Inspect one strategy · every ML change")):
                labels = rows.drop_duplicates("strategy").set_index("label").strategy.to_dict()
                detail_key = f"{key_prefix}_detail_strategy"
                if st.session_state.get(detail_key) not in labels:
                    st.session_state.pop(detail_key, None)
                strategy = labels[st.selectbox("Strategy to inspect", list(labels), key=detail_key)]
                base = result.metrics.loc[result.metrics.strategy.eq(strategy) & result.metrics.variant.eq("baseline")].iloc[0]
                st.caption(f"Original reference: CAGR {base.cagr:.2%} · Sharpe {base.sharpe:.3f} · Sortino {base.sortino:.3f} · Max drawdown {base.max_drawdown:.2%}.")
                _detail_table(rows.loc[rows.strategy.eq(strategy)], hide_strategy=True)
    st.download_button("Download ML changes (CSV)", rows.to_csv(index=False),
        file_name="ml_changes_by_strategy.csv", mime="text/csv", key=f"{key_prefix}_download")


def _detail_table(rows, hide_strategy=False):
    fields = {"label": "Strategy", "model_label": "Model", "change_cagr": "CAGR change",
        "change_sharpe": "Sharpe change", "change_sortino": "Sortino change",
        "change_max_drawdown": "Drawdown change", "change_avg_one_way_turnover": "Turnover change"}
    if hide_strategy:
        fields.pop("label")
    table = rows[list(fields)].rename(columns=fields)
    st.dataframe(table.style.format({"CAGR change": lambda x: change_format(x, True),
        "Sharpe change": "{:+.3f}", "Sortino change": "{:+.3f}",
        "Drawdown change": lambda x: change_format(x, True),
        "Turnover change": lambda x: change_format(x, True)}),
        height=min(650, 35 * (len(table) + 1) + 3), use_container_width=True, hide_index=True)
