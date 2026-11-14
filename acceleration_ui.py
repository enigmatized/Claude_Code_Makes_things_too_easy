"""Saved training/final views of the acceleration-to-cash experiment."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from app_charts import RETURN_MEASURES, return_curve
from final_comparison_ui import offer_final_comparison
from research.momentum_acceleration import METHODS, VARIANTS, load_acceleration_study


def acceleration_available(store=None, published=None, window="training"):
    if published is not None:
        return published.has_window(window)
    return bool(
        list(
            (Path(store.root) / "acceleration_studies").glob(
                f"*/{window}/complete.json"
            )
        )
    )


def render_acceleration(store=None, published=None, window="training"):
    if not acceleration_available(store, published, window):
        st.info(
            "No saved acceleration comparison is included yet. Compute it explicitly in research mode with python3 run_momentum_acceleration.py."
        )
        return
    if published is not None:
        study = published.study(window)
    else:
        paths = sorted(
            (Path(store.root) / "acceleration_studies").glob(
                f"*/{window}/complete.json"
            ),
            reverse=True,
        )
        directory = paths[0].parent.parent
        token = (str(directory), window, paths[0].read_text())
        key = "loaded_acceleration_" + window
        if st.session_state.get(key, (None,))[0] != token:
            st.session_state[key] = (token, load_acceleration_study(directory, window))
        study = st.session_state[key][1]
    prefix = "acceleration_" + window
    scope = study["scope"]
    st.subheader("Momentum acceleration — extra deceleration-to-cash rules")
    st.info(
        (
            "**Training data only.**"
            if window == "training"
            else "**Official test; previously viewed.**"
        )
        + f" {scope['first_return_month']} to {scope['last_return_month']} · {scope['months']} months. Saved results only."
    )
    st.write(
        "The original book takes the top 10 positive-volatility stocks per group. Group momentum is the "
        "equal-weight mean return of those selected names. Portfolio positions use only positive-momentum stocks "
        "inside positive-momentum groups; stock weights are proportional to momentum, with unused capital in cash. "
        "Groups and held stocks are therefore not allocated equal portfolio weights."
    )
    st.caption(
        "For example, if 50 stocks pass the screen and a positive basket has 6 positive-momentum members, "
        "that basket invests 6/50 = 12% of total capital. Its six positions share that 12% in proportion "
        "to their positive momentum. Other slots remain in cash."
    )
    st.latex(
        r"a_{i,t}=r_{i,t}-r_{i,t-1},\qquad a_{g,t}=\overline{r}_{g,t}-\overline{r}_{g,t-1}"
    )
    st.caption(
        "All returns in these signals are already earned. The basket uses its current selected members for both "
        "months; it does not subtract returns for two different memberships or join changing group IDs."
    )
    st.write(
        "A positive return can be slowing: +10% last month followed by +4% now gives −6 percentage points of "
        "acceleration. The original rule still permits the stock; the new rule can reduce its existing weight."
    )
    with st.expander("Declared rule — how deceleration moves allocation to cash"):
        st.write(
            "Stock-only: halve an existing weight when stock acceleration is negative. Basket-only: halve all "
            "existing weights in a basket when its acceleration is negative. Combined: apply both multipliers; "
            "a stock slowing inside a slowing basket keeps one quarter of its original allocation."
        )
        st.write(
            "No negative positions or shorts are introduced. Money removed stays in zero-return cash, with no "
            "renormalization. Zero/positive acceleration preserves the original weight. Missing comparisons "
            "leave the weight unchanged and remain visible as missing signals."
        )
        st.caption(
            "The 50% haircut and two-month comparison were declared before this run; no tuning or predictive "
            "classifier is involved. Cash protection can lower returns, and month-to-month changes can be noisy "
            "or duplicate information already present in momentum. This is a hypothesis, not an assumed improvement."
        )
    method = dict((label, key) for key, label in METHODS.items())[
        st.selectbox(
            "1. Asset basket — stock grouping method", list(METHODS.values()), key=prefix + "_method"
        )
    ]
    rows = study["metrics"].loc[study["metrics"].method.eq(method)].copy()
    rows["variant"] = pd.Categorical(rows.variant, VARIANTS, ordered=True)
    rows = rows.sort_values("variant")
    columns = [
        "variant",
        "cagr",
        "annualized_log_growth",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "avg_cash_weight",
        "avg_one_way_turnover",
    ]
    table = rows[columns].rename(
        columns={
            "variant": "Rule",
            "cagr": "CAGR",
            "annualized_log_growth": "Annual log growth",
            "annualized_volatility": "Volatility",
            "sharpe": "Sharpe",
            "max_drawdown": "Max drawdown",
            "avg_cash_weight": "Average cash",
            "avg_one_way_turnover": "Turnover",
        }
    )
    st.dataframe(
        table.style.format(
            {
                name: "{:.3f}" if name == "Sharpe" else "{:.2%}"
                for name in table.columns[1:]
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    measure = st.selectbox(
        "Return measure", list(RETURN_MEASURES), key=prefix + "_return"
    )
    monthly = study["monthly"].loc[study["monthly"].method.eq(method)]
    st.caption('2. Allocation strategy: 1M positive momentum + cash; compare the four deceleration rules below. No directional classifier.')
    offer_final_comparison(rows,monthly,protocol='Fixed acceleration rules',source=study['metadata'].get('created_utc','acceleration'),
        case_fields=['method','variant'],prefix=prefix,window=window)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.5, 0.25, 0.25],
    )
    for variant, color in zip(VARIANTS, ["#68716e", "#287d68", "#536da6", "#b36a2e"]):
        frame = monthly.loc[monthly.variant.eq(variant)].sort_values("return_month")
        book = frame.set_index("return_month").copy()
        book["log_return"] = np.log1p(book.simple_return)
        curve = return_curve(book, measure)
        fig.add_trace(
            go.Scatter(x=curve.index, y=curve, name=variant, line=dict(color=color)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=frame.return_month,
                y=frame.drawdown,
                showlegend=False,
                line=dict(color=color),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=frame.return_month,
                y=frame.cash_weight,
                showlegend=False,
                line=dict(color=color),
            ),
            row=3,
            col=1,
        )
    fig.update_yaxes(title_text=measure, row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_yaxes(title_text="Cash", tickformat=".0%", row=3, col=1)
    fig.update_xaxes(title_text="Earned month", row=3, col=1)
    fig.update_layout(
        template="plotly_white",
        height=800,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.12),
        margin=dict(b=110),
    )
    st.plotly_chart(fig, use_container_width=True, key=prefix + "_curves")
    with st.expander("Inspect basket/stock acceleration and actual allocation"):
        signals = study["signals"].loc[study["signals"].method.eq(method)]
        dates = sorted(pd.to_datetime(signals.date).unique())
        choices = {pd.Timestamp(d).strftime("%Y-%m-%d"): pd.Timestamp(d) for d in dates}
        if not choices:
            st.info(
                "No stocks passed the top-ten screen in this saved window; the book holds cash."
            )
            return
        date = choices[
            st.selectbox(
                "Decision snapshot",
                list(choices),
                index=len(choices) - 1,
                key=prefix + "_snapshot",
            )
        ]
        month = signals.loc[pd.to_datetime(signals.date).eq(date)].copy()
        groups = (
            month.groupby("correlation_basket")
            .agg(
                stocks=("stock_id", "size"),
                current_return=("basket_return_now", "first"),
                previous_return=("basket_return_previous", "first"),
                acceleration=("basket_acceleration", "first"),
                original_equity=("original_weight", "sum"),
                adjusted_equity=("Stock + basket deceleration", "sum"),
            )
            .reset_index()
        )
        # Compare the same current members in both months, never changing cluster IDs.
        comparison = go.Figure()
        for field, label, color in [
            ("previous_return", "Previous month", "#899594"),
            ("current_return", "Latest earned month", "#287d68"),
        ]:
            comparison.add_bar(
                x=groups.correlation_basket.astype(str),
                y=groups[field],
                name=label,
                marker_color=color,
                customdata=groups.acceleration,
                hovertemplate="Group %{x}<br>Return %{y:.2%}<br>Acceleration %{customdata:.2%}<extra>%{fullData.name}</extra>",
            )
        comparison.update_layout(
            template="plotly_white",
            barmode="group",
            title="Is each current basket speeding up or slowing down?",
            xaxis_title="Group at selected decision snapshot",
            yaxis=dict(title="Equal-weight basket return", tickformat=".0%"),
        )
        st.plotly_chart(
            comparison, use_container_width=True, key=prefix + "_basket_comparison"
        )
        st.caption(
            "Bars use the same current members in both months. The gap between latest and previous "
            "return is basket acceleration. Group numbers have no persistent identity across snapshots."
        )
        st.dataframe(
            groups.style.format(
                {
                    c: "{:.2%}"
                    for c in groups.columns
                    if c not in ["correlation_basket", "stocks"]
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        show = month[
            [
                "stock_id",
                "correlation_basket",
                "stock_return_previous",
                "stock_return_now",
                "stock_acceleration",
                "basket_acceleration",
                "original_weight",
                *VARIANTS[1:],
            ]
        ]
        st.dataframe(
            show.style.format(
                {
                    c: "{:.2%}"
                    for c in show.columns
                    if c not in ["stock_id", "correlation_basket"]
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    st.caption(
        "Stock and basket acceleration are saved features for inspection. Feeding them into a classifier would "
        "require a separate training/CV experiment; no existing ML model has been silently retrained here. Gross of costs; cash earns 0%."
    )
