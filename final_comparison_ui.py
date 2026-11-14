"""Session-only comparison of explicit saved test cases, retaining their protocols."""

import hashlib
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots
import streamlit as st

from app_charts import RETURN_MEASURES, return_curve
from research.models import default_model_registry


def _case_label(protocol, identity):
    models = default_model_registry()
    parts = []
    for field, value in identity.items():
        if str(value) == "none" or (field == "momentum_months" and value == 0):
            continue
        if field == "model_key":
            value = models[value].label if value in models else value
        elif field == "target":
            value = {"loss": "Next down", "upside": "Next up"}.get(value, value)
        elif field == "momentum_months":
            value = f"{int(value)}M"
        parts.append(str(value))
    return protocol + " · " + " · ".join(parts)


def offer_final_comparison(
    metrics, monthly, *, protocol, source, case_fields, prefix, window
):
    if window != "test":
        return
    if not st.button(
        "Add displayed portfolios to final comparison", key=prefix + "_add_comparison"
    ):
        return
    saved = dict(st.session_state.get("saved_final_comparisons", {}))
    for values, group in monthly.groupby(case_fields, dropna=False, sort=False):
        values = values if isinstance(values, tuple) else (values,)
        identity = dict(zip(case_fields, values))
        rows = metrics
        for field, value in identity.items():
            rows = rows.loc[rows[field].eq(value)]
        if len(rows) != 1:
            raise ValueError("A comparison case needs exactly one saved metric row.")
        dates = pd.to_datetime(group.return_month)
        if dates.min() < pd.Timestamp("2010-01-31") or dates.max() > pd.Timestamp(
            "2018-12-31"
        ):
            raise ValueError("Only official-test cases can enter this comparison.")
        token = hashlib.sha256(
            json.dumps(
                [protocol, str(source), identity], sort_keys=True, default=str
            ).encode()
        ).hexdigest()
        label = _case_label(protocol, identity)
        saved[token] = dict(
            label=label,
            protocol=protocol,
            source=str(source),
            metrics=rows.iloc[0].to_dict(),
            monthly=group.copy(),
        )
    st.session_state["saved_final_comparisons"] = saved
    # New additions should be visible immediately while preserving a prior selection.
    st.session_state.pop("final_comparison_cases", None)
    st.success(
        "Added saved cases. Open “Compare saved final portfolios” below the study tabs."
    )


def render_final_comparison():
    saved = st.session_state.get("saved_final_comparisons", {})
    with st.expander("Compare saved final portfolios", expanded=bool(saved)):
        if not saved:
            st.caption(
                "Add displayed portfolios from the K-means, feature-engineering or acceleration final sections. "
                "This comparison reads saved official-test paths; it never runs another test."
            )
            return
        if st.button("Clear final comparison", key="clear_final_comparison"):
            st.session_state["saved_final_comparisons"] = {}
            st.session_state.pop("final_comparison_cases", None)
            st.rerun()
        options = {}
        for key, entry in saved.items():
            label = entry["label"]
            if label in options:
                label += " (saved case " + str(len(options) + 1) + ")"
            options[label] = key
        labels = st.multiselect(
            "Saved portfolios to compare",
            list(options),
            default=list(options),
            key="final_comparison_cases",
        )
        entries = [saved[options[label]] for label in labels]
        if not entries:
            st.info("Choose at least one saved final portfolio.")
            return
        tables = []
        for entry in entries:
            dates = pd.to_datetime(entry["monthly"].return_month)
            tables.append(
                {
                    "Portfolio": entry["label"],
                    "Protocol": entry["protocol"],
                    "First earned month": str(dates.min().date()),
                    "Last earned month": str(dates.max().date()),
                    **{
                        field: entry["metrics"].get(field, np.nan)
                        for field in (
                            "cagr",
                            "annualized_log_growth",
                            "sharpe",
                            "max_drawdown",
                            "avg_cash_weight",
                        )
                    },
                }
            )
        table = pd.DataFrame(tables).rename(
            columns={
                "cagr": "CAGR",
                "annualized_log_growth": "Annual log growth",
                "sharpe": "Sharpe",
                "max_drawdown": "Max drawdown",
                "avg_cash_weight": "Average cash",
            }
        )
        st.dataframe(
            table.style.format(
                {
                    name: "{:.3f}" if name == "Sharpe" else "{:.2%}"
                    for name in (
                        "CAGR",
                        "Annual log growth",
                        "Sharpe",
                        "Max drawdown",
                        "Average cash",
                    )
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        if st.checkbox(
            "Inspect saved comparison provenance", key="final_comparison_provenance"
        ):
            st.json({entry["label"]: entry["source"] for entry in entries})
        calendars = [
            pd.DatetimeIndex(entry["monthly"].return_month).sort_values()
            for entry in entries
        ]
        if any(not dates.equals(calendars[0]) for dates in calendars[1:]):
            st.info(
                "These cases have different evaluation months. Select matching windows before charting; "
                "saved full-window metrics are not recalculated on an intersection."
            )
            return
        measure = st.selectbox(
            "Comparison return measure", RETURN_MEASURES, key="final_comparison_measure"
        )
        allocation_paths = all(
            {"drawdown", "cash_weight"}.issubset(entry["monthly"]) for entry in entries
        )
        fig = make_subplots(
            rows=3 if allocation_paths else 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
        )
        for index, entry in enumerate(entries):
            book = (
                entry["monthly"]
                .sort_values("return_month")
                .set_index("return_month")
                .copy()
            )
            book["log_return"] = np.log1p(book.simple_return)
            curve = return_curve(book, measure)
            line = dict(color=qualitative.Plotly[index % len(qualitative.Plotly)])
            fig.add_trace(
                go.Scatter(
                    x=curve.index,
                    y=curve,
                    name=entry["label"],
                    legendgroup=entry["label"],
                    line=line,
                ),
                row=1,
                col=1,
            )
            if allocation_paths:
                for row, field in ((2, "drawdown"), (3, "cash_weight")):
                    fig.add_trace(
                        go.Scatter(
                            x=book.index,
                            y=book[field],
                            name=entry["label"],
                            legendgroup=entry["label"],
                            showlegend=False,
                            line=line,
                        ),
                        row=row,
                        col=1,
                    )
        fig.update_yaxes(title_text=measure, row=1, col=1)
        if allocation_paths:
            fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
            fig.update_yaxes(title_text="Cash", tickformat=".0%", row=3, col=1)
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            height=800 if allocation_paths else 520,
        )
        st.plotly_chart(
            fig, use_container_width=True, key="final_saved_comparison_chart"
        )
        st.caption(
            "Each curve retains its own saved fitting protocol and original costs/cash assumptions. "
            "The test period was already viewed. Selecting a winner here does not create independent confirmation."
        )
