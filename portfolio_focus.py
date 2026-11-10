"""Presentation focus; these are portfolio families, separate from classifiers."""
import streamlit as st


def main_portfolio_focus(key):
    return st.checkbox("Show only the four main portfolio families", value=True, key=f'{key}_main_families_only',
        help="Equal weight; top-10 positive-volatility momentum; its up/down ML adjustments; "
             "full grouped-basket ML adjustments (with their original reference). "
             "Turn off to enable additional saved strategy/model/horizon comparisons. "
             "This display filter never trains models or changes results.")
