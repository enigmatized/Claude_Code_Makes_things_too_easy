"""Shared basket/model/allocation/target controls over completed saved cases."""

import streamlit as st

from research.models import default_model_registry
from portfolio_selection import MODEL_NONE, TARGET_LABELS, matching_cases


def choose(label, options, key, preferred=None, **kwargs):
    options = list(options)
    if not options:
        raise ValueError(f"No saved choices for {label}.")
    if st.session_state.get(key) not in options:
        st.session_state[key] = preferred if preferred in options else options[0]
    return st.selectbox(label, options, key=key, **kwargs)


def select_portfolio_case(
    cases, prefix, *, predictive=True, keys=None, preferred_basket=None
):
    """Return matching saved rows, not a newly composed or newly trained strategy.

    Model/rule choices use the union of targets; the last control resolves only
    available events. State is validated at every step when upstream choices change.
    """
    if cases.empty:
        st.info("No saved portfolio cases are available for this scope.")
        return cases.copy()
    keys = keys or {}
    key = lambda field: keys.get(field, prefix + "_" + field)
    cols = st.columns(4 if predictive else 2)
    with cols[0]:
        basket = choose(
            "1. Asset basket",
            cases.basket.drop_duplicates(),
            key("basket"),
            preferred_basket,
            help="Membership and screening of an existing saved portfolio. Selecting it does not retrain a model on this basket.",
        )
    selected = matching_cases(cases, basket=basket)
    if predictive:
        registry = default_model_registry()
        labels = {
            registry[k].label if k in registry else MODEL_NONE: k
            for k in selected.model_key.drop_duplicates()
        }
        with cols[1]:
            label = choose(
                "2. Prediction model",
                labels,
                key("model"),
                preferred="Logistic regression",
            )
        selected = matching_cases(selected, model_key=labels[label])
    with cols[2 if predictive else 1]:
        allocation = choose(
            "3. Allocation strategy" if predictive else "2. Allocation strategy",
            selected.allocation.drop_duplicates(),
            key("allocation"),
        )
    selected = matching_cases(selected, allocation=allocation)
    if predictive:
        labels = {
            label: target
            for label, target in TARGET_LABELS.items()
            if target in set(selected.target)
        }
        if not labels:
            labels = {"Not applicable — no prediction": "none"}
        with cols[3]:
            label = choose(
                "4. Prediction target",
                labels,
                key("target"),
                help="Only separately fitted, saved events are offered. Up and down are not interchangeable probabilities.",
            )
        selected = matching_cases(selected, target=labels[label])
        if len(labels) == 1 and "none" not in labels.values():
            st.caption(
                "Only "
                + next(iter(labels))
                + " is saved for this model, basket and allocation. Other directions are not evaluated in this study."
            )
    else:
        st.caption("No directional prediction model is used in this comparison.")
    if selected.empty:
        st.info(
            "This exact configuration has not been evaluated. No substitute result is loaded."
        )
    else:
        st.caption(
            "Selected basket: "
            + basket
            + " · Allocation: "
            + allocation
            + ". Saved results only; the original portfolio remains the reference."
        )
    return selected
