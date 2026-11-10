"""Descriptions of the recorded non-ML rules; outside calculation identities."""
from research.strategies import default_registry


def strategy_descriptions(config):
    result = {}
    for key, spec in default_registry().items():
        explanation = spec.explanation
        parts = [spec.description]
        if explanation:
            parts = [explanation.universe, explanation.selection, explanation.weighting,
                     "History: " + explanation.history, "Fallback: " + explanation.fallback]
        text = " ".join(parts).split(" The ML version")[0]
        if key == "annual_consistency_momentum_cash":
            text = text.replace("(default twelve)", f"({config.cash_momentum_months} months in this run)")
        result[key] = text
    return result
