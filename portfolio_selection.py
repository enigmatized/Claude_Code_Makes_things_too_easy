"""Presentation identities for existing saved portfolios; no new allocation math.

Selection and weighting are bundled in legacy strategy IDs. These explicit maps
describe those bundles without claiming that arbitrary pairings were backtested.
All original IDs and provenance columns remain in the returned case tables.
"""

import pandas as pd

from research.strategies import default_registry

TARGET_LABELS = {"Next down": "loss", "Next up": "upside"}
MODEL_NONE = "No directional model"

REGISTRY_PARTS = {
    "equal_weight_universe": ("All available stocks", "Equal weight"),
    "equal_weight_eligible": ("Stocks with 12 months of history", "Equal weight"),
    "high_volatility": ("Highest trailing-volatility stocks", "Equal weight"),
    "low_volatility": ("Lowest trailing-volatility stocks", "Equal weight"),
    "momentum_equal_weight": ("Strongest positive-momentum stocks", "Equal weight"),
    "momentum_weighted": ("Strongest positive-momentum stocks", "Momentum weights"),
    "training_high_vol_equal_weight": (
        "Frozen training high-volatility pool",
        "Equal weight",
    ),
    "training_high_vol_momentum_equal": (
        "Frozen training high-volatility pool",
        "Select positive momentum; equal weight",
    ),
    "training_high_vol_momentum_weighted": (
        "Frozen training high-volatility pool",
        "Select positive momentum; momentum weights",
    ),
    "hybrid_volatility_3m_sizing": (
        "Frozen training high-volatility pool",
        "Momentum / ordinary 3M volatility",
    ),
    "hybrid_downside_3m_sizing": (
        "Frozen training high-volatility pool",
        "Momentum / downside-weighted 3M risk",
    ),
    "training_sharpe_momentum_weighted": (
        "Frozen training Sharpe pool",
        "Select positive momentum; momentum weights",
    ),
    "training_sortino_momentum_weighted": (
        "Frozen training Sortino pool",
        "Select positive momentum; momentum weights",
    ),
    "annual_consistency_momentum_cash": (
        "Frozen annual-consistency pool",
        "Positive momentum + cash",
    ),
}

BOOK_PARTS = {
    "Equal weight": ("All available stocks", "Equal weight"),
    "Momentum + cash": ("Eligible stocks", "1M positive momentum + cash"),
    "K-means full groups": ("K-means full groups", "1M positive momentum + cash"),
    "K-means top 10": (
        "K-means top 10 positive-vol per group",
        "1M positive momentum + cash",
    ),
    "Ward full groups": ("Ward full groups", "1M positive momentum + cash"),
    "Ward top 10": (
        "Ward top 10 positive-vol per group",
        "1M positive momentum + cash",
    ),
}


def registry_cases(strategies, models=("none",), targets=("none",), **identity):
    registry = default_registry()
    rows = []
    for strategy in strategies:
        # Unknown future rules stay usable as intact bundles, never guessed apart.
        basket, allocation = REGISTRY_PARTS.get(
            strategy, (registry[strategy].label, "Saved allocation rule")
        )
        for model in models:
            for target in targets:
                rows.append(
                    dict(
                        identity,
                        strategy=strategy,
                        basket=basket,
                        allocation=allocation,
                        model_key=model,
                        target=target,
                    )
                )
    return pd.DataFrame(rows)


def book_cases(frame):
    """Describe existing engineered book rows, retaining recipe/window/fit identity."""
    result = frame.copy()
    result["basket"] = result.book.map(lambda book: BOOK_PARTS[book][0])
    result["allocation"] = result.book.map(lambda book: BOOK_PARTS[book][1])
    return result


def grouped_cases(metrics, method="kmeans"):
    """Describe only recorded original/directional group cases and their horizons."""
    from research.basket_momentum import TOP_VOL_VERSION
    from research.basket_adjustments import (
        DOWNSIDE_VERSION,
        UPSIDE_VERSION,
        BASKET_DOWNSIDE_VERSION,
        BASKET_UPSIDE_VERSION,
    )

    prefix = "K-means" if method == "kmeans" else "Ward"
    versions = {
        "Equal weight": ("All available stocks", "none", "Equal weight"),
        "Momentum + cash": ("Eligible stocks", "none", None),
        "Basket momentum + cash": (prefix + " full groups", "none", None),
        TOP_VOL_VERSION: (prefix + " top 10 positive-vol per group", "none", None),
        DOWNSIDE_VERSION: (prefix + " top 10 positive-vol per group", "loss", None),
        UPSIDE_VERSION: (prefix + " top 10 positive-vol per group", "upside", None),
        BASKET_DOWNSIDE_VERSION: (prefix + " full groups", "loss", None),
        BASKET_UPSIDE_VERSION: (prefix + " full groups", "upside", None),
    }
    result = metrics.loc[metrics.version.isin(versions)].copy()
    result["basket"] = result.version.map(lambda version: versions[version][0])
    result["target"] = result.version.map(lambda version: versions[version][1])
    result["allocation"] = [
        versions[v][2] or f"{int(m)}M positive momentum + cash"
        for v, m in zip(result.version, result.momentum_months)
    ]
    return result


def matching_cases(cases, **selection):
    """Strict tuple matching, leaving source/recipe/fit provenance columns intact."""
    mask = pd.Series(True, index=cases.index)
    for column, value in selection.items():
        mask &= cases[column].eq(value)
    return cases.loc[mask].copy()
