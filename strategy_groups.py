"""Presentation-only strategy families and lossless model comparison tables."""
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategyFamily:
    key: str
    title: str
    description: str
    strategies: tuple[str, ...]


FAMILIES = (
    StrategyFamily("benchmarks", "Benchmarks", "Equal-weight reference portfolios, with and without a history requirement.",
                   ("equal_weight_universe", "equal_weight_eligible")),
    StrategyFamily("volatility", "Volatility baskets", "Select stocks by how much their returns fluctuate; equal-weight the basket.",
                   ("high_volatility", "low_volatility")),
    StrategyFamily("momentum", "Momentum baskets", "Select recent positive-momentum stocks; compare equal and momentum-based weights.",
                   ("momentum_equal_weight", "momentum_weighted")),
    StrategyFamily("high_volatility_hybrids", "High-volatility hybrids", "Freeze a high-volatility pool, then vary momentum selection and risk sizing.",
                   ("training_high_vol_equal_weight", "training_high_vol_momentum_equal", "training_high_vol_momentum_weighted",
                    "hybrid_volatility_3m_sizing", "hybrid_downside_3m_sizing")),
    StrategyFamily("ratio_pools", "Sharpe / Sortino pools", "Freeze a pool by historical risk-adjusted performance, then weight positive momentum.",
                   ("training_sharpe_momentum_weighted", "training_sortino_momentum_weighted")),
    StrategyFamily("annual_winners_cash", "Annual winners + cash", "Require mostly positive historical years; buy positive momentum and retain inactive capital in cash.",
                   ("annual_consistency_momentum_cash",)),
)
OTHER = StrategyFamily("other", "Other strategies", "Additional registered portfolio rules.", ())


def family_for(strategy):
    return next((family for family in FAMILIES if strategy in family.strategies), OTHER)


def grouped_strategies(strategies):
    """Preserve membership and within-family input order; never drop unknown rules."""
    strategies = list(dict.fromkeys(strategies))
    return [(family, [key for key in strategies if family_for(key).key == family.key])
            for family in (*FAMILIES, OTHER)
            if any(family_for(key).key == family.key for key in strategies)]


def change_matrix(paired, metric):
    """One strategy per row, one model per column; no averaging or ranking."""
    if paired.duplicated(["strategy", "model_id"]).any():
        raise ValueError("Expected one recorded change per strategy/model pair.")
    strategies = paired.strategy.drop_duplicates().tolist()
    models = paired.model_id.drop_duplicates().tolist()
    labels = paired.drop_duplicates("model_id").set_index("model_id").model_label.to_dict()
    strategy_labels = paired.drop_duplicates("strategy").set_index("strategy").label.to_dict()
    return (paired.pivot(index="strategy", columns="model_id", values=metric)
            .reindex(index=strategies, columns=models)
            .rename(index=strategy_labels, columns=labels).rename_axis(index="Strategy", columns=None))


DELTA_METRICS = {
    "Annualized return (CAGR)": ("change_cagr", True, True),
    "Sharpe ratio": ("change_sharpe", False, True),
    "Sortino ratio": ("change_sortino", False, True),
    "Maximum drawdown": ("change_max_drawdown", True, True),
    "Monthly turnover": ("change_avg_one_way_turnover", True, False),
}


def change_format(value, percentage_points):
    if pd.isna(value):
        return "—"
    return f"{value * 100:+.2f} pp" if percentage_points else f"{value:+.3f}"
