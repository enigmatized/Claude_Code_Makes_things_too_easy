"""Compare fixed 2/3/6/12-month cash momentum, with four existing ML overlays."""
import json
from pathlib import Path

from research import RunConfig, default_model_registry
from research.cash_strategy import STRATEGY_KEY
from research.momentum_studies import create_momentum_study, load_momentum_study
from research.store import ExperimentStore


if __name__ == "__main__":
    store = ExperimentStore()
    study = create_momentum_study(store, RunConfig(
        ("equal_weight_universe", STRATEGY_KEY), include_ml=True,
        models=tuple(default_model_registry())), progress=print)
    folder = Path(__file__).resolve().parent / "0_5_Momentum_Lookbacks/outputs" / study["study_id"]
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "study_reference.json").write_text(json.dumps(study, indent=2) + "\n")
    loaded = load_momentum_study(store, study["study_id"])
    loaded["portfolio_comparison"].to_csv(folder / "portfolio_comparison.csv", index=False)
    print(f"Saved momentum study: {study['study_id']}")
    print(f"Review exports: {folder}")
    print(loaded["portfolio_comparison"].loc[lambda f: f.variant.eq("baseline"),
        ["lookback_months", "cagr", "sharpe", "max_drawdown", "avg_one_way_turnover", "avg_cash_weight"]].to_string(index=False))
