"""Run the same experiments from a terminal, without Streamlit."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from . import FeatureSpec, RunConfig, default_model_registry, default_registry
from .legacy import PROJECT_ROOT
from .store import ExperimentStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List registered strategies and exit")
    parser.add_argument("--strategies", nargs="+", default=["equal_weight_universe", "momentum_weighted"])
    parser.add_argument("--all", action="store_true", help="Run every registered strategy")
    parser.add_argument("--evaluation", choices=["training_validation", "test"], default="training_validation")
    parser.add_argument("--ml", action="store_true", help="Add the shared loss-probability adjustment")
    parser.add_argument("--models", nargs="+", default=["logistic"], help="Model IDs from --list-models")
    parser.add_argument("--all-models", action="store_true", help="Compare all four fixed models; enables ML")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--features", nargs="+", help="Explicit original-factor input columns; default all twenty")
    parser.add_argument("--feature-selection", choices=["none", "anova", "correlation_pruned", "random_forest"], default="none")
    parser.add_argument("--max-features", type=int, default=10)
    parser.add_argument("--correlation-threshold", type=float, default=.90)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--prediction-target", choices=["loss", "upside"], default="loss")
    parser.add_argument("--model-scope", choices=["pooled", "per_stock"], default="pooled")
    parser.add_argument("--upside-threshold", type=float, default=.6)
    parser.add_argument("--min-stock-history", type=int, default=36)
    parser.add_argument("--min-stock-class", type=int, default=6)
    parser.add_argument("--cash-momentum-months", type=int, default=12,
                        help="Momentum/history window for annual-consistency cash strategy only (2–12)")
    parser.add_argument("--output", type=Path, help="New output directory (must not already exist)")
    parser.add_argument("--include-weights", action="store_true")
    parser.add_argument("--fit-policy", choices=["expanding", "frozen_at_origin"], default="expanding")
    parser.add_argument("--force", action="store_true", help="Explicitly recompute instead of reusing an exact saved run")
    parser.add_argument("--saved", action="store_true", help="List persistent saved experiments")
    parser.add_argument("--load", metavar="RUN_ID", help="Load a saved experiment without model fitting")
    parser.add_argument("--import-report", type=Path, help="Import a compatible schema-2 CSV report without fitting")
    parser.add_argument("--artifact-root", type=Path, help="Override the local artifact library")
    args = parser.parse_args()
    registry = default_registry()
    store = ExperimentStore(args.artifact_root)
    if args.saved:
        for row in store.list_runs():
            print(f"{row['run_id']} | {row['label']} | {row['case_count']} cases | {row['origin']}")
        return
    if args.import_report:
        from .imports import import_csv_report
        print(f"Imported saved report: {import_csv_report(store, args.import_report)}")
        return
    if args.list_models:
        for model in default_model_registry().values():
            print(f"{model.key}: {model.label}")
        return
    if args.list:
        for spec in registry.values():
            print(f"{spec.key}: {spec.label}")
        return
    config = RunConfig(
        strategies=tuple(registry if args.all else args.strategies), evaluation=args.evaluation,
        include_ml=args.ml or args.all_models, adjustment_strength=args.strength,
        models=tuple(default_model_registry() if args.all_models else args.models),
        feature_spec=FeatureSpec(tuple(args.features) if args.features is not None else FeatureSpec().columns,
                                 args.feature_selection, args.max_features, args.correlation_threshold),
        fit_policy=args.fit_policy,
        cash_momentum_months=args.cash_momentum_months,
        prediction_target=args.prediction_target, model_scope=args.model_scope,
        upside_threshold=args.upside_threshold, min_stock_history=args.min_stock_history,
        min_stock_class=args.min_stock_class,
    )
    result = store.load(args.load) if args.load else store.get_or_run(config, force=args.force, progress=print)
    print(f"Result source: {result.cache_info['source']} | run {result.cache_info['run_id']}")
    print(result.metrics[["label", "model_label", "cagr", "annualized_volatility", "sharpe", "sortino", "max_drawdown"]].to_string(index=False))
    destination = args.output or PROJECT_ROOT / "0_2_Strategy_Runner/outputs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result.save(destination, include_weights=args.include_weights)
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
