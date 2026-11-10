"""Save matched basket overlays from explicitly named, already fitted runs."""
import argparse
from research.basket_adjustments import create_adjusted_basket_study
from research.basket_momentum import save_basket_study
from research.store import ExperimentStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basket-study", required=True)
    parser.add_argument("--loss-run", required=True)
    parser.add_argument("--upside-run", required=True)
    parser.add_argument("--strength", type=float, default=.5)
    parser.add_argument("--upside-threshold", type=float, default=.6)
    args = parser.parse_args()
    store = ExperimentStore()
    result = create_adjusted_basket_study(store, args.basket_study, args.loss_run,
        args.upside_run, args.strength, args.upside_threshold)
    directory = save_basket_study(result, store.root / "basket_studies")
    print(directory, flush=True)
    print(result["metrics"][["version", "model_key", "momentum_months", "cagr", "sharpe",
                             "max_drawdown", "avg_cash_weight"]].to_string(index=False))


if __name__ == "__main__":
    main()
