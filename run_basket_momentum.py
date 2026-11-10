"""Fit monthly correlation baskets on the training window and save the comparison.

Does not open the official test CSV and does not register an experiment-catalog run.
"""
import argparse
import json

from research.basket_momentum import create_basket_study, save_basket_study
from research.store import ExperimentStore


def main():
    parser = argparse.ArgumentParser(description="Score correlation-basket momentum against equal weight and ungated momentum.")
    parser.add_argument("--grouping", choices=["kmeans", "ward"], default="kmeans", help="Unsupervised grouping method; trading rules stay fixed.")
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--lookback", type=int, default=12, help="Months of return history used only to form the groups.")
    parser.add_argument("--momentum", nargs="+", type=int, default=[1, 2, 3], help="Holding windows, in months.")
    args = parser.parse_args()
    store = ExperimentStore()
    print(f"Clustering into {args.clusters} baskets on a {args.lookback}-month return path.")
    print("Holding windows:", ", ".join(f"{months} months" for months in args.momentum))
    print("A second book keeps the 10 highest positive-vol names in each basket and trades 1- and 2-month momentum.")
    result = create_basket_study(store.project_root, n_clusters=args.clusters, lookback=args.lookback,
                                 momentum_months=tuple(args.momentum), grouping_method=args.grouping)
    directory = save_basket_study(result, store.root / "basket_studies")
    print(json.dumps({"directory": str(directory), **{key: result["metadata"][key] for key in
        ["first_snapshot", "last_snapshot", "first_return_month", "last_return_month", "months",
         "momentum_months", "test_csv_loaded"]}}, indent=2))
    table = result["metrics"][["version", "momentum_months", "cagr", "annualized_volatility", "sharpe",
                               "max_drawdown", "avg_one_way_turnover", "avg_cash_weight", "avg_names"]]
    print(table.to_string(index=False))
    held = result["assignments"]
    if "version" in held.columns:
        top = held.loc[held.version.eq("Top 10 positive-vol baskets") & held.weight.gt(0)]
        if not top.empty:
            print("Top-10 max weight by horizon:")
            print(top.groupby("momentum_months").weight.max().to_string())


if __name__ == "__main__":
    main()
