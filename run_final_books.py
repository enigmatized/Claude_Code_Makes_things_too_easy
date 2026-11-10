"""Score the four books once on the official 2010–2018 test.

The recipe is the one written in context/context.md. A saved study blocks a
second pass. This does not pin the submission release.
"""
from research.final_books import list_final_books, save_final_books, score_final_books
from research.legacy import PROJECT_ROOT


def main():
    if list_final_books(PROJECT_ROOT / "artifacts"):
        raise SystemExit("A final-book study is already saved. This recipe is not run again.")
    result = score_final_books(PROJECT_ROOT, progress=print)
    directory = save_final_books(result, PROJECT_ROOT / "artifacts")
    print(directory, flush=True)
    view = result["metrics"]
    view = view.loc[view.model_key.isin(["none", "logistic"]) & view.momentum_months.isin([0, 1])]
    columns = ["version", "model_key", "momentum_months", "cagr", "annualized_volatility",
               "sharpe", "max_drawdown", "avg_cash_weight", "avg_names"]
    print(view[columns].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
