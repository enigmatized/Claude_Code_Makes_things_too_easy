"""Fit any missing upside forecasts, then score the four books on each feature cut.

Uses the training file only. The official test CSV is not opened. Basket
allocations are not written again; this saves the portfolio metrics.
"""
from research.basket_features import save_feature_book_scores, score_feature_books
from research.feature_studies import FOREST_RECIPES
from research.runner import RunConfig
from research.store import ExperimentStore


LOSS_RUNS = {
    "all_20": "20261109T202421_184e95372ab2",
    "anova_10": "20261109T202516_039a5a86258b",
    "pruned_10": "20261109T202612_7da383c77e5b",
    "forest_10": "20261109T202931_597510736fb4",
}
BASKET_STUDY = "artifacts/basket_studies/baskets_20261109T223026_c7be47ff"
MODELS = ("logistic", "naive_bayes", "random_forest", "hist_gradient_boosting")


def main():
    store = ExperimentStore()
    upside = {}
    for key, recipe in FOREST_RECIPES.items():
        config = RunConfig(
            strategies=("equal_weight_universe",), include_ml=True, models=MODELS,
            validation_history_months=60, prediction_target="upside", feature_spec=recipe,
            refit_months=12, fit_policy="expanding")
        print(f"upside {key}: {recipe.label}", flush=True)
        result = store.get_or_run(config, progress=print)
        upside[key] = result.cache_info["run_id"]
        print(f"upside {key}: {upside[key]}", flush=True)
        del result
    recipes = [
        {"key": key, "label": recipe.label, "loss_run_id": LOSS_RUNS[key], "upside_run_id": upside[key]}
        for key, recipe in FOREST_RECIPES.items()
    ]
    print("scoring four books", flush=True)
    scored = score_feature_books(store, BASKET_STUDY, recipes)
    directory = save_feature_book_scores(scored, store.root)
    print(directory, flush=True)
    view = scored["metrics"]
    view = view.loc[view.model_key.isin(["none", "logistic"]) & view.momentum_months.isin([0, 1])]
    print(view[["recipe", "version", "model_key", "momentum_months", "cagr", "sharpe",
                "max_drawdown", "avg_cash_weight"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
