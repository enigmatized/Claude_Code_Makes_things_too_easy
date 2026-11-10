"""Run the fixed, training-only four-recipe feature-selection study."""
from research import RunConfig, default_model_registry, default_registry
from research.feature_studies import create_feature_study, FOREST_RECIPES
from research.store import ExperimentStore


if __name__ == "__main__":
    study = create_feature_study(ExperimentStore(), RunConfig(
        strategies=tuple(default_registry()), models=tuple(default_model_registry()),
        include_ml=True, evaluation="training_validation", fit_policy="expanding",
        cash_momentum_months=3, adjustment_strength=.5), progress=print, recipes=FOREST_RECIPES)
    print(f"Saved study: {study['study_id']}")
    for recipe, run_id in study["runs"].items():
        print(f"{recipe}: {run_id}")
