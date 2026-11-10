"""Build the training-holdout tuning comparison from a saved feature study.

Does not open the official test CSV and does not register an experiment-catalog run.
"""
import argparse
import json

from research.feature_studies import list_feature_studies, load_feature_study
from research.store import ExperimentStore
from research.models import default_model_registry
from research.tuning_holdout import INNER_TEST_ORIGIN, create_tuning_holdout, frozen_feature_list, save_tuning_holdout


def main():
    parser = argparse.ArgumentParser(description="Tune on the early training years and score the later training years.")
    parser.add_argument("--recipe", default="anova_10", choices=["anova_10", "pruned_10"])
    parser.add_argument("--models", nargs="+", default=["logistic", "hist_gradient_boosting"],
                        choices=list(default_model_registry()))
    parser.add_argument("--target", default="loss", choices=["loss", "upside"])
    parser.add_argument("--allow-slow", action="store_true", help="Keep 100 trees for the random-forest search.")
    parser.add_argument("--feature-model", default="logistic", choices=list(default_model_registry()),
                        help="Saved selector fit supplying the fixed columns, independent of the prediction model.")
    parser.add_argument("--study", default=None, help="Feature-study id. Defaults to the latest saved study.")
    args = parser.parse_args()
    store = ExperimentStore()
    studies = list_feature_studies(store)
    if not studies:
        raise SystemExit("No feature-selection study is saved.")
    study_id = args.study or studies[0]["study_id"]
    study = load_feature_study(store, study_id)
    columns, source = frozen_feature_list(study, args.recipe, INNER_TEST_ORIGIN, model_id=args.feature_model)
    source["feature_study_id"] = study_id
    print("Frozen features at", source["fit_snapshot"], ":", ", ".join(columns))
    print("Models:", ", ".join(args.models))
    print("Prediction target:", args.target)
    result = create_tuning_holdout(store.project_root, columns, model_keys=tuple(args.models),
                                   feature_source=source, progress=print,
                                   prediction_target=args.target, allow_slow=args.allow_slow)
    directory = save_tuning_holdout(result, store.root / "tuning_studies")
    print(json.dumps({"directory": str(directory), **{key: result["metadata"][key] for key in
        ["origin_snapshot", "trained_through_return_month", "holdout_first_return_month", "holdout_last_return_month"]}},
        indent=2))


if __name__ == "__main__":
    main()
