"""Rubric checks against the saved evidence the graded notebook actually reads.

`research.reporting.readiness` scores the catalog run pinned in submission/release.json.
The notebook's official-test results come from the frozen final-book study, its tuning
evidence from the GridSearchCV holdout, the grouping search and the pinned engineering
study, and its declared choice from `notebook_views.CHOICE`. These checks read those
saves. Nothing is fitted and no course CSV is opened.
"""
from __future__ import annotations

import pandas as pd


def submission_checks() -> dict[str, bool]:
    from research.engineering_study import load_engineering_study
    from research.grouping import load_latest_grouping
    from research.notebook_views import CHOICE, artifact_root, latest_holdout, load_final_windows, pinned_engineering

    study, _training_meta, windows = load_final_windows()
    meta = study["metadata"]
    test_metrics = windows["test"][0]
    no_ml = set(test_metrics.loc[test_metrics.model_key.eq("none"), "version"])
    fitted = [names for by_target in meta.get("fitted_features", {}).values() for names in by_target.values()]

    holdout = latest_holdout()["metadata"]
    grouping = load_latest_grouping(artifact_root())["metadata"]
    engineering = pinned_engineering()
    eng_test_ok = (engineering / "test" / "complete.json").is_file()
    eng_trials = load_engineering_study(engineering, "training")["trials"] if eng_test_ok else pd.DataFrame()

    searched_models = set(holdout.get("chosen", {}))
    return {
        "Frozen official-test fit uses at most 10 features":
            bool(fitted) and meta.get("n_features", 99) <= 10 and all(len(names) <= 10 for names in fitted),
        "Four model types in the frozen official-test fit": len(set(meta.get("models", []))) >= 4,
        "GridSearchCV evidence for at least two models (training only)":
            len(searched_models) >= 2 and holdout.get("test_csv_loaded") is False
            and holdout.get("scoring") in {"neg_log_loss", "neg_brier_score"},
        "Extra GridSearchCV model (K nearest neighbors) on training only":
            grouping.get("scoring") is not None and grouping.get("test_csv_loaded") is not True
            and pd.Timestamp(grouping.get("holdout_last_return_month", "2100-01-01")) <= pd.Timestamp("2009-12-31"),
        "Tuned settings re-scored on the chosen book (2010–2018)":
            eng_test_ok and not eng_trials.empty and eng_trials.model_key.nunique() >= 2,
        "Equal weight plus a second non-ML strategy":
            "Equal weight" in no_ml and len(no_ml - {"Equal weight"}) >= 1,
        "Risk and reward metrics recorded":
            {"cagr", "annualized_log_growth", "annualized_volatility", "sharpe", "max_drawdown"}.issubset(test_metrics.columns),
        "Official-test models frozen before evaluation":
            meta.get("fit_policy") == "frozen_at_origin"
            and pd.Timestamp(meta["trained_through_return_month"]) < pd.Timestamp(meta["first_return_month"]),
        "Official equal-weight check passed": bool(meta.get("official_equal_weight_check")),
        "Final choice declared and present in the frozen run": CHOICE["version"] in no_ml,
    }
