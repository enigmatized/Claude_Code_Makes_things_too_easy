"""Declared training-only momentum-window studies over immutable saved runs."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
import sqlite3
import uuid

import numpy as np
import pandas as pd

from .cash_strategy import STRATEGY_KEY
from .store import CALCULATION_FILES, digest_file

LOOKBACKS = (2, 3, 6, 12)


def list_momentum_studies(store):
    if not store.catalog.exists():
        return []
    with sqlite3.connect(store.catalog) as connection:
        connection.row_factory = sqlite3.Row
        exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='momentum_studies'").fetchone()
        return [dict(row) for row in connection.execute("SELECT * FROM momentum_studies ORDER BY created_utc DESC")] if exists else []


def compare_momentum_runs(store, runs):
    if set(runs) != {str(months) for months in LOOKBACKS}:
        raise ValueError("A momentum study needs 2, 3, 6 and 12-month runs.")
    summaries = {str(months): store.summary(runs[str(months)]) for months in LOOKBACKS}
    reference = summaries["12"]
    expected = dict(reference["manifest"]["config"])
    expected.pop("cash_momentum_months")
    rows = []
    for months, summary in summaries.items():
        config = dict(summary["manifest"]["config"])
        actual_months = config.pop("cash_momentum_months")
        if config != expected or actual_months != int(months) or config["evaluation"] != "training_validation":
            raise ValueError("Only the cash momentum window may differ; use matched training-validation runs.")
        manifest, baseline = summary["manifest"], reference["manifest"]
        if manifest["source_sha256"] != baseline["source_sha256"] or manifest["versions"] != baseline["versions"]:
            raise ValueError("Momentum study sources and runtime must match.")
        if any(manifest["code_sha256"].get(name) != baseline["code_sha256"].get(name) for name in CALCULATION_FILES):
            raise ValueError("Momentum study calculation code must match.")
        if set(manifest["source_sha256"]) != {"mlfactor_long_only_train.csv"}:
            raise ValueError("Momentum studies must not include test data.")
        current = summary["monthly"].sort_values(["case_id", "return_month"]).reset_index(drop=True)
        original = reference["monthly"].sort_values(["case_id", "return_month"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(current[["case_id", "return_month", "snapshot_date"]],
                                      original[["case_id", "return_month", "snapshot_date"]])
        # The universe and all other trading rules must be numerically unchanged.
        control = current.strategy.ne(STRATEGY_KEY)
        np.testing.assert_array_equal(current.loc[control, "simple_return"], original.loc[control, "simple_return"])
        for _, row in summary["metrics"].loc[lambda f: f.strategy.eq(STRATEGY_KEY)].iterrows():
            parent = reference["metrics"].set_index("case_id").loc[row.case_id]
            record = {**row.to_dict(), "lookback_months": int(months), "run_id": runs[months]}
            for metric in ["cagr", "sharpe", "max_drawdown", "avg_one_way_turnover", "avg_cash_weight"]:
                record[f"change_{metric}_vs_12m"] = row[metric] - parent[metric]
            rows.append(record)
    return {"summaries": summaries, "portfolio_comparison": pd.DataFrame(rows)}


def create_momentum_study(store, config, progress=None):
    if os.environ.get("MLFACTOR_READ_ONLY") == "1":
        raise RuntimeError("Study creation is disabled during saved-results-only rendering.")
    if config.evaluation != "training_validation" or not {STRATEGY_KEY, "equal_weight_universe"}.issubset(config.strategies):
        raise ValueError("Use training validation with the cash strategy and equal-weight control.")
    store._initialize()
    study_id = "momentum_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    folder = store.root / "studies" / study_id
    folder.mkdir(parents=True)
    plan = {"study_id": study_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "label": "Cash momentum: 2 vs 3 vs 6 vs 12 months", "config": asdict(config),
        "lookbacks": list(LOOKBACKS),
        "protocol": "Training-only sensitivity comparison; annual pool, dates, classifiers, factors and cash policy fixed. No automatic winner or GridSearchCV."}
    (folder / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    runs = {}
    reference_screen, reference_predictions = None, None
    for months in LOOKBACKS:
        if progress:
            progress(f"Momentum lookback: {months} months")
        result = store.get_or_run(replace(config, cash_momentum_months=months), progress=progress)
        screen = result.screens["annual_consistency"]
        predictions = {key: batch.predictions for key, batch in result.prediction_batches.items()}
        if reference_screen is None:
            reference_screen, reference_predictions = screen.copy(), {k: v.copy() for k, v in predictions.items()}
        else:
            pd.testing.assert_frame_equal(screen, reference_screen)
            if predictions.keys() != reference_predictions.keys():
                raise AssertionError("Prediction model identities differ.")
            for key, frame in predictions.items():
                pd.testing.assert_frame_equal(frame, reference_predictions[key])
        runs[str(months)] = result.cache_info["run_id"]
        del result
    comparison = compare_momentum_runs(store, runs)
    table_path = folder / "portfolio_comparison.csv"
    comparison["portfolio_comparison"].to_csv(table_path, index=False)
    metadata = {**plan, "runs": runs, "checks": {
        "identical_annual_pool": True, "identical_predictions": True,
        "identical_other_strategy_returns": True, "same_evaluation_calendar": True,
        "test_csv_loaded": False, "hyperparameter_search": False},
        "file_sha256": {table_path.name: digest_file(table_path)}}
    path = folder / "study.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    with sqlite3.connect(store.catalog) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS momentum_studies (study_id TEXT PRIMARY KEY, label TEXT NOT NULL, created_utc TEXT NOT NULL, relative_path TEXT NOT NULL, sha256 TEXT NOT NULL)")
        connection.execute("INSERT INTO momentum_studies VALUES (?, ?, ?, ?, ?)",
            (study_id, metadata["label"], metadata["created_utc"], str(path.relative_to(store.root)), digest_file(path)))
    return metadata


def load_momentum_study(store, study_id):
    record = next((row for row in list_momentum_studies(store) if row["study_id"] == study_id), None)
    if record is None:
        raise ValueError("Unknown momentum study.")
    path = store.root / record["relative_path"]
    if digest_file(path) != record["sha256"]:
        raise ValueError("Momentum-study metadata failed its integrity check.")
    metadata = json.loads(path.read_text())
    for relative, digest in metadata["file_sha256"].items():
        if digest_file(path.parent / relative) != digest:
            raise ValueError("Momentum-study export failed its integrity check.")
    return {"metadata": metadata, **compare_momentum_runs(store, metadata["runs"])}
