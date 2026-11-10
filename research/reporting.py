"""Read a pinned experiment and create publication figures; never fit a model."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .legacy import PROJECT_ROOT
from .store import ExperimentStore, digest_file
from .feature_studies import list_feature_studies, load_feature_study


def feature_study_reference(store, study_id):
    """Validate a completed study before deliberately pinning it in a release."""
    record = next((row for row in list_feature_studies(store) if row["study_id"] == study_id), None)
    if record is None:
        raise ValueError("Unknown feature-selection study.")
    load_feature_study(store, study_id)
    return {"study_id": study_id, "sha256": record["sha256"]}


def load_pinned_feature_study(store, reference):
    records = {row["study_id"]: row for row in list_feature_studies(store)}
    record = records.get(reference.get("study_id"))
    if record is None or reference.get("sha256") != record["sha256"]:
        raise ValueError("Pinned feature-study identity/checksum does not match the saved evidence.")
    return load_feature_study(store, record["study_id"])


def load_release(path=None):
    path = Path(path or PROJECT_ROOT / "submission/release.json").resolve()
    release = json.loads(path.read_text())
    root = (path.parent / release["artifact_root"]).resolve()
    store = ExperimentStore(root)
    summary = store.summary(release["run_id"])
    summary["release"] = release
    if release.get("feature_study"):
        summary["feature_study"] = load_pinned_feature_study(store, release["feature_study"])
    return summary


def readiness(summary):
    manifest = summary["manifest"]
    config = manifest["config"]
    cases = summary["cases"]
    originals = cases.loc[cases.variant.eq("baseline")]
    tuning = manifest.get("tuning", {})
    choice = summary["release"].get("final_choice_case_id")
    fits = summary.get("model_fits", pd.DataFrame())
    within_limit = (not fits.empty and fits.n_features.le(10).all()) if config["feature_spec"].get("selection", "none") != "none" else len(config["feature_spec"]["columns"]) <= 10
    return {
        "Declared input subset has at most 10 features": bool(within_limit),
        "Four model types recorded": len(manifest.get("models", [])) >= 4,
        "GridSearchCV evidence for at least two models": tuning.get("engine") == "GridSearchCV" and len(set(tuning.get("models", []))) >= 2,
        "Two non-ML controls including equal weight": len(originals) >= 2 and "equal_weight_universe" in set(originals.strategy),
        "Risk and reward metrics recorded": {"cagr", "sharpe", "sortino", "max_drawdown"}.issubset(summary["metrics"]),
        "Official test models frozen before evaluation": config["evaluation"] == "test" and config.get("fit_policy") == "frozen_at_origin",
        "Final strategy/model choice recorded": bool(choice) and choice in set(cases.case_id),
    }


def predictive_table(summary):
    frame = summary["model_metrics"][["model", "brier_score", "log_loss", "roc_auc"]].copy()
    return frame.rename(columns={"model": "Predictor", "brier_score": "Brier score", "log_loss": "Log loss", "roc_auc": "ROC AUC"})


def portfolio_table(summary):
    selected = summary["release"]["focus_strategies"]
    frame = summary["metrics"].loc[lambda data: data.strategy.isin(selected)]
    labels = {"equal_weight_universe": "Equal weight", "momentum_weighted": "Momentum",
              "training_high_vol_momentum_weighted": "Frozen high-vol + momentum"}
    frame = frame.assign(Strategy=frame.strategy.map(lambda key: labels.get(key, key)))
    order = frame.model_label.drop_duplicates()
    return frame.pivot(index="model_label", columns="Strategy", values="cagr").reindex(order).rename_axis("Version")


def portfolio_figure(summary):
    selected = summary["release"]["focus_strategies"]
    cases = summary["cases"].loc[lambda frame: frame.strategy.isin(selected)]
    monthly = summary["monthly"]
    colors = ["#273c38", "#287d70", "#bd8235", "#536da6", "#97619c"]
    figure, axes = plt.subplots(len(selected), 2, figsize=(11.5, 3.2 * len(selected)), squeeze=False)
    for row, strategy in enumerate(selected):
        subset = cases.loc[cases.strategy.eq(strategy)]
        for index, (_, case) in enumerate(subset.iterrows()):
            data = monthly.loc[monthly.case_id.eq(case.case_id)].sort_values("return_month")
            x = [data.snapshot_date.iloc[0], *data.return_month]
            width = 2.2 if case.variant == "baseline" else 1.4
            style = "-" if case.variant == "baseline" else "--"
            axes[row, 0].plot(x, [1, *data.wealth], label=case.model_label, color=colors[index % len(colors)], linewidth=width, linestyle=style)
            axes[row, 1].plot(x, [0, *data.drawdown], color=colors[index % len(colors)], linewidth=width, linestyle=style)
        axes[row, 0].set_title(subset.label.iloc[0], loc="left", fontsize=10, fontweight="bold")
        axes[row, 0].set_ylabel("Growth of $1")
        axes[row, 1].set_ylabel("Drawdown")
        axes[row, 1].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
        for axis in axes[row]:
            axis.grid(alpha=.18)
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="best")
    figure.tight_layout()
    return figure


def benchmark_table():
    """Independent full-window accounting reference, separate from model validation."""
    relative = "part0_data_and_benchmark/outputs/equal_weight_metrics.csv"
    release = json.loads((PROJECT_ROOT / "submission/release.json").read_text())
    expected = release.get("reference_files", {}).get(relative)
    if not expected or digest_file(PROJECT_ROOT / relative) != expected:
        raise ValueError("The independent benchmark reference must match the pinned release hash.")
    frame = pd.read_csv(PROJECT_ROOT / relative)
    return frame[["split", "months", "cagr", "annualized_volatility", "sharpe", "max_drawdown"]].rename(columns={
        "split": "Period", "months": "Months", "cagr": "CAGR", "annualized_volatility": "Volatility",
        "sharpe": "Sharpe", "max_drawdown": "Max drawdown"})
