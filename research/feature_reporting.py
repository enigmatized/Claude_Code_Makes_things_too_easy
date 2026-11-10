"""Shared, read-only feature-study tables and publication figures.

These functions consume saved evidence. They neither fit estimators nor replay
portfolios, and can be imported without Streamlit by notebooks and Quarto.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_studies import feature_audit


RECIPE_COLORS = {"all_20": "#536da6", "anova_10": "#277d68",
                 "pruned_10": "#bd8235", "forest_10": "#99599b"}


def method_notes():
    return (Path(__file__).resolve().parents[1] /
            "0_3_Feature_Selection/method_notes.md").read_text()


def importance_table(study, model_id="logistic", recipe="forest_10", snapshot=None):
    fits = study["fits"].loc[lambda rows: rows.model_id.eq(model_id) & rows.recipe.eq(recipe)]
    if fits.empty:
        raise ValueError("No saved feature fit for the requested model and recipe.")
    snapshot = fits.fit_snapshot.min() if snapshot is None else pd.Timestamp(snapshot)
    rows = feature_audit(fits.loc[fits.fit_snapshot.eq(snapshot)]).sort_values("rank").reset_index(drop=True)
    if rows.empty:
        raise ValueError("No saved feature audit at that origin.")
    # Older studies remain readable without a tree-variation field.
    if "importance_std" not in rows:
        rows["importance_std"] = np.nan
    from pipeline import FEATURE_INFO  # The shared legacy bridge registers this module.
    groups = {name: group for name, group, meaning in FEATURE_INFO}
    meanings = {name: meaning for name, group, meaning in FEATURE_INFO}
    return rows.assign(group=rows.feature.map(groups), meaning=rows.feature.map(meanings))


def selection_overlap(study, model_id="logistic", snapshot=None):
    """Number of selected factors shared by each method at one historical fit."""
    fits = study["fits"].loc[lambda rows: rows.model_id.eq(model_id) & rows.recipe.ne("all_20")]
    snapshot = fits.fit_snapshot.min() if snapshot is None else pd.Timestamp(snapshot)
    audit = feature_audit(fits.loc[fits.fit_snapshot.eq(snapshot)])
    sets = {study["metadata"]["recipes"][key]["label"]:
            set(audit.loc[audit.recipe.eq(key) & audit.selected, "feature"])
            for key in study["metadata"]["runs"] if key != "all_20"}
    return pd.DataFrame({b: {a: len(sa & sb) for a, sa in sets.items()} for b, sb in sets.items()})


def prediction_table(study, model_id):
    rows = study["predictive_comparison"].loc[lambda frame: frame.model_id.eq(model_id)].copy()
    base = rows.loc[rows.recipe.eq("all_20")].iloc[0]
    rows["Brier change vs all 20"] = rows.brier_score - base.brier_score
    rows["Log-loss change vs all 20"] = rows.log_loss - base.log_loss
    return rows


def prediction_summary(study):
    """All fitted predictors with within-model deltas against all twenty factors."""
    models = study["metadata"]["config"]["models"]
    return pd.concat([prediction_table(study, key) for key in models], ignore_index=True)


def forest_findings(study, brief=False):
    """A reproducible short reading of saved scores, without selecting a winner."""
    rows = prediction_summary(study)
    forest = rows.loc[rows.recipe.eq("forest_10")]
    if forest.empty:
        return "This saved study has no random-forest selection recipe."
    target = study["metadata"]["config"].get("prediction_target", "loss")
    control_id = "historical_loss_rate" if target == "loss" else "historical_up_rate"
    control = study["predictive_comparison"].loc[lambda data: data.model_id.eq(control_id)].iloc[0]
    improved = int(forest["Brier change vs all 20"].lt(0).sum())
    beat_control = int(forest.brier_score.lt(control.brier_score).sum())
    text = (f"Forest selection lowers Brier score relative to all twenty factors for {improved} of "
            f"{len(forest)} predictors. {beat_control} of the forest-selected predictors beat the "
            f"historical event-frequency control on Brier score (control: {control.brier_score:.6f}).")
    return text if brief else text + (
        " These are training-validation comparisons; portfolio effects must be assessed separately. "
        "They do not declare a final feature/model choice or provide a tuned frozen-test result.")


def portfolio_table(study, model_id, strategies=None):
    rows = study["portfolio_comparison"].loc[lambda frame: frame.model_id.eq(model_id)]
    return rows.copy() if strategies is None else rows.loc[rows.strategy.isin(strategies)].copy()


def calibration_table(study, model_id):
    return study["calibration"].loc[lambda rows: rows.model_id.eq(model_id)].copy()


def feature_curves(study, strategy, model_id):
    """Saved portfolio paths in declared recipe order, original reference once."""
    histories = []
    for key, summary in study["summaries"].items():
        if key == "all_20":
            case = summary["cases"].loc[lambda rows: rows.strategy.eq(strategy) & rows.variant.eq("baseline")].iloc[0]
            monthly = summary["monthly"].loc[lambda rows: rows.case_id.eq(case.case_id)].sort_values("return_month")
            histories.append(("Non-ML reference (unchanged)", monthly, "#68716e"))
        case = summary["cases"].loc[lambda rows: rows.strategy.eq(strategy) & rows.model_id.eq(model_id)].iloc[0]
        monthly = summary["monthly"].loc[lambda rows: rows.case_id.eq(case.case_id)].sort_values("return_month")
        histories.append((study["metadata"]["recipes"][key]["label"], monthly,
                          RECIPE_COLORS.get(key, "#556b73")))
    return histories


def importance_figure(study, model_id="logistic", recipe="forest_10", snapshot=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    rows = importance_table(study, model_id, recipe, snapshot)
    figure, axis = plt.subplots(figsize=(10, 7))
    errors = rows.importance_std.fillna(0).to_numpy()
    axis.barh(rows.feature, rows.score, xerr=errors, capsize=2,
              color=np.where(rows.selected, "#277d68", "#c8cfcc"), error_kw={"elinewidth": .8})
    axis.invert_yaxis()
    axis.set_xlabel("Mean decrease in impurity · error bars: SD across trees")
    origin = pd.Timestamp(rows.fit_snapshot.iloc[0])
    axis.set_title(f"Random forest feature selection · fit {origin:%b %Y}", loc="left")
    axis.legend(handles=[Patch(color="#277d68", label="Selected"),
                         Patch(color="#c8cfcc", label="Not selected")], frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def portfolio_figure(study, model_id="logistic", strategy="annual_consistency_momentum_cash",
                     cumulative_log=False):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for label, monthly, color in feature_curves(study, strategy, model_id):
        dates = [monthly.snapshot_date.iloc[0], *monthly.return_month]
        values = [0, *monthly.log_return.cumsum()] if cumulative_log else [1, *monthly.wealth]
        axes[0].plot(dates, values, label=label, color=color)
        axes[1].plot(dates, [0, *monthly.drawdown], color=color)
    axes[0].set_ylabel("Cumulative log return" if cumulative_log else "Growth of $1")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title(f"Training validation · {model_id} · {strategy}", fontsize=10, loc="left")
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    for axis in axes:
        axis.grid(alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure
