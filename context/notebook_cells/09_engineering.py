"""Feature-engineering lessons from saved studies. No fitting and no test CSV.

The training study is the completed chronological holdout. The comparison
file in 0_7_Feature_Engineering is a previously viewed 2010–2018 notebook
result. Annual log growth for that file is log(1 + CAGR), the same identity
performance() uses.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.engineering_study import latest_engineering, load_engineering_study
from research.legacy import PROJECT_ROOT

COMPARISON_FILE = "0_7_Feature_Engineering/part7_model_comparison.csv"
HOLD_BOOK = "K-means top 10"
HOLD_MODEL = "logistic"
HOLD_TARGET = "loss"
_GROWTH = (
    ("annualized_log_growth", "Annual log growth"),
    ("cagr", "CAGR"),
    ("annualized_volatility", "Volatility"),
    ("sharpe", "Sharpe"),
    ("max_drawdown", "Max drawdown"),
)


def engineering_holdout_table() -> pd.DataFrame:
    """Unadjusted top-10 book against each recipe's logistic next-down overlay.

    Fixed and tuned rows are both shown. The window is the saved training holdout.
    """
    loaded = _training()
    metrics = loaded["metrics"]
    original = metrics.loc[metrics.book.eq(HOLD_BOOK) & metrics.status.eq("Original")]
    overlays = metrics.loc[
        metrics.book.eq(HOLD_BOOK)
        & metrics.model_key.eq(HOLD_MODEL)
        & metrics.target.eq(HOLD_TARGET)
        & metrics.status.isin(["Fixed", "Tuned"])
    ]
    frame = pd.concat([original, overlays], ignore_index=True)
    if frame.empty:
        raise ValueError(f"{HOLD_BOOK} is not in the engineering training metrics.")
    frame = frame.sort_values(["recipe", "status"]).reset_index(drop=True)
    rows = []
    for record in frame.itertuples(index=False):
        row = {"Recipe": record.recipe, "Status": record.status}
        for source, header in _GROWTH:
            row[header] = getattr(record, source)
        rows.append(row)
    show = pd.DataFrame(rows)
    show.attrs["directory"] = str(loaded["directory"])
    show.attrs["scope"] = loaded["scope"]
    return show


def engineering_comparison_table() -> pd.DataFrame:
    """Tuned top-20% books from the feature-engineering notebook comparison.

    That notebook already used 2010–2018. This function only reads its saved CSV.
    """
    frame = pd.read_csv(PROJECT_ROOT / COMPARISON_FILE)
    tuned = frame.loc[frame.tuning.eq("tuned")].copy()
    if tuned.empty:
        raise ValueError(f"{COMPARISON_FILE} has no tuned rows.")
    tuned["Annual log growth"] = np.log1p(tuned["cagr"].astype(float))
    show = pd.DataFrame({
        "Set": tuned["set"],
        "Model": tuned["model"].map(_model_name),
        "Annual log growth": tuned["Annual log growth"],
        "CAGR": tuned["cagr"],
        "Volatility": tuned["vol"],
        "Sharpe": tuned["sharpe"],
        "Max drawdown": tuned["max_dd"],
        "Test AUC": tuned["test_auc"],
    })
    return show.sort_values(["Set", "Model"]).reset_index(drop=True)


def engineering_note() -> str:
    """Three lessons: the holdout overlay, the viewed test comparison, and the limit."""
    loaded = _training()
    scope = loaded["scope"]
    table = engineering_holdout_table()
    base = table.loc[table["Status"].eq("Original")].iloc[0]
    tuned = table.loc[table["Status"].eq("Tuned")]
    fixed = table.loc[table["Status"].eq("Fixed")].set_index("Recipe")
    gaps = (tuned.set_index("Recipe")["CAGR"] - fixed["CAGR"]).abs()
    comparison = engineering_comparison_table()
    start = pd.Timestamp(scope["first_return_month"]).strftime("%B %Y")
    end = pd.Timestamp(scope["last_return_month"]).strftime("%B %Y")
    return (
        f"The completed engineering study scores {start} through {end} after a fit through "
        f"{scope['fit_origin']}. Its scope records that the test file was not loaded. "
        f"On {HOLD_BOOK}, logistic next-down overlays have annual log growth from "
        f"{tuned['Annual log growth'].min():.2%} to {tuned['Annual log growth'].max():.2%}, "
        f"against the unadjusted book at {base['Annual log growth']:.2%}. "
        f"Sharpe stays near {base['Sharpe']:.3f} and the largest tuned-minus-fixed CAGR is "
        f"{gaps.max():.3%}. "
        "The notebook comparison in 0_7_Feature_Engineering already scored a top-20% long-only rule "
        f"on 2010–2018. Tuned test AUC runs from {comparison['Test AUC'].min():.3f} to "
        f"{comparison['Test AUC'].max():.3f}. "
        "engineered10 has the highest annual log growth in that file, and the notebook traces the "
        "holdings to high 6-month volatility. A model-free top quintile of that volatility rank "
        "is recorded there at CAGR 18.4% and Sharpe 0.84, against equal weight at 13.1% and 0.87. "
        "Ten constructed inputs can still combine all twenty source factors. ANOVA-10 is the recipe "
        "limited to ten original factors. "
        + _final_clause(loaded["directory"])
    )


def _final_clause(directory):
    if (Path(directory) / "test" / "complete.json").is_file():
        return (
            "A completed 2010–2018 engineering stage is saved. This write-up does not use it to pick a recipe; "
            "Section 5 reads it only to ask whether tuning changed the chosen book."
        )
    return (
        "The engineering runner has no completed 2010–2018 file, so this submission does not add "
        "an engineered official-test portfolio from that runner."
    )


def _training():
    from research.notebook_views import pinned_engineering
    directory = pinned_engineering()     # pinned in notebook_views.ENGINEERING_STUDY, not "newest"
    loaded = load_engineering_study(directory, "training")
    if loaded["scope"].get("test_csv_loaded"):
        raise ValueError("The engineering training scope records that the test file was loaded.")
    loaded["directory"] = Path(directory)
    return loaded


def _model_name(key):
    names = {
        "logistic": "Logistic regression",
        "naive_bayes": "Gaussian naive Bayes",
        "random_forest": "Random forest",
        "grad_boosting": "Gradient boosting",
    }
    return names.get(str(key), str(key))
