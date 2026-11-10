"""Training-holdout return categories. KNN and logistic regression.

Reads the newest grouping study. It does not fit a model and does not open
a course CSV. The frozen 2010–2018 books are a different save.
"""
from __future__ import annotations

import pandas as pd

from research.grouping import load_latest_grouping
from research.notebook_views import artifact_root


_GROWTH = (
    ("annualized_log_growth", "Annual log growth"),
    ("cagr", "CAGR"),
    ("annualized_volatility", "Volatility"),
    ("sharpe", "Sharpe"),
    ("max_drawdown", "Max drawdown"),
)


def grouping_search_table() -> pd.DataFrame:
    """One row per GridSearchCV model: what was searched, what was chosen, the score."""
    metadata = _metadata()
    rows = []
    for record in metadata["search"]:
        rows.append({
            "Model": record["model"],
            "Searched": record["searched"],
            "Chosen": record["chosen"],
            "Fits": record["n_fits"],
            "Best score": record["best_score"],
        })
    frame = pd.DataFrame(rows)
    frame["Fits"] = frame["Fits"].astype("int64")
    frame["Best score"] = frame["Best score"].astype(float)
    return frame


def grouping_class_table() -> pd.DataFrame:
    """Holdout accuracy and log loss against a uniform five-class guess."""
    rows = []
    for record in _metadata()["classification"]:
        rows.append({
            "Model": record["model"],
            "Holdout accuracy": record["holdout_accuracy"],
            "Chance accuracy": record["chance_accuracy"],
            "Holdout log loss": record["holdout_log_loss"],
            "Uniform log loss": record["uniform_log_loss"],
        })
    return pd.DataFrame(rows)


def grouping_book_table() -> pd.DataFrame:
    """Top predicted category versus equal weight, on the training holdout only."""
    rows = []
    for record in _metadata()["portfolio"]:
        row = {"Book": record["book"]}
        for source, header in _GROWTH:
            row[header] = record[source]
        rows.append(row)
    return pd.DataFrame(rows)


def grouping_note() -> str:
    """KNN is one of the model types. The numbers are the saved holdout, not 2010–2018."""
    metadata = _metadata()
    search = {row["model_key"]: row for row in metadata["search"]}
    classes = {row["model_key"]: row for row in metadata["classification"]}
    books = {row["model_key"]: row for row in metadata["portfolio"]}
    knn = search["knn"]
    knn_class = classes["knn"]
    knn_book = books["knn"]
    logistic = search["logistic"]
    equal = books["none"]
    start = pd.Timestamp(metadata["holdout_first_return_month"]).strftime("%B %Y")
    end = pd.Timestamp(metadata["holdout_last_return_month"]).strftime("%B %Y")
    return (
        "K nearest neighbors is one of the model types. "
        f"GridSearchCV chose {knn['chosen']} for K nearest neighbors and {logistic['chosen']} "
        f"for multinomial logistic regression, using labels through {metadata['asof']} and the {metadata['scoring']} score. "
        f"On the {start}–{end} training holdout, K nearest neighbors accuracy is "
        f"{knn_class['holdout_accuracy']:.1%} against a chance accuracy of {knn_class['chance_accuracy']:.1%}. "
        f"Its holdout log loss is {knn_class['holdout_log_loss']:.3f}, and logistic regression holdout log loss is "
        f"{classes['logistic']['holdout_log_loss']:.3f}, against a uniform-guess log loss of {knn_class['uniform_log_loss']:.3f}. "
        f"Its top-category book has annual log growth {knn_book['annualized_log_growth']:.2%} "
        f"and CAGR {knn_book['cagr']:.2%}, against equal weight at "
        f"{equal['annualized_log_growth']:.2%} and {equal['cagr']:.2%}. "
        "This search did not open the official test file. It is not inside the frozen 2010–2018 books, "
        "which still use the four factory classifiers. "
        "The basket book groups by K-means on return paths; this model groups by the predicted return quintile."
    )


def _metadata():
    return load_latest_grouping(artifact_root())["metadata"]
