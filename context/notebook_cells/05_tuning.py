"""Read the newest training-holdout search. No fitting and no pipeline files."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.notebook_views import latest_holdout, model_label


SEARCH_COLUMNS = ["Model", "Searched", "Chosen", "Fits", "Best score"]
PORTFOLIO_COLUMNS = [
    "Strategy", "Model", "No ML CAGR", "Untuned CAGR", "Tuned CAGR", "Tuned minus untuned",
]
EQUAL_WEIGHT = "equal_weight_universe"


def tuning_search_table() -> pd.DataFrame:
    """One row per model in the saved GridSearchCV choice, registry order first."""
    metadata = latest_holdout()["metadata"]
    chosen = metadata.get("chosen") or {}
    keys = _registry_order(list(chosen) or list(metadata.get("model_keys") or []))
    rows = []
    for key in keys:
        record = _chosen_record(metadata, key)
        rows.append({
            "Model": model_label(key),
            "Searched": _compact(record.get("grid") or {}),
            "Chosen": _compact(record.get("params") or {}),
            "Fits": record.get("n_fits"),
            "Best score": record.get("best_score"),
        })
    frame = pd.DataFrame(rows, columns=SEARCH_COLUMNS)
    frame["Fits"] = pd.to_numeric(frame["Fits"], errors="coerce").astype("Int64")
    frame["Best score"] = pd.to_numeric(frame["Best score"], errors="coerce")
    return frame


def tuning_portfolio_table() -> pd.DataFrame:
    """Equal weight plus the two registry books whose average CAGR moved most."""
    loaded = latest_holdout()
    metadata = loaded["metadata"]
    metrics = loaded["metrics"]
    labels = _labels(metrics)
    models = _study_models(metadata, metrics)
    moves = {
        strategy: move
        for strategy in _strategy_keys(metadata, metrics)
        if (move := _average_move(metrics, strategy, models)) is not None
    }
    extras = sorted(
        (strategy for strategy in moves if strategy != EQUAL_WEIGHT),
        key=lambda strategy: (-abs(moves[strategy]), labels.get(strategy, strategy), strategy),
    )
    selected = []
    if EQUAL_WEIGHT in _strategy_keys(metadata, metrics):
        selected.append(EQUAL_WEIGHT)
    selected.extend(extras[:2])
    rows = []
    for strategy in selected:
        baseline = _cagr(metrics, strategy, "No ML", "none")
        for model in models:
            untuned = _cagr(metrics, strategy, "ML, not tuned", model)
            tuned = _cagr(metrics, strategy, "Tuned ML", model)
            rows.append({
                "Strategy": labels.get(strategy, strategy),
                "Model": model_label(model),
                "No ML CAGR": baseline,
                "Untuned CAGR": untuned,
                "Tuned CAGR": tuned,
                "Tuned minus untuned": tuned - untuned,
            })
    return pd.DataFrame(rows, columns=PORTFOLIO_COLUMNS)


def tuning_note() -> str:
    """Three sentences: the search, the equal-weight move, and what the official test used."""
    metadata = latest_holdout()["metadata"]
    start = pd.Timestamp(metadata["holdout_first_return_month"]).strftime("%B %Y")
    end = pd.Timestamp(metadata["holdout_last_return_month"]).strftime("%B %Y")
    scoring = metadata.get("scoring") or "its saved score"
    target = metadata.get("prediction_target")
    target_clause = f" and the {target} prediction target" if target else ""
    first = (
        f"GridSearchCV searched the {start}–{end} training holdout "
        f"with the {scoring} score{target_clause}."
    )
    portfolio = tuning_portfolio_table()
    label = _labels(latest_holdout()["metrics"]).get(EQUAL_WEIGHT, EQUAL_WEIGHT)
    gaps = portfolio.loc[portfolio["Strategy"].eq(label), "Tuned minus untuned"]
    move = float(gaps.abs().max()) if not gaps.empty else float("nan")
    if pd.isna(move):
        second = "The equal-weight holdout book has no tuned-minus-untuned CAGR in this save."
    else:
        moved = "so the portfolio barely moved" if move < 0.005 else "so tuning moved that portfolio"
        second = (
            "The largest absolute tuned-minus-untuned CAGR on the equal-weight holdout book "
            f"is {move:.2%}, {moved}."
        )
    third = (
        "These holdout rows are registry rules on the training cut; the official 2010–2018 books "
        "in the other cells use factory classifier settings, so this search is not the model inside that test."
    )
    return " ".join((first, second, third))


def _registry_order(keys):
    from research.models import default_model_registry
    registry = list(default_model_registry())
    unique = []
    for key in keys:
        if key not in unique and key not in (None, "", "none"):
            unique.append(key)
    return [key for key in registry if key in unique] + [key for key in unique if key not in registry]


def _study_models(metadata, metrics):
    if metadata.get("model_keys"):
        keys = list(metadata["model_keys"])
    elif metadata.get("chosen"):
        keys = list(metadata["chosen"])
    else:
        keys = list(metrics["model_key"].drop_duplicates())
    return _registry_order(keys)


def _strategy_keys(metadata, metrics):
    listed = [key for key in (metadata.get("strategies") or [])]
    present = list(dict.fromkeys([*listed, *metrics["strategy"].drop_duplicates()]))
    return present


def _labels(metrics):
    labels = {}
    for strategy, label in zip(metrics["strategy"], metrics["label"]):
        if strategy not in labels and pd.notna(label):
            labels[strategy] = str(label)
    return labels


def _chosen_record(metadata, model_key):
    record = dict((metadata.get("chosen") or {}).get(model_key) or {})
    if record.get("params"):
        return record
    # Chosen params live on the study metadata. Open tuning.json only when they are absent.
    directory = metadata.get("directory")
    path = Path(directory) / str(model_key) / "tuning.json" if directory else None
    if path is None or not path.is_file():
        return record
    saved = json.loads(path.read_text())
    record.setdefault("params", saved.get("chosen_params") or saved.get("params") or {})
    record.setdefault("grid", saved.get("grid") or {})
    record.setdefault("n_fits", saved.get("n_fits"))
    record.setdefault("best_score", saved.get("best_score"))
    return record


def _compact(mapping):
    if not mapping:
        return pd.NA
    parts = []
    for key, value in mapping.items():
        name = str(key).removeprefix("model__")
        if isinstance(value, (list, tuple)):
            shown = ", ".join(_format_scalar(item) for item in value)
        else:
            shown = _format_scalar(value)
        parts.append(f"{name}: {shown}")
    return "; ".join(parts)


def _format_scalar(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return format(value, ".12g")
    return str(value)


def _cagr(metrics, strategy, version, model_key):
    rows = metrics.loc[
        metrics["strategy"].eq(strategy)
        & metrics["version"].eq(version)
        & metrics["model_key"].eq(model_key),
        "cagr",
    ]
    if rows.empty or pd.isna(rows.iloc[0]):
        return float("nan")
    return float(rows.iloc[0])


def _average_move(metrics, strategy, models):
    deltas = []
    for model in models:
        untuned = _cagr(metrics, strategy, "ML, not tuned", model)
        tuned = _cagr(metrics, strategy, "Tuned ML", model)
        if pd.notna(untuned) and pd.notna(tuned):
            deltas.append(tuned - untuned)
    if not deltas:
        return None
    return sum(deltas) / len(deltas)
