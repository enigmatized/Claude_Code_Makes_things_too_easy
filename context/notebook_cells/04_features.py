"""Training-window feature cuts for the graded notebook.

Reads the newest feature-score save and the newest final-book save.
Does not fit, does not open a course CSV, and does not import the app.
The feature-score window is the training years recorded on that save,
not the official test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.notebook_views import (  # noqa: E402
    DISPLAY_MODEL,
    STORY,
    latest_feature_scores,
    load_final_windows,
    model_label,
)

# A move smaller than this is the notebook's "does not meaningfully change" line.
# 0.003 is 0.30 percentage points. The saved cuts sit near 0.20 points.
_SMALL_GAP = 0.003
_GAP_COLUMN = "Largest gap vs all 20"


def _recipe_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _recipe_columns(metadata, present):
    """Metadata recipe order, then any extra key alphabetically.

    Labels come from metadata when that save stored them. An unknown key
    keeps the key as its label. The empty recipe is the unadjusted book
    and is not a column.
    """
    present_keys = []
    seen = set()
    for value in present:
        key = _recipe_text(value)
        if key == "" or key in seen:
            continue
        seen.add(key)
        present_keys.append(key)
    labels = {}
    ordered = []
    recipes = metadata.get("recipes") if isinstance(metadata, dict) else None
    entries = []
    if isinstance(recipes, list):
        entries = [(item.get("key"), item) for item in recipes if isinstance(item, dict)]
    elif isinstance(recipes, dict):
        entries = list(recipes.items())
    for raw_key, item in entries:
        key = _recipe_text(raw_key)
        if key == "" or key not in seen or key in ordered:
            continue
        if isinstance(item, dict) and item.get("label"):
            labels[key] = str(item["label"])
        else:
            labels[key] = key
        ordered.append(key)
    extras = sorted(key for key in present_keys if key not in ordered)
    for key in extras:
        labels.setdefault(key, key)
    headers = []
    used = {}
    for key in ordered + extras:
        label = labels[key]
        count = used.get(label, 0)
        used[label] = count + 1
        headers.append(label if count == 0 else f"{label} ({key})")
    return ordered + extras, headers, labels


def _overlay_versions():
    return [item["version"] for item in STORY if item["role"] == "overlay"]


def _story_rows(metrics, model_key):
    found, missing = [], []
    for item in STORY:
        if item["role"] != "overlay":
            continue
        part = metrics.loc[
            metrics["version"].eq(item["version"])
            & metrics["momentum_months"].eq(item["momentum"])
            & metrics["model_key"].eq(model_key)
        ]
        part = part.loc[part["recipe"].map(_recipe_text).ne("")]
        if part.empty:
            missing.append(item["book"])
            continue
        found.append((item, part))
    return found, missing


def _fallback_model(metrics):
    """First overlay model_key, other than DISPLAY_MODEL, that has a story row."""
    versions = set(_overlay_versions())
    overlay = metrics.loc[metrics["version"].isin(versions)]
    seen = []
    for value in overlay["model_key"]:
        key = _recipe_text(value)
        if key in ("", "none", DISPLAY_MODEL) or key in seen:
            continue
        seen.append(key)
    for key in seen:
        found, _missing = _story_rows(metrics, key)
        if found:
            return key
    return seen[0] if seen else None


def _window_span(metadata) -> str:
    window = {}
    if isinstance(metadata, dict):
        window = metadata.get("window") or {}
    first = window.get("first_return_month")
    last = window.get("last_return_month")
    months = window.get("months")
    if first and last and months not in (None, ""):
        return f"{first} through {last}, {months} months"
    if first and last:
        return f"{first} through {last}"
    return "the months recorded on that save"


def _cagr_by_recipe(part) -> dict:
    found = {}
    recipes = part["recipe"].map(_recipe_text)
    for recipe, group in part.groupby(recipes, sort=False):
        if recipe == "":
            continue
        value = group["cagr"].iloc[0]
        if pd.isna(value):
            continue
        found[recipe] = float(value)
    return found


def _largest_row_gap(cagr_by_recipe, keys):
    """Signed CAGR minus all_20 for the recipe with the largest absolute gap."""
    if "all_20" not in keys or "all_20" not in cagr_by_recipe:
        return float("nan"), None
    base = cagr_by_recipe["all_20"]
    best_abs = -1.0
    best_gap = 0.0
    best_key = None
    for key in keys:
        if key == "all_20" or key not in cagr_by_recipe:
            continue
        gap = cagr_by_recipe[key] - base
        if abs(gap) > best_abs:
            best_abs = abs(gap)
            best_gap = gap
            best_key = key
    if best_key is None:
        return 0.0, "all_20"
    return best_gap, best_key


def feature_cut_table() -> pd.DataFrame:
    """One CAGR column per non-empty recipe for the story overlay books.

    The window is whatever the newest feature-score study recorded.
    Those months are training years, not the official test.
    """
    loaded = latest_feature_scores()
    metadata = loaded["metadata"]
    metrics = loaded["metrics"].copy()
    metrics["momentum_months"] = pd.to_numeric(metrics["momentum_months"], errors="coerce")
    found, missing = _story_rows(metrics, DISPLAY_MODEL)
    model_key = DISPLAY_MODEL
    fell_back = False
    if not found:
        fallback = _fallback_model(metrics)
        if fallback is not None:
            model_key = fallback
            fell_back = True
            found, missing = _story_rows(metrics, model_key)
    present = []
    parsed = []
    for item, part in found:
        cagr_by_recipe = _cagr_by_recipe(part)
        present.extend(cagr_by_recipe)
        parsed.append((item, cagr_by_recipe))
    keys, headers, labels = _recipe_columns(metadata, present)
    columns = ["Book", *headers]
    if "all_20" in keys:
        columns.append(_GAP_COLUMN)
    records = []
    gaps = []
    for item, cagr_by_recipe in parsed:
        row = {"Book": item["book"]}
        for key, header in zip(keys, headers):
            row[header] = cagr_by_recipe.get(key, float("nan"))
        gap, gap_key = _largest_row_gap(cagr_by_recipe, keys)
        if "all_20" in keys:
            row[_GAP_COLUMN] = gap
        label = None if gap_key is None else labels.get(gap_key, gap_key)
        gaps.append({"book": item["book"], "gap": gap, "recipe_key": gap_key, "recipe_label": label})
        records.append(row)
    frame = pd.DataFrame(records, columns=columns)
    for column in frame.columns:
        if column != "Book":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    largest = None
    for note in gaps:
        gap = note["gap"]
        if gap is None or pd.isna(gap):
            continue
        if largest is None or abs(gap) > abs(largest["gap"]):
            largest = {"book": note["book"], "gap": float(gap),
                       "recipe_key": note["recipe_key"], "recipe_label": note["recipe_label"]}
    span = _window_span(metadata)
    frame.attrs["window_span"] = span
    frame.attrs["window"] = (
        f"Training years recorded by the newest feature-score study ({span}), not the official test."
    )
    frame.attrs["missing_books"] = missing
    frame.attrs["model_key"] = model_key
    frame.attrs["fell_back"] = fell_back
    frame.attrs["largest_gap"] = largest
    shown = model_label(model_key)
    if fell_back:
        frame.attrs["model_note"] = (
            f"{DISPLAY_MODEL} is absent from the overlay rows at the story momentum. "
            f"The table uses {shown} ({model_key})."
        )
    else:
        frame.attrs["model_note"] = f"Overlay rows for {shown} ({model_key})."
    return frame


def _join_and(items) -> str:
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return str(items[0])
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(str(item) for item in items[:-1]) + f", and {items[-1]}"


def _name_sentence() -> str:
    study, _training_meta, _windows = load_final_windows()
    fitted = study["metadata"].get("fitted_features")
    label = model_label(DISPLAY_MODEL)
    if not isinstance(fitted, dict) or DISPLAY_MODEL not in fitted:
        return f"The newest final save does not record a list for {label}."
    chosen = fitted[DISPLAY_MODEL]
    if not isinstance(chosen, dict) or not chosen:
        return f"The newest final save does not record a list for {label}."
    parts = []
    for target, columns in chosen.items():
        if not isinstance(columns, (list, tuple)):
            return f"The newest final save does not record a list for {label}."
        parts.append((str(target), [str(name) for name in columns]))
    same = len({tuple(names) for _target, names in parts}) == 1
    if same:
        names = parts[0][1]
        targets = _join_and(target for target, _names in parts)
        listed = ", ".join(names)
        return (
            f"Inside the frozen fit, {len(names)} names were chosen for {label}, "
            f"and {targets} use the same list: {listed}."
        )
    grouped = []
    for target, names in parts:
        if names:
            grouped.append(f"{len(names)} {target} names ({', '.join(names)})")
        else:
            grouped.append(f"no {target} names")
    return f"Inside the frozen fit, {_join_and(grouped)} were chosen for {label}."


def _gap_sentence(table) -> str:
    span = table.attrs.get("window_span") or "the months recorded on that save"
    model_key = table.attrs.get("model_key", DISPLAY_MODEL)
    shown = model_label(model_key)
    where = (
        f"On the training years recorded by the newest feature-score study ({span}), "
        f"not the official test, "
    )
    who = shown
    if table.attrs.get("fell_back"):
        who = f"{shown} ({model_key}, because {DISPLAY_MODEL} is absent)"
    largest = table.attrs.get("largest_gap")
    if not largest or _GAP_COLUMN not in table.columns:
        return where + f"{who} has no all-20 CAGR, so no gap versus all 20 is shown."
    gap = float(largest["gap"])
    book = largest["book"]
    recipe = largest["recipe_label"] or largest["recipe_key"]
    amount = f"{abs(gap) * 100:.2f} percentage points"
    if gap < 0:
        direction = f"{amount} below all 20"
    elif gap > 0:
        direction = f"{amount} above all 20"
    else:
        direction = "0, matching all 20"
    sentence = (
        f"{where}the largest absolute CAGR gap versus all 20 for {who} "
        f"is {direction}, on {book} under {recipe}"
    )
    if abs(gap) < _SMALL_GAP:
        sentence += "; the cut does not meaningfully change the book"
        if recipe is not None and "10" in str(recipe):
            sentence += ", and it satisfies a ten-or-fewer feature limit"
    return sentence + "."


def feature_name_note() -> str:
    """Frozen-fit names, then the largest training-window CAGR gap versus all 20."""
    return _name_sentence() + " " + _gap_sentence(feature_cut_table())
