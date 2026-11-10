"""Official-test paragraph for the declared book.

Numbers come from the newest final-book save. This module does not fit a
model, open a course CSV, or import the app.
"""
from __future__ import annotations

from research.notebook_views import (
    CHOICE,
    DISPLAY_MODEL,
    STORY,
    load_final_windows,
    model_label,
    story_slice,
)

# A smaller CAGR gap is the same book, not a different result.
_SAME_BOOK = 0.002


def choice_text() -> str:
    """Four to six sentences on the declared book, read from the official test."""
    study, _training, windows = load_final_windows()
    metrics, _monthly = windows["test"]
    try:
        story = story_slice(metrics)
    except ValueError:
        story = metrics.iloc[0:0]

    version, model_key = _traded_key()
    chosen = _find(story, metrics, version, model_key)
    if chosen is None:
        return _missing(metrics, version, model_key)

    name = _book_name(version, chosen)
    if model_key != "none":
        name = f"{name}, using {model_label(DISPLAY_MODEL)}"
    equal = _find(story, metrics, "Equal weight", "none")
    sentences = [_lead(name, chosen, equal), _against_equal(chosen, equal)]
    support = _support(story, metrics, version, chosen, equal)
    if support:
        sentences.append(support)
    sentences.append(str(CHOICE["disclosure"]))
    recorded = _fit_clause(study.get("metadata") or {})
    if recorded:
        sentences.append(recorded)
    return " ".join(sentences)


def _traded_key():
    overlay = CHOICE.get("overlay_version")
    if overlay:
        return str(overlay), DISPLAY_MODEL
    return str(CHOICE["version"]), "none"


def _momentum(version, model_key) -> int:
    overlay = model_key not in (None, "none", "")
    for item in STORY:
        if item["version"] != version:
            continue
        if overlay and item["role"] != "overlay":
            continue
        if not overlay and item["role"] == "overlay":
            continue
        return int(item["momentum"])
    for item in STORY:
        if item["version"] == version:
            return int(item["momentum"])
    return 1


def _find(story, metrics, version, model_key):
    if not story.empty and {"version", "model_key"}.issubset(story.columns):
        part = story.loc[story.version.eq(version) & story.model_key.eq(model_key)]
        if not part.empty:
            return part.iloc[0]
    momentum = _momentum(version, model_key)
    part = metrics.loc[
        metrics.version.eq(version)
        & metrics.model_key.eq(model_key)
        & metrics.momentum_months.eq(momentum)
    ]
    if part.empty:
        return None
    return part.iloc[0]


def _book_name(version, row=None) -> str:
    if row is not None and "book" in row.index:
        book = row["book"]
        if isinstance(book, str) and book:
            return book
    for item in STORY:
        if item["version"] == version:
            return item["book"]
    return str(version)


def _missing(metrics, version, model_key) -> str:
    name = _book_name(version)
    if model_key != "none":
        name = f"{name}, {model_label(DISPLAY_MODEL)}"
    present = metrics.loc[metrics.version.eq(version) & metrics.model_key.eq(model_key)]
    if present.empty:
        return f"The declared choice, {name}, is not in the newest save."
    moment = _momentum(version, model_key)
    return (
        f"The declared choice, {name}, is not in the newest save at momentum {moment}."
    )


def _lead(name, chosen, equal) -> str:
    own = _four(chosen)
    if equal is None:
        return f"The book I would trade is {name}, with an official-test {own}."
    return (
        f"The book I would trade is {name}, with an official-test {own}, "
        f"against equal weight at {_four(equal)}."
    )


def _four(row) -> str:
    return (
        f"annual log growth of {_pct(row.annualized_log_growth)}, "
        f"CAGR of {_pct(row.cagr)}, volatility of {_pct(row.annualized_volatility)}, "
        f"a Sharpe of {_sharpe(row.sharpe)}, and a max drawdown of {_pct(row.max_drawdown)}"
    )


def _against_equal(chosen, equal) -> str:
    if equal is None:
        return "Equal weight is not in the newest save, so this cell does not invent that comparison."
    cagr = _versus(chosen.cagr, equal.cagr, _pct)
    sharpe = _versus(chosen.sharpe, equal.sharpe, _sharpe)
    return f"Its annual log growth and its CAGR are {cagr} and its Sharpe is {sharpe}."


def _versus(left, right, formatter) -> str:
    if formatter(left) == formatter(right):
        return "the same as equal weight"
    if float(left) > float(right):
        return "higher than equal weight"
    return "lower than equal weight"


def _support(story, metrics, traded_version, chosen, equal) -> str:
    pieces = []
    overlays = _overlay_phrases(story, metrics, traded_version, chosen)
    if overlays:
        pieces.append(f"For {model_label(DISPLAY_MODEL)}, {_join(overlays)}")
    basket = _find(story, metrics, "Basket momentum + cash", "none")
    if basket is not None and equal is not None:
        pieces.append(_basket_clause(_book_name("Basket momentum + cash", basket), basket, equal))
    if not pieces:
        return ""
    if len(pieces) == 1:
        return pieces[0] + "."
    return f"{pieces[0]}, while {pieces[1]}."


def _overlay_phrases(story, metrics, traded_version, chosen) -> list[str]:
    base = _unadjusted(story, metrics, traded_version, chosen)
    if base is None:
        return []
    phrases = []
    for version in _family_overlays(traded_version):
        row = _overlay_row(metrics, version)
        if row is None:
            continue
        phrases.append(_overlay_phrase(_book_name(version), row.cagr, base.cagr))
    return phrases


def _unadjusted(story, metrics, traded_version, chosen):
    if str(chosen.model_key) in {"none", ""}:
        return chosen
    declared = _find(story, metrics, str(CHOICE["version"]), "none")
    if declared is not None:
        return declared
    base_version = _family_base(traded_version)
    if base_version is None:
        return None
    return _find(story, metrics, base_version, "none")


def _family_overlays(version):
    text = str(version)
    if "Top 10" in text:
        return ("Top 10 + downside to cash", "Top 10 + upside tilt")
    if "Basket" in text:
        return ("Baskets + downside to cash", "Baskets + upside tilt")
    return ()


def _family_base(version):
    text = str(version)
    if "Top 10" in text:
        return "Top 10 positive-vol baskets"
    if "Basket" in text:
        return "Basket momentum + cash"
    return None


def _overlay_row(metrics, version):
    part = metrics.loc[
        metrics.version.eq(version)
        & metrics.momentum_months.eq(1)
        & metrics.model_key.eq(DISPLAY_MODEL)
    ]
    if part.empty:
        return None
    return part.iloc[0]


def _overlay_phrase(name, overlay_cagr, base_cagr) -> str:
    delta = float(overlay_cagr) - float(base_cagr)
    if abs(delta) < _SAME_BOOK:
        return f"{name} does not change the book"
    return (
        f"{name} changes CAGR by {delta * 100:+.1f} percentage points "
        "versus the unadjusted book"
    )


def _basket_clause(name, row, equal) -> str:
    cagr = _versus(row.cagr, equal.cagr, _pct)
    sharpe = _versus(row.sharpe, equal.sharpe, _sharpe)
    if cagr == sharpe:
        tail = cagr if cagr == "the same as equal weight" else f"both {cagr}"
    else:
        tail = f"a CAGR {cagr} and a Sharpe {sharpe}"
    return (
        f"the test CAGR for {name} is {_pct(row.cagr)} and its Sharpe is {_sharpe(row.sharpe)}, {tail}"
    )


def _join(phrases) -> str:
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]}, and {phrases[1]}"
    return ", ".join(phrases[:-1]) + ", and " + phrases[-1]


def _fit_clause(meta) -> str:
    if not isinstance(meta, dict):
        return ""
    pieces = []
    for key in ("fit_policy", "fit_snapshot", "n_features"):
        if key not in meta or meta[key] is None:
            continue
        pieces.append(f"{key} {meta[key]}")
    if not pieces:
        return ""
    return f"The final study metadata records {_join(pieces)}."


def _pct(value) -> str:
    return f"{float(value):.1%}"


def _sharpe(value) -> str:
    return f"{float(value):.3f}"
