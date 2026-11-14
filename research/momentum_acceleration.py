"""Declared deceleration-to-cash rules on the unchanged grouped top-ten book.

Acceleration here is a discrete change in monthly return, r_t - r_(t-1).
Current selected membership is held fixed across the two historical months.
There is no classification, parameter search or live fitting in the readers.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .accounting import portfolio_with_cash
from .basket_momentum import (
    FEATURE,
    assign_correlation_baskets,
    positive_volatility,
    select_top_positive_vol,
    concentrated_basket_weights,
    realized_returns,
    _metrics,
    _monthly,
    _digest,
)
from .legacy import KEYS, TARGET, load_panel, return_month

VARIANTS = (
    "Original",
    "Stock deceleration",
    "Basket deceleration",
    "Stock + basket deceleration",
)
METHODS = {"kmeans": "K-means", "ward": "Ward"}


def acceleration_signals(history, selected):
    """Already-earned changes for stocks and today's fixed-membership baskets."""
    if selected.duplicated(KEYS).any() or selected[KEYS + [FEATURE]].isna().any().any():
        raise ValueError("Need unique keyed selected stocks and current group IDs.")
    wide = realized_returns(history)
    parts = []
    for date, block in selected.groupby("date", sort=True):
        date = pd.Timestamp(date)
        past = date - pd.offsets.MonthEnd(1)
        window = wide.reindex([past, date])
        current = block.copy()
        current["stock_return_now"] = current.stock_id.map(window.loc[date])
        current["stock_return_previous"] = current.stock_id.map(window.loc[past])
        current["stock_acceleration"] = (
            current.stock_return_now - current.stock_return_previous
        )
        current["basket_return_now"] = np.nan
        current["basket_return_previous"] = np.nan
        for group, members in current.groupby(FEATURE, sort=True):
            paths = window.reindex(columns=members.stock_id)
            # Missing observations cannot be silently replaced with another universe.
            if not paths.notna().all().all():
                continue
            mask = current[FEATURE].eq(group)
            current.loc[mask, "basket_return_now"] = paths.loc[date].mean()
            current.loc[mask, "basket_return_previous"] = paths.loc[past].mean()
        current["basket_acceleration"] = (
            current.basket_return_now - current.basket_return_previous
        )
        parts.append(current)
    if parts:
        return pd.concat(parts, ignore_index=True)
    # An empty screen means an all-cash book, with a valid empty signal audit.
    return selected.assign(
        **{
            name: pd.Series(dtype=float)
            for name in (
                "stock_return_now",
                "stock_return_previous",
                "stock_acceleration",
                "basket_return_now",
                "basket_return_previous",
                "basket_acceleration",
            )
        }
    )


def adjust_for_deceleration(weights, signals, variant, strength=0.5):
    """Multiply only existing positions; removed capital remains zero-return cash."""
    if variant not in VARIANTS or not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Choose a declared variant and haircut in [0, 1].")
    if weights.duplicated(KEYS).any() or signals.duplicated(KEYS).any():
        raise ValueError("Weight/signal keys must be unique.")
    if not np.isfinite(weights.weight).all() or weights.weight.lt(0).any():
        raise ValueError("Baseline weights must be finite and nonnegative.")
    if weights.groupby("date").weight.sum().gt(1 + 1e-10).any():
        raise ValueError("Baseline stock exposure exceeds total capital.")
    unknown = signals[KEYS].merge(weights[KEYS], on=KEYS, how="left", indicator=True)
    if not unknown._merge.eq("both").all():
        raise ValueError("Signals reference stocks outside the baseline universe.")
    data = weights.merge(
        signals[KEYS + ["stock_acceleration", "basket_acceleration"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    multiplier = pd.Series(1.0, index=data.index)
    if variant in ("Stock deceleration", "Stock + basket deceleration"):
        multiplier *= np.where(data.stock_acceleration.lt(0), 1 - strength, 1.0)
    if variant in ("Basket deceleration", "Stock + basket deceleration"):
        multiplier *= np.where(data.basket_acceleration.lt(0), 1 - strength, 1.0)
    data["weight"] *= multiplier
    return data[KEYS + ["weight"]]


def _check(study, window):
    scope, monthly = study["scope"], study["monthly"]
    if (
        window not in ("training", "test")
        or scope["window"] != window
        or bool(scope["test_csv_loaded"]) != (window == "test")
    ):
        raise ValueError("Wrong acceleration evaluation scope.")
    dates = pd.to_datetime(monthly.return_month)
    lower, upper = (
        ("2000-01-31", "2009-12-31")
        if window == "training"
        else ("2010-01-31", "2018-12-31")
    )
    if (
        dates.empty
        or dates.min() < pd.Timestamp(lower)
        or dates.max() > pd.Timestamp(upper)
    ):
        raise ValueError("Acceleration dates cross the declared evaluation boundary.")
    if (
        dates.min() != pd.Timestamp(scope["first_return_month"])
        or dates.max() != pd.Timestamp(scope["last_return_month"])
        or dates.nunique() != scope["months"]
    ):
        raise ValueError("Acceleration dates do not match the saved scope.")
    if not (pd.to_datetime(monthly.snapshot_date) + pd.offsets.MonthEnd(1)).equals(
        dates
    ):
        raise ValueError("Acceleration decisions must precede earned returns.")


def create_acceleration_study(root, progress=print):
    """Write a plan, compute training first, then a separate fixed-rule final check.

    Strength is declared at 50%, never selected on test. The original positive
    momentum/cash rules remain unchanged; only the extra multipliers differ.
    """
    if os.environ.get("MLFACTOR_READ_ONLY") == "1":
        raise RuntimeError("Acceleration computation is disabled in view mode.")
    root = Path(root)
    directory = (
        root
        / "artifacts"
        / "acceleration_studies"
        / (
            "acceleration_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "_"
            + uuid.uuid4().hex[:8]
        )
    )
    directory.mkdir(parents=True)
    metadata = {
        "protocol": "grouped_momentum_deceleration",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "methods": list(METHODS),
        "variants": list(VARIANTS),
        "strength": 0.5,
        "stock_groups": 5,
        "top_per_group": 10,
        "grouping_months": 12,
        "positive_vol_months": 12,
        "momentum_months": 1,
        "acceleration": "r_t - r_(t-1), using already-earned simple monthly returns",
        "basket_membership": "current selected members held fixed for both observations",
        "missing_signal": "neutral multiplier 1; missing comparisons are not classified as deceleration",
        "cash_return": 0,
        "transaction_costs": 0,
        "new_directional_fits": 0,
        "tuning": "none",
        "test_previously_viewed": True,
        "code_sha256": _digest(Path(__file__)),
        "source_train_sha256": _digest(root / "mlfactor_long_only_train.csv"),
    }
    (directory / "study.json").write_text(json.dumps(metadata, indent=2) + "\n")
    train = load_panel("train", root)
    for window in ("training", "test"):
        if window == "training":
            history = train
            panel = train.loc[
                train.date.ge(pd.Timestamp("2004-12-31")), KEYS + [TARGET]
            ].copy()
        else:
            test = load_panel("test", root)
            if train.date.max() >= test.date.min():
                raise ValueError("Train/test snapshots overlap.")
            history = (
                pd.concat([train, test], ignore_index=True)
                .sort_values(KEYS)
                .reset_index(drop=True)
            )
            panel = test[KEYS + [TARGET]].copy()
        dates = pd.DatetimeIndex(panel.date.unique()).sort_values()
        frames = {name: [] for name in ("metrics", "monthly", "signals")}
        for method in METHODS:
            progress(f"{window}: {METHODS[method]} top-ten momentum/acceleration")
            with threadpool_limits(limits=1):
                assigned = assign_correlation_baskets(
                    history, dates, grouping_method=method
                )
            groups = panel[KEYS].merge(
                assigned[KEYS + [FEATURE]], on=KEYS, validate="one_to_one"
            )
            selected = select_top_positive_vol(
                groups, positive_volatility(history, dates)
            )
            baseline, _, _ = concentrated_basket_weights(history, panel, selected, 1)
            signals = acceleration_signals(history, selected)
            audit = signals.merge(
                baseline.rename(columns={"weight": "original_weight"}),
                on=KEYS,
                validate="one_to_one",
            )
            for variant in VARIANTS:
                allocation = adjust_for_deceleration(
                    baseline, signals, variant, metadata["strength"]
                )
                book = portfolio_with_cash(panel, allocation)
                identity = dict(method=method, variant=variant)
                frames["metrics"].append(
                    pd.DataFrame([_metrics(book, variant, 1)]).assign(**identity)
                )
                frames["monthly"].append(_monthly(book, variant, 1).assign(**identity))
                if variant != "Original":
                    audit[variant] = (
                        audit[KEYS]
                        .merge(allocation, on=KEYS, validate="one_to_one")
                        .weight.to_numpy()
                    )
            frames["signals"].append(audit.assign(method=method))
        stage = directory / window
        stage.mkdir()
        scope = {
            "window": window,
            "test_csv_loaded": window == "test",
            "first_return_month": str(return_month(panel.date).min().date()),
            "last_return_month": str(return_month(panel.date).max().date()),
            "months": len(dates),
            "source_test_sha256": (
                _digest(root / "mlfactor_long_only_test.csv")
                if window == "test"
                else None
            ),
        }
        study = {
            "scope": scope,
            **{
                name: pd.concat(parts, ignore_index=True)
                for name, parts in frames.items()
            },
        }
        _check(study, window)
        record = {"scope": scope, "frames": {}}
        for name in frames:
            path = stage / (name + ".parquet")
            study[name].to_parquet(path, index=False)
            record["frames"][name] = _digest(path)
        (stage / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
    return directory


def load_acceleration_study(directory, window="training"):
    directory = Path(directory)
    record = json.loads((directory / window / "complete.json").read_text())
    study = {
        "metadata": json.loads((directory / "study.json").read_text()),
        "scope": record["scope"],
    }
    for name, expected in record["frames"].items():
        path = directory / window / (name + ".parquet")
        if _digest(path) != expected:
            raise ValueError(f"Damaged acceleration frame: {name}")
        study[name] = pd.read_parquet(path)
    _check(study, window)
    return study
