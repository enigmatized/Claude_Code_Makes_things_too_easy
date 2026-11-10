"""Correlation baskets, then positive momentum, with the residual left in cash.

Each month, stocks with a complete trailing return path are grouped by a
K-means model. The inputs are the z-scored simple returns earned through that
snapshot, not the twenty factor ranks and not the still-unrealized forward
return. Euclidean distance on those paths groups stocks whose recent returns
moved together. The cluster id is a new within-month feature,
``correlation_basket``. The id is fit again next month, so it is not a sector
and the same number does not mean the same stocks later.

The trading signal is a short trailing return: one, two, or three months.
One month is the last earned simple return. Two and three months compound
those earned simple returns. A group whose own return over that same short
window is not positive is left in cash, including members that were
individually positive. Inside a positive group, only members with a positive
short-window return are held, in proportion to that return. Every other slot
stays in cash. Cash earns 0%. The twelve-month path is used only to decide
the groups, because a one-month number has no co-movement to cluster.

A second book makes each group smaller before that rule. Inside every group
it keeps the names with the highest positive volatility, at most ten. Positive
volatility is the root-mean-square of up months over the same twelve-month
path: down months contribute zero and still stay in the average. The smaller
book then trades one- and two-month momentum. Its capital is scaled to the
names that remain, so a name that missed the ten is out of the book.

The comparison books use the same months:
equal weight is fully invested; momentum-plus-cash uses the same positive-only
rule with no grouping model. This strategy is not part of the fourteen-rule
registry, and this module is not part of the experiment-catalog code hash.
The official test file is not opened.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .accounting import portfolio_with_cash
from .legacy import KEYS, PROJECT_ROOT, TARGET, load_panel, performance, portfolio_returns, return_month


LOOKBACK = 12
MOMENTUM_MONTHS = (1, 2, 3)
N_CLUSTERS = 5
FEATURE = "correlation_basket"
TOP_VOL_COUNT = 10
TOP_VOL_MOMENTUM = (1, 2)
TOP_VOL_VERSION = "Top 10 positive-vol baskets"
VERSIONS = ("Equal weight", "Momentum + cash", "Basket momentum + cash", TOP_VOL_VERSION)
RANDOM_STATE = 0


def realized_returns(history):
    """Pivot to earned months. The label on snapshot t is stored at t+1.

    The row indexed by a snapshot date contains only returns already earned
    by that date. Reindexing to the snapshot calendar leaves the first month
    empty, matching the signal builder used by the other strategies.
    """
    required = set(KEYS + [TARGET])
    if history.empty or not required.issubset(history.columns):
        raise ValueError("Need stock-month keys and forward returns.")
    frame = history[KEYS + [TARGET]].copy()
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("Return keys must be unique and nonmissing.")
    if not pd.api.types.is_datetime64_any_dtype(frame.date) or not frame.date.dt.is_month_end.all():
        raise ValueError("Dates must be parsed month-ends.")
    if not np.isfinite(frame[TARGET]).all() or frame[TARGET].lt(-1).any():
        raise ValueError("Need finite simple returns of at least -100%.")
    earned = frame.assign(earned_month=return_month(frame.date))
    wide = earned.pivot(index="earned_month", columns="stock_id", values=TARGET)
    calendar = pd.date_range(frame.date.min(), frame.date.max(), freq="ME")
    return wide.reindex(calendar)


def _cluster_window(window, n_clusters, random_state):
    """Fit one K-means model on the stocks with a complete, non-constant path."""
    labels = pd.Series(pd.NA, index=window.columns, dtype="Int64")
    stock_momentum = pd.Series(np.nan, index=window.columns, dtype=float)
    basket_momentum = pd.Series(np.nan, index=window.columns, dtype=float)
    complete = window.columns[window.notna().all()]
    if len(complete) == 0:
        return labels, stock_momentum, basket_momentum
    block = window[complete]
    stock_momentum.loc[complete] = (1 + block).prod(axis=0) - 1
    scale = block.std(ddof=0, axis=0)
    usable = scale.index[scale.gt(1e-12)]
    if len(usable) == 0:
        return labels, stock_momentum, basket_momentum
    k = min(int(n_clusters), len(usable))
    centered = block[usable] - block[usable].mean(axis=0)
    if k == 1:
        fitted = pd.Series(0, index=usable, dtype="int64")
    else:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state, algorithm="lloyd")
        fitted = pd.Series(model.fit_predict(np.asarray(centered / scale[usable]).T), index=usable)
    labels.loc[usable] = fitted.to_numpy()
    for basket, members in fitted.groupby(fitted):
        path = block[members.index].mean(axis=1)
        basket_momentum.loc[members.index] = float((1 + path).prod() - 1)
    return labels, stock_momentum, basket_momentum


def assign_correlation_baskets(history, decision_dates, n_clusters=N_CLUSTERS,
                               lookback=LOOKBACK, random_state=RANDOM_STATE, progress=None):
    """Attach the within-month basket feature at each decision snapshot.

    ``decision_dates`` are snapshot dates. A stock is eligible only when all
    ``lookback`` earned months through that snapshot are present and its path
    is not constant. Cluster numbers are meaningful inside that month only.
    """
    if not isinstance(n_clusters, int) or n_clusters < 2:
        raise ValueError("Use at least two correlation baskets.")
    if not isinstance(lookback, int) or lookback < 2:
        raise ValueError("The correlation window must cover at least two months.")
    wide = realized_returns(history)
    dates = pd.DatetimeIndex(pd.to_datetime(decision_dates)).sort_values()
    if dates.empty or not dates.is_unique:
        raise ValueError("Need at least one decision snapshot.")
    frames = []
    for number, date in enumerate(dates):
        date = pd.Timestamp(date)
        if progress is not None and number % 12 == 0:
            progress(f"Correlation baskets at {date:%Y-%m}")
        expected = pd.date_range(end=date, periods=lookback, freq="ME")
        if date not in wide.index:
            raise ValueError(f"{date.date()} is outside the realized-return calendar.")
        labels, stock_momentum, basket_momentum = _cluster_window(
            wide.reindex(expected), n_clusters, random_state)
        frames.append(pd.DataFrame({
            "date": date,
            "stock_id": labels.index.to_numpy(),
            "trailing_momentum": stock_momentum.to_numpy(dtype=float),
            FEATURE: pd.array(labels.to_numpy(), dtype="Int64"),
            "basket_momentum": basket_momentum.to_numpy(dtype=float),
        }))
    assigned = pd.concat(frames, ignore_index=True)
    assigned["eligible"] = assigned[FEATURE].notna()
    return assigned.sort_values(KEYS).reset_index(drop=True)


def attach_short_momentum(history, signals, months):
    """Replace the trading signal with a one-, two-, or three-month return.

    ``signals`` already carries the correlation group. Stock momentum is the
    compound of the last ``months`` earned simple returns. Group momentum is
    the compound of the equal-weight path of that group's current members.
    Neither uses the twelve-month compound that formed the groups.
    """
    if not isinstance(months, int) or months < 1:
        raise ValueError("Momentum must cover at least one earned month.")
    if signals.empty or not set(KEYS + [FEATURE]).issubset(signals.columns):
        raise ValueError("Need basket ids before the momentum window.")
    if signals[KEYS].isna().any().any() or signals.duplicated(KEYS).any():
        raise ValueError("Signal keys must be unique and nonmissing.")
    wide = realized_returns(history)
    pieces = []
    for date, block in signals.groupby("date", sort=True):
        date = pd.Timestamp(date)
        window = wide.reindex(pd.date_range(end=date, periods=months, freq="ME"))
        complete = window.notna().all()
        momentum = pd.Series(np.nan, index=window.columns, dtype=float)
        if bool(complete.any()):
            momentum.loc[complete] = (1 + window.loc[:, complete]).prod(axis=0) - 1
        basket_momentum = pd.Series(np.nan, index=block.stock_id.to_numpy(), dtype=float)
        labeled = block.dropna(subset=[FEATURE])
        for _, members in labeled.groupby(FEATURE, sort=True):
            ids = [stock for stock in members.stock_id if stock in complete.index and bool(complete[stock])]
            if not ids:
                continue
            path = window[ids].mean(axis=1)
            basket_momentum.loc[members.stock_id.to_numpy()] = float((1 + path).prod() - 1)
        frame = block[KEYS + [FEATURE]].copy()
        frame["trailing_momentum"] = frame.stock_id.map(momentum)
        frame["basket_momentum"] = frame.stock_id.map(basket_momentum)
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


def positive_volatility(history, decision_dates, lookback=LOOKBACK, progress=None):
    """Root-mean-square of up months. Down months contribute zero.

    Every month in the window stays in the average, matching the downside
    measure used elsewhere in the project. A complete window of only
    non-positive returns has positive volatility zero. An incomplete window
    is missing. Annualizing would not change the rank, so this stays monthly.
    """
    if not isinstance(lookback, int) or lookback < 1:
        raise ValueError("Positive volatility needs at least one earned month.")
    wide = realized_returns(history)
    dates = pd.DatetimeIndex(pd.to_datetime(decision_dates)).sort_values()
    if dates.empty or not dates.is_unique:
        raise ValueError("Need at least one decision snapshot.")
    frames = []
    for number, date in enumerate(dates):
        date = pd.Timestamp(date)
        if progress is not None and number % 12 == 0:
            progress(f"Positive volatility at {date:%Y-%m}")
        expected = pd.date_range(end=date, periods=lookback, freq="ME")
        if date not in wide.index:
            raise ValueError(f"{date.date()} is outside the realized-return calendar.")
        window = wide.reindex(expected)
        complete = window.notna().all()
        vol = pd.Series(np.nan, index=window.columns, dtype=float)
        if bool(complete.any()):
            ups = np.maximum(window.loc[:, complete].to_numpy(dtype=float), 0.0)
            vol.loc[complete] = np.sqrt(np.mean(ups * ups, axis=0))
        frames.append(pd.DataFrame({
            "date": date,
            "stock_id": vol.index.to_numpy(),
            "positive_volatility": vol.to_numpy(dtype=float),
        }))
    return pd.concat(frames, ignore_index=True)


def select_top_positive_vol(signals, volatility, count=TOP_VOL_COUNT):
    """Keep the highest positive-vol names inside each correlation basket.

    A name with positive volatility of zero is left out whenever the basket
    has any up move. Fewer than ``count`` qualifying names means the basket
    contributes only those names. Ties keep the lower stock id.
    """
    if not isinstance(count, int) or count < 1:
        raise ValueError("Keep at least one name from each basket.")
    if signals.empty or not set(KEYS + [FEATURE]).issubset(signals.columns):
        raise ValueError("Need basket ids before the volatility screen.")
    if volatility.empty or not set(KEYS + ["positive_volatility"]).issubset(volatility.columns):
        raise ValueError("Need a positive-volatility value for each stock-month.")
    merged = signals[KEYS + [FEATURE]].merge(
        volatility[KEYS + ["positive_volatility"]], on=KEYS, how="left", validate="one_to_one")
    labeled = merged.dropna(subset=[FEATURE])
    pieces = []
    for _, members in labeled.groupby(["date", FEATURE], sort=True):
        ranked = members.loc[np.isfinite(members.positive_volatility) & members.positive_volatility.gt(0)]
        ranked = ranked.sort_values(["positive_volatility", "stock_id"], ascending=[False, True]).head(count)
        if not ranked.empty:
            pieces.append(ranked)
    if not pieces:
        return merged.iloc[:0].copy()
    chosen = pd.concat(pieces, ignore_index=True)
    return chosen.sort_values(KEYS).reset_index(drop=True)


def concentrated_basket_weights(history, panel, selected, months):
    """Apply the grouped cash rule to the reduced baskets.

    ``selected`` is the book. Group momentum is recomputed on those names,
    and the invested fraction uses that count as N. Names that missed the
    screen are mapped back at weight zero so the accounting still sees the
    full panel; they do not add a second cash sleeve.
    """
    if selected.empty:
        weights = panel[KEYS].copy()
        weights["weight"] = 0.0
        return weights, selected, selected
    momentum = attach_short_momentum(history, selected[KEYS + [FEATURE]], months)
    if "positive_volatility" in selected.columns:
        momentum = momentum.merge(selected[KEYS + ["positive_volatility"]], on=KEYS, validate="one_to_one")
    _, grouped = _prepare_signals(momentum)
    subset = positive_momentum_cash_weights(grouped, by_basket=True)
    full = panel[KEYS].merge(subset, on=KEYS, how="left")
    full["weight"] = full.weight.fillna(0.0)
    return full, grouped, subset


def positive_momentum_cash_weights(signals, by_basket=False):
    """Give capital only to positive momentum. Leave every other slot in cash.

    With P positive names out of N names present, the invested fraction is P/N.
    Those weights are proportional to trailing momentum. When ``by_basket`` is
    set, P is counted inside each basket and a basket with non-positive
    basket momentum contributes nothing, so its own positive members stay in cash.
    """
    required = KEYS + ["eligible", "trailing_momentum"]
    if by_basket:
        required = required + [FEATURE, "basket_momentum"]
    if signals.empty or not set(required).issubset(signals.columns):
        raise ValueError("Need keyed momentum signals.")
    if signals[KEYS].isna().any().any() or signals.duplicated(KEYS).any():
        raise ValueError("Signal keys must be unique and nonmissing.")
    if not pd.api.types.is_bool_dtype(signals.eligible):
        raise ValueError("Eligibility must be boolean.")
    if signals.eligible.any() and not np.isfinite(signals.loc[signals.eligible, "trailing_momentum"]).all():
        raise ValueError("Eligible stocks need a finite trailing return.")
    if by_basket and signals.eligible.any():
        usable = signals.loc[signals.eligible]
        if usable[FEATURE].isna().any() or not np.isfinite(usable.basket_momentum).all():
            raise ValueError("Eligible baskets need an id and a finite basket return.")
    parts = []
    ordered = signals.sort_values(KEYS)
    for _, block in ordered.groupby("date", sort=True):
        allocation = block[KEYS].copy()
        allocation["weight"] = 0.0
        universe = len(block)
        chosen = block.loc[block.eligible]
        if by_basket:
            for _, members in chosen.groupby(FEATURE, sort=True):
                if members.basket_momentum.nunique() != 1:
                    raise ValueError("Each basket needs one momentum value on a snapshot.")
                if not float(members.basket_momentum.iloc[0]) > 0:
                    continue
                positive = members.loc[members.trailing_momentum.gt(0)]
                if positive.empty:
                    continue
                equity = len(positive) / universe
                allocation.loc[positive.index, "weight"] = equity * positive.trailing_momentum / positive.trailing_momentum.sum()
        else:
            positive = chosen.loc[chosen.trailing_momentum.gt(0)]
            if len(positive):
                equity = len(positive) / universe
                allocation.loc[positive.index, "weight"] = equity * positive.trailing_momentum / positive.trailing_momentum.sum()
        parts.append(allocation)
    weights = pd.concat(parts, ignore_index=True)
    if not np.isfinite(weights.weight).all() or weights.weight.lt(-1e-12).any():
        raise AssertionError("Momentum weights must be finite and nonnegative.")
    totals = weights.groupby("date").weight.sum()
    if totals.gt(1 + 1e-8).any():
        raise AssertionError("Stock exposure cannot exceed total capital.")
    weights["weight"] = weights.weight.clip(lower=0)
    return weights.sort_values(KEYS).reset_index(drop=True)


def _metrics(book, version, momentum_months):
    row = performance(book.simple_return).to_dict()
    row.update(version=version, momentum_months=int(momentum_months),
               avg_one_way_turnover=float(book.turnover_one_way.iloc[1:].mean()),
               avg_cash_weight=float(book.cash_weight.mean()) if "cash_weight" in book else 0.0,
               avg_names=float(book.n_held.mean()))
    return row


def _monthly(book, version, momentum_months):
    frame = book.reset_index()
    keep = frame[["snapshot_date", "return_month", "simple_return", "wealth", "drawdown", "turnover_one_way", "n_held"]].copy()
    keep["version"] = version
    keep["momentum_months"] = int(momentum_months)
    keep["cash_weight"] = frame.cash_weight if "cash_weight" in frame else 0.0
    keep["equity_weight"] = frame.equity_weight if "equity_weight" in frame else 1.0
    return keep


def _basket_table(signals, weights):
    data = signals.merge(weights, on=KEYS, validate="one_to_one")
    universe = data.groupby("date").size().rename("n_universe")
    eligible = data.loc[data.eligible].copy()
    eligible["positive"] = eligible.trailing_momentum.gt(0)
    summary = eligible.groupby(["date", FEATURE]).agg(
        n_stocks=("stock_id", "size"),
        basket_momentum=("basket_momentum", "first"),
        n_positive=("positive", "sum"),
        equity_weight=("weight", "sum")).reset_index()
    summary = summary.merge(universe, on="date", validate="many_to_one")
    summary["sleeve"] = summary.n_stocks / summary.n_universe
    summary["sleeve_cash"] = summary.sleeve - summary.equity_weight
    summary["invested"] = summary.basket_momentum.gt(0) & summary.n_positive.gt(0)
    summary["return_month"] = pd.DatetimeIndex(return_month(summary.date)).to_numpy()
    return summary.sort_values(["date", "basket_momentum"], ascending=[True, False]).reset_index(drop=True)


def _prepare_signals(momentum):
    """Basket membership comes from the correlation fit. The short return is the trade."""
    tradable = np.isfinite(momentum.trailing_momentum)
    plain = momentum.copy()
    plain["eligible"] = tradable
    grouped = momentum.copy()
    grouped["eligible"] = tradable & grouped[FEATURE].notna() & np.isfinite(grouped.basket_momentum)
    return plain, grouped


def create_basket_study(root, n_clusters=N_CLUSTERS, lookback=LOOKBACK,
                        momentum_months=MOMENTUM_MONTHS, validation_history_months=60,
                        random_state=RANDOM_STATE):
    """Score equal weight, each short momentum window, and the reduced baskets.

    The first ``validation_history_months`` snapshots are history only. The
    correlation groups use ``lookback`` earned months. The books then trade
    one-, two-, and three-month momentum. The twelve-month compound is not a
    holding signal. One- and two-month windows also trade the ten highest
    positive-vol names inside each group.
    """
    if os.environ.get("MLFACTOR_READ_ONLY") == "1":
        raise RuntimeError("Computation is disabled during artifact-only rendering.")
    horizons = tuple(momentum_months)
    if not horizons or any(not isinstance(months, int) or months < 1 or months > lookback for months in horizons):
        raise ValueError("Each momentum window must be a positive integer no longer than the correlation window.")
    if not isinstance(validation_history_months, int) or validation_history_months < max(24, lookback):
        raise ValueError("Keep at least 24 months, and a full correlation window, before the first book.")
    root = Path(root)
    train = load_panel("train", root)
    dates = pd.DatetimeIndex(train.date.unique()).sort_values()
    if not dates.to_period("M").equals(pd.period_range(dates.min(), dates.max(), freq="M")):
        raise ValueError("Training snapshots must be consecutive month-ends.")
    if len(dates) - validation_history_months < 2:
        raise ValueError("Need at least two evaluation months after the initial history.")
    decision_dates = dates[validation_history_months:]
    panel = train.loc[train.date.isin(decision_dates), KEYS + [TARGET]].copy()
    assigned = assign_correlation_baskets(train, decision_dates, n_clusters, lookback, random_state)
    groups = panel[KEYS].merge(assigned[KEYS + [FEATURE]], on=KEYS, how="left", validate="one_to_one")
    volatility = positive_volatility(train, decision_dates, lookback)
    selected = select_top_positive_vol(groups, volatility, TOP_VOL_COUNT)
    if not selected.empty and selected.groupby(["date", FEATURE]).size().gt(TOP_VOL_COUNT).any():
        raise AssertionError("The positive-vol screen kept more than the requested names.")
    equal_weight = portfolio_returns(panel)
    if pd.Timestamp(equal_weight.index.max()) >= pd.Timestamp("2010-01-31"):
        raise AssertionError("This comparison must end before the official test period.")
    monthly = [_monthly(equal_weight, "Equal weight", 0)]
    metrics = [_metrics(equal_weight, "Equal weight", 0)]
    basket_tables, assignments = [], []
    for months in horizons:
        momentum = attach_short_momentum(train, groups, months)
        plain, grouped = _prepare_signals(momentum)
        basket_weights = positive_momentum_cash_weights(grouped, by_basket=True)
        momentum_weights = positive_momentum_cash_weights(plain, by_basket=False)
        books = {
            "Momentum + cash": portfolio_with_cash(panel, momentum_weights),
            "Basket momentum + cash": portfolio_with_cash(panel, basket_weights),
        }
        for version, book in books.items():
            if not book.index.equals(equal_weight.index):
                raise AssertionError(f"{months}-month {version} does not share the evaluation months.")
            monthly.append(_monthly(book, version, months))
            metrics.append(_metrics(book, version, months))
        held = grouped.merge(basket_weights, on=KEYS, validate="one_to_one")
        held["momentum_months"] = months
        held["version"] = "Basket momentum + cash"
        if held.loc[held.weight.gt(0) & ~held.eligible].shape[0]:
            raise AssertionError("A stock outside a basket was given capital.")
        if held.loc[held.weight.gt(0) & held.basket_momentum.le(0)].shape[0]:
            raise AssertionError("A non-positive basket was given capital.")
        if held.loc[held.weight.gt(0) & held.trailing_momentum.le(0)].shape[0]:
            raise AssertionError("Non-positive momentum was given capital.")
        basket_tables.append(_basket_table(grouped, basket_weights).assign(
            momentum_months=months, version="Basket momentum + cash"))
        assignments.append(held)
        if months in TOP_VOL_MOMENTUM:
            full_weights, concentrated, subset_weights = concentrated_basket_weights(
                train, panel, selected, months)
            book = portfolio_with_cash(panel, full_weights[KEYS + ["weight"]])
            if not book.index.equals(equal_weight.index):
                raise AssertionError(f"{months}-month {TOP_VOL_VERSION} does not share the evaluation months.")
            monthly.append(_monthly(book, TOP_VOL_VERSION, months))
            metrics.append(_metrics(book, TOP_VOL_VERSION, months))
            if not concentrated.empty:
                screened = concentrated.merge(subset_weights, on=KEYS, validate="one_to_one")
                screened["momentum_months"] = months
                screened["version"] = TOP_VOL_VERSION
                if screened.loc[screened.weight.gt(0) & ~screened.eligible].shape[0]:
                    raise AssertionError("A stock outside a reduced basket was given capital.")
                if screened.loc[screened.weight.gt(0) & screened.basket_momentum.le(0)].shape[0]:
                    raise AssertionError("A non-positive reduced basket was given capital.")
                if screened.loc[screened.weight.gt(0) & screened.trailing_momentum.le(0)].shape[0]:
                    raise AssertionError("Non-positive momentum was given capital in a reduced basket.")
                outside = full_weights.loc[full_weights.weight.gt(0), KEYS].merge(
                    selected[KEYS], on=KEYS, how="left", indicator=True)
                if not outside["_merge"].eq("both").all():
                    raise AssertionError("A name outside the positive-vol screen was given capital.")
                basket_tables.append(_basket_table(concentrated, subset_weights).assign(
                    momentum_months=months, version=TOP_VOL_VERSION))
                assignments.append(screened)
    top_months = [month for month in horizons if month in TOP_VOL_MOMENTUM]
    versions = list(VERSIONS[:3]) + ([TOP_VOL_VERSION] if top_months else [])
    return {
        "monthly": pd.concat(monthly, ignore_index=True),
        "metrics": pd.DataFrame(metrics),
        "baskets": pd.concat(basket_tables, ignore_index=True),
        "assignments": pd.concat(assignments, ignore_index=True),
        "metadata": {
            "protocol": "correlation_basket_momentum",
            "model": "KMeans",
            "feature_name": FEATURE,
            "model_inputs": "z-scored trailing simple returns; one column per month in the correlation window",
            "n_clusters": n_clusters,
            "lookback_months": lookback,
            "correlation_lookback_months": lookback,
            "momentum_months": list(horizons),
            "top_positive_vol_count": TOP_VOL_COUNT,
            "top_positive_vol_months": top_months,
            "positive_volatility": "sqrt(mean(max(earned simple return, 0)**2)) over the correlation window; down months contribute zero",
            "random_state": random_state,
            "validation_history_months": validation_history_months,
            "first_snapshot": str(pd.Timestamp(decision_dates.min()).date()),
            "last_snapshot": str(pd.Timestamp(decision_dates.max()).date()),
            "first_return_month": str(pd.Timestamp(equal_weight.index.min()).date()),
            "last_return_month": str(pd.Timestamp(equal_weight.index.max()).date()),
            "months": int(len(equal_weight.index)),
            "versions": versions,
            "cash_return": 0.0,
            "test_csv_loaded": False,
            "official_test_period": "Jan 2010–Dec 2018, reserved for the final tab",
            "rule": "Groups use the 12-month correlation path. Holding uses 1-, 2-, and 3-month momentum. "
                    "Positive group momentum only; inside those groups, positive stock momentum only. Cash earns zero. "
                    "A second book keeps the 10 highest positive-vol names in each group and applies the same rule at 1 and 2 months. "
                    "That book is scaled to the names that remain.",
        },
    }


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_basket_study(result, root=None):
    base = Path(root) if root is not None else PROJECT_ROOT / "artifacts" / "basket_studies"
    directory = base / ("baskets_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    frames = {}
    for name in ["metrics", "monthly", "baskets", "assignments", "predictive", "fits"]:
        if name not in result:
            continue
        path = directory / f"{name}.parquet"
        result[name].to_parquet(path, index=False)
        frames[name] = _digest(path)
    metadata = dict(result["metadata"])
    metadata.update(created_utc=datetime.now(timezone.utc).isoformat(), frame_sha256=frames,
                    directory=str(directory))
    (directory / "study.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    return directory


def list_basket_studies(root):
    base = Path(root) / "basket_studies"
    if not base.is_dir():
        return []
    found = []
    for path in sorted(base.glob("*/study.json"), reverse=True):
        metadata = json.loads(path.read_text())
        if metadata.get("protocol") == "correlation_basket_momentum":
            found.append({"directory": path.parent, "metadata": metadata})
    return found


def load_basket_study(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "study.json").read_text())
    if metadata.get("protocol") != "correlation_basket_momentum":
        raise ValueError("This directory is not a correlation-basket study.")
    frames = {}
    for name, expected in metadata["frame_sha256"].items():
        path = directory / f"{name}.parquet"
        if not path.is_file() or _digest(path) != expected:
            raise ValueError(f"Saved basket file is missing or damaged: {name}.parquet")
        frames[name] = pd.read_parquet(path)
    return {"metadata": metadata, **frames}
