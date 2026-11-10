"""Read-only explanations and comparisons over saved basket studies.

No grouping, prediction fitting, weight construction or portfolio replay happens
here. Return-path illustrations use the already-saved one-month momentum values.
"""
import pandas as pd

from .basket_momentum import FEATURE


FULL_GROUP = "Basket momentum + cash"


def normalize_study(study):
    months = int(study["metadata"].get("lookback_months", 12))
    result = dict(study)
    for name in ("metrics", "monthly", "baskets", "assignments"):
        if name not in result:
            continue
        frame = result[name]
        additions = {}
        if "momentum_months" not in frame:
            additions["momentum_months"] = months
        if "model_key" not in frame:
            additions["model_key"] = "none"
        result[name] = frame.assign(**additions) if additions else frame
    return result


def require_basket_training(study):
    meta, monthly = study["metadata"], study["monthly"]
    if meta.get("protocol") != "correlation_basket_momentum" or meta.get("test_csv_loaded") is not False:
        raise ValueError("Advanced ML requires recorded training-only basket evidence.")
    if any("test" in name for name in meta.get("source_sha256", {})):
        raise ValueError("A test source cannot enter the training basket view.")
    dates = pd.to_datetime(monthly.return_month)
    if dates.empty or dates.min() < pd.Timestamp("2000-01-01") or dates.max() > pd.Timestamp("2009-12-31"):
        raise ValueError("Basket results must remain inside the 2000–2009 training years.")
    if dates.min() != pd.Timestamp(meta["first_return_month"]) or dates.max() != pd.Timestamp(meta["last_return_month"]):
        raise ValueError("Basket dates differ from the saved evaluation window.")
    if not (pd.to_datetime(monthly.snapshot_date) + pd.offsets.MonthEnd(1)).equals(dates):
        raise ValueError("Basket snapshots must earn returns the following month.")


def detail_frames(study, months, versions=None, model_key="none"):
    """Fetch only the requested inspection partitions in a deployed publication."""
    if "detail_loader" in study:
        return study["detail_loader"](months, versions, model_key)
    result = {}
    for name in ("baskets", "assignments"):
        frame = study[name]
        mask = frame.momentum_months.eq(months) & frame.model_key.isin(["none", model_key])
        if versions is not None:
            mask &= frame.version.isin(versions)
        result[name] = frame.loc[mask].copy()
    return result


def grouping_profiles(assignments, snapshot, lookback=12):
    """Illustrate the current members' standardized paths, without refitting.

    One-month momentum equals a return already earned at its saved snapshot.
    Early evaluation dates lack a full earlier path in this archive. Missing
    histories are excluded explicitly and coverage is returned to the renderer.
    Current-month memberships define the plotted groups; older numeric cluster
    IDs are never joined as though they were persistent sectors.
    """
    snapshot = pd.Timestamp(snapshot)
    frame = assignments.loc[assignments.version.eq(FULL_GROUP) &
                            assignments.momentum_months.eq(1) & assignments.model_key.eq("none")]
    if frame.duplicated(["date", "stock_id"]).any():
        raise ValueError("Expected one saved one-month signal per stock/snapshot.")
    current = frame.loc[pd.to_datetime(frame.date).eq(snapshot)].dropna(subset=[FEATURE])
    expected = pd.date_range(end=snapshot, periods=lookback, freq="ME")
    history = frame.loc[pd.to_datetime(frame.date).isin(expected)]
    wide = history.pivot(index="date", columns="stock_id", values="trailing_momentum").reindex(expected)
    ids = current.stock_id[current.stock_id.isin(wide.columns)]
    block = wide.reindex(columns=ids)
    usable = block.notna().all() & block.std(ddof=0).gt(1e-12)
    block = block.loc[:, usable]
    coverage = {"members": len(current), "illustrated": len(block.columns),
                "first_input_month": expected.min(), "last_input_month": expected.max()}
    if block.empty:
        return pd.DataFrame(), coverage
    standardized = (block - block.mean()) / block.std(ddof=0)
    labels = current.set_index("stock_id")[FEATURE]
    rows = []
    for group, members in labels.groupby(labels, sort=True):
        available = members.index.intersection(standardized.columns)
        if len(available):
            rows.append(pd.DataFrame({"input_month": expected, FEATURE: int(group),
                "mean_standardized_return": standardized[available].mean(axis=1).to_numpy(),
                "covered_members": len(available), "group_members": len(members)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), coverage


def selected_vol_ranks(assignments, snapshot):
    """Rank the saved screened names; never invent scores for excluded names."""
    frame = assignments.loc[pd.to_datetime(assignments.date).eq(pd.Timestamp(snapshot))].copy()
    if frame.empty:
        return frame
    frame = frame.sort_values([FEATURE, "positive_volatility", "stock_id"], ascending=[True, False, True])
    frame["Positive-vol rank in group"] = frame.groupby(FEATURE).cumcount() + 1
    return frame
