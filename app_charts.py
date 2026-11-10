"""Presentation helpers for paired portfolios and monthly allocation histories."""

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots


VARIANTS = {"baseline": "Original", "loss_adjusted": "With ML"}
RETURN_MEASURES = ("Growth of $1", "Cumulative log return", "Monthly log return")


def return_curve(book, measure="Growth of $1"):
    """Read saved portfolio returns, preserving earned-month timing.

    Wealth and cumulative logs include the initial funding snapshot. Monthly
    logs contain only earned returns, with no invented initial observation.
    Stock log returns must never be averaged to construct these series.
    """
    if measure == "Monthly log return":
        return book.log_return.copy()
    if measure == "Growth of $1":
        values, initial = book.wealth, 1.0
    elif measure == "Cumulative log return":
        values, initial = book.log_return.cumsum(), 0.0
    else:
        raise ValueError(f"Unknown return measure: {measure}")
    return pd.concat([pd.Series([initial], index=[book.snapshot_date.iloc[0]]), values])


def line_dash(variant, variants):
    if variant == "baseline":
        return "solid"
    adjusted = [key for key in variants if key != "baseline"]
    return ["dash", "dot", "dashdot", "longdash"][adjusted.index(variant) % 4]


def case_rows(result, strategy=None, model_ids=None):
    if hasattr(result, "cases") and not result.cases.empty:
        rows = result.cases
        if strategy is not None:
            rows = rows.loc[rows.strategy.eq(strategy)]
        if model_ids is not None:
            rows = rows.loc[rows.variant.eq("baseline") | rows.model_id.isin(model_ids)]
        return rows
    # Small legacy chart fixtures and previously saved two-variant results.
    rows = []
    for key in result.books:
        name, variant = key.split("::")
        if strategy is None or strategy == name:
            rows.append({"case_id": key, "strategy": name, "variant": variant,
                         "model_id": "none" if variant == "baseline" else "loss_adjusted",
                         "model_label": VARIANTS[variant], "label": name})
    return pd.DataFrame(rows)


def case_label(result, key):
    row = case_rows(result).set_index("case_id").loc[key]
    return f"{row.label} · {row.model_label}"


def paired_portfolio_keys(result, strategies, model_ids=None):
    """Choosing a strategy includes all of its available variants."""
    return [key for strategy in strategies for key in case_rows(result, strategy, model_ids).case_id]


def allocation_matrices(result, strategy, model_ids=None):
    """Return matching date × asset grids, including zero when an asset exits.

    Dates are allocation snapshots, not earned-return dates. Only assets held at
    least once are included. Ordering is shared across variants and based on
    average original weight. Missing grid cells mean no allocation, not missing
    returns; the underlying target weights are never interpolated.
    """
    matrices = {}
    for _, case in case_rows(result, strategy, model_ids).iterrows():
        if case.case_id not in result.weights:
            continue
        variant = "baseline" if case.variant == "baseline" else case.model_id
        matrices[variant] = result.weights[case.case_id].pivot(index="date", columns="stock_id", values="weight").fillna(0)
        if case.get("allows_cash", False):
            matrices[variant]["Cash"] = (1 - matrices[variant].sum(axis=1)).clip(lower=0)
    original = matrices["baseline"]
    assets = original.columns
    dates = original.index
    for matrix in matrices.values():
        assets = assets.union(matrix.columns, sort=False)
        dates = dates.union(matrix.index)
    matrices = {key: matrix.reindex(index=dates, columns=assets, fill_value=0).fillna(0)
                for key, matrix in matrices.items()}
    held = pd.concat([matrix.gt(0).any() for matrix in matrices.values()], axis=1).any(axis=1)
    order = matrices["baseline"].mean().loc[held].sort_values(ascending=False, kind="stable").index
    return {key: matrix.loc[:, order] for key, matrix in matrices.items()}


def allocation_heatmap(matrices, assets=None, labels=None):
    assets = list(matrices["baseline"].columns if assets is None else assets)
    count = len(matrices)
    labels = labels or VARIANTS
    figure = make_subplots(rows=count, cols=1, shared_xaxes=True, shared_yaxes=True,
                           vertical_spacing=min(0.12, 0.5 / max(1, count - 1)), subplot_titles=[labels.get(key, key) for key in matrices])
    maximum = max(float(matrix[assets].to_numpy().max()) for matrix in matrices.values())
    for row, (variant, matrix) in enumerate(matrices.items(), 1):
        figure.add_trace(go.Heatmap(
            x=matrix.index, y=[str(asset) for asset in assets], z=matrix[assets].to_numpy().T,
            coloraxis="coloraxis", name=labels.get(variant, variant),
            hovertemplate="Snapshot %{x|%b %Y}<br>Stock %{y}<br>Weight %{z:.2%}<extra>%{fullData.name}</extra>",
        ), row=row, col=1)
        figure.update_yaxes(type="category", categoryorder="array", categoryarray=[str(asset) for asset in assets],
                            autorange="reversed", title_text="Stock ID", row=row, col=1)
    height_per_panel = min(550, max(260, len(assets) * 16))
    figure.update_layout(
        height=height_per_panel * count + 120, template="plotly_white", margin=dict(l=10, r=10, t=45, b=10),
        coloraxis=dict(cmin=0, cmax=maximum or 1, colorscale=[[0, "#f4f7f5"], [0.3, "#9bc8b1"], [1, "#155a46"]],
                       colorbar=dict(title="Weight", tickformat=".1%")),
    )
    figure.update_xaxes(title_text="Rebalance snapshot", row=count, col=1)
    return figure


def composition_frames(matrices, assets=None, top_n=12):
    """Keep a fixed asset list through time; aggregate the remainder exactly."""
    assets = list(matrices["baseline"].columns[:top_n] if assets is None else assets)
    if "Cash" in matrices["baseline"].columns and "Cash" not in assets:
        assets.append("Cash")
    frames = {}
    other_count = len(matrices["baseline"].columns) - len(assets)
    for variant, matrix in matrices.items():
        frame = matrix[assets].rename(columns=lambda asset: "Cash" if asset == "Cash" else f"Stock {asset}").copy()
        if other_count:
            # Sum the omitted positions instead of renormalizing selected assets.
            frame[f"Other assets ({other_count})"] = matrix.drop(columns=assets).sum(axis=1)
        frames[variant] = frame
    return frames


def allocation_composition(matrices, assets=None, labels=None):
    labels = labels or VARIANTS
    frames = composition_frames(matrices, assets)
    figure = make_subplots(rows=len(frames), cols=1, shared_xaxes=True, shared_yaxes=True,
                           vertical_spacing=min(0.14, 0.5 / max(1, len(frames) - 1)), subplot_titles=[labels.get(key, key) for key in frames])
    for row, (variant, frame) in enumerate(frames.items(), 1):
        for index, asset in enumerate(frame.columns):
            color = "#c5ccca" if asset.startswith("Other assets") else qualitative.Dark24[index % 24]
            figure.add_trace(go.Scatter(
                x=frame.index, y=frame[asset], name=asset, legendgroup=asset, showlegend=row == 1,
                mode="lines", line=dict(color=color, width=0.5, shape="hv"), stackgroup=f"weights_{row}",
                hovertemplate=f"{asset}<br>Snapshot %{{x|%b %Y}}<br>Weight %{{y:.2%}}<extra>{labels.get(variant, variant)}</extra>",
            ), row=row, col=1)
        figure.update_yaxes(title_text="Portfolio weight", tickformat=".0%", range=[0, 1], row=row, col=1)
    figure.update_layout(height=340 * len(frames) + 130, template="plotly_white", hovermode="x unified",
                         margin=dict(l=10, r=10, t=45, b=10), legend=dict(orientation="h", y=-0.15))
    figure.update_xaxes(title_text="Rebalance snapshot", row=len(frames), col=1)
    return figure


def asset_weight_lines(matrices, assets, labels=None):
    labels = labels or VARIANTS
    figure = go.Figure()
    for index, asset in enumerate(assets):
        for variant, matrix in matrices.items():
            figure.add_trace(go.Scatter(
                x=matrix.index, y=matrix[asset], name=f"{'Cash' if asset == 'Cash' else f'Stock {asset}'} · {labels.get(variant, variant)}",
                mode="lines", line=dict(color=qualitative.Dark24[index % 24], width=2,
                                         dash=line_dash(variant, matrices), shape="hv"),
                hovertemplate="%{x|%b %Y}<br>Weight %{y:.2%}<extra>%{fullData.name}</extra>",
            ))
    figure.update_layout(template="plotly_white", height=430, hovermode="x unified",
                         xaxis_title="Rebalance snapshot", yaxis=dict(title="Target weight", tickformat=".1%"),
                         margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.2))
    return figure


def allocation_history_csv(matrices):
    """Export complete histories, independently of chart asset filters."""
    frames = []
    for variant, matrix in matrices.items():
        frames.append(matrix.rename_axis(columns="stock_id").stack(future_stack=True).rename("weight").reset_index().assign(variant=variant))
    return pd.concat(frames, ignore_index=True).to_csv(index=False)
