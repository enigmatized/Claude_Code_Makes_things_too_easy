"""Figures and compact tables shared by the Quarto report and slides. Saved results only.

analysis.ipynb draws the correlation map and the risk/return scatter inline so the grader can
read the plotting code. The report and slides call these functions so all three show the
same picture from the same saved files. No fitting, no course CSV.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.legacy import PROJECT_ROOT

OUTPUTS = PROJECT_ROOT / "part0_data_and_benchmark" / "outputs"
GLANCE_NAMES = {"Equal weight": "Equal weight (benchmark)",
                "Top 10 positive-vol": "Top 10 positive-vol (choice, no ML)",
                "Top 10 + downside to cash": "Top 10 + ML downside-to-cash",
                "Full baskets": "Full baskets (no ML)"}


def glance_table() -> pd.DataFrame:
    """Official-test candidates: frozen final books plus the two Section 7 study rows."""
    from research.notebook_views import risk_table, vol_neutral_table
    frozen = risk_table().loc[lambda d: d["Window"].eq("Official test") & d["Book"].isin(GLANCE_NAMES)]
    frozen = frozen.assign(Book=frozen["Book"].map(GLANCE_NAMES), Protocol="Frozen final books")
    study = vol_neutral_table("test_2010_2018").rename(columns={"Strategy": "Book"})
    study = study.loc[study["Book"].str.startswith(("Top volatility", "Vol-neutral ML + momentum"))]
    study = study.assign(Protocol="Section 7 study (top 20%)")
    columns = ["Book", "Protocol", "Annual log growth", "CAGR", "Volatility", "Sharpe", "Max drawdown"]
    return pd.concat([frozen[columns], study[columns]], ignore_index=True)


def correlation_figure(size=(8.5, 7)):
    """Training-data factor correlations, ordered by economic family."""
    corr = pd.read_csv(OUTPUTS / "training_feature_correlations.csv", index_col=0)
    families = pd.read_csv(OUTPUTS / "feature_dictionary.csv").set_index("feature")["family"]
    order = families.sort_values(kind="stable").index
    corr = corr.loc[order, order]
    figure, axis = plt.subplots(figsize=size)
    image = axis.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
    axis.set_yticks(range(len(order)), [f"{name} ({families[name]})" for name in order], fontsize=7)
    figure.colorbar(image, ax=axis, shrink=0.75, label="Pearson correlation, training stock-months")
    axis.set_title("Factor correlations cluster by family", loc="left", fontsize=11)
    figure.tight_layout()
    return figure


def strong_pairs(threshold=0.7) -> tuple[int, int]:
    """(pairs with |correlation| >= threshold, all pairs) in the training correlation matrix."""
    corr = pd.read_csv(OUTPUTS / "training_feature_correlations.csv", index_col=0)
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    return int((upper.abs() >= threshold).sum()), int(len(upper))


def risk_return_figure(size=(11, 5.8)):
    """Official test: volatility against CAGR for every distinct candidate, numbered."""
    from research.notebook_views import prepare_axes, risk_table, vol_neutral_table
    frozen = (risk_table().loc[lambda d: d["Window"].eq("Official test") & ~d["Book"].str.contains("upside tilt")]
              [["Book", "CAGR", "Volatility", "Sharpe"]].assign(Source="Frozen final books"))
    study = (vol_neutral_table("test_2010_2018").rename(columns={"Strategy": "Book"})
             .loc[lambda d: d["Book"].ne("Equal weight") & ~d["Book"].str.startswith("Vol-neutral ML + volatility")]
             [["Book", "CAGR", "Volatility", "Sharpe"]].assign(Source="Section 7 study"))
    points = pd.concat([frozen, study], ignore_index=True)
    figure, axis = plt.subplots(figsize=size)
    colors = plt.get_cmap("tab10").colors
    for n, row in enumerate(points.itertuples(), start=1):
        marker = "o" if row.Source == "Frozen final books" else "s"
        axis.scatter(row.Volatility, row.CAGR, s=90, marker=marker, color=colors[(n - 1) % 10], zorder=3,
                     label=f"{n}. {row.Book} — Sharpe {row.Sharpe:.2f}")
        axis.annotate(str(n), (row.Volatility, row.CAGR), fontsize=8, fontweight="bold",
                      xytext=(6, 4), textcoords="offset points")
    ew = points.set_index("Book").loc["Equal weight"]
    line = np.linspace(0, points.Volatility.max() * 1.05, 50)
    axis.plot(line, line * ew.CAGR / ew.Volatility, color="#68716e", lw=0.8, ls="--",
              label="Same CAGR per unit of volatility as equal weight")
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axis.set_xlabel("Annualized volatility")
    axis.set_ylabel("CAGR")
    axis.set_title("Official test 2010–2018 (circles: frozen final books; squares: Section 7 study)",
                   loc="left", fontsize=10)
    axis.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    prepare_axes(axis)
    figure.tight_layout()
    return figure


def cost_figure(window="test"):
    """CAGR of each book at 0, 10 and 25 bps per dollar traded."""
    from research.notebook_views import cost_table, prepare_axes
    table = cost_table(window).set_index("Book")
    columns = [c for c in table.columns if c.startswith("CAGR @")]
    figure, axis = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / len(columns)
    x = np.arange(len(table))
    shades = ["#1f4e79", "#4f81bd", "#9dc3e6"]
    for i, column in enumerate(columns):
        axis.bar(x + i * width - 0.4 + width / 2, table[column], width, label=column.replace("CAGR @ ", ""),
                 color=shades[i % len(shades)])
    axis.set_xticks(x, [f"{name}\n(turnover {t:.0%}/mo)" for name, t in
                        zip(table.index, table["Avg one-way turnover / month"])], fontsize=8)
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axis.set_ylabel("CAGR after costs")
    axis.legend(title="Cost per dollar traded", fontsize=8, frameon=False)
    axis.set_title("Official test 2010–2018: CAGR after trading costs", loc="left", fontsize=10)
    prepare_axes(axis)
    figure.tight_layout()
    return figure
