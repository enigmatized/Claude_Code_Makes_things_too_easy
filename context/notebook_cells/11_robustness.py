"""Robustness of the chosen book, from saved results only. No fitting, no course CSV.

Three questions a reader of the final choice will ask:

1. Did GridSearchCV tuning change the chosen book on 2010–2018?  (pinned engineering study,
   test stage: the same K-means top-10 book with fixed vs. tuned ANOVA-10 overlays)
2. What do trading costs do?  (saved monthly turnover of the frozen final books)
3. Is the extra return over equal weight statistically reliable?  (paired t-test and a
   circular block bootstrap on the saved monthly returns)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.engineering_study import load_engineering_study

BOOK = "K-means top 10"                      # the chosen book's name inside the engineering study
RECIPE = "anova_10"                          # the frozen official fit uses the ANOVA top 10
OVERLAY = {"loss": "Downside to cash", "upside": "Upside tilt"}
COST_BPS = (0, 10, 25)
BOOT_SEED, BOOT_DRAWS, BOOT_BLOCK = 0, 10_000, 6
WINDOW_NAMES = {"test": "Official test 2010–2018", "training": "Training window 2005–2009",
                "entire": "Entire path 2005–2018"}


def _model_label(key):
    from research.notebook_views import model_label
    return model_label(key)


# --------------------------------------------------------------------- 1. tuning on the test
def tuned_final_table() -> pd.DataFrame:
    """Chosen book on 2010–2018: factory (fixed) vs GridSearchCV-tuned overlay, per model and direction."""
    from research.notebook_views import pinned_engineering
    directory = pinned_engineering()
    test = load_engineering_study(directory, "test")
    trials = load_engineering_study(directory, "training")["trials"]
    metrics = test["metrics"].loc[lambda d: d.book.eq(BOOK)]
    original = metrics.loc[metrics.status.eq("Original")].iloc[0]
    rows = [{"Model": "No ML (the chosen book)", "Overlay": "—", "Tuned settings": "—",
             "Factory CAGR": original.cagr, "Tuned CAGR": original.cagr, "Tuned − factory CAGR": 0.0,
             "Factory Sharpe": original.sharpe, "Tuned Sharpe": original.sharpe}]
    part = metrics.loc[metrics.recipe.eq(RECIPE) & metrics.status.isin(["Fixed", "Tuned"])]
    for (model, target), block in part.groupby(["model_key", "target"], sort=True):
        fixed = block.loc[block.status.eq("Fixed")].iloc[0]
        tuned = block.loc[block.status.eq("Tuned")].iloc[0]
        best = trials.loc[trials.recipe.eq(RECIPE) & trials.model_key.eq(model)
                          & trials.target.eq(target) & trials["rank"].eq(1)]
        params = json.loads(best.params.iloc[0]).get("model_params", {}) if len(best) else {}
        rows.append({"Model": _model_label(model), "Overlay": OVERLAY[target],
                     "Tuned settings": ", ".join(f"{k}={v}" for k, v in params.items()) or "—",
                     "Factory CAGR": fixed.cagr, "Tuned CAGR": tuned.cagr,
                     "Tuned − factory CAGR": tuned.cagr - fixed.cagr,
                     "Factory Sharpe": fixed.sharpe, "Tuned Sharpe": tuned.sharpe})
    show = pd.DataFrame(rows)
    show.attrs["directory"] = directory.name
    show.attrs["scope"] = test["scope"]
    return show


def tuned_final_note() -> str:
    table = tuned_final_table()
    overlays = table.iloc[1:]
    gap = overlays["Tuned − factory CAGR"].abs()
    worst = overlays.loc[gap.idxmax()]
    scope = table.attrs["scope"]
    return (
        f"Study {table.attrs['directory']} refit every overlay once on all {scope['fit_rows']:,} training rows "
        f"through {scope['trained_through_return_month']}, with the settings GridSearchCV chose on training "
        f"labels (log loss), and scored {scope['months']} test months. On the chosen book, the largest "
        f"tuned-minus-factory CAGR change is {worst['Tuned − factory CAGR']:+.3%} "
        f"({worst['Model']}, {worst['Overlay'].lower()}); the median is {gap.median():.3%}. "
        "Tuning to a better log loss does not change the portfolio, so the factory-setting official "
        "books stand in for the tuned ones."
    )


# --------------------------------------------------------------------- shared: saved monthly books
def _books(window):
    """Monthly rows of the four books compared in the cost and significance checks."""
    from research.notebook_views import load_final_windows
    _study, _meta, windows = load_final_windows()
    monthly = windows[window][1]
    picks = {"Equal weight": ("Equal weight", "none", 0),
             "Top 10 positive-vol": ("Top 10 positive-vol baskets", "none", 1),
             "Top 10 + downside to cash (logistic)": ("Top 10 + downside to cash", "logistic", 1),
             "Full baskets": ("Basket momentum + cash", "none", 1)}
    out = {}
    for name, (version, model, momentum) in picks.items():
        block = monthly.loc[monthly.version.eq(version) & monthly.model_key.eq(model)
                            & monthly.momentum_months.eq(momentum)].sort_values("return_month")
        if block.empty:
            continue
        out[name] = block.set_index("return_month")
    return out


def _stats(r):
    r = pd.Series(r, dtype=float)
    log_growth = 12 * np.log1p(r).mean()
    vol = r.std(ddof=1) * np.sqrt(12)
    return np.expm1(log_growth), r.mean() * 12 / vol if vol > 0 else np.nan


# --------------------------------------------------------------------- 2. trading costs
def cost_table(window="test", bps=COST_BPS) -> pd.DataFrame:
    """CAGR and Sharpe after charging `bps` per dollar traded each month.

    Saved turnover is one-way and drift-aware, with cash counted as an asset, so the share
    of the portfolio sold equals the share bought. The charge is bps x 2 x one-way turnover:
    a full round trip on every dollar moved. Moves into cash need only the sale, so this
    slightly overstates their cost (conservative). The first month pays for buying in.
    """
    rows = []
    for name, block in _books(window).items():
        turnover = block.turnover_one_way.copy()
        if pd.isna(turnover.iloc[0]):
            turnover.iloc[0] = float(block.equity_weight.iloc[0])     # initial purchase
        row = {"Book": name, "Avg one-way turnover / month": float(block.turnover_one_way.iloc[1:].mean())}
        for cost in bps:
            net = block.simple_return - (cost / 10_000) * 2 * turnover.fillna(0)
            cagr, sharpe = _stats(net)
            row[f"CAGR @ {cost} bps"] = cagr
            row[f"Sharpe @ {cost} bps"] = sharpe
        rows.append(row)
    return pd.DataFrame(rows)


def cost_note(window="test") -> str:
    table = cost_table(window).set_index("Book")
    top, ew = table.loc["Top 10 positive-vol"], table.loc["Equal weight"]
    hi = max(COST_BPS)
    return (
        f"{WINDOW_NAMES[window]}: Top 10 positive-vol trades {top['Avg one-way turnover / month']:.0%} of the "
        f"portfolio each month against {ew['Avg one-way turnover / month']:.0%} for equal weight. At {hi} bps per "
        f"dollar traded its CAGR falls from {top[f'CAGR @ 0 bps']:.1%} to {top[f'CAGR @ {hi} bps']:.1%}, "
        f"and equal weight's from {ew[f'CAGR @ 0 bps']:.1%} to {ew[f'CAGR @ {hi} bps']:.1%}."
    )


# --------------------------------------------------------------------- 3. significance
def _block_bootstrap(diff, a, b, draws=BOOT_DRAWS, block=BOOT_BLOCK, seed=BOOT_SEED):
    """Circular block bootstrap: resample 6-month blocks to keep short-run dependence."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    starts = rng.integers(0, n, size=(draws, int(np.ceil(n / block))))
    index = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(draws, -1)[:, :n] % n
    mean_diff = diff[index].mean(axis=1) * 12
    sa, sb = a[index], b[index]
    sharpe = lambda x: x.mean(axis=1) / x.std(axis=1, ddof=1) * np.sqrt(12)
    return mean_diff, sharpe(sa) - sharpe(sb)


def significance_table(book="Top 10 positive-vol") -> pd.DataFrame:
    """Chosen book minus equal weight, month by month, in each saved window."""
    from scipy import stats
    rows = []
    for window in ("test", "training", "entire"):
        books = _books(window)
        a = books[book].simple_return
        b = books["Equal weight"].simple_return.reindex(a.index)
        diff = (a - b).to_numpy()
        t, p = stats.ttest_1samp(diff, 0.0)
        boot_mean, boot_sharpe = _block_bootstrap(diff, a.to_numpy(), b.to_numpy())
        sa, sb = _stats(a)[1], _stats(b)[1]
        rows.append({
            "Window": WINDOW_NAMES[window], "Months": len(diff),
            "Extra return / yr": diff.mean() * 12,
            "t-stat": t, "p-value": p,
            "95% CI low": np.percentile(boot_mean, 2.5), "95% CI high": np.percentile(boot_mean, 97.5),
            "Share of months ahead": float((diff > 0).mean()),
            "Sharpe difference": sa - sb,
            "Sharpe diff CI low": np.percentile(boot_sharpe, 2.5),
            "Sharpe diff CI high": np.percentile(boot_sharpe, 97.5),
        })
    return pd.DataFrame(rows)


def significance_note() -> str:
    table = significance_table().set_index("Window")
    test = table.loc[WINDOW_NAMES["test"]]
    entire = table.loc[WINDOW_NAMES["entire"]]
    def verdict(row):
        return "is" if row["p-value"] < 0.05 else "is not"
    return (
        f"On the official test the chosen book earns {test['Extra return / yr']:+.1%} a year more than equal weight "
        f"(arithmetic, t = {test['t-stat']:.2f}, p = {test['p-value']:.2f}; block-bootstrap 95% interval "
        f"{test['95% CI low']:+.1%} to {test['95% CI high']:+.1%}). That {verdict(test)} significant at 5%. "
        f"Over 2005–2018 the gap is {entire['Extra return / yr']:+.1%} a year (t = {entire['t-stat']:.2f}, "
        f"p = {entire['p-value']:.2f}). Its Sharpe is lower than equal weight's on the test by "
        f"{-test['Sharpe difference']:.2f}, with a bootstrap interval of {test['Sharpe diff CI low']:+.2f} to "
        f"{test['Sharpe diff CI high']:+.2f}. The higher growth is real in the sample but noisy, and it is "
        "bought with volatility."
    )
