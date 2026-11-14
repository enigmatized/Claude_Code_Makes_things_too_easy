"""Volatility-neutral ML and momentum blend, read from saved files. No fitting, no test CSV.

The numbers come from 0_7_Feature_Engineering/03_vol_neutral_momentum_ml.ipynb, which
saved part8_*.csv/json. Inputs were fixed before any result. Blend weights were chosen
on 2005-2009 walk-forward scores. The 2010-2018 window was already viewed in this project,
so it is a sanity check. Annual log growth is the saved performance() identity.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.legacy import PROJECT_ROOT

FOLDER = PROJECT_ROOT / "0_7_Feature_Engineering"
WINDOWS = {"oos_2005_2009": "Walk-forward 2005–2009", "test_2010_2018": "Official test 2010–2018"}
_COLUMNS = (
    ("annual_log_growth", "Annual log growth"),
    ("cagr", "CAGR"),
    ("volatility", "Volatility"),
    ("sharpe", "Sharpe"),
    ("max_drawdown", "Max drawdown"),
    ("alpha_ann", "Bucket alpha / yr"),
    ("alpha_t", "Alpha t"),
)


def _results() -> pd.DataFrame:
    frame = pd.read_csv(FOLDER / "part8_strategy_results.csv")
    if frame.empty or set(frame.window) != set(WINDOWS):
        raise ValueError("part8_strategy_results.csv must hold both saved windows.")
    return frame


def _meta() -> dict:
    return json.loads((FOLDER / "part8_meta.json").read_text())


def vol_neutral_table(window: str = "test_2010_2018") -> pd.DataFrame:
    """One row per strategy: growth, risk and bucket alpha for one saved window."""
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {sorted(WINDOWS)}.")
    frame = _results().loc[lambda d: d.window.eq(window)]
    show = frame[["strategy"] + [source for source, _ in _COLUMNS]].copy()
    show.columns = ["Strategy"] + [header for _, header in _COLUMNS]
    return show.reset_index(drop=True)


def vol_neutral_model_table() -> pd.DataFrame:
    """Per model and label: tuned CV score, walk-forward and test AUC, bucket rank correlation."""
    frame = pd.read_csv(FOLDER / "part8_model_table.csv")
    names = {"logistic": "Logistic regression", "naive_bayes": "Gaussian naive Bayes",
             "random_forest": "Random forest", "grad_boosting": "Gradient boosting"}
    show = pd.DataFrame({
        "Label": frame["label"], "Model": frame["model"].map(names),
        "CV AUC": frame["cv_auc_final"], "Walk-forward AUC": frame["oos_auc"], "Test AUC": frame["test_auc"],
        "Walk-forward bucket IC": frame["oos_bucket_IC"], "Test bucket IC": frame["test_bucket_IC"],
    })
    return show.reset_index(drop=True)


def vol_neutral_figure():
    """Left: the saved blend-weight curve. Right: bucket alpha with a 95% band, both windows."""
    import matplotlib.pyplot as plt

    meta, results = _meta(), _results()
    curve = pd.read_csv(FOLDER / "part8_blend_curve.csv").loc[lambda d: d.other.eq("momentum")]
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    left.plot(curve.w_ml, curve.oos_sharpe, "o-", color="#1d6a6a", label="Walk-forward 2005–2009 (chooses w)")
    left.plot(curve.w_ml, curve.test_sharpe, "s--", color="#b36a2e", label="Official test 2010–2018 (display only)")
    left.axvline(meta["chosen_weights"]["momentum"], color="#68716e", linestyle=":", linewidth=1)
    left.set_xlabel("Weight on the ML rank (1 − weight on momentum)")
    left.set_ylabel("Sharpe, top-20% book")
    left.set_title("Blending ML with 3-month momentum", loc="left", fontsize=10)
    left.legend(fontsize=8, frameon=False)
    picks = results.dropna(subset=["alpha_ann"])
    picks = picks.loc[~picks.strategy.str.contains("volatility \\(w=", regex=True)]
    names = list(dict.fromkeys(picks.strategy))
    for offset, (window, color) in zip((-0.18, 0.18), (("oos_2005_2009", "#1d6a6a"), ("test_2010_2018", "#b36a2e"))):
        part = picks.loc[picks.window.eq(window)].set_index("strategy").reindex(names)
        band = 1.96 * part.alpha_ann / part.alpha_t.replace(0, np.nan)
        right.barh(np.arange(len(names)) + offset, part.alpha_ann, height=0.34, xerr=band.abs(),
                   color=color, alpha=0.85, label=WINDOWS[window], error_kw={"elinewidth": 0.8})
    right.set_yticks(np.arange(len(names)), names, fontsize=8)
    right.axvline(0, color="black", linewidth=0.8)
    right.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    right.set_xlabel("Bucket alpha per year (log return), 95% band")
    right.set_title("Stock-picking skill inside volatility buckets", loc="left", fontsize=10)
    right.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor="none", loc="lower left")
    for axis in (left, right):
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def vol_neutral_note() -> str:
    """Plain-language summary built from the saved numbers, so the text cannot drift from them."""
    meta, results = _meta(), _results()
    test = results.loc[results.window.eq("test_2010_2018")].set_index("strategy")
    walk = results.loc[results.window.eq("oos_2005_2009")].set_index("strategy")
    weight = meta["chosen_weights"]["momentum"]
    blend = next(name for name in test.index if name.startswith("Vol-neutral ML + momentum"))
    neutral = "Vol-neutral ML ensemble"
    vol_rule = "Top volatility quintile (no ML)"
    models = vol_neutral_model_table()
    return (
        f"Ten inputs were fixed before any result: six cluster averages and four technical ranks. "
        f"Stocks are bucketed into {meta['vol_buckets']} volatility groups each month, and the label asks "
        f"whether a stock beats its own bucket. Tuned test AUC runs from {models['Test AUC'].min():.3f} to "
        f"{models['Test AUC'].max():.3f}, so the models still rank stocks only slightly better than chance. "
        f"Inside volatility buckets, the vol-neutral ensemble's top 20% beats its bucket by "
        f"{test.loc[neutral, 'alpha_ann']:.1%} a year on 2010–2018 (t = {test.loc[neutral, 'alpha_t']:.1f}) and "
        f"{walk.loc[neutral, 'alpha_ann']:.1%} a year on the 2005–2009 walk-forward (t = {walk.loc[neutral, 'alpha_t']:.1f}). "
        f"Neither is significant on its own. "
        f"The ML-plus-momentum blend uses weight {weight:.2f} on the ML rank, chosen on 2005–2009. "
        f"On 2010–2018 it has Sharpe {test.loc[blend, 'sharpe']:.2f} and max drawdown {test.loc[blend, 'max_drawdown']:.1%}, "
        f"against {test.loc['Equal weight', 'sharpe']:.2f} and {test.loc['Equal weight', 'max_drawdown']:.1%} for equal weight. "
        f"The volatility-only rule has Sharpe {test.loc[vol_rule, 'sharpe']:.2f} and volatility "
        f"{test.loc[vol_rule, 'volatility']:.1%}. "
        f"The ML-plus-volatility blend chose weight {meta['chosen_weights']['volatility']:.2f}, so it equals the volatility rule. "
        "The 2010–2018 window was already viewed, results are gross of costs, and these are 108 months on one path."
    )
