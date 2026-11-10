# Feature-selection experiment: fixed specification

This specification is implemented by `run_feature_selection.py`; each execution
also writes an immutable `plan.json` before reading its new results. It extends
the runner without changing source CSVs, portfolio rules or the original Part 1.

## Question

Does selecting at most ten original factors improve next-month loss prediction
and the resulting model-adjusted portfolios, compared with all twenty factors?

## Three recipes

| Recipe | Rule |
| --- | --- |
| All 20 | Existing model inputs, with no learned selection. |
| ANOVA top 10 | Rank factors by training-only ANOVA F score against `R1M_Usd < 0`; select ten varying factors, resolving ties by original input order. |
| Correlation-pruned up to 10 | Use the same ranking; greedily skip factors with absolute Pearson correlation >=0.90 with a higher-ranked selected factor, until ten are retained or candidates are exhausted. |

The F score measures one factor's separation between loss and non-loss outcomes.
It is a ranking heuristic, not an independence/significance claim for this panel.
No p-value threshold, selector-size search, correlation-threshold search or
hyperparameter search is performed. Constant training columns are excluded.
The pruned recipe never backfills with rejected duplicates just to reach ten.

Scores, correlations and support masks are learned only inside the sklearn model
pipeline. Selection precedes scaling, so all four models see the same mask for
the same recipe and historical origin. Future GridSearchCV can clone this
pipeline and refit selection inside each inner fold.

## Matched comparison

- Use training data only. The test CSV is not loaded.
- Initial history is 60 snapshots. Evaluate earned returns Jan 2005–Dec 2009.
- Refit on expanding historical data at Dec 2004, 2005, 2006, 2007 and 2008.
  Labels must be realized by that snapshot; its own next-month return is future.
- Use the same four fixed classifiers, thirteen portfolios, adjustment strength
  0.5, market accounting, eligibility and fallback rules for every recipe.
- Keep the same twenty candidate columns and stock-month rows. Only selected
  model inputs change. Original non-ML books must be identical across recipes.
- Save all probabilities, weights, books and feature diagnostics for review;
  the app must not refit while loading or plotting them.

There are 65 cases per recipe, or 195 stored cases across three independent runs.
The study review shows the original once rather than counting it three times.

## What gets recorded

At each fit: training cutoff and row count, candidate and chosen names, actual
selected count, each factor's score/rank, exclusion reason, correlated blocker,
and the full training-only correlation matrix. Names are allowed to change at
later annual fits. The union across validation years may exceed ten even though
each fitted model uses at most ten; a frozen submission fit would select once.

Compare Brier score, log loss, ROC AUC and the historical loss-rate control.
Separately compare CAGR, Sharpe, Sortino, drawdown, turnover and concentration
for the same model and portfolio. Show changes relative to that model's all-20
version, not just relative to a no-ML portfolio.

This is an initial validation comparison. No final feature list or final model
choice is declared from it. The submission release remains pinned to its prior
research draft until the user reviews the evidence.

References: [sklearn univariate feature selection](https://scikit-learn.org/1.4/modules/feature_selection.html),
[ANOVA F scoring](https://scikit-learn.org/1.4/modules/generated/sklearn.feature_selection.f_classif.html),
[leakage and preprocessing](https://scikit-learn.org/1.4/common_pitfalls.html#data-leakage).
