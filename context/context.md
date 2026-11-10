# Presentation flow

Written 2026-11-09 before any multi-tab rebuild. This is the working plan.
Living run notes stay in `docs/PROJECT_STATE.md`. If they disagree on a saved
run id, trust the living notes.

## The request

Show the app in this order.

1. Strategies with no model. The first tab is a lot of toying around. Keep it.
2. Basic ML. A simple model that predicts up or down, laid on a weighting rule.
   This tab needs software work: a strategy is a class with a weighting
   algorithm, and that algorithm can take an up-or-down model on top.
3. The favorite. Cluster stocks, keep the high positive-vol names, and weight
   the positive short-term momentum. That cluster is a feature made by a model.
   The same up-or-down model can sit on top of those weights later.
4. Feature selection, shown first on the four main books. All the other
   models stay available, one click away.
5. Hyperparameter tuning of those four books, with the selected features and
   without them.
6. Final test. The same four books first. Every other model and strategy
   mixture behind a click.

The four books are:

1. Equal weight.
2. High positive-vol names, short-term momentum weights.
3. Book 2, plus a simple up-or-down model adjusting the weights.
4. The correlation-basket book, plus that same up-or-down adjustment.

Keep the current research. Put the extra strategies and models below the four,
not in front of them. Another strategy can be added later. The notebook, the
Quarto report, and hosting come after this flow is right. Fancy features come
after the documents can be rebuilt from a saved result.

Do not open the 2010–2018 file to choose the recipe. The January 2007–December
2009 holdout decides. The official decade is one frozen run of the recipe that
remains.

## Where the work stands

| Step | Tab | Saved today | Blocked on |
| --- | --- | --- | --- |
| 1 | Strategies | Fourteen rules, no model | Nothing. Leave the calculations. |
| 2 | Basic ML | Loss and upside adjustments on the fourteen rules. The shared adjuster is `research/adjustments.py` | Nothing. Leave the registry at fourteen rules |
| 3 | Advanced ML | K-means groups, 1/2/3-month momentum, top-10 book, and the up-or-down overlay on the top-10 book and the full basket book. Training window January 2005–December 2009. Study `baskets_20261110T021017_e6fa43e2` | Nothing on this tab |
| 4 | Feature selection | Four books rescored on all-20, ANOVA-10, pruned-10, and forest-10. Score `features_20261110T022741_e6f5661f`. The fourteen rules remain below | Nothing on this tab. Tuning has not been rerun on the four books |
| 5 | Tuned model | Logistic and boosting grid search, ANOVA-10, holdout January 2007–December 2009, fourteen rules | The same search shown on the four books, with and without the feature cut |
| 6 | Final test | Saved once: `artifacts/final_books/final_20261110T025420_6d3405bd`. Equal weight matches the graded 13.096%. The page shows the four books | The submission pin stays the old expanding run. This pass is not run again |

Book 2 is the Advanced ML top-10 positive-vol book. It is outside the
fourteen-rule registry. Books 3 and 4 are that same up-or-down weight
adjustment on book 2 and on the full basket book. The training-window copies
stay in Advanced ML. The 2010–2018 copies are the saved final-book pass, and
they are still not registry cases.

## Order for anyone editing a tab

Work one tab at a time. Stop at the first missing piece from the previous tab.

1. Tab 1 stays. Display work only, and only if the four-book story needs a
   quieter default. Do not delete the other rules.
2. Tab 2 may introduce the strategy class and the optional up-or-down overlay.
   Stop there. Do not refit the official test.
3. Tab 3 may call that overlay on the basket book and on the top-10 book.
   Stop there. Judge it on the training window already used by that tab.
4. Feature selection may then rescore those four books. Stop before tuning.
5. Tuning may then search logistic regression and histogram boosting for those
   four, with the frozen feature list and with all 20. Score log loss through
   December 2006. Read portfolios on January 2007–December 2009.
6. Final test runs once, frozen, after that choice is written down. The page
   shows the four books, with every other saved mixture behind one control.

An agent on a later tab does not invent the missing overlay, does not open
`mlfactor_long_only_test.csv`, and does not delete `artifacts/`.

## Final tab recipe

Locked 2026-11-10 before `mlfactor_long_only_test.csv` is opened for this pass.
The January 2007–December 2009 holdout and the January 2005–December 2009
feature rescore already happened. Feature cuts moved logistic one-month CAGR
by less than 0.2 percentage points. Tuning was not rerun on these four books,
so the classifiers stay at the factory settings used on Advanced ML.

1. Equal weight of the test cross-section. Cash earns nothing. After the run,
   this book's CAGR is checked against the graded 13.096% benchmark. That
   check does not edit the other books.
2. Five K-means baskets on the trailing 12-month earned simple-return path
   through the decision date. The top-10 book keeps the highest positive-vol
   names in each basket, at most ten, then 1- and 2-month positive momentum
   with residual cash. Positive volatility is the root-mean-square of up
   months on that 12-month path. The full basket book uses the same cash rule
   at 1, 2, and 3 months.
3. ANOVA top 10, chosen inside the single fit at the first test snapshot,
   2009-12-31, from labels whose outcomes are known by that date. Loss and
   upside may keep different columns. The list is recorded from that fit.
4. Four factory classifiers, each fit once, pooled, at that same snapshot,
   for a loss event and for an upside event: logistic regression (C=1),
   Gaussian naive Bayes, random forest, and histogram gradient boosting.
   Fit policy `frozen_at_origin`. The ANOVA logistic search that chose C=0.1
   stays on the fourteen registry rules.
5. The training overlay, unchanged: strength 0.5. Downside multiplies weight
   by (1 − 0.5 × probability of loss) and leaves the haircut in cash. Upside
   multiplies weight by (1 + 0.5) when the probability of an up move is at
   least 0.6, then rescales to the original equity budget. The two directions
   stay separate.
6. If the 2010–2018 top-10 book does not keep the return lift it had in
   2005–2009, that lift was a training-window result. The probability overlay
   is not what raised return on the training window. This page keeps the
   recipe after the decade is scored.

The page shows equal weight, the top-10 book, that book with each direction,
and the basket book with each direction. Every other saved mixture stays
behind the existing control. Browsing reads the saved file. The submission
release stays unpinned, so the pending release sentence stays on the page.

Saved pass: `artifacts/final_books/final_20261110T025420_6d3405bd`. One fit at
2009-12-31, 140,739 training rows, ANOVA top 10, eight factory fits. Equal
weight CAGR 13.096%, volatility 15.63%, Sharpe 0.868. Logistic, one month:
top 10 CAGR 19.61% and Sharpe 0.591; full baskets CAGR 4.01% and Sharpe 0.528.
The top-10 CAGR lift over equal weight is smaller than in 2005–2009, and its
Sharpe is now below equal weight. The recipe above stays as written.

## Later, not this pass

- Notebook, Quarto report, slides, and a read-only host.
- New hand-made features, a block bootstrap around the holdout Sharpe, and a
  high-vol down-weight. Each one is scored on the holdout before anyone
  considers a new official run.
