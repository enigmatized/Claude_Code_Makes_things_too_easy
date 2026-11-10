# Tuning, final-test scope and GitHub handoff — 2026-11-10

The user requested a Git ignore check, an explanation of unsupervised tuning,
more strategies in Tuned model, and a clearer final comparison before pushing.

## What changed

Tuned model no longer applies the four-portfolio-family focus checkbox. That
filter hid thirteen existing strategies, leaving equal weight. All fourteen
saved basic rules are now available by default, with the existing four classifier
and separate next-up/down choices. The chooser lists rules represented in saved
study metadata; an unsupported model/target combination still shows no result.
It never substitutes another experiment or trains in response to navigation.

The tab explains three distinct objects: the classifier predicts the event;
chronological grid search chooses classifier settings using log loss; the
unchanged portfolio rule consumes those probabilities. A collapsible explanation
covers equal weight as the original reference and the ML-tilted portfolio as a
separate strategy. A two-stock numerical example makes normalization explicit.
The all-strategy comparison now offers CAGR, annual log growth, Sharpe, volatility,
drawdown and turnover, rather than hiding a Sharpe-only table behind the filter.
Existing predictive metric-header tooltips and feature/search audits remain.

Equal weight plus ML is a useful controlled experiment. Start with weights 1/N,
apply the stock-specific signal and account for the resulting portfolio. It is
no longer equal weight. For fully invested rules, downside haircuts renormalize
to 100% stocks; cash-capable rules retain the reduction in cash. The upside
threshold boost preserves the original equity budget. Uniform multipliers cancel
under normalization. None of this promises better returns.

Final test now explicitly says its saved 2010–2018 grouped portfolios use fixed
classifier settings, frozen ANOVA-10 features and K-means. They are **not** the
Tuned model tab's winners. The page adds:

- CAGR alongside annual log growth, including compatibility with older saved
  metrics that record CAGR only (`annual log growth = log1p(CAGR)`).
- Wealth, cumulative log return and monthly log return chart choices, reusing
  `app_charts.return_curve` on saved **portfolio** simple returns.
- Changes for each directional portfolio relative to its own original grouped
  book at the same momentum horizon, including cash differences.
- An explicit note that the entire-path view joins expanding-all-20 training
  forecasts to frozen-ANOVA-10 test forecasts. It is not one fitted protocol.
- Clear distinction between available test results and formal submission-release
  approval that is still pending in local research mode.

The official-test feature expander now describes its scope even when the viewer
chooses training-only or the joined path. All values come from existing saved
frames. No scientific calculation modules, model fitting, search, strategy replay,
published data, source datasets or formal report/release pin were changed.

## Unsupervised tuning is a different experiment

It makes sense to compare cluster counts, history lengths and grouping methods.
An initial declared grid might compare 3/5/8 groups and 6/12-month histories.
Evaluate compactness/separation using silhouette, check sensitivity to input
changes, and check group size/concentration. Numeric group IDs are arbitrary:
never assess stability by comparing raw IDs without accounting for relabeling.

For investment usefulness, apply the same momentum/screen/cash strategy to each
recipe on later **training** months. Assess return, risk and turnover separately.
Do not select grouping settings on the official-test portfolio results. Silhouette
measures geometry; it is not a return forecast or profit objective. K-means inertia
alone decreases as groups proliferate and is not a sufficient selection rule.
See [scikit-learn evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
and [silhouette](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).

Neither clustering hyperparameter search nor grouped-portfolio replay with tuned
classifiers has been run. These are two separate missing experiments. Ward's
training comparison does not become a tuned classifier or an official-test result
by being selected in the app. Both facts are explained in the tuning/final views.

## Existing evidence reused

Three frozen training-holdout studies remain unchanged:

- `holdout_20261109T202045_b0a51718`: downside logistic and histogram boosting.
- `holdout_20261110T012820_d20ce131`: downside naive Bayes and random forest.
- `holdout_20261110T013013_6f2fd4f9`: upside, all four classifiers.

All cover fourteen basic rules and January 2007–December 2009. ANOVA's fixed ten
inputs were selected at the December 2006 fit, using then-known labels. Model
search uses earlier chronological folds; models then stay frozen on the holdout.
The source recipe itself had been reviewed on these years, which remains disclosed.

Final test remains `final_20261110T025420_6d3405bd`; its source and every final
window are unchanged. Default publication manifest SHA-256 remains
`0dc4d709f54634430e060273bfe013f60fdcd5a3f6b424948504d533f3d8c895`.

## Git ignore and pushing

The current `.gitignore` already handles the new grouping work correctly:
raw CSVs, `artifacts/` (including tuning pipelines/database/results), older
publication directories, local credentials, environments and caches are ignored.
`published/project` Parquet/JSON, application code, requirements, docs, notebooks
and QMD sources remain included. Both the old grouping release and backup of the
previous default release are ignored. No ignored research/data/secret files are
already tracked, as checked with Git's own tracked/ignore queries.

No ignore-rule changes or history removal are needed for this update. Keep the
current Parquet files: the website reads those. Update the existing repository
and retain `deploy_app.py` / Python 3.12 on the host. No commit or push is made
by this work. Concurrent other-LLM notebook/grouping edits are preserved.

## Review and next work

Verification details are in
[the saved evidence](verification/tuning_final_review_20261110.json).
Tests check real saved model/target/strategy choices, no-fit deployment, metric
help, growth/log identities and correct original/horizon pairing. Manual browser
checks inspect the new tuning controls and final log chart.

Next: push the reviewed app changes. If extending the research afterward, declare
a training-only grouping search and/or replay tuned forecasts on grouped books,
then explicitly decide which frozen final-evaluation variants to compute. Do
not relabel the current final portfolios as tuned or Ward-based. Formal notebook,
Quarto/HTML, slides and release-signoff work stays separate.
