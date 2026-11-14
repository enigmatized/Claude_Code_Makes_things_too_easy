# Change log

## 2026-11-14 — Ignore local LLM assistant files

- Add targeted assistant history, session, cache, local-settings and private
  instruction rules; retain shared instructions, skills and project rules.
- Verify ignored/retained paths with Git and confirm no tracked files match.
  Update the state, staging guide and decision record; no research computation.

## 2026-11-10 — Report and slides rewritten to match the graded notebook

- Rewrite `submission/project_report.qmd` around the notebook's nine sections, with key
  numbers inlined from saved results (no hard-coded values), tuned-on-test, cost and
  significance sections, final choice and limitations.
- Rewrite `submission/presentation.qmd` as 16 code-free slides with a reading guide under
  each chart and speaker notes; checked for overflow with headless-Chrome screenshots.
- Add `context/notebook_cells/12_presentation.py` (glance table, correlation map,
  risk/return scatter, cost chart) exported via `research/notebook_views.py`; one new test.
- Old QMD/HTML archived in `submission/archives/pre_presentation_20261110/`.
  `build_submission.py --final` passes 10/10. No fitting, no `release.json` change.

## 2026-11-10 — Remove app header subtitle

- Remove the long arrow-separated project-stage subtitle from the app header.
  Verified its removal and Python syntax; no model/data changes or new tests.

## 2026-11-10 — Implement basket/model/allocation/target navigation

- Add pure saved-case descriptions and shared compatible selectors across the
  seven tabs; preserve original IDs, recipes, fitting scope and cash policies.
- Default Strategies/Basic ML to one configuration, keeping expanded comparisons
  and select-all charts. Collapse advanced source filters/saved runs.
- Preserve grouping/method lessons; move grouped feature results into the feature-
  effects subtab, with actual model/target/horizon choices.
- Reorder tuning/engineering/final controls; filter earlier final plots to the
  chosen pair and actual momentum horizon. Missing tuples do not trigger fits.
- Add a source/protocol-aware session final comparison with growth/log/drawdown/
  cash paths and a guard against mismatched windows. Fix the momentum allocation
  link to select the matching basket/run.
- Verify 240 tests, source-free seven-tab interactions and browser views. No new
  backtests/fits, export/data/dependency change or formal submission rewrite.
  Updated implementation handoff and evidence; no commit or push performed.

## 2026-11-10 — Basket-first navigation plan only

- Locate the existing fourth Momentum acceleration grouping subtab and distinguish
  local changes from the still-unpushed hosted update.
- Document the proposed basket/model/strategy/up-down flow for all seven tabs,
  with exceptions, saved-case coverage, protocol matching and staged rollout.
- Recommend a Basic ML pilot; separate presentation changes from new basket-
  specific fitting or arbitrary weighting combinations. No app/experiment/release
  changes and no new tests or fits. See `docs/BASKET_FIRST_UI_PLAN.md`.

## 2026-11-10 — Submission robustness: pinned study, tuned test, costs, significance, --final

- Pin the engineering study (`notebook_views.ENGINEERING_STUDY` =
  `engineering_20261110T171255_825ffa99`) instead of "newest"; training metrics are
  identical to `…172206` (max |CAGR diff| 0.0), and the pinned study has the 2010–2018 stage.
- New read-only `context/notebook_cells/11_robustness.py`: tuned-vs-factory overlays on the
  chosen book for 2010–2018 (largest change −0.027% CAGR), cost check at 0/10/25 bps on
  saved turnover (chosen book 19.6% → 15.8% test CAGR at 25 bps; equal weight 13.1% → 12.8%),
  and paired t-test + 6-month block bootstrap vs equal weight (test +9.6%/yr, p = 0.48).
- New `research/submission_checks.py`; `build_submission.py --final` now checks the evidence
  the notebook reads (10/10 pass) and keeps the old catalog-run checks as informational.
- Notebook: Sections 5b, 8b, 8c added (46 cells). `python3 build_submission.py --final` exit 0.
  New `tests/test_robustness_view.py`; 30 focused tests pass.


## 2026-11-10 — Momentum acceleration rules on grouped top-ten portfolios

- Explain equal-weight group signals versus actual momentum-proportional holdings.
- Add a shared causal stock/fixed-current-basket acceleration experiment with four
  original/stock/basket/combined rules; halved allocations move to cash, never shorts.
- Save K-means and Ward training and separately labeled previously viewed final
  evidence; original monthly books match all four existing references exactly.
- Add training/final app sections, metrics and log/growth/drawdown/cash charts,
  current-member basket bars and stock-level signal/allocation audits.
- Publish six small Parquet files; current release 476 files / 173,332,889 bytes,
  preserving 469 earlier frame hashes and the ignored previous default backup.
- Verify 229 tests, saved-only seven-tab AppTest, and real-browser views. No new
  dependencies, classifier fits, tuning or formal report/release change from this
  experiment. Document handoff in `docs/MOMENTUM_ACCELERATION.md`.

## 2026-11-10 — Presentation rewrite of the graded notebook + volatility-neutral ML study

- New `0_7_Feature_Engineering/03_vol_neutral_momentum_ml.ipynb`: volatility-neutral
  label, four GridSearchCV-tuned models, 2005–2009 walk-forward choice of an
  ML/momentum blend weight (0.50), frozen 2010–2018 check (previously viewed).
  Saves `part8_*.csv/json`.
- New read-only reader `context/notebook_cells/10_vol_neutral.py`, registered in
  `research/notebook_views.py`; new `tests/test_vol_neutral_view.py`.
- Rewrite `submission/analysis.ipynb` for presentation (20 → 37 cells): summary and
  rubric map, at-a-glance table, visible equal-weight recomputation, factor-correlation
  heatmap, displayed ANOVA-10 pipeline, declared grids and the exact GridSearchCV
  source lines, compact model pivots, Section 7 study, risk/return scatter, explicit
  choice trade-off and limitations. Still read-only; no fits, no course CSV opened.
- Earlier notebook preserved at `submission/archives/pre_presentation_20261110/`.
  Declared choice (Top 10 positive-vol, no overlay), `release.json`, QMD/HTML unchanged.
  18 focused tests pass.


## 2026-11-10 — Recognizable K-means top-ten strategy in Final test

- Rename the earlier-final subtab to K-means strategies; spell out grouping,
  top-ten positive-vol screening, one-month positive momentum and residual cash.
- Label original table/curve explicitly and retain actual two-month labels when
  showing all horizons. Default engineered final comparisons to K-means top ten.
- Confirm existing training/test CAGR 24.6807% / 19.6133%, test cash 51.5556%.
  Twelve focused tests and source-free AppTest pass; no experiments/publication
  data or formal release changed.

## 2026-11-10 — Feature-engineering notebook to deployed model pipeline

- Review/annotate the source notebook, preserve its original, correct full-precision
  PCA/IC export and missing-IC denominator/overfitting claims for future reruns.
- Add causal technical inputs and training-fitted cluster averages, cluster PCA,
  IC blends and economic composites; compare six recipes with all four classifiers.
- Run chronological classifier/recipe GridSearchCV and a separately frozen final
  phase. Save 96 variants / 582 portfolios per phase, with full local fitted state
  and resumable checksummed research cases; preserve original numerical modules.
- Add Feature engineering after Tuned model, with methods/results/tuning subtabs;
  add separately labeled earlier/new final sections including Ward-based books.
- Publish 470 files / 172,577,499 bytes, preserving all 457 previous frame hashes;
  verify a source-free deploy and raw/object/fitting guards. Ignore raw/model
  artifacts while keeping current published frames and review/report sources.
- Build the saved-results notebook and QMD/HTML supplement. Verify 221 tests,
  exact parity and browser charts. Formal submission pin remains unchanged.
  See `docs/verification/feature_engineering_20261110.json`.

## 2026-11-10 — Clearer tuning strategies and final-test comparisons

- Expose all fourteen saved basic-strategy holdouts by default in Tuned model;
  explain classifier tuning, equal-weight overlays and separate clustering search.
- Compare all saved rules on six metrics; retain model/target/feature/search audits
  and predictive metric-header help. No search or fitting on interaction.
- Add final annual log growth and two log-return chart views, original-relative
  changes, explicit fixed-K-means/untuned-Ward/tuning coverage and joined-path
  protocol notes. Separate available evaluation from pending formal release.
- Verify 214 tests. Git ignore/tracked checks pass with no ignore edits required;
  source datasets, experiments, current publication and formal release unchanged.
  See `docs/verification/tuning_final_review_20261110.json`.

## 2026-11-10 — Grouping With ML and saved Ward comparison

- Rename the third tab and split it into K-means, Hierarchical clustering and
  Adding a direction. Explain input scaling, centroids, merge trees, membership,
  momentum, screening, cash and separate directional forecasts.
- Add Ward clustering using the original strategy pipeline; save actual monthly
  linkage trees and render a shared interactive dendrogram without fitting.
- Save a training-only 49-book Ward study with four reused directional models;
  no new classifier fits, feature search, tuning or test evaluation.
- Extend `published/project` to 169,973,507 bytes / 458 files, keeping all prior
  frame hashes and adding method-specific lazy details. Preserve the old release.
- Verify 210 tests, real-data partitions, standalone deploy with datasets/archive
  absent and fitting/replay blocked, and browser tree/method switching. Update
  handoff/docs. Existing formal submission and other LLM changes are preserved.

## 2026-11-09 — Add GitHub ignore rules

- Add root `.gitignore` for CSV inputs/reports, experiment archives, older
  publication exports, generated reports, environments, caches and local secrets.
- Keep the current `published/project` Parquet/JSON release, theme, source code,
  notebooks/QMD and documentation available to Git. No local data deleted,
  repository initialized, commit created or remote push performed.
- Verify Git's actual matching through a temporary external repository: 18
  excluded and 20 retained examples pass; all 363 hosted-result files are kept.

## 2026-11-09 — Publish Features, Tuning and Final; clarify factor meanings and focus

- Add published readers/exports for registry and advanced feature studies, saved
  tuning trials/holdouts and frozen final results. Precompute final test, training
  and entire-path windows from saved monthly returns; export archived test reports.
- Default deploy root `published/project` now covers all six tabs (114,541,240
  bytes). Earlier publication frame checksums, raw data, experiments and formal
  submission/report/notebook/slides remain unchanged. No production fit/replay.
- Add descriptive feature-bar and recipe/reference curve hovers; explicitly show
  advanced strategy feature comparisons with their original full-group reference.
- Add default four-portfolio-family focus on Features/Tuned/Final with wider
  comparisons opt-in. Classifier algorithms stay available. Missing advanced
  tuning is explicit; focused tuning has equal-weight evidence only.
- Add portable reader, hover, focus, frozen/training boundary and precomputed-window
  tests. Verification: `docs/verification/project_deploy_20261109.json`.
  All 203 tests pass, with exact real-data parity, standalone all-six-tab use
  under failing source/fitting/replay guards, and browser hover/focus/window checks.

## 2026-11-09 — Grouping story before directional overlays

- Refactor Advanced ML into two subtabs: build/explain the grouped strategy,
  then add up/down predictions. Show K-means memberships, counts, standardized
  return-pattern heatmap and members before positive momentum/cash and top-ten
  positive-volatility comparisons. Separate starting-book/model/horizon controls.
- Derive path illustrations from saved one-month signals with explicit coverage;
  no refitting, raw CSV reads, new weights or strategy replay. Preserve actual
  baseline, model probability and weight results; add selected-name vol ranks.
- Add a training-validated, partitioned third-tab publication; default becomes
  `published/advanced_ml` (111,258,117 bytes). First-two-tab frames and older
  releases remain unchanged; last three deployed tabs remain pending.
- Add causality/coverage/ranking/lazy-read/offline/presentation tests and browser
  verification. See `docs/verification/advanced_refactor_20261109.json`.
  198 full-suite passes, then five focused checks after the final selector label;
  exact real-data parity and standalone no-source/no-fit deployment also pass.
  Formal submission, notebooks, Quarto/HTML/slides and research artifacts stay intact.

## 2026-11-09 — Collapsible Basic ML and two-tab published view

- Make probability quality, portfolio results and original-relative changes
  three independent default-collapsed panels for both prediction directions.
  Preserve shared filters and all-strategy chart defaults; avoid nested expanders.
- Add a manifest-backed read-only Basic ML adapter for the existing saved chooser,
  matched pooled/local comparison, diagnostics, charts and allocation inspection.
  Aggregate local/fallback counts; load weights and fit audits on request.
- Export four existing direction/scope runs: four pooled models, per-stock
  logistic only, 196 cases / 11,760 months, training validation only. Deduplicate
  identical Parquet content; no new fits, replays or source CSV reads required.
- Make `published/basic_ml` the default deploy root (56,589,983 bytes), preserving
  the original first-tab publication and pending states for the other four tabs.
- Add exact-parity/offline/lazy-read/empty-state/training-boundary tests. See
  `docs/verification/basic_ml_deploy_20261109.json` for verification evidence.
  All 193 tests passed, plus exact real-data parity, standalone no-CSV/no-catalog
  deployment, and browser panel/direction/scope interaction checks.
  Formal release/report/notebook/slides and research artifacts remain unchanged.

## 2026-11-09 — Publish the training-only Strategies tab

- Add explicit research/deploy modes, a Cloud view-only entry point and an
  immutable results export/reader. Deployment avoids the research store, raw
  CSVs and all unpublished tab readers; local compute controls remain available.
- Publish fourteen non-ML originals, 840 monthly observations, complete main-run
  target weights and baseline momentum-window comparisons: 4,504,572 bytes.
  Export existing saved evidence only; no model fit or strategy replay.
- Reuse return/drawdown/allocation/momentum charts; load allocation files lazily.
  Add training-source/date/timing guards and a prominent training-only notice.
- Add escaped per-strategy hover/focus tooltips to the first results table in
  both layouts, retaining the existing metric data and number formatting.
- 189 full-suite tests passed, then six focused publication tests after the
  final presentation/description refinements. Exact real-data curve/weight/table parity,
  standalone no-CSV/no-catalog deployment and actual browser interaction checks.
  See `docs/verification/strategies_deploy_20261109.json` and `docs/DEPLOYMENT.md`.
- No public deployment, raw-data/archive mutation, research rerun or formal
  submission/report pin change. Next published tab: Basic ML.

## 2026-11-10 — Final test tab can show training years and the full path

- The saved official score was already January 2010–December 2018 only. The
  chart title now says so.
- Which years: official test, the saved January 2005–December 2009 books, or
  those months compounded straight into the test (168 months). No new fit.

## 2026-11-10 — Frozen 2010–2018 score of the four books

- One fit at 2009-12-31 on the 140,739 training rows. ANOVA top 10, factory
  classifiers, loss and upside. Save `final_books/final_20261110T025420_6d3405bd`.
- Equal weight matches the graded benchmark: CAGR 13.096%, volatility 15.63%,
  Sharpe 0.868. Logistic, one month: top 10 CAGR 19.61% and Sharpe 0.591;
  full baskets CAGR 4.01% and Sharpe 0.528.
- The Final test tab opens on these books. The submission pin is unchanged.
  This pass is not run again.

## 2026-11-10 — Feature cuts on the four books

- Replay all-20, ANOVA-10, correlation-pruned-10, and forest-10 forecasts onto
  equal weight, the top-10 momentum book, and the up-or-down overlays.
- New upside forecasts for the three cuts, plus a matched all-20 upside run.
  Training file only. Save `basket_feature_scores/features_20261110T022741_e6f5661f`.
- Logistic, one month: the cuts move CAGR by less than 0.2 percentage points.
  The feature tab opens on these four books. The fourteen registry rules stay below.

## 2026-11-10 — Up-or-down overlay on the full basket book

- Apply the existing downside-to-cash and upside-tilt rules to Basket momentum
  + cash as well as the top-10 book. Version names stay separate.
- Replay saved all-20 forecasts. Zero new classifier fits. Save study
  `baskets_20261110T021017_e6fa43e2`. Earlier studies and the release pin stay.
  The official test file was not opened.
- On January 2005–December 2009, logistic regression, the overlay trims
  drawdown by holding more cash on the downside book and barely moves the
  upside book. The return lift remains the positive-vol screen.

## 2026-11-09 — Choose tuning model, basic strategy and prediction target

- Rework Tuned model into three selectors, then prediction scores and portfolio
  results; load only compatible saved evidence, preserve old downside files.
- Add Brier/log-loss/ROC-AUC header tooltips, one checked historical control,
  before/after settings and explicit unchanged-winner explanation.
- Extend shared tuning holdout/CLI to independently fitted up/down targets using
  existing adjustments. Save missing downside NB/forest and all-four upside
  comparisons on the same frozen ANOVA-10 list / 2007–2009 holdout. Existing
  results and release retained; no official test read or catalog mutation.
- 175 full-suite tests passed. Saved-only checks match eight model/target and
  sixteen portfolio views, original parity, provenance and growth identities.
  Browser pointer hovers verified all three tooltips; actual next-up selection
  and score table were visually reviewed without exceptions.
  See `docs/verification/tuning_controls_20261109.json`.

## 2026-11-09 — Directional basket overlays and clearer tuning purpose

- Explain tuning's target (next-month stock loss), scoring (log loss), fixed
  basic trading rules, frozen ANOVA-10 inputs and 2007–2009 training holdout.
- Add downside-to-cash and separately fitted upside-threshold tilts to top-10
  positive-volatility basket momentum at 1 and 2 months. Four-model selector,
  stock probability/original/adjusted weight audits, portfolio/cash curves.
- Reuse saved directional forecasts with zero new classifier fits; save study
  `baskets_20261109T225254_b6a479dd`. Earlier studies, registered strategies
  and release remain preserved. No official test CSV read. Shared loader is
  available to notebooks/Quarto without fitting.
- Verification: 171 full-suite passes before the final extra provenance test,
  then four focused checks; saved-only real-data four-model/horizon/legacy
  checks and actual browser review. See
  `docs/verification/basket_adjustments_20261109.json`.

## 2026-11-09 — Keep the 10 highest positive-vol names in each basket

- After the correlation groups, a second book keeps the 10 names in each
  group with the highest positive volatility and trades 1- and 2-month momentum.
- Positive volatility is the root-mean-square of up months over the 12-month
  grouping window. Down months contribute zero and stay in the average.
- The smaller book is scaled to the names that remain. The full 1-, 2-, and
  3-month books stay.
- Saved `artifacts/basket_studies/baskets_20261109T223026_c7be47ff`. On
  January 2005–December 2009 the 1-month top-10 book earned CAGR 24.68% and
  Sharpe 0.833, with 53.4% turnover and a 14.9% maximum weight. The 2-month
  top-10 book earned CAGR 16.14% and Sharpe 0.780. No costs. The test file
  was not opened.

## 2026-11-09 — Select the model beside feature portfolio results

- Added **Prediction model for portfolio comparison** directly above the
  portfolio table, with logistic first and all four saved basic models.
- Synchronize it with the shared feature-study selector so scores, calibration,
  inputs, portfolio tables and curves follow the chosen predictor.
- Verified 13 existing feature checks and all four saved model views, including
  exact portfolio values and both selector directions, with computation blocked.
  No research run, report or saved artifact changed.

## 2026-11-09 — Advanced momentum is 1, 2, and 3 months

- The correlation groups still use a 12-month return path. That path is no
  longer the holding signal.
- The Advanced ML tab now scores one-, two-, and three-month momentum, with
  and without the groups. One month is the last earned simple return.
- Saved `artifacts/basket_studies/baskets_20261109T213124_e664776f`. On
  January 2005–December 2009, one-month momentum plus cash earned CAGR 11.65%
  and Sharpe 0.693, with 47.9% monthly turnover. The grouped one-month book
  earned 10.07% and Sharpe 0.709. No costs. The test file was not opened.

## 2026-11-09 — Identify the predictor in feature-result tables

- Renamed the shared selector to **Prediction model to review** with help text.
- Added a **Prediction model** column to predictive and portfolio comparisons;
  the portfolio heading names the selected predictor. Explain the separate role
  of the forest input-selection recipe.
- Display-only change. Verified 13 feature checks and all four saved model
  selections with research computation blocked; no saved result or release change.

## 2026-11-09 — Correlation-basket momentum on the Advanced ML tab

- Added a monthly K-means feature, `correlation_basket`, from z-scored
  trailing 12-month simple returns. Factor ranks and the forward return are
  not inputs. The id is refit every month.
- Positive groups hold positive-momentum members, sized by that momentum.
  Negative groups, negative members, and stocks without the window stay in
  cash at 0%. The fourteen-rule registry is unchanged.
- Saved `artifacts/basket_studies/baskets_20261109T210228_f4ad9b2a` on the
  January 2005–December 2009 training window. Grouped book: CAGR 4.41%,
  Sharpe 0.550, max drawdown −10.3%, average cash 54%. Equal weight on the
  same months: CAGR 7.62%, Sharpe 0.433, drawdown −49.5%. The test file was
  not opened and the run is not in the experiment catalog.
- Browsing the tab does not fit. `python3 run_basket_momentum.py` rebuilds it.

## 2026-11-09 — Forest feature selection implemented across deliverables

- Added a cloneable forest selector, independent serialized settings, stable
  feature identity, tree-variation audit and explicit uninformative fallback.
  Selection stays inside historical training/future tuning folds.
- Saved matched study `features_20261109T202318_8b01fe74`: four recipes, four
  predictors, fourteen rules, 280 cases. Fixed pooled downside, annual fits,
  three-month cash momentum and strength 0.50. No test or tuning in this study.
- Added shared read-only tables, notes and static figures; app importance,
  overlap and calibration views; all four recipe curves; cumulative log view.
- Executed a new review notebook and updated/rendered submission IPYNB,
  QMD/HTML and slides. Added an optional checksummed feature-study release pin;
  retained the main run and archived prior submission sources/HTML.
- Findings: forest selection improves Brier for two of four models; none beats
  the historical control. This does not declare a final feature/model winner.
- Verified 155 research/app checks, exact four-table/70-book/70-weight all-20
  parity, four current cache matches and 2,004 preservation hashes. Read-only
  app review and report builds use saved evidence. New runs: 143.5 MiB, study
  plan-to-completion about 375.5 seconds. See docs/verification evidence.
- Reviewed app/report/slides in the browser, corrected slide overflow and
  result-table spacing, and verified the final-readiness gate still rejects
  pending final research. Both new slides fit their viewport.

## 2026-11-09 — Tuned model tab on a training holdout

- Added the **Tuned model** tab after Feature selection. It compares no ML,
  fixed ML, and tuned ML on one frozen feature list and one strategy at a time.
- The training file is split at December 2006. Search and both fits use labels
  known through that date. The tab scores January 2007–December 2009. The
  2010–2018 file is not opened.
- Saved `artifacts/tuning_studies/holdout_20261109T202045_b0a51718` from the
  ANOVA-10 list at the 2006-12-31 feature fit. Logistic chose C=0.1. Histogram
  boosting chose its factory defaults. Equal-weight Sharpe stayed near 0.23
  for all three versions. The result is not an experiment-catalog run.
- Browsing the tab does not fit. `python3 run_tuning_holdout.py` rebuilds it.

## 2026-11-09 — Plan random-forest feature selection and shared outputs

- Added `docs/RANDOM_FOREST_FEATURE_PLAN.md` with local textbook/sklearn
  references, a fourth feature recipe, importance/error-bar explanations,
  matched training-only comparison and concrete module integration steps.
- Planned shared saved-evidence views for the hosted app, readable code,
  submission notebook, QMD/HTML and slide deck. Documented the current chart's
  three-color truncation and the need for an explicit study release reference.
- Updated state, architecture, submission and feature-study guide links.
- Verification: local Markdown links and cited notebook cell contents checked.
  Documentation only; no fitting, experiments, release changes or rendering.

## 2026-11-09 — Basic ML performance charts default to all strategies

- Select all available strategies by default in Basic ML's Performance through
  time charts, respecting active display filters and subsequent manual choices.
- Preserve other stages' chart defaults; no model fitting or result changes.
- Verified nine app checks and saved-data review with fitting blocked: all 14
  Basic ML strategies start selected, manual changes persist, and Strategies
  still starts with its existing two-curve selection.

## 2026-11-09 — Separate feature-selection process and ML effects

- Split Feature selection into **Selected features & method** and **Effect on
  ML models**, sharing the saved-study/model controls and existing renderers.
- Put the chosen input list first, followed by rankings/exclusion explanations,
  historical selection and training-only correlations. Keep these available
  independently of portfolio filters.
- Show saved independent model-refit provenance before prediction and
  ML-adjusted portfolio comparisons. Label unchanged non-ML reference curves.
- Move tuning status into model effects; distinguish the available search helper
  from the unrun MLFactor panel search. No experiment or fitting was performed.
- Verified 22 focused checks, saved-data mask/refit/metric parity, unsupported
  strategy behavior and both actual browser subtabs. Preserved 2,099 hashes and
  four current exact cache matches; recorded evidence under docs/verification.


## 2026-11-09 — Chronological grid-search function, not yet run

- Added `research/tuning.py`: a feature recipe and one known training slice go
  in; `GridSearchCV` uses expanding month blocks; the winning pipeline is refit
  on that slice. Default grids are logistic `C`, boosting learning rate and
  leaves, forest depth and leaf size, and naive-Bayes smoothing.
- Refused searches above 40 fits, and a 100-tree forest search unless a smaller
  tree count is fixed and recorded or the slow run is requested explicitly.
  Negative log loss is the default score. Portfolio metrics are not a score.
- Saved results live in `artifacts/tuning/<id>/` (`pipeline.joblib`, trial and
  fold tables, `tuning.json` with file hashes). Loading does not fit. These
  directories are not catalog experiment runs.
- Did not open the source CSVs, fit a panel model, change runners, pins, or
  the test evaluation. `tests/test_tuning.py`: 6 passed. See [tuning](docs/TUNING.md).

## 2026-11-09 — Upside Basic ML and pooled/per-stock comparison

- Added strictly-positive next-month forecasts, threshold-based stock boosts and
  original-equity-budget normalization, preserving cash and excluded names.
- Added optional stock-only fits with minimum history/class checks and audited
  same-origin pooled fallback. Reused estimator, selector and accounting code.
- Extended configuration/provenance, cache identity, case IDs, save/load/summary,
  CSV replay and CLI options while keeping legacy loss/pooled runs readable.
- Basic ML now selects target/scope and saved evidence independently from the
  first tab, with matching prediction/calibration/portfolio/allocation views and
  matched pooled-versus-local predictive and portfolio comparison tables.
- Added chronological/isolation/fallback/accounting/cache/storage/UI checks and
  a declared matched-comparison runner. See
  [protocol](0_6_Basic_ML_Directions/README.md) and its recorded evidence.
- Saved four fixed training comparisons (196 total portfolio cases), totaling
  94.3 MiB. Local logistic probability scores are materially worse; report that
  negative finding alongside mixed portfolio effects.
- Verified 138 tests, exact four-model prediction/seventy-book historical parity,
  all four saved-only app views, browser diagnostics/scope comparison, four exact
  cache hits and preservation of 1,570 earlier data/artifact files.


## 2026-11-08 — Log-return views and CAGR

- Expose saved annual log growth beside CAGR in main, momentum, feature and
  archived test comparison tables.
- Add independent wealth/cumulative-log/monthly-log chart selectors to the
  shared strategy comparison renderer; retain earned-month timing and default
  wealth plots. Disable the log wealth axis for log-return data.
- Explain the relationship between growth conventions and preserve portfolio
  calculations, experiments, cache identities and the submission pin.
- Add checks for log-curve dates, compounding/CAGR identities and saved-only
  app interactions. See [return-metric usage](docs/RETURN_METRICS.md).
- Verified 125 passing research/app tests, real saved-data reads with computation
  blocked, both log views in both browser stages, 70 portfolio identities, 203
  preserved files and five exact cache matches. No experiment was computed.

## 2026-11-08 — implement five progressive research tabs

- Replaced the seven-tab composition with training-only Strategies, Basic ML,
  an undefined Advanced ML strategy placeholder, Feature selection and Final test.
- Added immutable case-filtered views; reused grouped metrics, curves,
  probability/calibration diagnostics, allocation tools and saved study readers.
  Preserved source configs/manifests and all scientific calculation behavior.
- Separated training/test state and renderer widget keys. Localized explicit
  training controls; original-only momentum results stay in the first stage.
  Model effects/relative weighting versus cash reductions are explained in Basic ML.
- Added multi-strategy feature review and unavailable-experiment states. Final
  primary results require pinned ready frozen evidence; archives and exploratory
  test selections are opt-in and cannot replace training evidence.
- Verified 121 passing research/app tests, seven final UI checks, real saved-only
  review and all five browser screens. Confirmed exact 14 baseline/70 paired-case
  results, five cache matches and 203 preserved calculation/data/catalog/archive/
  report hashes. Updated handoff and usage docs. No research run, advanced
  implementation, hyperparameter search or release-pin change occurred.

## 2026-11-08 — five-stage app refactor plan and LLM handoff

- Recorded the user's requested sequence: training-only non-ML strategies,
  basic downside-prediction ML and portfolio effects, an undefined advanced
  placeholder, chosen-strategy feature selection, and final test last.
- Added TAB_REFACTOR_HANDOFF with exact renderer reuse, small case-filter/state
  changes, widget-key boundaries, existing cash/normalization semantics, missing
  feature evidence, later tuning, Quarto reuse and future acceptance checks.
- Made the new plan authoritative over earlier navigation/advanced examples;
  updated AGENTS, state, architecture, roadmap, submission and discovery links.
- Documentation only, as requested. No app/backend/data/experiment/catalog/report
  or release change; no tests or experiments were rerun. Verified local Markdown
  links and hashes of 58 code/config/data/catalog/report files.

## 2026-11-08 — grouped strategy results and readable ML changes

- Added six presentation families for fourteen strategies, with unknown-rule
  fallback. Added display filters, collapsible family/strategy result tables and
  a compact sidebar chooser with family selection for the next run.
- Replaced repeated-name ML-change rows in the default layout with per-family
  strategy-by-model grids. Added a metric selector, consistent direction-aware
  shading, percentage-point units, detailed inspection and CSV export. Retained
  full tables and cross-family Select all; archived test results can filter by family.
- Verified 116 passing research/app tests plus the final six grouping checks,
  exact preservation of 280 displayed numeric delta cells, real browser review
  of the saved 70-case comparison, five unchanged exact cache matches and 189
  preserved input/archive/catalog/reference files. Saved a browser preview and
  updated handoff/usage/design docs. No experiment, calculation-code or release
  change was made.

## 2026-11-08 — visible training/test scope and historical test results

- Added actual loaded-period/protocol banners and chart titles. Pending sidebar
  settings cannot relabel the old result; saved choices identify their period.
- Labeled feature and momentum studies independently and separated the saved
  momentum study from the main comparison with a visible boundary.
- Added Test results: thirteen earlier non-ML strategies for 2010–2018, read
  directly from archives after reconciling shared records and filtering training
  rows. Added tables, curves, source hashes and downloads; registered test runs
  can be loaded separately. Expanding/frozen fitting policies remain distinct.
- Clarified that recent studies use training validation and the final selected,
  tuned, frozen test evaluation is pending. No new research test backtest,
  calculation-code change, saved-run migration or report-pin change occurred.
- Verified 168 passing tests, saved-only real app interaction, current exact
  momentum cache matches, old study loading, healthy app and preservation of
  189 input/archive/catalog/reference files. Updated usage and handoff documents.

## 2026-11-08 — configurable cash momentum and four-window study

- Added `cash_momentum_months` to RunConfig, CLI, sidebar, manifests, cases,
  diagnostics, CSV replay and cache identity. Default twelve preserves prior
  behavior; old configurations load as twelve. Earlier thirteen rules keep
  their original signals. Shorter windows genuinely need only their own history.
- Added a declared training-only 2/3/6/12-month study, matched annual-pool and
  prediction checks, immutable run links, checksum-verified saved readers and
  read-only app comparison/model/curve/download/allocation controls.
- Completed `momentum_20261108T211612_b4723b7c`: four ten-case runs with the same
  equal-weight control and four fixed ML overlays. Three months led training
  return/Sharpe; twelve led drawdown/turnover. No automatic final selection,
  test read, feature search or model GridSearchCV.
- Added the `0_5_Momentum_Lookbacks/` guide, findings, exports, chart and evidence;
  updated handoff/design/usage documents. Verified 162 passing tests, saved-only
  real app review, exact prior twelve-month results and preserved source data,
  earlier outputs and report pin.

## 2026-11-08 — annual consistency, positive momentum and cash

- Added a fourteenth non-ML rule: freeze stocks with at least three complete
  historical calendar years and a strict majority positive; buy only positive
  12-month momentum. Invest the positive fraction of available pool slots,
  weighted by momentum, and leave inactive slots in cash. Empty pools can hold
  100% cash. Thresholds were declared, not optimized.
- Added an opt-in cash accounting adapter using the unchanged Part 0 pipeline,
  with zero assumed cash return, drift-aware turnover and explicit cash records.
- Applied all four existing classifiers. For cash-capable rules only, ML weight
  reductions move into cash; existing fully invested behavior remains exact.
- Added cash histories/composition/downloads, average and rebalance cash, annual
  screen explanations, CSV replay support and calculation-code cache hashes.
- Saved training-only 70-case run `20261108T210046_37d1700ebf97` and the new
  `0_4_Consistency_Momentum_Cash/` guide, metrics, screen, chart and evidence.
- Verified 155 distinct passing tests, exact prior four-model probabilities and
  65 portfolio books/weights, real saved app/feature-study reads without fitting,
  healthy app and preservation of 114 original/reference files. Updated handoff
  and architecture documents; submission release and earlier outputs stay intact.

## 2026-11-08 — six-section project and submission design

- Documented the user's six goals and mapped each to existing evidence and gaps.
- Proposed a shared read-only cross-run case catalog, controlled comparisons and
  versioned multi-run presentation release for app/notebook/QMD/slides reuse.
- Specified readable comparison views, tuning artifacts, staged implementation
  and checks for provenance, numerical parity, compatibility and no-fit viewing.
- Updated handoff, architecture, roadmap, submission and usage links. Corrected
  the submission README's outdated feature-study status. This update changes
  documentation only; no runs, numerical code, rendered outputs or release pins
  were changed. Verified links in the changed Markdown documents.

## 2026-11-08 — training-only feature selection and app review

- Added cloneable ANOVA and correlation-pruned selectors inside each historical
  model pipeline, before scaling. Serialized selector settings in FeatureSpec;
  preserved exact no-selection/default logistic behavior.
- Saved actual names/counts, F scores, rankings, exclusion reasons and training
  correlations at every fit. Recorded learned selector code/settings in cache
  identity and checked actual fitted feature counts for submission readiness.
- Added a fixed study runner with a plan written before results, immutable run
  links, shared configuration/code/runtime checks and exact original-book checks.
- Completed study `features_20261108T153403_ddfe05c8`: all 20 vs ANOVA ten vs
  correlation-pruned ten; four fixed classifiers and thirteen portfolios each;
  training-only validation Jan 2005–Dec 2009. No test read or parameter search.
- Both selectors used ten inputs at all five annual fits. Changes in scores and
  portfolios were small and mixed; no model/recipe beat the historical control
  on Brier/log loss. No final subset/model winner was declared.
- Added the saved-results **Feature selection** tab: predictive/portfolio
  comparisons and paths, selection stability, explained audits, correlations,
  and downloads. Browsing does not fit models or selectors.
- Added `0_3_Feature_Selection/` protocol, findings, executed review notebook,
  HTML and exported evidence. Updated handoff, architecture, roadmap and usage
  documents. Preserved the existing submission release pin pending review.
- Verified 143 passing tests, real-data app interaction with computation
  disabled, and read-only notebook execution. Confirmed exact prior all-20
  predictions for all four models and all 65 books/weights; preserved 111
  original files and the recorded input hashes.

## 2026-11-08 — persistence and submission foundations

- Added project handoff instructions and state, architecture, requirements and
  decision documents, with a requirement to update them after each change.
- Verified the assignment's notebook, Quarto HTML, folded-code and presentation
  requirements directly from its source notebook.
- Selected SQLite + Parquet for durable saved results and CSV for exports.
- Implemented complete Parquet persistence, SQLite discovery/metrics, exact cache
  identities, integrity checks, atomic publication, historical CSV migration,
  cache-first app/CLI loading and explicit recomputation.
- Added frozen-at-origin model fits for submission comparisons while preserving
  the expanding research default and exact logistic/portfolio parity.
- Added a pinned readable notebook, Quarto QMD/HTML report and draft slides, plus
  a build command and incomplete-final-evidence gate. Report execution blocks
  shared computation and never fits models or runs searches.
- Pinned the current 65-case validation artifact and benchmark reference hash;
  rendered self-contained HTML with folded report code and hidden slide code.
- Researched NiceGUI, Panel, Reflex and Streamlit. Recommended a later NiceGUI
  prototype; no framework migration was made.
- Verified all 131 automated checks; manually verified real saved-result UI
  startup/loading with model computation disabled. Source data and original
  experiment outputs were preserved. Feature selection/tuning remain next.

## Earlier work — 2026-11-07

- Established data validation, equal-weight accounting and thirteen portfolio rules.
- Added a local Streamlit viewer with paired model charts and allocation histories.
- Extracted shared data preparation; added structured explanations, model and
  feature configuration, and four fixed classifiers.
- Exact logistic parity: 71,631 probabilities, 26 books and 26 weight histories.
  Saved a 65-case training-validation comparison; 119 checks passed.
- Automatic feature selection, tuning and cross-model ensembles were left for
  later experiments. The original Part 1 and earlier outputs were preserved.
