# Architecture

All-six-tab publication adds `published_analysis.py` and optional schema-1
`features`, `tuning`, `final` sections. Explicit export reads completed artifacts,
serializes feature summaries/audits and method notes, tuning frames/trials, and
frozen final frames. It precomputes final test/training/entire-path tables with
the existing saved-monthly helper; deployment only reads those windows. Frozen
fit/date and training-only guards run before export and again on reads. Earlier
archived test reports become Parquet; deployed CSV/catalog reads are unnecessary.

Existing feature, advanced-feature, tuning and final renderers accept optional
published readers; local research keeps its original data-source branches.
Frame/token caching is publication-specific. `portfolio_focus.py` owns a display
checkbox on Features/Tuning/Final, not an experiment config. It hides extra
registry comparisons by default, while retaining input/model diagnostics and
truthfully showing only available tuning coverage. Graph hover payloads use the
existing feature dictionary; no accounting, selection or prediction math moved.

Advanced ML's shared renderer now composes two subtabs: grouped-strategy teaching
and directional adjustments. `_render_grouped_strategy` selects only original
books; `_render_prediction_adjustments` selects one model, one starting book and
its own horizon. Independent widget keys isolate these choices. Membership/stock
inspection uses the same saved weights/probabilities as before.

`research/basket_reporting.py` supplies normalization for old studies, training
scope validation, partition-aware detail reads and path illustrations. It derives
already-earned one-month returns from saved stock momentum and uses current
memberships to display standardized historical patterns with coverage. It never
calls K-means, classifiers, accounting or source-data loaders. Calculation cache
identities and saved studies are unchanged.

Schema-1 optional `tabs.advanced_ml` declares original study provenance, summary
paths and model/horizon/version detail partitions. `PublishedBasketStudy` passes
a detail loader to the same renderer, avoiding source CSVs/catalog access and
eager allocation loading. The first two publication sections remain exact;
old releases stay readable and absent sections show pending states.

Basic ML now uses the same renderer in research and deploy modes. Prediction,
portfolio and original-relative change sections are sibling `st.expander`s,
collapsed by default. Nested content uses containers/optional details because
Streamlit 1.32 forbids nested expanders. `render_comparison` returns its active
family's strategies so the separate change panel preserves the same filter.
Other callers retain their existing inline paired results and grouping.

Publication schema 1 adds optional `tabs.basic_ml.runs`. `PublishedBasicStore`
provides read-only list/config/load/summary operations over those explicit
source IDs, reusing `saved_basic_source` and `scope_comparison` without SQLite,
fingerprinting, source data or fitting. Published `PredictionBatch` objects hold
only actual saved score/calibration tables; full predictions are omitted and
never synthesized. Target-weight files and fit audits load on request. The
shared reader validates training-only scope for model cases as well as originals.
Explicit publication tokens isolate UI caches from research sessions and older
releases; absent combinations do not fall back to a research-mode result.
Export deduplicates identical serialized frames by checksum. No scientific
calculation identity or existing experiment format changes. See [deployment](DEPLOYMENT.md).

Incremental publication boundary: `publish_results.py` → `published_results.py`
exports baseline-only first-tab results from explicit saved run/study IDs into
a version-1 JSON manifest and checksummed compressed Parquet. `PublishedStrategies`
reads summaries and lazy allocation frames without source data, catalog access,
fingerprint lookup, replay or fitting. The original experiment config/manifest
remain recorded provenance; filtered cases are not relabeled computations.
`app_mode.py` declares research (default) or deploy; `deploy_app.py` forces deploy
and the existing computation guard. Deployment branches before constructing a
research store and does not execute unpublished tab renderers. Local research
continues to use the original store and explicit compute pathways.

The existing `render_strategies`, wealth/return curves, allocation renderers and
momentum figure are shared across both modes. `require_training` validates the
first-tab data boundary in exports, readers and local views. `strategy_help.py`
prepares contextual explanations outside calculation cache identity; baseline
tables use escaped HTML to support per-row hover/focus, retaining `metrics_table`
data and formats. No calculation module or scientific cache identity changed.
See [deployment schema, commands and next-tab handoff](DEPLOYMENT.md).

Tuning now carries `prediction_target` and `upside_threshold` through the same
`create_tuning_holdout` → `tune_classifier` / frozen runner pathway. New study
metadata includes source/code hashes and fixed parameters; historical absent
targets mean loss. `run_tuning_holdout.py` exposes target, model keys, independent
feature-source model and the existing slow-forest opt-in. `tuning_ui.py` selects
model/strategy/target first, filters compatible saved studies, and displays only
saved scores/books. Column help uses Streamlit NumberColumn metadata; the
historical control is collapsed only after verifying score/row-count equality.
No portfolio/prediction math is copied into the UI. These readers remain reusable
by notebooks and Quarto; no release/schema migration or renderer-side fitting.
See [tuning](TUNING.md).

`research/basket_adjustments.py` composes saved directional predictions with
saved positive-volatility basket momentum weights using existing adjustments
and cash accounting. It validates input identity, exact keys, historical fit
cutoffs and common schedules, and reproduces original returns before replay.
It fits no estimator and does not change the fourteen-rule runner/cache identity.
`run_basket_adjustments.py` explicitly computes the new immutable study;
`basket_ui.py` only loads and filters it. Optional predictive/fit frames extend
the basket save/load format while preserving historical studies. Shared loader
supports notebook/Quarto reuse. Downside releases capital to cash; separate
upside preserves the baseline stock budget. This is distinct from ANOVA holdout
tuning. See [protocol and saved sources](BASKET_MOMENTUM.md).

The [forest feature-selection extension](RANDOM_FOREST_FEATURE_PLAN.md) is
implemented. `FeatureSpec` serializes independent forest selector settings;
`TrainingFeatureSelector` fits it inside each historical classifier pipeline,
records importance means/tree SD and handles uninformative rankings explicitly.
The store round-trips new and historical configurations. Calculation identities
changed; historical runs remain readable without silently becoming current hits.

`create_feature_study(..., recipes=...)` accepts a declared recipe mapping; its
default preserves the earlier three-recipe API. `run_feature_selection.py`
explicitly requests four recipes and fourteen rules. Study comparisons validate
all model calendars as well as exact original returns. The summary reader also
loads saved calibration bins. `research/feature_reporting.py` prepares shared
audit/overlap/prediction/portfolio tables, notes and static figures without UI
dependencies or fitting. App Plotly charts consume these same prepared data.
Stable recipe colors replace the old three-color zip that omitted a fourth.

`submission/release.json` retains its main run and can add a `feature_study`
reference containing study ID and metadata checksum. `load_release` validates
both that reference and the linked saved runs, retaining old-release support.
`build_submission.py --pin-feature-study ID` explicitly updates that reference.
The new study is pinned into notebook/report/slides; rendering remains read-only.
The older implementation description below records the preceding foundation.

The separately added **Tuned model** tab sits between Feature selection and
Final test and reads its own saved training holdout. The original five-stage
description below predates that addition; it does not require removing the tab.

The implemented five-stage app follows the
[tab refactor handoff](TAB_REFACTOR_HANDOFF.md): training-only originals, basic
ML, an undefined advanced placeholder, feature selection, and final test.
`result_views.py` filters existing cases/frames without rewriting source configs
or manifests. `training_result`, `test_result` and the ready pinned final result
have separate state. Existing renderers accept stage options and unique widget
prefixes. Scientific runner/accounting/storage behavior remains unchanged.

A shared read-only cross-run comparison/presentation layer can develop
incrementally above saved experiments for app, notebook, Quarto and slide reuse.
Use explicit case references and the latest five-stage story. The
[earlier presentation design](PROJECT_PRESENTATION_PLAN.md) supplies background;
a large generic comparison layer is not a prerequisite for moving the views.
The implemented release/report reader still pins one run.

```text
read-only CSVs → shared available-at-snapshot preparation
             → strategy → original weights
             → feature recipe → model fitting → loss probabilities
original weights + probabilities → common adjustment → shared accounting
             → saved experiment (SQLite index + Parquet/JSON artifacts)
             → app / analysis notebook / Quarto report / slides
```

Basic ML now separates `RunConfig.prediction_target` (`loss` / `upside`) and
`model_scope` (`pooled` / `per_stock`) from portfolio strategies. The default
loss/pooled path retains its exact numerical behavior. `models.event_fields`
keeps legacy loss columns readable and gives upside its own strictly-positive
label/probability/control/calibration fields. `_fit_slice` shares preprocessing
and estimator construction across pooled/local fits. Local eligibility uses
only then-known stock history and class counts; an origin-level pooled fit
supplies explicit fallbacks. Frozen policy freezes both local fits and fallbacks.

`adjustments.upside_probability` applies the recorded threshold/multiplier then
normalizes to the original equity budget, preserving baseline cash and excluded
names. Case metadata records target, scope and adjustment. New variants get
explicit case IDs; originals and historical loss/pooled IDs are preserved.
Store configuration, predictions, fit summaries and manifests record the new
choices; no fitted estimator objects are persisted.

`basic_ml_ui.py` selects checksummed saved configurations before loading a chosen
result. `basic_result`/`basic_loaded_result` are independent of `training_result`.
Scope comparison requires matched protocol/data/calculation/runtime and shared
models; it loads recorded metrics, never fits. Explicit Basic ML computation
writes only its own result state. `run_basic_ml_comparison.py` declares a fixed
training-only four-cell protocol before requesting runs from the existing store.

`research/data.py` owns validation, date handling, lagged signals and fitted pools.
`features.py` identifies the declared model inputs and serializes learned-selector
settings; IDs and forward returns cannot be model features. `selection.py` fits
ANOVA ranking or correlation-pruned ranking inside the sklearn pipeline, before
scaling. Each historical fit records its actual mask, rankings, exclusions and
training-only correlations. `models.py` owns estimator factories and chronological fitting.
`strategies.py` owns portfolio rules; `adjustments.py` converts probabilities to
weights. `runner.py` orchestrates computation and attaches explicit case metadata.
The unchanged Part 0 pipeline performs portfolio accounting.

`cash_strategy.py` adds the annual-consistency pool and positive-momentum/cash
rule. Its `StrategySpec.allows_cash=True` explicitly permits stock weights below
one. `accounting.py` supplies the residual as a temporary zero-return cash leg
to the unchanged Part 0 pipeline, preserving full-portfolio drift-aware turnover.
Saved stock weights contain real IDs only; books and diagnostics record cash.
Cash-strategy ML haircuts transfer capital into cash without renormalizing.
Other strategies retain their original fully invested accounting/adjustment.
The allocation reader adds a labeled Cash series for charts and downloads.

`RunConfig.cash_momentum_months` configures the cash rule independently of the
annual screen and other strategies (default twelve, supported integers 2–12).
Shared data preparation keeps original signals and adds shorter `cash_` signals
only when needed. Their consecutive-history requirement matches the window.
The runner and CSV importer use the same settings; rebalance inspection shows
the actual window. Configuration hashing distinguishes lookbacks.

`momentum_studies.py` records a 2/3/6/12-month training-only protocol before
computation and links immutable runs in a separate momentum-study catalog.
Creation verifies identical annual pools/predictions; read-only comparisons
check sources, code/runtime, configurations, calendars and other strategy returns.
`momentum_study_ui.py` presents baseline curves in Strategies and optional model
versions in Basic ML. Its callback changes only training state and can load full
allocations explicitly. Model settings, feature recipes and cash policy stay fixed.

Persistent storage belongs in `research/store.py`. SQLite indexes run identities
and comparison metrics. Parquet stores typed frames, including complete weights,
signals, predictions, books and diagnostics. JSON records configuration and
provenance. CSV remains an interchange/export format. No database server is needed.

Cache identity must include configuration, source content hashes, calculation
code hashes and numerical-library versions. A different recipe, model policy,
parameter set or code version cannot silently reuse an exact-match cache entry.
Historical runs remain browseable as recorded evidence even when they no longer
match current code. Their provenance must remain visible.

A complete saved run must restore allocation histories without model fitting.
Writes become discoverable only after all artifact files are complete. Verify
artifact integrity on load; never mark a partial/corrupt run as a cache hit.

The UI can load a saved run or explicitly load/compute a configuration. Report
rendering only reads a pinned saved run. Neither plotting nor changing a chart
selection may trigger model training. A later UI framework can call these same
backend interfaces.

`strategy_groups.py` defines presentation-only families and pivots recorded
strategy/model changes without aggregation. `strategy_results_ui.py` renders
family sections, model-column change grids, full tables and detailed inspection.
Comparison filters affect display only; the sidebar family chooser edits the
next run configuration. This taxonomy is outside calculation cache identity.
See [strategy grouping](STRATEGY_GROUPS.md).

`evaluation.py` provides read-only period/protocol labels and combines historical
non-ML test CSVs after checking shared metrics, return histories and calendars.
`evaluation_ui.py` displays actual scope and historical test evidence inside
Final test. `result_views.pinned_final_summary` requires the pinned run to pass
existing readiness checks and the frozen test policy before filling the primary
final slot. Editor controls never relabel an unchanged result. Training stages
reject test inputs; feature/momentum studies keep their own validation dates. Historical archive reads do not replay strategies, refit models or
create current-code runs. See [evaluation periods](EVALUATION_PERIODS.md).

`feature_study_ui.render_feature_studies` loads a study once, then composes two
subtabs with a shared model choice: `render_selected_features` for saved inputs
and process, and `render_feature_effects` for independently retrained-model
prediction/portfolio evidence. Portfolio filters affect only the latter's
portfolio section; selected inputs and prediction quality remain available for
unsupported or empty strategy selections. Non-ML curves are unchanged references.
Tuning status belongs in the model-effects subtab. No new feature/model fitting
is triggered by this navigation. The study-session cache includes its artifact
root as well as study ID and metadata hash.

`research/feature_studies.py` declares a three-recipe plan before computation,
links immutable runs and verifies common configuration, source/code/runtime,
original portfolios and evaluation calendars. SQLite indexes completed studies;
JSON records their plan and run IDs. `feature_study_ui.py` reads saved summaries
to compare recipes and inspect masks without fitting or loading full weights.
The selected-factor union across annual fits may exceed ten; each fitted model
still uses at most ten. Final frozen submission models select once at their origin.

`research/imports.py` migrates verified CSV archives without fitting.
`research/reporting.py` reads pinned summaries and makes report figures.
`build_submission.py` executes the readable notebook and Quarto under the
`MLFACTOR_READ_ONLY=1` computation guard. The release file also pins the independent
equal-weight reference's content hash; a changed release during rendering aborts
the build rather than silently publishing a mixed result.

Two model protocols must be distinct: expanding research fits may learn from
earlier evaluation outcomes once realized; frozen-at-origin fits use only history
known before evaluation starts. Use the latter for the assignment's official
test model fits. Previously viewed test results are still not fresh evidence.
