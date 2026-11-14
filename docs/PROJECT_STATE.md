# Project state — 2026-11-09

## Latest: local LLM assistant ignore rules (2026-11-14)

Reviewed repository files and tracked assistant paths; only shared `AGENTS.md`
is currently present. Added targeted ignores for local Codex, Claude, Cursor,
Gemini, Continue and Aider histories/caches/settings, plus `*.local.md` instruction
names for AGENTS/CLAUDE/GEMINI. Shared instructions, skills and rules remain
eligible for Git. See [staging scope](DEPLOYMENT.md#github-staging-scope).
Verified representative ignored paths and retained shared/project/publication
paths using `git check-ignore`; no tracked files match the new rules. This was
repository maintenance only; no experiments or model fits ran. Next: review the
ignore changes alongside the existing uncommitted work before staging.

## Latest: app header subtitle removed (2026-11-10)

Removed the arrow-separated project-stage subtitle below the app title, as
requested. Verified the text is absent from `app.py` and the file compiles.
No new tests or experiments needed for this text-only change. Next: review the
header after reloading the app; include this code change in the next site update.

## Latest: basket/model/strategy/target navigation implemented (2026-11-10)

The user subsequently requested implementation of the plan. Seven main tabs
remain; saved portfolio comparisons now begin with asset basket, prediction
model, allocation strategy and next-up/down target where applicable. Non-ML
views use only applicable controls. Methods/selection teaching sections remain
separate from portfolio effects. Only existing saved combinations are offered,
with stale downstream choices reset and missing directions explained.

`portfolio_selection.py` maps original registry/book IDs to display descriptions;
`portfolio_selection_ui.py` shares the selectors. Strategies/Basic ML start with
one configuration and retain earlier multi-model/strategy comparisons behind
Compare more. Basic training scope/saved runs are secondary details; model fits
retain their recorded universe. Grouped direction choices include method, full/
top-ten basket and real momentum horizon. Feature effects now sit in their own
subtab. Tuning and engineering retain their actual recipes/settings/protocols.

Final has the same selection flow and a session-only **Compare saved final
portfolios** list spanning its three labeled studies. Original metrics and paths
are preserved; source protocols remain visible. Different calendars cannot be
charted as matched tests, and publication/mode changes clear the list. The
momentum-window allocation link selects the correct basket and saved run.

Verification: **240 tests pass** (144.072 s, one native thread), including six new
saved-case/protocol checks; source-free seven-tab scope/model/direction/horizon
interactions and real-browser views pass. See
`docs/verification/basket_first_ui_20261110.json` and companion screenshots.
Manifest remains `5dc7835271b868a609798657d4fa043be7dc19623c87ef1356ba6e3f3bef9372`;
all 475 display-frame hashes verified. No new experiments, fits, backtests,
publication exports, dependencies or ignore-rule changes. Computational modules
remain exact; independent submission work is preserved. No commit/push performed.

Read [implementation and handoff](BASKET_FIRST_UI_PLAN.md). Next: user reviews
navigation, then can push code/docs plus current publication (including the earlier
acceleration extension). Basket-specific fitting, missing pairings and formal
report integration remain separate work. The planning-only entry below records
the previous turn and is superseded by this explicit implementation request.

## Latest: basket-first navigation proposal — not implemented (2026-11-10)

The user could not find the acceleration view and requested a tab-by-tab plan
for basket → prediction model → strategy → up/down, explicitly declining a
refactor for now. The current view is the fourth nested tab under Grouping With
ML; its final evaluation has a separate Final test subtab. Acceleration source
and display frames remain local/uncommitted at this review; this agent has not
pushed the change to the hosted site.

Saved [BASKET_FIRST_UI_PLAN.md](BASKET_FIRST_UI_PLAN.md): current control audit,
seven-tab proposal, exceptions for non-ML/method views, shared saved-case adapters,
coverage/provenance rules, deployment/report boundaries and an incremental rollout.
Recommend Basic ML as a pilot. Choosing a basket initially filters existing
portfolio evidence; basket-specific model retraining and arbitrary new pairings
would be separate scientific work. No application code, experiment, publication,
dependency or submission change accompanies this plan. No new tests or fits run;
documentation paths and whitespace checked. Previous test evidence below belongs
to the prior implementation, not a newly implemented UI proposal.

Next: user reviews the plan. Do not start the proposal based on older “implement”
requests in this history. Preserve concurrent submission work.

## Latest: submission robustness and a passing --final build (2026-11-10)

`python3 build_submission.py --final` exits 0. It now checks the evidence the notebook
reads (`research/submission_checks.py`, 10/10 true); the older catalog-run checks are kept
in `build_status.json` as `catalog_run_checks_informational`. The engineering study is
pinned (`ENGINEERING_STUDY` in `research/notebook_views.py`), so the training-only
`…172206` run no longer changes Section 6. New Sections 5b (tuned settings on the chosen
book, 2010–2018: no material change), 8b (costs: chosen book still ahead of equal weight
at 25 bps) and 8c (significance: not significant, p = 0.48 test).

Report and slides rewritten to the notebook's structure (same day): `project_report.qmd`
(sections numbered as in the notebook, inline numbers read from saved results) and
`presentation.qmd` (16 slides, no code, a "how to read it" line per chart, speaker notes).
Shared figures live in `context/notebook_cells/12_presentation.py`. Slide overflow checked
with headless-Chrome screenshots. Old QMD/HTML archived in
`submission/archives/pre_presentation_20261110/`. `--final` still 10/10.


## Latest: grouped momentum acceleration experiment (2026-11-10)

Added **Grouping With ML → Momentum acceleration** and a separate **Final test →
Momentum acceleration** section. Original K-means/Ward group formation, top-ten
positive-side-volatility screening, positive momentum and cash rules stay exact.
The app explicitly distinguishes equal-weight basket return signals from actual
momentum-weighted stock positions. The new experiment compares original, stock,
basket and combined deceleration-to-cash rules, with a declared 50% haircut.
Current members are fixed across the two return observations; all signals use
already-earned returns. No directional classifiers were fitted or retuned.

Study `acceleration_20261110T204704_5c0c838f` is complete: training January
2005–December 2009 (60 months), then a separately labeled previously viewed
2010–2018 check (108 months). Across both grouping methods, eight metric rows per
phase, 480/864 monthly rows and 6,000/10,800 stock audits. All four original
method/window books match saved references exactly. K-means test basket-only
CAGR falls 19.61% → 14.36%, drawdown shrinks −21.33% → −14.65%; show the trade-off.

Current publication: **476 files / 173,332,889 bytes**, preserving all 469 earlier
frame hashes. Old default preserved in ignored `project_before_acceleration_20261110`.
Deploy remains source/object/catalog/fit/replay free; existing ignore rules include
new code and current Parquet while excluding scientific artifacts and backups.
All **229 tests pass** (215.021 s, one native thread); source-free seven-tab AppTest
and real-browser training/basket/final checks pass. See
[MOMENTUM_ACCELERATION.md](MOMENTUM_ACCELERATION.md) and
`docs/verification/momentum_acceleration_20261110.json` plus PNGs.

Next: user reviews this fixed-rule comparison; a separately declared acceleration
ML input recipe remains optional and unrun. Formal report/slides/release pin are
not revised by this app experiment. Preserve the other LLM's independent notebook
rewrite and volatility-neutral study below. No commit or push performed.

## Latest: graded notebook rewritten for presentation (2026-11-10)

`submission/analysis.ipynb` now follows the assignment rubric (summary, rubric map,
sections 1–9, limitations). It stays read-only (`MLFACTOR_READ_ONLY=1`), executes
cleanly with nbclient from `submission/`, and shows the modeling code a grader expects
without refitting: the equal-weight CAGR recomputed from saved monthly returns, the
ANOVA-10 pipeline object, the declared grids and the exact `GridSearchCV` lines
from `research/tuning.py`. The previous notebook is archived under
`submission/archives/pre_presentation_20261110/`.

New Section 7 reads `0_7_Feature_Engineering/part8_*` from
`03_vol_neutral_momentum_ml.ipynb`: AUC ≈ 0.49–0.51 with a volatility-neutral label;
bucket alpha ≈ 2.2%/yr test (t 1.9), 3.0%/yr walk-forward (t 0.9); ML+momentum
blend (w = 0.50 chosen on 2005–2009) test Sharpe 1.01 vs equal weight 0.87.
The test window was already viewed; reported as a check.

Unchanged: declared choice, `release.json`, `project_report.qmd`, `presentation.qmd`
and rendered HTML. Note: `engineering_20261110T172206_74f70a89` (training-only, made by
a default `run_feature_engineering.py` run) is now the newest engineering study, so
Section 6 reads it instead of `engineering_20261110T171255_825ffa99`; its closing
sentence therefore says no 2010–2018 engineering stage is saved for that study.

Next: decide whether the report/slides adopt the new structure and Section 7; rebuild
with `python3 build_submission.py`.


## Latest: make the original K-means top-ten cash book recognizable (2026-11-10)

The requested K-means → top-ten positive-volatility → one-month positive-momentum
→ residual-cash strategy was already evaluated and published. It was displayed
as “Top 10 positive-vol” under “Earlier grouped portfolios”, obscuring its identity.
The Final subtab is now **K-means strategies**. Its explanation, original table row
and curve explicitly name K-means, the momentum window and cash. Expanded tables
label two-month variants correctly too. The feature-engineered final comparison
now defaults to the same K-means top-ten starting book in fresh sessions.

Saved original CAGR: 24.6807% on 2005–2009 training validation; 19.6133% on
2010–2018 test, with average test cash 51.5556%. These are the existing original
book, independent of the chosen directional classifier. No new fits/backtests or
publication changes. Twelve focused tests and a source-free deploy AppTest verify
visibility/defaults/labels and the original return value. See
`docs/verification/kmeans_final_visibility_20261110.json`. Next: push UI/docs;
all numerical evidence and the formal release remain unchanged.

## Latest: feature engineering, tuning and frozen final variants (2026-11-10)

Reviewed `0_7_Feature_Engineering/01_feature_engineering.ipynb`, preserved the
original under ignored notebook-review artifacts, and annotated its exploratory
scope. Corrected future-rerun exports for PCA/scaler/full-precision IC state,
missing-IC denominators and the claim that unsupervised construction cannot
involve overfitting. Existing exploratory outputs/CSVs remain intact.

Seven main tabs now include **Feature engineering** after Tuned model. Its
methods/results/tuning subtabs compare six recipes, four classifiers, separate
up/down targets and fixed/tuned settings on six unchanged one-month books,
including both K-means and Ward stock groups. Every learned feature is fitted
inside its chronological CV fold. The input budget is at most ten, but some
composites aggregate twenty source factors; rubric interpretation is not assumed.

Study `engineering_20261110T171255_825ffa99` is complete: training labels through
2006-12-31, frozen January 2007–December 2009 validation; separately frozen
2010–2018 final evaluation fitted once on all 140,739 training rows. Parameters
come from the earlier training CV, with no test retuning. Each stage has 96 fitted
recipe/classifier/direction/settings variants, 582 portfolio cases including six
originals; 20,952 training and 62,856 final monthly rows. There are 144 CV candidate
rows. Later stock-group/technical updates use only already-earned returns.

Final test now separates earlier grouped portfolios from feature-engineered
variants. The older ANOVA/factory final study and formal submission pin remain
unchanged. The extension has its own fixed/tuned protocol; old tuning filters or
statements do not imply those new results are absent. Shared plotted results
include CAGR/log growth, drawdown, cash and actual fitted input lists/counts.

Default `published/project` now contains **470 files / 172,577,499 bytes**, adding
about 2.6 MB while preserving all 457 earlier frame hashes. Prior default preserved
at `published/project_before_engineering_20261110`. The app needs no raw extended
CSV, forecast panel, fitted estimator or research catalog to review any new model.
New `.gitignore` comments describe these existing exclusions; all current release
files remain included and no ignored files are already tracked.

Verification: **221 full-suite tests pass** (219.838 s, one native thread), plus
exact training/final frame parity, every saved growth/cash identity, four old final
reference books, 288 fitted/prediction artifact checksums, source-free seven-tab
deployment under failing raw/archive/object/fit guards, and browser method/tuning/
Ward-final charts. Saved-result notebook and standalone QMD/HTML supplement are
built using the same published reader; formal notebook/QMD/slides/release remain
separate. See [full protocol/handoff](FEATURE_ENGINEERING.md) and
[evidence](verification/feature_engineering_20261110.json).

Next: review and push the code/docs/current publication. Do not automatically
choose a test winner or claim composites satisfy a limit on raw source factors.
Keep the original default models/results, earlier experiments and other LLM edits.

## Latest: tuning choices, final comparisons and Git ignore review (2026-11-10)

Tuned model now shows all fourteen saved basic strategies by default; the old
four-family focus had hidden thirteen. Keep four classifier choices and separate
up/down targets, frozen feature/search audits and metric tooltips. Added clear
classifier-versus-portfolio/clustering explanations and an all-strategy metric
comparison. Equal weight remains the untouched benchmark; ML produces a separate
weighted portfolio. Neither classifier settings nor grouping searches are fitted
by viewing the tab.

Final test now shows CAGR and annual log growth, wealth/cumulative-log/monthly-log
curves, and each overlay's change relative to its own original and horizon.
Coverage explicitly distinguishes fixed K-means test portfolios from untuned Ward
training results and separate classifier tuning holdouts. The joined path is
labeled as two protocols; local submission-release pending no longer implies
saved test results are absent. No new model, backtest or published data was made.

Git's actual ignore/tracked checks confirm raw CSVs, artifacts, old releases,
secrets and caches are excluded, current publication/source retained, and no
already-tracked ignored files. No `.gitignore` edits or Git history removal needed.
**214 tests passed** (190.292 s). See [handoff](TUNING_AND_FINAL_REVIEW.md) and
[verification](verification/tuning_final_review_20261110.json) for source-free
and browser verification. Next: push reviewed UI/docs. Clustering search,
grouped portfolios with tuned forecasts and new Ward/tuned official-test passes
remain distinct research work; do not label current final evidence as those.

## Latest: Grouping With ML — two unsupervised methods (2026-11-10)

The user confirms successful website deployment. The third main tab is now
**Grouping With ML**, with **K-means**, **Hierarchical clustering**, and **Adding
a direction**. K-means has clearer centroid/input explanations. Ward hierarchical
clustering uses the same standardized twelve-month histories and five groups;
an interactive actual saved merge tree explains its nested groups. Both feed the
same full-group/top-ten positive-momentum/cash rule, then the same four saved
up/down classifier overlays. A selector chooses grouping in the directional view.

New training-only Ward base `baskets_20261110T122422_88c2cc4a`; adjusted study
`baskets_20261110T122521_a62c9b0c`: 49 books / 2,940 monthly rows, zero new
classifier fits. One-month top-ten CAGR is 21.24% versus K-means 24.68%; this is
comparison evidence, not a declared improvement. Later feature/tuning/final
studies remain K-means and are labeled independently.

Default `published/project` now has 458 files / 169,973,507 bytes, preserving
all 362 prior frame checksums and adding lazy Ward partitions/tree data. Old
release preserved locally at `published/project_before_grouping_20261110`.
**210 tests pass**, plus exact real-data partition parity, a source-free six-tab
standalone deploy check for both methods/four classifiers/two starting books,
and real-browser tree/method-switch checks. No source CSV, earlier experiment,
formal submission pin or report source was changed. Other LLM edits remain intact.
See [grouping handoff](GROUPING_WITH_ML.md) and
[verification](verification/grouping_20261110.json). Next: review/push this hosted
presentation; decide separately whether to extend downstream research to Ward.

## Latest: GitHub ignore rules

The root `.gitignore` excludes CSV datasets/reports, the full experiment archive,
generated research/Quarto output, older publication directories, Python caches,
virtual environments and local credentials. It explicitly keeps the current
`published/project` release and `published/README.md`, along with application and
legacy Python modules, notebooks/QMD sources, requirements, docs and the theme.
No files are deleted, no project Git repository is initialized, and nothing is
committed or pushed by this change. Verification used Git's own ignore rules with
a temporary external repository: 18 excluded and 20 retained examples pass, and
all 363 current published files remain eligible. The resulting candidate set is
611 files / about 129.4 MB; its largest file is about 2.4 MB. See
[deployment](DEPLOYMENT.md) for the staging scope.

## Latest: all six tabs published; feature explanations and main-family focus

`published/project` is now the default deployment package: **114,541,240 bytes**.
The previous three-tab package's frame checksums are preserved; only about 3.3 MB
is added for Features, Tuning and Final test. The viewer needs no raw/result CSV,
catalog or model object, and never fits selectors/classifiers/K-means or replays
portfolios. Earlier publications and source experiments remain intact.

Feature selection publishes registry study `features_20261109T202318_8b01fe74`
and advanced score `features_20261110T022741_e6f5661f`: all four recipes/models,
input audits/correlations, calibration, registry curves, and 166 advanced-book
cases. Bar hovers explain feature names/meanings/categories and saved importance.
The initial curve hovers explain recipes and references. The app explicitly says
the advanced grouped/top-ten up/down classifiers are included, with separately
refitted recipe-specific models. K-means and all portfolio rules stay fixed.
The full-group original is now shown alongside its recipe-adjusted curves.
The feature-ranking audit describes downside fits; upside has independent fits.

Tuned Model reads the existing three holdout studies, including settings and
all 26 grid trial rows, for both targets and all four classifiers. It remains
the frozen 2007–2009 basic-strategy comparison. No advanced-family tuning is
invented. Final test reads `final_20261110T025420_6d3405bd`; test-only, training-only
and joined 168-month views are precomputed at export using the existing saved
monthly-return helper. Archived non-ML test reports are exported as Parquet too.
The final model still uses fixed parameters and ANOVA-10; formal release/report
sign-off remains separate and its pin is unchanged.

The requested “four main models” filter is interpreted as four portfolio families.
Feature selection, Tuned Model and Final test now start focused, with a checkbox
to enable additional saved comparisons. All four classifier algorithms stay
available. Focused tuning offers equal weight only and explicitly explains the
missing grouped/top-ten tuning; all fourteen basic rules remain available with
focus off. Foundation tabs retain their examples. This is presentation state,
not an experiment setting or new numerical result.

Verification: **203 full-suite tests passed** (126.347 seconds). Exact real-data
parity for all feature summaries/audits/calibration and 112 registry curve views,
166 advanced-feature books / 9,960 months, all tuning frames / 26 trial rows,
and final test/training/entire windows. A standalone copy with code and only the
publication exercises all six tabs, four feature classifiers, eight tuning
model/target combinations, final windows and archived comparisons under failing
CSV/store/fitting/replay guards. A real browser hover displays the financial
meaning on the importance bar; focus controls and final 168-month labels work.
Previous publication frame checksums and the formal release pin remain unchanged.

See [current deployment commands](DEPLOYMENT.md) and
[verification](verification/project_deploy_20261109.json). Next: review the
presentation, measure hosting resource use, then prepare the curated GitHub
deployment. Formal notebook/Quarto/slides integration and any advanced tuning
are separate work. Earlier entries below describe historical partial deployment.

## Latest: explain the grouped strategy, then add directional predictions

Advanced ML now contains **Build the grouped strategy** and **Add up/down
predictions** subtabs, preserving the six-main-tab navigation. The first explains
K-means on individually standardized 12-month earned-return paths, shows actual
saved group membership/counts and a standardized path heatmap, then teaches the
positive group/stock momentum and cash rule before the top-ten positive-volatility
screen. Its results contain original books only. The second chooses full groups
or top ten per group, horizon and classifier, then compares that original with
separate downside-to-cash and upside-tilt versions. Probability scores and stock
inspection remain available. No science or fitted model has changed.

The path illustration uses saved one-month stock momentum values, current-month
membership and explicit completeness coverage. Earlier cluster IDs are never
linked as sectors. The latest snapshot illustrates all 1,191 grouped stocks;
early snapshots without 12 archived observations show an honest missing-path
message. `research/basket_reporting.py` is a pure saved-frame reader/transformer,
reusable in notebooks/Quarto, not a clustering or portfolio runner.

Default deployment is now `published/advanced_ml`, **111,258,117 bytes**. It
retains exact first-two-tab evidence and adds the lead study
`baskets_20261110T021017_e6fa43e2`: 49 books / 2,940 monthly rows, 13,500 group
rows and 1,988,037 stock assignment/weight/probability rows, partitioned for
on-demand inspection. Original publications, raw datasets and experiments remain
intact. Features, Tuning and Final test are still unpublished. Training-only
source/timing guards apply to all three published tabs. No raw CSVs, catalog,
refitting or replay are needed to open the app.

Verification: **198 full-suite passes** (115.467 seconds), followed by five focused
presentation tests after the final selector-label clarification. Real-data review
matches all 49 advanced books, 2,940 return rows, 13,500 group rows and 1,988,037
assignment/weight/probability rows exactly, plus four models × three horizon curve
sets. First-two-tab summary frames match the prior publication exactly. A fresh
standalone deploy entry point with code and only this publication exercises all
four classifiers, both starting books, stock audits and independent horizons under
failing CSV/store/refit/replay guards. Browser checks cover both subtabs, explanations,
membership/profile charts, momentum/cash, top-ten and directional controls.

Verification and precise limits are in [the basket guide](BASKET_MOMENTUM.md),
[deployment handoff](DEPLOYMENT.md) and [evidence](verification/advanced_refactor_20261109.json).
Formal report/notebook/slides and submission release remain unchanged. Next:
publish the feature-selection tab using the same explicit-source/shared-reader
pattern; keep the newly separated unsupervised grouping and directional-model
explanations when integrating the presentation into Quarto.

Earlier one-/two-tab deployment entries below are historical.

## Latest: collapsible Basic ML and a self-contained two-tab publication

Basic ML now has three sibling panels, collapsed by default: prediction quality
(downside or up move), portfolio results and changes from each original portfolio.
Existing models, strategy accounting, filters and all-strategy chart defaults
are preserved in both local and deployed views. Nested group/settings expanders
were replaced within those panels to respect Streamlit 1.32's nesting limit.
The Basic ML page explicitly labels training-only chronological validation.

The default deploy root is now `published/basic_ml`: **56,589,983 bytes**. It
retains the first tab's exact evidence and adds 196 Basic ML portfolios / 11,760
monthly rows from the four already-completed January 2005–December 2009 runs:
`20261109T185402_0722a7ec725c` (downside pooled),
`20261109T185621_fe4017faa327` (upside pooled),
`20261109T185517_1bf238ea2712` (downside per stock),
`20261109T185736_3b19b90c1122` (upside per stock).
Pooled has all four classifiers; per-stock has the saved logistic pilot only.
All 20 factors, fixed parameters, annual expanding fits, 14 rules, 3-month cash
momentum. No feature selection, tuning, test results or new computation added.

`PublishedBasicStore` adapts the same chooser and provenance-matched scope
comparison to checked manifest files; it never constructs a research store.
Scores/calibration are summaries; stock-month probability panels are omitted.
Weights and fit audits load on demand. Matching Parquet content is stored once.
The original `published/strategies` publication remains unchanged and usable;
only Advanced ML, Features, Tuning and Final test remain unpublished.

Verification: **193 full-suite tests passed** (105.998 seconds). Exact real-data
parity across all 196 cases / 11,760 months: all three return-curve measures,
original-relative curves and change tables, score/control/calibration tables,
fit audits, local/fallback counts and every strategy/model allocation matrix.
Matched scope comparisons also agree exactly with the research reader. A fresh
standalone process with code and only the two-tab publication exercises all four
target/scope combinations, fit inspection and cash allocations with raw reads,
research-store construction, fitting and replay patched to fail. Actual browser
checks verified all three panels collapsed by default, opening/closing content,
upside/downside and pooled/per-stock switching. First-tab frames and the old
publication's manifest checksum are unchanged.

See [deployment protocol and next-tab handoff](DEPLOYMENT.md) and
[verification evidence](verification/basic_ml_deploy_20261109.json). No raw data,
prior experiment, research result, formal submission release, QMD, notebook,
HTML or slides were changed. Next: apply this publication pattern to Advanced
ML's separately saved basket study; no new strategy/model research is needed
just to publish its existing views.

The entries below record preceding steps; their one-tab deployment status is
historical.

## Latest app refactor: published view mode, starting with Strategies

Research remains the default (`python3 run_app.py`). Deploy/view mode
(`python3 run_app.py --mode deploy --port 8502`, or Cloud entry `deploy_app.py`)
loads only `published/strategies`, a **4,504,572-byte** self-contained export.
No raw CSV, model prediction panel, research catalog, source-file fingerprint
lookup or computation is required to browse it. Fourteen original portfolios,
840 earned-month rows (January 2005–December 2009), all main-run allocation
histories and saved baseline momentum-window curves are included. Allocations
load on demand. Source run `20261109T202421_184e95372ab2`; optional momentum
study `momentum_20261108T211612_b4723b7c`. The export was read from existing
saved evidence, with **zero fits and zero portfolio replays**.

Only the first deployed tab is implemented; the other five display honest
not-yet-published states without invoking their archive readers. Local tabs and
explicit compute controls remain available. A shared training guard rejects
test evaluation, test sources/dates, ML cases and mistimed snapshots in the first
tab. Both modes show a training-only notice and actual saved evaluation dates.
Strategy names now have hover and keyboard-focus explanations in grouped and
full-table layouts, using the shared metric formatting and recorded cash window.

Verification: **189 full-suite tests passed**, followed by the six focused
publication tests after the final presentation/description refinements. Exact real-data parity for
all fourteen metrics/allocations, 840 monthly rows, all three return views and
axis settings, and four-lookback tables/curves. A fresh-process standalone copy
with **no source CSVs or research artifacts/catalog** runs the deployed entry
point and allocation/momentum views while raw reads, store construction and
computation are patched to fail. Actual browser hover, keyboard focus, full
tables, allocation opt-in and pending-tab boundaries passed. See
[deployment handoff](DEPLOYMENT.md) and [evidence](verification/strategies_deploy_20261109.json).

Next: refactor Basic ML to publish selected models/targets/scopes, matched
baselines and diagnostics through the same manifest approach. Do not export every
intermediate model automatically or build a second UI/runner. No GitHub upload,
public deployment, research re-fit, source-data mutation or formal submission
release/report update was performed. The final-book/feature work below was
already present and remains intact.

## Latest: one frozen 2010–2018 pass of the four books

Saved study `artifacts/final_books/final_20261110T025420_6d3405bd`. One
`frozen_at_origin` fit at 2009-12-31, labels through that date, 140,739
training rows, eight factory fits. ANOVA kept the same ten factors for every
model and both events: Tot_Debt_Rev, Net_Debt, Share_Turn_3M, Debtequity,
Ocf_Ce, Div_Yld, Share_Turn_12M, Tev_Less_Mktcap, Total_Debt_Capital,
Share_Turn_6M. The December 2006 ANOVA list had Total_Liabilities_Total_Assets
where this fit has Ocf_Ce.

Equal weight matches the graded test row: CAGR 13.096%, volatility 15.63%,
Sharpe 0.868, max drawdown −20.6%, 108 months. Logistic regression, one-month
horizon, gross of costs: top 10 CAGR 19.61%, volatility 39.2%, Sharpe 0.591,
max drawdown −21.3%, cash 51.6%. Downside-to-cash: CAGR 15.64%, Sharpe 0.590,
drawdown −16.7%, cash 62.8%. Upside tilt leaves that logistic book unchanged.
Full baskets: CAGR 4.01%, Sharpe 0.528, cash 54.2%. Downside-to-cash: CAGR
3.15%, drawdown −9.7%, cash 64.4%.

On January 2005–December 2009 the top-10 book was CAGR 24.68% and Sharpe 0.833
against equal weight at 7.62% and 0.433. On this decade the CAGR gap shrinks
to about 6.5 percentage points and the Sharpe advantage reverses. The full
basket book's training-window edge does not carry over. Histogram boosting's
upside tilt is the overlay that moves the one-month top-10 book: CAGR 22.3%,
volatility 53.8%, Sharpe 0.53. The other classifiers stay near the unadjusted
book. The recipe is not changed after these numbers. The submission pin remains
`20261108T144257_f45441ba8e83`, draft. The Final test tab reads this file.
Browsing does not fit.

## Feature cuts barely move the four books

Feature selection now leads with the four presentation books. Saved score
`artifacts/basket_feature_scores/features_20261110T022741_e6f5661f`, January
2005–December 2009. Recipes: all 20, ANOVA-10, correlation-pruned 10, forest-10.
Equal weight and the top-10 momentum book do not use the factors. The overlays
replay saved forecasts. Logistic, one month, the largest CAGR gap versus all 20
is about 0.2 percentage points (top-10 upside tilt). Official test file was not
opened. Tuning has not been rerun on these four books.

## Up-or-down overlay on both Advanced ML books

The same two saved-forecast adjustments now sit on the full correlation-basket
book and on the top-10 positive-vol book. Lead study
`artifacts/basket_studies/baskets_20261110T021017_e6fa43e2`, January 2005–
December 2009. Source baskets `baskets_20261109T223026_c7be47ff`. Forecasts
`20261109T185402_0722a7ec725c` (loss) and `20261109T185621_fe4017faa327`
(upside), all 20 factors, pooled, zero new fits. The earlier top-10-only
overlay `baskets_20261109T225254_b6a479dd` stays on disk.

Logistic regression, one-month horizon, gross of costs: full baskets 10.07%
CAGR and Sharpe 0.709; downside-to-cash 7.92% and 0.710, drawdown −11.0%
versus −13.9%, cash 67.1% versus 57.8%; upside tilt 10.01% and 0.706 with
the original cash. Top 10 stays 24.68% and Sharpe 0.833; downside-to-cash
19.41% and 0.833, drawdown −16.1% versus −20.5%; upside tilt 24.98% and
0.841. The other three classifiers move the same way. The screen is still
what raises return. Official test file was not opened. Final test remains
pending. Feature selection has not rescored these four books.

## Tuned model controls

Tuned model now starts with **Model**, **Basic strategy**, **Prediction target**
(next down or next up). Each choice loads matching saved evidence; it never
fits while browsing. Probability scores precede portfolio results; controls
are deduplicated and have Brier / log-loss / ROC-AUC header tooltips. A settings
comparison and explicit same-defaults message explain identical tuned books.

All four predictors now have both targets. Original downside logistic/boosting
study stays `holdout_20261109T202045_b0a51718`; new downside NB/forest study is
`holdout_20261110T012820_d20ce131`, new all-four upside study is
`holdout_20261110T013013_6f2fd4f9` (UTC timestamps). Same frozen downside ANOVA-10
list, December 2006 fit, January 2007–December 2009 holdout, fourteen rules.
Upside uses the existing 60% threshold / 50% boost and preserves original cash;
downside retains the existing haircut/cash policy. Settings minimize log loss,
not a portfolio score. New files: 1,083,315 bytes; 63 search/refit + 12 frozen
prediction fits. No official test CSV read, catalog run, or release change.
No model beats the historical control on holdout Brier/log loss. See
[protocol](TUNING.md) and [verification](verification/tuning_controls_20261109.json).

Verification: **175 full-suite tests passed**; saved-only review matched all
eight model/target score tables and sixteen strategy tables to actual artifacts,
with fitting/replay blocked. Original non-ML holdout books match exactly across
all three studies; growth identities and new source/code hashes match. Actual
browser pointer hovers displayed all three metric tooltips; target switching
and the updated upside score table were visually reviewed without exceptions.
Next: review the saved tuning/feature/strategy evidence and choose the final
specification. Frozen official test and formal report/release integration remain
pending. Older downside-only tuning entries below are historical.

## Read first

**Advanced positive-volatility baskets now have separate downside/upside overlays
for all four basic models.** Lead saved study:
`artifacts/basket_studies/baskets_20261109T225254_b6a479dd`, Jan 2005–Dec 2009.
It retains all originals and uses their 1-/2-month top-10 momentum weights.
Downside multiplies by `1 - 0.5*p_loss`, leaving the haircut in cash. Separately
fitted upside boosts held names at p>=0.60 by 50%, then preserves the original
equity/cash budget. No combined overlay. Four-model selector updates metrics,
curves, scores and stock allocation audits. Inputs: all 20 factors, no selection
or tuning. Reuse saved annual pooled forecasts from
`20261109T185402_0722a7ec725c` and `20261109T185621_fe4017faa327`; **zero new
classifier fits**. Original basket books reproduce exactly. No test CSV read,
registered-rule changes, or formal release change. See [protocol](BASKET_MOMENTUM.md).

**Tuned model explicitly explains its task and feature source.** It tunes the
stock-loss classifier for log loss, not the trading rule or Sharpe. The saved
ANOVA-10 list is frozen at Dec 2006; holdout is Jan 2007–Dec 2009. The strategy
chooser applies the same predictor to a basic rule. Other tabs' feature choices
do not retrain it. Cash versus full-investment adjustment is stated explicitly.
The new basket overlays above are separate untuned/all-20 evidence.

Verification: **171 full-suite passes** before the final extra provenance test;
then **4 focused directional replay checks passed**, covering missing keys,
future fits, source hashes, cash budgets and read-only behavior. Real saved-only
review checked all four selectors against exact values, both horizons, original
parity, allocation audits and archived study compatibility with fitting/replay
blocked. Actual browser comparison/tuning screenshots were reviewed. See
[evidence](verification/basket_adjustments_20261109.json). Next: review these
exploratory results, then define a selected-feature/tuned hybrid comparison if
desired. Frozen final test and formal release integration remain pending.
Historical entries below retain their earlier status.

**The Advanced ML tab now keeps the 10 highest positive-vol names in each group.**
The groups still come from the 12-month correlation path, and the full 1-, 2-,
and 3-month books stay. The new book trades 1- and 2-month momentum on the
reduced groups. Positive volatility is the root-mean-square of up months over
those 12 earned months; down months contribute zero. Each month keeps 10 names
in each of the 5 groups, and the book is scaled to those 50 names. Saved
comparison `artifacts/basket_studies/baskets_20261109T223026_c7be47ff`,
January 2005–December 2009. One-month top 10: CAGR 24.68%, volatility 31.7%,
Sharpe 0.833, max drawdown −20.5%, turnover 53.4%, average cash 50.4%, about
25 names held, maximum weight 14.9%. Two-month top 10: CAGR 16.14%, Sharpe
0.780, max drawdown −29.5%, turnover 40.5%. Gross of costs. The 2010–2018
test file was not opened. See [basket momentum](BASKET_MOMENTUM.md).

**Feature portfolio results now have their own nearby model selector.**
**Prediction model for portfolio comparison** appears directly above the table,
ordered logistic regression, Gaussian naive Bayes, random forest, histogram
gradient boosting. It synchronizes with the existing shared feature-study
model selector; either control updates actual saved prediction/portfolio results
and feature inspection. This is display/state wiring only, with no new training,
artifact or report changes. Verification: 13 feature checks and saved-only
switching through all four models, exact portfolio values and bidirectional
selector synchronization. Next: review the existing saved comparisons.

**Feature-result tables explicitly identify their prediction model.** The
shared control is now **Prediction model to review**. Both predictive and
portfolio tables include **Prediction model**, and the portfolio heading repeats
the selected name. Rows compare independently refitted feature recipes for one
predictor; the forest feature recipe can supply inputs to any classifier.
This changes display labels only, with no research/artifact/report-pin changes.
Verification: 13 existing feature checks and real saved-only review of all four
model selections. Next research work remains the separately documented final
specification and evaluation; no new experiment is needed for these labels.

**The full-group 1-, 2-, and 3-month books remain in that same study.**
Ungated one-month momentum plus cash: CAGR 11.65%, Sharpe 0.693, turnover
47.9%. The same signal inside the full groups: CAGR 10.07%, Sharpe 0.709,
drawdown −13.9%. The earlier comparison before the vol screen is
`baskets_20261109T213124_e664776f`. The 12-month holding study remains at
`baskets_20261109T210228_f4ad9b2a`.

**The Tuned model tab compares three books on a holdout inside the training years.**
It sits after Feature selection and before Final test. The saved comparison is
`artifacts/tuning_studies/holdout_20261109T202045_b0a51718`. Features are the
ten ANOVA names from the 2006-12-31 fit. Labels through December 2006 choose
the settings. Reported returns are January 2007–December 2009, 36 months.
Logistic search chose `C=0.1` (10 fits). Histogram boosting chose its existing
defaults, learning rate 0.05 and 15 leaves (13 fits). On equal weight, no ML
CAGR was 2.62% and Sharpe 0.231; untuned and tuned logistic were 2.58% and
Sharpe 0.230. Boosting matched its untuned book. This window includes the
crisis and is not the 2010–2018 test. The test CSV was not opened and the
experiment catalog was not given a new run. See [tuning](TUNING.md).

Verification: `tests/test_tuning.py` passed, including the holdout split.
`test_app` tab-order check and the final-tab regression check passed. The full
suite was not rerun.

**Random-forest feature selection is implemented and saved.** Study
`features_20261109T202318_8b01fe74` compares all-20, ANOVA-10, pruned-10 and
forest-10 on all fourteen strategies and four pooled downside predictors.
Annual training-only fits evaluate Jan 2005–Dec 2009; cash momentum is three
months and adjustment strength 0.50 for every recipe. Each selected fit uses ten
inputs; the forest's five-origin union is eleven names. See
[findings](../0_3_Feature_Selection/05_random_forest_findings.md) and
[implementation handoff](RANDOM_FOREST_FEATURE_PLAN.md).

Forest selection lowers Brier score for logistic/naive Bayes but worsens it for
forest/boosting. None beats the historical control. This is fixed-model feature
evidence, distinct from the separate tuned ANOVA holdout above. Final feature/
model choice, tuning release integration and frozen official test remain pending.

The app shows importance bars with tree standard deviation, actual masks,
method overlap, calibration, portfolio deltas, CAGR and log-return paths. Shared
`research/feature_reporting.py` readers supply the app, new executed review
notebook, submission notebook, report and slides. The release adds a checksummed
feature-study pin while retaining its older main run. Prior submission sources
and HTML are archived under `submission/archives/pre_forest_20261109/`.

Verification: **155 research/app tests passed**. Saved-only real-data review
checked all methods, current cache matches, release identity, 280 portfolio
growth identities and exact all-20 parity with the preceding matched run:
four probability tables and seventy books/weight histories. All 2,004 recorded
pre-existing data/artifact hashes match. Both notebooks and Quarto HTML/slides
were executed/rendered with fitting disabled. The four new runs occupy 143.5 MiB;
the study took about 375.5 seconds. Actual browser review confirmed twenty
importance bars/ten selected inputs, all four recipe curves, the report and
both new slides. Slide layouts passed viewport-overflow checks. The final
readiness gate still rejects the incomplete final submission. See
[evidence](verification/random_forest_20261109.json). Next: review the new
feature findings and choose final specifications; do not treat this study as
tuned or as a final-test experiment. Historical entries below retain their
status at the time of each earlier update.

Basic ML's **Performance through time** chart now initially selects all available
strategies within its display filters. Subsequent chart selections persist;
other tabs keep their prior defaults. This is a viewer-only default change.
Nine app checks passed; real saved-data review with fitting blocked confirmed
all fourteen defaults, retained manual changes and the unchanged Strategies default.

**Feature selection now has two subtabs:** **Selected features & method** shows
the actual selected inputs first, methods/rankings/exclusion reasons, training
cutoffs, stability and correlations. **Effect on ML models** shows saved
independent refits per recipe, prediction scores and ML-adjusted portfolio
comparisons. Non-ML rules have no selected model inputs and remain unchanged
reference curves. Feature inspection and prediction scores no longer depend on
a supported portfolio selection. The shared model/study controls identify the
recorded evidence; the existing study is pooled downside, not a new upside/local
feature experiment. See [usage](../0_3_Feature_Selection/README.md).
This is a read-only UI reorganization; no retraining, scientific-code, catalog
or report-pin change was needed. Tuning status is inside model effects and now
correctly distinguishes the available helper from its unrun panel search.

Verification: **22 focused checks passed** (13 feature-study and 9 app checks).
Real-data saved-only review verified exact selected lists for two methods and
two models at recorded origins, all three recipe run/refit records, unchanged
portfolio scores and unsupported-strategy behavior. Both subtabs were reviewed
in the actual browser. All 2,099 data/artifact/catalog/calculation hashes match;
four current saved runs remain exact cache matches. No new run was computed.
See [evidence](verification/feature_subtabs_20261109.json). Next research work
remains the chosen feature recipe/strategy, then the unrun tuning and frozen test;
this UI split did not supply newer target/scope feature studies.

**A small chronological grid search is available and has not been run.**
`research/tuning.py` takes a `FeatureSpec` and one training slice, runs
`GridSearchCV` on month-forward folds, refits the winner, and can save the
pipeline under `artifacts/tuning/` for a later demo reload. That folder is not
an experiment-catalog run. Scoring is negative log loss. Portfolio results are
not part of the search. See [tuning](TUNING.md). The intended order remains
feature selection, then a correlated-group weighting strategy, then this search,
then one frozen test. No panel search, basket, release pin, or test evaluation
was run for this addition.

Verification: `python3 tests/test_tuning.py` — 6 tests passed. The full research
suite was not rerun. Existing runners still record `tuning: none`.

**Basic ML now has two targets and two training scopes.** The upside threshold
boost (`R1M_Usd > 0`, threshold 0.60, strength 0.50) is available alongside the
unchanged downside haircut. Choose pooled or separate per-stock training;
per-stock fits use >=36 observed months and >=6 examples of each class, otherwise
an audited same-origin pooled fallback. The first local comparison uses logistic
regression. Basic ML has its own saved result selection/state and matched scope
comparison; switching it cannot replace the Strategies source. Prediction quality,
calibration, portfolio effects, log/CAGR and allocation inspection work for both.
See [basic ML directions](../0_6_Basic_ML_Directions/README.md).

Numerical code/configuration identities changed intentionally. Historical runs
remain readable; they cannot silently satisfy a current-code cache request.
The new pooled-downside run exactly reproduces all four probability tables and
all seventy matching-setting historical books. Feature/momentum
studies remain the earlier downside/pooled evidence, not a newly crossed study.
Advanced ML stays a placeholder; report pin and final-test status stay unchanged.
The prior update entries below record their state at that time.

**Log returns and CAGR are now visible together.** Strategy/Basic ML tables,
momentum/feature studies and archived test tables expose saved annual log growth
beside CAGR. Strategy/Basic ML charts can show wealth, cumulative log return or
monthly log return. Only cumulative series have a funding-point observation;
monthly logs use earned dates. See [return metrics](RETURN_METRICS.md).
Numerical calculations, saved results and the Quarto pin are unchanged.

Verification: **125 research/app tests passed**. Real saved-only review checked
all 70 books, exact 14-original/70-comparison growth-table rows, all return views,
and feature/momentum/archived test growth columns. The actual browser rendered
both log views in Strategies and Basic ML. All 203 preservation hashes and five
exact cache matches remain valid. No experiment was computed during this update.
The newest already-saved source is `20261109T040626_e0a702234197` (three-month
cash momentum); the catalog has twelve training runs and no registered test run.
The older draft report pin remains unchanged. See
[evidence](verification/log_returns_20261108.json). Next research work remains
feature/strategy review, tuning and final submission preparation below.

**The requested five-tab refactor is implemented.** Strategies shows only
training-period originals; Basic ML combines downside prediction quality and
paired portfolio effects; Advanced ML strategy stays a placeholder; Feature
selection supports selected training strategies; Final test is last and keeps
pending final evaluation separate from historical/exploratory evidence.
[TAB_REFACTOR_HANDOFF.md](TAB_REFACTOR_HANDOFF.md) records the design and current
status. Rendering reuses existing helpers and a small read-only case filter;
scientific calculation code, saved experiments and report pin are unchanged.
Do not infer the advanced strategy. Tuning/final research and hosting remain pending.

The user clarified six presentation goals: regular strategies, simple ML,
an author-designed hybrid, feature-selection effects, tuning evidence, and an
overall comparison shared by the app, Quarto report and slides. The proposed
[presentation/refactor plan](PROJECT_PRESENTATION_PLAN.md) maps these goals to
existing evidence and staged implementation. The latest implementation makes
the annual-consistency cash rule's momentum/history window configurable and
saves a matched 2/3/6/12-month study with four fixed ML overlays; details are below.
The shared multi-run presentation
layer and tuning remain unimplemented.
The latest UI clarification labels loaded periods and independently saved
studies explicitly, and exposes historical non-ML test results in a separate tab;
see [evaluation periods](EVALUATION_PERIODS.md). No new test experiment was run.
The latest presentation update groups strategy results into six families,
replaces repeated-name delta rows with model-column grids, and adds compact
family selection and detailed/full-table views; see [strategy groups](STRATEGY_GROUPS.md).
The first fixed training-only feature comparison is complete and saved. Keep documentation current after
each change, as specified in the project-root AGENTS.md. Preserve prior
experiments and Part 1. Tuning and final submission decisions remain pending.

## Existing research

- CSVs: 140,739 training stock-month rows / 120 snapshots; 127,740 test rows / 108
  snapshots. Features are twenty percentile-like factor columns, not raw ratios.
- Snapshot t's R1M_Usd is the following month's simple return. Training earns
  Jan 2000–Dec 2009; test earns Jan 2010–Dec 2018.
- Part 0 validates data and independently checks equal-weight accounting.
- Fourteen non-ML portfolio rules are registered, including frozen training
  volatility, Sharpe and Sortino pools followed by monthly momentum, and the new
  annual-consistency positive-momentum rule with explicit cash.
- Four fixed classifiers: logistic regression, GaussianNB, random forest and
  histogram gradient boosting. One declared factor recipe per experiment;
  explicit manual subsets and learned ANOVA/correlation-pruned subsets are supported.
- Default research validation uses the first 60 training snapshots as initial
  history and evaluates Jan 2005–Dec 2009, with annual expanding fits.
- Saved full comparison: `0_2_Strategy_Runner/outputs/fixed_models_validation`.
  Thirteen originals plus 52 adjusted portfolios. No classifier beat the past
  loss-frequency control on Brier score or log loss in this experiment.
- The preceding refactor reproduced all 71,631 original logistic probabilities,
  26 books and 26 weight histories exactly. Its 119 checks passed.

## Completed foundation update

- Root AGENTS.md and this docs directory define the handoff/update workflow.
- `research/store.py` provides an SQLite run/metric catalog and complete, typed
  Parquet artifacts. App and CLI load exact saved results first; historical runs
  remain browseable. Full allocation histories restore without model fitting.
- `research/imports.py` upgrades compatible historical CSV reports by replaying
  saved predictions and checking archived books; it does not fit models.
- `fit_policy="expanding"` preserves research behavior. The separate
  `frozen_at_origin` policy fits once before evaluation and ignores all evaluation
  labels when fitting models. It is available in the app and CLI.
- `submission/analysis.ipynb`, `project_report.qmd`, and `presentation.qmd` read
  `submission/release.json`. `python3 build_submission.py` executes the notebook
  and renders HTML without fitting. Shared computation is blocked during rendering.
- The report and slides are rendered under `submission/_output/`. The current
  release pins run `20261108T144257_f45441ba8e83` and an independent benchmark hash.
  This remains the earlier draft: the new feature study has not been pinned into
  the release; tuning and final choice are still pending.
- `--final` checks recorded evidence and rejects an incomplete final build.
- UI alternatives were researched. NiceGUI is recommended for a later prototype;
  the working Streamlit app was retained.

Foundation verification: all **131 checks passed** (15 foundation, 43 original strategies,
73 runner/model/chart/app/storage/submission checks). Additional checks confirmed
the real app loaded 65 cases and historical allocations with computation disabled.
Exact original logistic probabilities and 26 portfolio books still match. Full
artifacts occupy about 32 MiB per 65-case run. No source CSV or original experiment
output was replaced.

The artifact catalog also has three new 65-case feature-comparison runs and a
registered study. Use `python3 -m research --saved` to inspect actual IDs.
The report pin, not the newest run, determines published figures.
Reuse matches complete configurations; cross-configuration prediction reuse is
not yet implemented. Chart filters can inspect saved subsets without computation.

## Completed first feature-selection study

- `research/selection.py` implements cloneable sklearn ANOVA ranking and greedy
  correlation pruning. Learned selection is inside each historical model fit,
  before scaling. Existing no-selection pipeline behavior is preserved.
- `FeatureSpec` serializes method, maximum count and pruning threshold; cache
  identities distinguish recipes. Actual masks/counts, scores, rankings,
  exclusion reasons and full training correlations are saved at every fit.
- `run_feature_selection.py` writes the three-recipe plan before results, uses
  training data only, and holds four models, thirteen rules and alpha=0.5 fixed.
- Study `features_20261108T153403_ddfe05c8` links all-20 run
  `20261108T153503_eba121c8904e`, ANOVA-10 run
  `20261108T153555_a41c41a1c46a` and pruned-10 run
  `20261108T153648_ce8ca5b600a2`. Each has 65 cases; original books match exactly.
- Both selectors used ten inputs at each of five annual fits, with identical
  masks across models at the same origin. ANOVA's union is 12 names; pruning's
  union is 11. Neither union is a single fixed final ten-factor model.
- Results are small and mixed. ANOVA helps logistic/naive Bayes probability
  scores; pruning helps naive Bayes most relative to its all-20 version. No
  model/recipe beats the historical control's Brier 0.249023 or log loss 0.691215.
- The app's **Feature selection** tab reads the registered study without fitting:
  matched predictive/portfolio scores, curves, selection history, audit,
  training-only correlation heatmap and CSV downloads.
- `0_3_Feature_Selection/` has the protocol, findings, executed read-only review
  notebook, HTML and compact exported evidence. The existing Quarto release pin
  is unchanged, pending user review and the later research stages.

Verification: all **143 tests passed** (15 foundation, 43 original strategies,
85 runner/model/selector/chart/app/storage/submission checks). The real saved
study was reviewed with runner computation disabled, including model/selector/
historical-fit changes. The live app is healthy at port 8501. The review notebook
executed under the read-only guard and exported HTML. All four all-20 prediction
tables and all 65 books/weight histories exactly match the preceding saved run.
Hashes confirmed preservation of 111 original files and the recorded input CSVs.
See `0_3_Feature_Selection/outputs/verification.json` for the compact evidence.

## Completed annual consistency / momentum / cash strategy

- `annual_consistency_momentum_cash` freezes a pool with at least three complete
  historical calendar years and strictly more than half positive. Uses only
  earned training labels available at origin; incomplete years are excluded and
  flat years count as nonpositive. The first pool contains 875 of 1,201 stocks.
- Monthly weights buy only strictly positive eligible 12-month momentum. If P
  of N available pool stocks qualify, invest P/N, proportional to momentum, and
  keep the rest in cash. Empty pools/no positive signals mean all cash.
- Cash earns an assumed zero monthly return. A small adapter includes it in
  preserved Part 0 drift-aware accounting/turnover; stock counts and concentration
  exclude cash. Old accounting, source modules and thirteen rules are unchanged.
- Cash-capable ML variants reduce stock weights without renormalizing; freed
  capital stays in cash. Old fully invested ML variants retain exact behavior.
- Saved training-only run `20261108T210046_37d1700ebf97`: fourteen rules, four fixed
  models, seventy cases, earned Jan 2005–Dec 2009, all twenty ML factors, no tuning.
  Original new rule: CAGR 2.34%, Sharpe 0.270, drawdown -19.60%, mean cash 45.93%.
  Logistic variant: CAGR 1.91%, Sharpe 0.270, drawdown -15.56%, mean cash 57.66%.
  These are exploratory trade-offs, not a declared improvement/final choice.
- The app shows annual screen details, cash history, labeled cash in allocation
  heatmaps/composition/downloads, rebalance cash and average cash in comparisons.
  CSV report import also preserves cash accounting. Saved browsing never fits.
- [Experiment guide](../0_4_Consistency_Momentum_Cash/README.md) documents exact
  rules, reproduction, results and outputs. No submission release change.

Verification: **155 distinct tests passed** (15 foundation, 43 original strategies,
97 research/app tests). The initial complete research suite passed 96; the final
12-test cash suite also passed, adding one app test after that discovery (eleven
overlap). All four prior probability tables and all 65 prior books/weight histories
match exactly. The real seventy-case app and older feature study were browsed
with computation disabled. The app health endpoint returned ok. Hash checks
preserved 114 original/reference files, including source CSVs and release pin.
See `0_4_Consistency_Momentum_Cash/outputs/verification.json`.

## Completed momentum-window study

The latest cash-strategy follow-up is complete: study
`momentum_20261108T211612_b4723b7c`, documented in
[momentum findings](../0_5_Momentum_Lookbacks/01_findings.md), compares four windows
on identical Jan 2005–Dec 2009 earned months. Its runs are 2m
`20261108T211647_c1a10def32e6`, 3m `20261108T211720_6c1ba3996026`, 6m
`20261108T211753_acaa1b294fe3`, 12m `20261108T211825_3c40639529d1`.
Each run has equal weight and the cash strategy, originals plus four models
(ten cases). Classifier inputs/settings, predictions, annual pool and control
returns match; only cash-strategy momentum and its history requirement change.

Non-ML CAGR / Sharpe / max drawdown / monthly turnover: 2m 2.63% / 0.290 /
-25.95% / 33.13%; 3m 4.42% / 0.437 / -22.03% / 26.57%; 6m 3.02% / 0.329 /
-22.50% / 18.85%; 12m 2.34% / 0.270 / -19.60% / 12.38%. The shorter window
is not automatically better. Three months leads return/Sharpe here; twelve
leads drawdown/turnover. No final window choice or GridSearchCV was performed.

`RunConfig.cash_momentum_months` defaults to twelve; old configurations load that
default. Other strategies retain their earlier signals. App Comparison has a
saved momentum-study view with model selection, wealth/drawdown/cash, downloads
and an allocation loader. Its sidebar supports custom lookbacks. Inspectors
show the actual selected-window signals; CSV import replays the recorded setting.

Verification: **162 tests passed** (15 foundation, 43 original strategies, 104
research/app). Seven new tests cover short-window arithmetic/eligibility/gaps,
timing, configuration/cache identity, matched studies and saved-only app review.
Real-data artifact-only app review and exact prior twelve-month parity are
recorded in the study's verification output. Source data, report pin and earlier
outputs are preserved. The study is training-only and uses twenty fixed model
factors; it is not a completed final submission experiment.

## Completed evaluation-scope UI clarification

The evaluation-scope UI update uses actual loaded-run metadata for banners and
chart titles. Pending sidebar settings cannot relabel the old result. The Test
results tab reads thirteen non-ML 2010–2018 histories from earlier CSV archives;
overlapping strategies are reconciled, training rows excluded and sources shown.
Registered test runs can be loaded when present; expanding and frozen protocols
are distinguished. At that update, the real library contained ten training-validation
runs and no registered test run. Latest cash/feature/window final test work is
still pending. This presentation-only update preserves calculation code, cache
identities, source files, experiment artifacts and the submission release pin.

Verification: **168 tests passed** (15 foundation, 43 original strategies, 110
research/app). Six new checks cover actual date labels, expanding/frozen policies,
archive reconciliation and wrong-period rejection, unchanged display when editing
settings, and saved test loading with independently labeled training studies.
The real app was exercised with computation blocked. Both saved studies load,
the four momentum runs remain exact cache matches, and hashes preserve 189
input/archive/artifact-catalog/reference files. App health is ok. See
[recorded evidence](verification/evaluation_scope_20261108.json).

## Completed strategy grouping

The strategy-grouping presentation update is complete. Six families cover all
fourteen registered rules; unknown extensions receive Other strategies.
Comparison has a display-only family filter, collapsible family/strategy metrics,
one-row-per-strategy model-change grids, metric selection, detailed inspection,
full tables and CSV export. Changes use percentage points where appropriate and
metric-aware directional shading. The sidebar strategy list starts collapsed
and supports family selection for the next run. Historical test results have
their own family filter. Numerical results and computation code are unchanged.

Verification: **116 research/app tests passed**, with six grouping checks rerun
after the final sidebar refinement. A real headless browser loaded the existing
70-case run `20261108T234216_191b0784406b` and all six families were visually
reviewed. All 280 matrix cells across five metrics match the recorded deltas
exactly. Five current saved runs remain exact cache matches; 189 input/archive/
catalog/reference files are preserved from this turn's baseline. The catalog
has eleven training-validation runs and no test runs; no new run was computed
for this UI change. App health is ok. See the
[verification](verification/strategy_groups_20261108.json) and
[browser preview](verification/grouped_strategy_changes.png).

## Completed five-stage app refactor

The five-stage implementation reuses existing metrics, charts, allocation,
prediction and study renderers with stage-specific options/widget prefixes.
`result_views.py` creates exact case-filtered views without rewriting the source
config/manifest and gates primary final results on the pinned ready frozen run.
Training/test state is independent. Source runs still load complete artifacts;
allocation rendering and historical test reads are opt-in. The earlier seven-tab
composition is replaced, while numerical computation/storage are preserved.

Verification: **121 research/app tests passed**, then the seven stage UI checks
passed after the last presentation refinement. Real saved-data review with
computation blocked checked 14 exact baseline cases, all 70 basic-ML cases,
multi-strategy feature results, unavailable cash-feature evidence and separate
historical test display. Five existing runs remain exact cache matches; all
203 calculation/data/catalog/archive/report hashes match. Five actual browser
screens were reviewed. No research run was computed, no final pin changed, and
the advanced slot/tuning/final test remain pending. See
[evidence](verification/five_tabs_20261108.json).

## Completed basic-ML direction/scope comparison — 2026-11-09

The declared four-cell study is saved in
`0_6_Basic_ML_Directions/outputs/20261109T185258Z`. Runs: loss/pooled
`20261109T185402_0722a7ec725c`, loss/local `20261109T185517_1bf238ea2712`,
upside/pooled `20261109T185621_fe4017faa327`, upside/local
`20261109T185736_3b19b90c1122`. All fourteen rules, same Jan 2005–Dec 2009
validation, three-month cash momentum, twenty factors, strength 0.50 and upside
threshold 0.60. Pooled runs use four models; local runs use logistic only.
No threshold/feature/hyperparameter search or test evaluation was performed.

Each model has 71,631 predictions. Local runs use stock-only fits for 70,792
rows and pooled fallback for 839. There are 5,904 actual stock fits plus five
pooled fallback fits per local experiment; 72 additional fit-summary rows record
fallback assignments. Four new artifacts total 94.3 MiB. The catalog now has
sixteen training runs and no registered test runs. Earlier artifacts are intact.

Common logistic Brier: loss pooled 0.249667 vs local 0.333672; upside pooled
0.250037 vs local 0.334235. Corresponding control Brier is 0.249023/0.249364.
Local probability quality is materially worse; no model in this comparison beats
its historical control on both probability scores. Portfolio effects are mixed;
a favorable local cash-portfolio CAGR alone does not demonstrate prediction skill.
See [full findings](../0_6_Basic_ML_Directions/01_findings.md).

Verification: **138 research/app tests passed**; real saved-data app review with
fitting blocked covered all four views, exact stored metric rows, allocation
sources, logs, legacy study reads and independent first-tab state. All four old
probability tables and seventy matching-setting books are exact; all 1,570 prior
data/artifact hashes match. Four new current-code exact cache matches were
verified. Actual browser review checked pooled/local upside diagnostics and
matched predictive/portfolio scope comparisons. See
[evidence](verification/basic_directions_20261109.json). The report pin remains
the earlier draft; new research does not satisfy tuning/final-test requirements.

Next context: use pooled models as the default research starting point and retain
local results as negative evidence. Review choices before further feature studies,
then chronological tuning and final frozen evaluation. Do not infer Advanced ML.

## Remaining research and submission work

0. Maintain the implemented five-stage composition and strict case/period
   boundaries in TAB_REFACTOR_HANDOFF. The advanced slot stays undefined. The
   next research work is the selected-strategy feature comparison and tuning,
   not another navigation rewrite.
1. Review the saved feature and momentum studies in the app and decide the small
   recipe/window set to carry forward. No final subset/model/window winner has
   been declared.
2. Add GridSearchCV for at least two tunable models, using explicit month-group
   chronological folds inside historical training data. Compare ML scoring and
   portfolio outcomes separately. Never select features before splitting folds.
3. Choose final specifications from training-period evidence. Use frozen training
   model fits for official test evaluation; disclose earlier test inspection.
4. Complete the final professional Quarto narrative and readable analysis
   notebook, then the approximately 15-minute presentation.
5. Consider cross-model probability ensembles later. Learned mixtures/stacking
   require chronological out-of-sample base predictions, not fitted predictions.

## Important limits

No prices, ticker map, market caps, daily returns, spreads or verified publication
lags are supplied. Raw extreme returns remain intact. Books are monthly and
long-only. The new explicitly marked cash rule may be partly or fully in cash;
the earlier thirteen remain fully invested. Sharpe risk-free return, Sortino
target and the assumed cash return are zero; costs are not deducted. The cash
rule's ML haircut reduces exposure; other ML adjustments renormalize. Some
momentum/risk variants concentrate heavily; no position cap was introduced.

The app is a research viewer, not the required deliverable. Report rendering
must load a deliberately pinned saved experiment, not whatever run is newest.

Next context: read this state and TAB_REFACTOR_HANDOFF, then the saved findings.
The navigation work is complete. Continue the separately requested research:
selected-strategy/feature evidence, chronological tuning, justified choices,
frozen final test and Quarto/slides. Keep the advanced slot undefined until the
user describes it. The report release remains the earlier draft; no new research
experiment or final test was performed for the UI refactor.
