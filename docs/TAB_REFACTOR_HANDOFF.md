# App tab refactor — implementation plan and LLM handoff

Latest request (2026-11-10): the subsequent “Okay, can you implement please”
authorized the basket/model/strategy/target flow. It is now implemented over
saved cases, with exceptions for nonpredictive/method views. See
[BASKET_FIRST_UI_PLAN.md](BASKET_FIRST_UI_PLAN.md) for actual scope, coverage
constraints, verification and next steps. The older planning-only instruction
is superseded. Seven main tabs remain; numerical results and the publication
are unchanged by this navigation update.

Feature-engineering addition authorized 2026-11-10: seven main stages, with
**Feature engineering** between Tuned model and Final test. Its methods, model/
portfolio comparisons and hyperparameter tuning are subtabs; it clusters factor
columns, while Grouping With ML clusters stocks. The extension's fixed/tuned
frozen variants have a separate Final test section. Earlier six-tab/no-Ward-final
statements refer to the earlier publication. See [current protocol](FEATURE_ENGINEERING.md).

Current tuning/final authority (2026-11-10): Tuned model no longer uses the
four-portfolio-family focus; all fourteen saved basic rules are available by
default. Features and Final retain their own display filters. The current final
study is explicitly fixed K-means/ANOVA-10, separate from tuning and Ward training.
See [latest review](TUNING_AND_FINAL_REVIEW.md). Older focused-equal-weight-only
statements below record the prior UI.

Latest authority (2026-11-10): the user requested and authorized a second
unsupervised grouping method. Tab 3 is now **Grouping With ML**, with K-means,
Hierarchical clustering (Ward) and Adding a direction. Both grouping methods
feed the same fixed strategies/overlays. The current hosted publication includes
both, with later feature/tuning/final sources still K-means. This supersedes
older two-subtab/advanced-placeholder instructions below. See
[handoff](GROUPING_WITH_ML.md).

Current: **all six main tabs are deploy-ready** via `published/project`.
Features includes both registry and advanced-book studies, descriptive factor-bar
and recipe-curve hovers. Tuning reads all saved model/target trials and holdout
results. Final reads precomputed saved test/training/entire-path windows and
optional archived comparisons. No runtime CSV/catalog/fitting/replay is allowed.
Features/Tuned/Final default to the four portfolio-family focus; focus off enables
the wider saved comparisons. Advanced families have no tuning study, so tuning
focus honestly has equal-weight coverage only. See [DEPLOYMENT.md](DEPLOYMENT.md).
Partial-deployment notes below are historical; formal submission pin is unchanged.

Latest Advanced ML refactor: two subtabs, **Build the grouped strategy** and
**Add up/down predictions**. Explain/visualize monthly K-means membership first,
then positive momentum/cash, then top-ten positive-vol screening; isolate the
directional models and adjusted comparisons in the second subtab. The first three
main tabs are deploy-ready through `published/advanced_ml`. The final three remain
pending. Read [basket presentation details](BASKET_MOMENTUM.md) and
[deployment handoff](DEPLOYMENT.md); the one-/two-tab notes below are historical.

Latest: Basic ML now joins Strategies in the self-contained deployment mode,
using `published/basic_ml` by default. Three independent collapsed panels contain
prediction quality, portfolio results and changes from originals in both modes.
Both directions, all four pooled models and the per-stock logistic pilot are
published with matched scope comparisons, lazy weights and lazy fit audits.
Other deployed tabs remain pending. See [current deployment handoff](DEPLOYMENT.md)
before continuing to Advanced ML; the earlier first-tab-only notes below are
historical. No original experiment or formal submission pin changed.

Latest incremental refactor: **Strategies** now has a self-contained published
view as well as local research mode. Both show training-only scope and strategy
name tooltips. Deployment opens only its selected publication; all other tabs
are pending and do not execute archive reads. Continue from
[DEPLOYMENT.md](DEPLOYMENT.md), extending one tab at a time and preserving common
renderers and exact saved results. This supersedes the old hosting deferral;
no actual online deployment or formal submission repin has occurred.

Latest tuning UI: three top controls (**Model**, **Basic strategy**,
**Prediction target**). Four model types × separately trained up/down are saved
on the same ANOVA-10 / 2007–2009 holdout. Model scores precede portfolio effects;
hover the Brier/log-loss/ROC-AUC headers. Historical control shown once; unchanged
winning parameters explained. Older downside-only prose below is historical.
See [current protocol and study IDs](TUNING.md). No main tab reordered and no
formal notebook/QMD/release change in this addition.

## Status and authority

Latest authorized addition: Advanced ML offers the original top-positive-vol
momentum book plus separate downside-to-cash and upside-relative-weight overlays.
Four saved pooled all-20 classifiers per direction, no tuning/selection. Select
the predictor above the results and an adjusted Basket book for stock weights
and probabilities. See [protocol](BASKET_MOMENTUM.md). Tuned model's introduction
now explains downside/log-loss tuning and its separate frozen ANOVA-10 holdout.
These additions supersede the older undefined-advanced descriptions below.

Current additions: **Advanced ML** is the correlation-basket momentum book in
[BASKET_MOMENTUM.md](BASKET_MOMENTUM.md). It reads
`artifacts/basket_studies/` and does not join the fourteen-rule registry.
A second book keeps the 10 highest positive-vol names in each group and
trades 1- and 2-month momentum.
**Tuned model** sits between Feature selection and
Final test, reading the separate saved ANOVA training holdout described in
[TUNING.md](TUNING.md). Feature selection also has a completed four-recipe
forest study shared with notebooks/QMD/slides; see
[forest implementation](RANDOM_FOREST_FEATURE_PLAN.md). Preserve both additions.
The original five-stage refactor below records the earlier composition.

**Implemented after the subsequent user request: "Implement please".**

The five-tab UI now composes existing renderers with `result_views.py` case
filters. Training and test selections use separate state. The advanced tab
now renders the saved basket study. Feature/tuning gaps and final readiness remain explicit.
The earlier tab refactor did not change the fourteen-rule experiments or the report release.
The original design below is retained as a reference; it is no longer a pending
navigation task. Existing source runs still load complete artifacts once per
selection; allocation plots are opt-in, not a new partial-file storage system.
This document records the latest requested navigation and supersedes the tab
sequence and advanced-strategy suggestions in PROJECT_PRESENTATION_PLAN.md.
That older document remains useful background for shared evidence and reporting.
App composition and view helpers changed. Scientific backend code, experiments,
the report pin and the undefined advanced strategy remain unchanged.

Read AGENTS.md, PROJECT_STATE.md and this file for future maintenance. The UI
refactor does not authorize new feature/model experiments or an inferred advanced
strategy. Resume those research tasks separately when requested.

## Subsequent authorized Basic ML extension — 2026-11-09

The user separately requested upside prediction with threshold-based heavier
allocation and comparison of pooled versus per-stock models. That extension is
implemented within Basic ML; it supersedes the loss-only content restriction
below. There is no new main tab or advanced strategy. See
[implementation and protocol](../0_6_Basic_ML_Directions/README.md).
Basic ML now selects its own saved experiment without changing Strategies state.
The original plan below documents the earlier refactor, not a prohibition on the
newly requested target/scope experiments. Final test and the report pin remain
pending/unchanged.

## Subsequent Tuned model tab — 2026-11-09

A sixth main tab, **Tuned model**, sits after Feature selection and before
Final test. It reads a saved training-holdout comparison: no ML, fixed ML, and
tuned ML on the feature list frozen at the December 2006 selection fit.
Reported returns are January 2007–December 2009. The 2010–2018 file is not
opened. Browsing does not fit. See [TUNING.md](TUNING.md).

## Subsequent Feature selection split — 2026-11-09

Feature selection retains its main-tab position and now contains two subtabs:
**Selected features & method**, then **Effect on ML models**. The first presents
actual names, method, audit and historical training evidence. The second
explains that each recipe refitted the model independently, then compares its
predictive and ML-adjusted portfolio outcomes. Non-ML strategies are unchanged
references; feature selection does not apply to their rules. The selected-model
control is shared. Unsupported portfolio choices cannot hide the feature audit
or predictive scores. The tuning-helper status moves into the second subtab.
This reuses the saved study; no new fit or experiment is part of the UI split.

## Requested five-tab story

| Order / tab | Dataset and visible content | Exclusions / status |
| --- | --- | --- |
| **1. Strategies** | Training-period non-ML strategies, grouped by the existing six families. Explain rules; compare return, volatility, Sharpe/Sortino, drawdown, turnover, cash and allocations. Include the non-ML momentum-window study. | No ML versions, models, probabilities, ML-change grids or ML controls in this tab. |
| **2. Basic ML** | Training-period loss-probability models; their prediction quality; the existing simple weighting adjustment; matched original versus ML portfolio results and changes. | No new ML algorithm or advanced strategy is needed. Preserve current numerical behavior. |
| **3. Advanced ML strategy** | Reserved placeholder for a future user-defined strategy. | The user does not want to explain, design or implement this strategy yet. |
| **4. Feature selection** | Select implemented strategies and models; compare saved factor recipes on training validation; explain selected factors and measure predictive and portfolio effects. | Missing comparisons stay visibly unavailable. No invented improvements or automatic fitting on interaction. |
| **5. Final test** | Later, apply the chosen training-derived specifications to the 2010–2018 test period and compare original, basic ML, selected-feature and any subsequently implemented advanced variants. | Final selected/tuned frozen-model test evaluation is pending. Historical test evidence is secondary and explicitly labeled. |

The advanced placeholder does not block using Feature selection with existing
strategies. Do not fill tab 3 with annual-consistency/cash, Sharpe/Sortino,
ensembles or another inferred design. Existing complicated **non-ML** rules
still belong in tab 1; the standard probability overlay on them belongs in tab 2.

Interpret the user's final "test training" tab as **evaluation on the test
period after training choices**, rather than training a model on test labels.

## Evaluation conventions

Tabs 1, 2 and 4 use training-period evidence. Preserve the existing default
**Jan 2005–Dec 2009 training-validation** window after initial 2000–2004 history.
Do not switch to scores on a model's own fitting rows, or silently extend one
strategy's comparison window. Non-ML weights and model forecasts use information
available at each historical rebalance. Show each view's actual recorded dates.

The full training dataset earns Jan 2000–Dec 2009. Default validation is a later
portion of that dataset; it is not the final 2010–2018 test period. A full-training
non-ML chart, if later requested, needs its own labeled window and comparable
warm-up rules. It is not necessary for this UI refactor.

The final test model policy is `frozen_at_origin`: train/select/tune from allowed
training history, fit before test evaluation, and do not refit ML on test labels.
Lagged non-ML signals can update as returns become known under existing timing
rules. Keep earlier exploratory expanding-test fits distinct. The test period
was already inspected; never describe it as untouched.

## What "Basic ML" means here

Use the existing four classifiers and the target `R1M_Usd < 0`. Their task is to
estimate a stock's probability of a negative next-month return. First explain a
simple equal-weight example, then allow the user to inspect selected existing
strategies. Reuse the existing historical loss-frequency prediction control.

Present two separate questions in tab 2:

1. **Does the model predict downside usefully?** Reuse Brier score, log loss,
   ROC AUC, accuracy at the documented threshold, and calibration. Explain score
   direction and the probability control. Do not equate these with investment
   performance or promise improvement; existing predictive evidence is weak.
2. **What changes when its probabilities affect portfolio weights?** Show the
   original and adjusted portfolios over identical months, their actual scores,
   the existing grouped change grids, cash/exposure and allocation differences.

The existing adjustment is:

```text
raw adjusted stock weight = original weight × (1 − strength × probability of loss)
```

For the thirteen fully invested rules, existing code normalizes these weights
back to 100% stock exposure: deallocation is relative between stocks. For the
explicit cash-capable rule, reductions remain in cash. Explain this difference;
do not silently change either policy while moving the UI. Moving every haircut
to cash for all rules would be a separate trading-rule experiment, not a layout
refactor. The current default strength is 0.5.

## Reuse map: move composition, keep computation

| Existing component | Reuse / small adaptation |
| --- | --- |
| `app.py::render_comparison` | Split its composition into baseline and basic-ML stage functions. It currently mixes absolute metrics, ML curves and changes. |
| `strategy_groups.py` | Keep the six families and lossless change-table pivot. |
| `strategy_results_ui.py::render_metric_groups` | Add an explicit case/variant filter so tab 1 receives baseline cases only. Keep the renderer and styling. |
| `strategy_results_ui.py::render_paired_groups` | Use in tab 2, with the training/strategy/model selection. Do not render in tab 1. |
| `app.py::wealth_chart` and `app_charts.py` | Reuse explicit case keys for baseline-only or paired plots; reuse allocation histories and diagnostics. |
| `app.py::render_models` | Move prediction scores/calibration into tab 2. Keep fitting details in a local expander. |
| `app.py::strategy_guide`, `explanation`, `render_data` | Make these contextual help sections instead of main navigation tabs. Suppress the currently unconditional ML explanation in tab 1. |
| `app.py::render_allocations` | Show baseline allocations within tab 1 and paired allocations within tab 2. Add view parameters and unique widget-key prefixes. |
| `momentum_study_ui.py` | Reuse `momentum_figure(..., model_id="none")` in tab 1. Adapt the wrapper to hide ML controls there; offer ML versions in tab 2 if desired. |
| `feature_study_ui.py` and `research/feature_studies.py` | Reuse tab 4's saved study reader, checks, scores, masks, correlations and curves. Extend selection to multiple supported strategies without copying the study calculation. |
| `evaluation_ui.py`, `research/evaluation.py` | Reuse period/protocol banners and historical archive reads in tab 5. Make the pending final experiment its primary status. |
| `research/runner.py`, `data.py`, `models.py`, `selection.py`, `adjustments.py`, `accounting.py`, `store.py` | Preserve scientific behavior and storage. A navigation change does not require rewriting these. |

Start with five small stage-composition functions in the app and parameterize
existing renderers. Add a small pure case-filter helper only where useful.
Do not build a second runner, duplicate accounting/model pipelines, create a
generic workflow engine, migrate the UI framework, or introduce a new database.
The broader cross-run presentation design can develop incrementally from the
existing case metadata and study links; it need not block this tab reorganization.

## View filters and state boundaries

Use explicit `(run_id, case_id)` references; case IDs repeat across runs. A view
filter selects existing cases and their corresponding metrics, books and weights.
Never rewrite a stored config or set `include_ml=False` on a loaded ML run to
pretend it was computed differently. Its baseline cases can be shown in tab 1
without recomputing a baseline-only experiment.

Before calling a tab renderer, check:

- Training tabs accept only `evaluation="training_validation"` runs/studies.
- Tab 1 accepts only `variant="baseline"` cases and does not call ML renderers.
- Tab 2 pairs each adjusted case with its own original, matching dates and rules.
- Feature comparisons vary the declared recipe while holding other intended
  components fixed; use the existing study validation checks.
- Final test primary results require the frozen fitting policy, explicit final
  specification selection and recorded readiness evidence. An expanding test
  run belongs in historical/exploratory evidence, not the final-results slot.

Replace the single shared `session_state["result"]` navigation assumption with
separate training and test selections/results. The current global period radio
and "load any newest run" fallback must not make tab 1 display test results.
An absent compatible run should show an honest empty state. Reuse store reads.

Keep a shared selected-strategy list for training comparisons, with per-tab
model/recipe controls as appropriate. Put model settings in tab 2 and feature
settings in tab 4; keep their compute actions explicit. Remove the global ML
toggle/model count/ML settings from the pre-tab introduction so tab 1 teaches
only non-ML concepts. General data context, dates and provenance can remain shared.

Every renderer reused in multiple tabs needs a distinct `key_prefix` for widget
state. Streamlit executes tab bodies even when they are not selected: tab
navigation must never train models, search parameters or replay backtests.
Use saved results and cached read-only views, with full weights loaded on demand.

## Feature selection, tuning and missing evidence

The first feature study covers the earlier thirteen rules, with all twenty
factors, ANOVA ten and correlation-pruned ten. It predates the annual-consistency
cash rule. The momentum study covers that cash rule at 2/3/6/12 months using all
twenty factors. These are separate studies, not a completed crossed experiment.

For unsupported selected strategies/recipes, show "No saved comparison yet"
and the evidence that is available. Defining and running the missing matched
experiments is later research work; never fabricate or silently compute them
as part of moving the UI. Preserve selector timing and the at-most-ten actual
inputs per fit. Annual selected-feature unions are not a single final subset.

Hyperparameter tuning remains required by the assignment but unimplemented.
Propose a later **Tuning** subsection within the training research area of tab 4,
before Final test, without adding an unrequested sixth main tab. It should show
declared grids, chronological folds, scores, chosen parameters and runtime from
training-only GridSearchCV for at least two tunable models. This is future scope;
the tab refactor itself must not implement a search or invent tuning results.

## Report, slides and saved research

Use the same five-part progression for the Quarto narrative and slides, with
advanced content omitted or marked pending until the user defines it. Keep
data/method/code details in contextual sections or appendices. The app remains
an exploration tool; readable code, notebook, QMD/HTML and slides remain deliverables.

Keep table definitions, case selection and numeric chart data shared. Interactive
and publication renderers can differ, but should not repeat calculations. Later,
extend the existing release specification to pin the appropriate training runs,
feature/tuning studies and final test run explicitly, preserving old single-run
release compatibility. Do not repin `submission/release.json` during this planning
or navigation work. Report rendering continues to load artifacts without fitting.
Online hosting remains deferred until research and deliverables are finished.

## Proposed implementation sequence

1. **Define case visibility and separate training/test state.** Extract small
   filters over existing metadata; retain saved-run provenance and reuse caches.
2. **Assemble tabs 1 and 2 from existing renderers.** Parameterize variant/case
   filters, contextual explanations and widget keys. Keep existing grouped
   layouts. Verify no ML content in tab 1 and matched ML diagnostics/results in 2.
3. **Add tab 3's placeholder and move feature evidence to tab 4.** Do not define
   advanced behavior. Support a selected strategy set and clear availability.
4. **Move final test to the last tab.** Primary final-results slot remains pending
   until appropriate saved evidence exists; archives remain secondary. Eliminate
   cross-period changes to training tabs from global selection or callbacks.
5. **Verify parity and presentation.** Exercise real saved data with computation
   blocked, update UI tests to inspect tabs by label rather than global chart/table
   offsets, review the browser, and update handoff/change log. No fitting should
   be required for a successful navigation refactor.
6. **Resume research separately.** Complete the selected feature/strategy studies,
   tuning and justified choices, then frozen test evaluation, Quarto and slides.
   The advanced branch waits for its own user specification and can remain pending.

## Acceptance checks for a future implementation

- Tab 1 shows only training-period originals, including baseline-only allocation
  and momentum views. No probability table, adjusted series or ML controls leak in.
- Tab 2 contains prediction quality/control/calibration and exact original/ML
  portfolio pairs for the same training window; metric direction and cash policy
  are explained. Previously recorded differences remain numerically identical.
- Tab 3 is a placeholder and triggers no strategy/model calculation or invented
  choice, even when all Streamlit tab bodies execute.
- Tab 4 shows supported selected strategies and multiple saved feature methods;
  unsupported combinations and missing tuning are clearly marked pending.
- Changing a test selection cannot change training-tab cases or their dates.
  Historical archives/expanding fits cannot masquerade as final frozen results.
- Model fitting/search/replay patched to raise still allows all saved-result
  navigation, downloads, filters and plots. Existing numerical regression checks
  retain their recorded expected values; only UI composition expectations change.
- Browser review confirms the five-tab story, group clarity and period labels.
  A notebook/report reader can resolve the same selected evidence without fitting.

## Starting evidence for the next LLM

Read the current state for full history. Useful existing references are:

- All-fourteen/all-four-model training run: `20261108T234216_191b0784406b`.
- Feature study: `features_20261108T153403_ddfe05c8`; see
  [feature findings](../0_3_Feature_Selection/02_findings.md).
- Cash momentum study: `momentum_20261108T211612_b4723b7c`; see
  [window findings](../0_5_Momentum_Lookbacks/01_findings.md).
- [Evaluation periods](EVALUATION_PERIODS.md), [grouping](STRATEGY_GROUPS.md)
  and [storage](SAVED_RESULTS.md) describe the implemented pieces.
- Earlier grouping verification: 116 research/app tests passed, six grouping
  checks rerun, and real-browser review; see
  [evidence](verification/strategy_groups_20261108.json).
- Submission release remains the earlier single-run draft
  `20261108T144257_f45441ba8e83`; tuning and final choice/test evidence are pending.

The app now has the five tabs above. Verification includes 121 passing
research/app tests, final UI checks, saved-only real-data review and browser
screenshots. See [implementation evidence](verification/five_tabs_20261108.json).
Scientific computation and the original report pin are unchanged.
