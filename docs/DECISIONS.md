# Design decisions

## 2026-11-14 — Ignore local assistant state; retain shared instructions

Use targeted `.gitignore` paths for assistant sessions, histories, caches and
private settings/instructions. Keep tracked `AGENTS.md` and allow shared
CLAUDE/GEMINI instructions, assistant configuration, skills and rule directories.
Avoid blanket assistant-directory or Markdown exclusions that would hide shared
project guidance. See [Git staging scope](DEPLOYMENT.md#github-staging-scope).

## 2026-11-10 — Implement the shared selection flow over saved cases

The subsequent user request authorizes the previously planning-only navigation.
Keep seven main tabs and use basket, prediction model, allocation strategy and
up/down target for portfolio views. Use fewer controls for nonpredictive and
method lessons. Explicitly describe existing bundled strategies instead of
rebuilding a universal weighting engine. Keep single-case defaults and existing
Compare more controls. Offer only evaluated tuples and never silently borrow
another scope/target/recipe. Basket selection does not retrain pooled models on
only those assets. Final comparisons preserve protocol, source and exact dates;
their list is session-only and resets across publication/mode changes. Numerical
results, research APIs and formal deliverable pins stay unchanged by this UI work.
See [implementation and handoff](BASKET_FIRST_UI_PLAN.md).

## 2026-11-10 — Plan basket-first navigation; keep the running app intact

The user explicitly requests planning, not implementation. Record a tab-by-tab
proposal for basket, prediction model, allocation strategy and prediction target.
Use fewer controls for nonpredictive/method views. Recommend a Basic ML pilot
with adapters over existing saved cases; do not imply unsupported cross-products
or retrain models on basket membership as part of rearranging the UI. The proposal
is not an accepted implementation commitment. See
[plan and LLM handoff](BASKET_FIRST_UI_PLAN.md).

## 2026-11-10 — Test acceleration as an explicit fixed cash rule first

Preserve the original top-ten grouping strategy. Equal weights define its group
return signal; invested member weights are proportional to positive momentum.
Acceleration is the change between two already-earned monthly returns. For basket
acceleration use current selected membership for both months; cluster IDs are not
persistent identities. Compare original, stock-only, basket-only and combined
negative-acceleration haircuts (50%, declared before computation). Multiply existing
positions, retaining cash rather than shorts or renormalization. Keep training and
the already-viewed test in separate app sections. Save the signal audits for later
ML research, but do not claim to have retrained classifiers with them. Publish only
display frames; preserve prior results, independent notebook work and formal pins.
See [complete protocol](MOMENTUM_ACCELERATION.md).

## 2026-11-10 — Name the full K-means momentum/cash rule in Final

The original strategy was included but hidden behind abbreviated book and subtab
names. Make K-means, top-ten positive-volatility screening, actual momentum window
and cash explicit. Feature-engineered final comparisons start from K-means top ten
by default in new sessions. Preserve saved book/version keys and all values; do
not rerun models or relabel a training CAGR as a test CAGR.

## 2026-11-10 — Feature-engineering methods get their own end-to-end stage

The user authorized integrating the 0_7 notebook, new input/model variants,
hyperparameter searches, frozen final results and deploy/local publication.
Use a new Feature engineering main tab after Tuned model, with methods/results/
tuning subtabs. Factor-column clustering differs from stock grouping, so keep
those concepts in separate teaching stages. Reuse the existing four classifier
factories, weight adjustments and accounting; do not fork the older model runner.
Declare six input recipes and small classifier/recipe grids before computation.
Refit every learned feature inside chronological folds, then freeze features and
classifiers at each evaluation origin. Carry fixed and trained-CV-selected settings
separately to final evaluation, without test retuning. Publish only display frames
and audits; keep estimator/probability panels local. Preserve original notebook/
CSV evidence and record review corrections. Ten model inputs can use twenty source
factors; disclose that distinction instead of claiming automatic rubric compliance.
See [design, review and handoff](FEATURE_ENGINEERING.md).

## 2026-11-10 — Expose saved tuning rules and distinguish final evidence

Remove the four-family filter from Tuned model: show all fourteen saved basic
strategies, not only equal weight. Retain feature/search protocol and separate
up/down targets. Explain that the search optimizes predictive log loss and the
ML overlay creates a different portfolio from its original equal-weight reference.
Do not silently run clustering tuning, advanced tuned-forecast replay or new
final tests. Label those as separate missing experiments. Final displays fixed
K-means evidence, log-return views and changes from correct originals; local
formal-release pending is separate from available test results. Keep published
results and earlier calculations exact. Existing Git ignore rules need no changes.
See [handoff](TUNING_AND_FINAL_REVIEW.md).

## 2026-11-10 — Compare grouping methods with fixed trading rules

Rename Advanced ML to Grouping With ML and use three subtabs: K-means,
hierarchical clustering, directional adjustments. Choose Ward as a second
unsupervised method, using identical standardized return paths and five groups.
Save the actual tree rather than refitting to draw it. Preserve the K-means
numerical default and all shared momentum/positive-volatility/cash math.
Reuse existing separate up/down forecasts for both starting universes; prediction
scores stay the same while portfolio weights can differ. Publish Ward as an
optional lazy method record, preserving every prior exported frame. Keep feature,
tuning and frozen test sources K-means until new experiments are explicitly run.
See [implementation/handoff](GROUPING_WITH_ML.md).

## 2026-11-09 — Track source and the current hosted release

Use the project-root `.gitignore` to omit datasets/CSV research reports, the full
experiment store, older published exports and generated/local files. Explicitly
retain `published/project` Parquet/JSON and source dependencies, because hosting
must work without raw data or the research archive. Keep notebooks, Quarto source,
documentation and the theme available for version control. Ignore rules change
Git's discovery only; they do not remove research files or create/push a repository.

## 2026-11-09 — Finish six-tab publication without inventing research

Export both registry and advanced feature evidence, all saved tuning studies/trials,
and the existing frozen final study. Preserve independently refitted recipe models,
fixed trading rules, dates and original source IDs. Keep classifiers separate from
feature selectors and K-means. Explain factor abbreviations in graph hovers and
show the full-group original beside its adjusted feature curves.

Interpret “four main models” as the four main portfolio families; put an on-by-default
focus checkbox on Features/Tuned/Final, keeping classifiers available and extra
saved comparisons opt-in. Foundation tabs keep their teaching examples. No advanced
tuned result exists: focused tuning exposes equal weight and says so, rather than
relabeling basic-strategy tuning as advanced research. This scope was a stated
assumption while optional clarification was pending.

Precompute final plot windows once from saved monthly returns at export, preserving
the expanding-all-20 training versus frozen-ANOVA-10 test distinction. Publish old
non-ML archive summaries as Parquet; omit fitted tuning objects and raw stock
probability panels. Rendering requires no research store, CSV, fitting or replay.
Preserve all older publications and the separate formal submission release.

## 2026-11-09 — Teach grouping before adding up/down predictors

Use two subtabs within Advanced ML, keeping the main progression unchanged.
Show how unsupervised K-means creates monthly memberships, then apply the saved
positive-momentum/cash rule and introduce the top-ten positive-volatility screen.
Keep all directional scores/controls/results in the second subtab, where the
starting full/top-ten book and its momentum horizon are explicit. Preserve all
saved numerical results and separate up/down rules; invent no grouping-only
portfolio backtest or combined forecast rule.

Illustrate past input patterns from saved one-month signals without refitting or
reading raw CSVs. Use current memberships, exclude incomplete/constant paths and
show coverage; temporary cluster IDs are not persistent sectors. Publish the
existing advanced study with lazy detail partitions, preserving earlier releases.
Do not run new strategy/model research just to explain or host the saved results.

## 2026-11-09 — Publish Basic ML with three collapsible questions

Use three separate default-collapsed panels for probability quality, absolute
portfolio results and changes from the originals, for both event directions.
Keep current science and chart defaults. Avoid nested expanders on Streamlit
1.32 by using group containers and optional settings/details inside each panel.

Publish the existing matched four-cell direction/scope study, with four pooled
models and the per-stock logistic pilot. Retain all twenty original inputs and
fixed settings; do not train additional local models for deployment. Use a small
read-only store adapter to reuse existing chooser/provenance comparison code.
Publish aggregate probability/calibration/count diagnostics, lazy target weights
and lazy fit audits, omitting raw probability panels and fitted objects. Preserve
the first publication; make the new two-tab release the default. Deduplicate
identical Parquet content. Leave later deployed tabs explicitly pending.

## 2026-11-09 — Publish a read-only first tab without forking the app

Use explicit research/deploy modes, not automatic environment detection. Keep
local research and its compute buttons; deployment reads an immutable curated
release and cannot fall back to the research archive or computation. Start with
Strategies and mark other deployed tabs pending rather than pretending the full
hosting refactor is done. Reuse existing chart/table/weight renderers, baseline
frames and accounting. Save per-case target weights separately for lazy loading.
Source CSVs, stock-month predictions and fitted objects are unnecessary for this
first tab. Include the saved baseline lookback comparison, but retain other
lookbacks' full allocation audits locally.

First-tab training-only scope is both prominent text and a validated data
boundary. Use escaped HTML for these small results tables because the existing
Streamlit canvas cannot provide cell-specific strategy tooltips; keep the shared
metric formatter and add keyboard-focus support. Publication provenance is
separate from `submission/release.json`. See [handoff](DEPLOYMENT.md).

## 2026-11-10 — Freeze the four books once for 2010–2018

The official pass uses the training-window recipe: equal weight, the top-10
positive-vol book, the full basket book, and the same two up-or-down
adjustments. ANOVA top 10 is chosen inside the single fit at 2009-12-31.
The four classifiers stay at factory settings, because the grid search was
not run on these books. Strength 0.5 and upside threshold 0.6 stay as they
were on Advanced ML. Save the result outside the experiment catalog. Leave
the submission pin alone. After the decade is scored, keep this recipe.

## 2026-11-10 — Same up-or-down overlay on the full basket book

Keep the top-10 overlays. Add the same two rules, with their own version
names, on Basket momentum + cash at 1, 2, and 3 months. Downside haircut
stays in cash. Upside tilt keeps the original equity budget. Do not blend
the two directions into one weight. Do not register a fifteenth strategy.
Score it on the training window already used by Advanced ML. Leave
2010–2018 closed until the four-book choice is written down.

## 2026-11-09 — Make tuned comparisons explicit by model, rule and event

Let users choose any of the four basic classifiers, one fixed basic rule and
next up/down. Train independent event models; retain log loss as the grid-search
score. Reuse the same frozen downside ANOVA-10 inputs across directions, clearly
labeling this rather than implying an upside feature search. Preserve original
downside evidence, add only the missing two downside/four upside model searches,
and filter saved evidence before display. Show prediction quality before
portfolio effects, explain unchanged winners, and show the historical control
once. Metric header tooltips describe probability quality, not portfolio success.
Preserve release/test status; browsing never trains or replays.

## 2026-11-09 — Directional predictions on positive-volatility baskets

Preserve the grouped top-10 screen and its existing momentum weights. Compare
separate downside-to-cash and upside-relative-weight overlays at 1 and 2 months
for all four classifiers. Strength 0.50; upside threshold 0.60. Excluded and
non-positive-momentum names remain at zero. Upside preserves baseline cash.
Reuse saved annual pooled all-20 forecasts without new training. Keep this
exploratory study separate from selected-feature tuning, the fourteen-rule
registry and final submission/test. No combined overlay is added.

Explain that Tuned model optimizes classifier log loss on the saved frozen
ANOVA-10 list; choosing a strategy does not tune its trading rule. Browsing
either tab must not fit models or replay books. Preserve the formal release.

## 2026-11-09 — Shrink each basket to its 10 highest positive-vol names

After K-means, keep up to 10 names per group by positive volatility. The
measure is `sqrt(mean(max(return, 0)^2))` on the same 12 earned months that
formed the groups. Down months contribute zero and stay in the average. Apply
the existing positive-momentum cash rule to that smaller set at 1 and 2 months.
Scale the book to the selected names, so a name that misses the screen is out
of the book. Keep the full-group 1-, 2-, and 3-month books.

## 2026-11-09 — Hold the baskets with 1-, 2-, and 3-month momentum

The 12-month return path stays the input to K-means, because one month cannot
describe co-movement. The weight and the cash decision use a separate signal:
the last one, two, or three earned months. Show all three windows. The earlier
12-month holding study remains saved and is not the rule the tab leads with.

## 2026-11-09 — Group co-moving stocks, then momentum, with cash

The Advanced ML strategy is a monthly K-means on trailing return paths, then
the positive-momentum cash rule inside each group. A negative group is cash
even when some members are individually positive. Compare it with equal weight
and with the same cash rule without groups, on the January 2005–December 2009
training window. Leave the fourteen registered rules and the official test
file alone. Cluster ids are not treated as persistent sectors.

## 2026-11-09 — Execute forest selection with shared saved evidence

The user's "Preform the plan" authorizes implementation and the new study.
Add a fixed forest-ranked top-ten recipe inside the existing training pipeline;
keep selector settings independent of downstream predictor settings. Record
importance means, tree SD, exact masks and training availability. Compare four
recipes under one current calculation version, with all fourteen rules, pooled
downside, annual fits, three-month cash momentum and strength 0.50.

Preserve historical studies, main release pin and earlier submission files.
Add an optional checksummed feature-study release reference, shared read-only
presentation helpers and canonical notes for app/notebook/QMD/slides. The study
has mixed results and declares no winner. It is distinct from the separately
saved ANOVA tuning holdout; advanced strategy and frozen final test stay pending.

## 2026-11-09 — Tune on an inner split of the training file

The Tuned model tab reports January 2007–December 2009. Settings and the
frozen ANOVA column list come from labels known at December 2006. The official
test file stays closed until the final tab. The comparison is three books:
the original rule, the same rule with the factory model, and the same rule
with the grid-search winner. Portfolio return is not the search score.

## 2026-11-09 — Plan forest selection across all deliverables

The user requested a plan to add random-forest feature selection and its
importance visualization, explanations, method comparisons and reusable
app/QMD/notebook/slide evidence. Record that plan in
[RANDOM_FOREST_FEATURE_PLAN.md](RANDOM_FOREST_FEATURE_PLAN.md). Propose a
training-fitted forest top-ten recipe alongside all-20, ANOVA and pruning;
independently retrain all four predictors with fixed trading rules. Reuse the
existing runner/store, add shared read-only presentation helpers and a small
optional study release reference. No new study, model, UI or release change is
claimed. Implementation and experiment execution are subsequent work.

## 2026-11-09 — Basic ML charts start with all strategies

Use every currently available strategy as the Basic ML performance-chart default.
Keep user selections after initialization and respect family/strategy filters.
Other stage defaults and numerical calculations are unchanged.

## 2026-11-09 — Feature inputs/process before retrained-model effects

Split the existing Feature selection main tab into two nested subtabs, preserving
its saved-study/model controls. Show selected inputs before the detailed audit
and stability charts. Put independent-retraining provenance, prediction quality,
ML-adjusted portfolio comparisons and tuning status in the second subtab.
Selection applies only to model inputs. Keep non-ML rules unchanged, shown only
as reference curves; do not imply feature selection was applied to them.

Do not refit to reorganize the app. The saved study already fitted each recipe's
models at each historical origin. Browsing reads that evidence. Feature/process
inspection and predictive quality must remain visible when no compatible
portfolio is selected. Missing newer strategy/target/scope studies remain
explicitly unavailable rather than being computed automatically.


## 2026-11-09 — Grid search is a separate saved pipeline

Keep experiment artifacts as predictions and provenance. Tuning writes its own
directory with the refit sklearn pipeline, because the demo has to reload the
model without fitting. Folds are expanding month blocks with the existing
one-month label lag. The search score is a probability score. A fit cap of 40,
and a refusal of the 100-tree forest unless a smaller forest is explicitly
fixed, keep the first run short. Do not run it until the feature recipe and
the correlated-basket strategy are chosen, and do not treat a tuning directory
as the official test.

## 2026-11-09 — Separate prediction target, training scope and portfolio rule

Add a strictly-positive next-month target as a second Basic ML pattern. Use a
fixed, configurable probability threshold (initial 0.60) and proportional boost
(initial multiplier 1.50) before normalizing to the original equity budget.
Preserve existing cash and zero-weight exclusions. Preserve the downside rule.

Support optional per-stock fits with stock-only preprocessing/estimation and
same-origin pooled fallback for fewer than 36 known months or six examples of
either event class. Keep the pooled past-frequency control identical across
scopes and score every evaluation row. Start the local comparison with logistic
regression; existing tree defaults were designed for pooled training.

Record predictions and fit provenance, not serialized estimator objects. Keep
Basic ML result state independent from the non-ML tab. Saved scope comparisons
require matched target, trading protocol, factors and source/code/runtime hashes.
Declare four training-only runs before computation; do not tune thresholds,
revisit the final test or change the submission pin in this extension.


## 2026-11-08 — Expose both portfolio growth conventions

Read existing `log_return`, `cagr` and `annualized_log_growth` artifacts rather
than change accounting or refit. Show annual log growth beside CAGR and offer
monthly/cumulative log views in the shared comparison chart. Keep its wealth
default and existing risk/ML-advantage panels. Monthly observations start at the
first earned month; cumulative series include the funding snapshot. A log wealth
axis is a separate display option. See [definitions](RETURN_METRICS.md).

## 2026-11-08 — implement the five-stage composition

The user's subsequent "Implement please" authorizes the earlier handoff. Replace
the seven navigation tabs with Strategies, Basic ML, Advanced ML strategy,
Feature selection and Final test. Keep advanced behavior undefined. Reuse
existing renderers with case-filtered views and distinct widget namespaces;
preserve scientific code, artifact formats and report pin.

Store training and exploratory test selections independently. Original cases
can be read from an existing model-containing run without recreating it or
rewriting its provenance. Final primary results must pass the existing readiness
checks and frozen policy; historic evidence is opt-in. Feature selections can
cover multiple supported strategies, with missing cash-rule experiments explicit.
No new research experiment, feature selection or tuning is part of this refactor.

## 2026-11-08 — five progressive tabs, planning only

The latest user request specifies training-only non-ML Strategies first, Basic
ML prediction quality plus portfolio effects second, an undefined Advanced ML
placeholder third, selected-strategy Feature selection fourth, and Final test
last. Record [the handoff](TAB_REFACTOR_HANDOFF.md); do not implement this turn.

Reuse existing rendering/computation/storage. Small explicit case filters and
separate training/test state are necessary; merely renaming tabs would leave
ML/test leakage in presentation. Preserve the current overlay's distinction
between fully invested normalization and cash-preserving reductions. Do not
invent a new advanced strategy or identify an earlier hybrid as that slot.
Training-only tuning remains later work before final test. This sequence
supersedes earlier navigation suggestions; hosting remains deferred.

## 2026-11-08 — group strategies and compare models across columns

The user's screenshot showed a dense change table repeating strategy names
for every model. Group the fourteen rules into six portfolio-construction
families; keep original and ML versions inside each rule. Default to collapsible
absolute-result sections and compact per-family delta matrices with a metric
selector. Retain detailed model rows, full tables and CSV exports. No averaging,
ranking or choice of a winner is introduced.

Use percentage-point units for return/drawdown/turnover changes. Color direction
follows the displayed metric, including lower-is-preferred turnover, with a
shared intensity scale across families. Display filtering must not fit models
or change saved configurations. The sidebar can choose families for the next
explicit computation; its long individual list starts collapsed.

## 2026-11-08 — explicit evaluation scope and historical test view

The user found the training/test presentation ambiguous. Use the loaded result's
recorded configuration/dates for prominent banners and plot titles; editing a
pending period must not relabel existing results. Label independently saved
feature/momentum studies with their own periods and add a visible section boundary.

Expose thirteen historical non-ML test portfolios in a separate Test results
view, reconciling shared records before deduplication. Read existing CSV evidence
without replay or fitting; do not silently promote archives into current runs.
Show registered test runs when present and distinguish expanding research from
frozen models. No new test experiment, final choice or report release change is
part of this UI fix. A chart showing one period needs no internal train/test line;
any future concatenated chart must mark both periods and their boundary.

## 2026-11-08 — separate momentum window from annual consistency

The user authorized comparing two, three, six and twelve months. Add a
cash-strategy-specific setting instead of changing the thirteen earlier rules.
Shorter momentum genuinely requires only that many consecutive known returns;
the annual screen still requires three complete years and stays fixed.
Keep twelve as the default to preserve the earlier result. Record the setting
in configuration, case metadata, diagnostics, audit, manifest and cache identity.

Compare all four windows with the same four fixed classifier overlays and equal
weight on training validation only. Check exact equality of annual pools and
prediction tables. This is strategy sensitivity analysis, not the assignment's
GridSearchCV evidence or an automatic final choice. Expose saved tables and
wealth/drawdown/cash curves in Comparison with explicit allocation loading.

## 2026-11-08 — annual winners, positive momentum and explicit cash

Implement the user's non-ML rule as a new registry entry; preserve the thirteen
existing rules. Define yearly consistency as a strict majority positive among
at least three complete January--December earned-return years known at the
evaluation origin. Flat years are nonpositive; incomplete years are excluded.
Freeze this historical pool. These are declared first settings, not tuned ones.

Each available pool member represents an equal capital slot. Invest the fraction
of slots with eligible strictly positive 12-month momentum, distribute that
equity budget proportional to positive momentum, and keep inactive slots in
cash. Empty pools/no positive signals mean all cash. Assume cash earns zero;
include cash in drift-aware turnover but not stock counts/concentration.

For this explicitly cash-capable rule, ML multiplies stock weights by
`1 - strength * P(loss)` and leaves released capital in cash. Existing rules
continue normalizing to full investment. Persist the policy and assumptions,
and show cash through time in the app. Do not infer a performance improvement
or a final strategy choice from the implementation.

## 2026-11-08 — six shared presentation goals and proposed refactor

Organize the research story around regular portfolios, a simple ML example, the
author's hybrid, feature-selection effects, tuning evidence and an overall
comparison. Reuse saved evidence across app, notebook, Quarto report and slides.
The proposed implementation adds explicit cross-run case references and a
versioned multi-run presentation release above the existing runner/store.
Keep the current numerical behavior, artifacts and release pin. The detailed
[plan](PROJECT_PRESENTATION_PLAN.md) distinguishes existing capabilities from
unimplemented work and candidate examples from final research choices.

## 2026-11-08 — first learned feature study

Compare all twenty original factors with two fixed <=10-factor recipes: ANOVA
ranking and greedy ANOVA ranking with absolute Pearson correlation pruning at
0.90. Keep four fixed models, thirteen trading rules, annual historical origins
and adjustment strength unchanged. Fit selectors inside each model pipeline,
before scaling; store actual masks and all exclusion diagnostics. Do not fit a
selector on the whole training decade before validating its earlier forecasts.

Use only training-period validation for this study. Register its predeclared
plan, underlying runs and matched comparisons separately from individual runs.
The app reviews saved evidence and does not refit on interaction. Keep the
submission release pinned until review and later tuning/final evaluation.
Derived composites and hyperparameter searches are separate future experiments.

## 2026-11-08 — durable research results

Use standard-library SQLite for the experiment index and SQL queries over key
metrics, with compressed Parquet for complete typed numeric tables. Keep CSV
exports and JSON manifests. This avoids a server dependency and supports complete
allocation restoration; a collection of summary CSVs alone cannot do that.

Saved comparisons are immutable records. Exact cache hits require matching input,
configuration, computation code and numerical runtime. Historical mismatches may
be viewed as recorded results, with provenance, but are not silently current runs.

## 2026-11-08 — submission and computation are separate

The app, notebook, Quarto report and slides share a result-reading layer. Rendering
reads pinned runs and never trains. A draft is clearly labeled until feature
selection, tuning, final choice and the submission backtest are complete.

## 2026-11-08 — model fitting policy

Preserve expanding research fits for existing experiments. Add a separately
identified frozen-at-origin policy for models applied to official test data.
Cache identities and reports must distinguish these protocols.

## 2026-11-08 — UI recommendation

Keep the working Streamlit viewer while stabilizing artifacts and the submission
pipeline. If the user wants a visual redesign, NiceGUI is the preferred candidate
for a custom Python interface; Panel is a strong analytical-dashboard alternative.
A framework migration is not implemented as part of this foundation update.
Evaluate with a small saved-results-only screen before replacing the current UI.

References: [NiceGUI](https://github.com/zauberzeug/nicegui),
[Panel Material template](https://panel.holoviz.org/reference/templates/Material.html),
[Reflex introduction](https://reflex.dev/docs/getting-started/introduction/).
