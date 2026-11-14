# Basket-first app navigation — implementation and handoff

Date: 2026-11-10. The initial request was planning only; the subsequent user
request “Okay, can you implement please” authorized this implementation.
The original proposal below is retained as the design reference.

## Implemented

The seven main tabs remain. Portfolio comparisons now use **Asset basket →
Prediction model → Allocation strategy → Prediction target**, where applicable.
Strategies uses only basket/allocation. Method lessons retain their own input/
fit context. All choices resolve completed saved cases; no new model training,
strategy replay, backtest, feature construction or hyperparameter search was run.

- `portfolio_selection.py` explicitly maps the fourteen registry bundles and
  grouped/engineered books to basket/allocation descriptions. It retains their
  original IDs, model/target/horizon and all source/fit/recipe columns. Mappings
  do not make arbitrary basket/weighting combinations interchangeable.
- `portfolio_selection_ui.py` provides the shared controls. Only actually saved
  tuples are offered. Upstream changes validate and reset stale dependent state;
  a missing direction is explained, never synthesized or read from another study.
- Strategies and Basic ML start with one configuration. **Compare more** restores
  the existing grouped/full tables, model choices and select-all charts. Toggling
  comparison mode resets stale chart filters. The advanced source-strategy filter
  is collapsed in deployment. Training scope and saved Basic ML runs are secondary
  details. Basket selection does not retrain pooled models on only those names.
- Grouping With ML keeps K-means/Ward explanations and the four subtabs. Adding a
  direction selects the actual grouped basket, model, momentum horizon and one
  event before results. The acceleration experiment remains nonpredictive and
  its fourth-subtab location is explicitly called out.
- Feature selection opens with the selected-features/method audit. Grouped-book
  portfolio effects now live inside **Effect on ML models** with the shared
  selection flow. Registry effects have their own compatible choices. Audit
  model/target context remains separate from the portfolio study when protocols
  differ; up/down feature fits are not interchanged.
- Tuned model uses the same flow over its actual studies, retaining original/
  fixed/tuned comparisons, log-loss objective, feature lists and trial tables.
  An absent study or model is not fitted by visiting the page.
- Feature-engineering results reorder the existing book/model/target choices,
  then retain recipe and fixed/tuned comparisons. Method construction and CV
  teaching views stay separate.
- Final preserves its distinct K-means, engineered and acceleration studies.
  The earlier K-means view filters to the selected original/adjusted pair, including
  the actual selected momentum horizon. No-model cases have no fabricated target.
  Earlier training/entire-path inspection remains explicitly labeled.
- `final_comparison_ui.py` lets users add displayed official-test cases to a
  **Compare saved final portfolios** list. It retains original metric values,
  source/study protocol and monthly paths. The list is session-only, can be
  cleared and supports growth/log-return, drawdown and cash charts. Different
  evaluation calendars are displayed but cannot be charted as a matched test.
  The app clears the list when switching publication or research/deploy context.
- The momentum-window “Load allocations” callback now selects the correct annual-
  consistency basket and saved run, rather than leaving an unrelated menu choice.

Deployment still reads the same `published/project` manifest and Parquet frames.
Manifest SHA256 remains
`5dc7835271b868a609798657d4fa043be7dc19623c87ef1356ba6e3f3bef9372`.
No export, new data, dependencies, database or ignore-rule change was needed for
this navigation work. The preceding acceleration release still needs to accompany
code when pushing to update a site that predates it. No commit/push was made here.

Formal notebook/QMD/slides/release pins were not edited by this implementation;
preserve the independent submission work. Report builders can later reuse the
pure saved-case descriptions, but report integration is not implied by UI changes.
Verification is recorded in `docs/verification/basket_first_ui_20261110.json` and
screenshots. Next: review the navigation locally, then push code/docs and the
current publication when ready. New combinations and basket-specific model
training remain separate explicit experiments.

## First: finding the existing acceleration change

The existing addition is the fourth nested tab under **Grouping With ML →
Momentum acceleration**. Its separate final evaluation is under **Final test →
Momentum acceleration**. It is not a new main tab, and the K-means subtab does
not automatically open it. Code locations: `basket_ui.py`, `acceleration_ui.py`
and `_final_book_sections` in `app.py`.

At this review, the new acceleration source and display frames are still
uncommitted local files. They have not been pushed by this agent. A hosted site
needs the code and `published/project` update; a running local app may need a
restart to reload imported presentation modules. Earlier publication roots do
not contain these results. Check the actual selected release rather than
recomputing models to resolve a navigation/deployment issue.

## Design reference: requested interaction

For portfolio comparisons with supervised ML:

**1. Asset basket → 2. Prediction model → 3. Allocation strategy →
4. Prediction target (next up / next down) → saved results.**

Keep the seven current main tabs. Reuse this order where it helps; do not force
model and direction controls onto nonpredictive teaching views. Single selection
should be the easy starting point, with “Compare more” exposing existing multiple
strategy/model charts. Keep the original strategy beside its adjusted version.

Use consistent names:

| Concept | Meaning | Examples |
|---|---|---|
| Asset basket | Which stocks may receive capital, and how membership is chosen | All available; historical Sharpe pool; K-means top ten per group |
| Grouping method | Unsupervised algorithm that creates stock groups | K-means; hierarchical Ward |
| Prediction model | Supervised algorithm estimating a stock's next-month event probability | Logistic regression; random forest; the other saved classifiers |
| Allocation strategy | Rule for distributing stock exposure and holding cash | Equal weight; positive momentum + cash |
| Prediction target | Event separately learned by a fitted model | Strictly positive next month; strictly negative next month |
| ML weight adjustment | How the probability affects the original allocation | Downside reduction; upside threshold boost |
| Feature recipe / settings | Model inputs and whether estimator settings were tuned | ANOVA ten; engineered composites; fixed/tuned |

The initial UI should explain the existing target-dependent adjustment instead
of adding another top-level selector. Down/up are different fitted cases, not a
chart-label switch. Flat returns belong to neither event, so do not substitute
`1 - probability_of_loss` for the separately fitted upside model.

### Important scope boundary

Choosing a basket initially filters **existing saved portfolio cases**. It does
not automatically retrain a pooled model using only that basket's members.
Existing pooled predictions are generally trained across stocks and then applied
to individual names; per-stock fits are a separately labeled existing mode.

Likewise, “K-means top ten” is currently a saved combination of grouping,
screening, positive momentum and cash. Many basic registry strategy IDs also
bundle selection and weighting. Moving selectors must not imply that every
basket can already be paired with every weighting rule, model and direction.

Start by giving these existing bundles clear basket/strategy descriptions and
filtering supported cases. Actually making every universe interchangeable with
every weighting rule would require a larger research/engine refactor and new
backtests. That is a separate possible project, not this first UI pass.

## Tab-by-tab plan

| Current main tab | Proposed first controls | Degree of fit | Relative effort |
|---|---|---|---|
| Strategies | Basket → allocation strategy | Skip prediction controls | Small to medium |
| Basic ML | Basket → model → allocation strategy → up/down | Primary example of the new order | Medium |
| Grouping With ML | Learn/inspect grouping, then basket → optional model → strategy → optional up/down | Keep unsupervised learning separate from direction prediction | Medium |
| Feature selection | Method audit first; basket → model → strategy → target in effects view | Preserve both existing teaching sections | Medium |
| Tuned model | Basket → model → strategy → target, followed by fixed/tuned comparison | Reorder saved comparisons, clarify objective | Small to medium |
| Feature engineering | Construction first; same four controls for results, then recipe/settings | Reuse existing recipe studies | Medium |
| Final test | Saved basket → model → strategy → target, then valid feature/settings comparisons | Needs careful matching of different saved protocols | Medium to larger |

These estimates are relative, not time commitments. The coverage audit determines
how many adapters are needed. Shared result selection and regression checks are
additional work; a universal plug-and-play strategy engine would be much larger.

### 1. Strategies

Current: strategy choices are largely a sidebar list; strategy IDs bundle asset
selection and allocation. The first tab already excludes ML and test data.

Proposed:

- Start with an asset-basket selector describing available/frozen screening.
- List only existing compatible allocation strategies for that basket. Keep
  multi-strategy comparison available rather than replacing the current charts.
- Keep tooltips, explanations, return/drawdown/cash and allocation history.
- Show “No prediction model in this tab”; no model/up/down dropdown is needed.
- Keep exact eligibility, frozen pools, momentum horizons, fallback behavior and
  cash rules from the selected saved strategy. Similar labels do not imply the
  same mechanics.

Files: `app.py:render_strategies`, `research/strategies.py` metadata and
`research/explanations.py`. Prefer presentation mappings first; avoid editing
`build_weights` implementations.

### 2. Basic ML — recommended pilot

Current: direction pattern and pooled/per-stock scope come first, followed by
saved-experiment and multiple-model controls; strategies come from the sidebar.

Proposed:

- Put basket, model, strategy and prediction target together above the results.
- Place pooled/per-stock scope, feature list, fit policy and saved-run provenance
  in a clearly visible secondary details panel. Scope remains part of result
  identity; changing it cannot silently fall back to a different experiment.
- Preserve the three collapsed sections: prediction quality, portfolio results,
  and changes from the original. Offer “Compare more models/strategies”.
- State the actual adjustment and cash policy. Some fully invested basic rules
  renormalize after downside reduction; cash-capable rules retain cash. An upside
  boost normally redistributes within the existing equity budget. Do not silently
  convert every strategy to a downside-to-cash rule in this presentation change.
- Distinguish all-scored-stock prediction metrics from basket-specific metrics.
  Existing aggregate scores cannot simply be relabeled as basket-only scores;
  producing missing basket-specific diagnostics requires an explicit export.
- In research mode, keep computation an explicit separate action. Dropdown changes
  browse saved results. Deployment offers only published combinations.

Files: `app.py:render_basic_ml`, `basic_ml_ui.py`, current chart/view helpers and
the existing `PublishedBasicStore` interface. Reuse existing fits and accounting.

### 3. Grouping With ML

Current subtabs: K-means; Hierarchical clustering; Adding a direction; Momentum
acceleration. The first two explain grouping and apply the saved strategy.

Proposed:

- Preserve K-means/Ward explanations, group visualizations and membership audits
  before the investment comparison. Grouping ML is not the direction classifier.
- In the directional comparison, choose a basket such as K-means full groups,
  K-means top ten, Ward full groups or Ward top ten; then model, saved allocation
  rule/horizon and target. Show the selected basket's construction immediately.
- If momentum + cash is the only compatible allocation rule, display its name as
  fixed instead of a one-option dropdown suggesting unimplemented alternatives.
- Keep Momentum acceleration discoverable as a rule experiment. It needs basket
  and rule controls; its current four variants have no direction classifier.
  Do not expose an ML model/target for these variants until those adjusted cases
  have actually been computed and published.
- Show original/stock/basket/combined acceleration together; keep the existing
  nonnegative holdings, cash and fixed-current-member return comparison.

Files: `basket_ui.py`, `acceleration_ui.py`, existing basket readers and exported
method metadata. Preserve `research/basket_momentum.py` numerical defaults.

### 4. Feature selection

Preserve **Selected features & method** and **Effect on ML models**.

- Method section: lead with selection method, fit date and selected inputs;
  retain random-forest importance, overlap and method explanations. Prediction
  model/target context is needed where masks or importances depend on that fit,
  but basket/weighting controls are not required for a general selection audit.
- Effects section: use basket → prediction model → allocation strategy → target,
  then select feature recipes to compare. Show each independently refitted model
  against the same trading rule and unadjusted reference.
- Map registry and grouped-book feature evidence through saved-case adapters.
  Respect actual direction/model coverage of each source; a Basic ML target
  selection must not relabel a separate downside-only feature study.
- Show the exact saved input lists, training origin and fitting policy. No feature
  selection “improvement” is inferred for the non-ML reference.

Files: `feature_study_ui.py`, `basket_feature_ui.py`,
`research/feature_reporting.py`, `published_analysis.py`.

### 5. Tuned model

- Reorder the current model/basic-strategy/target choices into the four-part
  sequence, using basket descriptions for the existing saved strategy bundles.
- Keep no-ML, fixed ML and tuned ML side by side under the selected configuration.
- Distinguish the **prediction target** (up/down) from the **search objective**
  (currently classification log loss). Selecting a portfolio does not mean
  the model was tuned to maximize that portfolio's Sharpe or CAGR.
- Display feature recipe, search grid, chronological folds, chosen settings and
  whether settings changed. Keep identical winners explained as identical.
- Show only strategies supported by that tuning study. Older basic-model tuning
  and engineered-recipe tuning have different protocols; do not mix their rows
  into an apparent single controlled experiment.

Files: `tuning_ui.py`, saved tuning readers. No new `GridSearchCV` or tuning
objective is needed to reorder these comparisons.

### 6. Feature engineering

- Preserve Methods & feature construction. This clusters/combines factor columns,
  whereas Grouping With ML clusters stocks; keep those lessons distinct.
- In Model & portfolio results, reorder current model/target/starting-portfolio
  controls to basket → model → compatible strategy → target. Existing “book”
  records combine a universe and allocation; initially map them without changing
  how they were computed.
- Then expose feature-recipe comparisons and Fixed/Tuned/Both. Retain original
  reference, actual input counts, construction audits and recipe CV provenance.
- In Hyperparameter tuning, show recipe/model/target and the real search metric.
  Basket/strategy controls are unnecessary for classifier CV itself; they can
  select a downstream portfolio illustration. Clearly label any shared model fit.
- Do not add acceleration as a feature here without a separately declared
  training/CV experiment; saved signal inspection is not a trained input recipe.

Files: `engineering_ui.py`, existing local/published engineering readers.
Preserve `research/engineering_study.py` and completed study identities.

### 7. Final test

- Offer the same order using **only available saved official-test cases**. Begin
  with the test window visibly fixed; historical/entire-path inspection can stay
  under separately labeled advanced controls.
- Preserve original portfolio and adjusted result together, with CAGR/log returns,
  Sharpe, drawdown and cash. Allow selected configurations to be added to a
  comparison list. Compare on the same earned months and disclose cost assumptions.
- Keep the older K-means fixed-model study, engineered fixed/tuned recipes and
  acceleration rules as distinct protocols. If a common selector is introduced,
  retain study/protocol labels in every row and avoid connecting incompatible
  training/test curves into a fictitious single experiment.
- The acceleration study has no direction classifier; skip model/up/down there.
  A missing final configuration is “not evaluated”, never replaced by training
  results, a different recipe or a new computation.
- Do not make the current upstream dropdown choices an automatic new final test.
  Supervised features/parameters must already be frozen from training. This test
  period was previously viewed; do not describe it as untouched.

Files: `app.py:_final_book_sections`, `final_book_ui.py`, shared engineering/
acceleration renderers and published analysis readers.

## Smallest useful shared change

Before touching tabs, inventory existing saved cases using their manifests and
metadata. Build a thin presentation adapter that returns supported combinations
and resolves a selection to its original saved case. A small table/helper is
enough to start; do not introduce a database, runner replacement or class hierarchy.

The display case record should preserve basket/strategy identity, grouping/screen,
model, target, model training scope, feature recipe, fixed/tuned status, adjustment
and cash policy, evaluation dates, fit origin/policy and source study/publication.
Use explicit “not applicable” for nonpredictive cases, not a fabricated classifier.
Do not infer identities from labels alone or merge different source protocols.

Only enumerate combinations actually represented in completed results. Because
the target control is last, preceding choices can use the union of available
targets; once all selections are known, validate the exact tuple. Clear stale
dependent selections when a basket/model changes. A missing up/down case remains
visibly unavailable, with the available alternative explained. No silent substitution.

The UI order is not execution order. Actual computation, when explicitly requested,
still resolves all settings first: causal basket and baseline weights; fitted
features/model for the chosen target; per-stock probabilities; weight adjustment;
cash-aware accounting; saved results. Selecting a model earlier on screen does
not fit it before the target is known.

## Deployment and submission

Prefer readers/adapters over migration. Keep existing artifact schemas, checksums,
case IDs, run configs and historical outputs. Only if existing metadata cannot
describe the choices, add a small optional display index during an explicit export;
older releases should remain readable. Never edit the manifest or Parquet in place.

Deploy remains saved-results only: no CSVs, research catalog, fitted objects,
training or strategy replay. Research can retain explicit compute buttons using
the existing runner. No newly calculated return series in presentation code.
Keep raw/research/backups ignored and the active publication included in Git.

Shared descriptions and case selection can later be reused by notebooks/QMD.
An initial UI change does not rewrite the formal report, notebook, slides or
release pin. Preserve the other LLM's submission work and rebuild deliverables
only when report integration is separately requested.

## Suggested implementation stages, if later requested

1. Audit supported cases and define names; no scientific changes.
2. Pilot the four controls on Basic ML. Verify usefulness before spreading it.
3. Add basket/strategy controls to Strategies and Grouping With ML, including
   clearer acceleration navigation. Preserve their teaching views.
4. Adapt feature-selection effects, Tuned model and engineering results, reusing
   the pilot's helpers where protocols permit.
5. Adapt Final last, after saved-case/provenance matching is verified. Then align
   report-facing labels and any explicitly requested submission views.

After every stage, compare saved monthly returns, allocations and model scores
with the same original cases; test unavailable combinations and stale widget state.
Check both modes and each direction; block source/object/fitting/replay access in
deployment. Confirm first-six-tab training boundaries and separate final dates.
Use the existing tests plus focused selection/coverage checks, not a new parallel
test framework. Update handoff docs after each implemented stage.

Recommended stopping point if time is limited: Basic ML pilot plus clearer names
and summaries in Grouping With ML. A complete mathematical decomposition of all
strategies and every possible basket/model/rule pairing is not needed for a
readable submission or a useful hosted viewer.
