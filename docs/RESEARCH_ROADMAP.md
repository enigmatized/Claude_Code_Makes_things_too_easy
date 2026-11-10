# Next research stages: feature selection, tuning, final comparison

The foundation and four-recipe feature study are complete. A separate ANOVA
training-holdout tuning comparison is available in the app. Final feature/model
choice, tuning release integration and official frozen test evaluation remain
planned. Update the state and change log after each stage. See
[forest findings](../0_3_Feature_Selection/05_random_forest_findings.md) and
[the saved tuning holdout](TUNING.md). The original roadmap below supplies
background; its earlier "not run" statements predate those additions.

The user-requested cash-strategy window study is also complete: see
[2/3/6/12-month findings](../0_5_Momentum_Lookbacks/01_findings.md). Review its
turnover/risk/cash trade-offs alongside feature evidence before choosing the
small set for later tuning. It is strategy sensitivity analysis; model
GridSearchCV and final window selection remain pending.

## Before more experiments: shared presentation and comparison

The [five-stage refactor](TAB_REFACTOR_HANDOFF.md) is implemented: training-only
strategies, basic ML, an undefined advanced placeholder, selected-strategy feature
analysis, and final test. Existing renderer composition and small case filters
preserve numerical behavior. Training and test state are separate.
Extend cross-run/report release interfaces incrementally as needed. Preserve
computation, artifacts and old release compatibility. Tuning remains a later
training-only stage before final test; it can live within the fourth stage.

## 1. Feature evidence with fixed models

The completed [first study](../0_3_Feature_Selection/02_findings.md) compares all
twenty factors with ANOVA top ten and correlation-pruned up to ten using four
fixed models and thirteen rules. Review its saved app results before choosing
the small recipe set for tuning. Predictive and portfolio differences are mixed;
no model/recipe beats the historical probability control on Brier/log loss.

Learned selection belongs inside the sklearn pipeline and each chronological
fold. A selector fitted once on all 2000–2009 observations cannot validate an
earlier 2005 forecast. FeatureSpec now serializes learned selectors and the model
pipeline refits them at each origin. Actual columns, counts and diagnostics are
saved. The ANOVA/pruned unions over five annual fits contain 12/11 names, while
each fitted model uses ten. Frozen submission fits select once before evaluation.

Derived features/composites are a separate recipe experiment with input lineage
and explicit availability. Do not silently change the evaluation universe when
a new feature lacks history. Do not assume composites evade the assignment's
ten-feature limit.

## 2. Training-only GridSearchCV

The callable version is [TUNING.md](TUNING.md) / `research/tuning.py`. It is not
yet run on the panel. The rules below still govern that run.

Use explicit month-group train/validation index lists; all stocks from a month
stay together. Within each historical outer fit, inner training labels must be
known at the inner validation origin. The existing one-month label convention
must be applied to splits, rather than using random rows or default stratified CV.

Use a proper probability score such as negative log loss as the initial search
objective. Keep portfolio scoring separate. Fit scaling and selection through
the Pipeline within GridSearchCV, then refit its chosen estimator on the allowed
outer training slice. Hyperparameters and selected features may differ by origin.

At least two models must have recorded tuning evidence. A small first grid can
include logistic C values and random-forest depth/leaf-size combinations; NB
smoothing and histogram-boosting learning-rate/leaf combinations can follow.
Start with a small declared grid, fixed seeds and controlled parallelism, and
record the grid before inspecting validation results.

Save the candidate parameters, per-fold/mean scores, chosen parameters, scoring
direction, training cutoffs, recipe identity and runtime. Persist compact CV
trial tables, not a complete allocation history for every grid candidate. Save
full predictions and portfolio cases for the selected comparison configurations.
Extend cache identity to include the selector, grid, CV schedule, scoring, chosen
model parameters and input lineage. Do not reuse a fixed-model artifact as a
tuned-model result.

## 3. Final decision and test experiment

Choose a small set of specifications based on training-period evidence. Compare
at least the market equal-weight portfolio and one declared non-ML strategy.
Include return, volatility, Sharpe/Sortino, drawdown, turnover and concentration.

The official test model fit uses `fit_policy="frozen_at_origin"`: train/tune on
training data known at Dec 2009, then apply the fitted model to later snapshots.
The separate expanding research policy remains available, but do not label it
as the assignment's frozen-training test model experiment. Lagged non-ML signals
may update once returns are realized under the documented month-end convention.

Disclose that the test period was inspected earlier. Do not present repeated
test-driven changes as new holdout evidence. Record a final choice and explain
why its training evidence and risk/reward trade-offs support it.

## 4. Publish from pinned artifacts

Pin the baseline, selected-feature and tuned comparisons explicitly. Revise the
report narrative to match them, execute the readable notebook, and build Quarto
HTML with folded code. Complete the approximately 15-minute presentation with
results, not code snippets. Final builds must pass the recorded-evidence gate.

## Later: ensembles

First compare a fixed probability average using aligned model predictions. A
learned blend or stack requires chronological out-of-sample training predictions;
never fit it on base models' in-sample predictions or final evaluation outcomes.
Keep predictor combination separate from blending completed portfolio weights.
