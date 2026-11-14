# Submission requirements and build pipeline

2026-11-10 (later): `--final` now uses `research/submission_checks.py`, which reads the frozen
final-book study, the GridSearchCV holdouts and the pinned engineering study, and passes.
The notebook adds 5b (tuned settings on the chosen book, 2010–2018), 8b (trading costs) and
8c (significance versus equal weight).


The separate momentum-acceleration app experiment is published with both training
and previously viewed final evidence. Its shared manifest reader can be reused
from notebooks/QMD without fitting; see [protocol](MOMENTUM_ACCELERATION.md).
No formal notebook/report/slides or `release.json` change is introduced by that
experiment; the independent graded-notebook rewrite below is preserved. Acceleration
has not yet been added to any classifier input recipe.

2026-11-10: `analysis.ipynb` was rewritten for presentation and now maps each rubric
item to a section. It adds a read-only Section 7 (volatility-neutral ML and the
ML+momentum blend) from `0_7_Feature_Engineering/part8_*`. The declared choice,
release pin, QMD sources and rendered HTML are unchanged.


The user authorized a feature-engineering extension and its separate final
variants. A new review notebook and standalone QMD supplement use published
saved evidence; formal report/slides/release pins are not automatically replaced.
Each engineered estimator has at most ten model inputs, but some aggregate twenty
source factors. This is not asserted to satisfy an instructor limit on underlying
sources. Original-factor ANOVA-10 remains an explicit reference. Both previously
viewed training/test evidence and the recipe-selection limitation must be disclosed.
See [notebook review and full experiment protocol](FEATURE_ENGINEERING.md).

The latest tuning/final UI clarifies that fourteen basic-rule classifier holdouts
are separate from the existing fixed K-means official-test portfolios. Added log
views/original-relative changes reuse saved evidence. Unsupervised tuning and
Ward/tuned-classifier official-test variants are not claimed complete. This
presentation review changes no formal notebook/QMD/HTML/slides or release pin;
see [handoff](TUNING_AND_FINAL_REVIEW.md).

The third app tab now compares K-means and Ward hierarchical grouping before
separate directional predictions. Shared published readers and
`research/grouping_charts.py` support later notebook/QMD integration; see
[Grouping With ML](GROUPING_WITH_ML.md). Ward evidence is training-only. Existing
feature/tuning/final sources stay K-means, and this change does not update the
formal submission pin, notebook, QMD/HTML or slide deck.

The hosting presentation now exports all six tabs, including registry/advanced
feature results, tuning trials and frozen final windows. These are saved-evidence
readers, reusable by reports; no new experiment or formal release is inferred.
The final saved experiment uses factory settings and ANOVA-10, independently of
the earlier tuning holdout. Advanced-family tuning remains absent and explicitly
labeled. The formal notebook/QMD/HTML/slides and `submission/release.json` are
unchanged by the deployment work. See [deployment](DEPLOYMENT.md).

The Advanced ML presentation now separates learned grouping and portfolio
construction from later directional classifiers. Shared saved-frame membership
and return-pattern helpers in `research/basket_reporting.py` support notebook/
Quarto reuse; the deployed third tab reads the existing basket study only.
See [basket presentation](BASKET_MOMENTUM.md). No source/rendered deliverable or
formal release pin changes accompany this UI/publication refactor.

Basic ML now also has a self-contained publication, reading saved predictive,
calibration, portfolio/change, weight and fit-audit tables through the same
training-only renderers. Both directions/scopes and the current classifier
coverage are recorded in the [deployment handoff](DEPLOYMENT.md). These pure
saved-result readers are available for report/notebook reuse; the formal
submission pin and rendered deliverables are unchanged by this UI/export step.

The app has a separate first-tab published-results package and explicit local
research/deploy modes; see [deployment handoff](DEPLOYMENT.md). Its pure readers
and exported metric/curve/weight frames can also feed notebooks and Quarto.
This is a presentation export from completed training evidence, not new research
or a change to the formal `submission/release.json`, QMD, notebook, HTML or slide
pin. No final/test evidence is included in the first-tab publication.

Tuning evidence now covers four classifiers × next up/down on the same frozen
ANOVA-10 training holdout; see [tuning](TUNING.md). Targets use separately trained
models; thresholds/strategies stay fixed. Three saved studies feed shared read-only
loaders for later notebook/QMD integration. This addition updates the app and
exploratory evidence only: formal submission sources, rendered outputs and
release pin remain unchanged; final selection/frozen official test are pending.

Additional exploratory evidence: [directional basket overlays](BASKET_MOMENTUM.md)
now compare four fixed all-20 pooled predictors on the advanced momentum book.
This is separate from the ANOVA-10 tuned holdout and is not the final <=10-input
submission specification. The shared basket loader supports future notebook/QMD
integration without fitting; the formal release and rendered deliverables have
not been changed by this addition.

Verified against `03_mlfactor_long_only_project.ipynb` on 2026-11-08.

## Required deliverables

| Deliverable | Assignment requirement |
| --- | --- |
| Coding/analysis notebooks | 175 points; accuracy, readability and code comments matter. |
| Feature analysis | Choose <=10 features; analyze whether restriction improves the strategy. |
| Models | Four model types; at least two have tunable hyperparameters. |
| Tuning | Use sklearn GridSearchCV on training data with standard ML scoring; examine portfolio effects. |
| Benchmarks | At least two non-ML strategies, including the equal-weight market portfolio. |
| Evaluation | Consider reward and risk; justify a final model/strategy choice. |
| Written report | 25 points; Quarto HTML, professional presentation, visualizations produced by code cells, folded code. |
| Presentation | 100 points; approximately 15 minutes, results and visualizations, no code snippets. |

Keep the `.qmd` source with its rendered `.html` for reproducibility. The notebook
explicitly asks for a Quarto HTML report; it does not separately spell out whether
the LMS requires uploading the source too. Providing both is the planned package.

## Additional basic-ML research evidence

The [direction/scope comparison](../0_6_Basic_ML_Directions/README.md) provides
saved training-only downside/upside and pooled/per-stock experiments. Shared
runner/store outputs, protocol JSON and comparison CSVs can be read by future
Quarto/slides work without fitting. The exploratory local comparison uses one
model type and all twenty supplied factors; it does not satisfy the complete
final assignment requirements by itself. The release pin is unchanged.

## Methodology before final evaluation

The assignment describes fitting models on training data, then applying them to
test data. Current exploratory expanding test refits must not be presented as
that frozen-training experiment. Use the explicit frozen-at-origin policy for
submission model fits and document allowed lagged non-ML signals separately.

Feature selection, preprocessing, tuning and any learned probability calibration
belong inside historical training folds. All stocks from a month stay together.
Do not use a random row split or default classifier CV. Rank models/features with
training-period validation; the test period was already inspected and that must
be disclosed.

## Planned final build

The latest [five-tab plan](TAB_REFACTOR_HANDOFF.md) also defines the intended
report/slide progression: non-ML training results, basic downside-prediction ML,
a reserved advanced branch, feature/tuning training evidence, then final test.
Advanced content stays pending until the user defines it. The app now follows
this progression using existing computation and presentation helpers. Report
sources and release schema/pin are unchanged; the Quarto narrative and final
research evidence still need completion.

The broader research questions still include feature effects, tuning and an
overall comparison. Map them into the five stages: tuning within the training
research stage, and comparative evidence within the training and final-test
results. These are not six required main tabs. Proposed shared case references
and a versioned multi-run release remain background in
[PROJECT_PRESENTATION_PLAN.md](PROJECT_PRESENTATION_PLAN.md). The implemented
draft currently reads one pinned run.

```text
explicit research/tuning commands → complete saved experiments
→ pin chosen run IDs and status in a release configuration
→ execute the readable analysis notebook against those artifacts
→ render Quarto HTML and slides from the same artifacts
→ validate final requirements and package code, QMD, HTML, notebook and slides
```

Rendering must not rerun a hyperparameter search or refit models. A draft report
can render incomplete research with its status stated. A final package must not
claim feature selection, tuning, final model choice or an official test backtest
until those are actually completed and traceable to saved runs.

The report should lead with the conclusion and comparative results. Keep data
background brief. Explain the feature-selection evidence rather than merely
listing columns. Keep methodological detail and code available in folded cells
and the analysis notebook, with reusable implementation in `research/`.

## Implemented draft build

The workspace is [submission/](../submission/README.md). Run
`python3 build_submission.py` from the project root. The executed notebook,
folded-code Quarto report and hidden-code draft slides use a pinned saved run;
no fitting or parameter search occurs. The report and slides were rendered and
checked as self-contained HTML. `--final` currently rejects the incomplete
research evidence, as intended. The short draft deck still needs the final
feature/tuning results and expansion to the required presentation length.

## Feature-selection evidence now available

The [random-forest selection extension](RANDOM_FOREST_FEATURE_PLAN.md) is
implemented and rendered: a fourth matched recipe, shared method notes/figures
for app/notebooks/report/slides, and a checksummed feature-study reference
alongside the existing release run. Study `features_20261109T202318_8b01fe74`
covers all fourteen rules and four fixed pooled downside models. Its ten-input
fits are training validation, not the chosen tuned final submission model.
The earlier three-recipe study described below remains unchanged.

The first training-only study is documented in
[0_3_Feature_Selection](../0_3_Feature_Selection/README.md), including a readable,
executed review notebook and saved-results app view. Each learned model uses ten
factors at each annual origin; masks can change between origins. This provides
initial validation evidence, not a final fixed ten or tuned submission model.
The readiness check uses the actual fitted count for learned selectors rather
than counting all candidate columns. The main release run remains the older
twenty-factor draft; a separate feature-study pin adds the new evidence without
claiming final readiness. The app's separate ANOVA tuning holdout is not yet
pinned into the submission. Final choice and frozen official test remain pending.
