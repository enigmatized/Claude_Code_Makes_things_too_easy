# Feature-selection study

**Random-forest selection is implemented and saved.** Study
`features_20261109T202318_8b01fe74` compares all twenty factors, ANOVA ten,
correlation-pruned ten and forest ten across four fixed pooled downside models
and all fourteen rules, including the three-month momentum/cash rule. Read
[the new findings](05_random_forest_findings.md),
[executed forest review notebook](04_random_forest_review.ipynb) and
[shared method notes](method_notes.md). The app, submission notebook, Quarto
report and slides read this evidence through shared helpers. The historical
three-recipe study described below is preserved.

Read [the fixed experiment plan](01_experiment_plan.md), then review the completed
study in the app's **Feature selection** tab. The app loads its saved runs; it
does not train while showing scores, selection histories or portfolio paths.

The first completed study is `features_20261108T153403_ddfe05c8`. Read
[the findings](02_findings.md) or execute the
[saved-results review notebook](03_feature_selection_review.ipynb).
An [HTML copy](outputs/03_feature_selection_review.html) is available too.
Effects are small and mixed; no final subset or winning model is declared.

Three recipes are compared with all four fixed models and thirteen portfolios:
all twenty factors, ANOVA top ten, and correlation-pruned up to ten. Everything
is fitted on historical training slices and evaluated on Jan 2005–Dec 2009.

Reproduce the current four-recipe study explicitly from the project root:

```bash
python3 run_feature_selection.py
```

Exact compatible runs are reused through the existing artifact store. Each
completed study links its run IDs in `artifacts/studies/<study_id>/study.json`
and in the SQLite study index. The plan is written before computation. Comparison
CSVs and fit diagnostics are alongside that file; full underlying allocations
and predictions remain in their linked immutable run directories.

Inside **Feature selection**, choose the saved study and model, then use the two
subtabs:

**Prediction model to review** controls both subtabs. The result tables show
one predictor at a time: logistic regression by default, or Gaussian naive
Bayes, random forest or histogram boosting. A second selector,
**Prediction model for portfolio comparison**, sits directly above the portfolio
table in that order, with logistic first. Both controls stay synchronized;
changing either updates feature inspection, prediction scores, calibration,
portfolio tables and curves using the saved model results. The **Prediction
model** columns repeat the chosen name. Rows compare feature recipes
for that predictor; **Random forest top 10** selects inputs and can supply them
to any of the four prediction models.

1. **Selected features & method**: choose a selection method and historical fit.
   The chosen inputs appear first, with their meaning, group, rank and score.
   Inspect all candidates/exclusion reasons, the selection history and the
   training-only correlation matrix below. This review works independently of
   the portfolio selection in the sidebar. The new study defaults to the forest
   importance chart with selected factors highlighted, tree-variation error
   bars, full selector settings and cross-method selected-feature overlap.
2. **Effect on ML models**: inspect saved retraining evidence and compare
   prediction scores, then select ML-adjusted portfolios to review returns and
   curves against the same model using all twenty factors. Each recipe's models
   were independently refitted at every historical origin; these are not old
   twenty-input models with columns hidden after fitting. The viewer loads
   those saved results and never starts training. Compare saved calibration,
   CAGR/annual log growth, and wealth or cumulative log-return paths.

Feature selection applies only to the ML component. The underlying non-ML
portfolio rules are unchanged and appear only as reference curves. The current
four-recipe study covers pooled downside models on all fourteen rules. It does
not supply feature-selection evidence for upside/per-stock variants. The older
three-recipe study still covers thirteen rules. Missing combinations are labeled;
the feature audit and predictive evidence remain available.

The separate **Tuned model** tab has a saved ANOVA training-holdout comparison;
see [tuning](../docs/TUNING.md). It is not a tuned forest-selection comparison
and its shorter validation window must not be mixed into this matched study.

Selection counts apply at each fit. A validation-period union can exceed ten;
do not present it as a fixed final subset. The next step after review is
training-only hyperparameter tuning, then a justified frozen submission fit.
