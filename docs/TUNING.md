# Chronological grid search

## What the app is tuning

Choose the three controls at the top of **Tuned model**, in order:

1. **Model:** logistic regression, Gaussian naive Bayes, random forest, or
   histogram gradient boosting.
2. **Basic strategy:** one of the fourteen existing rules. Its rule and momentum
   window stay fixed; the selected model adjusts its allocations.
3. **Prediction target:** next down (`R1M_Usd < 0`) or next up (`R1M_Usd > 0`).
   These are separately trained binary classifiers. A zero return is neither event.

The **target** is the event being predicted. The **optimization score** is log
loss for that event, minimized by chronological GridSearchCV. Portfolio return
and Sharpe are outcomes to inspect afterward, not this search's objective.
One pooled predictor per version scores every stock; changing the strategy does
not refit it or change its all-stock prediction scores.

Below the controls, the app shows prediction quality, then the chosen strategy's
no-ML / untuned-ML / tuned-ML portfolio table and paths. The historical event-rate
control is shown **once**, after checking its two saved rows agree. It predicts
one frozen training frequency, uses no factors and has no tunable settings.
The settings/trials expander shows original versus chosen values. If tuning
selects the existing settings, the app explicitly explains why the prediction
scores and portfolio results match.

Hover over the probability table's column headers:

| Metric | Meaning | Direction / reference |
| --- | --- | --- |
| Brier score | Mean squared error `(p - y)^2` of the event probability. | Lower; perfect 0, constant 50% gives 0.25. |
| Log loss | Negative log probability assigned to the actual outcome; confident mistakes cost more. | Lower; perfect 0, constant 50% gives about 0.693. Search objective. |
| ROC AUC | Ability to rank event cases above non-event cases across thresholds. | Higher; chance 0.5, perfect 1. Not accuracy or calibration. |

The saved comparisons use **ANOVA-10**, frozen at December 2006. This downside-
selected input list is deliberately reused for upside; a new upside feature
selection was not performed. Fitting and tuning use labels known by that date;
reported results earn January 2007–December 2009. The feature source, exact
names and saved-study chooser are in **Feature inputs and saved comparison**.
Changing the Feature selection tab does not retrain this study. The earlier
choice of ANOVA versus pruning was not blind to these holdout years; the
hyperparameter search itself does not score them.

Both downside versions multiply weights by `1 - 0.5 * P(loss)`. Fully invested
rules rescale to 100% stocks; the annual-consistency cash rule leaves reductions
in cash. Both upside versions boost original weights by 50% when `P(up) >= 0.60`,
then rescale to the original equity budget. Existing cash and excluded stocks
are preserved. Neither threshold nor adjustment strength is tuned.

The grouped positive-volatility overlays use separate untuned all-20-factor
models and a different evaluation window; see [basket momentum](BASKET_MOMENTUM.md).
The ANOVA tuning studies do not tune the forest-selected study or complete the
official frozen final test. Browsing only reads saved results.

## Saved model/target coverage — 2026-11-09 local time

Artifact timestamps below are UTC (November 10 for the two additions).

| Target | Models | Saved study |
| --- | --- | --- |
| Next down | Logistic, histogram boosting (preserved) | `holdout_20261109T202045_b0a51718` |
| Next down | Naive Bayes, random forest | `holdout_20261110T012820_d20ce131` |
| Next up | All four models | `holdout_20261110T013013_6f2fd4f9` |

All use the same ten names from `features_20261108T153403_ddfe05c8`, the
logistic ANOVA fit at 2006-12-31, and all fourteen basic rules. Random forest
retains 100 trees with `--allow-slow`; there is no reduced-tree comparison.
Search selects logistic `C=0.1`, naive Bayes smoothing `1e-5`, forest depth 4 /
leaf size 200, and boosting's existing learning rate 0.05 / 15 leaves, for both
targets. The two additions use 63 search/refit fits plus 12 frozen portfolio-
prediction fits, occupying 1,083,315 bytes. The original two-model study stays
unchanged. No model beats its historical control on holdout Brier/log loss.
Tuned forest improves those scores relative to the untuned forest but remains
worse than the control. This does not choose a final submission winner.

The UI filters saved studies by actual target, model and strategy, newest first.
Historical files lacking a target are interpreted as downside. Missing combinations
show an explicit unavailable message and CLI recipe; they never display another
target's scores or start a search. Studies remain independent of the experiment
catalog and formal release pin. Shared readers also work in notebooks and Quarto;
those rendered deliverables have not been changed by this UI/results addition.

See [verification](verification/tuning_controls_20261109.json).

## What it does

`research/tuning.py` answers one question: given a feature recipe and one
training slice whose labels are already known, which small set of model
settings has the best probability score?

```text
feature recipe + labeled training rows
    → chronological GridSearchCV
    → winning settings refit on that whole slice
    → pipeline.joblib + trial table + fold table
```

The recipe is a `FeatureSpec`. A fixed column list uses those columns. ANOVA
or correlation pruning is a step inside the pipeline, so each fold selects
from its own training rows. The saved model is the refit pipeline, ready for
`predict_proba`.

The score is `neg_log_loss` (higher, meaning less negative, is better).
`neg_brier_score` is the other allowed score. Portfolio return, Sharpe, and
drawdown are not the search objective. Compare those later, on the chosen
settings only.

## Why the folds are not sklearn's default

Default classifier CV stratifies rows rather than enforcing forward month
blocks. It can train on later dates than a validation observation and split
one month's stocks across the two sides.

`MonthForwardSplit` keeps every stock from a month together. Folds are
contiguous later blocks. The training side of a block grows as the block moves
forward. A row dated `D` earns its return at the next month-end. That row can
train a fold only when the validation block starts on or after that earned
month. This is the same availability rule as `ModelData.training_slice`.

The default is 3 folds with 24 months before the first block. The search then
refits the winner on every row in the slice you passed. Those rows were already
known at `asof`. The refit is the model you would use at that origin. The folds
only choose the settings.

## The grids are declared and small

| Model | Searched settings | Candidates | Fits with 3 folds, including the refit |
| --- | --- | ---: | ---: |
| Logistic regression | `C` in {0.1, 1, 10} | 3 | 10 |
| Histogram gradient boosting | learning rate {0.05, 0.1} and leaves {15, 31} | 4 | 13 |
| Random forest | depth {4, 6} and leaf size {50, 200} | 4 | 13 |
| Gaussian naive Bayes | `var_smoothing` in {1e-9, 1e-5} | 2 | 7 |

The fixed defaults used everywhere else sit inside these grids. A search that
would fit more than 40 models raises before it starts.

Random forest still uses 100 trees in the factory. That grid is refused unless
you pass `fixed_params={"model__n_estimators": 40}` or `allow_slow=True`. The
saved file records which choice you made. Logistic regression plus histogram
boosting is the fast pair that covers two tunable model types.

## Save and reload

Experiment runs still store predictions and provenance, not estimator objects.
A tuning directory is separate. It is not a row in `artifacts/catalog.sqlite`,
and loading it does not fit.

```text
artifacts/tuning/<id>/
    pipeline.joblib
    trials.parquet
    folds.parquet
    tuning.json
```

`load_tuned_model` checks the three file hashes, then returns the pipeline.
`predict_event_probability` applies it to a frame that still has the recipe's
input columns.

## Call to use when the features are chosen

This uses the training CSV only. It does not open the test file.

```python
from pathlib import Path
from research import FeatureSpec, prepare_data
from research.tuning import (
    load_tuned_model, new_tuning_directory, predict_event_probability,
    save_tuned_model, tune_classifier,
)

root = Path(".").resolve()
recipe = FeatureSpec(("Div_Yld", "Fcf_Yld"))  # replace with the chosen recipe
prepared = prepare_data(root, evaluation="training_validation")
asof = prepared.model.labels.label_available_at.max()
training = prepared.model.training_slice(asof, recipe.columns)

for model_key in ("logistic", "hist_gradient_boosting"):
    result = tune_classifier(training, recipe, model_key, asof)
    save_tuned_model(result, new_tuning_directory())

loaded = load_tuned_model("artifacts/tuning/<id>")
probabilities = predict_event_probability(loaded, training)
```

One logistic search plus one boosting search on that full training slice is
the intended first run. Repeating the search at every annual origin is a
stability check, and it multiplies the cost by about five. Do that only if
the presentation needs to show that the chosen settings moved.

## Training holdout tab

The app tab **Tuned model** sits after Feature selection and before Final test.
It compares three books that share a strategy and a frozen feature list:

| Version | What is fit |
| --- | --- |
| No ML | The portfolio rule only. |
| ML, not tuned | The factory settings, one fit. |
| Tuned ML | The grid-search winner, refit once on the same rows. |

The training file is cut in two:

- **Inner train.** Labels known through December 2006. The feature names are
  the ones the saved feature study selected at the December 2006 fit. The grid
  search uses only those rows.
- **Inner test.** Returns earned January 2007–December 2009. Both models are
  frozen at December 2006 and applied to these months. This is the table on
  the tab.
- **Official test.** January 2010–December 2018 stays on Final test. This
  comparison does not open that file.

The earlier feature study also scored 2007–2009, so picking ANOVA rather than
correlation pruning is not blind to the holdout. The column list used here is
still the December 2006 fit, and the hyperparameter search does not see the
holdout returns.

```bash
python3 run_tuning_holdout.py
python3 run_tuning_holdout.py --recipe pruned_10
python3 run_tuning_holdout.py --target upside --models logistic naive_bayes random_forest hist_gradient_boosting --allow-slow
```

The result is saved under `artifacts/tuning_studies/` and is not an
experiment-catalog run. Browsing the tab does not fit.
