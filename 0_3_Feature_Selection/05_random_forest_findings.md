# Random-forest selection: matched training findings

Completed study: `features_20261109T202318_8b01fe74`.

| Recipe | Saved run |
| --- | --- |
| All twenty | `20261109T202421_184e95372ab2` |
| ANOVA ten | `20261109T202516_039a5a86258b` |
| Correlation-pruned ten | `20261109T202612_7da383c77e5b` |
| Forest ten | `20261109T202931_597510736fb4` |

The study compares all fourteen current rules and four pooled downside
classifiers on the same Jan 2005–Dec 2009 earned months. Selection and classifier
fits expand annually using only then-known labels. Fixed allocation strength is
0.50; the annual-consistency cash rule uses three-month momentum. There is no
hyperparameter search or official test evaluation in this feature study.
The separate Tuned model holdout in the app is different evidence.

## What the forest selected

At the first fit (Dec 2004), in descending importance order:

1. Div_Yld
2. Ocf_Ce
3. Fcf_Yld
4. Tev_Less_Mktcap
5. Share_Turn_3M
6. Ev_Ebitda
7. Share_Turn_6M
8. Capex_Sales
9. Share_Turn_12M
10. Tot_Debt_Rev

The forest selects ten factors at each of five origins. Its annual union has
eleven names, not one final eleven-factor model. At the first origin it shares
eight names with ANOVA and six with pruning. The selector settings are 100
trees, depth 6, minimum leaf 50, sqrt feature subsampling and seed 0. The saved
audit records the full configuration, mean importance and standard deviation
across trees. These error bars are descriptive tree variation, not confidence
intervals. Importance does not show the direction of a factor's effect.

## Prediction quality

Lower Brier score and log loss are better. Historical control: Brier 0.249023,
log loss 0.691215. None of the sixteen model/recipe combinations beats it on
either score.

| Predictor | All-20 Brier | Forest-10 Brier | Change | Forest-10 log loss |
| --- | ---: | ---: | ---: | ---: |
| Logistic | 0.249667 | 0.249474 | -0.000192 | 0.692174 |
| Gaussian naive Bayes | 0.259759 | 0.254489 | -0.005270 | 0.702917 |
| Random forest | 0.249387 | 0.249515 | +0.000128 | 0.692300 |
| Histogram boosting | 0.249715 | 0.249769 | +0.000054 | 0.692835 |

Forest selection helps logistic and naive Bayes relative to all twenty, but
hurts forest and boosting. ANOVA gives logistic a slightly lower Brier score
than forest selection; pruning gives naive Bayes a slightly lower score.
This does not support declaring forest selection the best recipe.

## Portfolio example

Logistic overlay on the annual-consistency / positive three-month momentum /
cash rule, over the same five years:

| Inputs | CAGR | Sharpe | Max drawdown | Mean cash |
| --- | ---: | ---: | ---: | ---: |
| All twenty | 3.510% | 0.435 | -17.558% | 56.374% |
| ANOVA ten | 3.523% | 0.436 | -17.559% | 56.343% |
| Pruned ten | 3.514% | 0.435 | -17.564% | 56.357% |
| Forest ten | 3.522% | 0.436 | -17.561% | 56.328% |

The non-ML original has CAGR 4.42%, Sharpe 0.437 and max drawdown -22.03%.
These are very small differences between feature recipes. The downside overlay
also raises cash exposure, so its drawdown reduction is not evidence by itself
of useful classification. All results are gross of costs; assumed cash return
is zero. The full exports cover every predictor and all fourteen rules.

## Review and reproduce

- App: **Feature selection → Selected features & method → Random forest top 10**.
  Then inspect **Effect on ML models**, including calibration and log paths.
- [Executed review notebook](04_random_forest_review.ipynb).
- [Review HTML](outputs/features_20261109T202318_8b01fe74/04_random_forest_review.html).
- [Importance chart](outputs/features_20261109T202318_8b01fe74/forest_importance.png).
- [Shared method notes](method_notes.md).
- [Exported evidence](outputs/features_20261109T202318_8b01fe74/exports.json).

```bash
python3 run_feature_selection.py
MLFACTOR_READ_ONLY=1 python3 export_feature_study.py features_20261109T202318_8b01fe74
python3 build_submission.py --pin-feature-study features_20261109T202318_8b01fe74
```

The initial selector smoke fit took 5.57 seconds on 69,108 rows. The completed
study took about 375.5 seconds from written plan to saved completion; its four
full runs occupy 143.5 MiB. The four all-20 probability tables and all seventy
matching books/weight histories reproduce the preceding matched all-20 run
exactly. The earlier three-recipe study remains unchanged.

Next: review the feature/strategy trade-offs before fixing final specifications.
Keep any forest-selected tuning comparison explicit and distinct from the
existing ANOVA training holdout. Final choice, tuning release integration and
the official frozen-training test remain later work.
