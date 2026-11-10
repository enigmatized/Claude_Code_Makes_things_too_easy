# First feature-selection comparison

Reducing the inputs to ten produces small, mixed changes. Neither selection
recipe establishes a clear overall advantage. In particular, no model/recipe
beats the historical loss-frequency control on Brier score or log loss.

These are training-period validation results, earned Jan 2005–Dec 2009. The test
CSV was not loaded. Models and selectors refit annually on expanding historical
training slices. Read the [fixed protocol](01_experiment_plan.md) for timing and
selection rules. No hyperparameters or selection thresholds were searched.

## Prediction quality

Lower Brier score means smaller squared probability errors. The common
historical loss-frequency control scores **0.249023**, with log loss **0.691215**.
It is fitted from past outcomes, rather than from factor inputs.

| Fixed classifier | All 20 Brier | ANOVA 10 Brier | Pruned 10 Brier |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.249667 | 0.249434 | 0.249507 |
| Gaussian naive Bayes | 0.259759 | 0.257497 | 0.254021 |
| Random forest | 0.249387 | 0.249428 | 0.249548 |
| Histogram gradient boosting | 0.249715 | 0.249683 | 0.249826 |

ANOVA improves logistic and naive Bayes probability scores relative to their own
all-20 versions. Pruning improves naive Bayes most, but its absolute probability
scores remain worse than the control. Redundant factors may be particularly
unhelpful to naive Bayes' conditional-independence assumption; this result is
consistent with that explanation, without proving it. AUCs remain around
0.50–0.51. No significance claim is made from these differences.

## Portfolio effects

Selection changes only the ML probability inputs. Each portfolio keeps its
original trading rule, eligible stock pool and adjustment strength of 0.5.
Original non-ML portfolio calendars and returns match exactly across all recipes.

Across the 52 adjusted portfolios (13 rules × 4 models), changes in annualized
return versus the same model using all 20 inputs are:

| Recipe | Smallest change | Largest change | Mean change |
| --- | ---: | ---: | ---: |
| ANOVA 10 | −0.163 percentage points | +0.083 percentage points | +0.007 percentage points |
| Pruned 10 | −0.477 percentage points | +0.184 percentage points | −0.026 percentage points |

These 52 portfolios share data and often similar holdings; their average is a
description of the experiment, not independent statistical evidence. Prediction
quality and portfolio returns must be judged separately. For example, pruned
naive Bayes improves its Brier score but decreases the momentum-weighted
portfolio's CAGR by about 0.159 percentage points relative to all-20 naive Bayes.

The app shows each strategy/model comparison, including wealth, drawdown and
relative wealth. Returns are gross of trading costs; the ML adjustment keeps the
portfolio fully invested. Selection does not by itself solve market drawdowns.

## What was selected?

Both learned recipes used exactly ten factors at each of the five annual fits.
All four models received identical masks for the same recipe and fit date.
ANOVA selected 12 different names across the five fits; pruning selected 11.
Those unions are not single fitted models. A frozen submission model would
select once before its evaluation period and use at most ten inputs throughout.

The first fit, at December 2004, selected the following:

| ANOVA top 10 | Correlation-pruned 10 |
| --- | --- |
| Tot_Debt_Rev | Tot_Debt_Rev |
| Share_Turn_3M | Net_Debt |
| Debtequity | Total_Liabilities_Total_Assets |
| Ocf_Ce | Debtequity |
| Fcf_Yld | Ocf_Ce |
| Div_Yld | Fcf_Yld |
| Share_Turn_12M | Ocf_Tbv |
| Tev_Less_Mktcap | Div_Yld |
| Total_Debt_Capital | Share_Turn_12M |
| Share_Turn_6M | Tev_Less_Mktcap |

Names above follow original input order, not importance order. Review scores,
reasons for exclusions, meanings, correlations and changes over time in the app.
F scores rank association with loss events; they are not causal explanations or
valid panel-data significance tests. The supplied factor values are normalized
percentile-like inputs, not raw accounting ratios.

## Reproducibility and next step

Study: `features_20261108T153403_ddfe05c8`.

| Recipe | Immutable run ID |
| --- | --- |
| All 20 | 20261108T153503_eba121c8904e |
| ANOVA 10 | 20261108T153555_a41c41a1c46a |
| Pruned 10 | 20261108T153648_ce8ca5b600a2 |

The [review notebook](03_feature_selection_review.ipynb) reads these saved
artifacts and never trains. Its HTML export is in
[outputs/](outputs/03_feature_selection_review.html). The app's **Feature
selection** tab loads the same study and supports CSV downloads. Full predictions
and weight histories are preserved under the linked runs in `artifacts/`.

Review this evidence before choosing a final recipe. The next research stage is
a small, declared training-only GridSearchCV comparison with selection inside
each chronological fold. The submission release remains pinned to the earlier
draft; this validation study is not a tuned model or a final test backtest.
