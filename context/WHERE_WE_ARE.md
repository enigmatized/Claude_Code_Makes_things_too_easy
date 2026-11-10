# Where the project is

Snapshot 2026-11-09. Source of truth if this file is stale:
`docs/PROJECT_STATE.md`. Last living-doc write seen while making this snapshot:
2026-11-08 22:21 local. Newest saved run seen:
`artifacts/20261109T040626_e0a702234197` (created 2026-11-09 04:06 UTC).

## The assignment file is still the prompt

`03_mlfactor_long_only_project.ipynb` has five markdown cells and no code.
Points: notebook 175, Quarto report 25, presentation 100.

Notebook rubric inside those 175:

| Ask | Points called out | Status on 2026-11-09 |
| --- | --- | --- |
| Correct equal-weight market annualized return | 30 | Computed. Not yet the center of the turn-in notebook. |
| Accuracy of results | 70 | Training research is real. Official frozen test of a chosen spec is not. |
| Readability | 50 | Draft `submission/analysis.ipynb` is short. It is still a research draft. |
| Code comments | 25 | Logic is in `research/`. The turn-in notebook barely comments the calls. |

The six analysis tasks:

1. **≤10 features, and whether the subset helps the book.** Two attempts exist.
   Part 1 (`part1_feature_selection/`, 2026-10-24) locked a 10-name ridge list
   and found no clear improvement. The later study
   `features_20261108T153403_ddfe05c8` compares all 20 vs ANOVA-10 vs
   correlation-pruned 10 inside each annual fit. Effects are small and mixed.
   Masks change by year, so neither study is yet a single frozen ≤10 submission
   model. The pinned report run still uses all 20, so the readiness check
   "at most 10 features" is false.
2. **Four model types, at least two tunable.** Done as fixed classifiers:
   logistic regression, Gaussian naive Bayes, random forest, histogram
   gradient boosting. Logistic and the two tree models are the tunable ones.
   Naive Bayes is the fourth type.
3. **`GridSearchCV` on training data, then ask whether ML scores improve the
   book.** Not done. `build_status.json` records this check as false. No
   tuning artifacts exist.
4. **Two non-ML strategies, one of them equal weight.** Done many times over.
   Fourteen registered rules. The graded story should feature equal weight
   plus one chosen second rule, not all fourteen.
5. **Risk and reward.** CAGR, volatility, Sharpe, Sortino, drawdown, turnover,
   and cash are saved for the research runs. The final comparison of the
   chosen spec on the test window is not.
6. **Final model and trading-rule choice, justified.** Not recorded.
   `final_choice_case_id` is null.

## What one row is

Train `mlfactor_long_only_train.csv`: 140,739 rows, 120 month-end snapshots,
1999-12-31 through 2009-11-30. Those rows earn January 2000–December 2009.

Test `mlfactor_long_only_test.csv`: 127,740 rows, 108 snapshots, 2009-12-31
through 2018-11-30. Those rows earn January 2010–December 2018.

Columns: `date`, `stock_id`, 20 factor percentiles, `R1M_Usd`. No prices,
tickers, sectors, countries, or market cap. No missing values. Features sit
on about [0.01, 1] with 100 distinct values. `R1M_Usd` is the forward
one-month **simple** USD return (train min about −0.92, max about 27.2;
nothing ≤ −1).

Equal weight has no cap weights, so the month's portfolio return is the
cross-sectional mean of `R1M_Usd`. Portfolio log return, already saved, is
`log(1 + that simple portfolio return)`. Do not average stock log returns.

## The 30-point number

From `part0_data_and_benchmark/outputs/equal_weight_metrics.csv`, full test
window, 108 months:

- CAGR **13.096%**
- volatility 15.63%
- Sharpe 0.868
- max drawdown −20.64%

CAGR here is `wealth ** (12 / n) − 1`, the monthly version of the chapter 15b
compound return. The training-label window is a different number (CAGR about
14.50% over 120 months). The 2005–2009 validation window is a third number.
The graded figure is the **test-period** equal-weight annualized return.
`submission/release.json` pins the hash of that CSV.

## What already runs

- `research/` fits, accounts, and saves. Importing it does not start a run.
- `python3 run_app.py` serves the Streamlit app at http://localhost:8501.
  Five tabs: Strategies, Basic ML, Advanced ML (empty on purpose), Feature
  selection, Final test. Saved Parquet is read first. Browsing does not fit.
- Default research window: first 60 training snapshots as history, evaluate
  earned January 2005–December 2009, refit once a year, `fit_policy="expanding"`.
- `fit_policy="frozen_at_origin"` exists and is the one the assignment wants
  for the official test. It has not been used as the published test.
- Catalog at snapshot time: twelve training runs, no registered test run.
  Final test tab can still show older archived non-ML 2010–2018 CSVs. Those
  are historical, not the frozen chosen-model test.
- Draft package: `python3 build_submission.py` executes
  `submission/analysis.ipynb` and renders Quarto HTML with fitting blocked.
  `--final` correctly refuses. The pin is the older expanding run
  `20261108T144257_f45441ba8e83`, not the newest artifact and not the feature
  study.

## Evidence worth keeping straight

| Piece | Where | Use in the final story |
| --- | --- | --- |
| Data dictionary and equal-weight check | `part0_data_and_benchmark/` | The 30-point benchmark. Brief in the talk. |
| Early ridge ≤10 study | `part1_feature_selection/` | Historical. Do not silently replace it. Its decision was "no clear improvement." |
| Fourteen rules and four fixed models | `0_1_Basic_Strategies/`, `0_2_Strategy_Runner/`, run `20261108T144257_f45441ba8e83` and later 70-case runs | Pick one second non-ML rule from here. |
| ANOVA 10 vs pruned 10 | `0_3_Feature_Selection/` | Feature-selection analysis. Small, mixed. Not a frozen ten. |
| Cash + momentum window | `0_4_Consistency_Momentum_Cash/`, `0_5_Momentum_Lookbacks/` | Optional featured rule. 3-month led 2005–2009 return/Sharpe; 12-month led drawdown and turnover. Not a declared winner. |
| Newest run | `20261109T040626_e0a702234197` | Same fourteen rules, four models, all 20 factors, expanding, cash lookback 3 months, adjustment strength 0.8. Training validation only. |

Prediction quality on the first feature study is weak. Historical loss-rate
control Brier **0.249023**, log loss **0.691215**. No model/recipe beat that
control. AUCs sit around 0.50–0.51. Portfolio effects of dropping to ten
factors were tenths of a percentage point of CAGR, not a new strategy.

## Limits that belong in the talk

Gross of costs. Sharpe and Sortino use a zero risk-free/target rate. Cash, where
a rule holds it, earns zero. No market-cap weights. Extreme simple returns are
left in the raw books. The test file was already opened during earlier work, so
the 2010–2018 result is a disclosed evaluation, not a fresh holdout. The
advanced-ML tab stays empty until the user defines that strategy.
