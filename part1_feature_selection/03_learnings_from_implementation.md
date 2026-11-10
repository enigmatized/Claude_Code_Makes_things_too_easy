# Part 1 learnings from implementation

Status: **filled from the executed notebook** `02_part1_feature_selection.ipynb`. Numbers below are that run. This file does not re-decide the method in `01_plan.md`.

## 1. Subset the rule produced

`subset_final` has 10 columns, in greedy order: Ev_Ebitda, Net_Debt, Div_Yld, Fcf_Yld, Tot_Debt_Rev, Ocf_Ce, Share_Turn_12M, Tev_Less_Mktcap, Fcf_Tbv, Total_Liabilities_Total_Assets.

`weak_signal_fallback` is False.

Scorecard path: `feature_scorecard.csv`. Chosen rows: `subset_final.csv`.

Yearly lists, each chosen only on snapshots before the origin:

- Origin 60 (last seen 2004-11-30, 1 fold(s), fallback False): Fcf_Yld, Tev_Less_Mktcap, Ev_Ebitda, Netdebtyield, Ocf_Ce, Share_Turn_12M, Net_Debt, Div_Yld, Debtequity, Bb_Yld. Recall against subset_final: 0.70.
- Origin 72 (last seen 2005-11-30, 2 fold(s), fallback False): Fcf_Yld, Ev_Ebitda, Div_Yld, Netdebtyield, Ocf_Ce, Tev_Less_Mktcap, Net_Debt, Share_Turn_12M, Capex_Sales, Tot_Debt_Rev. Recall against subset_final: 0.80.
- Origin 84 (last seen 2006-11-30, 3 fold(s), fallback False): Fcf_Yld, Ev_Ebitda, Tev_Less_Mktcap, Netdebtyield, Div_Yld, Ocf_Ce, Share_Turn_12M, Net_Debt, Tot_Debt_Rev, Total_Liabilities_Total_Assets. Recall against subset_final: 0.90.
- Origin 96 (last seen 2007-11-30, 4 fold(s), fallback False): Div_Yld, Ev_Ebitda, Ocf_Ce, Fcf_Yld, Net_Debt, Tot_Debt_Rev, Ocf_Toa, Share_Turn_12M, Netdebtyield, Total_Liabilities_Total_Assets. Recall against subset_final: 0.80.
- Origin 108 (last seen 2008-11-30, 5 fold(s), fallback False): Ev_Ebitda, Net_Debt, Share_Turn_12M, Fcf_Yld, Ocf_Ce, Div_Yld, Tev_Less_Mktcap, Tot_Debt_Rev, Capex_Sales, Ocf_Toa. Recall against subset_final: 0.80.

Median recall of those lists against `subset_final` is 0.80. The precommitted stability line is 0.50.

The first origin has one lasso fold, so its frequency is 0 or 1. That origin is included in the median.

## 2. Did the diagnostics match the plan's scouting measurements?

- Train shape 140,739 rows and 120 snapshots, test shape 127,740 rows and 108 snapshots, dates as in the plan, no missing values, no duplicate keys, earned-return months disjoint.
- Multi-column clusters at distance 0.20 match the plan's scouting clusters (the leverage trio, the three turnover windows, and the two tangible-book cash-flow columns).
- Largest training |mean Spearman IC| is 0.0113. Columns with |t| >= 2: 0. The plan's scouting pass found none.
- 146 training rows with |return| > 1 account for 50.0% of the sum of squared training returns. Training max return 27.23. Test max return 30.18.
- Lasso convergence warnings inside the selection folds: 0. Path-picture warnings: 0.
- Mean-squared-error alpha equals the top of the grid on 6 of 6 full-sample folds, and those refits keep 0 nonzero coefficients. The 0.002 band is wide relative to the mean-squared-error gaps, so the side column records the sparsest alpha and, on this grid, an empty set. It does not vote.

## 3. Walk-forward inside the training years

Source file: `yearly_walkforward.csv`. Earned returns January 2005 through December 2009 (60 months).

- Ridge, all 20 features: annualized return 9.31%, volatility 23.98%, Sharpe 0.491, max drawdown -49.03%.
- Ridge, refreshed subset: annualized return 5.06%, volatility 21.32%, Sharpe 0.340, max drawdown -45.41%.
- Equal-weight universe: annualized return 7.62%, Sharpe 0.433.
- Monthly difference, subset minus all-20: mean -0.00377, textbook t -1.86, Newey-West t -1.51.
- Sharpe gap (subset minus all-20): -0.151.

This path is the evidence that the restriction was chosen before the year it traded. It is not the official test.

## 4. Test backtest, one pass

Return window: January 2010 through December 2018 (snapshots 2009-12-31 through 2018-11-30).

Source file: `strategy_comparison.csv`. The table is the raw view.

| Strategy | Ann. return | Ann. vol | Sharpe | Max drawdown | Mean IC | Months ahead of EW | Avg one-way turnover |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Equal-weight universe | 13.10% | 15.63% | 0.868 | -20.64% |  |  | 0.08% |
| Ridge, all 20 | 16.28% | 18.25% | 0.921 | -22.46% | 0.0015 | 57.41% | 13.06% |
| Ridge, subset | 15.60% | 17.34% | 0.926 | -20.89% | 0.0062 | 58.33% | 13.03% |
| Forest, all 20 | 19.81% | 20.85% | 0.975 | -25.45% | -0.0058 | 62.04% | 14.00% |
| Forest, subset | 17.71% | 20.84% | 0.890 | -25.64% | -0.0043 | 54.63% | 13.75% |

- Paired monthly difference, subset minus all-20, Ridge, raw returns: mean -0.00062, textbook t -0.42. Newey-West t = -0.44 (lag 3, S = 0.000211509).
- Same difference after capping each stock's return at 100%: mean 0.00060, textbook t 0.65. Sharpe gap 0.089.
- Same difference after the 10 bp turnover haircut: mean -0.00062, textbook t -0.42. Sharpe gap 0.005. The haircut ranking agrees with the raw Sharpe ranking. The haircut does not enter the decision label.
- Forest Sharpe gap (subset minus all-20) -0.085, paired mean -0.00149, textbook t -1.75. The forest does not agree with the ridge on the sign of the Sharpe gap.
- Average names held — ridge all-20 237.0, ridge subset 237.0, forest all-20 237.0, forest subset 237.0.

Capped-view Sharpes: ridge all-20 0.749, ridge subset 0.838.
Cost-haircut Sharpes: ridge all-20 0.912, ridge subset 0.918.

## 5. Decision sentence

Restricting the model to 10 features does not clearly improve the long-only top-quintile strategy.

Rule trace: the test Sharpe gap and the walk-forward Sharpe gap disagree.

The deployed list has 10 original columns: Ev_Ebitda, Net_Debt, Div_Yld, Fcf_Yld, Tot_Debt_Rev, Ocf_Ce, Share_Turn_12M, Tev_Less_Mktcap, Fcf_Tbv, Total_Liabilities_Total_Assets. Families represented: valuation multiple, leverage, dividend yield, cash-flow yield, cash flow, share turnover, enterprise value minus market cap, cash flow over tangible book. On the ridge that trades the test window, the preferred side of each kept column is: Ev_Ebitda (low percentile, ridge coef -0.001); Net_Debt (low percentile, ridge coef -0.002); Div_Yld (low percentile, ridge coef -0.001); Fcf_Yld (high percentile, ridge coef +0.001); Tot_Debt_Rev (low percentile, ridge coef -0.000); Ocf_Ce (low percentile, ridge coef -0.000); Share_Turn_12M (low percentile, ridge coef -0.001); Tev_Less_Mktcap (high percentile, ridge coef +0.001); Fcf_Tbv (low percentile, ridge coef -0.001); Total_Liabilities_Total_Assets (high percentile, ridge coef +0.000). Columns left out because |correlation| with a kept column is at least 0.80: Share_Turn_3M (|corr| 0.95 with Share_Turn_12M); Debtequity (|corr| 0.84 with Tot_Debt_Rev); Ocf_Tbv (|corr| 0.86 with Fcf_Tbv); Total_Debt_Capital (|corr| 0.84 with Tot_Debt_Rev); Share_Turn_6M (|corr| 0.98 with Share_Turn_12M). Eligible columns that lost a seat and are not that tightly tied to a kept column: Ocf_Toa, Netdebtyield, Bb_Yld, Capex_Sales, Ocf_Ta. Eligibility alone supplied at least 3 names, so the weak-signal fallback stayed off.

The feature list was not edited after these test numbers were computed.

## 6. Deviations from the plan

No deviation changes the subset or the test. The following are readings of details the plan left unspecified, fixed in the notebook before the test pass:

- Undefined Spearman months are dropped from an alpha's average. If every inner month is undefined, the largest alpha is used.
- Hit rate counts months with Spearman IC strictly above 0.
- An undefined permutation drop is stored as 0.
- Each fold's shuffles start at `RandomState(0)`, so a repeated fold is reused. The reuse does not read later months.
- The first backtest month has turnover 0. Reported average turnover starts at the second month. The haircut uses that series.
- `XGBRegressor(verbosity=0)` only quiets the log.
- "Large disagreement" of the yearly lists is operationalized as median recall below 0.50.
- The decision priority is: disagreement, cap-ranking flip, or unstable identity; then the similar-performance band; then improvement; otherwise no clear improvement.
- Ranking under the cap means which ridge book has the higher Sharpe.

## 7. Mistakes, dead ends, and fragile points

- The mean-squared-error side column does not distinguish columns on this grid. The 0.002 tolerance was written for a Spearman score near 0.01. Applied to negative mean squared error, it covers the whole path, the largest alpha wins, and that alpha zeros the coefficients. The score ignores the column. A later part that wants mean squared error to vote has to give it its own tolerance.
- On several folds the Spearman-chosen alpha is small, so lasso leaves most coefficients nonzero. Frequency is then a weak screen, eligibility is wide, and the cap of 10 is what cuts the list. The correlation gate still removes a later duplicate when one is reached before the cap fills. Distinct eligible columns can lose a seat purely because ten higher scores already filled the list.
- The crisis validation fold can have a negative inner Spearman even at the best alpha. Frequency still counts the columns that fold selects. One bad year pulls a column's frequency down. That is the stability the folds are there to measure.
- The first walk-forward origin has a single fold. Treat that year's list as a coarse draw.
- Ridge with alpha 1 is the same model for both column sets. A gap, or the absence of a gap, is about the columns. A later part can change the model.
- Name confidence is low for `Ocf_Ce` and `Netdebtyield`. The score does not use the glossary. The prose description does.
- A shallow forest can tie many names at the quintile cutoff. In this run every book held 237 names on average, so the tie rule kept the forest book at the same width as the ridge book.

## 8. What a reviewer can try

A better part 1 beats the raw-view test table above on the long-only book, or matches it with a simpler and more stable subset, with the test labels still unused during selection. The current baseline is ridge subset Sharpe 0.926 against ridge all-20 Sharpe 0.921, together with walk-forward Sharpe gap -0.151 and median yearly recall 0.80.

Each idea changes one knob:

- The score weights `0.40 / 0.40 / 0.20`.
- The correlation gate `0.80`, including the choice to keep two columns when their correlation is high and their lasso signs differ. `Net_Debt` and `Tev_Less_Mktcap` are the pair the 0.70 picture merges and the 0.80 gate does not.
- The eligibility gates.
- Choosing the lasso alpha from `lasso_mse_frequency`'s rule instead of from Spearman, with a tolerance that is meaningful in mean-squared-error units.
- A within-month rank of `R1M_Usd` as the fit target, in place of the clipped demeaned return.
- A fixed name count in the book, pre-committed before the test pass, in place of the 80th-percentile rule with ties.
