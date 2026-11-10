# Training high volatility plus monthly momentum: executed findings

Source: `05_high_volatility_momentum.ipynb`. Rule specification: `04_hybrid_rules.md`.

The fixed screen contains 241 stocks from 1203 with enough training history. Membership uses the 2000–2009 training returns only, and is frozen for the test period.

Monthly momentum selection takes up to the top 20% of eligible screened names with positive twelve-month returns. Scores determine weights without a cap. Gross results use risk-free return zero.

## January 2010–December 2018

| Strategy | CAGR | Volatility | Sharpe | Max drawdown | Average names |
| --- | --- | --- | --- | --- | --- |
| Equal-weight universe | 13.10% | 15.63% | 0.868 | -20.64% | 1182.8 |
| Monthly rolling high-vol basket, equal weight | 18.88% | 23.44% | 0.857 | -29.21% | 236.8 |
| Positive momentum, score weights | 8.71% | 19.67% | 0.523 | -28.81% | 236.8 |
| Training high-vol universe, equal weight | 17.83% | 21.53% | 0.871 | -28.27% | 236.7 |
| Training high vol + momentum, equal weight | 18.68% | 28.06% | 0.740 | -27.89% | 47.6 |
| Training high vol + momentum, score weights | 10.86% | 29.06% | 0.494 | -46.44% | 47.6 |

- Hybrid score weighting versus the frozen-list equal-weight control: CAGR difference -6.97 percentage points, Sharpe difference -0.377.
- Score weighting versus equal weights on the SAME selected momentum names: CAGR difference -7.83 percentage points, Sharpe difference -0.245.
- The score-weighted hybrid averages 47.6 actual holdings and 24.7 effective holdings. Its largest single-stock weight is 40.55%; average one-way turnover is 21.25%.
- Recorded hybrid fallback months: 0.

## Interpretation

The frozen training screen and monthly rolling high-volatility basket are different rules. The two controls separate changes in the candidate universe, monthly selection, and capital allocation. Compare reward, drawdowns, concentration, and turnover together rather than only final wealth.

The final training screen is not applied retrospectively to earlier training years. An honest training process check would refit it at earlier origins. The current result is a historical test-period comparison requested after earlier test results were seen, not fresh validation.

Source returns retain extreme observations. No parameter tuning, return clipping, cash timing, weight cap, or original-feature selection was added. Original basic-strategy results, Part 0, and Part 1 were preserved.
