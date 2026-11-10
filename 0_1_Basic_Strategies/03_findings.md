# Findings from the executed basic-strategy comparison

Source: `02_basic_strategy_comparison.ipynb`. Rules: twelve realized months, 20% baskets, monthly rebalancing, raw supplied returns, no ML or tuning.

Training comparison: earned January 2001–December 2009 (after warm-up). Test comparison: January 2010–December 2018. Earlier test inspection is documented; this is not a fresh holdout.

## Train

| Strategy | CAGR | Volatility | Sharpe (rf=0) | Max drawdown | Turnover |
| --- | --- | --- | --- | --- | --- |
| Equal-weight universe | 13.93% | 20.61% | 0.739 | -49.51% | 3.90% |
| Equal-weight eligible stocks | 13.70% | 20.54% | 0.731 | -49.51% | 3.91% |
| High volatility, equal weight | 28.97% | 34.61% | 0.906 | -57.95% | 14.79% |
| Low volatility, equal weight | 6.97% | 12.26% | 0.614 | -40.07% | 13.11% |
| Positive momentum, equal weight | 6.81% | 18.51% | 0.450 | -48.88% | 26.53% |
| Positive momentum, score weights | 10.91% | 21.19% | 0.597 | -53.45% | 24.58% |

## Test

| Strategy | CAGR | Volatility | Sharpe (rf=0) | Max drawdown | Turnover |
| --- | --- | --- | --- | --- | --- |
| Equal-weight universe | 13.10% | 15.63% | 0.868 | -20.64% | 2.89% |
| Equal-weight eligible stocks | 13.10% | 15.63% | 0.868 | -20.66% | 2.88% |
| High volatility, equal weight | 18.88% | 23.44% | 0.857 | -29.21% | 13.68% |
| Low volatility, equal weight | 10.23% | 10.86% | 0.955 | -11.51% | 14.21% |
| Positive momentum, equal weight | 11.08% | 17.17% | 0.699 | -23.59% | 25.06% |
| Positive momentum, score weights | 8.71% | 19.67% | 0.523 | -28.81% | 23.08% |

- Train: high-volatility CAGR minus low-volatility CAGR is +22.01%; volatility difference is +22.35%.
- Train: momentum score weights change CAGR by +4.10% and Sharpe by +0.147 relative to equal weights on the same names.
- Train: momentum score weighting holds 226.9 names on average but 114.4 effective holdings; its largest single weight is 32.02%.
- Test: high-volatility CAGR minus low-volatility CAGR is +8.64%; volatility difference is +12.58%.
- Test: momentum score weights change CAGR by -2.37% and Sharpe by -0.176 relative to equal weights on the same names.
- Test: momentum score weighting holds 236.8 names on average but 113.1 effective holdings; its largest single weight is 16.53%.

## Interpretation and next experiment

High return alone does not settle the comparison: assess volatility, drawdowns, turnover, concentration, and year-by-year behavior together. Positive historical momentum describes the past; it is not a guarantee of a continuing trend.

The eligible equal-weight control separates the history requirement from selection. The two momentum books isolate the effect of allocation on a common selected basket. These are more informative comparisons than choosing a winner from final wealth alone.

The next feature-selection experiment can compare a simple original-factor subset with all twenty features and these non-ML controls. Choose transformations, subsets, and model parameters using training-period chronological validation. Keep volatility and momentum separate from the original twenty-feature-subset requirement.

No feature-selection list or Part 1 result was changed. No costs, cash timing, or weight caps were added in response to these results. Extreme supplied returns remain in the books; inspect `outputs/largest_stock_return_contributions.csv` before drawing conclusions about robustness.
