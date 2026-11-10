# A short menu for later work

Finish the benchmark before choosing a second non-ML strategy. The following are
ideas, not claims of superior performance, and are not fitted in Part 0.

| Idea | Why study it | What this dataset requires |
| --- | --- | --- |
| Single-factor top-quintile book | Transparent non-ML comparator; separates a factor idea from ML complexity. | Choose factor, direction, and tie rule on training data. Rebalance monthly. |
| A few equally averaged factor ranks | Tests the earlier idea of combining related measurements. | Declare economic direction; do not interpret averaged ranks as raw ratios. |
| Inverse-volatility allocation | Course chapter 15b; compare equal capital with less capital in volatile names. | Estimate only from returns already realized before each rebalance; define minimum history and fallback. |
| Risk parity / minimum variance | Chapter 15b covers risk parity; covariance-aware weighting is a further experiment. | A roughly 1,200-stock covariance matrix from 120 training months is unstable. Study shrinkage or small groups before a large optimizer. |
| Momentum | Uses past performance as a transparent ranking signal. | Build trailing histories using earned-month timing; current `R1M_Usd` is future information at its snapshot. |
| Trend filter with cash | Course chapter 16; explores staying out during weak trends. | Requires an explicit cash return and extension of the current fully invested engine. |

For all comparisons, first report CAGR, annualized volatility, zero-rate Sharpe,
maximum drawdown, and turnover on identical months. Keep the number of variants
small enough that each one answers a question.

Additional tools, when a specific question calls for them:

- A log wealth axis makes proportional changes easier to compare. It does not
  change the underlying investment strategy.
- Sortino measures return relative to downside variability, with a stated target.
  Calmar compares CAGR with the magnitude of maximum drawdown; it depends heavily
  on the chosen historical window.
- Tracking error describes variability of returns relative to the equal-weight
  benchmark. Information ratio divides mean active return by that variability.
  Neither should be relabeled as the portfolio's Sharpe ratio.
- Rolling volatility and rolling Sharpe can show changing conditions, but short
  windows are noisy. Monthly data cannot reveal intramonth drawdowns or daily VaR.
- VaR and expected shortfall describe tail losses at a stated horizon. With only
  120 training months, a 1% empirical tail contains about one observation; avoid
  presenting that as a precise estimate.

Relevant local course sources:
[15b: asset allocation](../../p4dsf/chapters/15b_asset_allocation/asset_allocation.ipynb),
[16: trend following](../../p4dsf/chapters/16_trend_following/trend_following.ipynb),
[25: stock prediction](../../p4dsf/chapters/25_stock_prediction/stock_prediction.ipynb),
and [26a: PCA](../../p4dsf/chapters/26a_pca/pca.ipynb).
For Sharpe's distinction between mean differential return and its volatility,
see [William Sharpe's original discussion](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm).
