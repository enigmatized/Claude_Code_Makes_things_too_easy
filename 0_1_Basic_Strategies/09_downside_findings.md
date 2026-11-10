# Three-month downside-weighted sizing: executed findings

Source: `08_three_month_downside.ipynb`. Rules: `07_downside_rules.md`.

The new rule retains the hybrid stock selection and divides positive momentum by a three-month downside-weighted risk measure. The ordinary-volatility control uses the same three months, selected stocks, and 1% monthly risk floor.

## January 2010–December 2018, gross returns

| Strategy | CAGR | Volatility | Sharpe | Sortino (target 0) | Max drawdown |
| --- | --- | --- | --- | --- | --- |
| Equal-weight universe | 13.10% | 15.63% | 0.868 | 1.456 | -20.64% |
| Training high-vol universe, equal weight | 17.83% | 21.53% | 0.871 | 1.565 | -28.27% |
| Training high vol + momentum, equal weight | 18.68% | 28.06% | 0.740 | 1.513 | -27.89% |
| Training high vol + momentum, score weights | 10.86% | 29.06% | 0.494 | 0.830 | -46.44% |
| Hybrid, momentum / ordinary 3m volatility | 16.57% | 38.38% | 0.548 | 1.410 | -31.61% |
| Hybrid, momentum / downside-weighted 3m risk | 12.53% | 27.35% | 0.559 | 1.043 | -31.67% |

- Adding the downside term versus ordinary 3m risk sizing changes CAGR by -4.03 percentage points, Sharpe by +0.011, and Sortino by -0.367.
- Versus the prior momentum-only allocation, CAGR changes by +1.68 percentage points. This comparison changes both inverse-risk sizing and the floor, not just downside sensitivity.
- The new rule averages 47.6 held names and 20.0 effective holdings. Largest single-stock weight: 64.01%; average one-way turnover: 38.53%.

## What the names mean

The earlier rolling risk estimate used twelve monthly observations. This new sizing estimate uses three. The frozen training volatility screen and twelve-month momentum remain unchanged. Volatility is a nonnegative magnitude; negative returns receive an extra penalty. The custom measure is not itself Sharpe or Sortino.

Sharpe is mean excess return divided by its standard deviation. Sortino is mean return above a target divided by root-mean-square shortfalls below that target. Positive observations remain in the denominator count with zero shortfall. Both ratios here use the monthly sqrt(12) convention. See the sources in the notebook for definitions.

Three observations give a noisy risk estimate. This already examined historical period cannot establish an optimal risk measure, window, floor, or penalty. No parameter search, caps, costs, or return clipping were added. Earlier project files and outputs are preserved.
