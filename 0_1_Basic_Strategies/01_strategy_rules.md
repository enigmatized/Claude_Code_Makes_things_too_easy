# Rules specified before this comparison was run

## Calendar and available information

The return label stored on snapshot t is earned in month t+1. First map labels
to their earned months, then construct a trailing history available as of t.
For example, the December 2009 rebalance can use January–December 2009 returns
from the December 2008–November 2009 snapshots. It cannot use the January 2010
return stored on its own row.

At the December 2010 rebalance, already realized January–December 2010 returns
can update the rule. This is normal monthly rebalancing; no rule or parameter
is chosen from future test results. The end-of-month availability convention
matches Part 0. Intraday execution and data-publication delays are not modeled.

Each stock needs twelve consecutive known monthly returns ending at the
snapshot. Rolling calculations are made on a calendar grid, so a missing
stock-month breaks eligibility until there are twelve consecutive observations
again. Current membership is determined by that snapshot, without requiring
future survival. Never fill a missing observation with a zero return.

## Signals and weights

- **Volatility:** sample standard deviation of the last twelve realized monthly
  simple returns, multiplied by sqrt(12). This ranks individual stocks.
- **Momentum:** product of their twelve gross returns, minus one. This includes
  the most recently realized month. It measures accumulated return, not the
  slope of a fitted price trend or a prediction of the next month.
- **Basket size:** ceil(0.20 × eligible stocks), with a minimum of one. A tie
  goes to the lower stock ID, making the count exact and reproducible.
- **High volatility:** largest volatility signals, equal capital per selected stock.
- **Low volatility:** smallest volatility signals, equal capital per selected stock.
- **Momentum selection:** highest momentum among stocks with momentum strictly
  above zero; select up to the basket size. If there are fewer positive signals,
  hold all the positive names. Equal-weight and score-weight versions select
  exactly the same names.
- **Momentum score weights:** each selected positive momentum divided by the sum
  of selected momenta. There is no single-name cap. Stronger winners receive
  more capital, and the concentration panel shows how far this can go.

This deliberately simple twelve-month momentum differs from the prior-months
2–12 convention used by the [French momentum factor](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor.html).
We do not claim to replicate that factor: these are fully invested long-only
books in the class universe, with no market-cap data or short positions.

## Fallbacks and comparison windows

If no momentum signal is positive, both momentum portfolios hold eligible
stocks equally. This keeps the comparison fully invested and avoids adding a
cash model to Part 0. If no stock is eligible, any history-dependent strategy
holds the current universe equally and records a fallback. The equal-weight
universe control never needs a history fallback.

Drop the initial twelve-month warm-up from ALL strategy comparisons. Report
training and testing separately on identical earned-return dates. At the split
boundary, earlier training history seeds January 2010; histories are not reset.
An equal-weight eligible-stock control identifies effects caused by the history
requirement rather than ranking or allocation.

## What to compare

Use Part 0's same decimal simple-return engine, raw source labels, monthly
compounding, and initial-dollar drawdown. Show CAGR, sample volatility,
zero-rate Sharpe, maximum drawdown, turnover, holdings, and effective number of
holdings (1 / sum of squared weights). No parameter grid, hidden cost haircut,
or return clipping is applied. Turnover excludes initial funding and includes
drift in the previous holdings.

Compare yearly paths and wealth relative to the universe. A mean monthly active
return times twelve is labeled as such, not as the difference in CAGR. Favorable
historical risk or return is not by itself proof of a superior rule. Check
whether a conclusion depends on one year or extreme observation.

## What feature selection can do later

These rules provide reference portfolios against which feature-based models can
be judged. Their lagged volatility and momentum can also be candidate engineered
predictors, subject to the assignment's feature-subset interpretation. Keep
them distinct from the twenty original columns. A later selection experiment
should use training-only chronological validation and the same portfolio
accounting. Do not choose a feature list by optimizing its 2010–2018 comparison.
