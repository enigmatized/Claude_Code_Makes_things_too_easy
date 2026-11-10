# What the current strategies actually do

This explains the thirteen strategies already implemented. It describes their
rules, not which one will perform best. The proposed pipeline and model changes
are in [02_pipeline_and_model_plan.md](02_pipeline_and_model_plan.md).

## Four separate decisions

1. **Universe:** which stocks may we consider?
2. **Basket:** which of those stocks do we choose this month?
3. **Allocation:** what fraction of our money goes into each chosen stock?
4. **ML adjustment:** do loss predictions change those fractions?

The first three define the original portfolio. The fourth creates a comparison
version. A prediction model and a portfolio strategy are separate choices.

For example, “momentum portfolio + random-forest loss adjustment” means the
momentum rule chooses the original basket and weights. A random forest estimates
each stock's loss probability; the common adjustment changes its relative weight.
**Random forest is proposed for the next stage; it is not implemented yet.**

## Words used in the app

| Term | Meaning in this project |
| --- | --- |
| Available stock | A stock with a row at the current month-end snapshot. |
| Eligible stock | An available stock with 12 consecutive already-realized monthly returns. |
| Momentum | Total compounded return over those 12 months, including the latest known month. It is not a forecast. |
| Volatility | How much monthly returns vary; the original baskets estimate it from 12 monthly observations. |
| Positive momentum | The compounded 12-month return is greater than zero. |
| Equal weight | Divide capital evenly across the selected stocks. |
| Score weight | Give more capital to a larger positive momentum score. |
| Frozen high-volatility list | Stocks selected once using historical volatility known at evaluation's start. Membership then stays fixed. |
| Three-month risk sizing | Use the last three monthly returns to change weights inside an existing basket. |

With $100 and four equal-weight stocks, each receives $25. With positive
momentum scores 0.10 and 0.30, score weighting assigns 25% and 75% of capital.

## The thirteen portfolios

The keys match the code; the explanations describe the actual rules.

| Code key | Stocks selected | How money is divided | What this comparison asks |
| --- | --- | --- | --- |
| `equal_weight_universe` | Every currently available stock. | Equal weights. | How did the available universe perform? |
| `equal_weight_eligible` | Every eligible stock. | Equal weights. | What changes just from requiring enough return history? |
| `high_volatility` | Highest 20% of eligible stocks by trailing 12-month volatility; recomputed monthly. | Equal weights. | What happens when we concentrate in stocks with larger recent fluctuations? |
| `low_volatility` | Lowest 20% by the same volatility measure; recomputed monthly. | Equal weights. | What happens when we concentrate in stocks with smaller recent fluctuations? |
| `momentum_equal_weight` | Strongest positive 12-month momentum, up to 20% of eligible stocks. | Equal weights. | What happens when we buy recent winners? |
| `momentum_weighted` | Exactly the same momentum basket. | Proportional to positive momentum. | What changes when stronger recent winners get more money? |
| `training_high_vol_equal_weight` | Eligible stocks inside the frozen high-volatility list. | Equal weights. | What does the initial volatility screen contribute by itself? |
| `training_high_vol_momentum_equal` | Strongest positive momentum, up to 20% of eligible stocks inside that frozen list. | Equal weights. | What does monthly momentum selection add inside the fixed list? |
| `training_high_vol_momentum_weighted` | Exactly the same hybrid momentum basket. | Proportional to positive momentum. | What changes when the hybrid also weights by momentum strength? |
| `hybrid_volatility_3m_sizing` | Exactly the same hybrid momentum basket. | Positive momentum divided by ordinary 3-month risk, then normalized. | What does inverse-risk sizing add? |
| `hybrid_downside_3m_sizing` | Exactly the same hybrid momentum basket. | Positive momentum divided by custom downside-weighted 3-month risk, then normalized. | What changes when negative returns add an extra risk penalty? |
| `training_sharpe_momentum_weighted` | Strongest positive momentum, up to 20% of eligible stocks inside a frozen top-training-Sharpe pool. | Proportional to positive momentum. | Does a historical Sharpe screen help the momentum strategy? |
| `training_sortino_momentum_weighted` | Strongest positive momentum, up to 20% of eligible stocks inside a frozen top-training-Sortino pool. | Proportional to positive momentum. | Does a historical Sortino screen help the momentum strategy? |

“Top 20%” rounds the stock count upward; ties use lower stock ID. Momentum
portfolios hold fewer names if fewer than the target count have positive momentum.
For illustration, 1,000 eligible stocks might produce a frozen list near 200,
then a hybrid momentum basket near 40. Actual counts vary with eligibility.

The frozen list is fitted using training returns known at the first evaluation
snapshot: Dec 2004 for the app's default 2005–2009 validation, or Dec 2009 for
the 2010–2018 test comparison. It is different from the monthly rolling basket
used by `high_volatility`. All monthly momentum signals continue to update.

## Three-month risk and the downside variant

For the latest three known monthly returns:

```text
ordinary risk² = sample variance of the three returns
extra downside term = average of min(return, 0)² across ALL THREE months
custom risk = sqrt(ordinary risk² + extra downside term)
```

A positive return adds zero to the extra downside term. A negative return adds
its squared shortfall from zero. Both risk measures are annualized consistently
in the code. The denominator has a 1% monthly risk floor, converted to the same
annualized units, to prevent division by zero or very small risk estimates.

These two strategies change allocation, not the selected stock names. The
custom measure is not a Sharpe or Sortino ratio. The two new ratio-pool strategies
below use Sharpe/Sortino directly for their initial screen.

## Training Sharpe or Sortino pool, then momentum

These follow the same two-stage selection and momentum weighting as the frozen
high-volatility hybrid, changing the statistic used to choose the initial pool.

1. At the evaluation's first decision date, use only training returns already
   earned by then. Each stock needs at least 12 observed training returns.
2. Compute historical Sharpe or Sortino with a zero monthly reference return:
   `Sharpe = mean(return) / sample_std(return) × sqrt(12)`;
   `Sortino = mean(return) / sqrt(mean(min(return, 0)²)) × sqrt(12)`.
   Sortino counts all observed months, including positive months with zero
   downside shortfall. These are arithmetic-mean ratios, not CAGR/risk.
3. Freeze the highest 20% of finite ratios, rounding upward; break ties by lower
   stock ID. An undefined ratio is excluded and recorded, not treated as infinity.
   All-positive returns can therefore make a stock's Sortino undefined. Negative
   finite ratios are still ranked; there is no extra positive-ratio screen.
4. Each month inside this fixed pool, require 12 consecutive known returns and
   select the strongest positive momentum, up to 20% of eligible pool stocks.
   Allocate in proportion to those positive momentum scores and normalize to 100%.

The ratio is not recomputed to change membership during evaluation. Momentum
and allocations do update monthly. Both strategies use the same frozen-pool
fallbacks as the existing hybrid and can receive the same optional ML adjustment.

For the app's default validation, the pool is known at Dec 2004 and evaluated
over Jan 2005–Dec 2009. In test mode, it uses full training returns known at
Dec 2009 and evaluates Jan 2010–Dec 2018. Applying the final 2009 pool to earlier
training years would use future information and is not done.

The app exposes the selected pool, training scores, observation counts, and
cutoff in the Allocations tab. Full rankings, including exclusions, are exported
as `frozen_sharpe_screen.csv` and `frozen_sortino_screen.csv` in a run report.

Definitions: [Sharpe's original discussion](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm)
and [the Sortino calculation reference](https://www.cmegroup.com/education/files/rr-sortino-a-sharper-ratio.pdf).

## What ML currently changes

The app offers logistic regression, Gaussian naive Bayes, random forest and
histogram gradient boosting. Each estimates the probability of a negative return
next month for each stock, using the same declared original-factor recipe. The
default recipe has twenty factors; an explicit manual subset is also supported.
For original weights `w`, probabilities `p`, and adjustment strength `alpha`:

```text
adjusted scores = w × (1 − alpha × p)
adjusted weights = adjusted scores / sum(adjusted scores)
```

At strength 0.5, an 80% loss probability multiplies a stock's original weight
by 0.6 before normalization. A 20% probability gives a multiplier of 0.9.
The latter stock gains relative allocation, although its final weight also
depends on every other selected stock. This is not an expected-return estimate.

The adjustment keeps the same basket and invests all capital. If all stocks
receive the same probability, weights do not change. It cannot move the whole
portfolio into cash. A one-stock basket also cannot change through normalization.

## Fallbacks and comparison assumptions

- With no positive momentum, momentum strategies equal-weight eligible stocks.
- With no eligible history, history-dependent strategies equal-weight available
  stocks. Hybrids keep both fallbacks inside their frozen list.
- If no frozen-list stock is available, the hybrid fails explicitly; there is
  no implicit cash position or expansion outside its permitted universe.
- The two risk-sizing strategies preserve their parent hybrid's fallback.
- Portfolios rebalance monthly, are long-only, and sum to 100%. No position cap
  or transaction-cost deduction is included. Turnover measures trading, not costs.

A month-end weight is held over the following month. Trailing inputs use only
already-realized returns. Charts of allocations show target weights at rebalance,
not the weights after prices move during the holding month.

The app's comparison metrics summarize portfolio outcomes: CAGR is compounded
growth, Sharpe compares mean return with total return variability, Sortino uses
downside deviation, and maximum drawdown measures the deepest fall from an
earlier wealth peak. Sharpe's risk-free return and Sortino's target are zero here.
