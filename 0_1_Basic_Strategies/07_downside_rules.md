# Three-month risk, with an extra downside penalty

This adds one strategy to the preceding high-volatility/momentum hybrid, plus
an ordinary-volatility sizing control. Earlier strategies and outputs remain
intact. This implementation adjusts the hybrid's allocation within its existing
selected stock basket.

## What three-month volatility means here

Use the latest **three already-realized monthly returns** at each rebalance.
The existing rolling strategies estimated volatility from twelve monthly returns,
not one observation. The earlier frozen screen used full training-history
volatility. Three months is the new sizing lookback; monthly return frequency,
monthly rebalancing, and twelve-month momentum are unchanged.

This is not volatility of three-month cumulative returns across many quarters,
and it is not daily volatility estimated over a quarter: daily prices are absent.
With only three observations, these risk estimates can change sharply.

## The custom measure

Let `r` be those three monthly simple returns. Set:

```text
s² = sample variance(r), using denominator 3 − 1
D² = mean(min(r, 0)²), using ALL THREE observations
ordinary risk = sqrt(s²) × sqrt(12)
downside-weighted risk = sqrt(s² + D²) × sqrt(12)
```

The downside penalty coefficient is fixed at one, not optimized. A negative
return contributes an extra squared shortfall from zero, while a positive
return contributes no additional downside term. The result is always a
nonnegative risk magnitude. It is a custom measure, not a standard deviation,
Sharpe ratio, or Sortino ratio. With a zero downside penalty it equals ordinary
volatility; constant negative returns remain risky even if their standard
deviation is zero.

## Allocation rule and control

Keep the same frozen 241-stock training high-volatility list and the same
monthly top-positive-momentum basket as the previous hybrid. Membership remains
training-only; twelve-month momentum uses returns known by the snapshot.

The new strategy sets weight proportional to:

```text
positive twelve-month momentum / max(downside-weighted 3m risk, risk floor)
```

The control replaces the denominator with ordinary three-month volatility.
Both normalize scores to sum to one. Both use a **1% monthly risk floor**,
converted to annualized units by multiplying by sqrt(12), to avoid dividing by
zero or near-zero three-observation estimates. No weight cap is added. Thus the
new-versus-control difference isolates the added downside term, using identical
selected names, floors, and momentum signals. Relative to the previous hybrid,
both the inverse-risk sizing and its floor are new choices.

If the parent hybrid uses an equal-weight fallback, preserve it. Never buy a
stock outside the frozen training list. Missing historical risk on a selected
stock is an error, not an imputed value. Missing returns are never filled with
zero. Zero *shortfalls* for positive returns are part of the downside definition.

## Sharpe, downside deviation, and Sortino

Sharpe compares mean monthly excess return with the standard deviation of those
excess returns. The course implementation assumes risk-free return zero and
multiplies the monthly ratio by sqrt(12). It treats upward and downward
dispersion as volatility. [Sharpe's discussion](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm)
explains its differential-return basis.

Sortino compares mean return above a stated target with root-mean-square
shortfalls below that target. With target zero, positive-return observations
contribute zero shortfall but remain in the observation count. This is not the
standard deviation of only the negative returns. The notebook reports Sortino
over the full 108-month evaluation, not an unstable three-observation Sortino
used as an allocation signal. A zero denominator gives an undefined ratio.
See the [Sortino calculation reference](https://www.cmegroup.com/education/files/rr-sortino-a-sharper-ratio.pdf).

## Evaluation

Use January 2010–December 2018, matching the preceding frozen-screen hybrid.
There is no retrospective training backtest using the final training screen.
Report gross CAGR, volatility, Sharpe, Sortino (target zero), drawdown, turnover,
and concentration. Three months, the downside coefficient, and the risk floor
are fixed before this run, with no grid search or return clipping. The test
period has already been examined; this is another requested historical
experiment rather than fresh validation.

New outputs go under `outputs/downside/`. Run the notebook from top to bottom,
and run `python3 -m unittest -v test_downside.py` to check its calculations.
