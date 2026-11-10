# Frozen training Sharpe/Sortino pools with monthly momentum

Two new strategies are registered in the runner and app:

- **Training Sharpe pool + momentum** (`training_sharpe_momentum_weighted`).
- **Training Sortino pool + momentum** (`training_sortino_momentum_weighted`).

They change the initial stock screen of the existing frozen high-volatility
momentum strategy. Its two-stage basket fractions, momentum calculation,
weighting, and fallback rules are retained, making the screen choice explicit.

## Fixed rules

1. Use only training returns realized by the first evaluation snapshot. Require
   at least 12 observed training months per stock, without a future-survival rule.
2. Rank by annualized historical Sharpe or Sortino, using arithmetic mean monthly
   return and a zero reference return. Sharpe uses sample standard deviation;
   Sortino uses root-mean-square negative shortfalls across all observed months.
3. Freeze the top 20% of eligible stocks, rounding upward. Ties use lower stock
   ID. Undefined/nonfinite ratios are excluded and their reason is exported;
   zero downside is not converted into an infinite Sortino score. Negative
   finite ratios remain rankable. The screen is not updated during evaluation.
4. Each month, restrict to available pool members with 12 consecutive known
   monthly returns. Calculate compounded 12-month momentum. Choose the strongest
   positive momenta, up to 20% of these eligible pool members, rounding upward.
5. Set weights proportional to the selected positive momentum scores. Normalize
   to 100%. With no positive momentum, equal-weight eligible pool stocks. With
   no eligible history, equal-weight available pool stocks. No capital goes
   outside the pool; no available member is an explicit error.

The same optional loss-probability ML adjustment can be applied afterward.
There is no new ML model or feature search in this addition. Sharpe and Sortino
here choose stock pools, while the portfolio's own evaluation ratios describe
the subsequent strategy returns; these are different calculations.

## Calendars and pool sizes

| Run | Screen known at | Earned-return window | Sharpe pool | Sortino pool |
| --- | --- | --- | --- | --- |
| Training validation | Dec 2004 | Jan 2005–Dec 2009 | 240 stocks | 240 stocks |
| Previously viewed test | Dec 2009 | Jan 2010–Dec 2018 | 241 stocks | 241 stocks |

Equal pool sizes do not imply identical members. Membership is fitted separately
for each ratio and separately for each evaluation origin. A final 2009 pool is
never applied to earlier validation returns.

## Initial results

Gross results retain the original source returns. No parameters were searched
to obtain these results. The test period has already been inspected elsewhere
in the project, so these are historical comparisons, not fresh holdout evidence.

Original portfolios, without the ML adjustment:

| Strategy | 2005–2009 CAGR | Sharpe | Max drawdown |
| --- | ---: | ---: | ---: |
| Equal-weight universe | 7.62% | 0.433 | −49.51% |
| Training Sharpe pool + momentum | −4.62% | −0.103 | −53.26% |
| Training Sortino pool + momentum | −4.96% | −0.107 | −56.16% |

| Strategy | 2010–2018 CAGR | Sharpe | Max drawdown |
| --- | ---: | ---: | ---: |
| Equal-weight universe | 13.10% | 0.868 | −20.64% |
| Positive momentum, score weights | 8.71% | 0.523 | −28.81% |
| Training high vol + momentum, score weights | 10.86% | 0.494 | −46.44% |
| Training Sharpe pool + momentum | 7.30% | 0.538 | −21.10% |
| Training Sortino pool + momentum | 7.72% | 0.493 | −24.45% |

The requested rules work, but this initial experiment does not demonstrate an
advantage over equal weight. The ratio screens reduced test drawdowns versus
the high-volatility momentum hybrid and also produced lower growth. Their
training-validation outcomes were weak. No threshold changes were made in
response to either window.

## Inspect and reproduce

Refresh the app, select either strategy, and run the comparison. In Allocations,
expand the pool explanation to see selected names and training ratios. The
single-rebalance table also shows each held stock's training rank, Sharpe,
Sortino, and screen cutoff, beside original/ML weights.

Saved runs:

- [Training validation, including ML comparisons](outputs/ratio_pools_training_validation/strategy_comparison.csv).
- [Test comparison against existing controls](outputs/ratio_pools_test_comparison/strategy_comparison.csv).

Both run folders include manifests, calendar-aligned monthly returns, allocation
diagnostics, and separate `frozen_sharpe_screen.csv` / `frozen_sortino_screen.csv`
files with full fitted rankings and exclusions. The training-validation run
also includes predictions and model fit audits.

```bash
python3 -m research --strategies training_sharpe_momentum_weighted training_sortino_momentum_weighted --ml
python3 -m research --strategies equal_weight_universe training_sharpe_momentum_weighted training_sortino_momentum_weighted --evaluation test
```

Implementation: `research/screens.py` fits ratio pools; `research/strategies.py`
registers their momentum allocations; the runner shares predictions/accounting
and exports each screen separately. Other strategy calculations retain their
existing implementation.

Verification includes independent ratio arithmetic, zero-denominator exclusions,
minimum history, deterministic ties, future-label perturbations, frozen pool
membership, momentum weights, pool-specific exports, and real app interactions.
All 103 checks across the project foundation, original strategies, runner,
allocation charts, ratio screens, and app passed for this addition.

Definitions: [Sharpe](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm),
[Sortino calculation](https://www.cmegroup.com/education/files/rr-sortino-a-sharper-ratio.pdf).
