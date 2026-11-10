# Annual consistency + positive momentum + cash

This is the user's requested non-ML rule: buy stocks that historically had more
positive years, buy only while momentum is positive, and allow cash. Its ML
versions apply the existing next-month loss classifiers to the same stock weights.

The original run below uses twelve-month momentum. That window is now configurable
independently of the annual screen; see the [2/3/6/12-month study](../0_5_Momentum_Lookbacks/README.md).
Earlier results and the twelve-month default are preserved.

## Exact rules

1. At the first evaluation snapshot, use only training returns already earned.
   Compound January--December returns separately for each stock and calendar
   year. Count a year only if all twelve monthly returns are present.
2. Require at least **three complete years**, with **strictly more than half
   positive**. Flat years count as nonpositive. Freeze this pool for the whole
   evaluation. This is a declared first specification; thresholds were not tuned.
3. Each month, count the currently available stocks in that pool (`N`). A stock
   can be bought only with twelve consecutive known returns and strictly positive
   compounded trailing twelve-month momentum. Count those stocks (`P`).
4. Invest `P/N` of capital and keep the rest in cash. Distribute the invested
   capital in proportion to positive momentum. Stocks with zero, negative or
   unavailable momentum receive zero. With no available pool stocks or no
   positive candidates, hold **100% cash**.

For four currently available pool stocks with momentum **10%, 30%, -20%, 0%**,
the stock weights are **12.5%, 37.5%, 0%, 0%**, and cash is **50%**.
The two inactive stocks' capital slots remain in cash; stronger positive momentum
gets more of the active budget. Missing history also leaves a slot in cash.

Cash earns an explicitly assumed **0% monthly return**. There is no supplied
cash-yield series. Cash is included in price-drift-aware turnover; it is excluded
from stock counts and stock concentration. No fees or position caps are added.

## ML comparison

For this rule only, the optional adjustment is:

```text
adjusted stock weight = original stock weight × (1 − strength × P(next-month loss))
cash weight = 1 − sum(adjusted stock weights)
```

With default strength 0.5 and probability 0.8 for both held stocks in the example,
their weights become 7.5% and 22.5%, and cash becomes 70%. This adjustment cannot
buy an excluded stock or spend the original cash. Existing fully invested
strategies retain their original normalization and fallback behavior.

## Review and reproduce

Start `python3 run_app.py`, then load the saved run identified in
[outputs/run_reference.json](outputs/run_reference.json). It compares all fourteen
rules, including the thirteen earlier rules, with four fixed classifiers on the
same training-validation months. Choose **Annual consistency + positive momentum
+ cash** in the optional allocation explorer within Strategies (original only)
or Basic ML (paired versions). The app shows performance,
cash through time, stock/cash composition and the historical annual screen.
Saved-result loading does not fit a model.

To compute or reuse a smaller matched comparison from the project root:

```bash
python3 -m research --strategies equal_weight_universe momentum_weighted annual_consistency_momentum_cash --all-models
```

The complete saved comparison can be reproduced/reused with:

```bash
python3 -m research --all --all-models
```

The completed first run uses all twenty original factors for the optional fixed
classifiers. It does not select features, tune parameters, or read the test CSV.
It is training-period research, not the final frozen-test submission experiment.

Outputs contain the run reference, complete comparison metrics, the annual
screen (including exclusions), monthly returns/cash for this strategy, and
verification evidence. Full allocations, predictions and diagnostics live in the
immutable SQLite/Parquet artifact store. The existing report release stays pinned.

## First results

The first saved training-validation result (earned Jan 2005–Dec 2009) is:

| Portfolio | CAGR | Sharpe | Maximum drawdown | Average cash |
| --- | ---: | ---: | ---: | ---: |
| Equal-weight universe | 7.62% | 0.433 | -49.51% | 0% |
| Earlier fully invested momentum weighting | -1.74% | 0.031 | -53.45% | 0% |
| Annual consistency + momentum + cash | 2.34% | 0.270 | -19.60% | 45.93% |
| Same cash strategy + logistic adjustment | 1.91% | 0.270 | -15.56% | 57.66% |

The new rule reduced drawdown in this window while trailing equal weight on
return and Sharpe. ML reduced exposure further; its smaller drawdown alone does
not establish useful predictive skill. A future exposure-matched control would
help distinguish forecasting from the effect of simply holding more cash.
The saved CSV includes all four ML versions and all fourteen original rules.

See the [performance and cash figure](outputs/training_comparison.png) and
[verification evidence](outputs/verification.json). All 155 distinct tests passed;
earlier probabilities, books, weights, source data and the report pin were checked.

## Implementation

- `research/cash_strategy.py`: known-label annual screen and stock budget.
- `research/accounting.py`: residual cash adapter around preserved Part 0 accounting.
- `research/strategies.py`: normal registry entry, with explicit `allows_cash`.
- `research/adjustments.py`: opt-in cash-preserving ML adjustment.
- `tests/test_cash_strategy.py`: annual compounding, incomplete years, timing,
  budget arithmetic, cash turnover, cash-only periods, persistence and app checks.
