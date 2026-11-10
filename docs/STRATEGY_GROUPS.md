# Reading grouped strategy results

Strategies and Basic ML organize results by portfolio construction.
The same strategy can have an original and several ML versions; models stay
inside that strategy rather than becoming separate strategy families.

| Family | Included rules |
| --- | --- |
| Benchmarks | Equal-weight universe and equal-weight stocks with eligible history. |
| Volatility baskets | High- and low-volatility baskets. |
| Momentum baskets | Positive momentum with equal or momentum-based weights. |
| High-volatility hybrids | Frozen high-volatility pool, its momentum variants, and ordinary/downside three-month risk sizing. |
| Sharpe / Sortino pools | Frozen risk-adjusted pools followed by momentum weighting. |
| Annual winners + cash | Historical positive-year consistency, configurable positive momentum and cash. |

Each of the fourteen current strategies belongs to exactly one family. Unknown
extensions have an Other strategies group instead of disappearing.

## Comparing results

- **Strategy family** filters the displayed results, charts and ML changes.
  It does not change the saved run or trigger computation. All strategy families
  allows comparisons across groups.
- **Grouped strategies** shows collapsible family sections. Inside each,
  a strategy heading identifies the non-ML scores in Strategies, or Original
  and each selected model's actual performance in Basic ML. Stock-count and largest-weight columns are available in
  Full tables; the compact view focuses on return, risk, turnover and cash.
- **Full tables** retains a sortable comparison across all visible rows.
- **Select all strategies** applies to the current display family. Choose All
  strategy families first to chart every strategy in the loaded run.

## The ML-change overview

The old long table repeated each strategy once for every model. The default view
now has a bordered card for each family, one row per strategy and one column per
model. **Compare the ML change in** selects CAGR, Sharpe, Sortino, maximum
drawdown or turnover. All values are recorded changes against that strategy's
own non-ML original over the same months; nothing is averaged across models.

Return, drawdown and turnover changes are displayed in **percentage points
(pp)**. For example, a decimal change of 0.001 is +0.10 pp. Sharpe/Sortino changes
remain ratio units. Positive drawdown change means the worst drawdown is
shallower. Missing strategy/model pairs stay missing and display an em dash.

Cells use a shared intensity scale for the visible metric across families.
Green indicates the preferred direction for that metric and red the opposite;
for turnover, lower is shaded green. Turnover shading describes a trading-cost
consideration, not an improvement in portfolio returns or statistical confidence.

**Inspect one strategy · every ML change** retains the detailed model rows and
all change metrics, with original reference scores. **Download ML changes (CSV)**
exports all recorded model changes for the display filter, regardless of the
metric currently selected in the overview.

## Choosing strategies for a new experiment

The sidebar's **Choose strategy families or individual rules** keeps the long
individual-strategy list collapsed initially. Choose a family to narrow the
available rules and use **Select all in this family** when needed. The selection
applies to the next explicitly loaded/computed configuration. Changing it does
not replace the currently displayed result.

The historical evidence view inside Final test also has its own family filter. It remains
independent of the main comparison and retains its test-period labels.

## Implementation and evidence

`strategy_groups.py` contains presentation-only taxonomy and lossless table
reshaping, independent of the calculation registry/cache identity.
`strategy_results_ui.py` renders grouped metrics, matrices, details and downloads.
Tests check unique coverage, unknown extensions, missing/duplicate pairs,
format units, exact saved deltas and no-fit group/filter interactions.

This update adds no strategy, training, data transformation or backtest.
Saved outputs, scientific calculation code and submission release remain intact.
See the [verification record](verification/strategy_groups_20261108.json) and
[browser preview](verification/grouped_strategy_changes.png).
