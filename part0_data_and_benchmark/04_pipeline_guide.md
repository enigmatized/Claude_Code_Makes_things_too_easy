# A small pipeline for later experiments

The pipeline is one Python module so that notebooks can explain an experiment
without copying accounting code. Keep it small until an actual experiment needs
more. The source CSVs are read-only inputs; generated files go in `outputs/`.

```text
source CSV → validated stock-month panel → available-at-snapshot inputs
           → strategy/model → keyed portfolio weights → forward returns
           → monthly wealth, risk, benchmark comparison
```

## The shared functions

| Function | Contract |
| --- | --- |
| `load_panel('train')` | Validate schema, dates, keys, missingness, ranges; sort consistently. |
| `panel_audit(panel)` | Summarize rows, stocks, and snapshot/earned calendars. |
| `feature_frame(panel)` | Return only the twenty original features. |
| `expanding_month_folds(panel)` | Chronological date blocks; all rows from a month stay together. |
| `equal_weights(panel)` | One equal capital weight per available stock per snapshot. |
| `portfolio_returns(panel, weights)` | Align by date and ID; enforce long-only, fully invested weights; earn forward labels. |
| `performance(monthly_returns)` | CAGR, log growth, volatility, Sharpe, drawdown, and simple summaries. |

There is no clipping, stock screening, learned scaler, feature selector, or model
inside the shared accounting. Pass explicit zero weights for unheld stocks. A
strategy experiment changes weights while retaining the return and metric code.
The engine currently supports monthly fully invested long-only books. A cash
allocation or leverage strategy requires an explicit extension, not weights that
quietly fail to sum to one.

For example, from this directory:

```python
from pipeline import load_panel, equal_weights, portfolio_returns, performance

train = load_panel()  # training by default
weights = equal_weights(train)
book = portfolio_returns(train, weights)
print(performance(book['simple_return']))
```

## Later model workflow

1. Define a modest experiment using training data. Keep identifiers and forward
   returns out of the feature matrix. Handwritten composites belong in a short,
   explicit transformation function.
2. Split by ordered snapshot months. Fit scalers, PCA, selectors, and models on
   each fold's training rows, then transform its validation rows. A transformation
   that learns from later rows leaks information even if it never reads labels.
3. Convert predictions into weights using a declared rule. Define tie handling,
   minimum history, and fallback behavior before inspecting returns.
4. Pass those keyed weights to the shared accounting. Compare identical months
   against equal weight, measuring both reward and risk.
5. Freeze specifications before the official test evaluation. Record the actual
   chronology of earlier test inspection; rerunning a test is not fresh evidence.

The next-month label of the last historical snapshot is assumed available at the
next month-end rebalance. This is the assignment's timing convention. Real
execution would need intraday cutoffs, publication lags, and possibly a gap.
If future targets span multiple months, splits also need to account for label
overlap. Do not introduce random row splits into this stock-month panel.

## Returns and implementation assumptions

Aggregate SIMPLE returns across stocks, then compound across months. Log returns
are derived from the resulting portfolio return, and add through time.
`performance` uses twelve periods per year and sample standard deviations.
Its default Sharpe uses a zero monthly risk-free rate to match course chapter
15b. Supply a correctly aligned monthly risk-free Series for an excess-return
version; do not subtract an annual yield from every monthly observation.

Turnover compares new target weights with old weights after their realized
returns. A departing name is sold at that transition after its last supplied
holding-period return; a new name is bought. This uses the supplied returns and
does not invent a missing liquidation price. First-period turnover is NaN,
because funding from cash needs a separately stated convention.

The required benchmark is gross of costs. No cost estimate is silently deducted.
If later experiments use costs, declare buy/sell versus one-way conventions,
initial entry cost, timing, and a model for costs. The reported turnover is a
research estimate under the available-universe assumption, not proof of liquidity.

## Reproducibility and verification

Notebook 03 checks the shared result against `groupby(...).mean()`, a separate
dollar-allocation loop, and the equivalent log-compounding identity. Toy cases
expose cross-sectional log averaging, buy-and-hold differences, weight drift,
the first losing month's drawdown, and alignment errors. These are accounting
checks rather than tests of whether an investment idea is profitable.

Run `python3 -m unittest -v test_pipeline.py`. Run the two notebooks from top to
bottom to regenerate their tables, figures, and notebook-specific outputs.
`outputs/source_manifest.json` records hashes and library versions from the run.
The preservation audit compares assignment/Part 1 hashes before and after this
addition. Neither notebook imports or runs Part 1.
