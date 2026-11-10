# What the dataset is

One row is **one stock at one month-end**, with twenty characteristics and the
stock's subsequent one-month simple return. Repeated observations of the same
stock form a panel; these are not independent rows from unrelated companies.

For stock 1 at the first snapshot, `date = 1999-12-31` and `R1M_Usd = -0.042`.
Read this as: use the December snapshot to decide what to hold, then earn -4.2%
in January. The target is already a return. Do not apply `pct_change()` to it,
and do not shift it a second time.

| Field | Role |
| --- | --- |
| `date` | Snapshot date, when the assignment assumes characteristics are available. |
| `stock_id` | Anonymous security identifier. There is no local ticker lookup. |
| Twenty factor columns | Model inputs; dictionary printed in notebook 02. |
| `R1M_Usd` | Forward one-month simple return in decimal units; excluded from inputs. |

| | Training | Testing |
| --- | --- | --- |
| Rows | 140,739 | 127,740 |
| Snapshots | 120 | 108 |
| Distinct securities | 1,204 | 1,195 |
| Snapshot dates | Dec 1999–Nov 2009 | Dec 2009–Nov 2018 |
| Months when returns are earned | Jan 2000–Dec 2009 | Jan 2010–Dec 2018 |

The notebooks recompute these facts. There are no duplicate stock-months or
missing cells in these source files. The number of stocks varies by month.
An absent stock-month is not a zero return, and a stock is not required to exist
through the end of the dataset to enter a month's equal-weight portfolio.

## What the feature values mean

All twenty columns have values on a 100-point grid from 0.01 to 1.00. Their
distributions are consistent with cross-sectional percentile encoding. For
example, `Fcf_Yld = 0.80` is roughly a relative rank, not an 80% cash-flow yield.
The exact export/rounding recipe for these CSVs is not provided. MLFactor discusses
cross-sectional uniformization in its [preprocessing chapter](https://www.mlfactor.com/Data.html).

A high value means a relatively high characteristic, not automatically a good
investment. High debt and high dividend yield describe different economic ideas.
Raw accounting magnitudes cannot be recovered from these ranks. Ratios or
differences of ranks are engineered signals, not restored accounting quantities.

The dictionary follows the [MLFactor variable descriptions](https://www.mlfactor.com/data-description.html).
In particular, `Ocf_Ce` uses capital employed; `Ocf_Ta` and `Ocf_Toa` have distinct
asset denominators. Some entries in the preserved Part 1 glossary differ.

## What we know about origin, and what we do not

The assignment calls these MLFactor data. The [original book dataset](https://www.mlfactor.com/notdata.html)
contains 1,207 US-listed securities and a larger collection of characteristics.
These class files are a reduced extract. The files here do not provide an export
script, ticker mapping, exact original-universe membership rule, accounting
release lags, or evidence sufficient to certify freedom from survivorship bias.
They also do not separately document dividend, split, or delisting adjustments.
Use the supplied labels as instructed, while distinguishing this classroom
backtest from a verified executable investment record.

There are no raw prices, market caps, sector classifications, spreads, or daily
observations. `Net_Debt` and `Tev_Less_Mktcap` are ranks, not dollar values. A
capitalization-weighted index therefore cannot be recovered from these files.
The assignment's equally weighted market means the available stock universe;
it is not a downloaded S&P 500 series.

## What the training charts can teach us

Notebook 02 shows universe size, monthly cross-sectional return distributions,
an equal-weight wealth path and drawdown, six illustrative stock paths, extreme
observations, and factor dependence. A stock path compounds the provided returns
into indexed wealth. It is not a reconstructed share-price history.

The six paths are selected by complete training coverage and then lowest ID,
without ranking their performance. That makes them reproducible illustrations,
but not a random or representative sample. Complete-coverage selection is never
applied to the benchmark. Extreme observations are shown and retained; deciding
they are bad data would require source evidence we do not have.
