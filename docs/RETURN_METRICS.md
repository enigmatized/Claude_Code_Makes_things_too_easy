# Portfolio returns and growth

The pipeline already saves monthly simple returns, monthly log returns, wealth,
CAGR and annualized log growth. The app now exposes both growth conventions.
This is a presentation update; portfolio accounting and saved results are unchanged.

## In the app

In **Strategies** and **Basic ML**, the results tables show **CAGR** beside
**Annual log growth**. Under **Performance through time**, choose **Return measure**:

- **Growth of $1**: cumulative wealth, beginning with the initial dollar.
- **Cumulative log return**: sum of earned monthly portfolio log returns,
  beginning at zero at the funding snapshot.
- **Monthly log return**: each earned month's portfolio log return. There is no
  artificial zero return at the initial snapshot.

Each stage has independent display controls. Original and adjusted portfolios
use the same selected measure in Basic ML; drawdown and relative ML wealth
advantage retain their definitions. The **Logarithmic wealth scale** checkbox
changes only the wealth axis and is disabled in log-return views, whose values
can be negative. Choosing a view reads saved results without fitting models.

Final test now also offers these three chart measures and shows CAGR beside
annual log growth; it reads the saved portfolio simple returns. The entire path
explicitly joins two model protocols at the training/test boundary.

Momentum, feature-selection and archived test comparison tables also show
annual log growth. Their existing charts continue to show wealth. A ready saved
final or exploratory run uses the same comparison renderer and return selector.

## Calculation and interpretation

For portfolio simple return `r_t` over `n` monthly observations:

```text
monthly log return = log(1 + r_t)
cumulative log return = sum(log(1 + r_t)) = log(wealth)
annual log growth = 12 × mean(log(1 + r_t))
CAGR = exp(annual log growth) − 1 = wealth_final ** (12 / n) − 1
```

Log returns add across time; simple returns compound. Annual log growth is a
continuously compounded rate, formatted as a percentage. For example, 10% annual
log growth corresponds to about 10.52% CAGR. The two measures describe the same
realized growth and therefore rank identical-window portfolios the same way.

First calculate the portfolio's simple return from weighted stock **simple**
returns and cash, then take its logarithm. Averaging stock log returns would
describe a different calculation and is incorrect for our portfolio accounting.
Cash earns the existing assumed zero return. Sharpe and Sortino retain their
existing definitions using monthly simple returns, with risk-free/target zero.

## Saved artifacts and submission handoff

`part0_data_and_benchmark/pipeline.py` remains the numerical source:
`portfolio_returns()` records `log_return`; `performance()` records `cagr` and
`annualized_log_growth`. Research cash accounting reuses this source.
Saved portfolio books/monthly tables contain the log series; saved metric tables
and CSV exports contain both growth rates. SQLite indexes selected summary
metrics; complete metrics remain in Parquet. No schema migration is needed.

`app_charts.return_curve()` only transforms these saved columns for display and
preserves earned-month dates. Tests cover funding-point timing, a gain followed
by a reversing loss, cash/flat returns and the CAGR/log-growth identity.

The current Quarto draft already reports CAGR. When extending its narrative,
read `annualized_log_growth` and `log_return` from the same deliberately pinned
summary rather than recalculate portfolio accounting or refit models. This
change does not repin or overwrite the existing submission outputs.
