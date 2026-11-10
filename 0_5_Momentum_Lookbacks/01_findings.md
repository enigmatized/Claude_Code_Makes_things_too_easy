# First momentum-window comparison

Study: `momentum_20261108T211612_b4723b7c`. Read its
[saved protocol and run references](outputs/momentum_20261108T211612_b4723b7c/study_reference.json)
and [complete results](outputs/momentum_20261108T211612_b4723b7c/portfolio_comparison.csv).

These are gross **training-validation** results for January 2005–December 2009,
with the same 875-stock frozen annual-consistency pool. Cash earns an assumed
zero. The lookback changes momentum and its history requirement; all other rules,
dates, factors and four classifiers are held fixed. No test data were read.

## Non-ML results

| Momentum window | CAGR | Sharpe | Maximum drawdown | Average monthly one-way turnover | Average cash |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 months | 2.63% | 0.290 | -25.95% | 33.13% | 44.89% |
| 3 months | 4.42% | 0.437 | -22.03% | 26.57% | 44.17% |
| 6 months | 3.02% | 0.329 | -22.50% | 18.85% | 43.21% |
| 12 months | 2.34% | 0.270 | -19.60% | 12.38% | 45.93% |

Three months has the highest observed return and Sharpe among these four cash
rules. Twelve months has the shallowest drawdown and lowest turnover. Two months
reacts to a shorter history but is not the best performer here; its turnover is
about 2.68 times the twelve-month version and its drawdown is deeper. Similar
average cash allocations do not imply the same timing or stock selection.

The common equal-weight universe earns CAGR **7.62%**, Sharpe **0.433**, and
maximum drawdown **-49.51%** over these same months. The three-month cash rule
has a similar Sharpe and shallower drawdown, with lower compounded return.
No costs are deducted, so turnover remains an important comparison input.

## The same ML adjustments

All four models produce exactly the same prediction tables across windows.
Three months also has the highest CAGR and Sharpe within each model's four
versions, but every ML version earns less than its corresponding non-ML rule.
The logistic examples are:

| Window | CAGR | Sharpe | Maximum drawdown | Average cash |
| --- | ---: | ---: | ---: | ---: |
| 2 months | 2.13% | 0.288 | -20.75% | 56.92% |
| 3 months | 3.51% | 0.435 | -17.56% | 56.37% |
| 6 months | 2.42% | 0.327 | -17.93% | 55.61% |
| 12 months | 1.91% | 0.270 | -15.56% | 57.66% |

ML haircuts move capital into cash. Shallower drawdowns alone therefore do not
show useful predictive skill; an exposure-matched control would help assess that
separately. The existing prediction-quality evidence remains relevant.

## What this establishes

Twelve months was a starting convention, not a demonstrated optimum. Shorter
windows produce meaningfully different turnover, timing and results. Three
months is a candidate for subsequent research based on this validation window;
no final strategy/window is selected automatically, and the default remains
twelve for reproducibility. This strategy sensitivity study does not satisfy
the assignment's separate model GridSearchCV requirement.

See [wealth, drawdown and cash histories](outputs/momentum_20261108T211612_b4723b7c/lookback_comparison.png)
and [verification evidence](outputs/momentum_20261108T211612_b4723b7c/verification.json).
The same saved tables drive the app and are available to notebook/Quarto readers.
