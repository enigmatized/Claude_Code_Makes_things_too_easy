# Correlation-basket momentum

## App progression and deployment — 2026-11-09

Advanced ML now has **two subtabs**, keeping the six-main-tab navigation:

1. **Build the grouped strategy.** Explain the learned memberships before showing
   investment results. Inspect a saved decision month: actual stocks per group,
   the groups' average standardized input patterns and a member list. Then compare
   equal weight, ungated positive momentum/cash and full-group momentum/cash at
   1/2/3 months. Finally introduce the positive-volatility screen and compare the
   top-10-per-group book with the full groups at the same 1/2-month window.
2. **Add up/down predictions.** Choose the starting book (full groups or top ten
   stocks per group), saved momentum horizon and one of four pooled classifiers.
   Compare that exact original with separate downside-to-cash and upside-relative
   weight versions. Inspect probability quality separately from portfolio results;
   opt into stock weights/probabilities for either book.

The first subtab contains no directional-model chooser, prediction scores or
adjusted portfolios. K-means itself is unsupervised ML; it learns memberships
without next-month direction labels. Grouping alone does not define a portfolio,
so no grouping-only investment backtest is invented. The grouped cash rule and
its comparisons are existing saved books, not new strategy variants.

### What the grouping illustration actually shows

`research/basket_reporting.py` works only with saved frames. At a chosen snapshot,
the current cluster IDs define membership. Its path illustration pivots the
saved **one-month stock momentum** (an already-earned return) over the preceding
12 calendar months, centers/scales each stock using population standard deviation,
and averages standardized paths within the current groups. Older cluster IDs
are never treated as persistent sectors. It does not fit K-means or a classifier,
rebuild weights, replay returns or read source CSVs.

Coverage is explicit. The first eleven decision months have insufficient earlier
one-month observations in the saved evaluation archive, so the heatmap has an
honest unavailable state while actual membership counts remain visible. Missing
or constant paths in later snapshots are excluded from the illustration, with
the number included reported. On the default November 2009 snapshot all **1,191
of 1,191** grouped stocks are illustrated. Where coverage is complete these are
the saved members' standardized mean input paths; otherwise they describe the
available subset. They are dimensionless patterns, not percentage returns or
forecasts. The positive-volatility ranking table ranks only the recorded selected
names; scores for excluded stocks are not fabricated.

### Saved-data boundary

Both modes prominently label training-only earned returns January 2005–December
2009. `require_basket_training` validates protocol/test-source flag, input-source
fingerprints when present, actual date bounds and next-month earning alignment.
The lead study remains `baskets_20261110T021017_e6fa43e2`; its 49 books, 2,940
monthly rows, 13,500 group rows and 1,988,037 stock assignment/weight/probability
rows are unchanged. The newer explanatory/presentation code is outside scientific
calculation cache identity. All four models, 1/2/3-month full-group rules and
1/2-month top-ten rules remain available. Three-month top ten is explicitly absent.

The default deployment publication is `published/advanced_ml` (111,258,117 bytes,
about 106 MiB), containing the first three app tabs. Earlier one-/two-tab
publications are preserved. Summary scores/curves are loaded directly; saved
memberships and allocations are partitioned by model/horizon/version and loaded
only for the requested illustration/inspection. No raw CSV, experiment catalog,
replay or model fitting is required. See [deployment commands](DEPLOYMENT.md) and
[verification](verification/advanced_refactor_20261109.json). The formal report,
notebook, slides and submission release have not been changed by this UI refactor.

The sections below record the existing science and earlier UI, retained as history.

Advanced ML tab. The fourteen existing rules are unchanged, and this book is
not an experiment-catalog run.

## Directional predictions on both books

Lead study: `artifacts/basket_studies/baskets_20261110T021017_e6fa43e2`.
It keeps every original book, the top-10 overlays, and the same two
adjustments on **Basket momentum + cash** at 1, 2, and 3 months. Those
version names are `Baskets + downside to cash` and `Baskets + upside tilt`.
The earlier top-10-only file `baskets_20261109T225254_b6a479dd` remains.
Zero new classifier fits. The official test file was not opened.

Logistic regression, January 2005–December 2009, gross of costs:

| Book | CAGR | Sharpe | Max drawdown | Average cash |
| --- | --- | --- | --- | --- |
| Full baskets, 1 month | 10.07% | 0.709 | −13.9% | 57.8% |
| Full baskets + downside, 1 month | 7.92% | 0.710 | −11.0% | 67.1% |
| Full baskets + upside, 1 month | 10.01% | 0.706 | −14.0% | 57.8% |
| Top 10, 1 month | 24.68% | 0.833 | −20.5% | 50.4% |
| Top 10 + downside, 1 month | 19.41% | 0.833 | −16.1% | 61.8% |
| Top 10 + upside, 1 month | 24.98% | 0.841 | −20.6% | 50.4% |

Downside-to-cash gives back return and holds more cash, and the drawdown is
smaller. Upside tilt leaves cash where it was and barely moves either book.
The return lift is still the positive-vol screen.

## Directional predictions on the top-10 book

Earlier top-10-only study: `artifacts/basket_studies/baskets_20261109T225254_b6a479dd`.
Every original book below is retained. New prediction overlays start from the
same top-positive-volatility groups and **1-/2-month positive-momentum weights**.
Those original weights are already proportional to positive stock momentum
inside each positive group. Excluded/non-positive names remain at weight zero.

| Version | Rule | Cash policy |
| --- | --- | --- |
| Original top 10 | Existing positive group/stock momentum rule | Existing momentum cash |
| Downside to cash | `w * (1 - 0.5 * P(next return < 0))` | Every haircut stays in cash; no rescaling |
| Upside tilt | `w * (1 + 0.5 * I(P(next return > 0) >= 0.60))`, rescaled to the original equity budget | Baseline cash unchanged; qualifying stocks gain relative weight |

A 10% position with 80% predicted loss becomes 6%, releasing 4% of portfolio
capital to cash. In the upside version a qualifying 10% starts at 15% before
rescaling all positions to the original stock budget. If every held stock
qualifies equally, relative weights do not change. These are separate versions,
not a combined overlay. The upside model is fitted separately, not `1-p_loss`.

Choose logistic regression, Gaussian naive Bayes, random forest or histogram
gradient boosting in the app. Tables, both curve comparisons, cash and scores
use that model. Choose an adjusted **Basket book** to see original momentum
weight, predicted probability and adjusted weight per stock/month.

### Inputs and reproducibility

- Downside forecasts: `20261109T185402_0722a7ec725c`.
- Separately fitted upside forecasts: `20261109T185621_fe4017faa327`.
- Both: all **20 factors**, **no feature selection**, fixed settings, pooled
  stock-month training, annual expanding historical fits. Scores cover the
  full forecast universe, not only screened names.
- **Zero new classifier fits.** Reuse saved forecasts and original basket study
  `baskets_20261109T223026_c7be47ff`. Check input hashes, exact keys, historical
  cutoffs, common fitting origins/settings, and exact original-return replay.
- Earned returns January 2005–December 2009, training only. Gross of costs,
  cash earns zero. The official test CSV was not opened.
- This is separate from Tuned model's frozen **ANOVA-10 / 2007–2009** holdout;
  it does not establish the <=10-feature final submission model.

```bash
python3 run_basket_adjustments.py \
  --basket-study artifacts/basket_studies/baskets_20261109T223026_c7be47ff \
  --loss-run 20261109T185402_0722a7ec725c \
  --upside-run 20261109T185621_fe4017faa327
```

Saved metadata includes source checksums, recipes, model settings, fitting
policies and code hashes. Allocation audits retain per-stock fit cutoffs.
Notebooks/Quarto can use `research.basket_momentum.load_basket_study(path)`
without Streamlit, fitting or replay. Filter `model_key` to `none` plus the
chosen classifier, then select `momentum_months`. Optional `predictive` and
`fits` frames retain both directional sources. Old studies remain readable.
The formal submission release has not been repinned.

### Logistic example: same 2005–2009 months

| Momentum | Version | CAGR | Sharpe | Max drawdown | Average cash |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 month | Original top 10 | 24.68% | 0.833 | −20.55% | 50.43% |
| 1 month | Downside to cash | 19.41% | 0.833 | −16.12% | 61.82% |
| 1 month | Upside tilt | 24.98% | 0.841 | −20.57% | 50.43% |
| 2 months | Original top 10 | 16.14% | 0.780 | −29.51% | 45.30% |
| 2 months | Downside to cash | 12.73% | 0.782 | −23.31% | 57.87% |
| 2 months | Upside tilt | 16.40% | 0.792 | −29.05% | 45.30% |

The haircut lowers market exposure, which itself can reduce return and drawdown.
It does not establish useful forecasting. Source probability scores do not beat
the historical-frequency control on Brier/log loss. Small upside improvements
and training-period strategy exploration do not establish a final winner.
Selected-feature/tuned hybrid comparisons and the frozen official test remain
future work. Evidence: [verification](verification/basket_adjustments_20261109.json).

## Two different windows

The groups and the holding signal are not the same horizon.

- **Groups.** K-means (`k=5`) sees the z-scored path of the last 12 earned
  simple returns. That is long enough to describe co-movement. A one-month
  number has no path to cluster. The cluster id is `correlation_basket`. It
  is fit again next month.
- **Momentum.** The holding rule is scored three times, at 1, 2, and 3
  months. One month is the last earned simple return. Two and three months
  compound those earned returns. The twelve-month compound is not a weight.

A group whose own short-window return is negative is left in cash, including
members that were individually positive. Inside a positive group, only members
with a positive return over that same window are held, in proportion to it.
Cash earns 0%. Rebuild with `python3 run_basket_momentum.py`.

## Positive-volatility screen

The full-group books stay. A second book runs after the groups are formed.
Inside each group it keeps the 10 names with the highest positive volatility,
then applies the same momentum rule at 1 and 2 months. The 3-month book does
not get this screen.

Positive volatility is `sqrt(mean(max(return, 0)^2))` over the same 12 earned
months used to form the groups. Only up months count. A down month contributes
zero and still stays in the average. A name with no up month is left out.
Ties keep the lower stock id. On this window every group had 10 qualifying
names, so the book screens 50 names each month.

The smaller book is scaled to those 50 names. A name that misses the screen
is out of the book. Cash is only the momentum de-allocation inside the 50.
Group momentum is recomputed on the 10 names that remain.

## Original saved comparison (preserved)

`artifacts/basket_studies/baskets_20261109T223026_c7be47ff`

Training validation only: decisions 2004-12-31 through 2009-11-30, earned
returns January 2005–December 2009 (60 months). The 2010–2018 file was not
opened. Gross of costs. This is not the graded equal-weight test figure.
The earlier 1/2/3 comparison, before this screen, remains at
`baskets_20261109T213124_e664776f`. The 12-month holding rule remains at
`baskets_20261109T210228_f4ad9b2a`. Neither is the lead study.

| Momentum | Book | CAGR | Volatility | Sharpe | Max drawdown | Turnover | Average cash |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| — | Equal weight | 7.62% | 23.2% | 0.433 | −49.5% | 3.7% | 0% |
| 1 month | Momentum + cash | 11.65% | 18.1% | 0.693 | −21.5% | 47.9% | 46.4% |
| 1 month | Basket momentum + cash | 10.07% | 15.0% | 0.709 | −13.9% | 46.6% | 57.8% |
| 1 month | Top 10 positive-vol baskets | 24.68% | 31.7% | 0.833 | −20.5% | 53.4% | 50.4% |
| 2 months | Momentum + cash | 6.33% | 12.3% | 0.560 | −24.9% | 32.7% | 45.2% |
| 2 months | Basket momentum + cash | 5.21% | 10.0% | 0.556 | −18.0% | 33.5% | 56.5% |
| 2 months | Top 10 positive-vol baskets | 16.14% | 22.2% | 0.780 | −29.5% | 40.5% | 45.3% |
| 3 months | Momentum + cash | 7.33% | 12.7% | 0.618 | −21.1% | 26.4% | 44.4% |
| 3 months | Basket momentum + cash | 5.98% | 10.6% | 0.602 | −14.4% | 26.8% | 55.5% |

The full groups still do not raise the return above ungated short momentum.
They hold more cash and a smaller drawdown. The positive-vol screen does
raise the return. It holds about 25 of the 50 screened names. The largest
single weight is 14.9% at one month and 16.6% at two months. Volatility
rises with that concentration. The one-month drawdown is −20.5%. The
two-month drawdown is −29.5%, deeper than the full two-month group. The
strongest one-month result is April 2009, +44.4%, with 40 names held.
One-month turnover is 53%. No costs are deducted.
