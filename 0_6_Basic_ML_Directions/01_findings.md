# First direction/scope comparison — 2026-11-09

The per-stock logistic models have much worse probability scores than the pooled
models on the same training-validation forecasts. Neither the pooled nor local
models beat their historical event-frequency control on Brier score or log loss.
Keep the pooled setup as the starting point; retain local fitting as useful
negative research evidence, rather than presume more models improve predictions.

## Fixed protocol

Fourteen original portfolio rules; earned Jan 2005–Dec 2009; first 60 snapshots
as initial history; annual expanding fits; all twenty supplied factors; strength
0.50; upside threshold 0.60; cash-strategy momentum three months. Pooled runs use
all four fixed classifiers; local runs use logistic regression. Local fits need
36 known months and six observations in each event class, otherwise pooled fallback.
No test experiment, feature search, threshold tuning or GridSearchCV was performed.

The protocol was saved before computation in
[plan.json](outputs/20261109T185258Z/plan.json). Exact run provenance and measured
sizes are in [comparison.json](outputs/20261109T185258Z/comparison.json).

## Saved experiments

| Target | Scope | Models | Cases | Run ID | Seconds | MiB |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| loss | pooled | 4 | 70 | `20261109T185402_0722a7ec725c` | 64.8 | 35.91 |
| loss | per_stock | 1 | 28 | `20261109T185517_1bf238ea2712` | 74.5 | 15.69 |
| upside | pooled | 4 | 70 | `20261109T185621_fe4017faa327` | 64.6 | 28.81 |
| upside | per_stock | 1 | 28 | `20261109T185736_3b19b90c1122` | 74.4 | 13.94 |

Observed elapsed times include saving and are machine-specific. Total new
artifacts: **94.3 MiB**. Earlier runs remain intact.

## Common logistic model: prediction quality

| Target | Predictor | Scope | Brier ↓ | Log loss ↓ | ROC AUC |
| --- | --- | --- | ---: | ---: | ---: |
| loss | Logistic regression | pooled | 0.249667 | 0.692578 | 0.500801 |
| loss | Historical loss-rate control | pooled | 0.249023 | 0.691215 | 0.494807 |
| loss | Logistic regression | per_stock | 0.333672 | 1.140251 | 0.502040 |
| loss | Historical loss-rate control | per_stock | 0.249023 | 0.691215 | 0.494807 |
| upside | Logistic regression | pooled | 0.250037 | 0.693300 | 0.498507 |
| upside | Historical up-rate control | pooled | 0.249364 | 0.691891 | 0.493470 |
| upside | Logistic regression | per_stock | 0.334235 | 1.143798 | 0.501863 |
| upside | Historical up-rate control | per_stock | 0.249364 | 0.691891 | 0.493470 |

These are chronological validation predictions, not scores on fitted rows.
The controls use pooled, then-known event prevalence in both scopes. Each model
has exactly 71,631 forecasts on the same stock-month keys. Both local runs have
70,792 stock-only forecasts and 839 pooled fallbacks (about 1.17%). Each has
5,981 fit-summary rows: 5 pooled fallback model fits, 5,904 actual local fits,
and 72 stock/origin fallback summaries. Models are not serialized.

Local probabilities range widely but the observed positive frequency stays
near the unconditional rate across their calibration bins. That poor calibration
and approximately 0.50 AUC argue against treating confident local forecasts as
reliable signals. Limited monthly samples relative to twenty inputs are a
plausible contributor; these scores alone do not isolate a single cause.

Pooled upside logistic probabilities meet the 0.60 trading threshold on 8.94%
of stock-months, versus 44.11% locally. A more frequently triggered boost does
not imply better forecasts. No threshold was chosen to maximize these results.

## Example: annual consistency + three-month momentum + cash

| Version | Scope | CAGR | Annual log growth | Sharpe | Max drawdown | Average cash |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original | — | 4.42% | 4.33% | 0.437 | -22.03% | 44.17% |
| loss adjustment | pooled | 3.51% | 3.45% | 0.435 | -17.56% | 56.37% |
| loss adjustment | per_stock | 3.69% | 3.63% | 0.472 | -15.88% | 57.85% |
| upside adjustment | pooled | 4.33% | 4.24% | 0.429 | -22.26% | 44.17% |
| upside adjustment | per_stock | 4.58% | 4.48% | 0.452 | -21.51% | 44.17% |

The downside versions increase cash, so risk changes include lower exposure.
The upside versions preserve the original 44.17% average cash allocation.
One local upside portfolio has a small realized CAGR gain, despite poor
prediction quality. This is not enough to establish a dependable ML advantage.
Other rules have mixed changes; see the complete recorded CSVs rather than
selecting a winner from a favorable row.

## Verification and next handoff

- 138 research/app tests passed: separate event labels (including flats),
  threshold accounting, local fit isolation, temporal/frozen-fit guards,
  fallback, cache identity, artifact roundtrip/CSV replay and saved-only UI.
- All four existing pooled probability tables and all seventy portfolio
  books match earlier experiments at matching settings exactly. Reference
  sources are `20261108T234216_191b0784406b` for the thirteen original rules
  and `20261108T211720_6c1ba3996026` for the three-month cash rule.
- The fourteen original return histories match across all four new runs.
- Real app review with fitting disabled checked every target/scope, saved
  scores/portfolios, log views, allocation probability/source columns, legacy
  studies and separation from the first tab. All 1,570 prior data/artifact
  hashes match. All four new runs are exact current-configuration cache hits.
- Real browser review covered pooled/local upside calibration and the matched
  predictive/portfolio scope comparison. See
  [verification](../docs/verification/basic_directions_20261109.json).

Feature studies still cover the earlier loss/pooled configurations; these
upside/local results are not a new feature-selection comparison. Submission
tuning, <=10 final inputs, justified choices and frozen final test remain
pending. Quarto/slide work can consume these saved CSVs/Parquet summaries
without refitting; the release pin and Advanced ML placeholder are unchanged.
