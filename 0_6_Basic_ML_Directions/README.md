# Basic ML: downside/upside and pooled/per-stock

In the app, open **Basic ML → Basic ML pattern** and choose **Downside
protection** or **Upside threshold boost**. **Model training scope** selects
**Pooled across stocks** or **Separate model per stock**. Each combination reads
its own saved experiment. The local saved-experiment chooser shows its actual
feature recipe and cash-momentum window. Changing controls never trains a model.

Prediction quality appears before the original/adjusted portfolio comparison.
**Performance through time** initially selects all available strategies in Basic
ML (within the current family/strategy filters). You can then narrow the chart;
your choices persist while reviewing results. Other tabs keep their existing defaults.
The allocation inspector includes the selected event probability and, for local
fits, the prediction source/history/fallback reason. **Compare pooled and
per-stock portfolios** reads matched saved experiments, with common models and
identical trading settings, factors, data and calculation provenance. That
comparison includes predictive scores as well as portfolio results.

The first matched research comparison is declared in `outputs/*/plan.json` before
computation. `comparison.json` records actual run IDs, elapsed time, artifact
size, fit-summary counts and prediction-source counts. CSVs collect the evidence
for later notebooks/Quarto. See [findings](01_findings.md) once the run completes.

## Two weighting patterns

The existing downside rule predicts `R1M_Usd < 0`:

```text
raw weight = original weight × (1 − strength × P(loss))
```

Fully invested strategies normalize back to 100% equities. The cash-capable
strategy leaves the haircut in cash. This behavior is preserved exactly.

The new upside rule fits the separate target `R1M_Usd > 0`:

```text
raw weight = original weight × (1 + strength × I(P(up) >= threshold))
weight = raw weight / sum(raw weights) × original equity budget
```

The initial fixed threshold is 0.60 and strength is 0.50. For two equal positions,
if only the first meets the threshold, their normalized weights become 60%/40%.
If the strategy originally invested 80% and held 20% cash, the corresponding
weights become 48%/32% and cash stays 20%. A zero-weight stock stays excluded.
All-cash portfolios stay in cash. If all held names get the same multiplier,
weights are unchanged. The adjustment is relative and does not promise higher
absolute weight when other names receive the same boost.

Flat returns are neither a strictly positive return nor a loss. Consequently,
`P(up)` is not obtained by silently complementing the saved loss probabilities.
The upside classifiers are fitted separately. The direction probability does not
predict return size. Accuracy in the score table uses 0.50; the portfolio gating
threshold is separately recorded. No threshold tuning has been performed.

## How a model predicts a stock

**Pooled (the original design):** each algorithm learns one model at each
historical origin using all then-known stock-month rows. It predicts each stock
using that stock's current factor values. `stock_id` is a grouping/join key, not
a model input. With five annual origins and four algorithms, there are twenty
pooled fits per target.

**Per stock (optional experiment):** the algorithm fits separate preprocessing
and an estimator for each eligible stock at that same origin. Only that stock's
known labels/features are used. Learned selectors, when requested, also fit
within its local training slice. Stock models need at least 36 observed historical
months and at least six observations of each target class. Otherwise, a
same-origin pooled model supplies the prediction. New/unseen stocks also use
this explicit fallback. The pooled fallback is fitted once per algorithm/origin,
not once per stock.

Predictions record `prediction_source`, `stock_history_months` and
`fallback_reason`; fit summaries additionally identify the stock and actual
training window. Each scope is evaluated on exactly the same stock-month rows,
including fallbacks. Both use the same **pooled historical event-frequency
control**, so a different benchmark cannot explain a score difference.

The first per-stock comparison uses logistic regression. The backend supports
all registered algorithms, but the current tree leaf-size defaults were chosen
for pooled datasets and can produce nearly constant local predictions. Avoid
interpreting those unchanged defaults as well-tuned per-stock models.

## Data size and interpretation

The prediction table still has one row per stock-month/algorithm. Per-stock
training adds thousands of short fit-summary rows, rather than multiplying
forecast rows by the number of stocks. Fitted estimator objects are not stored;
we save predictions, training provenance, metrics and portfolio histories.
More compute is needed, but storage is manageable for the current panel. Inspect
actual bytes and runtimes in the recorded comparison rather than extrapolate
from model-object sizes.

The more serious constraint is sample size: at the first validation origin the
longest individual history is about 60 monthly observations for twenty supplied
factors, whereas pooled training has tens of thousands of rows. Local models may
fit noise. Chronological forecasts can expose that behavior; fitting a stock
model does not itself establish that it overfits or outperforms. Compare Brier
score/log loss with the control, calibration, return, drawdown and turnover.
Neither scope nor target is declared the winner in advance.

All experiments are training validation, earned Jan 2005–Dec 2009 by default.
Inputs and labels are joined only after labels become available. The first
matched comparison uses three-month momentum for the cash strategy, all twenty
ML factors, annual expanding refits and no feature/tuning search. Official frozen
final-test evaluation, <=10 final factors and GridSearchCV remain separate work.
The Advanced ML tab stays undefined.

## Reproduce explicitly

```bash
python3 run_basic_ml_comparison.py --cash-momentum-months 3
python3 -m research --all --all-models --prediction-target upside --upside-threshold .6 --cash-momentum-months 3
python3 -m research --all --ml --models logistic --prediction-target upside --model-scope per_stock --cash-momentum-months 3
```

The first command creates a new timestamped comparison folder, declares the
protocol and requests four saved experiments. Exact current configurations reuse
results; changed numerical code/configurations receive new cache identities.
Earlier runs and outputs are preserved and remain readable. In the app, the
local **Model settings and explicit training run** expander can request other
settings. The actual loaded-run caption, not pending controls, defines the result.
Basic ML experiments have independent state from the first tab's source.
