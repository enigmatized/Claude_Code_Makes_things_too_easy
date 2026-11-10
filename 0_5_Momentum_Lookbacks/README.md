# Momentum lookback: 2, 3, 6 and 12 months

The annual-consistency cash strategy now has a separate configurable momentum
window. This study asks whether a shorter window better matches the user's goal
of buying historically consistent stocks only while they are rising recently.

The completed first study is documented in [findings](01_findings.md), including
all four non-ML results, ML trade-offs, run references and verification. Three
months led this training window on return and Sharpe; twelve retained the lowest
turnover and shallowest drawdown. The default remains twelve pending review.

## Protocol declared before results

- Compare **2, 3, 6 and 12 months** on identical earned-return months in the
  training-validation period, January 2005 through December 2009.
- Freeze the same majority-positive-calendar-year stock pool before evaluation;
  retain its minimum of three complete years. This annual screen does not shrink
  when the momentum window shrinks.
- Each window requires exactly that many consecutive known monthly returns for
  momentum eligibility. Missing months break eligibility. Compound those returns;
  zero/negative momentum gets zero stock weight.
- Keep the same positive-stock budget rule, cash return of zero, accounting,
  monthly rebalancing, costs convention and ML adjustment strength of 0.5.
- Compare each original with the four existing fixed classifiers using all
  twenty original factors. Predictions must match exactly across windows; the
  lookback changes allocation signals, not classifier inputs or fits.
- Include equal weight as a common non-ML control. No test CSV, feature search,
  model GridSearchCV or automatic final lookback choice is part of this study.

This is a strategy-parameter sensitivity study. Choosing a window based on its
validation results is a research choice, not evidence from an independent new
holdout. Return, risk, turnover and cash exposure should be reviewed together.

## App review

Start `python3 run_app.py`. In **Strategies**, scroll to **Momentum lookback:
2, 3, 6 or 12 months?** for the four non-ML versions and their wealth/drawdown/cash
histories. **Basic ML** contains a separately controlled version of the same study
with model selection. The baseline view exports only original cases; the model
view can export all twenty rows. Changes use the same version at twelve months.

Choose a window under **Lookback to inspect allocations**, then click **Load this
lookback's saved allocations**. The corresponding training stage opens its
allocation explorer with the actual stock/cash histories. Training selection may
change, but test selection stays separate. Loading and plotting never fit models.

For a new custom comparison, **Cash strategy momentum months** inside each
stage's explicit training-run expander changes this rule only; other registered
strategies keep their earlier twelve-month rules. Its default remains twelve, preserving earlier results.

## Python and CLI

```python
from research import ExperimentRunner, RunConfig

result = ExperimentRunner().run(RunConfig(
    strategies=("equal_weight_universe", "annual_consistency_momentum_cash"),
    cash_momentum_months=2,
))
```

```bash
python3 -m research --strategies equal_weight_universe annual_consistency_momentum_cash --cash-momentum-months 2
python3 run_momentum_study.py
```

The study command writes its protocol before computing, reuses exact saved runs
where available, and registers a completed study only after checking matching
calendars, annual pool, predictions and control returns. Each invocation has a
new study identity. Experiment artifacts remain immutable.

Completed study exports are under `outputs/<study_id>/`; the catalogued plan,
checksums and run links live under `artifacts/studies/<study_id>/`. The app,
notebooks and future Quarto sections can use the same read-only API:

```python
from research.momentum_studies import load_momentum_study
from research.store import ExperimentStore

study = load_momentum_study(ExperimentStore(), "STUDY_ID")
table = study["portfolio_comparison"]
```

`RunConfig` and cache identity record the window. Older configurations without
the field load as twelve months. Raw saved signals retain the original twelve
month columns for other strategies and add `cash_` columns for shorter windows;
the cash strategy's rebalance inspector displays its actual selected window.
CSV report import uses the saved lookback when replaying allocations.
