# Strategy runner and local app

Compare fourteen portfolio rules with four fixed next-month loss classifiers.
Data preparation, feature configuration, model fitting, portfolio construction,
and accounting have separate responsibilities. Original notebooks and their
results remain available.

The fourteenth rule is [annual consistency + positive momentum + cash](../0_4_Consistency_Momentum_Cash/README.md).
It buys only positive momentum from a frozen majority-positive-year pool; inactive
capital slots and optional ML weight reductions stay in cash. The earlier thirteen
rules retain their fully invested behavior.

Persistent loading is now implemented: SQLite indexes complete Parquet artifacts.
The app's **Load or compute comparison** reuses an exact saved run; saved runs
can also be browsed without fitting. See [saved results](../docs/SAVED_RESULTS.md)
and the [submission workspace](../submission/README.md). The experiment policy can
be expanding research fits or frozen-at-origin submission fits.

The app explicitly identifies each stage's evaluation period. **Final test**
contains optional historical non-ML 2010–2018 evidence and saved exploratory test
runs; it keeps these separate from a ready pinned final experiment.
The recent studies use training validation; final tuned test evaluation is still
pending. See [evaluation periods and controls](../docs/EVALUATION_PERIODS.md).

The Strategies and Basic ML layouts [groups related strategies](../docs/STRATEGY_GROUPS.md).
Original/model results sit under family and strategy headings; ML-change grids
show strategies as rows and models as columns. A family filter, metric selector,
detailed view, full-table option and CSV export support closer comparisons.

Reading order:

1. [Plain-language strategy guide](01_strategy_guide.md).
2. [Architecture plan and its implementation status](02_pipeline_and_model_plan.md).
3. [Sharpe/Sortino pool experiment](03_ratio_pool_momentum.md).
4. [Shared data pipeline and fixed-model comparison](04_data_and_model_comparison.md).

## Start the app

From the project directory:

```bash
python3 run_app.py
```

Open **http://localhost:8501**; stop with Ctrl+C. Use
`python3 run_app.py --port 8502` for another port. The launcher also works when
called by absolute path from another directory. It runs the Streamlit app using
the same Python environment, bound to this computer.

Dependencies are already installed in the course environment. Tested versions
are in `requirements-app.txt`; a separate environment can install them with
`python3 -m pip install -r requirements-app.txt`.

Choose Strategies, Prediction models, Feature inputs, and Evaluation period,
then **Load or compute comparison**. The preview shows the number of portfolios before you
run them. With thirteen strategies and all four models, there are **65 cases:
13 originals and 52 adjusted portfolios**. Original portfolios are computed once.
Each model's predictions are shared across every portfolio.

The app defaults to all four models and all twenty original factors. The
Python/CLI default remains logistic regression for compatibility; choose more
models explicitly there. You can disable ML for an original-only run.

## App views

- **Strategies:** non-ML training-validation originals, grouped by family. Review
  performance, the original momentum-window comparison, optional allocations,
  strategy explanations and data/timing context. No predictor controls/results.
- **Basic ML:** probability quality and the historical loss-rate control, followed
  by original/adjusted portfolio results and grouped changes. Optional allocation
  inspection and the complete source report remain available. Model and chart
  controls are specific to this stage.
- **Advanced ML strategy:** placeholder only, awaiting the user's specification.
- **Feature selection:** select training strategies supported by the saved study;
  compare all-20/ANOVA/pruned results and inspect factor audits/correlations. Missing
  comparisons are labeled unavailable; tuning is a later planned subsection.
- **Final test:** the pinned ready frozen-model result remains pending. Historical
  CSV results and exploratory saved test runs are opt-in and cannot affect the
  training tabs.

The sidebar selects a saved training experiment and shared strategy subset.
Training compute controls are explicit, inside their respective stages. Browsing,
plotting or switching tabs never fits models. Existing scientific code and
persistent artifact formats are preserved.

Allocation timelines show target weights at rebalance snapshots: December's
month-end targets earn January's returns. A zero allocation means unheld, including
after a stock leaves the available universe. Heatmaps share asset order and
color scale across models. Composition charts show a fixed set of twelve large
average original positions plus Other assets, retaining 100% of capital.
Asset filters never renormalize weights. Downloads contain complete histories
for the versions selected in the allocation view.

## Shared data and timing

The preparation layer validates source CSVs, maps forward labels to earned
months, builds lagged signals, fits required frozen pools, and defines a shared
evaluation calendar. Model input tables exclude IDs as predictors and exclude
forward returns entirely. The label table is separate and contains earned-month
and assumed availability dates.

At decision t, training joins only labels available through t. A label on source
row t is earned and assumed available at t+1 month-end. This is the existing
course execution convention. Earlier evaluation returns may update later signals
and scheduled model fits after they become known.

Default training validation uses the first 60 training snapshots as initial
history, earning Jan 2005–Dec 2009 on the remaining 60 months. It never opens the
test CSV. Frozen pools are fitted at Dec 2004, using only then-realized training
returns. Test mode earns Jan 2010–Dec 2018, with initial pools fitted at Dec 2009.

All compared cases share the same rows, calendars, history requirements, and
scheduled expanding fits. Missing returns are never filled with zero; calendar
gaps break rolling eligibility. Source extreme values are retained. Test results
were already inspected previously and are not an untouched holdout.

## Models and features

| Model ID | Fixed estimator | Preprocessing |
| --- | --- | --- |
| `logistic` | LogisticRegression: C=1, lbfgs, max_iter=1000, seed 0 | Training-fitted StandardScaler |
| `naive_bayes` | GaussianNB: var_smoothing=1e-9 | None |
| `random_forest` | 100 trees, depth 6, minimum leaf 50, sqrt feature sampling, seed 0 | None |
| `hist_gradient_boosting` | 100 iterations, learning rate .05, 15 leaves, minimum leaf 50, L2=1, seed 0 | None |

Histogram boosting's random early stopping is disabled. All models predict the
same event, `R1M_Usd < 0`, from the same declared inputs. Fits occur every twelve
decision months on expanding known history. A one-class sample uses its observed
constant class probability and records that fallback.

`FeatureSpec` identifies either all twenty original factors or an explicit
manually declared subset, including input order and lineage. The feature recipe
is shared across models in a run. Different recipes are separate runs. The
learned preprocessing belongs to each model's sklearn Pipeline, with a named
passthrough insertion point for later selectors. Automatic feature search,
learned selection, engineered composites, calibration fitting, hyperparameter
tuning, and cross-model ensembles are not performed in this stage.

The common adjustment remains:

```text
adjusted weight ∝ original weight × (1 − strength × predicted loss probability)
```

Strength defaults to .5 and must be below one. Zero strength reproduces originals
exactly; uniform probabilities do not change relative weights. This preserves
each basket and stays fully invested. There is no implicit cash position or
position cap. Accounting uses gross returns, drift-aware turnover, and zero
reference returns for Sharpe/Sortino.

## Python API

```python
from research import ExperimentRunner, FeatureSpec, RunConfig, default_model_registry

config = RunConfig(
    strategies=("equal_weight_universe", "momentum_weighted"),
    include_ml=True,
    models=tuple(default_model_registry()),
    feature_spec=FeatureSpec(),  # or FeatureSpec(columns=("Ocf_Toa", "Fcf_Yld"))
)
result = ExperimentRunner().run(config)
print(result.metrics)
print(result.paired)
print(result.model_metrics)
print(result.cases)
allocation = result.allocation(
    "momentum_weighted", "loss_adjusted", "2004-12-31", model_id="random_forest"
)
result.save("0_2_Strategy_Runner/outputs/my_new_run")
```

The output directory must not already exist. Add `include_weights=True` to
`save` or `to_zip` for complete weight histories. The case table carries
`case_id, strategy, model_id, feature_spec_id, adjustment_id`. Treat case IDs as
opaque; use the metadata rather than splitting strings.

Reports include cases, data audit, source/code hashes, exact model parameters,
feature recipes, explanations, monthly returns, portfolio diagnostics, and
model-specific predictions, fits and calibration. `result.prediction_batches`
is keyed by model ID; `result.predictions` remains a first-model compatibility
view. Existing original/logistic case IDs are retained for the default recipe.

## Terminal

```bash
python3 -m research --list
python3 -m research --list-models
python3 -m research --all --all-models
python3 -m research --strategies momentum_weighted --ml --models logistic random_forest
python3 -m research --strategies momentum_weighted --ml --features Ocf_Toa Fcf_Yld
python3 -m research --all --evaluation test
```

Runs save into new timestamped output directories. Override with `--output`;
add `--include-weights` for full histories. The full fixed-model comparison
was developed on training validation; test mode remains available explicitly.

## Extension points

| Module | Responsibility |
| --- | --- |
| `research/data.py` | Available inputs, separate labels, common fit schedule, lagged signals, frozen screens, data audit. |
| `research/features.py` | Explicit model-input recipe, stable identity and lineage. |
| `research/models.py` | ModelSpec registry, sklearn pipelines, shared chronological fitting and diagnostics. |
| `research/explanations.py` | Structured plain-language strategy rules used by UI and manifests. |
| `research/strategies.py` | Portfolio function registry and cached allocation dependencies. |
| `research/adjustments.py` | Independent probability-to-weight transformations. |
| `research/runner.py` | Experiment configuration, case identities, orchestration and accounting. |
| `app.py`, `app_charts.py` | Model-aware controls, explanations and charts. |

A new portfolio still registers a function returning keyed weights and a monthly
audit via StrategySpec. A new predictor registers ModelSpec with an estimator
factory supporting fit/predict_proba; it needs no portfolio changes. Future
ensembles should implement the same probability-table contract. Learned ensemble
weights and feature selection require chronological training-only validation.

## Verification

```bash
python3 -m unittest discover -s part0_data_and_benchmark -p 'test_*.py'
python3 -m unittest discover -s 0_1_Basic_Strategies -p 'test_*.py'
python3 -m unittest discover -s tests -p 'test_*.py'
```

Checks cover saved logistic prediction parity, original portfolio parity,
availability cutoffs, no test reads during training validation, all-model future
perturbations, shared fit dates/inputs, nonduplicated originals, feature subsets,
model-specific allocation inspection/exports, and actual app interactions.
