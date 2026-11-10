# Saved experiments and SQL

The app and CLI reuse exact saved experiments before computing. A run restores
all portfolio weights, signals, predictions, books and diagnostics. Allocation
charts therefore work without refitting models.

Monthly portfolio tables/books already include `simple_return`, `log_return`
and `wealth`; complete metric tables include `cagr` and `annualized_log_growth`.
The app's return-measure selector and growth columns read these artifacts.
See [return metrics](RETURN_METRICS.md) for definitions and display behavior.

The five-stage app uses separate training and test selections. Strategies can
read baseline cases from a saved comparison that also contains models; the
view does not rewrite that experiment's configuration or manifest. Basic ML
reads the same source with its model filters. Final test requires a deliberately
pinned ready frozen result for its primary slot; exploratory tests are separate.
Training compute buttons are explicit and preserve exact-match reuse.

For explicitly cash-capable cases, stored weights contain real stocks only and
may sum to less than one. The residual is cash; books and diagnostics store
`equity_weight` and `cash_weight`. Cash earns an assumed zero return and is included
in turnover. App allocation charts/downloads add a labeled Cash series. Earlier
fully invested cases retain their original frames and remain readable.

`cash_momentum_months` is part of the saved configuration and cache identity.
Historical configurations missing it load as twelve months. Registered momentum
studies link the 2/3/6/12-month runs, with checksummed metadata/exports and matched
protocol checks. App study review reads those artifacts without fitting models.

The Basic ML tab selects its own saved target/scope experiment. Its explicit
compute action updates `basic_result`, leaving the first tab's `training_result`
intact. Saved configurations distinguish `prediction_target`, `model_scope`,
`upside_threshold`, `min_stock_history` and `min_stock_class`; older runs load
loss/pooled defaults. Per-stock prediction rows record their actual local or
pooled fallback source. Fit summaries record stock/history/fallback details.
Estimator objects are not persisted. See
[basic ML directions](../0_6_Basic_ML_Directions/README.md) for fixed comparisons.

## Storage

- `artifacts/catalog.sqlite`: experiment discovery and queryable key metrics.
- `artifacts/<run_id>/artifact.json`: configuration, provenance, table paths and
  file hashes, with its own hash indexed in SQLite.
- `artifacts/<run_id>/**/*.parquet`: complete typed, losslessly compressed tables.
- Existing experiment CSVs remain unchanged as readable historical exports.

The first full 65-case run occupies about 32 MiB in this format. SQLite is built
into this Python environment; no database server is required. PyArrow is already
installed and is pinned in `requirements-app.txt`.

The cache key includes the full configuration, source content hashes, calculation
code, model settings and numerical runtime. A matching configuration with changed
data or code is a new computation. Storage/UI/report code is outside the numerical
identity so improving a viewer does not unnecessarily refit models.

Files are written to a temporary run directory before publication in SQLite.
Incomplete runs are not listed. Loads check metadata and artifact integrity;
damaged records are not silently recomputed. Historical experiments remain
browseable as recorded evidence when their code is no longer current.

## App and CLI

The app loads a saved result on startup when one is available. Choose a saved
comparison and select **Load saved comparison** to browse another without fitting.
The editor's **Load or compute comparison** uses an exact current-code match if
available, otherwise computes and saves a new run. **Recompute even if saved**
is an explicit override. Changing chart selections does not fit models.

Automatic reuse currently matches complete experiment configurations. Use chart
and model-view filters to inspect a subset of a saved comparison without creating
a new experiment. Reusing individual prediction tables across different run
configurations is a later optimization, distinct from the complete-run cache.

```bash
python3 -m research --saved
python3 -m research --load RUN_ID
python3 -m research --all --all-models
python3 -m research --all --all-models --force
```

CSV exports continue to use new output directories and never overwrite earlier
exports. Every durable run already includes complete weights; `--include-weights`
only controls whether those histories also appear in the CSV export.

For compatible schema-2 historical CSV reports:

```bash
python3 -m research --import-report 0_2_Strategy_Runner/outputs/fixed_models_validation
```

Import verifies source and allocation-code hashes, replays saved probabilities,
and checks every monthly book against the archive. It fits no models. The
historical model-code provenance is preserved; an import does not pretend that
old predictions were fitted by current code.

Use `--artifact-root PATH` or `MLFACTOR_ARTIFACT_ROOT` for a separate library.
App tests use temporary libraries rather than polluting the research catalog.

## SQL example

From the project directory:

```python
import sqlite3
import pandas as pd

with sqlite3.connect("artifacts/catalog.sqlite") as connection:
    table = pd.read_sql_query("""
        SELECT r.label, r.evaluation, m.strategy, m.model_id,
               m.cagr, m.sharpe, m.max_drawdown
        FROM case_metrics AS m
        JOIN runs AS r USING (run_id)
        ORDER BY r.created_utc DESC, m.strategy, m.model_id
    """, connection)
print(table)
```

Record IDs and feature IDs make later feature/tuning comparisons traceable. Keep
large CV searches compact: store candidate scores/parameters, then full books
only for the configurations actually compared as portfolios.

## Feature studies

Run `python3 run_feature_selection.py` for the fixed three-recipe, four-model,
thirteen-strategy training-validation experiment. It reuses exact compatible
runs and registers a new study linking them. Individual recipes are also
available with `--feature-selection anova` or `--feature-selection
correlation_pruned` on the research CLI; `--max-features` and
`--correlation-threshold` are explicit configuration values, not searches.

`artifacts/studies/<study_id>/` contains the plan written before results,
`study.json`, portfolio/prediction comparisons and fit diagnostics. The SQLite
`studies` table indexes completed metadata by checksum. Full allocations and
predictions stay in the three linked runs. Selector diagnostics are stored with
each fit: actual names/counts, rankings, exclusions and training correlations.

The **Feature selection** tab loads a registered study, verifies that the runs
share the same non-feature settings and original books, and plots saved results.
It never computes a new study on a dropdown change. See the
[first findings](../0_3_Feature_Selection/02_findings.md) for pinned IDs.
