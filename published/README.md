# Published presentation data

`project/` is the current default **seven-tab** publication, about 173.3 MB. It
includes K-means and Ward hierarchical grouping with the same trading rules;
adds feature selection (including advanced books), both tuning targets/trials,
and frozen test/training/entire-path views to the earlier three-tab package.
No raw data, experiment catalog, fitted estimators or runtime model/backtest
execution is needed. It also includes the completed feature-engineering training/tuning/final study,
with six input recipes and all four classifiers. Momentum acceleration adds
separate stock/basket deceleration-to-cash comparisons for both grouping methods,
with training and previously viewed test evidence. The 476-file release preserves
all 469 prior frame hashes. Earlier publications and the pre-acceleration default
are preserved outside Git; see [acceleration protocol](../docs/MOMENTUM_ACCELERATION.md).

`advanced_ml/` preserves the three-tab publication. It preserves the
first two tabs and adds saved K-means memberships, grouping-pattern illustrations,
momentum/cash and top-ten comparisons, and directional adjustments. About 111.3 MB,
with allocation/membership partitions loaded on demand. No raw CSVs, experiment
catalog, model fitting or portfolio replay. The final three tabs remain pending.

`basic_ml/` preserves the immutable two-tab publication: the same Strategies
evidence plus both event directions, four pooled classifiers and per-stock
logistic comparisons. It contains scores, calibration, portfolio histories,
paired changes, lazy allocation histories and fit audits; no raw stock-month
prediction panels or source datasets. It is about 56.6 MB. The remaining four
deployed tabs stay pending.

`strategies/` preserves the first immutable, self-contained first-tab publication.
It contains only saved non-ML training results, allocation histories, explanations,
and the optional baseline momentum-window comparison. It does not contain the
source dataset or the research experiment catalog.

Preview with `python3 run_app.py --mode deploy --port 8502` from the project root.
Use `--published-root published/strategies` or `--published-root published/basic_ml`
to inspect the earlier releases.
See [deployment modes and next-tab handoff](../docs/DEPLOYMENT.md) for export
commands, schema, provenance, limitations and future tabs. Do not edit generated
Parquet or the manifest in place; publish another reviewed directory instead.
