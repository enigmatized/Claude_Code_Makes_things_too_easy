# Feature engineering — notebook review and end-to-end integration

Updated 2026-11-10. User authorization covers new input/model variants, training,
tuning, frozen final evaluation, local/deployed app support and GitHub readiness.

## Notebook review

`0_7_Feature_Engineering/01_feature_engineering.ipynb` usefully explores factor
redundancy, technical signals from already-earned returns, sign-aligned averages,
within-cluster PCA, IC-weighted blends, greedy decorrelation and economic
composites. Its forward-label timing for technical inputs is appropriate. The
synthetic price/MA ratio reconstructs relative price levels from returns; it is
not an actual price series supplied by the dataset.

The original snapshot is preserved under
`artifacts/notebook_reviews/20261110/01_feature_engineering.ipynb`, SHA-256
`d5cf0928008069fdc1a345dddb7588f75a014e1d9874c2caa4406ac7198f93b7`.
The notebook now has a visible integration review. Its PCA/scaler export includes
full loadings/means/scales for future reruns; IC weights retain full precision;
positive-IC month percentages exclude missing IC observations. Existing numeric
outputs and extended CSVs were not rerun/overwritten by the review.

The main concerns addressed in the app workflow are:

- Full-training composite fits are inappropriate for historical validation
  snapshots inside that training period. Fit every learned transform within each
  chronological CV fold and freeze it before the later evaluation window.
- Unsupervised fits are not immune to overfitting: choosing groups/components
  after inspecting returns is still model selection. PCA maximizes explained
  feature variation, not predictive return information.
- IC weights use labels. Their own fit-sample IC is optimistic; a future-period
  diagnostic is needed. IC/Spearman and Q5−Q1 log spreads are factor diagnostics,
  not the realized return of a long-only portfolio.
- Ten composite columns may aggregate twenty original factors. Disclose both
  counts; do not assert formal rubric compliance without checking whether it
  limits underlying sources or final model inputs.
- The original notebook has already viewed both training and test results.
  New results are retrospective research, not fresh untouched evidence.
- Never feed the extended CSV wholesale into ML: it contains `R1M_Usd` and
  `fwd_log_ret`. The new estimator accepts only explicit key/source/technical
  fields and rejects these forward labels in its feature constructor.

References: [sklearn preprocessing leakage](https://scikit-learn.org/1.4/common_pitfalls.html#data-leakage),
[PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

## Declared experiment

A new **Feature engineering** main tab follows Tuned model and precedes Final
test. Its subtabs are **Methods & feature construction**, **Model & portfolio
results**, and **Hyperparameter tuning**. This groups factor *columns*, distinct
from stock grouping in Grouping With ML. The separate tuning section avoids
adding another main navigation tab.

Six declared recipes are independently trained with all four existing
classifiers and separately fitted loss/upside events:

| Recipe | Constructed model inputs |
| --- | --- |
| ANOVA-10 reference | Ten original factors, selected within each fold for the chosen event. |
| Technical-only | Ten of the eleven notebook technical ranks; no raw factor used by the classifier. |
| Cluster averages + technical | Six sign-aligned factor-group averages + four technical ranks; tuned group count can be four. |
| Cluster PCA + technical | One training-scaled PC1 per factor group, including singleton groups, + four technical ranks. |
| IC blends + technical | All-factor IC blend and ICIR-greedy/decorrelated blend + eight technical ranks. |
| Economic composites + technical | Five fixed rank-based heuristics + five technical ranks. |

Technical definitions follow the source notebook: last earned return; 3/6/12M
return means; 3M−12M difference; synthetic price versus 3/6/12 price SMA; 12−1
momentum; 6/12M log-return volatility. The technical-only recipe takes the first
ten in the declared ordering, omitting 12M volatility. Cluster recipes use last
return, average crossover, 12−1 momentum and 12M volatility. Economic recipes add
6M price/MA to those four. The order/lineage is explicit in `engineering.py`.

Rolling calculations use a calendar grid. Missing/gapped histories stay missing
until cross-sectional ranking, then receive neutral 0.5. Nonfinite log history,
including total loss, is unavailable. No recipe drops a different stock-month
universe. Current-month re-ranking requires a complete scoring cross-section;
call prediction on whole months, not arbitrary stock batches. Current observable
cross-sectional ranks are not a training fit on future outcomes.

Factor clustering uses average linkage on `1 − |Spearman correlation|`, not the
Ward stock-group algorithm. Fixed factor-group count is six. Signs align to the
member with highest average absolute within-group correlation. PCA loadings align
to that group's average. Economic averages combine percentile ranks, not raw
ratios; the yield/financing blend includes net-debt yield and should not be
interpreted as literal shareholder payout.

## Tuning and timing

The protocol is written to `study.json` before computation. Training-stage labels
are known through 2006-12-31; evaluate earned January 2007–December 2009 returns.
Each model is frozen through that holdout. Three expanding month folds with at
least 24 initial months choose parameters before the holdout. All stock rows from
a month stay together and label availability is enforced.

Two declared classifier configurations are searched for each model: logistic
`C={1,.1}`, Bayes smoothing `{1e-9,1e-5}`, forest `(depth=6,leaf=50)` versus
`(depth=4,leaf=200)` at 100 trees, and histogram learning rate `{.05,.1}`. Cluster
recipes additionally search four/six groups; IC recipes search pruning cap
`.4/.6`. Thus every model has two or four candidate configurations, each refitting
its feature map in each fold. Output count stays at most ten.

`EngineeredClassifier` retains continuous returns as its external `y` so the IC
recipe can fit future-return ranks as in the notebook. Its internal estimator
receives binary next-up/down labels. GridSearchCV's explicit scorer applies
standard sklearn binary **log loss**, with the same event mapping. It does not
optimize portfolio performance. This explicit scorer avoids accidentally applying
binary log loss to continuous returns. Fit-only feature cache keys hash every
input and label plus recipe settings; identical maps may be reused across
algorithms without crossing folds or using validation rows.

After training completes, a separately requested final phase carries those
chosen settings unchanged, refits feature construction and fixed/tuned classifiers
once using all training data at the first test snapshot (2009-12-31), then scores
2010–2018. No test-label model/IC/PCA fit or retuning occurs. Lagged technical
signals and stock clusters update using returns already earned, under the
existing trading protocol. Both fixed/tuned variants are shown without selecting
a winner after the official-test results.

## Same portfolio rules

Each variant overlays six baseline books at one-month momentum: equal weight,
ungrouped positive momentum/cash, K-means full groups, K-means top ten, Ward full
groups and Ward top ten. Stock-group settings remain five groups / twelve-month
return paths; top-ten uses positive-side volatility. The original group/weight/
accounting helpers are reused. Non-positive-momentum holdings stay at zero and
residual capital stays in cash.

Downside uses `w*(1-.5*p_loss)`; reductions stay in cash for the cash-capable
books, while the equal-weight overlay renormalizes to 100% stocks. Upside uses a
50% boost at `p_up >= .6`, rescaled to the original equity budget. Separate up/down
models remain separate portfolios, not a combined signal. No trading costs are
deducted and cash earns zero. The same baseline stock/month keys apply to every
recipe/model/target/version.

## Storage and display boundary

`research/engineering.py` owns technical inputs, fitted maps and estimators.
`research/engineering_study.py` explicitly orchestrates computation, saved cases,
provenance and aggregate readers. `engineering_ui.py` only reads/plots.
`engineering_publication.py` publishes and checks aggregate training/test frames.
Earlier scientific modules, model cache identities and experiment files are
preserved. New studies live under ignored `artifacts/engineering_studies/`.

A complete case saves both fitted models, keyed fixed/tuned probabilities,
full-precision feature/model audit, portfolio metrics/monthly returns, predictive
scores and CV trials/folds. Completed case frames are checksummed and can resume;
code/data hash changes reject resumption. Aggregates become visible only after
`complete.json` is written. Each stage is independently complete; local final
view reads the latest completed final stage rather than hiding it behind an
in-progress newer training study.

The publication adds optional schema-1 `tabs.engineering`, preserving every prior
frame checksum. It exports only metrics, curves, probabilities' score summaries,
construction audits and CV tables. No extended CSV, raw prediction panel or
fitted estimator is needed by the host. Missing releases show an explicit pending
state. The existing final books retain their own historical protocol; new
engineered fixed/tuned variants appear in a separately labeled final section.

## Commands and report reuse

```bash
python3 run_feature_engineering.py
python3 run_feature_engineering.py --study artifacts/engineering_studies/STUDY
python3 run_feature_engineering.py --phase final --study artifacts/engineering_studies/STUDY
python3 publish_results.py --extend-publication published/project \
  --engineering-study artifacts/engineering_studies/STUDY \
  --output published/NEW_RELEASE
python3 run_app.py
python3 run_app.py --mode deploy --port 8502
```

`--study` resumes completed cases instead of starting a new full run. `--recipes`
and `--models` declare a smaller new study if explicitly requested. Exports are
immutable. Review the new publication, preserve the old default release locally,
then promote the reviewed directory to `published/project` before pushing.

`0_7_Feature_Engineering/02_saved_model_results.ipynb` and
`0_7_Feature_Engineering/engineering_results.qmd` read the same published frames
and shared chart helper. Rendering uses `MLFACTOR_READ_ONLY=1`. These are saved-
evidence review supplements, not a repin of the formal submission or its slides.

`.gitignore` already excludes extended CSVs and the entire new artifact tree;
its comments now describe the general app release. Current published Parquet,
source/notebooks/QMD/docs remain eligible. The small historical `fit_params.json`
is already tracked and remains as an original reference; it is not used as the
canonical fitted state. No Git index/history change is required.

## Verification and handoff

See `docs/verification/feature_engineering_20261110.json` for actual source IDs,
sizes, counts, tests and browser/standalone evidence after computation. No model
selection or performance improvement should be claimed just because a recipe
exists. Review training and final predictive/portfolio results together, with
source-count and previously-viewed-test caveats. Formal project report/signoff
remains a separate task. Preserve other LLM edits under `context/` and the older
supervised `research/grouping.py`; that quintile classifier study is distinct.

## Completed release and evidence

Study: `engineering_20261110T171255_825ffa99`. Training has 582 portfolio cases /
20,952 monthly rows / 96 fitted variants / 144 CV candidate rows; final has
582 cases / 62,856 monthly rows / 96 separately frozen fitted variants. The
training fit uses 97,788 known-label rows, final uses all 140,739 training rows.
There are 192 saved fixed/tuned fitted objects; 288 local model/probability
checksums are recorded. The first full suite exposed a month-end error in a new
test fixture, corrected with period-based month shifting. The prior logistic
parity test also required the established one-native-thread environment; it
passes without changing its tolerance or the original model code. Final suite:
221 passes (219.838 s), plus focused checks after UI changes.

Publication SHA-256: `4df420855fe8e06be10ecf8cf52011fb6eae4e1147d9150fbab2737b04b65e1d`;
470 files / 172,577,499 bytes. All 457 earlier frame hashes remain unchanged.
Old default remains under ignored `published/project_before_engineering_20261110`.
The default live package is `published/project`; no commit/push was made.

The study does not demonstrate automatic improvement: on the training holdout,
fixed logistic downside log loss is 0.700112 for ANOVA-10 and 0.700181 for
technical-only; the other composite logistic variants are also slightly worse.
This is one prespecified example, not a test-selected winner. The app exposes
the full prediction/portfolio comparisons rather than hiding negative findings.

Real-data validation checks exact exported frames, every variant's growth/cash
identities, and agreement to numerical precision with the four previously saved final controls.
A fresh code-plus-publication copy exercises seven main tabs, all methods/four
classifiers/both directions, fixed/tuned trials and Ward final curves with CSV,
catalog, fitted-object loading and computation patched to fail. Browser charts
cover the factor tree, PCA variance/loadings, IC weights, recipe/forest trials
and final Ward/HGB log-return/drawdown/cash plots. A pre-existing sidebar state
warning was removed while preserving the default all-strategy selection.

The companion notebook is executed and Quarto supplement rendered. Its Plotly
figure uses a self-contained encoded srcdoc frame to avoid notebook AMD-loader
collisions, Quarto moving dependencies ahead of the title, and large data-URL
navigation limits. Browser verification confirms 21 traces, correct document
placement and successful rendering with all HTTP/HTTPS requests blocked. Formal submission release/hash remains unchanged.
