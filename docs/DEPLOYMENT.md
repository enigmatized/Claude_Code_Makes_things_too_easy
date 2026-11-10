# Research and published view modes

Latest visibility fix: the first Final subtab is **K-means strategies**, with
explicit original top-ten / one-month momentum / cash labels. Engineered final
comparisons default to the same starting book. This is a code/docs-only update;
`published/project` and its 470 files remain unchanged. No dependency or hosting
setting change is required.

Current release (2026-11-10): **seven main tabs**, including Feature engineering
and its new final variants, are published in `published/project` (470 files /
172,577,499 bytes). Only about 2.6 MB is added for the new training/final model
summaries, construction audits and search/fold tables. No raw technical dataset,
forecast panel or fitted model object is uploaded. `deploy_app.py`, Python 3.12 and
existing dependency pins stay the same. Push code, docs/review sources and the
updated current release. Older publications/backups and all new scientific
artifacts stay ignored. See [extension protocol](FEATURE_ENGINEERING.md).

Latest UI review (2026-11-10): Tuned model exposes all fourteen saved rules;
Final test adds log-return views, original-relative changes and explicit fixed
K-means protocol coverage. No publication update or raw data is required for
this UI change: the 458-file `published/project` remains exact. Git ignore rules
already include that release and code while excluding raw/research/old/local
files; no ignored files are already tracked. See
[review and push handoff](TUNING_AND_FINAL_REVIEW.md).

Current (2026-11-10): the user confirms the site deployed successfully.
The updated `published/project` has **169,973,507 bytes / 458 files**, including
K-means and Ward hierarchical grouping in the renamed **Grouping With ML** tab.
Its three subtabs share research/deploy renderers. Push the updated code,
`requirements-app.txt` (SciPy 1.13.1) and `published/project` to update the site.
No hosting setting or entry-point change is needed. Preserve deployment's
read-only boundary; no raw CSV/catalog/fitting/replay is introduced. See
[new export command, provenance and next steps](GROUPING_WITH_ML.md).

All **six main tabs** now support deployment.
Local research retains all six tabs and its explicit compute buttons. Deployment
reuses the same renderers against a self-contained publication. Older publications
still show honest pending states for sections they do not contain.
The user reports successful website deployment.

## GitHub staging scope

The root `.gitignore` keeps the source code, notebooks/QMD sources, docs, pinned
requirements, `.streamlit/config.toml` and current `published/project` release
available for Git. It excludes CSV datasets/reports, `artifacts/`, research
`outputs/`, older published releases, generated Quarto builds, caches, virtual
environments and local credentials. The current release's Parquet and JSON files
are deliberately included: the deployed app needs them. Files remain on disk.

After initializing your repository, review `git status --short` and
`git add --dry-run .` before staging. Use `deploy_app.py` as the Streamlit Cloud
entry point. The ignore-file change does not initialize Git, create a remote,
commit or push anything. Local research/report reproduction still needs the
excluded datasets and experiment archive; hosting uses only the published release.

## Run either mode locally

```bash
python3 run_app.py                         # research, default port 8501
python3 run_app.py --mode deploy --port 8502
```

`MLFACTOR_APP_MODE=research|deploy` is the equivalent setting for a direct
`streamlit run app.py` launch. Do not infer mode from hostname or current folder.
`deploy_app.py` is the Community Cloud entry point: it explicitly sets
deploy mode and `MLFACTOR_READ_ONLY=1` before opening the app. Research mode is
the default; the website cannot switch itself into research mode through a UI
control. The first-tab and Basic ML compute callbacks also reject deploy mode.

Preview a different publication with:

```bash
python3 run_app.py --mode deploy --published-root /path/to/new-publication --port 8502
```

Alternatively set `MLFACTOR_PUBLISHED_ROOT`. By default the reader uses
`published/project` relative to the Python source, independent of the current
working directory. Missing, corrupt or unsupported publications show an error;
they never fall back to a local archive, source CSV or new computation.

## Export completed evidence

The first publication was made with:

```bash
python3 publish_results.py \
  --run 20261109T202421_184e95372ab2 \
  --momentum-study momentum_20261108T211612_b4723b7c
```

This reads an existing completed experiment and the optional completed momentum
study, verifying their saved artifacts. It does not call a runner, train a model,
search settings or replay a portfolio. It does not require the source CSVs.
Exports are immutable: an existing destination is rejected. Publish later runs
with `--output published/releases/your-release-name`, review them, then point
deployment to that directory. Use `--artifact-root` for a non-default local archive.
Do not automatically publish every intermediate experiment.

Existing research runs already save per-case monthly portfolios, metrics and
weights, plus model diagnostics. Models therefore do not need refitting to make
chart exports. Use `--basic-run` with explicit saved run IDs to add model cases.
Without that option, the original first-tab-only export still works. Existing
publications are never updated in place.

## Current default: all six tabs

`published/project` is **114,541,240 bytes** (about 109 MiB). It preserves the
preceding three-tab package's frame checksums and adds roughly 3.3 MB for feature,
tuning and final-test presentation data. Deployment uses no source/result CSVs,
research catalog, persisted estimators, fitting or portfolio replay. Final-window
compounding/scoring is prepared once at export from existing monthly portfolios.
It does not rerun the underlying strategies.

```bash
python3 publish_results.py \
  --run 20261109T202421_184e95372ab2 \
  --momentum-study momentum_20261108T211612_b4723b7c \
  --basic-run 20261109T185402_0722a7ec725c 20261109T185621_fe4017faa327 \
              20261109T185517_1bf238ea2712 20261109T185736_3b19b90c1122 \
  --basket-study artifacts/basket_studies/baskets_20261110T021017_e6fa43e2 \
  --feature-study features_20261109T202318_8b01fe74 \
  --feature-book-study artifacts/basket_feature_scores/features_20261110T022741_e6f5661f \
  --tuning-study artifacts/tuning_studies/holdout_20261109T202045_b0a51718 \
      artifacts/tuning_studies/holdout_20261110T012820_d20ce131 \
      artifacts/tuning_studies/holdout_20261110T013013_6f2fd4f9 \
  --final-study artifacts/final_books/final_20261110T025420_6d3405bd \
  --output published/project
```

Choose a new destination for later releases. Any feature/tuning/final flag makes
`published/project` the default output. Existing exports are never overwritten.
Optional schema-1 sections `features`, `tuning` and `final` are handled by
`published_analysis.py`; old one-/two-/three-tab publications remain readable.

Feature export contains all four recipes and classifiers, the actual selection
audits/correlations/calibration/fit summaries, registry portfolio curves and the
advanced-book feature comparison. It embeds canonical method notes, so a hosted
reader need not find the notebook/report directories. The advanced models are
included: recipe-specific up/down classifiers are applied to full-group and
top-ten momentum portfolios. Feature selection does not regroup K-means or change
the volatility/momentum/cash rules. Detailed input rankings below the book results
describe downside fits; upside uses its own fitted recipe and can select different
names. Feature-bar hover shows financial meaning, category, importance, tree SD
and selection; the first portfolio graph also explains each recipe/reference.

Tuning export preserves all three training-holdout studies, four classifier types,
both events, trial scores, chosen settings and frozen 2007–2009 portfolio curves.
It excludes serialized estimators. Header definitions for Brier/log loss/ROC AUC
remain. These studies tune basic rules, not the advanced grouped/top-ten books.

Final export preserves the existing frozen 2010–2018 study and precomputes its
test-only (5,292 portfolio-month rows), training-only (2,940) and entire-path
(8,232) tables/curves with the existing `window_books` helper. Joined curves
clearly distinguish expanding all-20 training forecasts from the frozen ANOVA-10
test fit; no single frozen fit is claimed across the whole 2005–2018 path.
The dotted boundary marks the test start. Earlier saved non-ML report comparisons
are exported as optional Parquet tables. Their CSV files may be read during
explicit export, but the deployed viewer reads none. Raw stock data is never
needed for export. The final study uses factory settings and ANOVA-10, not the
Tuned Model holdout winner; the formal submission/report pin remains separate.

### Default presentation focus

“Four main models” was interpreted as **four portfolio families**: equal weight,
top-ten positive-vol momentum, its directional adjustments, and grouped-basket
directional adjustments (including original references). The four classifier
algorithms remain available. A checkbox at the top of Feature selection, Tuned
Model and Final test is checked by default. Turn it off to enable the wider
registry strategy and every-model/horizon comparisons. Foundation tabs keep their
teaching examples. These controls affect presentation only.

Feature masks/diagnostics remain accessible in focused mode. If a small research
library has no advanced feature-book score, its registry evidence remains
reviewable rather than inventing advanced results. The tuning focus honestly
offers **equal weight only**, because the other main families have no saved tuning
search; switching it off reveals fourteen basic rules. Final historical mixtures
are opt-in after switching focus off. No missing tuned advanced result is inferred
from the feature study or frozen final test.

Verification and read-only/period guard checks:
[six-tab deployment evidence](verification/project_deploy_20261109.json).
No online publication, cloud-resource guarantee, model rerun or formal Quarto/
notebook/slide release update is part of this export step.

## Earlier Advanced ML three-tab publication

`published/advanced_ml` contains the same first-two-tab evidence plus the lead
saved basket study `baskets_20261110T021017_e6fa43e2`. Size: **111,258,117 bytes**.
The original `published/strategies` and `published/basic_ml` remain intact and
can be selected with `--published-root`.

```bash
python3 publish_results.py \
  --run 20261109T202421_184e95372ab2 \
  --momentum-study momentum_20261108T211612_b4723b7c \
  --basic-run 20261109T185402_0722a7ec725c 20261109T185621_fe4017faa327 \
              20261109T185517_1bf238ea2712 20261109T185736_3b19b90c1122 \
  --basket-study artifacts/basket_studies/baskets_20261110T021017_e6fa43e2 \
  --output published/advanced_ml
```

Export to a new directory for subsequent releases. `--basket-study` chooses
`published/advanced_ml` as the default output; without it the older defaults
below still apply. This export only reads completed artifacts; it reads no raw
CSV, refits no cluster/classifier and replays no portfolios.

Optional schema-1 section `tabs.advanced_ml` records source study ID/metadata
checksum, the original metadata and summary frame paths. Its `details` maps split
membership/basket frames by momentum window, model and portfolio version.
`PublishedBasketStudy` validates training scope and supplies a read-only detail
loader to the shared basket renderer. Summary tables/curves do not load every
allocation partition. The grouping illustration loads the saved original full-group
one-month partition; inspection loads only selected versions/models/horizons.
Actual probability/weight audits are preserved, but raw factor/source data and
fitted objects are not included. The original saved fit audit is exported for
report reuse without being loaded by default.

Advanced ML has two subtabs: build/explain the grouped strategy, then add
directional forecasts. The saved membership/path illustration, positive-only
cash math, top-ten screen, model controls and coverage limits are described in
[the basket guide](BASKET_MOMENTUM.md). Missing study sections on older publications
remain honest pending states. All first-two-tab frames stay unchanged.

Verification: [advanced refactor evidence](verification/advanced_refactor_20261109.json).

## Basic ML publication and collapsible sections (preserved two-tab release)

The earlier package is `published/basic_ml`, **56,589,983 bytes** (about 54 MiB).
It retains the same first-tab evidence and adds four matched Basic ML runs:

| Target / scope | Saved run | Published classifiers |
| --- | --- | --- |
| Downside / pooled | `20261109T185402_0722a7ec725c` | Logistic, Gaussian NB, random forest, histogram boosting |
| Upside / pooled | `20261109T185621_fe4017faa327` | Same four classifiers |
| Downside / per stock | `20261109T185517_1bf238ea2712` | Logistic only, with recorded pooled fallbacks |
| Upside / per stock | `20261109T185736_3b19b90c1122` | Logistic only, with recorded pooled fallbacks |

All use the same 14 basic rules, all 20 original factors, January 2005–December
2009 training validation, 3-month cash momentum, annual expanding fits, strength
0.5 and upside threshold 0.6. There are 196 portfolio cases and 11,760 earned-month
rows across these four source runs. No feature selection, tuning, test returns,
new fits or backtest replays are introduced by publishing them.

```bash
python3 publish_results.py \
  --run 20261109T202421_184e95372ab2 \
  --momentum-study momentum_20261108T211612_b4723b7c \
  --basic-run 20261109T185402_0722a7ec725c 20261109T185621_fe4017faa327 \
              20261109T185517_1bf238ea2712 20261109T185736_3b19b90c1122 \
  --output published/basic_ml
```

For subsequent exports choose a new output directory. With `--basic-run`, the
default output is `published/basic_ml`; without it, `published/strategies`.
The earlier first-tab publication stays intact and can be opened with
`--published-root published/strategies`; Basic ML then shows an honest pending
state. Missing combinations never reuse a research-mode browser result.

In both modes Basic ML now has three separate, default-collapsed panels:

1. Does the model predict downside / an up move?
2. What changes in the portfolios?
3. Change from each original portfolio.

The comparison's family/model filters apply to the matching change tables. All
strategies remain selected by default for its performance chart. Streamlit 1.32
prohibits nested expanders, so family groups inside the second panel use bordered
containers; settings/fit details and one-strategy change inspection use optional
checkboxes inside their panel. Other tabs retain their existing grouping.
Collapsing a panel controls presentation, not model execution; browsing never fits
and summary charts are prepared from saved data even while a panel is collapsed.

The optional `tabs.basic_ml.runs` section of schema version 1 records each source
run/config/metadata checksum and paths to cases, metrics, monthly portfolios,
paired changes, per-model probability scores/calibration, fit audit tables and
per-case target weights. Pooled/per-stock prediction-source counts are aggregated
where recorded (legacy pooled runs do not have that column). Raw stock-month
prediction panels, source data and fitted estimators are omitted. Identical
Parquet files, especially shared original weights, are stored once by checksum.

`PublishedBasicStore` is a read-only adapter over this manifest, not a new
database or runner. Its list/config/load/summary methods let the existing saved
chooser and provenance-matched scope comparison work without SQLite or source
fingerprints. Summary batches contain scores/calibration, with no fabricated
stock-level predictions. Fit tables load only when requested. Allocation files
load for the chosen strategy/models; charts reuse the normal renderer and
rebalance inspection shows actual original/adjusted weights. Individual stock
probabilities, full signals and complete report ZIPs remain in research mode.
The model-specific momentum-lookback study remains local; its original-only
comparison is already available in the published Strategies tab.

Verification: [Basic ML deployment evidence](verification/basic_ml_deploy_20261109.json).
The standalone check copies code and only this publication into an empty
directory, with no raw CSVs/catalog/archive, and exercises all four combinations,
fit details and cash allocations under failing raw-read/fitting/replay guards.

## Original Strategies-only publication

The publication included in the workspace is **4,504,572 bytes** (about 4.3 MiB): fourteen
non-ML strategy results and 840 earned-month rows, January 2005–December 2009;
fourteen complete target-weight histories; and original-only 2/3/6/12-month
cash-momentum comparison tables and curves. It has no source CSVs, individual
stock returns/factors, prediction panels, fitted models, SQLite catalog or full
research archive.

`manifest.json` is schema version 1. `tabs.strategies` records:

- Source run ID, original artifact metadata checksum and original run manifest,
  including source/code fingerprints and recorded configuration. These describe
  the source experiment, which can contain hidden ML cases; they are not rewritten
  to pretend the source computation was baseline-only.
- Paths for baseline cases, portfolio metrics and monthly chart data, plus
  separate per-case allocation files. Each Parquet file has a SHA-256 checksum.
- Recorded strategy explanations, with the actual cash-momentum window.
- Optional momentum-study ID, original metadata, original-only summaries and
  performance comparison table. Those are distinct source runs with their own
  recorded timing and provenance.

The reader initially loads only the three main summary tables (101,419 bytes of
compressed Parquet) and the manifest. The momentum section additionally loads
its small comparison/curve tables. Allocation files are verified and read only
when the user opts into allocation exploration and selects that strategy.
Target-weight heatmaps, composition, asset lines and CSV downloads reuse the
existing renderers. Published rebalance inspection contains stock IDs and target
weights; full signal/frozen-pool audit fields remain in research mode. The main
publication uses 3-month cash momentum. The optional lookback chart compares four
windows but does not publish allocation histories for the three other source
runs; the page explicitly identifies that distinction.

## Training boundary and tooltips

The Strategies page prominently says **Training data only** in both modes, then
shows the actual saved evaluation window. A shared guard checks training-validation
configuration, train-only source splits, absence of test source fingerprints,
non-ML cases, 2000–2009 earned dates, recorded date bounds and snapshot-to-next-month
alignment. Local research and exports use the same check. No test-period series
can silently enter the first tab.

Strategy names in the first results table have per-row hover and keyboard-focus
explanations, including selection, weighting, history and fallback. Grouped mode
uses one table per family; Full tables uses one comparison table. Streamlit 1.32
canvas dataframes cannot display cell-specific help, so these small non-ML tables
use escaped HTML and the existing shared metric formatter. The same numbers and
formats are retained. ML tables keep their existing dataframe behavior. HTML
tables scroll horizontally and provide hover/focus help; dataframe sorting is
not part of this small presentation table.

## Remaining delivery work

1. Review presentation/data scope and profile cloud resource use.
2. Prepare the curated GitHub/hosting package and deploy when authorized.
3. Integrate the selected evidence and final choices into the formal notebook,
   Quarto/HTML and slides; keep that submission release distinct from hosting.
4. Advanced-family tuning remains separate future research, not a hosting task.

Keep a versioned manifest and explicit case references; add each tab's reader
and capability only when its self-contained files exist. Keep common table/chart
preparation available to notebooks and Quarto. Do not duplicate research math,
invent another runner or build a separate hosted UI. The public release is
separate from the formal `submission/release.json` pin.

For each tab, verify exact saved-data parity, no source CSV/catalog access,
no fitting/replay, honest missing-result behavior and browser presentation.
Verification for this first step: [evidence](verification/strategies_deploy_20261109.json).

## Later hosting package

Include the app/shared Python modules, their legacy Python dependencies, pinned
Python requirements, theme and selected `published/` directory. Exclude raw
datasets, research artifacts, caches and unrelated archives from the deployment
package. The standalone check copies only those Python sources and published
results into an empty temporary directory; both allocations and the optional
momentum section work there. Community Cloud configuration, GitHub upload and
public deployment have not been performed. Resource use on that service remains
to be measured.
