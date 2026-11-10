# MLFactor long-only project

The app now has explicit **research** and **deploy/view** modes. Start local
research with `python3 run_app.py`; preview all seven published tabs with
`python3 run_app.py --mode deploy --port 8502`. Deploy mode reads a self-contained
172.6 MB results export, with no raw dataset, research catalog or model fitting.
All seven tabs are published. Training comparisons and the frozen official test
are explicitly separated. Both modes clearly label the first tab as training-only,
and strategy names in its results tables have hover/focus explanations.
Basic ML has three collapsed-by-default sections for prediction quality,
portfolio results and changes from the originals. Both up/down patterns,
four pooled classifiers and the per-stock logistic comparison have saved views.
**Grouping With ML** has three subtabs: **K-means**, **Hierarchical clustering**
(Ward, with its actual saved merge tree), and **Adding a direction**. Both methods
feed the same momentum/cash and top-ten positive-volatility rules, then the same
four saved up/down models. See [grouping guide](docs/GROUPING_WITH_ML.md).
Feature selection and Final test default to the four main portfolio families;
turn their focus filter off to review the wider saved comparisons. Tuned model
shows all fourteen saved basic strategies by default, with four classifier and
separate up/down choices.
Feature-importance hovers explain the abbreviated factor names. Advanced
portfolio feature comparisons are included; K-means and trading rules stay fixed.
The grouped/top-ten portfolios have no tuned study yet. Final test explicitly
labels its fixed K-means models and offers CAGR/log-return views and changes
from each original portfolio. See [the tuning/final review](docs/TUNING_AND_FINAL_REVIEW.md).
**Feature engineering** follows Tuned model, with construction explanations,
PCA/IC charts, model/portfolio comparisons and its own tuning subtab. Six recipes
feed all four classifiers; their fixed/tuned frozen final variants are included
in Final test on K-means and Ward starting books. See
[the full notebook review and protocol](docs/FEATURE_ENGINEERING.md).
See [deployment modes and tab-by-tab handoff](docs/DEPLOYMENT.md).

For a new context, start with [AGENTS.md](AGENTS.md) and the
[current project state](docs/PROJECT_STATE.md). The handoff documents must be
updated after each substantive change. See [the research roadmap](docs/RESEARCH_ROADMAP.md)
for feature selection, training-only tuning and the final evaluation sequence.

The app's [stage layout](docs/TAB_REFACTOR_HANDOFF.md) progresses from
training-only Strategies to Basic ML, grouped momentum with directional ML overlays,
Feature selection, the separate Tuned model training holdout and Final test.
See [basket rules and forecast sources](docs/BASKET_MOMENTUM.md) and
[what is actually tuned](docs/TUNING.md) for their separate inputs and periods.
Tuned model starts with model, basic strategy and next-up/down selectors. All
four models have saved comparisons on the same frozen ANOVA-10 training holdout;
hover the probability-score headers for definitions and inspect original versus
chosen settings to understand the tuning result. Browsing never retrains.
Existing charts, diagnostics and grouped
results are reused; case filters and separate training/test state keep each
stage's evidence isolated. The earlier
[shared presentation plan](docs/PROJECT_PRESENTATION_PLAN.md) remains background
for evidence shared with Quarto and slides.

For the local Python app, run `python3 run_app.py` from this directory and open
**http://localhost:8501**. The [strategy runner guide](0_2_Strategy_Runner/README.md)
explains the UI, reusable Python API, CLI, and optional shared ML loss adjustment.
It compares fourteen strategies using shared portfolio accounting, including
frozen training Sharpe/Sortino pools followed by monthly momentum allocation.
The app includes strategy explanations, shared data preparation, configurable
original-factor inputs, and four fixed loss-prediction models. See the
[model comparison and implementation](0_2_Strategy_Runner/04_data_and_model_comparison.md).

The [feature-selection study](0_3_Feature_Selection/README.md) compares all twenty
factors with training-only ANOVA ten, correlation-pruned ten and random-forest
ten, across the same four models and fourteen strategies. Open **Feature selection** in the app for
two subtabs: **Selected features & method** for actual inputs and their selection
process, and **Effect on ML models** for retrained-model scores and ML-adjusted
portfolio comparisons. Non-ML rules remain unchanged reference benchmarks. Saved
findings and a readable executed notebook accompany the study. Forest importance
bars, tree-variation error bars and method-overlap tables are available, and the
same evidence is pinned into the submission notebook, Quarto report and slides.
See [forest findings](0_3_Feature_Selection/05_random_forest_findings.md).

Basic ML also offers an [upside threshold boost and per-stock comparison](0_6_Basic_ML_Directions/README.md).
Choose the prediction pattern and training scope in that tab. The existing
pooled models learn from all stocks; optional per-stock fits use only each
stock's history and record a pooled fallback when there is insufficient data.
The two targets have separate prediction diagnostics and original/adjusted
portfolio comparisons. Matched pooled/per-stock scores are available locally.

The app now loads saved experiments before computing. A local SQLite catalog
indexes full Parquet results, including allocation histories. Read
[saved-result usage and SQL examples](docs/SAVED_RESULTS.md).

The new [annual consistency + momentum + cash strategy](0_4_Consistency_Momentum_Cash/README.md)
requires a majority of positive historical calendar years and buys only strictly
positive momentum. Inactive stock budgets stay in cash. Its optional ML variants
transfer weight reductions into cash. It is the fourteenth registered rule;
the earlier thirteen retain their existing behavior.

The [momentum-window comparison](0_5_Momentum_Lookbacks/README.md) tests two,
three, six and twelve months with the same annual stock pool and four ML models.
Review baseline versions in **Strategies → Momentum lookback**, and the model
versions in **Basic ML**. Each stage has an explicit training-comparison expander
for requesting a new lookback/model configuration.

The loaded run's banner and chart titles state its actual evaluation period.
The **Final test** tab leaves the final frozen-model result pending until a
ready experiment is explicitly pinned. Its optional historical evidence shows
earlier 2010–2018 non-ML results and exploratory saved tests. These selections
cannot change training tabs; the recent studies remain training validation.
See [how the evaluation periods work](docs/EVALUATION_PERIODS.md).

Strategies and Basic ML [group strategies into six families](docs/STRATEGY_GROUPS.md),
with collapsible result sections and one-row-per-strategy ML-change grids.
Filter a family, select a change metric, or switch to full tables. Model details
and CSV exports remain available. The sidebar also supports family selection
inside a compact strategy chooser.

Results show **CAGR** and **Annual log growth** side by side. In Strategies and
Basic ML, use **Return measure** above the charts to switch between growth of
$1, cumulative log return and monthly log return. These views read the saved
portfolio results. See [return definitions and accounting](docs/RETURN_METRICS.md).

For deliverables, run `python3 build_submission.py` to build the readable notebook,
Quarto report and draft slides from a deliberately pinned saved experiment.
See [submission requirements](docs/SUBMISSION.md) and the
[submission workspace](submission/README.md). These are research drafts until
final feature choice, tuning integration and official test analysis are complete.
The [UI framework comparison](docs/UI_OPTIONS.md) recommends a later NiceGUI prototype.

Start with [Part 0: data and benchmark](part0_data_and_benchmark/00_readme.md).
It explains the stock-month panel, provides training-data charts, and independently
checks the monthly equal-weight portfolio before any modeling.

Next, explore [basic non-ML strategies](0_1_Basic_Strategies/00_readme.md): high
and low volatility, positive momentum, and the effect of weighting momentum
stocks differently. The comparison uses Part 0's shared accounting.

The [original assignment](03_mlfactor_long_only_project.ipynb) is the source of the
requirements. [Part 1](part1_feature_selection/00_readme.md) is preserved as the
earlier feature-selection experiment. Return to it after the data and benchmark
are familiar; its complexity is not a prerequisite for using Part 0.

The two source CSVs stay in this directory. Each experiment writes generated
calculations and figures beneath its own `outputs/` folder.
