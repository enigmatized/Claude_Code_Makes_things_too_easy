# Graded notebook plan

Rough-draft plan for the 175-point analysis notebook. Written 2026-11-10.
The draft notebook is `context/analysis_rough_draft.ipynb`. It calls
`research/notebook_views.py`, which reads the newest saved study of each
kind. Books to lead with are `STORY`, `DISPLAY_MODEL`, and `CHOICE` at the
top of that module. Re-executed 2026-11-10 with annual log growth beside
CAGR on the training window, the official test, and the combined path, and
with all four factory classifiers as rows. K nearest neighbors is an extra
model type from `artifacts/grouping_studies/grouping_20261110T044153_b86c80c0`
(training holdout only). This plan does not touch the deploy package,
`submission/analysis.ipynb`, or `03_mlfactor_long_only_project.ipynb`.

## Where the draft will live

| File | Role |
| --- | --- |
| `03_mlfactor_long_only_project.ipynb` | The five prompt cells. Leave them. |
| `submission/analysis.ipynb` | The file `build_submission.py` executes. Replace it only after this draft is accepted. |
| `context/analysis_rough_draft.ipynb` | The rough copy, when it is written. Not part of the Streamlit Cloud repo. |

The Quarto report (`submission/project_report.qmd`, 25 points) is a later pass.
It should call the same display functions, fold the code, and lead with the
decision. Do not rewrite it while the deploy session is in `submission/`.

## What the cloud repo is allowed to hold

Streamlit Community Cloud gets the app entry point and the small published
chart frames the other session is exporting. It does not get:

- `mlfactor_long_only_train.csv` or `mlfactor_long_only_test.csv`
- `artifacts/` (fitted models, prediction panels, the SQLite catalog)
- `*.joblib` pipelines

The notebook stays on this machine. It reads saved portfolio months and
metric tables. Those frames are book-level (one row per strategy per month),
not the stock panel. Execute the notebook once locally so the uploaded file
already contains the tables and figures. A grader who only opens the file
can read the results. A grader who reruns it needs those saved frames, not
the course CSVs and not the fitted models.

## Shape of every code cell

One comment, then one or two calls, then a display. The comment says which
saved file is read, which months, and why that output is the graded object.
Training loops, `GridSearchCV`, and `load_panel("test")` do not appear in
the notebook. `MLFACTOR_READ_ONLY=1` is set in the setup cell.

Display helpers belong in a new read-only module, added when the draft is
written. Suggested name: `research/notebook_views.py`. It returns a small
dataframe or a matplotlib figure. The notebook does not reshape data.

## The 175 points

### Equal-weight annualized return — 30

One cell. Test window only: January 2010–December 2018, 108 months.

The number is CAGR **13.096%**, volatility 15.63%, Sharpe 0.868, max
drawdown −20.64%. It is already in
`part0_data_and_benchmark/outputs/equal_weight_metrics.csv` and it matched
the frozen four-book pass. `benchmark_table()` is the existing reader.
Show the test row. The cell comment states the identity: each month is the
cross-sectional mean of simple `R1M_Usd`, then
`((1 + r).prod()) ** (12 / n) - 1`.

Do not put the training-file CAGR (about 14.5% over 120 months) or the
2005–2009 CAGR (7.62%) in this cell. Those are different windows.

### Accuracy of results — 70

The prompt lists six tasks. Each one needs a labeled table and one sentence
that answers the task. Mixing windows without a label is the usual way to
lose this.

1. **Ten or fewer features, and whether the cut helps.** Training rescore
   `artifacts/basket_feature_scores/features_20261110T022741_e6f5661f`.
   All 20 versus ANOVA-10, pruned-10, and forest-10. Logistic, one month:
   CAGR moves by less than 0.2 percentage points. The frozen test used the
   ANOVA list chosen at 2009-12-31 (10 names, recorded on the final-book
   study). Sentence to show: the cut did not improve the book; it meets the
   assignment limit. Describe the method (ANOVA rank at the freeze date).
   A name list can sit under the table. Slides still do not read the names.
2. **Four model types, two of them tunable.** Logistic regression, Gaussian
   naive Bayes, random forest, histogram gradient boosting. Logistic, the
   forest, and boosting have hyperparameters. Show all four on the same
   book and the same months. The official four-book file already has them.
3. **`GridSearchCV` on training data, and whether the portfolio moved.**
   This is the open accuracy hole. The 2010–2018 four-book pass used factory
   classifier settings. The search that does exist is the training holdout
   (fit through December 2006, score January 2007–December 2009) on the
   older rules: logistic chose `C=0.1`, boosting kept its defaults, and the
   books barely moved. The notebook can show that saved search and say, in
   the same cell, that the official four-book numbers are factory settings.
   It cannot say the displayed 2010–2018 books were the searched models.
4. **Two non-ML books, one of them equal weight.** Equal weight, and the
   one-month top-10 positive-volatility momentum book. Full baskets are the
   third line, because that is the comparison that lost. Leave the other
   twelve registry rules out of this notebook.
5. **Risk and reward.** Same rows, same window: CAGR, volatility, Sharpe,
   max drawdown, average cash. Sortino and turnover only if the saved frame
   already has them. Gross of costs. Cash earns nothing.
6. **A final choice, justified from the tables above.** Not pinned yet.
   Proposed sentence, still waiting on a yes:

   > Trade the one-month top-10 positive-volatility book, with no
   > probability overlay. On 2010–2018 it compounds at 19.6% against the
   > equal-weight 13.1%. Volatility is 39% against 16%, and the Sharpe is
   > 0.59 against 0.87. The upside overlay did not change that book. The
   > downside overlay cut the drawdown and also cut the return, with the
   > Sharpe unchanged. Full baskets were calmer and earned less than equal
   > weight.

   Say once that 2010–2018 was scored after the recipe was written down,
   and the recipe was not changed after the numbers were seen.

### Readability — 50

Six sections, in the prompt's order. Each section is a heading, one or two
sentences, one table, and at most one chart. The growth chart is the
official test. A second chart is the 2005–2018 path with a line at January
2010. No section dumps every model and every horizon. "Show every model"
stays in the app.

### Code comments — 25

The 25 points are the short comments in the notebook, not the comments in
`research/`. One comment per cell is enough when it names the file, the
window, and the graded question. A comment that restates the Python does
not help.

## Cells

Setup. Set the project root on `sys.path` and `MLFACTOR_READ_ONLY=1`.
Import the display helpers.

1. Heading: equal-weight test return.
   Call `benchmark_table()` and keep the test row.
2. Heading: the second non-ML book, same 108 months.
   Call a helper that reads
   `artifacts/final_books/final_20261110T025420_6d3405bd`
   and returns equal weight, top 10, and full baskets.
   Chart those three wealth paths.
3. Heading: four classifiers on that top-10 book.
   Same file. One table, logistic / naive Bayes / forest / boosting, for
   the unadjusted book and the two overlays. One sentence: upside logistic
   matches the unadjusted book; boosting upside does move it, and the
   table is where that is visible.
4. Heading: did ten features help?
   Read the feature-score metrics. Label the window January 2005–December
   2009. One sentence on the 0.2 percentage point move.
5. Heading: what `GridSearchCV` searched.
   Read the saved holdout study
   `artifacts/tuning_studies/holdout_20261109T202045_b0a51718`.
   Show the searched settings and the holdout portfolio delta.
   The following sentence says the official test used factory settings.
6. Heading: risk and reward on three windows.
   Official test, training window January 2005–December 2009, and the
   stitched 168-month path. Logistic overlays plus the two non-ML books.
   The path chart marks January 2010. Caption: training overlays are
   expanding all-20 forecasts; test months are the one frozen ANOVA fit.
7. Heading: the choice.
   Markdown only. The sentence above, plus the Sharpe caveat, plus the
   one-line disclosure that the test was already scored.

No cell calls `score_final_books`, `ExperimentRunner.run`, or `KMeans`.

## Numbers the draft must match

Logistic, one-month holding, gross of costs. Already recomputed from the
saved monthly returns.

| Book | 2005–2009 | 2010–2018 | 2005–2018 |
| --- | --- | --- | --- |
| Equal weight | 7.62%, Sharpe 0.43, DD −49.5% | 13.10%, Sharpe 0.87, DD −20.6% | 11.11%, Sharpe 0.66, DD −49.5% |
| Top 10 positive-vol | 24.68%, Sharpe 0.83 | 19.61%, Sharpe 0.59 | 21.40%, Sharpe 0.66 |
| Top 10 + downside | 19.41%, Sharpe 0.83 | 15.64%, Sharpe 0.59 | 16.97%, Sharpe 0.66 |
| Top 10 + upside | 24.98%, Sharpe 0.84 | 19.61%, Sharpe 0.59 | 21.50%, Sharpe 0.67 |
| Full baskets | 10.07%, Sharpe 0.71 | 4.01%, Sharpe 0.53 | 6.14%, Sharpe 0.59 |
| Baskets + downside | 7.92%, Sharpe 0.71 | 3.15%, Sharpe 0.53 | 4.83%, Sharpe 0.59 |

The −49.5% drawdown on the long path is the 2008 drop inside the training
window. The official-test equal-weight drawdown is −20.6%.

## What this plan does not do

- It does not pin `final_choice_case_id` or run `build_submission.py --final`.
- It does not start a new grid search.
- It does not open the official test CSV.
- It does not edit the deploy entry point, `published/`, or the release pin.
