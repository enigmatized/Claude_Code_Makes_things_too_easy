# What gets presented and turned in

The user will present this project. Four surfaces, one set of numbers.

| Surface | Role | Current file |
| --- | --- | --- |
| The app | Live presentation. Walk the five tabs on a **pinned** saved run. | `python3 run_app.py` → http://localhost:8501 |
| Slides | Graded presentation, about 15 minutes, 100 points. | `submission/presentation.qmd` → `submission/_output/presentation.html` |
| Quarto report | Graded written report, 25 points. HTML with folded code. | `submission/project_report.qmd` → `submission/_output/project_report.html` |
| Notebook | Graded coding/analysis, 175 points. | Prompt is `03_mlfactor_long_only_project.ipynb`. Draft analysis is `submission/analysis.ipynb`. |

The professor's deck rules: the room already knows the data, so do not teach
the columns. Describe the selection **analysis**, do not read out the ten
names. No code on slides. Say what each chart shows. Content is graded, not
only polish. Expect interruptions.

The app is the user's presentation tool. It is not a separate graded file.
Use it to show the same pinned results the slides use. Do not live-fit during
the talk.

## Notebook shape the user asked for

The turn-in `.ipynb` calls functions that live in `.py` files. A cell is a
few calls, then displayed output. A reader should get the result without
scrolling through training code.

`submission/analysis.ipynb` is already close:

- `research.reporting.load_release`, `benchmark_table`, `predictive_table`,
  `portfolio_table`, `portfolio_figure`, `readiness`
- a `RunConfig` cell that stays off unless `REBUILD_EXPERIMENT` is set
- builds set `MLFACTOR_READ_ONLY=1` and refuse to fit

That notebook is a draft about one old expanding training run. It still says
feature selection, tuning, and the final choice are pending. The finished
notebook should keep this shape and call report functions that read the
pinned artifacts for each graded section:

1. Equal-weight test annualized return (the 30-point cell).
2. The second non-ML strategy on the same months.
3. Four models, with the GridSearchCV table for the two that were tuned.
4. All-features versus the ≤10 subset, same model and same trading rule.
5. Risk and reward for those books.
6. The frozen test of the chosen spec, and one short markdown justification.

Keep the original five prompt cells in `03_mlfactor_long_only_project.ipynb`.
Put the callable notebook either below those cells or in
`submission/analysis.ipynb`, and say in the first markdown cell which file is
the turn-in. Do not delete the prompt.

Comments that earn the 25 comment points belong next to the calls: what the
function reads, which window, and why the output is the graded object. The
long comments stay in `research/`.

## Report and slides

`python3 build_submission.py` from the project root:

- reads `submission/release.json` (do not silently pin the newest run)
- executes the analysis notebook
- renders the QMD report and reveal.js slides
- does not refit and does not search hyperparameters

`--final` must keep failing until the readiness checks in
`research/reporting.py` are all true:

- declared inputs ≤ 10
- four model types
- GridSearchCV evidence for at least two models
- two non-ML books including `equal_weight_universe`
- CAGR, Sharpe, Sortino, max drawdown
- `evaluation == "test"` and `fit_policy == "frozen_at_origin"`
- `final_choice_case_id` points at a real case

Current `submission/build_status.json` (draft pin
`20261108T144257_f45441ba8e83`): four models, two non-ML controls, and risk
metrics pass. The feature limit, GridSearchCV, frozen test, and final choice
fail.

Quarto source stays next to the HTML. Code cells in the report stay folded.
Slide cells stay hidden (`echo: false` is already set). Figures in the report,
the slides, the notebook, and the app have to be the same pinned run.

## Talk outline that matches the professor and the app

About one minute each unless noted. The app is the picture; the slide is the
sentence.

1. Question and the test equal-weight number. Do not tour the columns.
2. Second non-ML rule: the rule in one sentence, then return, vol, drawdown.
3. What the loss model is asked to predict, and that probability scores barely
   beat a historical loss rate.
4. Whether giving those probabilities to the weights changed the book.
5. Feature cut: what the comparison did, and whether the book moved. Two minutes.
6. What GridSearchCV searched, and whether the portfolio moved with the ML score.
7. The chosen spec on 2010–2018, frozen fit, risk and reward, and the limit
   that this test was already inspected.

Leave the full fourteen-strategy grid and the factor-name audit in the report
appendix and in the app, not on the timed slides.
