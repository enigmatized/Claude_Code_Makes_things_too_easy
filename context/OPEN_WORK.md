# Open work for the finish

Agreed order, as of 2026-11-09:

1. Feature selection. This is the current piece of work.
2. A strategy that groups correlated stocks and weights by those groups.
   The Advanced ML tab now has that book: monthly K-means baskets, positive
   momentum inside the group, and cash otherwise. See
   [../docs/BASKET_MOMENTUM.md](../docs/BASKET_MOMENTUM.md).
3. Hyperparameter tuning. The Tuned model tab has one saved training
   holdout: ANOVA-10 frozen at December 2006, scored on January 2007–December
   2009. See [../docs/TUNING.md](../docs/TUNING.md).
4. One frozen test of the chosen spec on 2010–2018, after the training choice
   is written down. That file has not been used for the tuning comparison or
   the basket comparison.

Do the science before rewriting the talk. Do not start by refactoring tabs.
The app tabs are already implemented. Advanced ML reads the saved basket study.

Re-read `docs/PROJECT_STATE.md` first. If the other session already added
GridSearchCV, a frozen test, or a new release pin, skip the step it finished.

## 1. Choose the small story, on training evidence only

Decide, and write the decision down before looking for a better test result:

- The second non-ML strategy. The intended new rule groups correlated stocks
  and assigns weights through those groups. Existing rules stay as comparisons.
- The feature recipe the final model will use (a real ≤10 set, or a selector
  that is fit once at the frozen origin and then stays ≤10).
- Which two models get a small `GridSearchCV`, and the grid, written down
  before the search runs.

Training evidence to use: January 2005–December 2009 validation, not the test
CSV. Part 1 and `0_3_Feature_Selection/02_findings.md` both say cutting
features does not clearly help. That is an acceptable conclusion if the
notebook shows the comparison. The assignment asks for the analysis, not for
a subset that wins.

A reasonable featured non-ML rule is one the app can already explain in a
sentence: momentum-weighted, frozen high-vol plus momentum, or the 3-month
cash rule. Pick one. The other thirteen can stay in the app.

## 2. GridSearchCV, training folds only

The primitive is `research/tuning.py`. Do not run it until the feature recipe
and the grouped-stock strategy are chosen. When you do, follow
[../docs/TUNING.md](../docs/TUNING.md): one full training slice, test CSV
closed, logistic plus histogram boosting. The rules below are already what
that function enforces.

- Month-group folds. Every stock from a month stays in the same fold.
- Inner validation labels have to be known at that origin. No random row
  split and no default stratified CV.
- Search objective is a probability score (negative log loss is the plan in
  `docs/RESEARCH_ROADMAP.md`). Portfolio results are a separate table.
- At least two models. The fast declared pair is logistic `C` and histogram
  boosting's learning rate and leaf count. Random forest is in the same file
  but refuses a 100-tree search unless you explicitly allow the wait or fix
  a smaller tree count. The grids are already written in `DECLARED_GRIDS`.
- Save candidates, fold scores, chosen parameters, and runtime. Save full
  books only for the chosen settings, not for every trial.
- Compare tuned versus the current fixed settings on the same outer months
  and the same trading rule.

## 3. Freeze a spec and run the official test once

`fit_policy="frozen_at_origin"`, `evaluation="test"`. Fit on training data
known at the December 2009 snapshot. Apply that fit to the test snapshots.
Lagged non-ML signals may update under the existing month-end rule. Say in
the report that 2010–2018 was already inspected.

Do not iterate on the test. If the book looks bad, that is the result.

Then set `submission/release.json` on purpose (`build_submission.py --pin-run`
only after the run is the one we mean to publish). Set
`final_choice_case_id`. Keep the part-0 equal-weight CSV hash unless that
file is regenerated on purpose.

## 4. Thin notebook, report, slides

Follow [DELIVERABLES.md](DELIVERABLES.md).

- Extend `research/reporting.py` so the notebook can show the benchmark,
  the tuning table, the feature comparison, and the frozen test with function
  calls.
- Execute the notebook under the read-only build.
- Rewrite `project_report.qmd` and `presentation.qmd` so they state the
  conclusion first and match the pin. The current deck is a short draft and
  is not the 15-minute presentation.
- `python3 build_submission.py --final` should pass.
- Practice the app on that same pin: Strategies (two rules), Basic ML
  (scores versus the book), Feature selection, Final test. Skip Advanced.

## Explicitly out of scope unless the user asks

- Filling in an advanced ML strategy.
- Cross-model stacking or ensembles (`docs/RESEARCH_ROADMAP.md` lists this as later).
- A new UI framework. NiceGUI was a later idea; Streamlit stays.
- Rebuilding Part 0 or Part 1.
- Flashcards. Graded hand-in comes first.
- Recomputing every saved run. Old artifacts stay as historical evidence.

## Checks before calling it done

- Readiness dict in the notebook is all true, from the pinned run, not from memory.
- Test equal-weight CAGR on the slide matches
  `part0_data_and_benchmark/outputs/equal_weight_metrics.csv` (13.096% over
  108 test months) unless that file was intentionally rebuilt.
- Report HTML, slide HTML, and the app are opened and clicked, not only rendered.
- No slide contains a code chunk. Report code is folded.
- `docs/PROJECT_STATE.md` and `CHANGELOG.md` get a dated note after the run,
  per `AGENTS.md`, once this session is the one allowed to edit them.
