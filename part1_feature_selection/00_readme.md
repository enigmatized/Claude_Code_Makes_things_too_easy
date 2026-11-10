# Part 1 — feature subset selection

Read these files in filename order.

| Order | File | What it is |
| --- | --- | --- |
| 00 | `00_readme.md` | This map. |
| 01 | `01_plan.md` | Method locked before the notebook is built. Decisions, formulas, and the reason for each step. |
| 02 | `02_part1_feature_selection.ipynb` | Implementation of `01_plan.md`. Run top to bottom. |
| 03 | `03_learnings_from_implementation.md` | What the executed notebook produced. Filled only from that run. |

A later review reads `01` and `03` together. `01` says what was supposed to happen. `03` says what the executed notebook showed, including results that disagreed with the plan.

Rules for anyone trying to beat this part:

- The submitted feature list has at most 10 of the original factor columns.
- Training rows are the only rows that may choose those columns.
- The test file is for one backtest of a frozen list. Seeing a test result and then editing the list invalidates the comparison.
- `03` is filled from notebook outputs. It is not a place to invent a better subset.
