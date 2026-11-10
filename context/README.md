# Context for the finish

Snapshot taken 2026-11-09 for the next working session. Another LLM is also in
this project. This folder is a reading pack. It does not replace the living
handoff.

## Read in this order

1. This file.
2. [WHERE_WE_ARE.md](WHERE_WE_ARE.md) — what already runs, and what the grade still lacks.
3. [DELIVERABLES.md](DELIVERABLES.md) — app demo, slides, Quarto, and the thin notebook.
4. [OPEN_WORK.md](OPEN_WORK.md) — the remaining science and packaging, in order.
5. [NOTEBOOK_PLAN.md](NOTEBOOK_PLAN.md) — cell plan for the graded notebook. Draft: `context/analysis_rough_draft.ipynb`.

Then open the living handoff, which may be newer than this snapshot:

- [../AGENTS.md](../AGENTS.md)
- [../docs/PROJECT_STATE.md](../docs/PROJECT_STATE.md)
- [../docs/SUBMISSION.md](../docs/SUBMISSION.md)
- [../docs/TAB_REFACTOR_HANDOFF.md](../docs/TAB_REFACTOR_HANDOFF.md)

If `docs/PROJECT_STATE.md` or `CHANGELOG.md` is newer than these files, trust
those for run IDs, test counts, and what just changed. Re-read
`submission/release.json` and `submission/build_status.json` before editing
the report.

## Do not collide

Do not edit these while the other session may still be writing them:

- `research/`, `app.py`, `app_charts.py`, `*_ui.py`, `result_views.py`, `strategy_groups.py`
- `docs/PROJECT_STATE.md`, `docs/DECISIONS.md`, `CHANGELOG.md`
- `artifacts/`, `submission/release.json`, source CSVs
- experiment notebooks under `0_*` and `part*`

Safe to add: files in this `context/` folder, and later a clearly named
submission notebook once the other session has stopped.

Before any dev work, check file modification times and whether a Python,
Streamlit, or Quarto process is running in this directory.

## What this session is for

Finish the assignment in
[../03_mlfactor_long_only_project.ipynb](../03_mlfactor_long_only_project.ipynb)
well enough to present and turn in. The research app, saved runs, and draft
Quarto pipeline already exist. The missing work is the graded evidence
(a real ≤10-feature final spec, GridSearchCV, a frozen test, a written
choice) and the presentation package around that evidence.
