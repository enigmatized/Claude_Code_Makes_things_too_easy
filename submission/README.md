# Submission workspace

This is a reproducible draft pipeline, not a completed final submission.

- `analysis.ipynb`: readable coding/analysis notebook using the shared library and saved evidence.
- `project_report.qmd`: professional report source; code folded in its HTML output.
- `presentation.qmd`: draft slide source; code hidden on slides.
- `release.json`: deliberately pinned main experiment, optional checksummed feature study and report focus.
- `_output/project_report.html` and `_output/presentation.html`: rendered outputs.
- `build_status.json`: the pinned run and outstanding final-analysis evidence.

From the project root:

```bash
python3 build_submission.py
```

The build reads a complete saved experiment, executes the small analysis notebook,
and renders Quarto. It never refits models or runs a parameter search. Quarto is
installed on this machine. The full Python environment is listed in the root
`requirements.txt`; the app-only subset is in `requirements-app.txt`. The Quarto
CLI is installed separately. Builds enforce `MLFACTOR_READ_ONLY=1` and verify
the release pin stayed unchanged while rendering.

To change the report's experiment explicitly:

```bash
python3 -m research --saved
python3 build_submission.py --pin-run RUN_ID
```

Do not automatically use the newest run: that would change a report silently.
The release also pins the independent benchmark CSV's hash; update that reference
deliberately if its source experiment is regenerated.
`--final` rejects a build whose pinned run lacks recorded final evidence. At
present, the completed four-recipe forest study is pinned and rendered alongside
the older main experiment, with independent scopes stated. Final feature/model
choice, integration of the separate tuning holdout and the official frozen test
result remain pending.

To pin feature evidence explicitly, use
`python3 build_submission.py --pin-feature-study STUDY_ID`. The optional release
reference verifies the study metadata checksum and its constituent saved runs.
Old releases without this field remain readable. Shared read-only helpers in
`research/feature_reporting.py` supply the app, review notebook, submission
notebook, report and slides; changing chart controls never refits models.
The earlier sources and HTML are preserved under `archives/pre_forest_20261109/`.

The exact LMS upload packaging is not stated in the assignment notebook. Plan to
provide readable source/notebooks, the QMD sources and rendered HTML documents,
plus the final presentation. Keep report/slides figures tied to the same saved
experiments. Revise the draft slides to about 15 minutes after the remaining
analysis; do not present the current short research draft as the final deck.
