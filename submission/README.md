# Submission workspace

The notebook, report, and slides read the saved four-book studies and the feature-engineering holdout. `build_submission.py --final` checks the evidence the notebook reads (`research/submission_checks.py`) and passes. `release.json` still pins the older catalog run; its checks are kept in `build_status.json` as informational.

- `analysis.ipynb`: readable coding/analysis notebook using the shared library and saved evidence.
- `project_report.qmd`: written report; same section numbers as `analysis.ipynb`, code folded in its HTML output.
- `presentation.qmd`: 16-slide deck for the 15-minute talk; no code, a "how to read it" line under each chart, speaker notes (press `S`).
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
`--final` rejects a build whose pinned catalog run lacks recorded final evidence.
The narrative in the notebook and the Quarto files is ahead of that pin: it reads
the saved four-book pass, the training GridSearchCV holdout, and the engineering
holdout directly. Do not point `--final` at a run until `release.json` records
that choice.

To pin feature evidence explicitly, use
`python3 build_submission.py --pin-feature-study STUDY_ID`. The optional release
reference verifies the study metadata checksum and its constituent saved runs.
Old releases without this field remain readable. Shared read-only helpers in
`research/feature_reporting.py` supply the app, review notebook, submission
notebook, report and slides; changing chart controls never refits models.
The earlier sources and HTML are preserved under `archives/pre_forest_20261109/`.

The exact LMS upload packaging is not stated in the assignment notebook. Upload
this folder's `analysis.ipynb`, the QMD sources, and the rendered HTML documents.
Annual log growth is the lead return. CAGR stays beside it. The feature-engineering
section uses the completed 2007–2009 study and the previously viewed comparison
in `0_7_Feature_Engineering`.
