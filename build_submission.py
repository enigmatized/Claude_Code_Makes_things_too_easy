"""Build draft notebook, Quarto report and slides from a pinned saved experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import nbformat
from nbclient import NotebookClient

from research.reporting import load_release, readiness, feature_study_reference
from research.store import ExperimentStore


ROOT = Path(__file__).resolve().parent
SUBMISSION = ROOT / "submission"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin-run", help="Explicitly select a saved run for future renders")
    parser.add_argument("--pin-feature-study", help="Explicitly select a checksummed saved feature study")
    parser.add_argument("--final", action="store_true", help="Require recorded final-analysis evidence before rendering")
    args = parser.parse_args()
    release_path = SUBMISSION / "release.json"
    release = json.loads(release_path.read_text())
    store = ExperimentStore((SUBMISSION / release["artifact_root"]).resolve())
    if args.pin_run:
        store.summary(args.pin_run)  # Validate existence and integrity before changing the pin.
        release["run_id"] = args.pin_run
    if args.pin_feature_study:
        release["feature_study"] = feature_study_reference(store, args.pin_feature_study)
    if args.pin_run or args.pin_feature_study:
        release_path.write_text(json.dumps(release, indent=2) + "\n")
    summary = load_release(release_path)
    release_hash = hashlib.sha256(release_path.read_bytes()).hexdigest()
    checks = readiness(summary)
    if args.final and not all(checks.values()):
        missing = "\n".join(f"- {key}" for key, passed in checks.items() if not passed)
        raise SystemExit("Final submission evidence is incomplete:\n" + missing)
    quarto = shutil.which("quarto")
    if not quarto:
        raise SystemExit("Install the Quarto CLI before building the report.")
    print(f"Reading saved experiment {summary['run_id']}; no fitting or search", flush=True)
    notebook_path = SUBMISSION / "analysis.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    previous = os.environ.get("MLFACTOR_READ_ONLY")
    os.environ["MLFACTOR_READ_ONLY"] = "1"
    try:
        NotebookClient(notebook, timeout=120, kernel_name="python3",
                       resources={"metadata": {"path": str(SUBMISSION)}}).execute()
        nbformat.write(notebook, notebook_path)
        environment = dict(os.environ, QUARTO_PYTHON=sys.executable)
        subprocess.run([quarto, "render", str(SUBMISSION), "--execute"], cwd=ROOT, env=environment, check=True)
    finally:
        if previous is None:
            os.environ.pop("MLFACTOR_READ_ONLY", None)
        else:
            os.environ["MLFACTOR_READ_ONLY"] = previous
    if hashlib.sha256(release_path.read_bytes()).hexdigest() != release_hash:
        raise SystemExit("Release configuration changed during rendering; rebuild before using the outputs.")
    status = {"run_id": summary["run_id"], "release_sha256": release_hash,
              "feature_study": summary["release"].get("feature_study"),
              "mode": "final" if args.final else "draft", "checks": checks,
              "outputs": ["analysis.ipynb", "_output/project_report.html", "_output/presentation.html"]}
    (SUBMISSION / "build_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print("Built readable notebook, report HTML and slide HTML.", flush=True)


if __name__ == "__main__":
    main()
