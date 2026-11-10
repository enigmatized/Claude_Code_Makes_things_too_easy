"""Export one completed feature study as compact tables and static figures.

Usage: python3 export_feature_study.py STUDY_ID
The exporter only reads saved evidence; run_feature_selection.py computes it.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from research.feature_studies import load_feature_study, feature_audit
from research.feature_reporting import (importance_table, importance_figure, portfolio_figure,
                                        prediction_summary, selection_overlap, forest_findings)
from research.store import ExperimentStore, digest_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_id")
    args = parser.parse_args()
    store = ExperimentStore()
    study = load_feature_study(store, args.study_id)
    folder = Path(__file__).resolve().parent / "0_3_Feature_Selection/outputs" / args.study_id
    folder.mkdir(parents=True, exist_ok=True)
    tables = {"importance": importance_table(study), "predictive_comparison": prediction_summary(study),
              "portfolio_comparison": study["portfolio_comparison"], "calibration": study["calibration"],
              "selection_audit": feature_audit(study["fits"])}
    for name, frame in tables.items():
        frame.to_csv(folder / f"{name}.csv", index=False)
    selection_overlap(study).to_csv(folder / "selection_overlap.csv")
    figures = {"forest_importance": importance_figure(study),
               "portfolio_wealth": portfolio_figure(study),
               "portfolio_log_returns": portfolio_figure(study, cumulative_log=True)}
    for name, figure in figures.items():
        figure.savefig(folder / f"{name}.svg", bbox_inches="tight")
        figure.savefig(folder / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(figure)
    (folder / "findings.txt").write_text(forest_findings(study) + "\n")
    provenance = {"study_id": args.study_id, "runs": study["metadata"]["runs"],
                  "mode": "saved-results-only", "files": {
                      path.name: digest_file(path) for path in folder.iterdir() if path.is_file() and path.name != "exports.json"}}
    (folder / "exports.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(folder)


if __name__ == "__main__":
    main()
