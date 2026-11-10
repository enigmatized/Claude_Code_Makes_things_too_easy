"""Read-only stage views over existing results, preserving recorded provenance."""
from dataclasses import replace
import json
from pathlib import Path

from research.legacy import PROJECT_ROOT


def select_result_view(result, *, baseline_only=False, strategies=None, models=None,
                       evaluation="training_validation"):
    if result.config.evaluation != evaluation:
        raise ValueError(f"This view requires {evaluation} results.")
    cases = result.cases
    if strategies is not None:
        missing = set(strategies) - set(cases.strategy)
        if missing:
            raise ValueError(f"No recorded cases for: {', '.join(sorted(missing))}")
        cases = cases.loc[cases.strategy.isin(strategies)]
    if baseline_only:
        cases = cases.loc[cases.variant.eq("baseline")]
    elif models is not None:
        cases = cases.loc[cases.variant.eq("baseline") | cases.model_id.isin(models)]
    case_ids = set(cases.case_id)
    model_ids = set(cases.loc[cases.variant.ne("baseline"), "model_id"])
    batches = {key: batch for key, batch in result.prediction_batches.items() if key in model_ids}

    def subset(frame):
        return frame.loc[frame.case_id.isin(case_ids)].copy() if "case_id" in frame else frame.iloc[:0].copy()

    view = replace(result, cases=cases.copy(), metrics=subset(result.metrics),
        paired=subset(result.paired), monthly=subset(result.monthly), diagnostics=subset(result.diagnostics),
        weights={key: value for key, value in result.weights.items() if key in case_ids},
        books={key: value for key, value in result.books.items() if key in case_ids},
        prediction_batches=batches, predictions=next(iter(batches.values()), None))
    if hasattr(result, "cache_info"):
        view.cache_info = dict(result.cache_info)
    # Config/manifest identify the source computation, including any hidden cases.
    # They are deliberately not rewritten to describe a new experiment.
    return view


def pinned_final_summary(store, release_path=None):
    """Only a pinned, fully ready frozen test run can fill the final-results slot."""
    path = Path(release_path or PROJECT_ROOT / "submission/release.json")
    if not path.is_file():
        return None
    release = json.loads(path.read_text())
    if (path.parent / release["artifact_root"]).resolve() != store.root:
        return None
    record = next((row for row in store.list_runs() if row["run_id"] == release["run_id"]), None)
    if record is None or record["evaluation"] != "test":
        return None
    summary = store.summary(record["run_id"])
    config = summary["manifest"]["config"]
    if config.get("fit_policy") != "frozen_at_origin":
        return None
    from research.reporting import readiness
    summary["release"] = release
    return summary if all(readiness(summary).values()) else None
