"""Versioned presentation exports, shared by the app, notebooks and reports.

Export reads completed experiments; it never replays a strategy or fits a model.
The reader needs neither source CSVs nor the research catalog. Allocation frames
are verified and loaded only when the corresponding strategy is inspected.
"""
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import pandas as pd

from research.runner import RunResult
from research.models import PredictionBatch
from research.store import config_from_dict, digest_file
from result_views import select_result_view
from strategy_help import strategy_descriptions


def require_training(config, manifest, monthly, baseline_only=True):
    """Reject test evidence even if its display label has been edited."""
    evaluation = config.evaluation if hasattr(config, "evaluation") else config["evaluation"]
    audit = manifest.get("data_audit", {})
    splits = audit.get("source_splits")
    sources = manifest.get("source_sha256", {})
    if evaluation != "training_validation" or splits != ["train"] or any("test" in name for name in sources):
        raise ValueError("Publication requires training-only sources and evaluation.")
    dates = pd.to_datetime(monthly.return_month)
    if dates.empty or dates.min() < pd.Timestamp("2000-01-01") or dates.max() > pd.Timestamp("2009-12-31"):
        raise ValueError("Publication must stay inside the 2000–2009 training period.")
    if dates.min() != pd.Timestamp(manifest["first_return_month"]) or dates.max() != pd.Timestamp(manifest["last_return_month"]):
        raise ValueError("Published monthly dates do not match their recorded evaluation period.")
    if not (pd.to_datetime(monthly.snapshot_date) + pd.offsets.MonthEnd(1)).equals(dates):
        raise ValueError("Each snapshot must earn its return in the following month.")
    if baseline_only and (not monthly.variant.eq("baseline").all() or not monthly.model_id.eq("none").all()):
        raise ValueError("The Strategies publication accepts only non-ML original portfolios.")


def export_strategies(store, run_id, destination, momentum_study_id=None, basic_run_ids=(), basket_study=None,
                      *, feature_study_id=None, feature_book_study=None, tuning_studies=(), final_book_study=None):
    """Create a new self-contained publication; existing exports are immutable."""
    source = store.load(run_id)
    result = select_result_view(source, baseline_only=True)
    require_training(result.config, result.manifest, result.monthly)
    record = next(row for row in store.list_runs() if row["run_id"] == run_id)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Choose a new output directory; publication already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (".publishing_" + uuid.uuid4().hex)
    temporary.mkdir()
    checksums = {}
    content_paths = {}

    def write(name, frame):
        path = temporary / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, compression="zstd", index=True)
        digest = digest_file(path)
        if digest in content_paths:
            path.unlink()  # Matching originals/weights are stored once per release.
            return content_paths[digest]
        checksums[name] = digest
        content_paths[digest] = name
        return name

    try:
        frames = {name: write(f"tables/{name}.parquet", getattr(result, name))
                  for name in ("cases", "metrics", "monthly")}
        weights = {}
        for case_id, frame in result.weights.items():
            token = hashlib.sha256(case_id.encode()).hexdigest()[:24]
            weights[case_id] = write(f"allocations/{token}.parquet", frame[["date", "stock_id", "weight"]])
        tab = {"source_run_id": run_id, "source_artifact_sha256": record["artifact_sha256"],
               "source_manifest": result.manifest, "frames": frames, "weights": weights,
               "descriptions": {key: text for key, text in strategy_descriptions(result.config).items()
                                if key in set(result.cases.strategy)}}
        if momentum_study_id:
            from research.momentum_studies import load_momentum_study
            study = load_momentum_study(store, momentum_study_id)
            summaries = {}
            for months, summary in study["summaries"].items():
                cases = summary["cases"].loc[lambda f: f.variant.eq("baseline")].copy()
                monthly = summary["monthly"].loc[lambda f: f.case_id.isin(cases.case_id)].copy()
                require_training(summary["manifest"]["config"], summary["manifest"], monthly)
                summaries[months] = {"manifest": summary["manifest"], "frames": {
                    "cases": write(f"momentum/{months}/cases.parquet", cases),
                    "monthly": write(f"momentum/{months}/monthly.parquet", monthly)}}
            table = study["portfolio_comparison"].loc[lambda f: f.variant.eq("baseline")].copy()
            tab["momentum"] = {"study_id": momentum_study_id, "metadata": study["metadata"],
                "summaries": summaries, "comparison": write("momentum/comparison.parquet", table)}
        tabs = {"strategies": tab}
        if basic_run_ids:
            runs = {}
            for basic_id in dict.fromkeys(basic_run_ids):
                basic = store.load(basic_id)
                if not basic.config.include_ml or not basic.prediction_batches:
                    raise ValueError("Basic ML publication requires saved prediction models.")
                require_training(basic.config, basic.manifest, basic.monthly, baseline_only=False)
                base = f"basic_ml/{basic_id}"
                basic_frames = {name: write(f"{base}/tables/{name}.parquet", getattr(basic, name))
                                for name in ("cases", "metrics", "monthly", "paired")}
                basic_weights = {}
                for case_id, frame in basic.weights.items():
                    token = hashlib.sha256(case_id.encode()).hexdigest()[:24]
                    basic_weights[case_id] = write(f"{base}/allocations/{token}.parquet", frame[["date", "stock_id", "weight"]])
                models = {}
                for model_id, batch in basic.prediction_batches.items():
                    models[model_id] = {name: write(f"{base}/models/{model_id}/{name}.parquet", getattr(batch, name))
                                        for name in ("metrics", "calibration", "fits")}
                    models[model_id]["prediction_sources"] = {
                        str(key): int(count) for key, count in batch.predictions.prediction_source.value_counts().items()
                    } if "prediction_source" in batch.predictions else {}
                record = next(row for row in store.list_runs() if row["run_id"] == basic_id)
                runs[basic_id] = {"source_run_id": basic_id, "source_record": record,
                    "source_artifact_sha256": record["artifact_sha256"], "source_manifest": basic.manifest,
                    "frames": basic_frames, "weights": basic_weights, "models": models}
            tabs["basic_ml"] = {"runs": runs}
        if basket_study is not None:
            from research.basket_momentum import load_basket_study
            from research.basket_reporting import normalize_study, require_basket_training
            basket_path = Path(basket_study)
            advanced = normalize_study(load_basket_study(basket_path))
            require_basket_training(advanced)
            frames = {name: write(f"advanced_ml/{name}.parquet", advanced[name])
                      for name in ("metrics", "monthly", "predictive", "fits") if name in advanced}
            details = {}
            for name in ("baskets", "assignments"):
                details[name] = {}
                for (months, model, version), frame in advanced[name].groupby(
                        ["momentum_months", "model_key", "version"], sort=False):
                    key = json.dumps([int(months), model, version])
                    token = hashlib.sha256(key.encode()).hexdigest()[:24]
                    details[name][key] = write(f"advanced_ml/{name}/{token}.parquet", frame)
            tabs["advanced_ml"] = {"source_study_id": basket_path.name,
                "source_metadata_sha256": digest_file(basket_path / "study.json"),
                "metadata": advanced["metadata"], "frames": frames, "details": details}
        if feature_study_id:
            from published_analysis import export_features
            tabs['features'] = export_features(store, feature_study_id, feature_book_study, write)
        if tuning_studies:
            from published_analysis import export_tuning
            tabs['tuning'] = export_tuning(tuning_studies, write)
        if final_book_study:
            from published_analysis import export_final
            tabs['final'] = export_final(final_book_study, basket_study, write, store.project_root)
        metadata = {"publication_schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "published_tabs": list(tabs), "tabs": tabs, "file_sha256": checksums}
        (temporary / "manifest.json").write_text(json.dumps(metadata, indent=2, default=str, allow_nan=False) + "\n")
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


class PublishedStrategies:
    """Read only this release, verifying each requested frame before use."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.metadata = json.loads((self.root / "manifest.json").read_text())
        if self.metadata.get("publication_schema") != 1:
            raise ValueError("Unsupported published-results schema.")
        self.tab = self.metadata["tabs"]["strategies"]
        self._frames = {}
        self.token = digest_file(self.root / "manifest.json")

    def frame(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or relative not in self.metadata["file_sha256"]:
            raise ValueError("Published frame is not declared inside this release.")
        if relative not in self._frames:
            if not path.is_file() or digest_file(path) != self.metadata["file_sha256"][relative]:
                raise ValueError(f"Published result is missing or damaged: {relative}")
            self._frames[relative] = pd.read_parquet(path)
        return self._frames[relative].copy()

    def result(self):
        return _read_result(self, self.tab)

    def allocations(self, result, strategy, models=None):
        return _read_allocations(self, self.tab, result, strategy, models)

    def basic_store(self):
        return PublishedBasicStore(self)

    def baskets(self):
        if "advanced_ml" not in self.metadata["tabs"]:
            return None
        return PublishedBasketStudy(self)

    def analysis_section(self, name):
        if name not in self.metadata['tabs']:
            return None
        if name == 'acceleration':
            from acceleration_publication import PublishedAcceleration
            return PublishedAcceleration(self)
        if name == 'engineering':
            from engineering_publication import PublishedEngineering
            return PublishedEngineering(self)
        from published_analysis import PublishedFeatures, PublishedTuning, PublishedFinal
        return {'features': PublishedFeatures, 'tuning': PublishedTuning, 'final': PublishedFinal}[name](self)

    def momentum(self):
        record = self.tab.get("momentum")
        if record is None:
            return None
        summaries = {}
        for months, item in record["summaries"].items():
            frames = {name: self.frame(path) for name, path in item["frames"].items()}
            require_training(item["manifest"]["config"], item["manifest"], frames["monthly"])
            summaries[months] = {**frames, "manifest": item["manifest"]}
        return {"metadata": record["metadata"], "summaries": summaries,
                "portfolio_comparison": self.frame(record["comparison"])}


def _read_result(publication, record, basic=False):
    frames = {key: publication.frame(path) for key, path in record["frames"].items()}
    manifest = record["source_manifest"]
    config = config_from_dict(manifest["config"])
    require_training(config, manifest, frames["monthly"], baseline_only=not basic)
    cases = frames["cases"]
    if cases.case_id.duplicated().any() or (not basic and not cases.variant.eq("baseline").all()):
        raise ValueError("Published cases must be distinct, correctly scoped portfolios.")
    expected = set(cases.case_id)
    if set(frames["metrics"].case_id) != expected or set(frames["monthly"].case_id) != expected:
        raise ValueError("Published metrics, histories and cases do not match.")
    if basic:
        expected_variant = "loss_adjusted" if config.prediction_target == "loss" else "upside_adjusted"
        if not config.include_ml or set(cases.variant) != {"baseline", expected_variant}:
            raise ValueError("Basic ML publication has incompatible target/cases.")
        if set(cases.loc[cases.variant.ne("baseline"), "model_id"]) != set(record["models"]):
            raise ValueError("Published model cases and probability summaries differ.")
    batches = {}
    for key, item in record.get("models", {}).items():
        # No stock-month probabilities or model fitting objects are needed to
        # render probability scores/calibration. Detailed fit audits load on demand.
        batches[key] = PredictionBatch(predictions=pd.DataFrame(), fits=pd.DataFrame(),
            metrics=publication.frame(item["metrics"]), calibration=publication.frame(item["calibration"]))
    books = {key: group.set_index("return_month").sort_index()
             for key, group in frames["monthly"].groupby("case_id", sort=False)}
    paired = frames.pop("paired", pd.DataFrame())
    result = RunResult(config=config, **frames, books=books, weights={}, paired=paired,
        diagnostics=pd.DataFrame(), signals=pd.DataFrame(), screen=pd.DataFrame(),
        predictions=next(iter(batches.values()), None), prediction_batches=batches, manifest=manifest)
    result.cache_info = {"run_id": record["source_run_id"], "source": "published results", "cache_hit": True}
    return result


def _read_allocations(publication, record, result, strategy, models=None):
    cases = result.cases.loc[result.cases.strategy.eq(strategy)]
    if models is not None:
        cases = cases.loc[cases.variant.eq("baseline") | cases.model_id.isin(models)]
    weights = {}
    for key in cases.case_id:
        frame = publication.frame(record["weights"][key])
        if frame.duplicated(["date", "stock_id"]).any() or set(frame.date) != set(result.books[key].snapshot_date):
            raise ValueError("Published allocation dates/stock keys do not match this portfolio.")
        weights[key] = frame
    signals = next(iter(weights.values()))[["date", "stock_id"]].iloc[:0]
    allocated = replace(result, weights=weights, signals=signals)
    allocated.cache_info = dict(result.cache_info)
    return allocated


class PublishedBasicStore:
    """Small read-only adapter for the existing Basic ML chooser/comparison UI.

    Exposes saved-result reads only. It never creates ExperimentStore, fingerprints
    a source CSV, accesses SQLite, or offers a compute method.
    """

    def __init__(self, publication):
        self.publication = publication
        self.root = publication.root
        self.publication_token = publication.token
        self.runs = publication.metadata["tabs"].get("basic_ml", {}).get("runs", {})
        self._results = {}

    def list_runs(self):
        return [item["source_record"] for item in self.runs.values()]

    def configuration(self, run_id):
        return config_from_dict(self.runs[run_id]["source_manifest"]["config"])

    def load(self, run_id):
        if run_id not in self._results:
            self._results[run_id] = _read_result(self.publication, self.runs[run_id], basic=True)
        return self._results[run_id]

    def summary(self, run_id):
        result = self.load(run_id)
        return {"metrics": result.metrics, "model_metrics": result.model_metrics,
                "manifest": result.manifest}

    def allocations(self, result, strategy, models=None):
        record = self.runs[result.cache_info["run_id"]]
        return _read_allocations(self.publication, record, result, strategy, models)

    def prediction_sources(self, result, model_id):
        return self.runs[result.cache_info["run_id"]]["models"][model_id]["prediction_sources"]

    def model_fits(self, result, model_id):
        item = self.runs[result.cache_info["run_id"]]["models"][model_id]
        return self.publication.frame(item["fits"])


class PublishedBasketStudy:
    """Advanced summaries plus on-demand saved membership/weight partitions."""

    def __init__(self, publication, record=None):
        self.publication = publication
        self.record = record if record is not None else publication.metadata["tabs"]["advanced_ml"]

    def for_method(self, method):
        if method == "kmeans":
            return self
        record = self.record.get("grouping_methods", {}).get(method)
        return PublishedBasketStudy(self.publication, record) if record else None

    def study(self):
        from research.basket_reporting import require_basket_training
        frames = {name: self.publication.frame(path) for name, path in self.record["frames"].items()
                  if name != "fits"}
        study = {"metadata": self.record["metadata"], **frames, "detail_loader": self.details}
        require_basket_training(study)
        return study

    def details(self, months, versions=None, model_key="none"):
        result = {}
        for name, partitions in self.record["details"].items():
            frames = []
            for key, path in partitions.items():
                horizon, model, version = json.loads(key)
                if horizon == months and model in {"none", model_key} and (versions is None or version in versions):
                    frames.append(self.publication.frame(path))
            result[name] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
                columns=["date", "stock_id", "momentum_months", "model_key", "version"])
        return result


def export_grouping_comparison(source_publication, basket_study, destination):
    """Extend an immutable publication with a saved Ward study; preserve other tabs.

    This is a publication operation only: no data loading, clustering or replay.
    """
    from research.basket_momentum import load_basket_study
    from research.basket_reporting import normalize_study, require_basket_training
    source = PublishedStrategies(source_publication)
    advanced = normalize_study(load_basket_study(basket_study))
    require_basket_training(advanced)
    if advanced['metadata'].get('grouping_method') != 'ward':
        raise ValueError('The additional grouping comparison must be a Ward study.')
    reference = source.baskets().study()
    for key in ('n_clusters', 'lookback_months', 'momentum_months', 'first_return_month',
                'last_return_month', 'top_positive_vol_count', 'top_positive_vol_months',
                'directional_adjustment', 'adjustment_strength', 'upside_threshold', 'forecast_sources'):
        if reference['metadata'][key] != advanced['metadata'][key]:
            raise ValueError(f'Grouping comparisons must share {key}.')
    # Shared controls must be numerically identical before comparing learned memberships.
    for name in ('monthly', 'metrics'):
        control = reference[name].loc[lambda f: f.version.isin(['Equal weight', 'Momentum + cash'])]
        new = advanced[name].loc[lambda f: f.version.isin(['Equal weight', 'Momentum + cash'])]
        pd.testing.assert_frame_equal(control.reset_index(drop=True), new.reset_index(drop=True))
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'Choose a new publication directory: {destination}')
    temporary = destination.parent / ('.publishing_' + uuid.uuid4().hex)
    metadata = json.loads(json.dumps(source.metadata))
    try:
        # Validate all prior exported evidence before preserving it in the new release.
        for relative, expected in metadata['file_sha256'].items():
            if digest_file(source.root / relative) != expected:
                raise ValueError(f'Damaged source publication: {relative}')
        shutil.copytree(source.root, temporary)
        def write(name, frame):
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, compression='zstd', index=True)
            metadata['file_sha256'][name] = digest_file(path)
            return name
        frames = {name: write(f'grouping/ward/{name}.parquet', advanced[name])
                  for name in ('metrics', 'monthly', 'predictive', 'fits', 'hierarchy') if name in advanced}
        details = {}
        for name in ('baskets', 'assignments'):
            details[name] = {}
            for (months, model, version), frame in advanced[name].groupby(
                    ['momentum_months', 'model_key', 'version'], sort=False):
                key = json.dumps([int(months), model, version])
                token = hashlib.sha256(key.encode()).hexdigest()[:24]
                details[name][key] = write(f'grouping/ward/{name}/{token}.parquet', frame)
        metadata['tabs']['advanced_ml'].setdefault('grouping_methods', {})['ward'] = {
            'source_study_id': Path(basket_study).name,
            'source_metadata_sha256': digest_file(Path(basket_study) / 'study.json'),
            'metadata': advanced['metadata'], 'frames': frames, 'details': details}
        metadata['created_utc'] = datetime.now(timezone.utc).isoformat()
        metadata['parent_manifest_sha256'] = digest_file(source.root / 'manifest.json')
        (temporary / 'manifest.json').write_text(json.dumps(metadata, indent=2, default=str, allow_nan=False) + '\n')
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
