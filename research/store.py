"""Durable experiment results: SQLite discovery and complete typed artifacts.

No model fitting happens in load/list/import. Exact reuse checks configuration,
source contents, calculation code and numerical runtime. Historical records can
still be viewed when they no longer match current code.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import uuid

import numpy as np
import pandas as pd

from .features import FeatureSpec
from .legacy import PROJECT_ROOT
from .models import PredictionBatch, default_model_registry, event_fields
from .runner import ExperimentRunner, RunConfig, RunResult


CALCULATION_FILES = [f"research/{name}.py" for name in
    ["runner", "data", "features", "selection", "models", "strategies", "screens", "adjustments", "legacy", "explanations", "cash_strategy", "accounting"]]
CALCULATION_FILES += ["part0_data_and_benchmark/pipeline.py"]
CALCULATION_FILES += [f"0_1_Basic_Strategies/{name}.py" for name in ["strategies", "hybrid", "downside"]]


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_from_dict(values):
    values = dict(values)
    values["strategies"] = tuple(values["strategies"])
    values["models"] = tuple(values.get("models", ["logistic"]))
    recipe = values.pop("feature_spec", {})
    recipe = dict(recipe)
    recipe["columns"] = tuple(recipe.get("columns", FeatureSpec().columns))
    values["feature_spec"] = FeatureSpec(**recipe)
    return RunConfig(**values)


def _model_identity(models):
    return [{key: model[key] for key in ["model_id", "estimator", "parameters", "preprocessing"]} for model in models]


def identity_from_manifest(manifest):
    """Original computation hashes remain authoritative for a historical import."""
    identity = {"store_identity_version": 1, "config": asdict(config_from_dict(manifest["config"])),
        "sources": manifest["source_sha256"],
        "calculation_code": {name: manifest["code_sha256"].get(name, "not-present-in-recorded-version") for name in CALCULATION_FILES},
        "runtime": manifest["versions"], "models": _model_identity(manifest.get("models", []))}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()


class ExperimentStore:
    def __init__(self, root=None, project_root=PROJECT_ROOT):
        default = os.environ.get("MLFACTOR_ARTIFACT_ROOT") or str(Path(project_root) / "artifacts")
        self.root = Path(root or default).resolve()
        self.project_root = Path(project_root).resolve()
        self.catalog = self.root / "catalog.sqlite"

    def fingerprint(self, config):
        import platform
        import sklearn
        models = default_model_registry()
        splits = ["train", "test"] if config.evaluation == "test" else ["train"]
        manifest = {"config": asdict(config),
            "source_sha256": {f"mlfactor_long_only_{s}.csv": digest_file(self.project_root / f"mlfactor_long_only_{s}.csv") for s in splits},
            "code_sha256": {name: digest_file(PROJECT_ROOT / name) for name in CALCULATION_FILES},
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
            "models": [models[key].metadata(config.feature_spec, config.prediction_target) for key in config.models] if config.include_ml else []}
        return identity_from_manifest(manifest)

    def _initialize(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.catalog, timeout=30) as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                  run_id TEXT PRIMARY KEY, cache_key TEXT NOT NULL, created_utc TEXT NOT NULL,
                  label TEXT NOT NULL, evaluation TEXT NOT NULL, model_count INTEGER NOT NULL,
                  case_count INTEGER NOT NULL, feature_id TEXT NOT NULL, origin TEXT NOT NULL,
                  artifact_sha256 TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS cache_lookup ON runs(cache_key, created_utc);
                CREATE TABLE IF NOT EXISTS case_metrics (
                  run_id TEXT NOT NULL, case_id TEXT NOT NULL, strategy TEXT NOT NULL,
                  model_id TEXT NOT NULL, feature_spec_id TEXT NOT NULL,
                  cagr REAL, annualized_volatility REAL, sharpe REAL, sortino REAL,
                  max_drawdown REAL, monthly_turnover REAL,
                  PRIMARY KEY (run_id, case_id));
            """)

    def list_runs(self):
        if not self.catalog.exists():
            return []
        with sqlite3.connect(self.catalog) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute("SELECT * FROM runs ORDER BY created_utc DESC, run_id DESC")]

    def find(self, config):
        key = self.fingerprint(config)
        return next((row for row in self.list_runs() if row["cache_key"] == key), None)

    def save(self, result, label=None, origin="computed", cache_key=None):
        self._initialize()
        cache_key = cache_key or identity_from_manifest(result.manifest)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:12]
        temporary = self.root / (".writing_" + run_id)
        destination = self.root / run_id
        temporary.mkdir()
        maps = {}
        hashes = {}

        def write_frame(relative, frame):
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=True, compression="zstd")
            hashes[relative] = digest_file(path)
            return relative

        try:
            for name in ["metrics", "paired", "monthly", "diagnostics", "signals", "screen", "cases"]:
                maps[name] = write_frame(f"tables/{name}.parquet", getattr(result, name))
            for group in ["weights", "books", "screens"]:
                maps[group] = {}
                for key, frame in getattr(result, group).items():
                    token = hashlib.sha256(key.encode()).hexdigest()[:24]
                    maps[group][key] = write_frame(f"{group}/{token}.parquet", frame)
            maps["prediction_batches"] = {}
            for key, batch in result.prediction_batches.items():
                token = hashlib.sha256(key.encode()).hexdigest()[:24]
                maps["prediction_batches"][key] = {name: write_frame(f"models/{token}/{name}.parquet", getattr(batch, name))
                    for name in ["predictions", "fits", "metrics", "calibration"]}
            metadata = {"artifact_schema": 1, "run_id": run_id, "config": asdict(result.config),
                "manifest": result.manifest, "data_audit": result.data_audit,
                "strategy_screen_keys": result.strategy_screen_keys, "frame_paths": maps,
                "file_sha256": hashes, "origin": origin}
            meta_path = temporary / "artifact.json"
            meta_path.write_text(json.dumps(metadata, indent=2, default=str, allow_nan=False) + "\n")
            artifact_hash = digest_file(meta_path)
            # A partially written run never enters the catalog.
            temporary.rename(destination)
            created = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(self.catalog, timeout=30) as connection:
                connection.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (run_id, cache_key, created, label or f"{result.config.feature_spec.label} · {result.config.evaluation} · {result.config.prediction_target}/{result.config.model_scope} · {len(result.cases)} portfolios",
                     result.config.evaluation, len(result.prediction_batches), len(result.cases),
                     result.config.feature_spec.feature_id, origin, artifact_hash))
                fields = ["cagr", "annualized_volatility", "sharpe", "sortino", "max_drawdown", "avg_one_way_turnover"]
                records = []
                for _, row in result.metrics.iterrows():
                    numeric = [float(row[name]) if pd.notna(row[name]) and math.isfinite(float(row[name])) else None for name in fields]
                    records.append((run_id, row.case_id, row.strategy, row.model_id, row.feature_spec_id, *numeric))
                connection.executemany("INSERT INTO case_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", records)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            # An uncatalogued directory is safe to remove; an existing run is never overwritten.
            if destination.exists() and not any(row["run_id"] == run_id for row in self.list_runs()):
                shutil.rmtree(destination)
            raise
        result.cache_info = {"run_id": run_id, "source": "computed" if origin == "computed" else "imported saved report",
                             "cache_hit": False, "cache_key": cache_key, "origin": origin}
        return run_id

    def _record(self, run_id, verify_frames=True):
        record = next((row for row in self.list_runs() if row["run_id"] == run_id), None)
        if record is None:
            raise ValueError(f"Unknown saved experiment: {run_id}")
        folder = self.root / record["run_id"]
        if not folder.is_dir() or not (folder / "artifact.json").is_file() or digest_file(folder / "artifact.json") != record["artifact_sha256"]:
            raise ValueError("Saved experiment metadata is missing or damaged.")
        metadata = json.loads((folder / "artifact.json").read_text())
        if metadata["artifact_schema"] != 1:
            raise ValueError("Unsupported saved experiment schema.")
        for relative, expected in (metadata["file_sha256"].items() if verify_frames else []):
            path = (folder / relative).resolve()
            if not path.is_relative_to(folder.resolve()) or not path.is_file() or digest_file(path) != expected:
                raise ValueError(f"Saved experiment file is missing or damaged: {relative}")
        return record, folder, metadata

    def configuration(self, run_id):
        """Read checksummed configuration for a chooser, without loading frames."""
        return config_from_dict(self._record(run_id, verify_frames=False)[2]["config"])

    def load(self, run_id):
        record, folder, metadata = self._record(run_id)
        maps = metadata["frame_paths"]
        frames = {name: pd.read_parquet(folder / maps[name]) for name in
                  ["metrics", "paired", "monthly", "diagnostics", "signals", "screen", "cases"]}
        groups = {name: {key: pd.read_parquet(folder / path) for key, path in maps[name].items()}
                  for name in ["weights", "books", "screens"]}
        batches = {key: PredictionBatch(**{name: pd.read_parquet(folder / path) for name, path in paths.items()})
                   for key, paths in maps["prediction_batches"].items()}
        result = RunResult(config=config_from_dict(metadata["config"]), **frames, **groups,
            predictions=next(iter(batches.values()), None), manifest=metadata["manifest"],
            strategy_screen_keys=metadata["strategy_screen_keys"], prediction_batches=batches,
            data_audit=metadata["data_audit"])
        result.cache_info = {"run_id": run_id, "source": "saved experiment", "cache_hit": True,
                             "cache_key": record["cache_key"], "origin": record["origin"]}
        return result

    def summary(self, run_id):
        """Small report-facing read, without loading weights or fitting models."""
        record, folder, metadata = self._record(run_id)
        tables = {name: pd.read_parquet(folder / metadata["frame_paths"][name])
                  for name in ["metrics", "paired", "monthly", "cases"]}
        model_metrics = []
        model_fits = []
        calibrations = []
        for index, (key, paths) in enumerate(metadata["frame_paths"]["prediction_batches"].items()):
            frame = pd.read_parquet(folder / paths["metrics"])
            frame["model_id"] = [key, event_fields(metadata["config"].get("prediction_target", "loss"))["control_id"]]
            model_metrics.append(frame if index == 0 else frame.iloc[:1])
            model_fits.append(pd.read_parquet(folder / paths["fits"]).assign(model_id=key))
            calibrations.append(pd.read_parquet(folder / paths["calibration"]).assign(model_id=key))
        tables["model_metrics"] = pd.concat(model_metrics, ignore_index=True) if model_metrics else pd.DataFrame()
        tables["model_fits"] = pd.concat(model_fits, ignore_index=True) if model_fits else pd.DataFrame()
        tables["model_calibration"] = pd.concat(calibrations, ignore_index=True) if calibrations else pd.DataFrame()
        return {**tables, "run_id": run_id, "manifest": metadata["manifest"],
                "data_audit": metadata["data_audit"], "record": record}

    def get_or_run(self, config, force=False, progress=None):
        key = self.fingerprint(config)
        if not force:
            record = next((row for row in self.list_runs() if row["cache_key"] == key), None)
            if record:
                if progress:
                    progress(f"Loading saved experiment {record['run_id']} (no fitting)")
                return self.load(record["run_id"])
        result = ExperimentRunner(self.project_root).run(config, progress=progress)
        self.save(result, cache_key=key)
        return result
