"""Publish/read saved acceleration rules without datasets, fitting or replay."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid

from research.store import digest_file
from research.momentum_acceleration import load_acceleration_study, _check


def export_acceleration(source_root, study_directory, destination):
    from published_results import PublishedStrategies

    source = PublishedStrategies(source_root)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Choose a new publication directory.")
    loaded = {
        window: load_acceleration_study(study_directory, window)
        for window in ("training", "test")
    }
    tmp = destination.parent / (".publishing_" + uuid.uuid4().hex)
    try:
        for name, digest in source.metadata["file_sha256"].items():
            if digest_file(source.root / name) != digest:
                raise ValueError("Damaged source publication.")
        shutil.copytree(source.root, tmp)
        meta = json.loads(json.dumps(source.metadata))
        section = {
            "study_id": Path(study_directory).name,
            "metadata": loaded["training"]["metadata"],
            "windows": {},
        }
        for window, study in loaded.items():
            frames = {}
            for name in ("metrics", "monthly", "signals"):
                relative = f"acceleration/{window}/{name}.parquet"
                path = tmp / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                study[name].to_parquet(path, index=False, compression="zstd")
                frames[name] = relative
                meta["file_sha256"][relative] = digest_file(path)
            section["windows"][window] = {"scope": study["scope"], "frames": frames}
        section["source_study_sha256"] = digest_file(
            Path(study_directory) / "study.json"
        )
        meta["tabs"]["acceleration"] = section
        meta["published_tabs"] = list(meta["tabs"])
        meta["parent_manifest_sha256"] = source.token
        meta["created_utc"] = datetime.now(timezone.utc).isoformat()
        (tmp / "manifest.json").write_text(
            json.dumps(meta, indent=2, allow_nan=False) + "\n"
        )
        tmp.rename(destination)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return destination


class PublishedAcceleration:
    def __init__(self, publication):
        self.publication = publication
        self.section = publication.metadata["tabs"]["acceleration"]

    def has_window(self, window):
        return window in self.section["windows"]

    def study(self, window="training"):
        record = self.section["windows"][window]
        result = {
            "metadata": self.section["metadata"],
            "scope": record["scope"],
            **{
                name: self.publication.frame(path)
                for name, path in record["frames"].items()
            },
        }
        _check(result, window)
        return result
