"""Explicit runtime choice; deployment never falls back to research computation."""
import os
from pathlib import Path


def app_mode():
    mode = os.environ.get("MLFACTOR_APP_MODE", "research").lower()
    if mode not in {"research", "deploy"}:
        raise ValueError("MLFACTOR_APP_MODE must be research or deploy.")
    return mode


def published_root():
    return Path(os.environ.get("MLFACTOR_PUBLISHED_ROOT") or
                Path(__file__).resolve().parent / "published" / "project")
