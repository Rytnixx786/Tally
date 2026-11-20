"""Filesystem locations. Large downloads live outside the repo, in TALLY_DOWNLOAD_DIR."""

from __future__ import annotations

import os
from pathlib import Path

from tally.config import ROOT


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env reader (KEY=VALUE lines). Existing environment variables win."""
    p = path or ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value:  # an empty value means "use the default", not "set to empty"
            os.environ.setdefault(key.strip(), value)


def download_root() -> Path:
    """Where datasets and model weights are downloaded. Defaults to <repo>/data/downloads."""
    load_dotenv()
    raw = os.environ.get("TALLY_DOWNLOAD_DIR")
    return Path(raw) if raw else ROOT / "data" / "downloads"


def dataset_dir() -> Path:
    return download_root() / "datasets" / "tally"


def hf_home() -> Path:
    return download_root() / "huggingface"


INTERIM_DIR = ROOT / "data" / "interim"
SPLITS_DIR = ROOT / "data" / "splits"
RECORDINGS_DIR = ROOT / "data" / "recordings"
RUNS_DIR = ROOT / "reports" / "runs"
