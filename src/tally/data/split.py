"""Stratified, cluster-grouped 60/20/20 split with a committed manifest (BUILD_SPEC §3.2)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from tally.config import ROOT, ExperimentCfg
from tally.data import dedup

MANIFEST_PATH = ROOT / "data" / "splits" / "manifest.json"
SPLITS = ("dev", "tune", "test")


@dataclass(frozen=True)
class SplitResult:
    frame: pd.DataFrame  # input columns + cluster_id + split
    manifest: dict[str, Any]


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def assign_splits(df: pd.DataFrame, cluster_ids: list[int], cfg: ExperimentCfg) -> pd.Series:
    """Five stratified group folds: 3 -> dev, 1 -> tune, 1 -> test (60/20/20)."""
    ratios = cfg.split.ratios
    if not np.isclose([ratios["dev"], ratios["tune"], ratios["test"]], [0.6, 0.2, 0.2]).all():
        raise ValueError("assign_splits implements 60/20/20 via 5 folds; other ratios need a new splitter")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.split.seed)
    fold = np.full(len(df), -1)
    for k, (_, test_idx) in enumerate(sgkf.split(df["text"], df["intent"], groups=cluster_ids)):
        fold[test_idx] = k
    if (fold < 0).any():
        raise RuntimeError("some rows were not assigned to a fold")
    names = np.array(["dev", "dev", "dev", "tune", "test"])
    return pd.Series(names[fold], index=df.index, name="split")


def build(df: pd.DataFrame, cfg: ExperimentCfg, *, dataset_revision: str, dataset_sha256: str) -> SplitResult:
    d = cfg.split.dedup
    clusters = dedup.cluster(
        df["text"].tolist(),
        shingle_chars=d.shingle_chars,
        num_perm=d.num_perm,
        threshold=d.jaccard_threshold,
        seed=cfg.split.seed % (2**31),
    )
    out = df.copy()
    out["cluster_id"] = clusters
    out["split"] = assign_splits(out, clusters, cfg)
    sizes = out.groupby("cluster_id").size()
    manifest: dict[str, Any] = {
        "dataset": {"repo_id": cfg.dataset.repo_id, "revision": dataset_revision, "sha256": dataset_sha256},
        "seed": cfg.split.seed,
        "dedup": d.model_dump(),
        "n_rows": len(out),
        "n_clusters": int(sizes.size),
        "n_multi_row_clusters": int((sizes > 1).sum()),
        "largest_cluster": int(sizes.max()),
        "splits": {},
    }
    for s in SPLITS:
        part = out[out["split"] == s]
        ids = part["ticket_id"].tolist()
        manifest["splits"][s] = {
            "n": len(ids),
            "sha256": ids_sha256(ids),
            "intent_counts": {k: int(v) for k, v in part["intent"].value_counts().sort_index().items()},
        }
    leaks = out.groupby("cluster_id")["split"].nunique()
    if (leaks > 1).any():
        raise RuntimeError("a near-duplicate cluster spans more than one split")
    return SplitResult(frame=out, manifest=manifest)


def write(result: SplitResult, *, manifest_path: Path = MANIFEST_PATH, ids_dir: Path | None = None) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ids_dir = ids_dir or manifest_path.parent
    for s in SPLITS:
        ids = sorted(result.frame.loc[result.frame["split"] == s, "ticket_id"])
        (ids_dir / f"{s}_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")


def read_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def verify(result: SplitResult, manifest: dict[str, Any]) -> list[str]:
    """Return a list of differences between a freshly built split and the committed manifest."""
    problems: list[str] = []
    for s in SPLITS:
        got = result.manifest["splits"][s]["sha256"]
        want = manifest["splits"][s]["sha256"]
        if got != want:
            problems.append(f"{s}: sha256 {got[:12]} != committed {want[:12]}")
    if result.manifest["dataset"]["sha256"] != manifest["dataset"]["sha256"]:
        problems.append("dataset file sha256 differs from the committed manifest")
    return problems
