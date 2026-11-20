"""Measure what the local GPU actually drew during a run (BUILD_SPEC §6: measure, don't assume TDP)."""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from tally.cost import Throughput


@dataclass
class HardwareRecord:
    setup: str
    split: str
    prompt_version: str
    variant: str
    n_requests: int
    wall_seconds: float
    concurrency: int
    gpu_name: str | None
    mean_power_w: float | None
    max_power_w: float | None
    n_power_samples: int

    def throughput(self) -> Throughput:
        return Throughput(self.n_requests, self.wall_seconds, self.concurrency, self.mean_power_w)


def _query(fields: str) -> list[str] | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = (
            subprocess.run(
                [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            .stdout.strip()
            .splitlines()
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return [x.strip() for x in out[0].split(",")] if out else None


class PowerSampler:
    """Polls nvidia-smi every `interval_s` in a background thread."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        q = _query("name")
        self.gpu_name = q[0] if q else None

    def _loop(self) -> None:
        while not self._stop.is_set():
            q = _query("power.draw")
            if q:
                with contextlib.suppress(ValueError):
                    self.samples.append(float(q[0]))
            self._stop.wait(self.interval_s)

    def __enter__(self) -> PowerSampler:
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self.wall_seconds = time.perf_counter() - self._t0


def save(record: HardwareRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=2) + "\n", encoding="utf-8")


def load(path: Path) -> HardwareRecord:
    return HardwareRecord(**json.loads(path.read_text(encoding="utf-8")))
