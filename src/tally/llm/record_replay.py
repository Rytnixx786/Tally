"""Record every external model response; replay them bit-for-bit (ADR-008).

Modes:
- live:   call the provider, record the response (overwrites nothing; first write wins)
- replay: serve recorded responses; a miss is an error, so a reproduction never calls out silently
- auto:   replay on hit, call live on miss and record it
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Literal

from tally.llm.types import CallResult, ChatRequest

Mode = Literal["live", "replay", "auto"]
# Gateway rejections that depend only on the request (guardrail 400, budget 402, ...) are part of the
# outcome and are recorded so that a replay reproduces them.
DETERMINISTIC_FAILURES = frozenset({400, 402, 413, 422})


class ReplayMissError(RuntimeError):
    pass


def request_key(req: ChatRequest) -> str:
    blob = json.dumps(req.key_payload(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Recorder:
    """Append-only JSONL store, one file per (provider, model)."""

    def __init__(self, root: Path, mode: Mode) -> None:
        self.root = root
        self.mode = mode
        self._cache: dict[Path, dict[str, dict[str, object]]] = {}
        self._lock = threading.Lock()

    def _file(self, req: ChatRequest) -> Path:
        safe = req.model.replace("/", "__").replace(":", "_")
        return self.root / req.provider / f"{safe}.jsonl"

    def _index(self, path: Path) -> dict[str, dict[str, object]]:
        if path not in self._cache:
            idx: dict[str, dict[str, object]] = {}
            if path.exists():
                with path.open(encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            row = json.loads(line)
                            idx.setdefault(row["request_key"], row)
            self._cache[path] = idx
        return self._cache[path]

    def get(self, req: ChatRequest) -> CallResult | None:
        row = self._index(self._file(req)).get(request_key(req))
        return CallResult.from_json(dict(row)) if row is not None else None

    def put(self, result: CallResult, req: ChatRequest) -> None:
        if not (result.ok or result.http_status in DETERMINISTIC_FAILURES):
            return  # transient failures (timeouts, 429, 5xx) are retried, never recorded
        path = self._file(req)
        with self._lock:
            idx = self._index(path)
            if result.request_key in idx:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            row = result.to_json()
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            idx[result.request_key] = row

    def count(self, req: ChatRequest) -> int:
        return len(self._index(self._file(req)))
