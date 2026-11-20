"""Deterministic ids, so a replayed run produces byte-identical warehouse rows."""

from __future__ import annotations

import hashlib
import uuid

NAMESPACE = uuid.UUID("8f0d2c1e-6b7a-4c55-9a3e-2f61d4b0a7c9")


def decision_id(run_id: str, setup: str, split: str, ticket_id: str, variant: str = "") -> str:
    return str(uuid.uuid5(NAMESPACE, f"decision:{run_id}:{setup}:{split}:{ticket_id}:{variant}"))


def request_id(decision: str, call_index: int) -> str:
    return str(uuid.uuid5(NAMESPACE, f"request:{decision}:{call_index}"))


def trace_id_int(decision: str) -> int:
    return int.from_bytes(hashlib.sha256(f"trace:{decision}".encode()).digest()[:16], "big") or 1


def span_id_int(key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"span:{key}".encode()).digest()[:8], "big") or 1
