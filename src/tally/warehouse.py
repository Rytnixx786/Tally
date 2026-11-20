"""Load run artifacts and reference tables into the raw schema; read marts back."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from psycopg.types.json import Jsonb

from tally import prompts
from tally.config import ROOT, CostModelCfg, TenantsCfg
from tally.paths import load_dotenv


def dsn(role: str = "writer") -> str:
    load_dotenv()
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "tally")
    if role == "writer":
        user, pw = (
            os.environ.get("POSTGRES_USER", "tally_writer"),
            os.environ.get("POSTGRES_PASSWORD") or "tally_writer_local",
        )
    elif role == "lexicon":
        user, pw = "lexicon_ro", os.environ.get("LEXICON_RO_PASSWORD") or "lexicon_ro_local"
    else:
        raise ValueError(role)
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


def connect(role: str = "writer") -> psycopg.Connection[Any]:
    return psycopg.connect(dsn(role))


def _upsert(
    conn: psycopg.Connection[Any],
    table: str,
    cols: Sequence[str],
    rows: list[tuple[Any, ...]],
    key: Sequence[str],
) -> int:
    if not rows:
        return 0
    placeholders = ", ".join(["%s"] * len(cols))
    updates = (
        ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in key) or f"{key[0]} = EXCLUDED.{key[0]}"
    )
    sql = (
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT ({', '.join(key)}) DO UPDATE SET {updates}"
    )
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)


def _none(v: Any) -> Any:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def load_reference(conn: psycopg.Connection[Any], ten: TenantsCfg, cost_cfg: CostModelCfg) -> None:
    _upsert(
        conn,
        "raw.tenants",
        ["tenant_id", "display_name", "owner_project", "monthly_budget_usd", "traffic_mix"],
        [
            (k, v.display_name, v.owner_project, v.monthly_budget_usd, Jsonb(v.weights))
            for k, v in ten.tenants.items()
        ],
        ["tenant_id"],
    )
    _upsert(
        conn,
        "raw.price_book",
        [
            "provider",
            "model",
            "input_per_mtok",
            "output_per_mtok",
            "cached_input_per_mtok",
            "checked_on",
            "source",
        ],
        [
            (
                p.provider,
                p.model,
                p.input_per_mtok,
                p.output_per_mtok,
                p.cached_input_per_mtok,
                p.checked_on,
                p.source,
            )
            for p in cost_cfg.price_book
        ],
        ["provider", "model", "checked_on"],
    )
    _upsert(
        conn,
        "raw.prompt_registry",
        ["prompt_version", "feature", "prompt_hash", "template_path", "n_fewshot", "description"],
        [
            (
                pv.version,
                prompts.FEATURE,
                pv.prompt_hash,
                str(pv.template_path.relative_to(ROOT)).replace("\\", "/"),
                pv.n_fewshot,
                pv.description,
            )
            for pv in prompts.REGISTRY.values()
        ]
        + [("none", prompts.FEATURE, "none", "-", 0, "No prompt: setup A (TF-IDF + logistic regression)")],
        ["prompt_version"],
    )
    conn.commit()


def load_run(conn: psycopg.Connection[Any], run_dir: Path) -> dict[str, int]:
    dec = pd.read_parquet(run_dir / "decisions.parquet")
    cal = pd.read_parquet(run_dir / "calls.parquet")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    run_id = manifest["run_id"]
    with conn.cursor() as cur:
        cur.execute("DELETE FROM raw.eval_outcomes WHERE run_id = %s", (run_id,))
        cur.execute("DELETE FROM raw.call_costs WHERE run_id = %s", (run_id,))
    cols = [
        "run_id",
        "decision_id",
        "ticket_id",
        "split",
        "arm",
        "tenant_id",
        "setup",
        "prompt_version",
        "gold_intent",
        "gold_category",
        "predicted_intent",
        "predicted_category",
        "valid",
        "confidence",
        "calibrated_confidence",
        "escalated",
        "final_request_id",
        "flags",
        "decided_at",
    ]
    rows = []
    for d in dec.to_dict("records"):
        d["decided_at"] = datetime.fromtimestamp(float(d["decided_at"]), tz=UTC)
        rows.append(tuple(_none(d[c]) for c in cols))
    n_dec = _upsert(conn, "raw.eval_outcomes", cols, rows, ["decision_id"])
    n_cost = _upsert(
        conn,
        "raw.call_costs",
        ["request_id", "run_id", "cost_usd", "cost_method"],
        [(r["request_id"], run_id, float(r["cost_usd"]), r["cost_method"]) for r in cal.to_dict("records")],
        ["request_id"],
    )
    _upsert(conn, "raw.run_manifest", ["run_id", "manifest"], [(run_id, Jsonb(manifest))], ["run_id"])
    conn.commit()
    return {"decisions": n_dec, "call_costs": n_cost}


def init_schema(admin_dsn: str) -> None:
    """Apply deploy/postgres/init/sql/schema.sql (for a Postgres not started by docker compose)."""
    sql = (ROOT / "deploy" / "postgres" / "init" / "sql" / "schema.sql").read_text(encoding="utf-8")
    load_dotenv()
    sql = sql.replace(
        ":'writer_pw'", "'" + (os.environ.get("POSTGRES_PASSWORD") or "tally_writer_local") + "'"
    )
    sql = sql.replace(
        ":'lexicon_pw'", "'" + (os.environ.get("LEXICON_RO_PASSWORD") or "lexicon_ro_local") + "'"
    )
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        conn.execute(sql.encode())


def read_sql(query: str, role: str = "writer", params: Sequence[Any] | None = None) -> pd.DataFrame:
    with connect(role) as conn, conn.cursor() as cur:
        cur.execute(query.encode(), params)
        cols = [c.name for c in (cur.description or [])]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def count_spans(run_id: str) -> int:
    with connect() as conn:
        row = conn.execute(
            "SELECT count(*) FROM raw.otel_spans WHERE attributes->>'tally.run_id' = %s", (run_id,)
        ).fetchone()
    return int(row[0]) if row else 0
