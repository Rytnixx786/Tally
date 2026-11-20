from __future__ import annotations

import os

import pandas as pd
import pytest

from tally.config import load_cost_model, load_experiment, load_roi, load_tenants

INTENTS = {
    "track_order": (
        "ORDER",
        [
            "where is my order {i}",
            "track my package number {i}",
            "status of order {i} please",
            "i want to know where order {i} is",
            "can you track shipment {i}",
        ],
    ),
    "get_refund": (
        "REFUND",
        [
            "i want my money back for {i}",
            "refund purchase {i} now",
            "give me a refund on {i}",
            "how do i get a refund for item {i}",
            "request refund for order {i}",
        ],
    ),
    "recover_password": (
        "ACCOUNT",
        [
            "forgot my password {i}",
            "reset my login key {i}",
            "cannot remember pwd {i}",
            "help me recover password {i}",
            "lost access, reset password {i}",
        ],
    ),
}


def synthetic_tickets(per_intent: int = 30) -> pd.DataFrame:
    rows = []
    n = 0
    for intent, (cat, templates) in INTENTS.items():
        for i in range(per_intent):
            t = templates[i % len(templates)].format(i=1000 + i * 7)
            split = "dev" if i % 5 in (0, 1, 2) else ("tune" if i % 5 == 3 else "test")
            rows.append(
                {
                    "ticket_id": f"bx-{n:05d}",
                    "text": t,
                    "intent": intent,
                    "category": cat,
                    "flags": "BLQZ" if i % 4 == 0 else "BL",
                    "cluster_id": n,
                    "split": split,
                    "tenant_id": ["support", "billing", "sales"][n % 3],
                }
            )
            n += 1
    return pd.DataFrame(rows)


@pytest.fixture
def tickets() -> pd.DataFrame:
    return synthetic_tickets()


@pytest.fixture(scope="session")
def exp_cfg():  # type: ignore[no-untyped-def]
    return load_experiment()


@pytest.fixture(scope="session")
def cost_cfg():  # type: ignore[no-untyped-def]
    return load_cost_model()


@pytest.fixture(scope="session")
def roi_cfg():  # type: ignore[no-untyped-def]
    return load_roi()


@pytest.fixture(scope="session")
def tenants_cfg():  # type: ignore[no-untyped-def]
    return load_tenants()


def pg_dsn() -> str | None:
    """DSN for integration tests: TALLY_TEST_DSN, or the local compose Postgres if it answers."""
    dsn = os.environ.get("TALLY_TEST_DSN")
    if dsn:
        return dsn
    try:
        import psycopg

        from tally.warehouse import dsn as default_dsn

        d = default_dsn()
        with psycopg.connect(d, connect_timeout=2):
            return d
    except Exception:
        return None
