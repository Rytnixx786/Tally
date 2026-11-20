"""Download and validate the Bitext customer-support dataset.

The dataset is never committed (CDLA-Sharing-1.0); it is downloaded at a pinned revision.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tally.config import DatasetCfg
from tally.paths import dataset_dir

DATA_FILE = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
DEFAULT_REVISION = "430d1a89bd93bd1fa23c16f29dd53e73f0087443"  # resolved 2026-11-20

EXPECTED_INTENTS: frozenset[str] = frozenset(
    {
        "cancel_order",
        "change_order",
        "change_shipping_address",
        "check_cancellation_fee",
        "check_invoice",
        "check_payment_methods",
        "check_refund_policy",
        "complaint",
        "contact_customer_service",
        "contact_human_agent",
        "create_account",
        "delete_account",
        "delivery_options",
        "delivery_period",
        "edit_account",
        "get_invoice",
        "get_refund",
        "newsletter_subscription",
        "payment_issue",
        "place_order",
        "recover_password",
        "registration_problems",
        "review",
        "set_up_shipping_address",
        "switch_account",
        "track_order",
        "track_refund",
    }
)
EXPECTED_CATEGORIES: frozenset[str] = frozenset(
    {
        "ACCOUNT",
        "ORDER",
        "REFUND",
        "CONTACT",
        "INVOICE",
        "PAYMENT",
        "FEEDBACK",
        "DELIVERY",
        "SHIPPING",
        "SUBSCRIPTION",
        "CANCEL",
    }
)
EXPECTED_ROWS = 26_872


@dataclass(frozen=True)
class LoadedDataset:
    frame: pd.DataFrame  # columns: ticket_id, text, intent, category, flags
    revision: str
    sha256: str


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(cfg: DatasetCfg, revision: str | None = None) -> tuple[Path, str]:
    from huggingface_hub import hf_hub_download

    rev = revision or cfg.revision or DEFAULT_REVISION
    path = hf_hub_download(
        repo_id=cfg.repo_id,
        filename=DATA_FILE,
        repo_type="dataset",
        revision=rev,
        local_dir=dataset_dir(),
    )
    return Path(path), rev


def normalise_frame(raw: pd.DataFrame, cfg: DatasetCfg) -> pd.DataFrame:
    """Rename to Tally's column names and assign stable ticket ids (row order of the pinned file)."""
    missing = {cfg.text_column, cfg.label_column, cfg.category_column, cfg.flags_column} - set(raw.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    df = pd.DataFrame(
        {
            "ticket_id": [f"bx-{i:05d}" for i in range(len(raw))],
            "text": raw[cfg.text_column].astype(str).str.strip(),
            "intent": raw[cfg.label_column].astype(str).str.strip(),
            "category": raw[cfg.category_column].astype(str).str.strip(),
            "flags": raw[cfg.flags_column].fillna("").astype(str).str.strip(),
        }
    )
    return df


def validate(df: pd.DataFrame, *, expect_full: bool = True) -> None:
    """Assert the label sets the whole study depends on. Trust the data, not the dataset card."""
    intents = set(df["intent"].unique())
    cats = set(df["category"].unique())
    if intents - EXPECTED_INTENTS:
        raise ValueError(f"unexpected intents: {sorted(intents - EXPECTED_INTENTS)}")
    if cats - EXPECTED_CATEGORIES:
        raise ValueError(f"unexpected categories: {sorted(cats - EXPECTED_CATEGORIES)}")
    if expect_full:
        if len(df) != EXPECTED_ROWS:
            raise ValueError(f"expected {EXPECTED_ROWS} rows, got {len(df)}")
        if intents != EXPECTED_INTENTS or cats != EXPECTED_CATEGORIES:
            raise ValueError("label sets differ from the pinned revision")
    if df["text"].eq("").any():
        raise ValueError("empty ticket text found")
    if not df.groupby("intent")["category"].nunique().eq(1).all():
        raise ValueError("an intent maps to more than one category")


def load(cfg: DatasetCfg, revision: str | None = None) -> LoadedDataset:
    path, rev = download(cfg, revision)
    raw = pd.read_csv(path)
    df = normalise_frame(raw, cfg)
    validate(df)
    return LoadedDataset(frame=df, revision=rev, sha256=file_sha256(path))


def intent_to_category(df: pd.DataFrame) -> dict[str, str]:
    pairs = df[["intent", "category"]].drop_duplicates()
    return dict(zip(pairs["intent"], pairs["category"], strict=True))
