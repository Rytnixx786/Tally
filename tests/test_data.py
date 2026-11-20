from __future__ import annotations

import pandas as pd
import pytest

from tally.config import TenantsCfg
from tally.data import bitext, dedup, split, tenants
from tests.conftest import synthetic_tickets


def test_normalise_collapses_placeholders_and_punctuation() -> None:
    assert dedup.normalise("Where's my {{Order Number}}?!") == "where s my slot"
    assert dedup.normalise("Café  ÉTÉ") == "cafe ete"


def test_jaccard_bounds() -> None:
    assert dedup.jaccard(set(), set()) == 1.0
    assert dedup.jaccard({"a"}, {"b"}) == 0.0
    assert dedup.jaccard({"a", "b"}, {"a"}) == 0.5


def test_cluster_merges_near_duplicates_only() -> None:
    texts = [
        "i want to cancel my order number 12345 please",
        "I want to cancel my order number 12345, please!",
        "how do i reset the password for my account",
        "where can i leave a review about your service",
    ]
    ids = dedup.cluster(texts, threshold=0.8)
    assert ids[0] == ids[1]
    assert len({ids[0], ids[2], ids[3]}) == 3


def test_cluster_is_transitive_union() -> None:
    texts = ["abcdefghij klmnop", "abcdefghij klmnop", "zzzz yyyy xxxx"]
    ids = dedup.cluster(texts, threshold=0.9)
    assert ids[0] == ids[1] != ids[2]


def test_validate_rejects_unknown_labels() -> None:
    df = pd.DataFrame(
        {"ticket_id": ["a"], "text": ["x"], "intent": ["fly_to_mars"], "category": ["ORDER"], "flags": [""]}
    )
    with pytest.raises(ValueError, match="unexpected intents"):
        bitext.validate(df, expect_full=False)


def test_validate_rejects_intent_with_two_categories() -> None:
    df = pd.DataFrame(
        {
            "ticket_id": ["a", "b"],
            "text": ["x", "y"],
            "intent": ["track_order", "track_order"],
            "category": ["ORDER", "REFUND"],
            "flags": ["", ""],
        }
    )
    with pytest.raises(ValueError, match="more than one category"):
        bitext.validate(df, expect_full=False)


def test_tenant_assignment_is_deterministic_and_respects_zero_weights(tenants_cfg: TenantsCfg) -> None:
    a = tenants.assign_tenant("bx-00001", "ORDER", tenants_cfg)
    assert a == tenants.assign_tenant("bx-00001", "ORDER", tenants_cfg)
    raw = tenants_cfg.model_dump()
    for t in raw["tenants"].values():
        t["weights"]["ORDER"] = 0.0
    raw["tenants"]["support"]["weights"]["ORDER"] = 1.0
    only_support = TenantsCfg.model_validate(raw)
    assert {tenants.assign_tenant(f"bx-{i:05d}", "ORDER", only_support) for i in range(50)} == {"support"}


def test_tenant_mix_follows_weights(tenants_cfg: TenantsCfg) -> None:
    got = [tenants.assign_tenant(f"bx-{i:05d}", "PAYMENT", tenants_cfg) for i in range(3000)]
    share_billing = got.count("billing") / len(got)
    w = {k: v.weights["PAYMENT"] for k, v in tenants_cfg.tenants.items()}
    assert abs(share_billing - w["billing"] / sum(w.values())) < 0.03


def test_tenant_with_no_weight_anywhere_fails(tenants_cfg: TenantsCfg) -> None:
    with pytest.raises(ValueError, match="no tenant"):
        tenants.assign_tenant("bx-1", "NOT_A_CATEGORY", tenants_cfg)


def test_split_keeps_clusters_together_and_hits_ratios(exp_cfg) -> None:  # type: ignore[no-untyped-def]
    base = synthetic_tickets(per_intent=60).drop(columns=["split", "cluster_id"])
    res = split.build(base, exp_cfg, dataset_revision="test", dataset_sha256="0" * 64)
    sizes = res.frame["split"].value_counts(normalize=True)
    assert abs(sizes["dev"] - 0.6) < 0.08 and abs(sizes["test"] - 0.2) < 0.08
    assert res.frame.groupby("cluster_id")["split"].nunique().max() == 1
    assert set(res.manifest["splits"]) == {"dev", "tune", "test"}


def test_verify_detects_a_changed_split(exp_cfg) -> None:  # type: ignore[no-untyped-def]
    base = synthetic_tickets(per_intent=60).drop(columns=["split", "cluster_id"])
    res = split.build(base, exp_cfg, dataset_revision="t", dataset_sha256="0" * 64)
    assert split.verify(res, res.manifest) == []
    tampered = {
        **res.manifest,
        "splits": {**res.manifest["splits"], "test": {**res.manifest["splits"]["test"], "sha256": "x" * 64}},
    }
    assert any("test" in p for p in split.verify(res, tampered))


def test_ids_sha256_is_order_independent() -> None:
    assert split.ids_sha256(["b", "a"]) == split.ids_sha256(["a", "b"])
