from __future__ import annotations

import copy

import pytest
import yaml
from pydantic import ValidationError

from tally.config import (
    CONFIG_DIR,
    CostModelCfg,
    ExperimentCfg,
    RoiCfg,
    TenantsCfg,
)


def _raw(name: str) -> dict:  # type: ignore[type-arg]
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def test_all_configs_validate(exp_cfg, cost_cfg, roi_cfg, tenants_cfg) -> None:  # type: ignore[no-untyped-def]
    assert exp_cfg.split.ratios == {"dev": 0.6, "tune": 0.2, "test": 0.2}
    assert set(tenants_cfg.tenants) == {"support", "billing", "sales"}
    assert cost_cfg.cache_hit_cost_usd > 0
    assert roi_cfg.tickets_per_month > 0


def test_ratios_must_sum_to_one() -> None:
    raw = _raw("experiment.yaml")
    raw["split"]["ratios"]["test"] = 0.3
    with pytest.raises(ValidationError, match="sum to 1"):
        ExperimentCfg.model_validate(raw)


def test_prompt_selection_can_never_use_test() -> None:
    raw = _raw("experiment.yaml")
    raw["prompts"]["selection_split"] = "test"
    with pytest.raises(ValidationError, match="never use the test split"):
        ExperimentCfg.model_validate(raw)


def test_unknown_keys_are_rejected() -> None:
    raw = _raw("experiment.yaml")
    raw["surprise"] = 1
    with pytest.raises(ValidationError):
        ExperimentCfg.model_validate(raw)


def test_tenant_ids_follow_contract_slug() -> None:
    raw = _raw("tenants.yaml")
    raw["tenants"]["Bad Tenant"] = copy.deepcopy(raw["tenants"]["support"])
    with pytest.raises(ValidationError, match="CONTRACTS"):
        TenantsCfg.model_validate(raw)


def test_price_book_fails_closed(cost_cfg: CostModelCfg) -> None:
    assert cost_cfg.price_for("openai", "gpt-4o-mini").input_per_mtok == 0.15
    with pytest.raises(KeyError):
        cost_cfg.price_for("openai", "a-model-nobody-priced")


def test_roi_base_values_must_be_in_sensitivity_grid() -> None:
    raw = _raw("roi.yaml")
    raw["wage_usd_per_hour"] = 7.77
    with pytest.raises(ValidationError, match="sensitivity grid"):
        RoiCfg.model_validate(raw)


def test_fct_requests_decision_id_tests_allow_non_triage_tenants() -> None:
    from tally.config import ROOT

    doc = yaml.safe_load((ROOT / "dbt" / "models" / "marts" / "_marts.yml").read_text(encoding="utf-8"))
    models = {m["name"]: m for m in doc["models"]}
    col = next(c for c in models["fct_requests"]["columns"] if c["name"] == "decision_id")
    for t in col["data_tests"]:
        assert isinstance(t, dict) and "feature = 'ticket_triage'" in str(t)
