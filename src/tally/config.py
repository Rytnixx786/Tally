"""Typed configuration. Every YAML file in config/ is validated here before anything runs."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"

SetupName = Literal["A_tfidf", "B_local", "C_hosted", "D_cascade"]
SplitName = Literal["dev", "tune", "test"]
SETUP_ORDER: tuple[SetupName, ...] = ("A_tfidf", "B_local", "C_hosted", "D_cascade")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------- experiment


class DatasetCfg(_Strict):
    repo_id: str
    revision: str | None = None
    text_column: str
    label_column: str
    category_column: str
    flags_column: str


class DedupCfg(_Strict):
    method: Literal["minhash"] = "minhash"
    shingle_chars: int = Field(5, ge=2, le=12)
    num_perm: int = Field(128, ge=16)
    jaccard_threshold: float = Field(0.8, gt=0.0, lt=1.0)


class SplitCfg(_Strict):
    seed: int
    ratios: dict[SplitName, float]
    dedup: DedupCfg

    @field_validator("ratios")
    @classmethod
    def _sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        if set(v) != {"dev", "tune", "test"}:
            raise ValueError("ratios must define dev, tune and test")
        if abs(sum(v.values()) - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1")
        return v


class SetupACfg(_Strict):
    enabled: bool = True
    c_grid: list[float]
    learning_curve_per_intent: list[int]


class SetupBCfg(_Strict):
    enabled: bool = True
    base_url_env: str = "CINCH_BASE_URL"
    api_key_env: str = "CINCH_API_KEY"
    model: str
    constraint: Literal["guided_choice", "structured_outputs", "none"] = "guided_choice"
    top_logprobs: int = Field(20, ge=1, le=20)
    max_tokens: int = 16
    concurrency: int = Field(16, ge=1)
    cache_arm: bool = False


class SetupCCfg(_Strict):
    enabled: bool = True
    provider: Literal["openai", "gemini", "anthropic"]
    model: str
    max_tokens: int = 16
    concurrency: int = Field(8, ge=1)
    max_tickets_per_split: int | None = None


class SetupDCfg(_Strict):
    enabled: bool = True
    tau_grid_step: float = Field(0.02, gt=0.0, le=0.5)
    objective: Literal["min_cost_per_correct"] = "min_cost_per_correct"
    min_relative_accuracy: float = Field(0.98, gt=0.0, le=1.0)


class SetupsCfg(_Strict):
    A_tfidf: SetupACfg
    B_local: SetupBCfg
    C_hosted: SetupCCfg
    D_cascade: SetupDCfg


class PromptsCfg(_Strict):
    candidates: list[str]
    fewshot_per_intent: int = Field(2, ge=0, le=10)
    selection_split: SplitName = "tune"

    @field_validator("selection_split")
    @classmethod
    def _never_test(cls, v: str) -> str:
        if v == "test":
            raise ValueError("prompt selection must never use the test split")
        return v


class StatsCfg(_Strict):
    bootstrap_resamples: int = Field(2000, ge=100)
    bootstrap_seed: int = 7
    ci_level: float = Field(0.95, gt=0.5, lt=1.0)
    ece_bins: int = Field(15, ge=2)


class BudgetCfg(_Strict):
    max_hosted_spend_usd: float = Field(5.0, ge=0.0)


class ExperimentCfg(_Strict):
    dataset: DatasetCfg
    split: SplitCfg
    setups: SetupsCfg
    prompts: PromptsCfg
    stats: StatsCfg
    budget: BudgetCfg


# ---------------------------------------------------------------- tenants


class TenantCfg(_Strict):
    display_name: str
    owner_project: str
    monthly_budget_usd: float = Field(ge=0.0)
    weights: dict[str, float]

    @field_validator("weights")
    @classmethod
    def _positive(cls, v: dict[str, float]) -> dict[str, float]:
        if any(w < 0 for w in v.values()):
            raise ValueError("tenant weights must be non-negative")
        return v


class TenantsCfg(_Strict):
    seed: int
    tenants: dict[str, TenantCfg]

    @field_validator("tenants")
    @classmethod
    def _slug(cls, v: dict[str, TenantCfg]) -> dict[str, TenantCfg]:
        import re

        for k in v:
            if not re.fullmatch(r"[a-z0-9-]{2,32}", k):
                raise ValueError(f"tenant id {k!r} must match [a-z0-9-]{{2,32}} (CONTRACTS §1)")
        return v


# ---------------------------------------------------------------- cost


class LocalGpuCfg(_Strict):
    card: str
    purchase_usd: float = Field(gt=0)
    lifetime_years: float = Field(gt=0)
    utilisation: float = Field(gt=0, le=1)
    board_power_w: float = Field(gt=0)
    system_overhead_w: float = Field(ge=0)
    electricity_usd_per_kwh: float = Field(ge=0)
    alt_cloud_gpu_usd_per_hour: float = Field(gt=0)


class PriceEntry(_Strict):
    provider: str
    model: str
    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)
    # Price for prompt tokens served from the provider's prompt cache. None = billed at the full input price.
    cached_input_per_mtok: float | None = Field(default=None, ge=0)
    checked_on: str
    source: str


class CpuCfg(_Strict):
    usd_per_hour: float = Field(ge=0)


class CostModelCfg(_Strict):
    local_gpu: LocalGpuCfg
    method: Literal["owned_gpu", "cloud_gpu_hour"] = "owned_gpu"
    cpu: CpuCfg
    cache_hit_cost_usd: float = Field(gt=0)
    price_book: list[PriceEntry]

    def price_for(self, provider: str, model: str) -> PriceEntry:
        for p in self.price_book:
            if p.provider == provider and p.model == model:
                return p
        # Fail closed: an unknown hosted model must never be costed at $0.
        raise KeyError(f"no price-book entry for provider={provider!r} model={model!r}")


# ---------------------------------------------------------------- roi


class RoiSensitivity(_Strict):
    minutes_saved_per_correct: list[float]
    wage_usd_per_hour: list[float]
    misroute_penalty_minutes: list[float]


class RoiCfg(_Strict):
    minutes_saved_per_correct: float = Field(ge=0)
    wage_usd_per_hour: float = Field(ge=0)
    misroute_penalty_minutes: float = Field(ge=0)
    tickets_per_month: int = Field(gt=0)
    sensitivity: RoiSensitivity

    @model_validator(mode="after")
    def _base_in_grid(self) -> RoiCfg:
        s = self.sensitivity
        if self.minutes_saved_per_correct not in s.minutes_saved_per_correct:
            raise ValueError("base minutes_saved_per_correct must appear in its sensitivity grid")
        if self.wage_usd_per_hour not in s.wage_usd_per_hour:
            raise ValueError("base wage_usd_per_hour must appear in its sensitivity grid")
        if self.misroute_penalty_minutes not in s.misroute_penalty_minutes:
            raise ValueError("base misroute_penalty_minutes must appear in its sensitivity grid")
        return self


# ---------------------------------------------------------------- loaders


def _load(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def load_experiment(path: Path | None = None) -> ExperimentCfg:
    return ExperimentCfg.model_validate(_load(path or CONFIG_DIR / "experiment.yaml"))


def load_tenants(path: Path | None = None) -> TenantsCfg:
    return TenantsCfg.model_validate(_load(path or CONFIG_DIR / "tenants.yaml"))


def load_cost_model(path: Path | None = None) -> CostModelCfg:
    return CostModelCfg.model_validate(_load(path or CONFIG_DIR / "cost_model.yaml"))


def load_roi(path: Path | None = None) -> RoiCfg:
    return RoiCfg.model_validate(_load(path or CONFIG_DIR / "roi.yaml"))


@lru_cache(maxsize=1)
def default_experiment() -> ExperimentCfg:
    return load_experiment()
