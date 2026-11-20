"""Prompt registry: versioned Jinja templates, few-shot selection from the dev split only."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml
from jinja2 import Environment, StrictUndefined

from tally.config import CONFIG_DIR

PROMPT_DIR = CONFIG_DIR / "prompts"
FEATURE = "ticket_triage"


@dataclass(frozen=True)
class PromptVersion:
    version: str
    template_path: Path
    n_fewshot: int
    description: str

    @property
    def prompt_hash(self) -> str:
        """sha256 of the unrendered template, first 12 hex chars (CONTRACTS §1)."""
        return hashlib.sha256(self.template_path.read_bytes()).hexdigest()[:12]


REGISTRY: dict[str, PromptVersion] = {
    "ticket_triage.zero_shot.v1": PromptVersion(
        "ticket_triage.zero_shot.v1",
        PROMPT_DIR / "ticket_triage.zero_shot.v1.jinja",
        0,
        "Label list with one-line descriptions, no examples",
    ),
    "ticket_triage.fewshot_static.v3": PromptVersion(
        "ticket_triage.fewshot_static.v3",
        PROMPT_DIR / "ticket_triage.fewshot_static.v3.jinja",
        2,
        "Label list + a fixed set of 2 dev-split examples per intent (cache-friendly prefix)",
    ),
}


@lru_cache(maxsize=1)
def intent_descriptions() -> dict[str, str]:
    data = yaml.safe_load((PROMPT_DIR / "intents.yaml").read_text(encoding="utf-8"))
    out: dict[str, str] = data["intents"]
    return out


def select_fewshot(dev: pd.DataFrame, per_intent: int, seed: int = 0) -> list[dict[str, str]]:
    """Deterministic examples: per intent, the dev tickets with the smallest hash(seed, ticket_id).
    Only ever called with the dev split (enforced by tests/test_leakage.py)."""
    if per_intent <= 0:
        return []
    if set(dev["split"].unique()) != {"dev"}:
        raise ValueError("few-shot examples must come from the dev split only")
    keyed = dev.assign(
        _k=[hashlib.sha256(f"{seed}:{t}".encode()).hexdigest() for t in dev["ticket_id"]]
    ).sort_values(["intent", "_k"])
    picked = keyed.groupby("intent", sort=True).head(per_intent)
    return [
        {"ticket_id": str(r["ticket_id"]), "text": str(r["text"]), "intent": str(r["intent"])}
        for r in picked.to_dict("records")
    ]


_env = Environment(undefined=StrictUndefined, keep_trailing_newline=False, autoescape=False)


def render_system(version: str, examples: list[dict[str, str]]) -> str:
    pv = REGISTRY[version]
    tmpl = _env.from_string(pv.template_path.read_text(encoding="utf-8"))
    intents = sorted(intent_descriptions().items())
    return tmpl.render(intents=intents, examples=examples if pv.n_fewshot else []).strip()


def build_messages(system: str, ticket_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Message: {ticket_text}\nLabel:"},
    ]
