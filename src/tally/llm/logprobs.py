"""Label confidence from token logprobs, renormalised over the label set (ADR-004).

At each generated position we keep only candidate tokens that extend the current prefix towards
*some* valid label, and renormalise the chosen token's probability over them. The label
confidence is the product of those per-position probabilities. This removes the mass that raw
(pre-mask) logprobs place on tokens the structured-output constraint would block, and it removes
the length bias of plain sequence probabilities across labels with different token counts.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from tally.llm.types import TokenLogprob


def parse_label(content: str, labels: Sequence[str]) -> str | None:
    """Exact label after trimming whitespace, quotes, a trailing period, or a JSON {"intent": ...}."""
    text = content.strip()
    if text.startswith("{"):
        import json

        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict) and isinstance(obj.get("intent"), str):
            text = obj["intent"]
        else:
            return None
    text = text.strip().strip("\"'`").rstrip(".").strip().lower()
    return text if text in set(labels) else None


def label_confidence(
    token_logprobs: Sequence[TokenLogprob] | None, labels: Sequence[str], chosen: str | None
) -> float | None:
    if not token_logprobs or chosen is None:
        return None
    label_set = list(labels)
    prefix = ""
    conf = 1.0
    for tl in token_logprobs:
        cands: dict[str, float] = dict(tl.top)
        cands[tl.token] = tl.logprob

        def extends(tok: str, prefix: str = prefix) -> bool:
            s = (prefix + tok).lstrip()
            return bool(s) and any(lab.startswith(s) for lab in label_set)

        valid = {tok: lp for tok, lp in cands.items() if extends(tok)}
        if tl.token not in valid:
            if not (prefix + tl.token).strip():  # leading whitespace token: neutral
                prefix += tl.token
                continue
            break  # tokens after the label (e.g. EOS-like text) carry no label information
        denom = sum(math.exp(lp) for lp in valid.values())
        conf *= math.exp(tl.logprob) / denom
        prefix += tl.token
        if prefix.strip() == chosen:
            break
    return min(max(conf, 0.0), 1.0)
