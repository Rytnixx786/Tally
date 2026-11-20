"""Experiment harness: runs setups A-D on the sealed splits and writes run artifacts.

Run artifacts (reports/runs/<run_id>/) are the hand-off to the warehouse:
  decisions.parquet  one row per (setup, split, arm, ticket)
  calls.parquet      one row per model call (cost included)
  manifest.json      everything needed to explain or reproduce the run
  *.json             selection, calibration, tau sweep, learning curve
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress

from tally import calibration, cost, hardware, ids, prompts
from tally.config import (
    CONFIG_DIR,
    CostModelCfg,
    ExperimentCfg,
    TenantsCfg,
)
from tally.data import bitext, split, tenants
from tally.llm.client import ChatClient, Endpoint
from tally.llm.record_replay import Mode, Recorder
from tally.llm.types import CallResult
from tally.paths import INTERIM_DIR, RECORDINGS_DIR, RUNS_DIR, SPLITS_DIR, load_dotenv
from tally.setups import a_tfidf, cascade
from tally.setups.llm import LlmOutcome, LlmSpec, make_request, run_batch

console = Console()
TICKETS_PATH = INTERIM_DIR / "tickets.parquet"
HARDWARE_DIR = RECORDINGS_DIR / "hardware"
PRIMARY = "primary"
PROMPT_SELECTION = "prompt_selection"


class SpendNotAllowedError(RuntimeError):
    pass


# ------------------------------------------------------------------ data


def prepare_tickets(exp: ExperimentCfg, ten: TenantsCfg, *, write_manifest: bool) -> pd.DataFrame:
    ds = bitext.load(exp.dataset)
    res = split.build(ds.frame, exp, dataset_revision=ds.revision, dataset_sha256=ds.sha256)
    df = res.frame.copy()
    df["tenant_id"] = tenants.assign(df, ten)
    if write_manifest:
        split.write(res)
    else:
        problems = split.verify(res, split.read_manifest())
        if problems:
            raise RuntimeError("split does not match the committed manifest: " + "; ".join(problems))
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TICKETS_PATH, index=False)
    return df


def load_tickets() -> pd.DataFrame:
    if not TICKETS_PATH.exists():
        raise FileNotFoundError("run `tally data` first")
    df = pd.read_parquet(TICKETS_PATH)
    manifest = split.read_manifest()
    for s in split.SPLITS:
        got = split.ids_sha256(df.loc[df["split"] == s, "ticket_id"].tolist())
        if got != manifest["splits"][s]["sha256"]:
            raise RuntimeError(f"tickets.parquet {s} split differs from data/splits/manifest.json")
    return df


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ------------------------------------------------------------------ run state


@dataclass
class RunOptions:
    run_id: str
    mode: Mode
    exp: ExperimentCfg
    ten: TenantsCfg
    cost_cfg: CostModelCfg
    setups: tuple[str, ...]
    allow_spend: bool = False
    cache_arm: bool = False  # also run B on test with the gateway semantic cache ON (needs Cinch restarted)
    limit: int | None = None  # smoke tests only: first N tickets per split (never for reported runs)


@dataclass
class Harness:
    opts: RunOptions
    tickets: pd.DataFrame
    decisions: list[dict[str, Any]] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        load_dotenv()
        self.recorder = Recorder(RECORDINGS_DIR / "responses", self.opts.mode)
        self.labels = tuple(sorted(self.tickets["intent"].unique()))
        self.cat_of = bitext.intent_to_category(self.tickets)
        dev = self.tickets[self.tickets["split"] == "dev"]
        self.fewshot = prompts.select_fewshot(dev, self.opts.exp.prompts.fewshot_per_intent)
        self.systems = {v: prompts.render_system(v, self.fewshot) for v in self.opts.exp.prompts.candidates}
        self.run_started = time.time()

    # ------------------------------------------------------------ helpers

    def split_df(self, name: str) -> pd.DataFrame:
        df = self.tickets[self.tickets["split"] == name].sort_values("ticket_id").reset_index(drop=True)
        if self.opts.limit is not None and name != "dev":
            df = df.head(self.opts.limit)
        return df

    def _decision_row(
        self,
        *,
        setup: str,
        split_name: str,
        arm: str,
        prompt_version: str,
        t: Any,
        label: str | None,
        valid: bool,
        confidence: float | None,
        escalated: bool,
        call_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        did = call_rows[0]["decision_id"]
        ok_all = all(c["http_status"] == 200 for c in call_rows)
        final = call_rows[-1]
        correct = bool(ok_all and valid and label == t.intent)
        started = min(c["started_at"] for c in call_rows)
        latency = float(sum(c["latency_ms"] for c in call_rows))
        return {
            "run_id": self.opts.run_id,
            "decision_id": did,
            "setup": setup,
            "split": split_name,
            "arm": arm,
            "prompt_version": prompt_version,
            "ticket_id": t.ticket_id,
            "tenant_id": t.tenant_id,
            "flags": t.flags,
            "gold_intent": t.intent,
            "gold_category": t.category,
            "predicted_intent": label,
            "predicted_category": self.cat_of.get(label) if label else None,
            "is_correct": correct,
            "valid": bool(valid and ok_all),
            "confidence": confidence,
            "calibrated_confidence": None,
            "escalated": escalated,
            "n_calls": len(call_rows),
            "final_request_id": final["request_id"],
            "total_cost_usd": float(sum(c["cost_usd"] for c in call_rows)),
            "total_latency_ms": latency,
            "decided_at": started + latency / 1000.0,
        }

    def _call_row(
        self,
        *,
        decision_id: str,
        call_index: int,
        setup: str,
        split_name: str,
        arm: str,
        prompt_version: str,
        tenant_id: str,
        res: CallResult,
        label: str | None,
        confidence: float | None,
        is_escalation: bool = False,
        reused: bool = False,
        cost_usd: float = 0.0,
        cost_method: str = "",
        server: str | None = None,
        gen_ai_provider: str | None = None,
    ) -> dict[str, Any]:
        return {
            "request_id": ids.request_id(decision_id, call_index),
            "decision_id": decision_id,
            "run_id": self.opts.run_id,
            "call_index": call_index,
            "setup": setup,
            "split": split_name,
            "arm": arm,
            "prompt_version": prompt_version,
            "tenant_id": tenant_id,
            "provider": gen_ai_provider or res.provider,
            "recorder_provider": res.provider,
            "model": res.model_requested,
            "model_response": res.model_response,
            "response_id": res.response_id,
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "cached_input_tokens": res.cached_input_tokens,
            "latency_ms": res.latency_ms,
            "started_at": res.started_at,
            "http_status": res.http_status,
            "error": res.error,
            "finish_reason": res.finish_reason,
            "cache_status": res.cache_status or "BYPASS",
            "billed_usd": res.billed_usd,
            "confidence": confidence,
            "validation_passed": label is not None and res.ok,
            "predicted_intent": label,
            "is_escalation": is_escalation,
            "reused_response": reused,
            "replayed": res.replayed,
            "cost_usd": cost_usd,
            "cost_method": cost_method,
            "server_address": server,
            "request_key": res.request_key,
        }

    # ------------------------------------------------------------ setup A

    def run_a(self) -> None:
        exp = self.opts.exp
        dev, tune, test = self.split_df("dev"), self.split_df("tune"), self.split_df("test")
        console.print("[bold]A_tfidf[/]: fitting on dev, choosing C on tune")
        fit = a_tfidf.fit_select(dev, tune, exp.setups.A_tfidf.c_grid, seed=exp.split.seed % 1000)
        self.artifacts["a_selection"] = {"chosen_c": fit.c, "tune_accuracy_by_c": fit.tune_accuracy_by_c}
        for split_name, df in (("tune", tune), ("test", test)):
            preds = a_tfidf.predict(fit.model, df)
            hw_path = HARDWARE_DIR / f"A_tfidf__{split_name}.json"
            if self.opts.mode in ("replay", "auto") and hw_path.exists():
                spt = hardware.load(hw_path).wall_seconds / max(len(df), 1)
            else:
                spt = preds.seconds_per_ticket
                hardware.save(
                    hardware.HardwareRecord(
                        "A_tfidf", split_name, "-", "", len(df), spt * len(df), 1, None, None, None, 0
                    ),
                    hw_path,
                )
            per_call_cost = cost.cpu_cost(self.opts.cost_cfg, spt)
            for i, t in enumerate(df.itertuples()):
                did = ids.decision_id(self.opts.run_id, "A_tfidf", split_name, str(t.ticket_id))
                res = CallResult(
                    request_key="sklearn",
                    provider="sklearn",
                    model_requested=a_tfidf.MODEL_NAME,
                    model_response=a_tfidf.MODEL_NAME,
                    response_id=None,
                    content=preds.labels[i],
                    finish_reason="stop",
                    input_tokens=0,
                    output_tokens=0,
                    cached_input_tokens=0,
                    latency_ms=spt * 1000,
                    started_at=self.run_started + i * spt,
                    http_status=200,
                )
                call = self._call_row(
                    decision_id=did,
                    call_index=0,
                    setup="A_tfidf",
                    split_name=split_name,
                    arm=PRIMARY,
                    prompt_version="none",
                    tenant_id=str(t.tenant_id),
                    res=res,
                    label=preds.labels[i],
                    confidence=float(preds.confidence[i]),
                    cost_usd=per_call_cost,
                    cost_method="cpu_seconds",
                )
                self.calls.append(call)
                self.decisions.append(
                    self._decision_row(
                        setup="A_tfidf",
                        split_name=split_name,
                        arm=PRIMARY,
                        prompt_version="none",
                        t=t,
                        label=preds.labels[i],
                        valid=True,
                        confidence=float(preds.confidence[i]),
                        escalated=False,
                        call_rows=[call],
                    )
                )
        console.print("[bold]A_tfidf[/]: learning curve on test")
        self.artifacts["learning_curve"] = a_tfidf.learning_curve(
            dev, test, exp.setups.A_tfidf.learning_curve_per_intent, fit.c, seed=exp.split.seed % 1000
        )

    # ------------------------------------------------------------ LLM setups

    def _spec_b(self, variant: str = "") -> LlmSpec:
        b = self.opts.exp.setups.B_local
        return LlmSpec("cinch", b.model, b.constraint, b.max_tokens, b.top_logprobs, variant)

    def _spec_c(self) -> LlmSpec:
        c = self.opts.exp.setups.C_hosted
        constraint = "json_schema" if c.provider in ("openai", "gemini") else "none"
        return LlmSpec(c.provider, c.model, constraint, c.max_tokens, 0)

    def _headers(self, request_id: str, tenant_id: str, prompt_version: str) -> dict[str, str]:
        return {
            "X-Request-ID": request_id,
            "X-Tenant-ID": tenant_id,
            "X-Team-ID": "tally",
            "X-Feature": prompts.FEATURE,
            "X-Prompt-Version": prompt_version,
            "X-Prompt-Compaction": "false",  # Cinch compresses prompts by default; that would change the task
        }

    def _run_llm(
        self,
        setup: str,
        spec: LlmSpec,
        endpoint: Endpoint,
        df: pd.DataFrame,
        split_name: str,
        prompt_version: str,
        arm: str,
        concurrency: int,
    ) -> list[LlmOutcome]:
        system = self.systems[prompt_version]
        dids = [
            ids.decision_id(self.opts.run_id, setup, split_name, t, f"{prompt_version}|{arm}")
            for t in df["ticket_id"]
        ]
        reqs = [make_request(spec, system, text, self.labels) for text in df["text"]]
        heads = [
            self._headers(ids.request_id(d, 0), ten, prompt_version)
            for d, ten in zip(dids, df["tenant_id"], strict=True)
        ]
        hw_path = HARDWARE_DIR / f"{setup}__{split_name}__{prompt_version}__{arm}.json"

        async def go(progress_cb: Callable[[int], None]) -> list[LlmOutcome]:
            async with ChatClient(endpoint, self.recorder) as client:
                return await run_batch(
                    client, reqs, heads, self.labels, concurrency=concurrency, progress=progress_cb
                )

        console.print(
            f"[bold]{setup}[/] {split_name} {prompt_version} [{arm}] n={len(df)} mode={self.recorder.mode}"
        )
        with Progress(console=console, transient=True) as prog:
            task = prog.add_task(f"{setup}/{split_name}", total=len(df))
            with hardware.PowerSampler() as sampler:
                outcomes = asyncio.run(go(lambda k: prog.advance(task, k)))
        n_live = sum(not o.result.replayed for o in outcomes)
        if n_live == len(outcomes) and len(outcomes) > 0:
            rec = hardware.HardwareRecord(
                setup,
                split_name,
                prompt_version,
                arm,
                len(df),
                sampler.wall_seconds,
                concurrency,
                sampler.gpu_name if setup == "B_local" else None,
                float(np.mean(sampler.samples)) if sampler.samples and setup == "B_local" else None,
                float(np.max(sampler.samples)) if sampler.samples and setup == "B_local" else None,
                len(sampler.samples) if setup == "B_local" else 0,
            )
            hardware.save(rec, hw_path)
        elif n_live:
            console.print(
                f"[yellow]{n_live} of {len(outcomes)} calls ran live; keeping the earlier hardware record[/]"
            )

        costs, method = self._llm_costs(setup, outcomes, hw_path)
        for i, (t, o) in enumerate(zip(df.itertuples(), outcomes, strict=True)):
            call = self._call_row(
                decision_id=dids[i],
                call_index=0,
                setup=setup,
                split_name=split_name,
                arm=arm,
                prompt_version=prompt_version,
                tenant_id=str(t.tenant_id),
                res=o.result,
                label=o.label,
                confidence=o.confidence,
                cost_usd=costs[i],
                cost_method=method,
                server=endpoint.base_url,
                gen_ai_provider="vllm" if setup == "B_local" else spec.provider,
            )
            self.calls.append(call)
            self.decisions.append(
                self._decision_row(
                    setup=setup,
                    split_name=split_name,
                    arm=arm,
                    prompt_version=prompt_version,
                    t=t,
                    label=o.label,
                    valid=o.valid,
                    confidence=o.confidence,
                    escalated=False,
                    call_rows=[call],
                )
            )
        return outcomes

    def _llm_costs(self, setup: str, outcomes: list[LlmOutcome], hw_path: Path) -> tuple[list[float], str]:
        cfg = self.opts.cost_cfg
        if setup == "C_hosted":
            c = self.opts.exp.setups.C_hosted
            return (
                [
                    cost.hosted_cost(
                        cfg,
                        c.provider,
                        c.model,
                        o.result.input_tokens,
                        o.result.output_tokens,
                        o.result.cached_input_tokens,
                    )
                    if o.result.ok
                    else 0.0
                    for o in outcomes
                ],
                "price_book",
            )
        if not hw_path.exists():
            # A non-primary arm (e.g. cache_on) serves misses on the same GPU at the same per-request cost,
            # so it reuses the primary arm's measured throughput. Hits are priced separately below.
            primary_path = hw_path.with_name(hw_path.name.rsplit("__", 1)[0] + f"__{PRIMARY}.json")
            if not primary_path.exists():
                raise FileNotFoundError(f"missing throughput record {hw_path}; run B live once to measure it")
            hw_path = primary_path
        tp = hardware.load(hw_path).throughput()
        mean = cost.local_cost_per_request_mean(cfg, tp)
        served = [o.result.ok and o.result.cache_status != "HIT" for o in outcomes]
        weights = [
            float(o.result.input_tokens + o.result.output_tokens) if s else 0.0
            for o, s in zip(outcomes, served, strict=True)
        ]
        total = mean * sum(served)
        shares = cost.apportion(total, weights) if any(served) else [0.0] * len(outcomes)
        out = [
            cfg.cache_hit_cost_usd if (o.result.ok and o.result.cache_status == "HIT") else sh
            for o, sh in zip(outcomes, shares, strict=True)
        ]
        return out, f"{cfg.method}_token_weighted"

    def _estimate_hosted_spend(self, df: pd.DataFrame, prompt_version: str) -> float:
        c = self.opts.exp.setups.C_hosted
        sys_tokens = len(self.systems[prompt_version]) / 4.0
        in_tok = sum(sys_tokens + len(t) / 4.0 + 12 for t in df["text"])
        return cost.hosted_cost(self.opts.cost_cfg, c.provider, c.model, int(in_tok), 8 * len(df))

    # ------------------------------------------------------------ orchestration

    def run(self) -> None:
        exp = self.opts.exp
        want = set(self.opts.setups)
        if "A_tfidf" in want:
            self.run_a()
        if not want & {"B_local", "C_hosted", "D_cascade"}:
            return

        b_end = Endpoint.for_provider(
            "cinch", base_url_env=exp.setups.B_local.base_url_env, api_key_env=exp.setups.B_local.api_key_env
        )
        tune, test = self.split_df("tune"), self.split_df("test")

        # Prompt selection on tune with the local model (free at the margin); never on test.
        b_tune: dict[str, list[LlmOutcome]] = {}
        for pv in exp.prompts.candidates:
            b_tune[pv] = self._run_llm(
                "B_local",
                self._spec_b(),
                b_end,
                tune,
                "tune",
                pv,
                PROMPT_SELECTION,
                exp.setups.B_local.concurrency,
            )
        acc = {
            pv: float(np.mean([o.label == g for o, g in zip(outs, tune["intent"], strict=True)]))
            for pv, outs in b_tune.items()
        }
        chosen = max(exp.prompts.candidates, key=lambda pv: (acc[pv], -exp.prompts.candidates.index(pv)))
        self.artifacts["prompt_selection"] = {
            "split": "tune",
            "setup": "B_local",
            "accuracy": acc,
            "chosen": chosen,
        }
        console.print(f"prompt selection on tune: {acc} -> [bold]{chosen}[/]")
        # The chosen prompt's tune run doubles as B's primary tune decisions.
        for d in self.decisions + self.calls:
            if d["setup"] == "B_local" and d["split"] == "tune" and d["prompt_version"] == chosen:
                d["arm"] = PRIMARY

        b_test = self._run_llm(
            "B_local", self._spec_b(), b_end, test, "test", chosen, PRIMARY, exp.setups.B_local.concurrency
        )

        if self.opts.cache_arm or exp.setups.B_local.cache_arm:
            outs = self._run_llm(
                "B_local",
                self._spec_b("cinch_cache_on"),
                b_end,
                test,
                "test",
                chosen,
                "cache_on",
                exp.setups.B_local.concurrency,
            )
            live = [o for o in outs if not o.result.replayed]
            if live and all(o.result.cache_status is None for o in live):
                raise RuntimeError(
                    "cache arm ran against a gateway with the semantic cache OFF (no X-Semantic-Cache-Status "
                    "header). Restart Cinch with CINCH_SEMANTIC_CACHE=true, delete the cache_on recordings, retry."
                )

        if not want & {"C_hosted", "D_cascade"}:
            self.artifacts["chosen_prompt"] = chosen
            return

        c_cfg = exp.setups.C_hosted
        c_end = Endpoint.for_provider(c_cfg.provider)
        c_out: dict[str, list[LlmOutcome]] = {}
        for split_name, df in (("tune", tune), ("test", test)):
            sub = df if c_cfg.max_tickets_per_split is None else df.head(c_cfg.max_tickets_per_split)
            if self.opts.mode != "replay":
                est = self._estimate_hosted_spend(sub, chosen)
                missing = sum(
                    self.recorder.get(make_request(self._spec_c(), self.systems[chosen], t, self.labels))
                    is None
                    for t in sub["text"]
                )
                est_missing = est * missing / max(len(sub), 1)
                console.print(
                    f"C_hosted {split_name}: {missing}/{len(sub)} not recorded, est. spend ${est_missing:.2f}"
                )
                if missing and not self.opts.allow_spend:
                    raise SpendNotAllowedError(
                        f"C_hosted would spend about ${est_missing:.2f}. Re-run with --confirm-spend "
                        "(or TALLY_ALLOW_SPEND=1) after checking the estimate."
                    )
                if est_missing > exp.budget.max_hosted_spend_usd:
                    raise SpendNotAllowedError(
                        f"estimate ${est_missing:.2f} exceeds budget.max_hosted_spend_usd"
                    )
                if missing and not c_end.api_key:
                    raise SpendNotAllowedError(f"no API key for provider {c_cfg.provider}")
            c_out[split_name] = self._run_llm(
                "C_hosted", self._spec_c(), c_end, sub, split_name, chosen, PRIMARY, c_cfg.concurrency
            )

        if "D_cascade" in want:
            self.run_d(chosen, {"tune": b_tune[chosen], "test": b_test}, c_out)
        self.artifacts["chosen_prompt"] = chosen

    # ------------------------------------------------------------ setup D

    def _inputs(
        self, split_name: str, b: list[LlmOutcome], c: list[LlmOutcome], temperature: float
    ) -> tuple[cascade.CascadeInputs, pd.DataFrame]:
        df = self.split_df(split_name).head(len(c))
        b = b[: len(c)]
        calls = pd.DataFrame(self.calls)
        prim = calls[(calls["split"] == split_name) & (calls["arm"] == PRIMARY)]
        b_cost = prim[prim["setup"] == "B_local"].set_index("decision_id")["cost_usd"]
        c_cost = prim[prim["setup"] == "C_hosted"].set_index("decision_id")["cost_usd"]
        pv = self.artifacts.get("_chosen")
        b_ids = [
            ids.decision_id(
                self.opts.run_id,
                "B_local",
                split_name,
                t,
                f"{pv}|{PROMPT_SELECTION if split_name == 'tune' else PRIMARY}",
            )
            for t in df["ticket_id"]
        ]
        c_ids = [
            ids.decision_id(self.opts.run_id, "C_hosted", split_name, t, f"{pv}|{PRIMARY}")
            for t in df["ticket_id"]
        ]
        gold = df["intent"].to_numpy()
        conf = np.array([o.confidence if o.confidence is not None else np.nan for o in b], dtype=float)
        cal = np.where(
            np.isnan(conf), np.nan, calibration.apply_temperature(np.nan_to_num(conf, nan=0.5), temperature)
        )
        inp = cascade.CascadeInputs(
            b_correct=np.array([o.label == g for o, g in zip(b, gold, strict=True)]),
            b_valid=np.array([o.valid for o in b]),
            b_conf_cal=cal,
            b_cost=b_cost.loc[b_ids].to_numpy(dtype=float),
            b_latency_ms=np.array([o.result.latency_ms for o in b]),
            c_correct=np.array([o.label == g for o, g in zip(c, gold, strict=True)]),
            c_cost=c_cost.loc[c_ids].to_numpy(dtype=float),
            c_latency_ms=np.array([o.result.latency_ms for o in c]),
        )
        return inp, df

    def run_d(self, chosen: str, b: dict[str, list[LlmOutcome]], c: dict[str, list[LlmOutcome]]) -> None:
        exp = self.opts.exp
        self.artifacts["_chosen"] = chosen
        tune = self.split_df("tune").head(len(c["tune"]))
        bt = b["tune"][: len(c["tune"])]
        conf = np.array([o.confidence if o.confidence is not None else 0.0 for o in bt], dtype=float)
        correct = np.array([o.label == g for o, g in zip(bt, tune["intent"], strict=True)])
        has = np.array([o.confidence is not None for o in bt])
        temp = calibration.fit_temperature(conf[has], correct[has])
        calib: dict[str, Any] = {"temperature": temp, "fit_split": "tune", "bins": exp.stats.ece_bins}
        for split_name in ("tune", "test"):
            bo = b[split_name][: len(c[split_name])]
            gold = self.split_df(split_name).head(len(bo))["intent"].to_numpy()
            p = np.array([o.confidence if o.confidence is not None else 0.0 for o in bo], dtype=float)
            y = np.array([o.label == g for o, g in zip(bo, gold, strict=True)])
            q = calibration.apply_temperature(p, temp)
            calib[split_name] = {
                "n": len(p),
                "ece_raw": calibration.ece(p, y, exp.stats.ece_bins),
                "ece_calibrated": calibration.ece(q, y, exp.stats.ece_bins),
                "reliability_raw": [vars(r) for r in calibration.reliability(p, y, exp.stats.ece_bins)],
                "reliability_calibrated": [
                    vars(r) for r in calibration.reliability(q, y, exp.stats.ece_bins)
                ],
            }
        self.artifacts["calibration"] = calib

        inp_tune, _ = self._inputs("tune", b["tune"], c["tune"], temp)
        points = cascade.sweep(inp_tune, exp.setups.D_cascade.tau_grid_step)
        c_acc_tune = float(inp_tune.c_correct.mean())
        best, why = cascade.choose_tau(points, c_acc_tune, exp.setups.D_cascade.min_relative_accuracy)
        tau = best.tau
        inp_test, _ = self._inputs("test", b["test"], c["test"], temp)
        self.artifacts["tau"] = {
            "chosen_tau": tau,
            "rule": why,
            "objective": exp.setups.D_cascade.objective,
            "min_relative_accuracy": exp.setups.D_cascade.min_relative_accuracy,
            "tune_sweep": [pt.as_dict() for pt in points],
            "test_sweep": [
                pt.as_dict() for pt in cascade.sweep(inp_test, exp.setups.D_cascade.tau_grid_step)
            ],
            "oracle": {
                "tune": cascade.oracle(inp_tune).as_dict(),
                "test": cascade.oracle(inp_test).as_dict(),
            },
        }
        console.print(f"D_cascade: T={temp:.3f}, tau={tau:.2f} ({why})")

        first_call: dict[str, dict[str, Any]] = {}
        for row in self.calls:
            if row["call_index"] == 0:
                first_call.setdefault(row["decision_id"], row)
        for split_name in ("tune", "test"):
            inp, df = self._inputs(split_name, b[split_name], c[split_name], temp)
            esc = cascade.escalate_mask(inp, tau)
            b_arm = PROMPT_SELECTION if split_name == "tune" else PRIMARY
            for i, t in enumerate(df.itertuples()):
                did = ids.decision_id(
                    self.opts.run_id, "D_cascade", split_name, str(t.ticket_id), f"{chosen}|{PRIMARY}"
                )
                srcs = [("B_local", b_arm, b[split_name][i], False)]
                if esc[i]:
                    srcs.append(("C_hosted", PRIMARY, c[split_name][i], True))
                rows = []
                for k, (src_setup, src_arm, o, is_esc) in enumerate(srcs):
                    src_did = ids.decision_id(
                        self.opts.run_id, src_setup, split_name, str(t.ticket_id), f"{chosen}|{src_arm}"
                    )
                    src = first_call[src_did]
                    row = self._call_row(
                        decision_id=did,
                        call_index=k,
                        setup="D_cascade",
                        split_name=split_name,
                        arm=PRIMARY,
                        prompt_version=chosen,
                        tenant_id=str(t.tenant_id),
                        res=o.result,
                        label=o.label,
                        confidence=o.confidence,
                        is_escalation=is_esc,
                        reused=True,
                        cost_usd=float(src["cost_usd"]),
                        cost_method=str(src["cost_method"]),
                        server=src["server_address"],
                        gen_ai_provider=str(src["provider"]),
                    )
                    rows.append(row)
                self.calls.extend(rows)
                final = c[split_name][i] if esc[i] else b[split_name][i]
                d = self._decision_row(
                    setup="D_cascade",
                    split_name=split_name,
                    arm=PRIMARY,
                    prompt_version=chosen,
                    t=t,
                    label=final.label,
                    valid=final.valid,
                    confidence=b[split_name][i].confidence,
                    escalated=bool(esc[i]),
                    call_rows=rows,
                )
                d["calibrated_confidence"] = None if np.isnan(inp.b_conf_cal[i]) else float(inp.b_conf_cal[i])
                self.decisions.append(d)

    # ------------------------------------------------------------ artifacts

    def write(self) -> Path:
        out = RUNS_DIR / self.opts.run_id
        out.mkdir(parents=True, exist_ok=True)
        dec = pd.DataFrame(self.decisions).sort_values(
            ["setup", "split", "arm", "prompt_version", "ticket_id"]
        )
        cal = pd.DataFrame(self.calls).sort_values(
            ["setup", "split", "arm", "prompt_version", "decision_id", "call_index"]
        )
        dec.to_parquet(out / "decisions.parquet", index=False)
        cal.to_parquet(out / "calls.parquet", index=False)
        for k, v in self.artifacts.items():
            if not k.startswith("_"):
                (out / f"{k}.json").write_text(
                    json.dumps(v, indent=2, default=float) + "\n", encoding="utf-8"
                )
        manifest = self.manifest(dec, cal)
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        console.print(f"wrote {out}")
        return out

    def manifest(self, dec: pd.DataFrame, cal: pd.DataFrame) -> dict[str, Any]:
        e = self.opts.exp
        models = (
            cal[cal["setup"].isin(["B_local", "C_hosted"])]
            .groupby("setup")["model_response"]
            .agg(lambda s: sorted({str(x) for x in s if x is not None}))
            .to_dict()
        )
        return {
            "run_id": self.opts.run_id,
            "mode": self.opts.mode,
            "setups": list(self.opts.setups),
            "config_sha": {p.name: file_sha(p) for p in sorted(CONFIG_DIR.glob("*.yaml"))},
            "prompt_hashes": {v: prompts.REGISTRY[v].prompt_hash for v in e.prompts.candidates},
            "split_manifest_sha": file_sha(SPLITS_DIR / "manifest.json"),
            "chosen_prompt": self.artifacts.get("chosen_prompt"),
            "chosen_tau": (self.artifacts.get("tau") or {}).get("chosen_tau"),
            "temperature": (self.artifacts.get("calibration") or {}).get("temperature"),
            "models_seen": models,
            "local_model": e.setups.B_local.model,
            "local_constraint": e.setups.B_local.constraint,
            "vllm_image": os.environ.get("TALLY_VLLM_IMAGE", "vllm/vllm-openai:v0.6.3.post1"),
            "hosted": {"provider": e.setups.C_hosted.provider, "model": e.setups.C_hosted.model},
            "cost_method": self.opts.cost_cfg.method,
            "n_decisions": len(dec),
            "n_calls": len(cal),
            "replayed_calls": int(cal["replayed"].sum()) if "replayed" in cal else 0,
        }
