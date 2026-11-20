"""`tally` command line. `tally reproduce` is the one command that rebuilds everything."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import pandas as pd
import typer
from rich.console import Console

from tally.config import ROOT, SETUP_ORDER, load_cost_model, load_experiment, load_tenants
from tally.paths import RUNS_DIR, hf_home, load_dotenv

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Cost per correct outcome for LLM traffic."
)
console = Console()
DEFAULT_RUN = "main"


def _setups(arg: str) -> tuple[str, ...]:
    if arg == "all":  # every setup enabled in config/experiment.yaml
        cfg = load_experiment().setups
        return tuple(s for s in SETUP_ORDER if getattr(cfg, s).enabled)
    names = tuple(s.strip() for s in arg.split(","))
    full = {s.split("_")[0]: s for s in SETUP_ORDER} | {s: s for s in SETUP_ORDER}
    try:
        return tuple(full[n] for n in names)
    except KeyError as exc:
        raise typer.BadParameter(f"unknown setup {exc}; use A,B,C,D or all") from exc


@app.command()
def data(
    write_manifest: bool = typer.Option(False, help="Regenerate and overwrite data/splits/manifest.json."),
) -> None:
    """Download Bitext (pinned revision), cluster near-duplicates, split 60/20/20, assign tenants."""
    load_dotenv()
    os.environ.setdefault("HF_HOME", str(hf_home()))
    from tally.harness import prepare_tickets

    df = prepare_tickets(load_experiment(), load_tenants(), write_manifest=write_manifest)
    console.print(
        df.groupby("split").size().to_dict(), "tickets; tenants:", df["tenant_id"].value_counts().to_dict()
    )


@app.command()
def run(
    run_id: str = typer.Option(DEFAULT_RUN),
    mode: str = typer.Option(None, help="replay | live | auto (default: TALLY_MODE or replay)"),
    setups: str = typer.Option("all", help="Comma list of A,B,C,D, or 'all' (= enabled in experiment.yaml)."),
    confirm_spend: bool = typer.Option(False, "--confirm-spend", help="Allow paid hosted-API calls."),
    cache_arm: bool = typer.Option(False, help="Also run B on test with the gateway semantic cache ON."),
    limit: int = typer.Option(None, help="Smoke test only: first N tickets of tune/test."),
) -> None:
    """Run the setups on the sealed splits and write reports/runs/<run_id>/."""
    load_dotenv()
    from tally.harness import Harness, RunOptions, load_tickets
    from tally.llm.record_replay import Mode

    m = mode or os.environ.get("TALLY_MODE", "replay")
    if m not in ("replay", "live", "auto"):
        raise typer.BadParameter("mode must be replay, live or auto")
    allow = confirm_spend or os.environ.get("TALLY_ALLOW_SPEND") == "1"
    opts = RunOptions(
        run_id,
        cast(Mode, m),
        load_experiment(),
        load_tenants(),
        load_cost_model(),
        _setups(setups),
        allow,
        cache_arm,
        limit,
    )
    h = Harness(opts, load_tickets())
    h.run()
    h.write()


@app.command()
def publish(
    run_id: str = typer.Option(DEFAULT_RUN),
    otlp: str = typer.Option(None, help="OTLP/HTTP endpoint (collector :4318 or tally-ingest :8090)."),
    wait_s: int = typer.Option(180, help="Seconds to wait for spans to land in Postgres."),
) -> None:
    """Load run artifacts into raw.*, export the run's spans over OTLP, and wait for ingestion."""
    load_dotenv()
    from tally import telemetry, warehouse

    run_dir = RUNS_DIR / run_id
    endpoint = otlp or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    with warehouse.connect() as conn:
        warehouse.load_reference(conn, load_tenants(), load_cost_model())
        counts = warehouse.load_run(conn, run_dir)
    console.print(f"loaded {counts}")
    calls = pd.read_parquet(run_dir / "calls.parquet")
    decisions = pd.read_parquet(run_dir / "decisions.parquet")
    expected = len(calls) + len(decisions)
    sent = telemetry.emit(calls, decisions, endpoint)
    console.print(f"exported {sent} spans to {endpoint}")
    deadline = time.time() + wait_s
    while True:
        got = warehouse.count_spans(run_id)
        if got >= expected:
            console.print(f"[green]{got} spans in raw.otel_spans[/]")
            break
        if time.time() > deadline:
            console.print(f"[red]only {got}/{expected} spans arrived[/]")
            raise typer.Exit(1)
        time.sleep(2)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def dbt(ctx: typer.Context) -> None:
    """Run dbt in dbt/ with the repo's profile, e.g. `tally dbt build`."""
    load_dotenv()
    args = ctx.args or ["build"]
    exe = Path(sys.executable).with_name("dbt.exe" if os.name == "nt" else "dbt")
    cmd = [str(exe), *args, "--project-dir", str(ROOT / "dbt"), "--profiles-dir", str(ROOT / "dbt")]
    raise typer.Exit(subprocess.call(cmd))


@app.command()
def report(
    run_id: str = typer.Option(DEFAULT_RUN),
    source: str = typer.Option("warehouse", help="warehouse (marts) | artifacts (run parquet files)"),
) -> None:
    """Compute metrics with CIs, figures, the README headline and the case-study numbers."""
    load_dotenv()
    from tally.report import build

    out = build(run_id, source=source)
    console.print(f"report written to {out}")


@app.command()
def dashboard(port: int = 8501) -> None:
    """Launch the Streamlit dashboard."""
    exe = Path(sys.executable).with_name("streamlit.exe" if os.name == "nt" else "streamlit")
    raise typer.Exit(
        subprocess.call([str(exe), "run", str(ROOT / "dashboard" / "app.py"), "--server.port", str(port)])
    )


@app.command()
def reproduce(run_id: str = typer.Option(DEFAULT_RUN)) -> None:
    """Everything, end to end, in replay mode: data -> run -> publish -> dbt build -> report."""
    load_dotenv()
    os.environ["TALLY_MODE"] = "replay"
    steps = [
        ["data"],
        ["run", "--run-id", run_id, "--mode", "replay"],
        ["publish", "--run-id", run_id],
        ["dbt", "build"],
        ["report", "--run-id", run_id],
    ]
    for step in steps:
        console.rule(" ".join(step))
        rc = subprocess.call([sys.executable, "-m", "tally.cli", *step])
        if rc != 0:
            console.print(f"[red]step failed: {' '.join(step)}[/]")
            raise typer.Exit(rc)
    console.print("[green]reproduce: done[/]")


@app.command("db-init")
def db_init(admin_dsn: str = typer.Option(..., help="Superuser DSN of an existing Postgres.")) -> None:
    """Create roles, schemas and raw tables on a Postgres not started by docker compose."""
    from tally import warehouse

    warehouse.init_schema(admin_dsn)
    console.print("schema initialised")


if __name__ == "__main__":
    app()
