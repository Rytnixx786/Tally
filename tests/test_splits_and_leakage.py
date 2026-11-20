"""The sealed-test-split guarantees (AGENTS.md: never weaken these tests)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tally import prompts
from tally.data import split
from tally.harness import TICKETS_PATH
from tally.paths import RUNS_DIR, SPLITS_DIR
from tally.setups import a_tfidf

MANIFEST = SPLITS_DIR / "manifest.json"


def test_committed_manifest_is_complete() -> None:
    m = split.read_manifest()
    assert m["dataset"]["revision"] and len(m["dataset"]["sha256"]) == 64
    assert m["n_rows"] == 26_872
    assert sum(m["splits"][s]["n"] for s in split.SPLITS) == m["n_rows"]
    for s in split.SPLITS:
        assert len(m["splits"][s]["intent_counts"]) == 27


@pytest.mark.skipif(not TICKETS_PATH.exists(), reason="run `tally data` first")
def test_local_tickets_match_committed_manifest() -> None:
    from tally.harness import load_tickets

    df = load_tickets()  # raises if any split differs from the manifest
    assert df.groupby("cluster_id")["split"].nunique().max() == 1


@pytest.mark.skipif(not TICKETS_PATH.exists(), reason="run `tally data` first")
def test_split_id_files_match_manifest() -> None:
    m = split.read_manifest()
    for s in split.SPLITS:
        ids = (SPLITS_DIR / f"{s}_ids.txt").read_text(encoding="utf-8").split()
        assert split.ids_sha256(ids) == m["splits"][s]["sha256"]


def test_fewshot_refuses_non_dev_rows(tickets: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="dev split only"):
        prompts.select_fewshot(tickets, 2)


def test_fewshot_examples_are_dev_only_and_deterministic(tickets: pd.DataFrame) -> None:
    dev = tickets[tickets["split"] == "dev"]
    ex = prompts.select_fewshot(dev, 2)
    assert {e["ticket_id"] for e in ex} <= set(dev["ticket_id"])
    assert ex == prompts.select_fewshot(dev, 2)
    assert len(ex) == 2 * dev["intent"].nunique()


def test_tfidf_selection_never_sees_test(tickets: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[pd.Series] = []
    real = a_tfidf.build_pipeline

    def spy(c: float, seed: int = 0):  # type: ignore[no-untyped-def]
        pipe = real(c, seed)
        fit = pipe.fit

        def record(x, y):  # type: ignore[no-untyped-def]
            seen.append(pd.Series(x))
            return fit(x, y)

        pipe.fit = record  # type: ignore[method-assign]
        return pipe

    monkeypatch.setattr(a_tfidf, "build_pipeline", spy)
    dev, tune = tickets[tickets["split"] == "dev"], tickets[tickets["split"] == "tune"]
    a_tfidf.fit_select(dev, tune, [1.0, 4.0])
    test_texts = set(tickets.loc[tickets["split"] == "test", "text"])
    assert all(not (set(s) & test_texts - set(dev["text"])) for s in seen)


@pytest.mark.skipif(not (RUNS_DIR / "main" / "manifest.json").exists(), reason="no main run yet")
def test_main_run_tuned_only_on_tune() -> None:
    run = RUNS_DIR / "main"
    ps = json.loads((run / "prompt_selection.json").read_text())
    assert ps["split"] == "tune"
    dec = pd.read_parquet(run / "decisions.parquet")
    assert not ((dec["arm"] == "prompt_selection") & (dec["split"] != "tune")).any()
    cal = run / "calibration.json"
    if cal.exists():
        assert json.loads(cal.read_text())["fit_split"] == "tune"
