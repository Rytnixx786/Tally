"""Run the Streamlit dashboard headlessly against the warehouse and fail on any exception."""

from __future__ import annotations

import pytest

from tally.config import ROOT
from tally.paths import RUNS_DIR
from tests.conftest import pg_dsn


@pytest.mark.integration
def test_dashboard_renders_without_exceptions() -> None:
    if pg_dsn() is None or not (RUNS_DIR / "main" / "metrics.json").exists():
        pytest.skip("needs the warehouse and a published `main` run")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value.startswith("Tally")
    assert len(at.tabs) == 8
    labels = [m.label for m in at.metric]
    assert any(label.startswith("A TF-IDF") for label in labels)
