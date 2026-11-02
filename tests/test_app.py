"""Smoke test of the Streamlit interface (renders every tab, no model server)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_app_renders_and_captures(no_ml):
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    labels = {t.label for t in app.tabs}
    assert {"Capture", "Red team", "Replay", "Investigator", "Experiments", "Evidence"} <= labels
    app.text_area[0].input("Hello PSI")
    app.button[0].click().run()
    assert not app.exception
    assert any("prompt_id" in str(j.value) for j in app.json)
