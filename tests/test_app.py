"""Smoke test of the Streamlit interface (renders every tab, no model server)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_app_renders_and_captures(no_ml):
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    labels = {t.label for t in app.tabs}
    assert {"Capture", "Red team", "Replay", "Investigator", "Experiments", "Evidence",
            "Replay explorer", "Results v2"} <= labels
    app.text_area[0].input("Hello PSI")
    button(app, "Send").click().run()
    assert not app.exception
    assert any("prompt_id" in str(j.value) for j in app.json)


def test_app_tampering_demo_detects_change(no_ml):
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    for prompt in ("First prompt", "Second prompt"):
        app.text_area[0].input(prompt)
        button(app, "Send").click().run()
    for name in ("Change the response and rebuild the hash chain (no key)",
                 "Delete the last record (truncation)"):
        next(s for s in app.selectbox if s.label == "Tampering").set_value(name).run()
        tamper = next(b for b in app.button if b.label == "Tamper with a copy and verify it")
        tamper.click().run()
        assert not app.exception
        assert any("Tampered copy" in e.value and "problem" in e.value for e in app.error), name
    verify = next(b for b in app.button if b.label == "Verify the evidence store")
    verify.click().run()
    assert any("no problem found" in s.value for s in app.success)


def test_app_results_tab_renders(no_ml):
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert "Results v2" in {t.label for t in app.tabs}
    assert any("within 100 replays" in m.label for m in app.metric)


def test_app_loads_real_examples(no_ml):
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    button(app, "Load real examples").click().run()
    assert not app.exception
    assert any("10 interaction(s) stored" in s.value for s in app.success)
    button(app, "Load real examples").click().run()
    assert any("10 already present" in s.value for s in app.success)
    button(app, "Verify the evidence store").click().run()
    assert any("no problem found" in s.value for s in app.success)
    assert any(m.label == "Malicious" for m in app.metric)  # replay explorer rendered
