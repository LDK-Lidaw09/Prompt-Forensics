"""The five-minute demonstration runs and detects every tampering it shows."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_demo_runs_and_detects_tampering(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "demo.py"), "--out", str(tmp_path)],
        capture_output=True, text=True, check=True, cwd=ROOT,
    )
    assert "0 problem(s) found" in result.stdout
    assert result.stdout.count("-> DETECTED") == 4
    assert "NOT DETECTED" not in result.stdout
    assert (tmp_path / "evidence.sqlite").exists()
