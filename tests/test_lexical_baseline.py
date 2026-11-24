"""Lexical baseline: Wilson interval and the archived table quoted in Section 6.4.3."""

import importlib.util
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analysis_v2" / "lexical_baseline.py"
TABLE = ROOT / "results" / "v2" / "tables" / "lexical_baseline.csv"


def _load():
    spec = importlib.util.spec_from_file_location("lexical_baseline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wilson_interval():
    wilson = _load().wilson
    p, lo, hi = wilson(2384, 6148)
    assert round(p, 4) == 0.3878 and round(lo, 4) == 0.3757 and round(hi, 4) == 0.4000
    p, lo, hi = wilson(0, 16984)
    assert p == 0 and lo == 0 and round(hi, 4) == 0.0002
    p, lo, hi = wilson(10, 10)
    assert p == 1 and round(hi, 9) == 1 and round(lo, 4) == 0.7225
    assert all(math.isnan(x) for x in wilson(0, 0))


def test_write_csv_replaces_file(tmp_path):
    module = _load()
    out = tmp_path / "t.csv"
    out.write_text("old")
    module.write_csv(pd.DataFrame([{"a": 1}]), out)
    assert out.read_text().splitlines() == ["a", "1"]
    assert list(tmp_path.iterdir()) == [out]


def test_archived_table_matches_thesis():
    t = pd.read_csv(TABLE)
    assert list(t.columns) == ["features", "group", "n", "measure", "flagged", "rate", "lo", "hi"]
    get = lambda f, g: t[(t.features == f) & (t.group == g)].iloc[0]
    word = "word 1-2 grams"
    assert (get(word, "attack, test").n, get(word, "attack, test").flagged) == (18383, 18383)
    assert (get(word, "legitimate, test").n, get(word, "legitimate, test").flagged) == (16984, 0)
    assert get(word, "all, test").rate == 1.0
    char = "character 3-5 grams"
    assert get(char, "attack, test").flagged == 18382 and get(char, "legitimate, test").flagged == 2
    left = "word 1-2 grams, source left out"
    for src, pct in (("HarmBench", 38.8), ("JailbreakDB", 67.9), ("WildJailbreak", 68.8)):
        assert round(100 * get(left, f"attack, test, {src} (source left out)").rate, 1) == pct
        assert get(left, f"legitimate, test ({src} left out)").flagged == 0
