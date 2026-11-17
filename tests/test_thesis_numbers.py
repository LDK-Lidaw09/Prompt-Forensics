"""The figures quoted in the thesis must match the archived result tables."""

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_thesis_numbers.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_thesis_numbers", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # needed by dataclasses
    spec.loader.exec_module(module)
    return module


def test_thesis_figures_match_tables():
    module = _load()
    mismatches = [
        f"{fig.where}: thesis {fig.printed}, table {value:.4f}"
        for fig, value, ok in module.check()
        if not ok
    ]
    assert not mismatches, "\n".join(mismatches)


def test_every_figure_selects_one_row():
    module = _load()
    for fig in module.FIGURES:
        module.lookup(module.DEFAULT_TABLES, fig)  # raises if 0 or >1 rows
