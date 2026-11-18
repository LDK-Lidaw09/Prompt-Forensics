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
    mismatches = []
    for fig in module.FIGURES + module.load_table_figures():
        value = module.lookup(module.DEFAULT_TABLES, fig)
        if abs(value - fig.expected) > 0.5 * 10 ** -fig.decimals + 1e-9:
            mismatches.append(f"{fig.where}: thesis {fig.printed}, table {value:.4f}")
    assert not mismatches, "\n".join(mismatches)
    assert all(ok for _, _, ok in module.check())


def test_every_figure_selects_one_row():
    module = _load()
    for fig in module.FIGURES:
        module.lookup(module.DEFAULT_TABLES, fig)  # raises if 0 or >1 rows


def test_wrong_figure_is_reported():
    module = _load()
    good = module.FIGURES[2]  # 72.7% after 100 replays
    wrong = module.Figure(good.where, "72.8%", good.table, good.row, good.column)
    module.FIGURES = [good, wrong]
    oks = [ok for fig, _, ok in module.check() if fig in (good, wrong)]
    assert oks == [True, False]


def test_reference_quantiles_match_pairs():
    module = _load()
    bad = [f"{where}: thesis {printed}, pairs {value:.4f}"
           for where, printed, value, ok in module.check_reference_quantiles() if not ok]
    assert not bad, "\n".join(bad)


def test_reference_tables_are_frozen():
    module = _load()
    assert module.verify_frozen_tables() == []


def test_every_table_cell_selects_one_row():
    module = _load()
    for fig in module.load_table_figures():
        module.lookup(module.DEFAULT_TABLES, fig)
