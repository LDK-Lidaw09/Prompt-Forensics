"""Check that the key figures quoted in the thesis match the archived result tables.

Each entry below names one figure as it is printed in the thesis, the table of
results/v2/tables/ it comes from, and how to read it there. The script rounds
the archived value to the number of decimals printed in the thesis and reports
any figure that differs. No figure is recomputed from raw data: the point is to
show that every quoted number can be traced to a committed file.

The thesis itself is not in this repository, so the printed figures are written
below by hand. When a figure changes in the thesis, its entry here must be
updated as well; the check then confirms that the new figure matches its table.

Usage:
    python scripts/check_thesis_numbers.py            # prints a report, exit 1 on mismatch
    python scripts/check_thesis_numbers.py --tables DIR
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_TABLES = Path(__file__).resolve().parents[1] / "results" / "v2" / "tables"
# Every numeric cell of the chapter 6 tables that has an archived source, with
# the table, row and column it is read from (extracted from the manuscript).
FIGURES_CSV = DEFAULT_TABLES.parent / "thesis_figures.csv"


@dataclass(frozen=True)
class Figure:
    """One figure quoted in the thesis and where to find it."""

    where: str            # place in the thesis
    printed: str          # figure as printed, e.g. "72.7%" or "0.884"
    table: str            # CSV file in results/v2/tables/
    row: dict             # column -> value selecting exactly one row (None = empty)
    column: str           # column holding the value

    @property
    def percent(self) -> bool:
        return self.printed.endswith("%")

    @property
    def decimals(self) -> int:
        text = self.printed.rstrip("%")
        return len(text.split(".")[1]) if "." in text else 0

    @property
    def expected(self) -> float:
        return float(self.printed.rstrip("%"))


ATT = {"mode": "attacks"}
SAFE = {"mode": "safe"}
MISTRAL_ATT = {"victim": "Mistral-7B", "mode": "attacks", "group": "all"}
MISTRAL_SAFE = {"victim": "Mistral-7B", "mode": "safe", "group": "all"}
INV_ATT = {"run_mode": "attacks", "rule": "rule_without_invented"}
ANN_ATT = {"run_mode": "attacks", "rule": "rule_as_annotated"}
ANN_SAFE = {"run_mode": "safe", "rule": "rule_as_annotated"}

FIGURES: list[Figure] = [
    # Section 6.3: tail risk of the victim model (Evaluator verdicts).
    Figure("6.3, attack pools with an unsafe answer after 1 replay", "25.4%",
           "tail_risk_curve.csv", {**ATT, "m": 1}, "tail"),
    Figure("6.3, after 10 replays", "52.8%",
           "tail_risk_curve.csv", {**ATT, "m": 10}, "tail"),
    Figure("6.3, after 100 replays", "72.7%",
           "tail_risk_curve.csv", {**ATT, "m": 100}, "tail"),
    Figure("6.3, lower bound of the 95% interval", "72.1%",
           "tail_risk_curve.csv", {**ATT, "m": 100}, "lo"),
    Figure("6.3, upper bound of the 95% interval", "73.3%",
           "tail_risk_curve.csv", {**ATT, "m": 100}, "hi"),
    Figure("6.3, strict rate (malicious only) after 100 replays", "35.5%",
           "tail_risk_curve.csv", {**ATT, "m": 100}, "tail_strict"),
    Figure("6.3, attack answers judged unsafe one by one", "25.7%",
           "evaluator_tail_risk.csv", MISTRAL_ATT, "rate_unsafe"),
    Figure("6.3, legitimate pools with an unsafe answer", "6.0%",
           "evaluator_tail_risk.csv", MISTRAL_SAFE, "tail_unsafe"),
    Figure("6.3, legitimate answers judged unsafe one by one", "0.79%",
           "evaluator_tail_risk.csv", MISTRAL_SAFE, "rate_unsafe"),
    # Section 6.2: stability.
    Figure("6.2, mean cosine Consistency, attack prompts", "0.884",
           "replay_stability.csv", MISTRAL_ATT, "cons_score_mean"),
    Figure("6.2, mean cosine Consistency, legitimate prompts", "0.948",
           "replay_stability.csv", MISTRAL_SAFE, "cons_score_mean"),
    Figure("Table 6.3, attack pools with no unsafe answer, cosine", "0.885",
           "stability_vs_verdict.csv", ATT, "cos_mean_k0"),
    Figure("Table 6.3, attack pools with an unsafe answer, cosine", "0.884",
           "stability_vs_verdict.csv", ATT, "cos_mean_k1"),
    Figure("Table 6.3, AUC of low cosine Consistency (attacks)", "0.51",
           "stability_vs_verdict.csv", ATT, "auc_low_cos_predicts_any_unsafe"),
    Figure("Table 6.3, AUC of low BERTScore Consistency (attacks)", "0.63",
           "stability_vs_verdict.csv", ATT, "auc_low_bert_predicts_any_unsafe"),
    # Section 6.3.5: human check of the Evaluator.
    Figure("6.3.5, weighted precision of the Evaluator", "87%",
           "evaluator_check_invented_rule.csv",
           {**ANN_ATT, "measure": "evaluator_precision"}, "value"),
    Figure("6.3.5, weighted recall of the Evaluator", "40%",
           "evaluator_check_invented_rule.csv",
           {**ANN_ATT, "measure": "evaluator_recall"}, "value"),
    Figure("6.3.5, precision without invented-data answers", "38%",
           "evaluator_check_invented_rule.csv",
           {**INV_ATT, "measure": "evaluator_precision"}, "value"),
    Figure("6.3.5, recall without invented-data answers", "37%",
           "evaluator_check_invented_rule.csv",
           {**INV_ATT, "measure": "evaluator_recall"}, "value"),
    Figure("6.3.5, share of attack answers unsafe without invented data", "26.4%",
           "evaluator_check_invented_rule.csv",
           {**INV_ATT, "measure": "share_unsafe_annotation"}, "value"),
    Figure("6.3.5, pool check, attack pools with one flagged answer", "40.0%",
           "pool_check_status.csv", {"stratum": "A1 attaque, k=1"}, "share"),
    Figure("6.3.5, pool check, attack pools with 2-5 flagged answers", "36.7%",
           "pool_check_status.csv", {"stratum": "A2 attaque, k=2-5"}, "share"),
    Figure("6.3.5, pool check, attack pools with 1-5 malicious answers", "46.7%",
           "pool_check_status.csv", {"stratum": "A3 attaque, 1-5 malicious"}, "share"),
    # Section 6.4: Investigators (decision field).
    Figure("Table 6.7, Qwen3-14B recall", "97.9%",
           "investigator_summary.csv", {"judge": "Qwen3-14B", "rule": "decision"}, "recall"),
    Figure("Table 6.7, Qwen3-14B false-positive rate", "0.03%",
           "investigator_summary.csv", {"judge": "Qwen3-14B", "rule": "decision"}, "fpr"),
    Figure("Table 6.7, Ministral-8B recall", "80.1%",
           "investigator_summary.csv", {"judge": "Ministral-8B", "rule": "decision"}, "recall"),
    Figure("Table 6.7, Phi-4-reasoning false-positive rate", "23.0%",
           "investigator_summary.csv", {"judge": "Phi-4-reasoning", "rule": "decision"}, "fpr"),
    # Section 6.2.6 / 5.5.2: whitespace control.
    Figure("6.2, same verdict, base vs original order (attacks)", "90.6%",
           "whitespace_baseline.csv", {**ATT, "comparison": "base vs orig"}, "same_verdict"),
    Figure("6.2, same verdict, original vs perturbed whitespace (attacks)", "79.5%",
           "whitespace_baseline.csv", {**ATT, "comparison": "orig vs pert"}, "same_verdict"),
    # Cells of chapter 6 tables checked one by one at the final reading.
    Figure("Table 6.1, legitimate pools with BERTScore Consistency >= 0.9", "84.7%",
           "replay_stability.csv", MISTRAL_SAFE, "cons_bert_score_frac_ge0.9"),
    Figure("Table B.6, legitimate pairs with cosine >= 0.85", "95.8%",
           "tau_sweep.csv", {"victim": "Mistral-7B", "mode": "safe", "tau": 0.85}, "frac_pairs_ge"),
    Figure("Table 6.5, JailbreakDB-style third, all families", "73.6%",
           "tail_risk_by_source.csv", {"source": "JailbreakDB", "family": None}, "tail"),
    Figure("Table 6.5, WildJailbreak third, upper bound", "86.0%",
           "tail_risk_by_source.csv", {"source": "WildJailbreak", "family": None}, "hi"),
    Figure("Table 6.6, attack precision, upper bound", "0.942",
           "evaluator_check_invented_rule.csv",
           {**ANN_ATT, "measure": "evaluator_precision"}, "ci_high"),
    Figure("Table 6.6, attack share unsafe (annotation), lower bound", "44.6%",
           "evaluator_check_invented_rule.csv",
           {**ANN_ATT, "measure": "share_unsafe_annotation"}, "ci_low"),
    Figure("Table 6.6, attack share unsafe (annotation), upper bound", "67.5%",
           "evaluator_check_invented_rule.csv",
           {**ANN_ATT, "measure": "share_unsafe_annotation"}, "ci_high"),
    Figure("Table 6.6, legitimate precision, lower bound", "0.161",
           "evaluator_check_invented_rule.csv",
           {**ANN_SAFE, "measure": "evaluator_precision"}, "ci_low"),
    Figure("Table 6.6, legitimate precision, upper bound", "0.508",
           "evaluator_check_invented_rule.csv",
           {**ANN_SAFE, "measure": "evaluator_precision"}, "ci_high"),
    Figure("Table 6.6, legitimate recall, upper bound", "0.053",
           "evaluator_check_invented_rule.csv",
           {**ANN_SAFE, "measure": "evaluator_recall"}, "ci_high"),
    Figure("Table 6.6, legitimate share unsafe (annotation)", "17.7%",
           "evaluator_check_invented_rule.csv",
           {**ANN_SAFE, "measure": "share_unsafe_annotation"}, "value"),
    Figure("Table 6.6, legitimate share unsafe (annotation), lower bound", "5.8%",
           "evaluator_check_invented_rule.csv",
           {**ANN_SAFE, "measure": "share_unsafe_annotation"}, "ci_low"),
    Figure("Table 6.6, legitimate share unsafe (annotation), upper bound", "33.3%",
           "evaluator_check_invented_rule.csv",
           {**ANN_SAFE, "measure": "share_unsafe_annotation"}, "ci_high"),
    Figure("Table 6.11, legitimate prompts at T=0, lower bound", "0.4%",
           "sensitivity_temperature.csv", {"temperature": 0.0, "run_mode": "safe"}, "any_unsafe_lo"),
    Figure("Table 6.11, legitimate Consistency at T=1.0", "0.916",
           "sensitivity_temperature.csv", {"temperature": 1.0, "run_mode": "safe"}, "consistency_mean"),
    Figure("Table 6.12, same-run recall with the answer, lower bound", "97.2%",
           "sensitivity_ablation.csv",
           {"baseline": "same run, temperature sample", "run_mode": "attacks"}, "with_lo"),
    Figure("Table 6.12, same-run recall with the answer, upper bound", "99.3%",
           "sensitivity_ablation.csv",
           {"baseline": "same run, temperature sample", "run_mode": "attacks"}, "with_hi"),
]


# Table 6.2: quantiles of the similarity between answers to different prompts,
# recomputed from the archived pairs (5,000 pairs per run mode and kind).
REFERENCE_PAIRS = "reference_distribution_pairs.csv"
TABLE_6_3 = {
    # (run_mode, kind): {column: (5th, median, 95th, 99th) as printed}
    ("attacks", "within_domain"): {"cosine": ("0.339", "0.499", "0.660", "0.738"),
                                   "bert_f1": ("0.771", "0.808", "0.846", "0.866")},
    ("attacks", "random"): {"cosine": ("0.318", "0.458", "0.601", "0.664"),
                            "bert_f1": ("0.769", "0.806", "0.839", "0.856")},
    ("safe", "within_domain"): {"cosine": ("0.389", "0.516", "0.648", "0.713"),
                                "bert_f1": ("0.747", "0.809", "0.842", "0.855")},
    ("safe", "random"): {"cosine": ("0.341", "0.464", "0.593", "0.653"),
                         "bert_f1": ("0.750", "0.807", "0.838", "0.852")},
}
QUANTILES = (0.05, 0.50, 0.95, 0.99)


def check_reference_quantiles(tables: Path = DEFAULT_TABLES) -> list[tuple[str, str, float, bool]]:
    """Compare the quantiles printed in Table 6.2 with the archived pairs."""
    pairs = pd.read_csv(tables / REFERENCE_PAIRS)
    results = []
    for (mode, kind), columns in TABLE_6_3.items():
        group = pairs[(pairs.run_mode == mode) & (pairs.kind == kind)]
        if len(group) != 5000:
            raise LookupError(f"{REFERENCE_PAIRS}: {mode}/{kind} has {len(group)} pairs, expected 5000")
        for column, printed in columns.items():
            for q, text in zip(QUANTILES, printed):
                value = float(group[column].quantile(q))
                where = f"Table 6.2, {mode} {kind}, {column} quantile {q:g}"
                results.append((where, text, value, round(value, 3) == float(text)))
    return results


@functools.lru_cache(maxsize=None)
def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def lookup(tables: Path, fig: Figure) -> float:
    """Return the archived value of one figure, in the unit printed in the thesis."""
    frame = _read(tables / fig.table)
    mask = pd.Series(True, index=frame.index)
    for column, value in fig.row.items():
        mask &= frame[column].isna() if value is None else frame[column] == value
    rows = frame[mask]
    if len(rows) != 1:
        raise LookupError(f"{fig.table}: {fig.row} selects {len(rows)} rows, expected 1")
    value = float(rows.iloc[0][fig.column])
    return value * 100 if fig.percent else value


def matches(value: float, fig: Figure) -> bool:
    """True when the printed figure is the archived value rounded to its decimals.

    Half a unit of the last printed decimal is allowed either way, so that a
    value lying exactly half-way may be rounded up or down.
    """
    return abs(value - fig.expected) <= 0.5 * 10 ** -fig.decimals + 1e-9


def load_table_figures(path: Path = FIGURES_CSV) -> list[Figure]:
    """Read the cell-by-cell figures of the chapter 6 tables (thesis_figures.csv)."""
    frame = pd.read_csv(path, dtype=str)
    return [Figure(f"{row.place} [{row.figure}]",
                   row.printed + ("%" if row.kind == "pct" else ""),
                   row.table, json.loads(row.selector), row.column)
            for row in frame.itertuples()]


def verify_frozen_tables(tables: Path = DEFAULT_TABLES) -> list[str]:
    """Return the tables whose SHA-256 differs from results/v2/tables/SHA256SUMS."""
    changed = []
    for line in (tables / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        actual = hashlib.sha256((tables / name).read_bytes()).hexdigest()
        if actual != digest:
            changed.append(name)
    return changed


def check(tables: Path = DEFAULT_TABLES) -> list[tuple[Figure, float, bool]]:
    """Compare every figure with its table; return (figure, archived value, ok)."""
    results = []
    for fig in FIGURES + load_table_figures():
        value = lookup(tables, fig)
        ok = matches(value, fig)
        results.append((fig, value, ok))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    args = parser.parse_args()

    changed = verify_frozen_tables(args.tables)
    for name in changed:
        print(f"CHANGED  {name} differs from SHA256SUMS: the reference tables are frozen")
    rows = [(fig.where, fig.printed, value, ok, fig.table, "%" if fig.percent else "")
            for fig, value, ok in check(args.tables)]
    rows += [(where, printed, value, ok, REFERENCE_PAIRS, "")
             for where, printed, value, ok in check_reference_quantiles(args.tables)]
    failures = 0
    for where, printed, value, ok, table, unit in rows:
        status = "OK " if ok else "MISMATCH"
        failures += not ok
        print(f"{status}  {printed:>7}  archived {value:.4f}{unit}  {where}  [{table}]")
    print(f"\n{len(rows) - failures} of {len(rows)} figures match their table.")
    print("Not covered: Table B.7 (answer lengths) and the failure causes of Table B.5,")
    print("which come from the raw files of the Hugging Face dataset.")
    return 1 if failures or changed else 0


if __name__ == "__main__":
    sys.exit(main())
