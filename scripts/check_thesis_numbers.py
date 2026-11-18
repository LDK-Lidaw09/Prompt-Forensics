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
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_TABLES = Path(__file__).resolve().parents[1] / "results" / "v2" / "tables"


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
    Figure("Table 6.6, attack pools with no unsafe answer, cosine", "0.885",
           "stability_vs_verdict.csv", ATT, "cos_mean_k0"),
    Figure("Table 6.6, attack pools with an unsafe answer, cosine", "0.884",
           "stability_vs_verdict.csv", ATT, "cos_mean_k1"),
    Figure("Table 6.6, AUC of low cosine Consistency (attacks)", "0.51",
           "stability_vs_verdict.csv", ATT, "auc_low_cos_predicts_any_unsafe"),
    Figure("Table 6.6, AUC of low BERTScore Consistency (attacks)", "0.63",
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
    Figure("Table 6.10, Qwen3-14B recall", "97.9%",
           "investigator_summary.csv", {"judge": "Qwen3-14B", "rule": "decision"}, "recall"),
    Figure("Table 6.10, Qwen3-14B false-positive rate", "0.03%",
           "investigator_summary.csv", {"judge": "Qwen3-14B", "rule": "decision"}, "fpr"),
    Figure("Table 6.10, Ministral-8B recall", "80.1%",
           "investigator_summary.csv", {"judge": "Ministral-8B", "rule": "decision"}, "recall"),
    Figure("Table 6.10, Phi-4-reasoning false-positive rate", "23.0%",
           "investigator_summary.csv", {"judge": "Phi-4-reasoning", "rule": "decision"}, "fpr"),
    # Section 6.2.6 / 5.5.2: whitespace control.
    Figure("6.2, same verdict, base vs original order (attacks)", "90.6%",
           "whitespace_baseline.csv", {**ATT, "comparison": "base vs orig"}, "same_verdict"),
    Figure("6.2, same verdict, original vs perturbed whitespace (attacks)", "79.5%",
           "whitespace_baseline.csv", {**ATT, "comparison": "orig vs pert"}, "same_verdict"),
    # Cells of chapter 6 tables checked one by one at the final reading.
    Figure("Table 6.2, legitimate pools with BERTScore Consistency >= 0.9", "84.7%",
           "replay_stability.csv", MISTRAL_SAFE, "cons_bert_score_frac_ge0.9"),
    Figure("Table 6.4, legitimate pairs with cosine >= 0.85", "95.8%",
           "tau_sweep.csv", {"victim": "Mistral-7B", "mode": "safe", "tau": 0.85}, "frac_pairs_ge"),
    Figure("Table 6.8, JailbreakDB-style third, all families", "73.6%",
           "tail_risk_by_source.csv", {"source": "JailbreakDB", "family": None}, "tail"),
    Figure("Table 6.8, WildJailbreak third, upper bound", "86.0%",
           "tail_risk_by_source.csv", {"source": "WildJailbreak", "family": None}, "hi"),
    Figure("Table 6.9, attack precision, upper bound", "0.942",
           "evaluator_check_invented_rule.csv",
           {**ANN_ATT, "measure": "evaluator_precision"}, "ci_high"),
    Figure("Table 6.9, attack share unsafe (annotation), lower bound", "44.6%",
           "evaluator_check_invented_rule.csv",
           {**ANN_ATT, "measure": "share_unsafe_annotation"}, "ci_low"),
    Figure("Table 6.9, attack share unsafe (annotation), upper bound", "67.5%",
           "evaluator_check_invented_rule.csv",
           {**ANN_ATT, "measure": "share_unsafe_annotation"}, "ci_high"),
    Figure("Table 6.9, legitimate precision, lower bound", "0.161",
           "evaluator_check_invented_rule.csv",
           {**ANN_SAFE, "measure": "evaluator_precision"}, "ci_low"),
    Figure("Table 6.9, legitimate precision, upper bound", "0.508",
           "evaluator_check_invented_rule.csv",
           {**ANN_SAFE, "measure": "evaluator_precision"}, "ci_high"),
    Figure("Table 6.9, legitimate recall, upper bound", "0.053",
           "evaluator_check_invented_rule.csv",
           {**ANN_SAFE, "measure": "evaluator_recall"}, "ci_high"),
    Figure("Table 6.9, legitimate share unsafe (annotation)", "17.7%",
           "evaluator_check_invented_rule.csv",
           {**ANN_SAFE, "measure": "share_unsafe_annotation"}, "value"),
    Figure("Table 6.9, legitimate share unsafe (annotation), lower bound", "5.8%",
           "evaluator_check_invented_rule.csv",
           {**ANN_SAFE, "measure": "share_unsafe_annotation"}, "ci_low"),
    Figure("Table 6.9, legitimate share unsafe (annotation), upper bound", "33.3%",
           "evaluator_check_invented_rule.csv",
           {**ANN_SAFE, "measure": "share_unsafe_annotation"}, "ci_high"),
    Figure("Table 6.15, legitimate prompts at T=0, lower bound", "0.4%",
           "sensitivity_temperature.csv", {"temperature": 0.0, "run_mode": "safe"}, "any_unsafe_lo"),
    Figure("Table 6.15, legitimate Consistency at T=1.0", "0.916",
           "sensitivity_temperature.csv", {"temperature": 1.0, "run_mode": "safe"}, "consistency_mean"),
    Figure("Table 6.17, same-run recall with the answer, lower bound", "97.2%",
           "sensitivity_ablation.csv",
           {"baseline": "same run, temperature sample", "run_mode": "attacks"}, "with_lo"),
    Figure("Table 6.17, same-run recall with the answer, upper bound", "99.3%",
           "sensitivity_ablation.csv",
           {"baseline": "same run, temperature sample", "run_mode": "attacks"}, "with_hi"),
]


def lookup(tables: Path, fig: Figure) -> float:
    """Return the archived value of one figure, in the unit printed in the thesis."""
    frame = pd.read_csv(tables / fig.table)
    mask = pd.Series(True, index=frame.index)
    for column, value in fig.row.items():
        mask &= frame[column].isna() if value is None else frame[column] == value
    rows = frame[mask]
    if len(rows) != 1:
        raise LookupError(f"{fig.table}: {fig.row} selects {len(rows)} rows, expected 1")
    value = float(rows.iloc[0][fig.column])
    return value * 100 if fig.percent else value


def check(tables: Path = DEFAULT_TABLES) -> list[tuple[Figure, float, bool]]:
    """Compare every figure with its table; return (figure, archived value, ok)."""
    results = []
    for fig in FIGURES:
        value = lookup(tables, fig)
        ok = round(value, fig.decimals) == round(fig.expected, fig.decimals)
        results.append((fig, value, ok))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    args = parser.parse_args()

    results = check(args.tables)
    failures = 0
    for fig, value, ok in results:
        unit = "%" if fig.percent else ""
        status = "OK " if ok else "MISMATCH"
        failures += not ok
        print(f"{status}  {fig.printed:>7}  archived {value:.4f}{unit}  {fig.where}  [{fig.table}]")
    print(f"\n{len(results) - failures} of {len(results)} figures match their table.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
