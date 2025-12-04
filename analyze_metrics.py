"""
Analyse des résultats du LLM-as-judge à partir du CSV produit par run_scenarios.py.

Entrée : un CSV contenant au minimum les colonnes :
    - judge_model
    - attack_type                 (ex: "Prompt Injection — PII/PHI Disclosure")
    - gold_label                  (ex: "malicious" pour tes 240 attaques)
    - detected_as_malicious       (0/1)

Sorties :
    - metrics_attack : 1 ligne par judge_model, avec une colonne attack_type
      (la "famille" d'attaque, ex: "Prompt Injection", "Belief Elicitation")
    - metrics_global : métriques globales par judge_model (identiques ici,
      mais pratique si un jour tu mixes plusieurs familles).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def compute_metrics_for_group(sub: pd.DataFrame) -> pd.Series:
    """Calcule TP / FP / FN / TN + precision / recall / f1 / accuracy pour un groupe."""
    y_true = (sub["gold_label"] == "malicious").astype(int)
    y_pred = sub["detected_as_malicious"].astype(int)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(sub) if len(sub) > 0 else 0.0

    return pd.Series(
        {
            "n": len(sub),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }
    )


def analyze(csv_path: str | Path, save_prefix: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lit le CSV et retourne (metrics_attack, metrics_global).

    - metrics_attack : 1 ligne par judge_model, avec attack_type = famille d'attaque
                       (ex: "Belief Elicitation", "Prompt Injection")
    - metrics_global : métriques globales par judge_model
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required_cols = {"judge_model", "attack_type", "gold_label", "detected_as_malicious"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # ---- Construire la "famille" d'attaque (partie avant le '—') ----
    raw_type = df["attack_type"].fillna("")
    attack_family = raw_type.str.split("—").str[0].str.strip()
    attack_family = attack_family.replace("", "unknown")

    # Pour ce run, on prend une seule étiquette de famille (normalement un seul type filtré)
    unique_families = attack_family.unique()
    if len(unique_families) == 1:
        family_label = unique_families[0]
    else:
        # Si tu lances un run "All attacks", on met un label générique
        family_label = "mixed_attacks"

    # 1) Métriques par juge (c'est ton "Judge_results_with_attacks")
    metrics_attack = (
        df.groupby(["judge_model"])
        .apply(compute_metrics_for_group)
        .reset_index()
    )
    # on insère la colonne attack_type juste après judge_model
    metrics_attack.insert(1, "attack_type", family_label)

    # 2) Métriques globales par juge (identiques tant que tu ne mixes pas les familles)
    metrics_global = (
        df.groupby(["judge_model"])
        .apply(compute_metrics_for_group)
        .reset_index()
    )

    if save_prefix:
        save_prefix = str(save_prefix)
        metrics_attack.to_csv(f"{save_prefix}_by_attack.csv", index=False)
        metrics_global.to_csv(f"{save_prefix}_global.csv", index=False)
        print(
            f"[INFO] Saved metrics to:\n"
            f"  {save_prefix}_by_attack.csv\n"
            f"  {save_prefix}_global.csv"
        )

    return metrics_attack, metrics_global


def main():
    parser = argparse.ArgumentParser(
        description="Compute F1 / precision / recall / accuracy from judge CSV."
    )
    parser.add_argument("csv", help="Path to CSV produced by run_scenarios.py")
    parser.add_argument(
        "--save-prefix",
        "-o",
        help="Prefix for saving metrics CSVs (e.g. 'metrics_judges'). If omitted, no CSV is saved.",
        default=None,
    )
    args = parser.parse_args()

    metrics_attack, metrics_global = analyze(args.csv, args.save_prefix)

    print("\n=== Metrics by judge_model (with attack_type label) ===")
    print(metrics_attack.sort_values(["judge_model"]))

    print("\n=== Global metrics by judge_model ===")
    print(metrics_global.sort_values("judge_model"))


if __name__ == "__main__":
    main()
