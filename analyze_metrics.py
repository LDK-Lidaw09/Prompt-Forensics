
"""
Analyse des résultats du LLM-as-judge à partir du CSV produit par run_scenarios.py.

Entrée : un CSV contenant au minimum les colonnes :
    - judge_model
    - attack_type
    - gold_label           (ex: "malicious" pour tes 240 attaques)
    - detected_as_malicious (0/1)

Sorties :
    - tableau de métriques par (judge_model, attack_type)
    - tableau de métriques globales par judge_model
    - optionnel : sauvegarde en CSV des tableaux de métriques
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def compute_metrics_for_group(sub: pd.DataFrame) -> pd.Series:
    """
    Calcule TP / FP / FN / TN + precision / recall / f1 / accuracy
    pour un sous-ensemble de lignes (un groupe).
    """
    # Vrai label : 1 = malicious, 0 = non-malicious
    y_true = (sub["gold_label"] == "malicious").astype(int)
    y_pred = sub["detected_as_malicious"].astype(int)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
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

    - metrics_attack : métriques par (judge_model, attack_type)
    - metrics_global : métriques par judge_model (toutes attaques confondues)

    Si save_prefix est donné, sauvegarde aussi deux CSV :
        f"{save_prefix}_by_attack.csv"
        f"{save_prefix}_global.csv"
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required_cols = {"judge_model", "attack_type", "gold_label", "detected_as_malicious"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # 1) Métriques par juge ET par type d'attaque
    metrics_attack = (
        df.groupby(["judge_model", "attack_type"])
        .apply(compute_metrics_for_group)
        .reset_index()
    )

    # 2) Métriques globales par juge (toutes attaques confondues)
    metrics_global = (
        df.groupby(["judge_model"])
        .apply(compute_metrics_for_group)
        .reset_index()
    )

    if save_prefix:
        save_prefix = str(save_prefix)
        metrics_attack.to_csv(f"{save_prefix}_by_attack.csv", index=False)
        metrics_global.to_csv(f"{save_prefix}_global.csv", index=False)
        print(f"[INFO] Saved metrics to:\n  {save_prefix}_by_attack.csv\n  {save_prefix}_global.csv")

    return metrics_attack, metrics_global


def main():
    parser = argparse.ArgumentParser(description="Compute F1 / precision / recall / accuracy from judge CSV.")
    parser.add_argument("csv", help="Path to CSV produced by run_scenarios.py")
    parser.add_argument(
        "--save-prefix",
        "-o",
        help="Prefix for saving metrics CSVs (e.g. 'metrics_judges'). If omitted, no CSV is saved.",
        default=None,
    )
    args = parser.parse_args()

    metrics_attack, metrics_global = analyze(args.csv, args.save_prefix)

    # Affichage console lisible
    print("\n=== Metrics by judge_model AND attack_type ===")
    print(metrics_attack.sort_values(["judge_model", "attack_type"]))

    print("\n=== Global metrics by judge_model ===")
    print(metrics_global.sort_values("judge_model"))


if __name__ == "__main__":
    main()
