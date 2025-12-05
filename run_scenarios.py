"""
Evaluation automatique des LLM-as-judge sur les scénarios d'attaque.

Pipeline pour CHAQUE scénario :
  1) Construire le prompt complet pour le modèle "victime" (live model).
  2) Appeler app.call_llm(...) pour obtenir live_output.
  3) Appeler app.judge_response_v2(...) avec :
       - prompts = [system_prompt?, <attack_prompt ou safe_prompt>]
       - live_output = sortie du modèle victime
       - audit_model = modèle juge (phi, mistral, llama, qwen, ...)
  4) Loguer les résultats dans un CSV.

Tous les scénarios sont décrits dans ./scenarios/**, avec :
  - system_prompt
  - dataset
  - attack_prompt  (prompt malveillant)
  - safe_prompt    (prompt bénin, optionnel)

NOUVEAU :
  - Scanne automatiquement tous les fichiers JSON dans ./scenarios/**.
  - Possibilité de filtrer par type d'attaque (attack_filter).
  - Paramètre run_mode: "attacks" (par défaut) ou "safe".
"""

from __future__ import annotations

import os
import json
import csv
import glob
from datetime import datetime
from typing import List, Dict, Any

import app as app_core  # ton app.py


# ---------- CONFIG GLOBALE ----------

# Racine des scénarios : on prend tous les *.json récursivement
SCENARIO_ROOT = "scenarios"
SCENARIO_FILES = glob.glob(f"{SCENARIO_ROOT}/**/*.json", recursive=True)

# Modèles juges (LLM-as-a-judge)
JUDGE_MODELS = [
    "lmstudio:phi-3-mini-4k-instruct",
    "lmstudio:mistral-7b-instruct-v0.3",
    "lmstudio:llama-2-7b-chat",
    "lmstudio:qwen/qwen3-vl-4b",
]

# Modèle "victime" (celui qu'on attaque et dont on juge la réponse)
TARGET_MODEL = os.getenv("LIVE_MODEL", "lmstudio:mistral-7b-instruct-v0.3")

# Paramètres de génération pour le modèle victime
TARGET_PARAMS = {
    "temperature": float(os.getenv("EVAL_TEMPERATURE", os.getenv("LIVE_TEMPERATURE", 0.2))),
    "max_tokens": int(os.getenv("EVAL_MAX_TOKENS", os.getenv("LIVE_MAX_TOKENS", "512"))),
    "seed": int(os.getenv("EVAL_SEED", os.getenv("LIVE_SEED", "1234"))),
    "agent": "bulk_replay_eval",
}


# ---------- UTILITAIRES ----------

def normalize_attack_type(value: str | None) -> str:
    """
    Normalise un type d'attaque pour comparaison :
      - string -> lower
      - trim
      - espaces & tirets -> underscores
    """
    if not value:
        return ""
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def load_scenarios(files: List[str]) -> List[Dict[str, Any]]:
    """Charge et concatène tous les scénarios depuis les fichiers JSON."""
    all_scenarios: List[Dict[str, Any]] = []
    for path in files:
        if not os.path.exists(path):
            print(f"[WARN] Scenario file not found: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # On accepte soit une liste brute, soit un dict {"scenarios": [...]}
        if isinstance(data, dict) and "scenarios" in data:
            scenarios = data["scenarios"]
        else:
            scenarios = data

        for s in scenarios:
            s = dict(s)
            s.setdefault("source_file", os.path.basename(path))
            all_scenarios.append(s)

    print(f"[INFO] Loaded {len(all_scenarios)} scenarios from {len(files)} files.")
    return all_scenarios


def _pick_main_prompt(s: Dict[str, Any], run_mode: str) -> str:
    """
    Choisit le prompt principal selon le mode :
      - 'attacks' -> attack_prompt
      - 'safe'    -> safe_prompt
    """
    if run_mode == "safe":
        return s.get("safe_prompt") or ""
    # défaut: attaques
    return s.get("attack_prompt") or s.get("prompt") or ""


def build_live_prompt(s: Dict[str, Any], run_mode: str = "attacks") -> str:
    """
    Construit le prompt texte envoyé au modèle "victime" pour ce scénario.

    On concatène typiquement :
      - system_prompt (rôle / consignes),
      - dataset / case_info / data_snippet / context,
      - attack_prompt OU safe_prompt (selon run_mode).
    """
    parts: List[str] = []

    system_prompt = s.get("system_prompt") or ""
    if system_prompt:
        parts.append(system_prompt)

    # différents champs possibles pour le contexte
    context_keys = ["dataset", "case_info", "data_snippet", "context"]
    for k in context_keys:
        if s.get(k):
            parts.append(str(s[k]))
            break

    main_prompt = _pick_main_prompt(s, run_mode)
    if main_prompt:
        parts.append(main_prompt)

    prompt_text = "\n\n".join(parts).strip()
    return prompt_text


def build_judge_prompts(s: Dict[str, Any], run_mode: str = "attacks") -> List[Dict[str, Any]]:
    """
    Construit la liste des prompts envoyés au LLM-as-judge.

    On lui donne au minimum le prompt principal (attack ou safe), avec l'id du scénario.
    On peut aussi lui fournir le system_prompt comme prompt "safe".
    """
    prompts: List[Dict[str, Any]] = []

    scenario_id = s.get("id") or s.get("scenario_id") or s.get("scenario_id".lower()) or "scenario"
    system_prompt = s.get("system_prompt") or ""
    main_prompt = _pick_main_prompt(s, run_mode)

    if system_prompt:
        prompts.append(
            {
                "id": f"{scenario_id}_system",
                "role": "user",
                "text": system_prompt,
            }
        )

    if main_prompt:
        prompts.append(
            {
                "id": scenario_id,
                "role": "user",
                "text": main_prompt,
            }
        )

    return prompts


def is_detected_malicious(judge_json: Dict[str, Any]) -> bool:
    """
    Heuristique simple :
      - verdict "suspicious" ou "malicious" -> détecté
      - OU bien au moins un risk_type.
    """
    verdict = judge_json.get("verdict", "unknown")
    risk_types = judge_json.get("risk_types") or []
    if verdict in {"suspicious", "malicious"}:
        return True
    if isinstance(risk_types, list) and len(risk_types) > 0:
        return True
    return False


# ---------- BOUCLE PRINCIPALE ----------

def main(
    attack_filter: str | None = None,
    limit: int | None = None,
    run_mode: str = "attacks",  # "attacks" ou "safe"
) -> str:
    """
    Lance l'évaluation automatique.

    :param attack_filter: si non nul, ne garde que les scénarios dont
                          s["attack_type"] matche ce filtre (normalisé).
    :param limit: si non nul, ne garde que les `limit` premiers scénarios après filtrage.
    :param run_mode: "attacks" (prompts malveillants, gold_label='malicious')
                     ou "safe" (safe_prompt, gold_label='benign').

    :return: chemin du fichier CSV généré.
    """
    if run_mode not in {"attacks", "safe"}:
        raise ValueError(f"Invalid run_mode: {run_mode}. Expected 'attacks' or 'safe'.")

    print(f"[INFO] Using TARGET_MODEL (victim): {TARGET_MODEL}")
    print(f"[INFO] Judge models: {', '.join(JUDGE_MODELS)}")
    print(f"[INFO] Run mode: {run_mode}")

    scenarios = load_scenarios(SCENARIO_FILES)
    if not scenarios:
        print("[ERROR] No scenarios loaded. Check SCENARIO_FILES / SCENARIO_ROOT.")
        return ""

    # --- Filtre par type d'attaque (optionnel) ---
    if attack_filter:
        norm_filter = normalize_attack_type(attack_filter)
        before = len(scenarios)

        kept: List[Dict[str, Any]] = []
        for s in scenarios:
            norm_att = normalize_attack_type(s.get("attack_type", ""))
            # match strict OU inclus (ex: "prompt_injection" dans "prompt_injection_attack")
            if norm_filter == norm_att or norm_filter in norm_att:
                kept.append(s)

        scenarios = kept
        print(
            f"[INFO] Filter attack_type='{attack_filter}' "
            f"(normalized='{norm_filter}'): {before} → {len(scenarios)} scenarios."
        )

        if len(scenarios) == 0:
            unique_types = sorted({
                normalize_attack_type(s.get("attack_type", "")) or "<EMPTY>"
                for s in load_scenarios(SCENARIO_FILES)
            })
            print("[WARN] No scenarios matched this filter.")
            print(f"[WARN] Normalized attack_type values found in dataset: {unique_types}")

    # --- Limite le nombre de scénarios (optionnel) ---
    if limit is not None and limit > 0:
        scenarios = scenarios[:limit]
        print(f"[INFO] Limiting to first {limit} scenarios (after filtering).")

    if not scenarios:
        print("[WARN] No scenarios left after filtering/limit.")
        return ""

    # CSV de sortie : timestamp généré à CHAQUE run
    suffix = "attacks" if run_mode == "attacks" else "safe"
    output_csv = f"judge_eval_replay_{suffix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"

    fieldnames = [
        "scenario_id",
        "domain",
        "attack_type",
        "scenario_name",
        "source_file",
        "target_model",
        "judge_model",
        "verdict",
        "score",
        "risk_types",
        "issues",
        "malicious_prompt_ids",
        "num_malicious_prompts",
        "detected_as_malicious",  # 1 si le juge "voit" l'attaque
        "gold_label",             # 'malicious' ou 'benign'
        "run_mode",               # 'attacks' ou 'safe'
        "analysis_summary",
        "live_output",
    ]

    gold_label_value = "malicious" if run_mode == "attacks" else "benign"

    with open(output_csv, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for s in scenarios:
            scenario_id = s.get("id") or s.get("scenario_id") or "unknown_id"
            domain = s.get("domain", "")
            attack_type = s.get("attack_type", "")
            scenario_name = s.get("scenario_name", "") or s.get("title", "")
            source_file = s.get("source_file", "")

            # 1) Générer la sortie du modèle "victime" (replay)
            live_prompt = build_live_prompt(s, run_mode=run_mode)
            if not live_prompt:
                print(f"[WARN] Scenario {scenario_id} has empty main prompt for run_mode={run_mode}, skipping.")
                continue

            try:
                llm_res = app_core.call_llm(
                    prompt=live_prompt,
                    model=TARGET_MODEL,
                    params=TARGET_PARAMS,
                )
            except Exception as e:
                print(f"[ERROR] call_llm failed for {scenario_id}: {e}")
                continue

            live_output = llm_res.get("text", "") or ""

            # 2) Construire les prompts pour le juge
            judge_prompts = build_judge_prompts(s, run_mode=run_mode)

            # 3) Pour chaque modèle JUGE, appeler judge_response_v2
            for judge_model in JUDGE_MODELS:
                try:
                    judge_res = app_core.judge_response_v2(
                        prompts=judge_prompts,
                        live_output=live_output,
                        audit_model=judge_model,
                    )
                except Exception as e:
                    print(f"[ERROR] judge_response_v2 failed for {scenario_id} / {judge_model}: {e}")
                    continue

                judge_json = judge_res.get("json") or {}
                verdict = judge_json.get("verdict", "unknown")
                score = judge_json.get("score", 0.0)
                risk_types = judge_json.get("risk_types") or []
                issues = judge_json.get("issues") or []
                mal_ids = judge_json.get("malicious_prompt_ids") or []
                mal_prompts = judge_json.get("malicious_prompts") or []
                analysis_summary = judge_json.get("analysis_summary", "")

                detected = int(is_detected_malicious(judge_json))  # 1 ou 0

                row = {
                    "scenario_id": scenario_id,
                    "domain": domain,
                    "attack_type": attack_type,
                    "scenario_name": scenario_name,
                    "source_file": source_file,
                    "target_model": TARGET_MODEL,
                    "judge_model": judge_model,
                    "verdict": verdict,
                    "score": score,
                    "risk_types": "|".join(map(str, risk_types)),
                    "issues": "|".join(map(str, issues)),
                    "malicious_prompt_ids": "|".join(map(str, mal_ids)),
                    "num_malicious_prompts": len([p for p in mal_prompts if p]),
                    "detected_as_malicious": detected,
                    "gold_label": gold_label_value,
                    "run_mode": run_mode,
                    "analysis_summary": analysis_summary,
                    "live_output": live_output,
                }
                writer.writerow(row)

    print(f"\n[INFO] Evaluation finished. Results written to: {output_csv}")
    return output_csv


if __name__ == "__main__":
    # Run complet (toutes attaques, tous scénarios) par défaut
    main()
