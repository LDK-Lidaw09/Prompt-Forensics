# adapters/lmstudio_client.py
# ---------------------------------------------------------------------
# Client LM Studio avec détection automatique du modèle chargé.
# - Interroge /v1/models pour savoir ce qui est dispo sur le port
# - Sélectionne le modèle en fonction de l'agent (live / self_audit / adversary)
# - Appelle /v1/chat/completions (et remonte l'erreur clairement en cas d'échec)
# ---------------------------------------------------------------------

from __future__ import annotations
import os
import time
from typing import List, Tuple, Dict, Any

import httpx

# Base par défaut pour le "live" (tu peux aussi passer base_url à chat)
LM_BASE = os.getenv("LIVE_API_BASE", "http://127.0.0.1:1234/v1").rstrip("/")

# Cache très simple pour /v1/models (évite de spamer le serveur)
_models_cache = {"ts": 0.0, "list": []}


def list_lmstudio_models(base_url: str | None = None, ttl_sec: int = 3) -> List[str]:
    """
    Retourne la liste des model.id chargés dans LM Studio, ex:
      ["phi-3-mini-4k-instruct", "mistral-7b-instruct-v0.3", "qwen/qwen3-vl-4b"]
    """
    base = (base_url or LM_BASE).rstrip("/")
    now = time.time()
    if (now - _models_cache["ts"] < ttl_sec) and _models_cache["list"]:
        return _models_cache["list"]

    try:
        r = httpx.get(f"{base}/models", timeout=5)
        r.raise_for_status()
        ids = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
        _models_cache["ts"] = now
        _models_cache["list"] = ids
        return ids
    except Exception:
        return []


def _to_model_id(s: str) -> str:
    """Convertit 'lmstudio:xyz' -> 'xyz', sinon renvoie s tel quel."""
    if not s:
        return s
    return s.split(":", 1)[1] if s.startswith("lmstudio:") else s


def resolve_model_id(
    preferred: str,
    fallbacks: List[str],
    base_url: str | None = None,
) -> Tuple[str, str]:
    """
    Essaie d'abord 'preferred', sinon parcourt 'fallbacks' dans l'ordre,
    sinon retourne le premier modèle chargé, sinon "".
    Renvoie (model_id, reason) où reason ∈ {"preferred_available","fallback_selected","first_loaded","no_model_available"}.
    """
    wanted = _to_model_id(preferred)
    models = list_lmstudio_models(base_url)

    if wanted and wanted in models:
        return wanted, "preferred_available"

    for fb in fallbacks:
        fb_id = _to_model_id(fb)
        if fb_id in models:
            return fb_id, "fallback_selected"

    if models:
        return models[0], "first_loaded"

    return "", "no_model_available"


def _post_json(url: str, payload: Dict[str, Any], timeout: int = 120) -> Tuple[httpx.Response, int]:
    t0 = time.time()
    r = httpx.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=timeout)
    dt = int((time.time() - t0) * 1000)
    return r, dt


def chat(
    model_or_alias: str | None,
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 512,
    seed: int | None = None,
    agent: str = "live",
    base_url: str | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Appelle LM Studio en sélectionnant automatiquement le bon modèle selon l'agent.
    - agent ∈ {"live","self_audit","adversary"}
    - model_or_alias peut être ignoré (None) : la résolution auto fera le travail.
    Retourne (text, meta) où meta inclut usage/timing/error/model_used/selected.
    """
    base = (base_url or LM_BASE).rstrip("/")

    # 1) Préférences par agent (modifiable)
    if agent == "live":
        preferred = os.getenv("LIVE_MODEL", "lmstudio:mistral-7b-instruct-v0.3")
        fallbacks = ["lmstudio:phi-3-mini-4k-instruct", "lmstudio:qwen/qwen3-vl-4b"]
    elif agent == "self_audit":
        preferred = os.getenv("SELF_AUDIT_MODEL", "lmstudio:phi-3-mini-4k-instruct")
        fallbacks = ["lmstudio:mistral-7b-instruct-v0.3", "lmstudio:qwen/qwen3-vl-4b"]
    elif agent == "adversary":
        preferred = os.getenv("ADVERSARY_MODEL", "lmstudio:qwen/qwen3-vl-4b")
        fallbacks = ["lmstudio:phi-3-mini-4k-instruct", "lmstudio:mistral-7b-instruct-v0.3"]
    else:
        preferred = os.getenv("LIVE_MODEL", "lmstudio:mistral-7b-instruct-v0.3")
        fallbacks = []

    # Si un model_or_alias explicite est fourni, on le force comme preferred.
    if model_or_alias:
        preferred = model_or_alias

    model_id, reason = resolve_model_id(preferred, fallbacks, base_url=base)
    if not model_id:
        return "", {
            "error": f"[LMStudio {agent}] aucun modèle disponible sur {base}. "
                     f"Charge un modèle dans LM Studio ou vérifie le port.",
            "model_used": "",
            "selected": "no_model_available",
        }

    # 2) Appel /v1/chat/completions (le format qui marche dans tes tests)
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r, dt_ms = _post_json(f"{base}/chat/completions", payload)
    if r.status_code == 200:
        js = r.json()
        text = js.get("choices", [{}])[0].get("message", {}).get("content", "")
        return text, {
            "raw": js,
            "usage": js.get("usage", {}),
            "timing_ms": dt_ms,
            "model_used": model_id,
            "selected": reason,
        }

    # 3) Erreur lisible
    return "", {
        "error": f"[LMStudio {agent}] {r.status_code} {r.text}",
        "timing_ms": dt_ms,
        "model_used": model_id,
        "selected": reason,
    }
