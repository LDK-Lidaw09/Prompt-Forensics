"""Live capture of one interaction (thesis section 4.2, demo UI and API).

:func:`capture_interaction` is what happens when a user sends a prompt through
the demo UI or the HTTP API:

1. build the model input (with the recent session history if sessions are on);
2. call the model;
3. build the canonical payload, compute the Prompt ID and store the
   interaction, signed, in the evidence store;
4. optionally run the self-audit and the Investigator on the answer and attach
   their verdicts to the stored interaction as annotations;
5. append both turns to the session transcript.

:func:`replay_interaction` sends a stored prompt again and stores the answer
as a ``replay`` record linked to its source. By default it reuses the model and
the generation parameters recorded in the source's canonical payload, and it
stores, with the replay, the recorded configuration, the configuration used and
every difference between them (see :func:`compare_configuration`).

The payload's ``prompt`` field is the exact text sent to the model (history
included), so the stored evidence matches what the model actually saw.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from .canonical import build_interaction_payload, sha256_hex
from .config import get_settings
from .llm import call_llm, sent_param_keys, split_model_id
from .signing import load_or_create_key
from .storage import EvidenceStore, utc_now
from .agents.repository_investigator import investigate
from .agents.self_audit import self_audit


@lru_cache(maxsize=1)
def get_store() -> EvidenceStore:
    """The evidence store configured by ``DB_PATH`` and ``SIGNING_KEY_PATH``."""
    settings = get_settings()
    return EvidenceStore(settings.db_path, load_or_create_key(settings.signing_key_path))


def _compose_with_history(prompt_text: str, history: list[dict[str, Any]]) -> str:
    if not history:
        return prompt_text
    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in history
    )
    return (
        "You are in a multi-turn chat. Use the conversation history when relevant.\n\n"
        f"Conversation history:\n{transcript}\n\n"
        "Current user message:\n"
        f"{prompt_text}\n\n"
        "If prior turns answer the user's need, you may summarize and build on them."
    )


def capture_interaction(
    prompt_text: str,
    *,
    model: str | None = None,
    user_id: str = "anonymous",
    session_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    run_self_audit: bool = True,
    run_investigator: bool = False,
    store: EvidenceStore | None = None,
) -> dict[str, Any]:
    """Send one prompt to the model and store the interaction as evidence.

    ``session_id`` turns on multi-turn mode for that session. Model errors are
    raised as :class:`prompt_forensics.llm.LLMError`; nothing is stored then.
    """
    settings = get_settings()
    store = store or get_store()
    model = model or settings.live_model
    params = {
        "temperature": settings.live_temperature if temperature is None else temperature,
        "max_tokens": settings.live_max_tokens if max_tokens is None else max_tokens,
        "seed": settings.seed_base if seed is None else seed,
    }

    history = store.session_messages(session_id, settings.session_window) if session_id else []
    context_hash = sha256_hex("\n---\n".join(m["content"] for m in history)) if history else None
    model_input = _compose_with_history(prompt_text, history)

    response = call_llm(model_input, model, params)

    payload = build_interaction_payload(
        prompt=model_input,
        model=model,
        params=params,
        timestamp=utc_now(),
        user_id=user_id,
        session_id=session_id,
        context_hash=context_hash,
    )
    stored = store.append_interaction(
        payload=payload,
        response_text=response.text,
        record_type="live",
        model_used=response.model,
        provider=response.provider,
        metadata={"user_message": prompt_text, **response.provenance()},
    )

    result: dict[str, Any] = {
        "response": response.text,
        "prompt_id": stored.prompt_id,
        "content_hash": stored.content_hash,
        "response_hash": stored.response_hash,
        "chain_hash": stored.chain_hash,
        "signed": stored.signed,
        "context_hash": context_hash,
        "history_used": bool(history),
        "provenance": response.provenance(),
    }

    if run_self_audit:
        audit = self_audit(response.text, settings.judge_model)
        store.add_annotation(stored.id, "self_audit", audit["json"], settings.judge_model)
        result["audit"] = audit["json"]

    if run_investigator:
        report = investigate(
            [{"id": stored.prompt_id, "role": "user", "text": prompt_text}],
            response.text,
            settings.judge_model,
        )
        store.add_annotation(stored.id, "investigator", report["json"], settings.judge_model)
        result["investigator"] = report["json"]

    if session_id:
        store.append_message(session_id, "user", prompt_text)
        store.append_message(session_id, "assistant", response.text)

    return result


#: Generation parameters that a replay takes from the recorded payload.
REPLAY_PARAMS = ("temperature", "max_tokens", "seed")

#: What a replay also depends on but the record of the prototype does not hold.
#: They are listed in every replay so that the report states what was not checked.
NOT_RECORDED = (
    "chat template applied by the server",
    "server defaults for sampling parameters that were not sent",
    "serving software and its version",
    "hardware and load of the server",
)


def recorded_configuration(source: dict[str, Any]) -> dict[str, Any]:
    """Model and generation parameters recorded in a stored interaction.

    They are read from the signed canonical payload, not from the settings, so
    that a replay starts from what was used at capture time. Raises
    ``ValueError`` when the stored payload no longer matches its Prompt ID,
    i.e. when the record was changed after capture; the chain and the
    signatures of the whole store are checked by ``EvidenceStore.verify``.
    """
    canonical = source.get("canonical_payload") or ""
    if canonical and sha256_hex(canonical) != source.get("prompt_id"):
        raise ValueError(f"record {source.get('prompt_id')!r}: payload does not match its Prompt ID")
    try:
        payload = json.loads(source.get("canonical_payload") or "{}")
    except (TypeError, ValueError):
        payload = {}
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    return {
        "model": payload.get("model") or source.get("model_requested"),
        "model_used": source.get("model_used"),
        "system": payload.get("system"),
        "params": dict(params),
    }


def compare_configuration(
    recorded: dict[str, Any],
    used_model: str,
    used_params: dict[str, Any],
    overrides: dict[str, Any],
    served_model: str | None = None,
    sent_keys: frozenset[str] | None = None,
    used_system: str | None = None,
) -> dict[str, Any]:
    """Compare the configuration of a replay with the recorded one.

    Each difference is listed with its origin: ``override`` when the caller
    asked for another value, ``not_recorded`` when the record holds no value
    and a default was used, ``served`` when the server reports another model
    than the one recorded (``model_used``) or than the one requested
    (``served_model``), ``not_sent`` when the provider ignores a recorded
    parameter (``sent_keys``). ``faithful`` is true only when no difference exists.
    """
    divergences: list[dict[str, Any]] = []
    if recorded.get("model") is None:
        origin = "override" if overrides.get("model") is not None else "not_recorded"
        divergences.append({"field": "model", "recorded": None, "used": used_model, "origin": origin})
    elif used_model != recorded["model"]:
        divergences.append({"field": "model", "recorded": recorded["model"], "used": used_model, "origin": "override"})
    if served_model and served_model != split_model_id(used_model)[1]:
        divergences.append({"field": "served_model", "recorded": split_model_id(used_model)[1],
                            "used": served_model, "origin": "served"})
    if served_model and recorded.get("model_used") and served_model != recorded["model_used"]:
        divergences.append({"field": "model_used", "recorded": recorded["model_used"], "used": served_model,
                            "origin": "served"})
    if (recorded.get("system") or None) != (used_system or None):
        divergences.append({"field": "system", "recorded": recorded.get("system"), "used": used_system,
                            "origin": "override"})
    elif recorded.get("system") and sent_keys is not None and "system" not in sent_keys:
        divergences.append({"field": "system", "recorded": recorded["system"], "used": None, "origin": "not_sent"})
    rec_params = recorded.get("params") or {}
    for key in sorted(set(rec_params) | set(used_params)):
        if key in rec_params and sent_keys is not None and key not in sent_keys:
            divergences.append({"field": key, "recorded": rec_params[key], "used": None, "origin": "not_sent"})
        elif key not in rec_params:
            origin = "override" if overrides.get(key) is not None else "not_recorded"
            divergences.append({"field": key, "recorded": None, "used": used_params.get(key), "origin": origin})
        elif key not in used_params:
            divergences.append({"field": key, "recorded": rec_params[key], "used": None, "origin": "not_sent"})
        elif used_params[key] != rec_params[key]:
            origin = "override" if overrides.get(key) is not None else "not_recorded"
            divergences.append({"field": key, "recorded": rec_params[key], "used": used_params[key], "origin": origin})
    return {
        "recorded": recorded,
        "used": {"model": used_model, "model_used": served_model, "system": used_system,
                 "params": dict(used_params)},
        "divergences": divergences,
        "faithful": not divergences,
        "not_recorded": list(NOT_RECORDED),
    }


def replay_interaction(
    source_prompt_id: str,
    *,
    model: str | None = None,
    seed: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    store: EvidenceStore | None = None,
) -> dict[str, Any]:
    """Send a stored prompt again and store the answer as a ``replay`` record.

    The replayed text is the stored ``prompt_text``, i.e. exactly what the
    model saw the first time (history included). The model and the parameters
    ``temperature``, ``max_tokens`` and ``seed`` are taken from the recorded
    payload of the source; an argument that is not ``None`` overrides the
    recorded value, and a parameter absent from the record falls back to the
    settings. The new record points back to its source through
    ``metadata.source_prompt_id`` and holds, in ``metadata.replay_config``, the
    recorded and used configurations and their differences, which are also
    returned. Raises ``KeyError`` if the Prompt ID is unknown and
    :class:`~prompt_forensics.llm.LLMError` if the model call fails.
    """
    settings = get_settings()
    store = store or get_store()
    source = store.get_interaction(source_prompt_id)
    if source is None:
        raise KeyError(f"unknown prompt_id {source_prompt_id!r}")

    recorded = recorded_configuration(source)
    overrides = {"temperature": temperature, "max_tokens": max_tokens, "seed": seed}
    defaults = {"temperature": settings.live_temperature, "max_tokens": settings.live_max_tokens,
                "seed": settings.seed_base}
    requested_model = model
    model = model or recorded["model"] or settings.live_model
    params: dict[str, Any] = {}
    for key in REPLAY_PARAMS:
        if overrides[key] is not None:
            params[key] = overrides[key]
        elif key in recorded["params"]:
            params[key] = recorded["params"][key]
        else:
            params[key] = defaults[key]
    for key, value in recorded["params"].items():   # other recorded parameters are sent as recorded
        params.setdefault(key, value)

    call_params = dict(params)
    if recorded["system"] and not call_params.get("system"):
        call_params["system"] = recorded["system"]       # the recorded system message is sent again
    response = call_llm(source["prompt_text"], model, call_params)
    config = compare_configuration(recorded, model, params, {**overrides, "model": requested_model}, served_model=response.model,
                                   sent_keys=sent_param_keys(model),
                                   used_system=call_params.get("system") if recorded["system"] else None)
    payload = build_interaction_payload(
        prompt=source["prompt_text"], model=model, params=params, timestamp=utc_now(),
        system=recorded["system"], user_id="replay",
    )
    stored = store.append_interaction(
        payload=payload,
        response_text=response.text,
        record_type="replay",
        model_used=response.model,
        provider=response.provider,
        metadata={"source_prompt_id": source_prompt_id, "replay_config": config, **response.provenance()},
    )
    return {
        "response": response.text,
        "baseline": source["response_text"],
        "prompt_id": stored.prompt_id,
        "content_hash": stored.content_hash,
        "source_content_hash": source["content_hash"],
        "response_hash": stored.response_hash,
        "model": model,
        "params": params,
        "replay_config": config,
    }


def describe_model(model: str) -> str:
    """``"lmstudio:mistral"`` -> ``"mistral (lmstudio)"``, for display."""
    provider, name = split_model_id(model)
    return f"{name} ({provider})"
