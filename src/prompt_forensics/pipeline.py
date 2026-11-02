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
as a ``replay`` record linked to its source.

The payload's ``prompt`` field is the exact text sent to the model (history
included), so the stored evidence matches what the model actually saw.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .canonical import build_interaction_payload, sha256_hex
from .config import get_settings
from .llm import call_llm, split_model_id
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
    model saw the first time (history included). The new record points back
    to its source through ``metadata.source_prompt_id``. Raises ``KeyError``
    if the Prompt ID is unknown and :class:`~prompt_forensics.llm.LLMError` if
    the model call fails.
    """
    settings = get_settings()
    store = store or get_store()
    source = store.get_interaction(source_prompt_id)
    if source is None:
        raise KeyError(f"unknown prompt_id {source_prompt_id!r}")

    model = model or source["model_requested"] or settings.live_model
    params = {
        "temperature": settings.live_temperature if temperature is None else temperature,
        "max_tokens": settings.live_max_tokens if max_tokens is None else max_tokens,
        "seed": settings.seed_base if seed is None else seed,
    }
    response = call_llm(source["prompt_text"], model, params)
    payload = build_interaction_payload(
        prompt=source["prompt_text"], model=model, params=params, timestamp=utc_now(),
        user_id="replay",
    )
    stored = store.append_interaction(
        payload=payload,
        response_text=response.text,
        record_type="replay",
        model_used=response.model,
        provider=response.provider,
        metadata={"source_prompt_id": source_prompt_id, **response.provenance()},
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
    }


def describe_model(model: str) -> str:
    """``"lmstudio:mistral"`` -> ``"mistral (lmstudio)"``, for display."""
    provider, name = split_model_id(model)
    return f"{name} ({provider})"
