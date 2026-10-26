"""One client for every model backend used by PSI.

The victim and the judges are all reached through :func:`call_llm`. The model
identifier selects the backend (see :mod:`prompt_forensics.config`):

=================  ===========================================  ==================
Model id           Backend                                      Base URL setting
=================  ===========================================  ==================
``name``           OpenAI-compatible server (vLLM in the thesis) ``LIVE_API_BASE``
``vllm:name``      same as above, explicit
``lmstudio:name``  LM Studio (OpenAI-compatible)                ``LMSTUDIO_API_BASE``
``openai:name``    OpenAI Chat Completions                      ``OPENAI_API_BASE``
``anthropic:name`` Anthropic Messages API                       fixed
``sim:name``       offline echo simulator (demos and tests)     none
=================  ===========================================  ==================

Design rules:

- **Failures raise** :class:`LLMError`. An unreachable server, an HTTP error or
  an empty choice list is never turned into an empty answer, because an empty
  answer would silently enter the similarity metrics.
- **The requested model is the model used.** There is no fallback to another
  loaded model; if the server does not serve that model, the call fails.
- **The seed is always forwarded** when given, for every OpenAI-compatible
  backend, because per-replay seeds are part of the chain of custody (4.3.1).
- The prompt is sent as a single ``user`` message. A ``system`` message is
  added only when ``params["system"]`` is set. The thesis experiments do not
  set it: the scenario's system prompt is part of the user message instead
  (see :func:`prompt_forensics.scenarios.build_victim_prompt`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import requests

from .config import get_settings

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class LLMError(RuntimeError):
    """Raised when a model call fails. The message says which backend and why."""


@dataclass
class LLMResponse:
    """Text returned by a model, plus provenance metadata."""

    text: str
    model: str          # model id actually sent to the backend
    provider: str       # vllm | lmstudio | openai | anthropic | sim
    timing_ms: int
    usage: dict[str, Any] = field(default_factory=dict)
    # Reasoning text, when the server returns it separately from the answer
    # (for example vLLM started with --reasoning-parser).
    reasoning: str | None = None
    finish_reason: str | None = None

    def provenance(self) -> dict[str, Any]:
        """Metadata to store next to the text."""
        return {
            "model_used": self.model,
            "provider": self.provider,
            "timing_ms": self.timing_ms,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
        }


def split_model_id(model: str) -> tuple[str, str]:
    """Split ``"lmstudio:mistral"`` into ``("lmstudio", "mistral")``.

    Unprefixed ids go to the OpenAI-compatible ``LIVE_API_BASE`` server. Hugging
    Face ids contain a slash but no colon, so they are never mistaken for a prefix.
    """
    prefix, sep, rest = model.partition(":")
    if sep and prefix in {"vllm", "lmstudio", "openai", "anthropic", "sim"}:
        return prefix, rest
    return "vllm", model


def call_llm(prompt: str, model: str | None = None, params: Mapping[str, Any] | None = None) -> LLMResponse:
    """Send ``prompt`` to ``model`` and return its answer.

    Recognised ``params`` keys:

    - ``temperature`` (float), ``max_tokens`` (int), ``seed`` (int)
    - ``repetition_penalty`` (float): vLLM sampling extension
    - ``system`` (str): optional system message
    - ``guided_json`` (dict): JSON schema for vLLM constrained decoding
    - ``guided_decoding_backend`` (str): vLLM backend for ``guided_json``
    - ``extra_body`` (dict): extra fields merged into the request, for example
      ``{"chat_template_kwargs": {"enable_thinking": False}}`` for Qwen3.

    Unknown keys (such as ``agent``, kept for logging) are ignored.
    """
    settings = get_settings()
    params = dict(params or {})
    provider, model_name = split_model_id((model or settings.live_model).strip())
    if not model_name:
        raise LLMError(f"empty model id for provider {provider!r}")

    temperature = float(params.get("temperature", settings.live_temperature))
    max_tokens = int(params.get("max_tokens", settings.live_max_tokens))
    messages = []
    if params.get("system"):
        messages.append({"role": "system", "content": str(params["system"])})
    messages.append({"role": "user", "content": prompt})

    if provider == "sim":
        return LLMResponse(
            text=f"[simulator] Echo: {prompt[:200]}", model=model_name, provider="sim", timing_ms=0
        )

    if provider == "anthropic":
        return _call_anthropic(model_name, messages, temperature, max_tokens, settings)

    base_url = {
        "vllm": settings.live_api_base,
        "lmstudio": settings.lmstudio_api_base,
        "openai": settings.openai_api_base,
    }[provider]
    headers = {}
    if provider == "openai":
        if not settings.openai_api_key:
            raise LLMError("openai: OPENAI_API_KEY is not set")
        headers["Authorization"] = f"Bearer {settings.openai_api_key}"

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if params.get("seed") is not None:
        payload["seed"] = int(params["seed"])
    if params.get("repetition_penalty") is not None:
        # vLLM extension (used by the Investigator runs); OpenAI ignores it.
        payload["repetition_penalty"] = float(params["repetition_penalty"])
    if params.get("guided_json") is not None:
        # vLLM-specific: restrict generation to JSON matching this schema.
        payload["guided_json"] = params["guided_json"]
        payload["guided_decoding_backend"] = params.get("guided_decoding_backend", "outlines")
    payload.update(params.get("extra_body") or {})

    return _call_openai_compatible(provider, base_url, payload, headers, settings.request_timeout_s)


def _call_openai_compatible(
    provider: str, base_url: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: int
) -> LLMResponse:
    url = f"{base_url}/chat/completions"
    started = time.perf_counter()
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
    except requests.RequestException as exc:
        raise LLMError(f"{provider}: cannot reach {url}: {exc}") from exc
    timing_ms = int((time.perf_counter() - started) * 1000)

    if response.status_code != 200:
        raise LLMError(f"{provider}: HTTP {response.status_code} from {url}: {response.text[:500]}")
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise LLMError(f"{provider}: response without choices: {str(data)[:500]}")

    message = choices[0].get("message") or {}
    return LLMResponse(
        text=message.get("content") or "",
        model=data.get("model") or payload["model"],
        provider=provider,
        timing_ms=timing_ms,
        usage=data.get("usage") or {},
        reasoning=message.get("reasoning_content"),
        finish_reason=choices[0].get("finish_reason"),
    )


def _call_anthropic(
    model_name: str, messages: list[dict[str, str]], temperature: float, max_tokens: int, settings
) -> LLMResponse:
    if not settings.anthropic_api_key:
        raise LLMError("anthropic: ANTHROPIC_API_KEY is not set")
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    body: dict[str, Any] = {
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [m for m in messages if m["role"] != "system"],
    }
    if system:
        body["system"] = system
    headers = {
        "x-api-key": settings.anthropic_api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    started = time.perf_counter()
    try:
        response = requests.post(
            ANTHROPIC_API_URL, json=body, headers=headers, timeout=settings.request_timeout_s
        )
    except requests.RequestException as exc:
        raise LLMError(f"anthropic: cannot reach the API: {exc}") from exc
    timing_ms = int((time.perf_counter() - started) * 1000)
    if response.status_code != 200:
        raise LLMError(f"anthropic: HTTP {response.status_code}: {response.text[:500]}")

    data = response.json()
    text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
    return LLMResponse(
        text=text,
        model=data.get("model", model_name),
        provider="anthropic",
        timing_ms=timing_ms,
        usage=data.get("usage") or {},
        finish_reason=data.get("stop_reason"),
    )


def list_models(provider: str = "lmstudio") -> list[str]:
    """Model ids served by an OpenAI-compatible backend (empty list if unreachable).

    Used by the demo UI to populate model pickers; never used to pick a model
    automatically.
    """
    settings = get_settings()
    base_url = settings.lmstudio_api_base if provider == "lmstudio" else settings.live_api_base
    try:
        response = requests.get(f"{base_url}/models", timeout=5)
        response.raise_for_status()
    except requests.RequestException:
        return []
    return [m["id"] for m in response.json().get("data", []) if m.get("id")]
