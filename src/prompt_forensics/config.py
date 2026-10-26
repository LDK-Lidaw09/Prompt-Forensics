"""Runtime settings, read once from environment variables.

Every tunable value used by the package is declared here, with the default
used in the thesis experiments. A ``.env`` file in the working directory is
loaded automatically when ``python-dotenv`` is installed (see ``.env.example``).

Model identifiers select the backend through an optional prefix
(see :mod:`prompt_forensics.llm`):

- no prefix or ``vllm:``  -> OpenAI-compatible server at ``LIVE_API_BASE`` (vLLM)
- ``lmstudio:``           -> LM Studio server at ``LMSTUDIO_API_BASE``
- ``openai:``             -> OpenAI API (needs ``OPENAI_API_KEY``)
- ``anthropic:``          -> Anthropic API (needs ``ANTHROPIC_API_KEY``)
- ``sim:``                -> offline echo simulator, for demos and tests only
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

try:  # Optional: load a local .env file if python-dotenv is available.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional
    pass


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value and value.strip() else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value and value.strip() else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """All runtime settings. Build it with :func:`get_settings`."""

    # --- Model serving -------------------------------------------------------
    # OpenAI-compatible endpoint used for unprefixed model ids (vLLM on RunPod
    # in the thesis). vLLM listens on port 8000 by default.
    live_api_base: str
    # LM Studio endpoint, used for ids prefixed with "lmstudio:".
    lmstudio_api_base: str
    openai_api_base: str
    openai_api_key: str
    anthropic_api_key: str
    request_timeout_s: int

    # --- Models (thesis section 5) -------------------------------------------
    # Victim model whose behaviour is replayed.
    live_model: str
    # Model used by the Evaluator Agent, the self-audit and the demo judges.
    judge_model: str
    # Model used by the toy red-team generator in the demo UI.
    adversary_model: str

    # --- Victim generation parameters (thesis 4.3.1) -------------------------
    live_temperature: float
    live_max_tokens: int
    seed_base: int

    # --- Similarity metrics (thesis 5.2.2 and 5.2.3) -------------------------
    embed_model_name: str
    # Embeddings run on CPU by default so they do not compete with vLLM for GPU
    # memory on the same machine.
    embed_device: str
    embed_batch_size: int
    bertscore_lang: str
    # Empty string means "bert_score default for the language" (roberta-large
    # for English), which is what the thesis reports.
    bertscore_model_type: str
    similarity_threshold: float

    # --- Storage and integrity (thesis 4.2) ----------------------------------
    db_path: str
    signing_key_path: str

    # --- Chat sessions (demo UI) ---------------------------------------------
    session_mode: bool
    session_window: int

    # --- Dataset --------------------------------------------------------------
    scenario_root: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the settings, read from the environment on first call."""
    return Settings(
        live_api_base=_env_str("LIVE_API_BASE", "http://127.0.0.1:8000/v1").rstrip("/"),
        lmstudio_api_base=_env_str(
            "LMSTUDIO_API_BASE", _env_str("LLM_API_BASE", "http://127.0.0.1:1234/v1")
        ).rstrip("/"),
        openai_api_base=_env_str("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/"),
        openai_api_key=_env_str("OPENAI_API_KEY", ""),
        anthropic_api_key=_env_str("ANTHROPIC_API_KEY", ""),
        request_timeout_s=_env_int("LIVE_REQUEST_TIMEOUT", 300),
        live_model=_env_str("LIVE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
        judge_model=_env_str("JUDGE_MODEL", _env_str("SELF_AUDIT_MODEL", "Qwen/Qwen3-14B")),
        adversary_model=_env_str("ADVERSARY_MODEL", "Qwen/Qwen3-14B"),
        live_temperature=_env_float("LIVE_TEMPERATURE", 0.2),
        live_max_tokens=_env_int("LIVE_MAX_TOKENS", 512),
        seed_base=_env_int("LIVE_SEED", 1234),
        embed_model_name=_env_str("EMBED_MODEL_NAME", "Alibaba-NLP/gte-large-en-v1.5"),
        embed_device=_env_str("EMBED_DEVICE", "cpu"),
        # Batch size 1 is what produced the thesis numbers. Larger batches are
        # faster but padding can change the last decimals of the embeddings.
        embed_batch_size=_env_int("EMBED_BATCH_SIZE", 1),
        bertscore_lang=_env_str("BERTSCORE_LANG", "en"),
        bertscore_model_type=_env_str("BERTSCORE_MODEL_TYPE", ""),
        similarity_threshold=_env_float("SIMILARITY_THRESHOLD", 0.9),
        db_path=_env_str("DB_PATH", "./psi_evidence.sqlite"),
        signing_key_path=_env_str("SIGNING_KEY_PATH", "./keys/psi_signing_key.pem"),
        session_mode=_env_bool("SESSION_MODE", False),
        session_window=_env_int("SESSION_WINDOW", 6),
        scenario_root=_env_str("SCENARIO_ROOT", "data/scenarios"),
    )
