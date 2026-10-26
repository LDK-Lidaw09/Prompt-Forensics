"""Self-audit of one model output (demo UI).

A judge model reads an output and returns its confidence, probable sources,
risk flags, a summary and a recommended action. It is shown in the demo UI
next to each answer; it is not used in the thesis experiments.
"""

from __future__ import annotations

from typing import Any

from ..jsonparse import as_str_list, clamp01, coerce_float, extract_json_object
from ..llm import LLMError, call_llm


def build_prompt(output_text: str) -> str:
    # Instructions go in the user message: some local servers ignore system messages.
    return (
        "ROLE: You are an AI self-audit agent.\n"
        "TASK: Given the model output below, return ONLY a single VALID JSON object.\n"
        "NO prose, NO code fences, NO markdown — JSON object only.\n"
        "Schema (keys required):\n"
        "{\n"
        '  "confidence": float in [0,1],\n'
        '  "probable_sources": string[],\n'
        '  "risk_flags": string[],\n'
        '  "summary": string,\n'
        '  "recommended_action": string\n'
        "}\n\n"
        "MODEL_OUTPUT_START\n"
        f"{output_text}\n"
        "MODEL_OUTPUT_END\n"
    )


def self_audit(output_text: str, judge_model: str) -> dict[str, Any]:
    """Return ``{"ok", "json", "raw_text"}``. Never raises for model errors."""
    try:
        response = call_llm(
            build_prompt(output_text), judge_model, {"temperature": 0.0, "max_tokens": 320, "seed": 42}
        )
    except LLMError as exc:
        return {"ok": False, "raw_text": "", "json": _fallback(f"audit_call_failed: {exc}")}

    obj = extract_json_object(response.text)
    if obj is None:
        return {"ok": False, "raw_text": response.text, "json": _fallback("audit_parse_error")}

    confidence = coerce_float(obj.get("confidence"))
    return {
        "ok": True,
        "raw_text": response.text,
        "json": {
            "confidence": clamp01(confidence) if confidence is not None else None,
            "probable_sources": as_str_list(obj.get("probable_sources")),
            "risk_flags": as_str_list(obj.get("risk_flags")),
            "summary": str(obj.get("summary") or ""),
            "recommended_action": str(obj.get("recommended_action") or "manual_review"),
        },
    }


def _fallback(flag: str) -> dict[str, Any]:
    return {
        "confidence": None,
        "probable_sources": [],
        "risk_flags": [flag],
        "summary": "Audit could not parse model output into valid JSON.",
        "recommended_action": "manual_review",
    }
