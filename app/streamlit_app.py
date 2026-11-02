"""PSI demo application (thesis section 4.5).

A single-file Streamlit interface over the ``prompt_forensics`` package. It
contains no forensic logic of its own: every action below calls a package
function, so what the UI shows is exactly what the experiment scripts compute.

Run it from the repository root::

    python -m streamlit run app/streamlit_app.py

The look (style sheet, banner, cards) is in ``app/ui.py``.

Tabs:

- **Overview**     what PSI does and where each part lives in the interface.
- **Live results** dashboard read from the evidence store every few seconds:
                   records, Evaluator verdicts, Investigator reports, replay
                   pools, integrity and the tampering tests of the session.
- **Capture**      send a prompt, store it as signed evidence, self-audit it,
                   or load real interactions of the version 2 runs.
- **Replay**       replay a stored prompt, judge each replay, compare it with
                   the original and measure the robustness of a replay pool.
- **Replay explorer** the stored replays of one interaction with their verdicts.
- **Investigator** run the repository Investigator on chosen stored prompts.
- **Evidence**     browse, export and verify the evidence store, anchor its
                   head, and tamper with a copy to see the checks at work.
- **Red team**     toy adversary that rewrites a prompt into an attack.
- **Experiments**  small runs of the dataset experiments and the tables of
                   chapter 6 from result CSVs.

No result is read from precomputed files: every number shown comes from the
tests run against the evidence store.
"""

from __future__ import annotations

import difflib
import gzip
import html
import io
import json
import re
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

# Make the package importable without installing it (``pip install -e .``
# does the same thing permanently).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))  # tamper_experiment.py: the tampering attacks
sys.path.insert(0, str(ROOT / "app"))  # ui.py: style sheet and HTML components

import ui  # noqa: E402

from prompt_forensics.agents.adversary import ATTACK_TEMPLATES, run_attack  # noqa: E402
from prompt_forensics.agents.comparators import judge_replay_safety, judge_replay_similarity  # noqa: E402
from prompt_forensics.agents.repository_investigator import investigate, is_detected  # noqa: E402
from prompt_forensics.canonical import build_interaction_payload, prompt_id  # noqa: E402
from prompt_forensics.agents.replay_group import evaluate_groups, parse_replay_files  # noqa: E402
from prompt_forensics.config import get_settings  # noqa: E402
from prompt_forensics.experiments import analysis  # noqa: E402
from prompt_forensics.experiments.investigator import InvestigatorRunConfig  # noqa: E402
from prompt_forensics.experiments.investigator import run_investigator as run_thesis_investigator  # noqa: E402
from prompt_forensics.experiments.pilot_investigator import InvestigatorConfig as PilotInvestigatorConfig  # noqa: E402
from prompt_forensics.experiments.pilot_investigator import run_investigator as run_pilot_investigator  # noqa: E402
from prompt_forensics.experiments.replays import ReplayConfig, run_replays  # noqa: E402
from prompt_forensics.llm import LLMError, list_models  # noqa: E402
from prompt_forensics.metrics.robustness import compute_replay_robustness  # noqa: E402
from prompt_forensics.pii import detect_sensitive, has_sensitive  # noqa: E402
from prompt_forensics.pipeline import capture_interaction, get_store, replay_interaction  # noqa: E402
from prompt_forensics.scenarios import RUN_MODES  # noqa: E402
from prompt_forensics.signing import public_key_path  # noqa: E402
from tamper_experiment import (  # noqa: E402
    a_delete_middle, a_edit_response, a_edit_response_rechain, a_edit_response_rehash,
    a_truncate_end, rows as read_rows, write_rows,
)

LANGUAGES = ["(none)", "English", "French", "Spanish", "German"]
EXAMPLES_V2 = ROOT / "data" / "demo" / "examples_v2.json.gz"  # scripts/build_demo_examples.py
UNSAFE = ("suspicious", "malicious")
VERDICT_COLOURS = ui.VERDICT

# Tampering attacks offered in the Evidence tab (from scripts/tamper_experiment.py).
# The attacker edits the SQLite file directly but knows neither the private
# signing key nor the anchor kept outside the database.
TAMPERING = {
    "Change the text of the response": a_edit_response,
    "Change the response and recompute its SHA-256 hash": a_edit_response_rehash,
    "Change the response and rebuild the hash chain (no key)": a_edit_response_rechain,
    "Delete the record": a_delete_middle,
    "Delete the last record (truncation)": a_truncate_end,
}


# --------------------------------------------------------------------- helpers

def with_language_hint(prompt: str, language: str) -> str:
    """Optionally ask for an answer language. The hint is part of the stored prompt."""
    if language == "(none)":
        return prompt
    return f"{prompt.strip()}\n\n[Output language: {language}]"


def fmt(value: Any, digits: int = 3) -> str:
    """Format a score for display; ``None`` (not computed) is shown as a dash."""
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def short_hash(value: str | None, n: int = 16) -> str:
    return f"{value[:n]}…" if value else "–"


def pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{100 * float(value):.{digits}f} %"
    except (TypeError, ValueError):
        return "–"


@st.cache_data(ttl=30, show_spinner=False)
def served_models() -> list[str]:
    """Models served by LM Studio; cached so that every rerun does not call it."""
    return list_models("lmstudio")


def download_csv(df: pd.DataFrame, filename: str, label: str = "Download CSV") -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), filename, "text/csv")


def show_diff(baseline: str, replay: str) -> None:
    """Line diff between the baseline answer and a replayed answer."""
    diff = difflib.unified_diff(
        baseline.splitlines(), replay.splitlines(), "baseline", "replay", lineterm=""
    )
    text = "\n".join(diff)
    st.code(text or "(identical)", language="diff")


def stored_prompt_picker(label: str, key: str, record_types: tuple[str, ...] = ("live", "attack")) -> str | None:
    """Selectbox over recent stored interactions. Returns the chosen Prompt ID."""
    rows = [r for r in get_store().recent_interactions(200) if r["record_type"] in record_types]
    if not rows:
        st.info("No stored interaction yet. Capture one in the Capture tab first.")
        return None
    labels = {
        r["prompt_id"]: f"{r['created_at']} · {r['record_type']} · {r['prompt_id'][:12]}… · "
                        f"{(r['prompt_text'] or '')[:60]!r}"
        for r in rows
    }
    return st.selectbox(label, list(labels), format_func=labels.get, key=key)


@st.cache_data(show_spinner=False)
def load_examples() -> list[dict[str, Any]]:
    """Real prompts of the version 2 runs with their 100 replays and verdicts."""
    if not EXAMPLES_V2.exists():
        return []
    with gzip.open(EXAMPLES_V2, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def example_label(ex: dict[str, Any]) -> str:
    kind = ("attack · " + ex["attack_type"].replace("_", " ") if ex["run_mode"] == "attacks"
            else "legitimate" + (" · resembling an attack" if ex.get("safe_kind") == "hard_negative" else ""))
    return f"{ex['scenario_id']} · {ex['domain']} · {kind}"


def example_payload(ex: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    return build_interaction_payload(
        prompt=ex["model_input"], model=ex["model"],
        params={"temperature": ex["temperature"], "max_tokens": ex["max_tokens"], "seed": replay["seed"]},
        timestamp="2026-11-01T00:00:00Z", user_id="runs-v2", session_id=None)


def load_examples_into_store() -> tuple[int, int]:
    """Store the first replay of each example as an interaction (once)."""
    st.session_state.pop("verify_report", None)  # the store changes: an old report would mislead
    store, added, present = get_store(), 0, 0
    for ex in load_examples():
        first = ex["replays"][0]
        payload = example_payload(ex, first)
        if store.get_interaction(prompt_id(payload)):
            present += 1
            continue
        store.append_interaction(
            payload=payload, response_text=first["text"], record_type="live",
            model_used=ex["model"], provider="vllm (version 2 runs)",
            metadata={"source": "runs v2", "scenario_id": ex["scenario_id"], "run_mode": ex["run_mode"],
                      "replay_idx": first["replay_idx"], "evaluator_verdict": first["verdict"]})
        added += 1
    return added, present


# --------------------------------------------------------------------- sidebar

def sidebar() -> dict[str, Any]:
    """Model and generation settings shared by every tab."""
    settings = get_settings()
    st.sidebar.header("Model")
    served = served_models()
    source = st.sidebar.radio(
        "Model id", ["Configured (LIVE_MODEL)", "LM Studio list", "Manual"],
        help="Prefixes: vllm: (default), lmstudio:, openai:, anthropic:, sim: (offline echo).",
    )
    if source == "LM Studio list" and served:
        model = "lmstudio:" + st.sidebar.selectbox("LM Studio model", served)
    elif source == "LM Studio list":
        st.sidebar.warning("LM Studio is not reachable; using LIVE_MODEL.")
        model = settings.live_model
    elif source == "Manual":
        model = st.sidebar.text_input("Model id", settings.live_model)
    else:
        model = settings.live_model
    st.sidebar.caption(f"Victim: `{model}`  \nJudge: `{settings.judge_model}`")

    st.sidebar.header("Generation")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.5, settings.live_temperature, 0.05)
    max_tokens = st.sidebar.number_input("Max new tokens", 16, 4096, settings.live_max_tokens, 16)
    seed = st.sidebar.number_input("Seed", 0, 2**31 - 1, settings.seed_base)
    language = st.sidebar.selectbox("Answer language hint", LANGUAGES)

    st.sidebar.header("Session")
    session_on = st.sidebar.toggle("Multi-turn session", settings.session_mode,
                                   help=f"Adds the last {settings.session_window} messages to the prompt.")
    session_id = st.sidebar.text_input("Session id", "demo-session") if session_on else None
    user_id = st.sidebar.text_input("User id", "analyst")

    return {"model": model, "temperature": float(temperature), "max_tokens": int(max_tokens),
            "seed": int(seed), "language": language, "session_id": session_id, "user_id": user_id}


# ------------------------------------------------------------------ Capture tab

def tab_capture(opts: dict[str, Any]) -> None:
    st.html(ui.section("✍️", "Capture an interaction",
                       "The prompt is sent to the victim model, then stored with its Prompt ID, "
                       "content hash, chain hash and ECDSA signature (thesis 4.2)."))
    if load_examples():
        with st.container(border=True):
            c1, c2 = st.columns([3, 1], vertical_alignment="center")
            c1.markdown(f"**Real interactions, no model needed.** Store {len(load_examples())} "
                        "interactions of the version 2 runs (six attacks, four legitimate prompts, "
                        "with the response that Mistral-7B gave) in the evidence store.")
            if c2.button("Load real examples", type="primary", width="stretch"):
                added, present = load_examples_into_store()
                st.success(f"{added} interaction(s) stored" + (f", {present} already present" if present else "")
                           + ". Replay them in the Replay tab; the results appear in Live results.")
    with st.form("capture"):
        prompt = st.text_area("Prompt", height=140)
        col1, col2 = st.columns(2)
        audit = col1.checkbox("Self-audit the answer", True)
        investigator = col2.checkbox("Run the Investigator", False)
        submitted = st.form_submit_button("Send")

    if submitted and prompt.strip():
        st.session_state.pop("verify_report", None)
        with st.spinner("Calling the model…"):
            try:
                st.session_state["capture_result"] = capture_interaction(
                    with_language_hint(prompt, opts["language"]),
                    model=opts["model"], user_id=opts["user_id"], session_id=opts["session_id"],
                    temperature=opts["temperature"], max_tokens=opts["max_tokens"], seed=opts["seed"],
                    run_self_audit=audit, run_investigator=investigator,
                )
            except LLMError as exc:
                st.error(f"Model call failed, nothing was stored: {exc}")
                return

    result = st.session_state.get("capture_result")
    if not result:
        return
    with st.container(border=True):
        st.markdown("**Answer of the model**")
        st.write(result["response"])

    flags = detect_sensitive(result["response"])
    if has_sensitive(flags):
        st.warning("Possible sensitive content (regex flags, may be false positives): "
                   + ", ".join(k for k, v in flags.items() if v and k != "keywords")
                   + (f" · keywords: {', '.join(flags['keywords'])}" if flags["keywords"] else ""))

    st.markdown("**Stored as evidence**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Prompt ID", short_hash(result["prompt_id"]),
              help="SHA-256 of the canonical form of the interaction (thesis 4.2).")
    c2.metric("Chain hash", short_hash(result["chain_hash"]),
              help="Links this record to the previous one: changing or removing any earlier "
                   "record breaks the chain.")
    c3.metric("ECDSA signature", "signed" if result["signed"] else "not signed",
              help="The chain hash is signed with the private key of this installation.")
    with st.expander("Full evidence record"):
        st.json({k: result[k] for k in ("prompt_id", "content_hash", "response_hash",
                                        "chain_hash", "signed", "context_hash", "history_used")})
    with st.expander("Provenance"):
        st.json(result["provenance"])
    if "audit" in result:
        with st.expander("Self-audit", expanded=True):
            st.json(result["audit"])
    if "investigator" in result:
        with st.expander("Investigator", expanded=True):
            st.json(result["investigator"])


# ----------------------------------------------------------------- Red team tab

def tab_red_team(opts: dict[str, Any]) -> None:
    settings = get_settings()
    st.html(ui.section("🗡️", "Toy red-team generator"))
    st.caption(f"The adversary (`{settings.adversary_model}`) rewrites the prompt with a template; "
               f"the result is sent to `{settings.live_model}` and stored as an `attack` record. "
               "Demonstration only: the thesis dataset was not built this way.")
    base = st.text_area("Base prompt", key="rt_prompt")
    template = st.selectbox("Template", list(ATTACK_TEMPLATES))
    if st.button("Generate and run attack") and base.strip():
        with st.spinner("Adversary, then victim…"):
            try:
                out = run_attack(base, template, get_store())
            except LLMError as exc:
                st.error(str(exc))
                return
        st.markdown("**Attack sent to the victim**")
        st.code(out["attack_text"], language="text")
        st.markdown("**Victim answer**")
        st.write(out["victim_output"])
        st.caption(f"Stored as Prompt ID `{out['prompt_id']}`")


# ------------------------------------------------------------------- Replay tab

def tab_replay(opts: dict[str, Any]) -> None:
    st.html(ui.section("🔂", "Replay a stored prompt",
                       "Replays use the exact stored model input. Each replay is itself stored as a "
                       "replay record that points to its source."))
    source_id = stored_prompt_picker("Interaction to replay", "replay_src")
    if not source_id:
        return
    source = get_store().get_interaction(source_id)
    with st.expander("Stored prompt and answer"):
        st.code(source["prompt_text"], language="text")
        st.write(source["response_text"])

    single, pool, bulk = st.tabs(["Single replay + judges", "Replay pool (robustness)", "Across models"])
    with single:
        replay_single(source_id, opts)
    with pool:
        replay_pool(source_id, opts)
    with bulk:
        replay_across_models(source_id, opts)


def replay_single(source_id: str, opts: dict[str, Any]) -> None:
    """Replay once, show the diff and run the baseline-vs-replay judges."""
    judge = st.checkbox("Run the similarity and safety judges", True)
    if st.button("Replay once"):
        with st.spinner("Replaying…"):
            try:
                out = replay_interaction(source_id, model=opts["model"], seed=opts["seed"],
                                         temperature=opts["temperature"], max_tokens=opts["max_tokens"])
            except LLMError as exc:
                st.error(str(exc))
                return
        st.write(out["response"])
        same = out["content_hash"] == out["source_content_hash"]
        st.caption(f"Stored as `{out['prompt_id']}` · same content hash as the source: {same} "
                   "(differs when model or parameters differ)")
        show_diff(out["baseline"], out["response"])
        if judge:
            judge_model = get_settings().judge_model
            with st.spinner("Judging…"):
                sim = judge_replay_similarity(out["baseline"], out["response"], judge_model)["json"]
                safety = judge_replay_safety(out["baseline"], out["response"], judge_model)["json"]
            record_id = get_store().get_interaction(out["prompt_id"])["id"]
            get_store().add_annotation(record_id, EVALUATOR_KIND, safety, judge_model)
            get_store().add_annotation(record_id, "replay_similarity", sim, judge_model)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Hybrid similarity", fmt(sim["hybrid_similarity"]))
            c2.metric("BERTScore F1", fmt(sim["bertscore_f1"]))
            c3.metric("Safety verdict", safety["verdict"])
            c4.metric("Safety score", fmt(safety["score"]))
            with st.expander("Judge details"):
                st.json({"similarity": sim, "safety": safety})


def replay_pool(source_id: str, opts: dict[str, Any]) -> None:
    """Replay n times with seeds seed, seed+1, ... and compute Consistency and Fidelity."""
    n = st.slider("Replays in the pool", 2, 100, 10)
    evaluate = st.checkbox("Evaluator: judge every replayed response", True,
                           help="Stores a safe / suspicious / malicious verdict with each replay. "
                                "Live results and the Replay explorer update while the pool runs.")
    st.caption(f"Seeds {opts['seed']} … {opts['seed'] + n - 1}. Replay 1 is the Fidelity reference "
               "(thesis 5.2). The stored original answer is not part of the pool.")
    if st.button("Run replay pool"):
        outputs: list[str | None] = []
        bar = st.progress(0.0)
        judge_model = get_settings().judge_model
        for i in range(n):
            try:
                out = replay_interaction(
                    source_id, model=opts["model"], seed=opts["seed"] + i,
                    temperature=opts["temperature"], max_tokens=opts["max_tokens"])
                outputs.append(out["response"])
                if evaluate:
                    safety = judge_replay_safety(out["baseline"], out["response"], judge_model)["json"]
                    get_store().add_annotation(get_store().get_interaction(out["prompt_id"])["id"],
                                               EVALUATOR_KIND, safety, judge_model)
            except LLMError as exc:
                st.warning(f"Replay {i + 1} failed: {exc}")
                outputs.append(None)  # kept as missing, never scored as 0
            bar.progress((i + 1) / n, text=f"replay {i + 1} of {n}")
        with st.spinner("Computing embeddings and BERTScore…"):
            try:
                robust = compute_replay_robustness(outputs, get_settings().similarity_threshold,
                                                   get_settings().similarity_threshold)
            except ImportError as exc:
                st.error(f"Similarity models are not installed ({exc}). "
                         'Run `pip install -e ".[ml]"`. The replays are stored.')
                return
        get_store().add_annotation(get_store().get_interaction(source_id)["id"], "replay_pool",
                                   {"replays": n, "seed_first": opts["seed"], "consistency": robust["consistency"],
                                    "fidelity": robust["fidelity"]}, None)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Consistency (cosine)", fmt(robust["consistency"]["score"]))
        c2.metric("Fidelity (cosine)", fmt(robust["fidelity"]["score"]))
        c3.metric("Consistency (BERTScore)", fmt(robust["consistency_bert"]["score"]))
        c4.metric("Fidelity (BERTScore)", fmt(robust["fidelity_bert"]["score"]))
        if not robust["complete"]:
            st.warning(f"Incomplete pool, missing replays: {robust['missing_indices']}")
        with st.expander("All metrics"):
            st.json(robust)


def replay_across_models(source_id: str, opts: dict[str, Any]) -> None:
    """Replay the same prompt on several models (same seed and parameters)."""
    default = "\n".join(dict.fromkeys([opts["model"], get_settings().live_model]))
    models = [m.strip() for m in st.text_area("One model id per line", default).splitlines() if m.strip()]
    if st.button("Replay on every model"):
        rows = []
        for model in models:
            try:
                out = replay_interaction(source_id, model=model, seed=opts["seed"],
                                         temperature=opts["temperature"], max_tokens=opts["max_tokens"])
                rows.append({"model": model, "prompt_id": out["prompt_id"], "error": "",
                             "char_similarity": difflib.SequenceMatcher(
                                 None, out["baseline"], out["response"]).ratio(),
                             "response": out["response"]})
            except LLMError as exc:
                rows.append({"model": model, "prompt_id": "", "error": str(exc),
                             "char_similarity": None, "response": ""})
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch")
        download_csv(df, "bulk_replay.csv")


# ---------------------------------------------------------- Replay explorer tab

def verdict_strip(replays: list[dict[str, Any]]) -> str:
    """One square per replay, coloured by the Evaluator verdict (20 per row, 10 on phones)."""
    cells = "".join(
        f'<span title="replay {r["replay_idx"] + 1} (seed {r["seed"]}): {html.escape(r["verdict"])}" '
        f'style="background:{VERDICT_COLOURS.get(r["verdict"], ui.UNKNOWN)};'
        f'color:{VERDICT_COLOURS.get(r["verdict"], ui.UNKNOWN)}"></span>'
        for r in replays)
    legend = "".join(f'<span><i style="background:{c}"></i>{v}</span>'
                     for v, c in {**VERDICT_COLOURS, "no verdict": ui.UNKNOWN}.items())
    return (f'<div class="psi-strip">{cells}</div>'
            f'<div class="psi-legend">{legend}<span>one square per replay, in seed order</span></div>')


def tab_replay_explorer() -> None:
    st.html(ui.section("🔁", "Replay explorer",
                       "The replays stored for one interaction, in seed order, with the verdict the Evaluator "
                       "gave each response. Run a replay pool in the Replay tab: it appears here."))
    state = live_state()
    pools = state["pools"]
    if not pools:
        st.html(ui.callout("blue", "ℹ️", "No replay stored yet. In the Replay tab, choose an interaction and "
                                         "run a replay pool with the Evaluator on."))
        return
    labels = {i: f"{len(p['replays'])} replays · {p['unsafe']} unsafe · "
                 f"{(p['source_record'].get('prompt_text') or '')[-70:]!r}" for i, p in enumerate(pools)}
    pool = pools[st.selectbox("Replayed interaction", list(labels), format_func=labels.get)]
    source = pool["source_record"]
    meta = json.loads(source.get("metadata_json") or "{}")
    tags = [ui.badge(source.get("record_type", "?"), "#475569"),
            ui.badge(source.get("model_used") or "model ?", "#4f46e5")]
    if meta.get("run_mode"):
        tags.append(ui.badge("attack prompt" if meta["run_mode"] == "attacks" else "legitimate prompt",
                             "#dc2626" if meta["run_mode"] == "attacks" else "#16a34a"))
    st.html(f'<div class="psi-prompt"><div class="meta">{"".join(tags)}</div>'
            f'<div class="text">{html.escape((source.get("prompt_text") or "")[-1500:])}</div></div>')

    verdicts = pool["verdicts"]
    counts = {v: sum(x == v for x in verdicts) for v in VERDICT_COLOURS}
    st.html(ui.stat_cards([
        (str(len(verdicts)), "Replays", "stored for this interaction", ui.ACCENT),
        (str(counts["safe"]), "Safe", "judged by the Evaluator", VERDICT_COLOURS["safe"]),
        (str(counts["suspicious"]), "Suspicious", "judged by the Evaluator", VERDICT_COLOURS["suspicious"]),
        (str(counts["malicious"]), "Malicious", "judged by the Evaluator", VERDICT_COLOURS["malicious"]),
        (f"#{pool['first_unsafe']}" if pool["first_unsafe"] else "none", "First unsafe replay",
         "in seed order", "#7c3aed"),
    ]))
    replays = [{"replay_idx": i, "seed": seed, "verdict": v or "not judged", "text": r["response_text"]}
               for i, (r, seed, v) in enumerate(zip(pool["replays"], pool["seeds"], verdicts))]
    with st.container(border=True):
        st.html(verdict_strip(replays))
    if pool["judged"] < len(verdicts):
        st.caption(f"{len(verdicts) - pool['judged']} replay(s) without a usable Evaluator verdict are shown "
                   "in grey (not judged, or judged unknown).")
    if pool["first_unsafe"] and pool["first_unsafe"] > 1:
        st.html(ui.callout(
            "amber", "🔎", f"An examiner who replayed this prompt <b>once</b> would have seen a safe response. "
                           f"The first unsafe response appears at <b>replay {pool['first_unsafe']}</b>, and "
                           f"<b>{pool['unsafe']} of the {len(verdicts)}</b> responses are unsafe."))
    elif pool["first_unsafe"] == 1:
        st.html(ui.callout("red", "⚠️", "The first replay is already judged unsafe."))
    elif pool["judged"]:
        st.html(ui.callout("green", "✅", "No replay of this interaction was judged unsafe."))

    shown = st.segmented_control("Show", ["unsafe responses", "safe responses", "all"], default="all") or "all"
    selected = ([r for r in replays if r["verdict"] in UNSAFE] if shown == "unsafe responses"
                else [r for r in replays if r["verdict"] == "safe"] if shown == "safe responses"
                else replays)
    badge_colour = {"safe": "green", "suspicious": "orange", "malicious": "red"}
    for r in selected[:30]:
        colour = badge_colour.get(r["verdict"], "gray")
        with st.expander(f"Replay {r['replay_idx'] + 1} · seed {r['seed']}  :{colour}-badge[{r['verdict']}]"):
            st.write(r["text"])
    if len(selected) > 30:
        st.caption(f"First 30 of {len(selected)} shown.")


# ------------------------------------------------------------- Investigator tab

def tab_investigator(opts: dict[str, Any]) -> None:
    st.html(ui.section("🕵️", "Repository Investigator",
                       "Give the Investigator a set of stored prompts and one model answer; it reports "
                       "which prompts look malicious and why (thesis 4.4)."))
    rows = get_store().recent_interactions(200)
    if not rows:
        st.info("No stored interaction yet.")
        return
    by_id = {r["prompt_id"]: r for r in rows}
    chosen = st.multiselect(
        "Prompts to investigate", list(by_id),
        format_func=lambda pid: f"{by_id[pid]['record_type']} · {pid[:12]}… · {by_id[pid]['prompt_text'][:60]!r}",
    )
    answer_from = st.selectbox("Answer to analyse (from)", chosen or ["–"])
    if st.button("Investigate") and chosen:
        answer = get_store().get_interaction(answer_from)["response_text"]
        prompts = [{"id": pid, "role": "user", "text": by_id[pid]["prompt_text"]} for pid in chosen]
        with st.spinner("Investigating…"):
            result = investigate(prompts, answer, get_settings().judge_model)
        report = result["json"]
        if not result["ok"]:
            st.error(f"Investigator failed: {result['error']}")
        get_store().add_annotation(get_store().get_interaction(answer_from)["id"], INVESTIGATOR_KIND,
                                   report, get_settings().judge_model)
        c1, c2, c3 = st.columns(3)
        c1.metric("Verdict", report["verdict"])
        c2.metric("Score (1 = safe)", fmt(report["score"]))
        c3.metric("Detected", "yes" if is_detected(report) else "no")
        st.json(report)
        with st.expander("Raw judge text"):
            st.code(result["raw_text"] or "(empty)", language="text")


# -------------------------------------------------------------- Experiments tab

def _log_to(container) -> Any:
    """A ``progress`` callback that writes into a Streamlit container."""
    return lambda message: container.write(message)


def _read_uploads(files) -> pd.DataFrame:
    return pd.concat([pd.read_csv(io.BytesIO(f.getvalue())) for f in files], ignore_index=True)


def tab_experiments(opts: dict[str, Any]) -> None:
    st.html(ui.section("🧬", "Dataset experiments",
                       "Small interactive runs. For the full experiments use the scripts in scripts/ "
                       "(see docs/REPRODUCIBILITY.md)."))
    replays, investigator, analyse, advanced = st.tabs(
        ["Replay robustness", "Investigator (5.4)", "Analyse results", "Advanced"])
    with advanced:
        st.caption("Tools of the pilot study (December 2025) and of the first runs.")
        pilot = st.expander("Pilot investigator (repository Investigator)")
        grouped = st.expander("Grouped evaluator (score uploaded replay files)")

    with replays:
        run_mode = st.selectbox("Run mode", RUN_MODES, key="rr_mode")
        family = st.text_input("Attack family filter (empty = all)", key="rr_family")
        limit = st.number_input("Max scenarios", 1, 1000, 3, key="rr_limit")
        n = st.number_input("Replays per scenario", 2, 20, 10, key="rr_n")
        if st.button("Run replays"):
            with st.status("Running replays…", expanded=True) as status:
                csv_path, jsonl_path = run_replays(
                    ReplayConfig(run_mode=run_mode, attack_filter=family or None, limit=int(limit),
                                 num_replays=int(n), target_model=opts["model"]),
                    progress=_log_to(status))
                status.update(label="Done", state="complete")
            df = pd.read_csv(csv_path)
            st.dataframe(df, width="stretch")
            st.caption(f"Written to `{csv_path}` and `{jsonl_path}`. "
                       "The JSONL file is the input of the Investigator tab.")

    with investigator:
        st.caption("Thesis protocol: one judge investigates each scenario's prompt together with one "
                   "replay output, and returns a binary decision and a maliciousness score.")
        replay_file = st.text_input("Replay JSONL file", key="inv_replays")
        judge = st.text_input("Judge model", get_settings().judge_model, key="inv_judge")
        limit = st.number_input("Max scenarios", 1, 50_000, 5, key="inv_limit")
        output = st.text_input("Output CSV", "results/investigator_ui.csv", key="inv_out")
        if st.button("Run investigator") and replay_file:
            with st.status("Running…", expanded=True) as status:
                run_thesis_investigator(
                    InvestigatorRunConfig(replays=replay_file, output_csv=output, judge_model=judge,
                                          limit=int(limit), workers=4),
                    progress=_log_to(status))
                status.update(label="Done", state="complete")
            st.dataframe(pd.read_csv(output), width="stretch")

    with pilot:
        st.caption("Pilot-study protocol (repository Investigator, December 2025 results).")
        run_mode = st.selectbox("Run mode", RUN_MODES, key="pilot_mode")
        family = st.text_input("Attack family filter (empty = all)", key="pilot_family")
        limit = st.number_input("Max scenarios", 1, 1000, 5, key="pilot_limit")
        judges = st.text_area("Judge models, one per line", get_settings().judge_model, key="pilot_judges")
        no_think = st.checkbox("Disable thinking (Qwen3-style models)", False)
        if st.button("Run pilot investigator"):
            with st.status("Running…", expanded=True) as status:
                path = run_pilot_investigator(
                    PilotInvestigatorConfig(
                        run_mode=run_mode, attack_filter=family or None, limit=int(limit),
                        judge_models=[j.strip() for j in judges.splitlines() if j.strip()],
                        target_model=opts["model"], disable_thinking=no_think),
                    progress=_log_to(status))
                status.update(label="Done", state="complete")
            st.dataframe(pd.read_csv(path), width="stretch")

    with grouped:
        st.caption("Upload replay outputs (JSONL/JSON/CSV with scenario_id, replay_idx and text). "
                   "Every scenario pool is scored with the exact metrics and reviewed by the judge.")
        files = st.file_uploader("Replay files", accept_multiple_files=True, key="grp_files")
        if files and st.button("Evaluate groups"):
            records, errors = parse_replay_files([(f.name, f.getvalue()) for f in files])
            for error in errors:
                st.warning(error)
            with st.spinner(f"Evaluating {len(records)} replays…"):
                scenario_rows, replay_rows = evaluate_groups(
                    records, get_settings().judge_model, get_settings().similarity_threshold)
            scenarios_df, replays_df = pd.DataFrame(scenario_rows), pd.DataFrame(replay_rows)
            st.dataframe(scenarios_df, width="stretch")
            download_csv(scenarios_df, "grouped_scenarios.csv", "Download scenario table")
            download_csv(replays_df, "grouped_replays.csv", "Download replay table")

    with analyse:
        kind = st.radio("Table", ["Replay stability (6.1/6.2)", "Tail-risk (6.3)",
                                  "Investigator (6.4-6.6)", "Pilot detection"], horizontal=True)
        attacks = st.file_uploader("Attacks CSV", type="csv", accept_multiple_files=True, key="an_atk")
        safe = st.file_uploader("Safe CSV (optional for 6.1/6.3)", type="csv",
                                accept_multiple_files=True, key="an_safe")
        if not attacks:
            return
        try:
            if kind.startswith("Replay"):
                st.dataframe(analysis.replay_tables(_read_uploads(attacks)), width="stretch")
                if safe:
                    st.dataframe(analysis.replay_tables(_read_uploads(safe)), width="stretch")
            elif kind.startswith("Tail"):
                atk_df = _read_uploads(attacks)
                for name, table in analysis.tail_risk_tables(atk_df).items():
                    st.markdown(f"**{name}**")
                    st.dataframe(table, width="stretch")
                st.markdown("**by pool size (6.3bis)**")
                st.dataframe(analysis.tail_risk_by_pool_size(atk_df), width="stretch")
                if safe:
                    st.json(analysis.paired_tail_risk(atk_df, _read_uploads(safe)))
            elif kind.startswith("Investigator"):
                if not safe:
                    st.info("Precision needs the safe CSV too.")
                    return
                tables = analysis.investigator_tables({"judge": (_read_uploads(attacks), _read_uploads(safe))})
                for name, table in tables.items():
                    st.markdown(f"**{name}**")
                    st.dataframe(table, width="stretch")
            else:
                frames = [_read_uploads(attacks)] + ([_read_uploads(safe)] if safe else [])
                per_judge, per_family = analysis.pilot_detection_metrics(pd.concat(frames, ignore_index=True))
                st.dataframe(per_judge, width="stretch")
                st.dataframe(per_family, width="stretch")
        except (KeyError, ValueError) as exc:
            st.error(f"These files do not fit this analysis: {exc}")


# ----------------------------------------------------------------- Evidence tab

EVIDENCE_COLUMNS = ["id", "created_at", "record_type", "model_used", "prompt_text", "prompt_id", "chain_hash"]


def chain_html(records: list[dict[str, Any]], problems: list[str], limit: int = 6) -> str:
    """The last records of a store as linked blocks; records named in a problem are red."""
    bad = {int(m) for p in problems for m in re.findall(r"interaction (\d+)", p)}
    ids = [r["id"] for r in records]
    # Show the last records, or the first altered one and the records around it.
    start = max(0, len(records) - limit)
    if bad and min(bad) in ids:
        start = min(start, max(0, ids.index(min(bad)) - 1))
    shown = records[start:start + limit]
    blocks = []
    if start:
        blocks.append(f'<div class="psi-more">… {start} earlier</div>')
    for i, r in enumerate(shown):
        broken = r["id"] in bad
        prompt = html.escape((r.get("prompt_text") or "")[-70:].replace("\n", " "))
        prev = html.escape((r.get("prev_chain_hash") or "")[:8])
        state = "⚠ altered" if broken else ("🔏 signed" if r.get("signature") else "not signed")
        blocks.append(
            f'<div class="psi-block{" bad" if broken else ""}">'
            f'<div class="h"><span>#{r["id"]} · {html.escape(r["record_type"])}</span><span>{state}</span></div>'
            f'<div class="p">…{prompt}</div>'
            f'<div class="m">prev {prev}…</div>'
            f'<div class="m">hash {html.escape(r["chain_hash"][:8])}…</div></div>')
        if i < len(shown) - 1:
            link_bad = broken or shown[i + 1]["id"] in bad
            blocks.append(f'<div class="psi-link{" bad" if link_bad else ""}">→</div>')
    if start + limit < len(records):
        blocks.append(f'<div class="psi-more">… {len(records) - start - limit} later</div>')
    return f'<div class="psi-chain">{"".join(blocks)}</div>'


def store_records(db_path: str | Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    try:
        return read_rows(conn)
    finally:
        conn.close()


def show_report(report, what: str) -> None:
    if report.ok:
        st.success(f"**{what}**: all {report.checked} records verified, no problem found.", icon="✅")
    else:
        st.error(f"**{what}**: {len(report.problems)} problem(s) in {report.checked} records.\n\n"
                 + "\n".join(f"- {problem}" for problem in report.problems), icon="🚨")


def tab_evidence() -> None:
    settings = get_settings()
    store = get_store()
    st.html(ui.section("🔏", "Evidence store",
                       "Append-only: UPDATE and DELETE are refused by triggers. Each record is chained "
                       "to the previous one and its chain hash is signed (thesis 4.2)."))
    st.caption(f"Database: `{settings.db_path}`")

    counts = store.counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("Interactions", counts["total"])
    c2.metric("Live captures", counts["by_type"].get("live", 0))
    c3.metric("Replays and attacks", counts["by_type"].get("replay", 0) + counts["by_type"].get("attack", 0))

    df = pd.DataFrame(store.recent_interactions(500))
    if df.empty:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1], vertical_alignment="center")
            c1.markdown("**The store is empty.** Load real interactions of the version 2 runs "
                        "(no model needed), or capture one in the Capture tab.")
            if c2.button("Load the examples here", key="ev_load", type="primary", width="stretch"):
                load_examples_into_store()
                st.rerun()
    else:
        shown = df[[c for c in EVIDENCE_COLUMNS if c in df.columns]]
        st.dataframe(shown, width="stretch", hide_index=True)
        download_csv(df, "psi_evidence.csv")
        with st.expander("Inspect a record"):
            pid = st.selectbox("Record", df["prompt_id"],
                               format_func=lambda v: f"{v[:16]}… · "
                               f"{df.loc[df['prompt_id'] == v, 'prompt_text'].iloc[0][:60]!r}")
            row = store.get_interaction(pid)
            st.json({**row, "canonical_payload": json.loads(row["canonical_payload"]),
                     "metadata_json": json.loads(row["metadata_json"])})
            st.write("Annotations", store.annotations_for(row["id"]))

    st.html(ui.section("✅", "Verify integrity",
                       "Recomputes every hash, walks the hash chain and checks every signature against a "
                       "trusted public key. Anchors (heads of the chain kept outside the database) also "
                       "reveal truncation and rewriting with a stolen key."))
    local_key = Path(public_key_path(settings.signing_key_path))
    col1, col2 = st.columns(2)
    uploaded_key = col1.file_uploader("Trusted public key (.pub.pem)", key="pubkey",
                                      help=f"Without one, the key of this installation is used: `{local_key}`.")
    uploaded_anchors = col2.file_uploader("Anchors (.json, optional)", key="anchors",
                                          accept_multiple_files=True)
    b1, b2 = st.columns(2)
    if b1.button("Verify the evidence store", type="primary"):
        trusted = (uploaded_key.getvalue().decode("ascii") if uploaded_key
                   else local_key.read_text() if local_key.exists() else None)
        anchors = [json.loads(f.getvalue()) for f in uploaded_anchors or []]
        st.session_state["verify_report"] = store.verify(trusted, anchors=anchors)
    if "verify_report" in st.session_state:  # kept when another button reruns the page
        report = st.session_state["verify_report"]
        show_report(report, "Verification")
        st.html(chain_html(store_records(settings.db_path), report.problems))
    b2.download_button("Download an anchor of the current head",
                       json.dumps(store.anchor(), indent=2).encode("utf-8"), "psi_anchor.json",
                       "application/json",
                       help="Keep this file outside the system (or have it time-stamped with "
                            "scripts/anchor_evidence.py): it fixes the history up to now.")

    st.html(ui.section("🧨", "Tampering demonstration",
                       "Tampers with a temporary copy of the store, as an attacker who can edit the SQLite "
                       "file but knows neither the private key nor the anchor would do, then verifies the "
                       "copy. The store itself is never modified."))
    if df.empty:
        st.info("Capture at least one interaction first.")
        return
    ids = df.sort_values("id")["id"].tolist()
    attack_name = st.selectbox("Tampering", list(TAMPERING))
    target = st.selectbox("Record to tamper with", ids,
                          disabled=attack_name.startswith("Delete the last"))
    if st.button("Tamper with a copy and verify it", type="primary"):
        trusted = local_key.read_text() if local_key.exists() else None
        anchor = store.anchor()
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "tampered.sqlite"
            shutil.copy(settings.db_path, copy)
            conn = sqlite3.connect(copy)
            try:
                conn.execute("DROP TRIGGER IF EXISTS interactions_no_update")
                conn.execute("DROP TRIGGER IF EXISTS interactions_no_delete")
                records = read_rows(conn)

                class _OneRecord:  # the truncation removes exactly one record
                    def randint(self, a, b):
                        return 1

                TAMPERING[attack_name](records, ids.index(target), _OneRecord(), None)
                write_rows(conn, records)
                conn.commit()
            finally:
                conn.close()
            report = type(store)(copy).verify(trusted, anchors=[anchor])
            tampered = store_records(copy)
        problems = report.problems
        st.session_state.setdefault("tamper_tests", []).append({
            "time": pd.Timestamp.now(tz="UTC").strftime("%H:%M:%S"), "name": attack_name,
            "record": "last" if attack_name.startswith("Delete the last") else target,
            "detected": not report.ok, "problem": problems[0] if problems else ""})
        show_report(report, "Tampered copy")
        st.markdown("**Tampered copy** (records named by a problem are in red)")
        st.html(chain_html(tampered, report.problems))
        if any(p.startswith("anchor") for p in report.problems):
            st.caption("Only the anchor kept outside the database reveals this change.")


# --------------------------------------------------------------- Overview tab

def tab_overview() -> None:
    st.html(ui.section("🧭", "What PSI does",
                       "Every prompt–response interaction with a large language model is treated as a "
                       "forensic artefact. After an incident, PSI helps an examiner answer three questions."))
    steps = [
        ("1", "Record", "#2563eb", "Has the logged interaction been changed since it was recorded?",
         "Canonical form, SHA-256 Prompt ID, hash chain, ECDSA signature and external anchors (thesis 4.2).",
         ["Evidence", "Capture"]),
        ("2", "Replay", "#7c3aed", "How does the logged response compare with the responses obtained by "
         "replaying its prompt?",
         "Replays under a documented sampling configuration, one recorded seed per replay; Consistency "
         "and Fidelity (thesis 4.3, 5.2).", ["Replay explorer", "Replay"]),
        ("3", "Investigate", "#db2777", "Does the interaction show signs of an attempted attack?",
         "Evaluator (harm of each replayed response) and Investigator (signs of an attempted attack), "
         "both LLM judges (thesis 4.4).", ["Live results", "Investigator"]),
    ]
    cards = "".join(
        f'<div class="psi-card"><div class="num" style="background:{colour}">{num}</div>'
        f'<h4>{title}</h4><p class="q">{question}</p><p class="d">{detail}</p>'
        f'<div class="tabs">{"".join(f"<span class=psi-tag>{t}</span>" for t in tabs)}</div></div>'
        for num, title, colour, question, detail, tabs in steps)
    st.html(f'<div class="psi-grid">{cards}</div>')

    st.html(ui.section("🧪", "From an interaction to a report"))
    flow = [("Interaction", "prompt, context, response"), ("Evidence record", "hash, chain, signature"),
            ("Replay pool", "100 seeds, same configuration"), ("Evaluator", "verdict on each response"),
            ("Investigator", "signs of an attack, report")]
    parts = []
    for i, (name, what) in enumerate(flow):
        parts.append(f'<div class="step"><b>{name}</b><span>{what}</span></div>')
        if i < len(flow) - 1:
            parts.append('<div class="arrow">→</div>')
    st.html(f'<div class="psi-flow">{"".join(parts)}</div>')

    st.html(ui.section("🗺️", "Suggested tour",
                       "Keep Live results open in a second browser tab: it updates while the tests run."))
    st.html('<div class="psi-tour">'
            '<div><b>Capture</b>: send a prompt (or load real examples) and store it as signed evidence.</div>'
            '<div><b>Replay</b>: run a replay pool on a stored interaction, the Evaluator judging each '
            'response.</div>'
            '<div><b>Replay explorer</b>: see how the replays of that interaction mix safe and unsafe '
            'responses.</div>'
            '<div><b>Evidence</b>: verify the store, then tamper with a copy and see which check detects '
            'the change.</div></div>')


# ------------------------------------------------------------ Live results tab
#
# Everything below is computed from the evidence store at each refresh: the
# records and judgements written by the tests run in this interface (or by any
# script writing to the same database). Nothing is read from precomputed files.

REFRESH_SECONDS = 3
EVALUATOR_KIND = "replay_safety"      # verdict of the judge on one replayed response
INVESTIGATOR_KIND = "investigator"    # Investigator report on an interaction


def store_annotations(db_path: str | Path) -> list[dict[str, Any]]:
    """Every judgement of the store, with the record it is attached to."""
    path = Path(db_path)
    if not path.exists():
        return []
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT a.id, a.interaction_id, a.created_at, a.kind, a.model, a.result_json, "
            "i.prompt_id, i.record_type FROM annotations a JOIN interactions i ON i.id = a.interaction_id "
            "ORDER BY a.id").fetchall()
    except sqlite3.OperationalError:  # store not initialised yet
        return []
    finally:
        conn.close()
    out = []
    for r in rows:
        item = dict(r)
        try:
            item["result"] = json.loads(item.pop("result_json") or "{}")
        except json.JSONDecodeError:
            item["result"] = {}
        out.append(item)
    return out


def replay_pools(records: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group the stored replays by the interaction they replay, with their verdicts."""
    verdict_of = {a["interaction_id"]: str(a["result"].get("verdict") or "unknown")
                  for a in annotations if a["kind"] == EVALUATOR_KIND}
    by_prompt = {r["prompt_id"]: r for r in records}
    pools: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        if r["record_type"] != "replay":
            continue
        source = json.loads(r.get("metadata_json") or "{}").get("source_prompt_id")
        if source:
            pools.setdefault(source, []).append(r)
    out = []
    for source, replays in pools.items():
        replays.sort(key=lambda r: r["id"])
        seeds = [json.loads(r.get("canonical_payload") or "{}").get("params", {}).get("seed") for r in replays]
        verdicts = [verdict_of.get(r["id"]) for r in replays]
        judged = [v for v in verdicts if v in VERDICT_COLOURS]  # "unknown": the judge gave no usable verdict
        unsafe_idx = [i for i, v in enumerate(verdicts) if v in UNSAFE]
        src = by_prompt.get(source, {})
        out.append({"source": source, "source_record": src, "replays": replays, "seeds": seeds,
                    "verdicts": verdicts, "judged": len(judged), "unsafe": len(unsafe_idx),
                    "first_unsafe": unsafe_idx[0] + 1 if unsafe_idx else None})
    out.sort(key=lambda p: -p["replays"][-1]["id"])
    return out


def live_state() -> dict[str, Any]:
    """Counts, judgements, integrity and tampering tests, read now."""
    settings = get_settings()
    store = get_store()
    records = store_records(settings.db_path) if Path(settings.db_path).exists() else []
    annotations = store_annotations(settings.db_path)
    local_key = Path(public_key_path(settings.signing_key_path))
    report = store.verify(local_key.read_text() if local_key.exists() else None) if records else None
    tamper = st.session_state.get("tamper_tests", [])
    return {"records": records, "annotations": annotations, "report": report,
            "pools": replay_pools(records, annotations), "tamper": tamper,
            "by_type": {t: sum(r["record_type"] == t for r in records) for t in ("live", "replay", "attack")}}


def live_hero() -> None:
    state = live_state()
    report = state["report"]
    integrity = ("–" if report is None else "intact" if report.ok
                 else f"{len(report.problems)} problem(s)")
    tamper = state["tamper"]
    pools = [p for p in state["pools"] if p["judged"]]
    chips = [("🔏 SHA-256 · hash chain · ECDSA", False), ("🔁 replays under recorded seeds", False),
             ("⚖️ Evaluator + Investigator judges", False), (f"● live · refreshed every {REFRESH_SECONDS} s", False)]
    if get_settings().live_model.startswith("sim:"):
        chips.append(("Simulator mode: the model only echoes the prompt", True))
    st.html(ui.hero(
        "Prompt Scene", "Investigation",
        "Record, replay and investigate interactions with large language models. "
        "The numbers below come from the tests run in this session's evidence store.",
        chips,
        [(str(len(state["records"])), "records in the evidence store"),
         (str(state["by_type"]["replay"]), "replays stored"),
         (integrity, "integrity of the store, checked now"),
         (f"{sum(t['detected'] for t in tamper)} / {len(tamper)}" if tamper else "–",
          "tampering tests detected"),
         (f"{sum(p['unsafe'] > 0 for p in pools)} / {len(pools)}" if pools else "–",
          "replayed prompts with an unsafe replay")]))


def tab_live() -> None:
    st.html(ui.section("📡", "Live results",
                       "Computed from the evidence store at each refresh: every capture, replay, judgement "
                       "and tampering test run in this interface appears here as soon as it is stored."))
    auto = st.toggle("Refresh automatically", True, key="live_auto",
                     help=f"Re-reads the store every {REFRESH_SECONDS} seconds.")
    if auto:
        st.fragment(run_every=REFRESH_SECONDS)(live_panel)()
    else:
        live_panel()


def live_panel() -> None:
    state = live_state()
    records, annotations, report = state["records"], state["annotations"], state["report"]
    if not records:
        st.html(ui.callout("blue", "ℹ️", "The evidence store is empty. Capture an interaction, load the "
                                         "examples in the Evidence tab, or replay a prompt: the results "
                                         "appear here as the tests run."))
        return
    evaluator = [a for a in annotations if a["kind"] == EVALUATOR_KIND]
    investigator = [a for a in annotations if a["kind"] == INVESTIGATOR_KIND]
    integrity_colour = VERDICT_COLOURS["safe"] if report and report.ok else VERDICT_COLOURS["malicious"]
    st.html(ui.stat_cards([
        (str(len(records)), "Records", "captures, replays and attacks", ui.ACCENT),
        (str(state["by_type"]["replay"]), "Replays", f"of {len(state['pools'])} prompt(s)", "#7c3aed"),
        (str(len(evaluator)), "Replays judged", "Evaluator verdicts stored", "#0891b2"),
        (str(len(investigator)), "Investigator reports", "stored with their record", "#db2777"),
        ("intact" if report.ok else f"{len(report.problems)} problem(s)", "Integrity",
         f"{report.checked} records verified now", integrity_colour),
    ]))

    col1, col2 = st.columns(2)
    with col1, st.container(border=True):
        st.markdown("##### Evaluator verdicts on replayed responses")
        counts = {v: sum(str(a["result"].get("verdict")) == v for a in evaluator)
                  for v in (*VERDICT_COLOURS, "unknown")}
        if evaluator:
            data = pd.DataFrame({"verdict": list(counts), "replays": list(counts.values())})
            chart = alt.Chart(data).mark_bar(cornerRadiusEnd=6).encode(
                x=alt.X("replays:Q", title="replayed responses", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("verdict:N", sort=list(counts), title=None),
                color=alt.Color("verdict:N", legend=None, scale=alt.Scale(
                    domain=list(counts), range=[*VERDICT_COLOURS.values(), ui.UNKNOWN])),
                tooltip=["verdict", "replays"]).properties(height=190)
            st.altair_chart(chart, width="stretch")
        else:
            st.caption("No judged replay yet. In the Replay tab, run a replay pool with the Evaluator on.")
    with col2, st.container(border=True):
        st.markdown("##### Investigator reports")
        if investigator:
            detected = sum(is_detected(a["result"]) for a in investigator)
            st.html(ui.stat_cards([
                (str(detected), "flagged as an attempted attack", "", VERDICT_COLOURS["malicious"]),
                (str(len(investigator) - detected), "not flagged", "", VERDICT_COLOURS["safe"])]))
            last = investigator[-1]
            st.caption(f"Last report: record #{last['interaction_id']} · verdict "
                       f"{last['result'].get('verdict')} · {last['created_at']}")
        else:
            st.caption("No report yet. Run the Investigator in the Capture or Investigator tab.")

    pools = state["pools"]
    if pools:
        with st.container(border=True):
            st.markdown("##### Replay pools")
            table = pd.DataFrame([{
                "Prompt": (p["source_record"].get("prompt_text") or "")[-80:].replace("\n", " "),
                "Replays": len(p["replays"]),
                "Judged": p["judged"],
                "Unsafe": p["unsafe"],
                "Unsafe share": 100 * p["unsafe"] / p["judged"] if p["judged"] else None,
                "First unsafe replay": p["first_unsafe"],
            } for p in pools])
            st.dataframe(table, width="stretch", hide_index=True, column_config={
                "Unsafe share": st.column_config.ProgressColumn("Unsafe share", format="%.0f %%",
                                                                min_value=0, max_value=100)})
            judged = [p for p in pools if p["judged"]]
            if judged:
                st.caption(f"{sum(p['unsafe'] > 0 for p in judged)} of {len(judged)} judged prompt(s) have at "
                           "least one unsafe replay. Open the Replay explorer to see each pool.")

    tamper = state["tamper"]
    if tamper:
        with st.container(border=True):
            st.markdown("##### Tampering tests of this session")
            st.dataframe(pd.DataFrame([{
                "Time": t["time"], "Tampering": t["name"], "Record": t["record"],
                "Detected": "yes" if t["detected"] else "NO", "First problem found": t["problem"]}
                for t in reversed(tamper)]), width="stretch", hide_index=True)

    with st.container(border=True):
        st.markdown("##### Latest activity")
        events = [{"time": r["created_at"], "event": f"{r['record_type']} record #{r['id']} stored",
                   "detail": (r.get("prompt_text") or "")[-70:].replace("\n", " ")} for r in records[-15:]]
        events += [{"time": a["created_at"], "event": f"{a['kind'].replace('_', ' ')} on record #{a['interaction_id']}",
                    "detail": str(a["result"].get("verdict") or a["result"].get("summary") or "")[:70]}
                   for a in annotations[-15:]]
        events.sort(key=lambda e: e["time"], reverse=True)
        st.dataframe(pd.DataFrame(events[:15]), width="stretch", hide_index=True)
        st.caption(f"Last refresh: {pd.Timestamp.now(tz='UTC').strftime('%H:%M:%S')} UTC")


# ------------------------------------------------------------------------ main

def main() -> None:
    st.set_page_config(page_title="PSI · Prompt Forensics", page_icon="🔎", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(ui.CSS, unsafe_allow_html=True)
    opts = sidebar()
    st.fragment(run_every=REFRESH_SECONDS)(live_hero)()
    tabs = st.tabs([":material/explore: Overview", ":material/monitoring: Live results",
                    ":material/edit_note: Capture", ":material/autorenew: Replay",
                    ":material/replay: Replay explorer", ":material/policy: Investigator",
                    ":material/verified_user: Evidence", ":material/swords: Red team",
                    ":material/science: Experiments"])
    with tabs[0]:
        tab_overview()
    with tabs[1]:
        tab_live()
    with tabs[2]:
        tab_capture(opts)
    with tabs[3]:
        tab_replay(opts)
    with tabs[4]:
        tab_replay_explorer()
    with tabs[5]:
        tab_investigator(opts)
    with tabs[6]:
        tab_evidence()
    with tabs[7]:
        tab_red_team(opts)
    with tabs[8]:
        tab_experiments(opts)


main()
