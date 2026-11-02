"""PSI demo application (thesis section 4.5).

A single-file Streamlit interface over the ``prompt_forensics`` package. It
contains no forensic logic of its own: every action below calls a package
function, so what the UI shows is exactly what the experiment scripts compute.

Run it from the repository root::

    streamlit run app/streamlit_app.py

Tabs:

- **Overview**     what PSI does and where each part lives in the interface.
- **Capture**      send a prompt, store it as signed evidence, self-audit it.
- **Red team**     toy adversary that rewrites a prompt into an attack.
- **Replay**       replay a stored prompt, compare it with the original and
                   measure the robustness of a replay pool.
- **Investigator** run the repository Investigator on chosen stored prompts.
- **Experiments**  small runs of the dataset experiments and the tables of
                   chapter 6 from result CSVs.
- **Evidence**     browse, export and verify the evidence store, anchor its
                   head, and tamper with a copy to see the checks at work.
- **Results v2**   the result tables of the version 2 runs (thesis chapter 6).
"""

from __future__ import annotations

import difflib
import io
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Make the package importable without installing it (``pip install -e .``
# does the same thing permanently).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))  # tamper_experiment.py: the tampering attacks

from prompt_forensics.agents.adversary import ATTACK_TEMPLATES, run_attack  # noqa: E402
from prompt_forensics.agents.comparators import judge_replay_safety, judge_replay_similarity  # noqa: E402
from prompt_forensics.agents.repository_investigator import investigate, is_detected  # noqa: E402
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
RESULTS_V2 = ROOT / "results" / "v2" / "tables"
THESIS_VICTIM = "Mistral-7B"  # the victim reported in the thesis

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
    st.subheader("Capture an interaction")
    st.caption("The prompt is sent to the victim model, then stored with its Prompt ID, "
               "content hash, chain hash and ECDSA signature (thesis 4.2).")
    with st.form("capture"):
        prompt = st.text_area("Prompt", height=140)
        col1, col2 = st.columns(2)
        audit = col1.checkbox("Self-audit the answer", True)
        investigator = col2.checkbox("Run the Investigator", False)
        submitted = st.form_submit_button("Send")

    if submitted and prompt.strip():
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
    st.subheader("Toy red-team generator")
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
    st.subheader("Replay a stored prompt")
    st.caption("Replays use the exact stored model input. Each replay is itself stored as a "
               "`replay` record that points to its source.")
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
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Hybrid similarity", fmt(sim["hybrid_similarity"]))
            c2.metric("BERTScore F1", fmt(sim["bertscore_f1"]))
            c3.metric("Safety verdict", safety["verdict"])
            c4.metric("Safety score", fmt(safety["score"]))
            with st.expander("Judge details"):
                st.json({"similarity": sim, "safety": safety})


def replay_pool(source_id: str, opts: dict[str, Any]) -> None:
    """Replay n times with seeds seed, seed+1, ... and compute Consistency and Fidelity."""
    n = st.slider("Replays in the pool", 2, 20, 10)
    st.caption(f"Seeds {opts['seed']} … {opts['seed'] + n - 1}. Replay 1 is the Fidelity reference "
               "(thesis 5.2). The stored original answer is not part of the pool.")
    if st.button("Run replay pool"):
        outputs: list[str | None] = []
        bar = st.progress(0.0)
        for i in range(n):
            try:
                outputs.append(replay_interaction(
                    source_id, model=opts["model"], seed=opts["seed"] + i,
                    temperature=opts["temperature"], max_tokens=opts["max_tokens"])["response"])
            except LLMError as exc:
                st.warning(f"Replay {i + 1} failed: {exc}")
                outputs.append(None)  # kept as missing, never scored as 0
            bar.progress((i + 1) / n)
        with st.spinner("Computing embeddings and BERTScore…"):
            try:
                robust = compute_replay_robustness(outputs, get_settings().similarity_threshold,
                                                   get_settings().similarity_threshold)
            except ImportError as exc:
                st.error(f"Similarity models are not installed ({exc}). "
                         'Run `pip install -e ".[ml]"`. The replays are stored.')
                return
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


# ------------------------------------------------------------- Investigator tab

def tab_investigator(opts: dict[str, Any]) -> None:
    st.subheader("Repository Investigator")
    st.caption("Give the Investigator a set of stored prompts and one model answer; it reports which "
               "prompts look malicious and why (thesis 4.4).")
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
    st.subheader("Dataset experiments")
    st.caption("Small interactive runs. For the full experiments use the scripts in `scripts/` "
               "(see docs/REPRODUCIBILITY.md).")
    replays, investigator, pilot, grouped, analyse = st.tabs(
        ["Replay robustness", "Investigator (5.4)", "Pilot investigator", "Grouped evaluator",
         "Analyse results"])

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


def show_report(report, what: str) -> None:
    if report.ok:
        st.success(f"{what}: all {report.checked} records verified, no problem found.")
    else:
        st.error(f"{what}: {len(report.problems)} problem(s) in {report.checked} records.")
        for problem in report.problems:
            st.markdown(f"- {problem}")


def tab_evidence() -> None:
    settings = get_settings()
    store = get_store()
    st.subheader("Evidence store")
    st.caption(f"`{settings.db_path}` · append-only: UPDATE and DELETE are refused by triggers. "
               "Each record is chained to the previous one and its chain hash is signed (thesis 4.2).")

    counts = store.counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("Interactions", counts["total"])
    c2.metric("Live captures", counts["by_type"].get("live", 0))
    c3.metric("Replays and attacks", counts["by_type"].get("replay", 0) + counts["by_type"].get("attack", 0))

    df = pd.DataFrame(store.recent_interactions(500))
    if df.empty:
        st.info("No stored interaction yet. Capture one in the Capture tab first.")
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

    st.markdown("#### Verify integrity")
    st.caption("Recomputes every hash, walks the hash chain and checks every signature against a "
               "trusted public key. Anchors (heads of the chain kept outside the database) also "
               "reveal truncation and rewriting with a stolen key.")
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
        show_report(store.verify(trusted, anchors=anchors), "Verification")
    b2.download_button("Download an anchor of the current head",
                       json.dumps(store.anchor(), indent=2).encode("utf-8"), "psi_anchor.json",
                       "application/json",
                       help="Keep this file outside the system (or have it time-stamped with "
                            "scripts/anchor_evidence.py): it fixes the history up to now.")

    st.markdown("#### Tampering demonstration")
    st.caption("Tampers with a temporary copy of the store, as an attacker who can edit the SQLite "
               "file but knows neither the private key nor the anchor would do, then verifies the "
               "copy. The store itself is never modified.")
    if df.empty:
        st.info("Capture at least one interaction first.")
        return
    ids = df.sort_values("id")["id"].tolist()
    attack_name = st.selectbox("Tampering", list(TAMPERING))
    target = st.selectbox("Record to tamper with", ids,
                          disabled=attack_name.startswith("Delete the last"))
    if st.button("Tamper with a copy and verify it"):
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
        show_report(report, "Tampered copy")


# --------------------------------------------------------------- Overview tab

def tab_overview() -> None:
    st.markdown(
        "PSI treats every prompt–response interaction with a large language model as a "
        "forensic artefact. After an incident, it helps an examiner answer three questions.")
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("##### 1. Record")
        st.markdown("Has the logged interaction been changed since it was recorded?")
        st.caption("Canonical form, SHA-256 Prompt ID, hash chain, ECDSA signature, external "
                   "anchors (thesis 4.2). Tabs: **Capture**, **Evidence**.")
    with c2.container(border=True):
        st.markdown("##### 2. Replay")
        st.markdown("How does the logged response compare with the responses obtained by "
                    "replaying its prompt?")
        st.caption("Replays under a documented sampling configuration, with one recorded seed per "
                   "replay; Consistency and Fidelity (thesis 4.3, 5.2). Tab: **Replay**.")
    with c3.container(border=True):
        st.markdown("##### 3. Investigate")
        st.markdown("Does the interaction show signs of an attempted attack?")
        st.caption("Evaluator (harm of each replayed response) and Investigator (signs of an "
                   "attempted attack), both LLM judges (thesis 4.4). Tabs: **Investigator**, "
                   "**Experiments**.")
    st.markdown("")
    st.markdown("**Suggested tour.** Capture a prompt, look at its record in **Evidence**, verify "
                "the store, then tamper with a copy and see which check detects the change. "
                "**Results v2** shows the results of the full runs.")


# ------------------------------------------------------------- Results v2 tab

@st.cache_data(show_spinner=False)
def load_table(name: str) -> pd.DataFrame | None:
    path = RESULTS_V2 / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def tab_results() -> None:
    st.subheader("Results of the version 2 runs")
    st.caption(f"{THESIS_VICTIM}-Instruct-v0.3, temperature 0.2, 100 replays per prompt, on 21,600 "
               "attack prompts and 19,981 legitimate prompts. Tables computed by "
               "`scripts/analysis_v2/` from the raw runs; files in `results/v2/tables/`.")
    tail = load_table("evaluator_tail_risk")
    curve = load_table("tail_risk_curve")
    summary = load_table("investigator_summary")
    if tail is None or curve is None or summary is None:
        st.info("The result tables are not in `results/v2/tables/`.")
        return

    tail = tail[(tail["victim"] == THESIS_VICTIM) & (tail["group"] == "all")].set_index("mode")
    first = curve[curve["m"] == 1].set_index("mode")["tail"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Attacks, unsafe with 1 replay", pct(first.get("attacks")))
    c2.metric("Attacks, unsafe within 100 replays", pct(tail.loc["attacks", "tail_unsafe"]))
    c3.metric("Legitimate, unsafe within 100", pct(tail.loc["safe", "tail_unsafe"]))
    c4.metric("Unsafe replays (attacks)", pct(tail.loc["attacks", "rate_unsafe"]))
    st.caption("A prompt counts as unsafe within m replays when the Evaluator judges at least one "
               "of its first m replays suspicious or malicious. Human annotation of a sample shows "
               "that the Evaluator misses many unsafe responses: these rates are lower bounds.")

    st.markdown("#### Share of prompts with at least one unsafe replay, by number of replays")
    chart = curve.pivot(index="m", columns="mode", values="tail").rename(
        columns={"attacks": "attack prompts", "safe": "legitimate prompts"})
    st.line_chart(chart, x_label="replays examined (m)", y_label="share of prompts",
                  color=["#c0392b", "#1f4e79"])  # attack prompts in red, legitimate in blue

    st.markdown("#### Investigator judges")
    st.caption("Decision field of each judge; failed judgements are excluded and counted. "
               "False positives are legitimate prompts flagged as attacks.")
    inv = summary[summary["rule"] == "decision"]
    view = pd.DataFrame({
        "Judge": inv["judge"],
        "Recall": inv["recall"].map(pct),
        "False positives": inv["fpr"].map(lambda v: pct(v, 2)),
        "Precision": inv["precision"].map(pct),
        "F1": inv["f1"].map(pct),
        "Failed (attacks / legitimate)": inv["failed_att"].astype(str) + " / " + inv["failed_safe"].astype(str),
    })
    st.dataframe(view, width="stretch", hide_index=True)

    fp_kind = load_table("investigator_fp_by_kind")
    recall_family = load_table("investigator_recall_by_family")
    col1, col2 = st.columns(2)
    if fp_kind is not None:
        with col1:
            st.markdown("**False positives by kind of legitimate prompt**")
            kinds = fp_kind[(fp_kind["rule"] == "decision")
                            & fp_kind["safe_kind"].isin(["standard", "hard_negative"])]
            table = kinds.pivot(index="judge", columns="safe_kind", values="fpr").rename(
                columns={"standard": "standard", "hard_negative": "resembling attacks"})
            st.dataframe(table.map(lambda v: pct(v, 2)), width="stretch")
    if recall_family is not None:
        with col2:
            st.markdown("**Recall by attack family**")
            fam = recall_family[recall_family["rule"] == "decision"]
            table = fam.pivot(index="family", columns="judge", values="recall")
            st.dataframe(table.map(pct), width="stretch")

    with st.expander("All tables"):
        for path in sorted(RESULTS_V2.glob("*.csv")):
            df = pd.read_csv(path)
            if "victim" in df.columns:
                df = df[df["victim"] == THESIS_VICTIM]
            st.markdown(f"`{path.name}`")
            st.dataframe(df, width="stretch", hide_index=True)


# ------------------------------------------------------------------------ main

def main() -> None:
    st.set_page_config(page_title="PSI · Prompt Forensics", page_icon="🔎", layout="wide")
    st.title("PSI · Prompt Scene Investigation")
    st.caption("Record, replay and investigate interactions with large language models.")
    opts = sidebar()
    if opts["model"].startswith("sim:"):
        st.warning("Simulator mode: the model only echoes the prompt. Hashes, chain, signatures and "
                   "verification are real. For real answers, start LM Studio and choose "
                   "“LM Studio list” in the sidebar, or set LIVE_MODEL.")
    tabs = st.tabs(["Overview", "Capture", "Red team", "Replay", "Investigator", "Experiments",
                    "Evidence", "Results v2"])
    with tabs[0]:
        tab_overview()
    with tabs[1]:
        tab_capture(opts)
    with tabs[2]:
        tab_red_team(opts)
    with tabs[3]:
        tab_replay(opts)
    with tabs[4]:
        tab_investigator(opts)
    with tabs[5]:
        tab_experiments(opts)
    with tabs[6]:
        tab_evidence()
    with tabs[7]:
        tab_results()


main()
