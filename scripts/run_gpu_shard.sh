#!/usr/bin/env bash
# Version 2 runs on one GPU: one shard of the plan written by plan_gpu_runs.py.
#
# Per shard, in order:
#   1. replays with the victim model (deep pools: 100 replays, shallow: 10),
#      with Consistency / Fidelity (cosine + BERTScore);
#   1a. 500 deep-pool prompts (both shards) replayed again as is and with
#      extra whitespace, to test whether the canonical form may drop it;
#   1b. replays of the deep-pool prompts with a second victim (Llama-3.1-8B,
#      10 replays), to check that the results do not hold for one model only;
#   2. Evaluator (Qwen3-14B) on every replay of both victims;
#   3. LLM-as-Investigator, five judges, one served at a time.
#
# Run from the repository root on a pod with one GPU (80 GB) and 150 GB of
# disk, one pod per shard, in the background so that a closed terminal does
# not stop it:
#
#     export HF_TOKEN=hf_...
#     nohup bash scripts/run_gpu_shard.sh 0 > shard0.log 2>&1 &     # pod 1
#     nohup bash scripts/run_gpu_shard.sh 1 > shard1.log 2>&1 &     # pod 2
#
# The script can be started again after a stop: finished steps are skipped
# and the runs resume from their output files. At the end, the results of the
# shard are packed in results/v2/shard<k>.tar.gz.
#
# Backup: every 30 minutes and at the end, the results of the shard are
# copied to a private dataset repository on the Hugging Face Hub
# (<user>/prompt-forensics-runs-v2, or HF_BACKUP_REPO), so that nothing is
# lost with a pod. HF_TOKEN then needs the write permission; BACKUP=0 turns
# it off.
#
# Settings (environment): SHARDS (2), DEEP (5000 prompts), DEEP_REPLAYS (100),
# REPLAYS (10), VICTIM2 (empty to skip step 1b), VICTIM2_REPLAYS (10),
# SKIP_INSTALL=1, JUDGES (space-separated model ids), BACKUP (1),
# HF_BACKUP_REPO, BACKUP_EVERY (seconds, 1800).
set -euo pipefail

SHARD=${1:?usage: run_gpu_shard.sh SHARD (0, 1, ...)}
SHARDS=${SHARDS:-2}
DEEP=${DEEP:-5000}
DEEP_REPLAYS=${DEEP_REPLAYS:-100}
REPLAYS=${REPLAYS:-10}
ROOT=data/scenarios_v2
PLAN=results/v2/plan
OUT=results/v2/shard$SHARD
VICTIM=mistralai/Mistral-7B-Instruct-v0.3
# Better aligned than Mistral-7B and the usual victim of jailbreak benchmarks.
VICTIM2=${VICTIM2-meta-llama/Llama-3.1-8B-Instruct}
VICTIM2_REPLAYS=${VICTIM2_REPLAYS:-10}
EVALUATOR=Qwen/Qwen3-14B
# Investigator judges of the thesis; the Evaluator model comes first so that
# it is served only once.
JUDGES=${JUDGES:-"Qwen/Qwen3-14B google/gemma-3-12b-it mistralai/Ministral-8B-Instruct-2410 meta-llama/Llama-3.1-8B-Instruct microsoft/Phi-4-reasoning"}

export LIVE_API_BASE=http://127.0.0.1:8000/v1
export EMBED_DEVICE=cuda EMBED_BATCH_SIZE=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/steps.log"; }
done_step() { [ -f "$OUT/.done_$1" ]; }
mark_done() { touch "$OUT/.done_$1"; log "done: $1"; }

# ---- Environments (same as benchmark_gpu.sh) -----------------------------------
VLLM_VERSION=${VLLM_VERSION:-0.10.1.1}
make_venv() {
    [ -d "$1" ] && return
    python3 -m venv "$1" || { rm -rf "$1"; apt-get update -qq && apt-get install -y -qq python3-venv && python3 -m venv "$1"; }
}
make_venv .venv-vllm
make_venv .venv
if [ "${SKIP_INSTALL:-0}" != 1 ] && ! done_step install; then
    .venv-vllm/bin/pip install -q --upgrade pip
    .venv-vllm/bin/pip install -q "vllm==$VLLM_VERSION" "transformers<5" hf_transfer
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q -e ".[ml]" hf_transfer
    mark_done install
fi
# shellcheck disable=SC1091
. .venv/bin/activate
if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is not set (export HF_TOKEN=hf_..., no space after =)"; exit 1
fi

# Fail now, not after 40 hours, if a gated model has not been accepted on
# huggingface.co with this token.
python - $VICTIM $VICTIM2 $JUDGES <<'EOF'
import sys
from huggingface_hub import hf_hub_download
missing = []
for model in dict.fromkeys(sys.argv[1:]):
    try:
        hf_hub_download(model, "config.json")
    except Exception as exc:  # gated, unknown or no network
        missing.append(f"{model}: {type(exc).__name__}")
if missing:
    sys.exit("no access to:\n  " + "\n  ".join(missing) + "\naccept the licence on huggingface.co with the account of HF_TOKEN")
EOF

serve() {  # model, GPU memory share, extra vllm options
    .venv-vllm/bin/vllm serve "$1" --port 8000 --gpu-memory-utilization "$2" --max-model-len 8192 ${3:-} \
        > "$OUT/vllm_${1##*/}.log" 2>&1 &
    VLLM_PID=$!
    VLLM_LOG="$OUT/vllm_${1##*/}.log"
    until curl -sf "$LIVE_API_BASE/models" > /dev/null; do
        kill -0 "$VLLM_PID" 2> /dev/null || { echo "vLLM stopped, see $VLLM_LOG"; exit 1; }
        sleep 5
    done
    log "serving $1"
}
trap '[ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2> /dev/null; [ -n "${BACKUP_PID:-}" ] && kill "$BACKUP_PID" 2> /dev/null' EXIT
alive() {  # after each run: a server that died mid-run leaves failed calls behind
    kill -0 "$VLLM_PID" 2> /dev/null && return 0
    log "vLLM stopped during the last run, see $VLLM_LOG; fix it and relaunch (--resume redoes the failed calls)"
    exit 1
}
stop() {
    kill "$VLLM_PID"
    wait "$VLLM_PID" 2> /dev/null || true
    VLLM_PID=
    sleep 10  # let the GPU memory go
}
forget() {  # free the disk taken by a model that is no longer needed
    rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${1//\//--}"
}

{ nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; echo "vllm $VLLM_VERSION"; git rev-parse HEAD; } \
    > "$OUT/environment.txt"

# ---- 0. Plan (deterministic: both pods build the same one) ----------------------
if ! done_step plan; then
    python scripts/plan_gpu_runs.py --scenario-root "$ROOT" --out "$PLAN" --overwrite \
        --deep "$DEEP" --deep-replays "$DEEP_REPLAYS" --replays "$REPLAYS" --shards "$SHARDS"
    mark_done plan
fi

# ---- Backup to the Hugging Face Hub ---------------------------------------------
backup() {
    [ "${BACKUP:-1}" = 1 ] || return 0
    python scripts/backup_to_hf.py "$OUT" --path-in-repo "shard$SHARD" ${HF_BACKUP_REPO:+--repo "$HF_BACKUP_REPO"} \
        --message "shard $SHARD: $(tail -n 1 "$OUT/steps.log" 2> /dev/null | cut -c1-80)" >> "$OUT/backup.log" 2>&1
}
if [ "${BACKUP:-1}" = 1 ]; then
    # The first copy also checks the token: better to stop now than to find
    # out after two days that nothing was saved.
    log "backup check"
    if ! backup || ! python scripts/backup_to_hf.py "$PLAN" --path-in-repo plan \
            ${HF_BACKUP_REPO:+--repo "$HF_BACKUP_REPO"} >> "$OUT/backup.log" 2>&1; then
        echo "backup to the Hugging Face Hub failed (see $OUT/backup.log): HF_TOKEN needs the write permission, or set BACKUP=0"
        exit 1
    fi
    ( while sleep "${BACKUP_EVERY:-1800}"; do backup || true; done ) &
    BACKUP_PID=$!
fi

POOLS=()  # "<run mode> <replays>" pairs of this shard
for mode in attacks safe; do
    for n in "$DEEP_REPLAYS" "$REPLAYS"; do
        if [ -d "$PLAN/shard$SHARD/${mode}_r$n" ]; then POOLS+=("$mode $n"); fi
    done
done

# ---- 1. Replays + Consistency / Fidelity -------------------------------------------
if ! done_step replays; then
    serve "$VICTIM" 0.45
    for pool in "${POOLS[@]}"; do
        read -r mode n <<< "$pool"
        if [ "$n" -ge 50 ]; then workers="--scenario-workers 4 --replay-workers 32"
        else workers="--scenario-workers 16 --replay-workers 10"; fi
        log "replays $mode x$n"
        # shellcheck disable=SC2086
        python scripts/run_replays.py --run-mode "$mode" --scenario-root "$PLAN/shard$SHARD/${mode}_r$n" \
            --num-replays "$n" --cosine-mapping raw $workers --max-inflight-calls 128 --resume \
            --output-csv "$OUT/replays_${mode}_r$n.csv" --output-jsonl "$OUT/replays_${mode}_r$n.jsonl"
        alive
    done
    # Whitespace experiment: the same prompts as is and with extra spacing
    # that the canonical form removes, same seeds (see experiments/plan.py).
    for mode in attacks safe; do
        for variant in orig pert; do
            root="$PLAN/shard$SHARD/ws_${mode}_$variant"
            [ -d "$root" ] || continue
            log "whitespace $mode $variant"
            python scripts/run_replays.py --run-mode "$mode" --scenario-root "$root" --num-replays "$REPLAYS" \
                --cosine-mapping raw --scenario-workers 16 --replay-workers 10 --max-inflight-calls 128 --resume \
                --output-csv "$OUT/whitespace_${mode}_$variant.csv" --output-jsonl "$OUT/whitespace_${mode}_$variant.jsonl"
            alive
        done
    done
    stop
    forget "$VICTIM"
    mark_done replays
fi

REPLAY_FILES=()
EVAL_FILES=()
for pool in "${POOLS[@]}"; do
    read -r mode n <<< "$pool"
    REPLAY_FILES+=("$OUT/replays_${mode}_r$n.jsonl")
    EVAL_FILES+=("$OUT/replays_${mode}_r$n.jsonl")
done
for f in "$OUT"/whitespace_*.jsonl; do
    # *_pairs.jsonl hold pair similarities, not replays.
    if [ -f "$f" ] && [[ "$f" != *_pairs.jsonl ]]; then EVAL_FILES+=("$f"); fi
done

# ---- 1b. Second victim on the deep-pool prompts -----------------------------------
if [ -n "$VICTIM2" ]; then
    vtag=$(basename "$VICTIM2" | tr 'A-Z.' 'a-z_')
    if ! done_step victim2; then
        serve "$VICTIM2" 0.45
        for mode in attacks safe; do
            [ -d "$PLAN/shard$SHARD/${mode}_r$DEEP_REPLAYS" ] || continue
            log "replays $mode x$VICTIM2_REPLAYS with $VICTIM2"
            python scripts/run_replays.py --run-mode "$mode" --model "$VICTIM2" \
                --scenario-root "$PLAN/shard$SHARD/${mode}_r$DEEP_REPLAYS" --num-replays "$VICTIM2_REPLAYS" \
                --cosine-mapping raw --scenario-workers 16 --replay-workers 10 --max-inflight-calls 128 --resume \
                --output-csv "$OUT/replays_${mode}_r${VICTIM2_REPLAYS}_$vtag.csv" \
                --output-jsonl "$OUT/replays_${mode}_r${VICTIM2_REPLAYS}_$vtag.jsonl"
            alive
        done
        stop  # its weights stay on disk if it is also a judge
        case " $JUDGES " in *" $VICTIM2 "*) ;; *) forget "$VICTIM2" ;; esac
        mark_done victim2
    fi
    for mode in attacks safe; do
        f="$OUT/replays_${mode}_r${VICTIM2_REPLAYS}_$vtag.jsonl"
        if [ -f "$f" ]; then EVAL_FILES+=("$f"); fi
    done
fi

# ---- 2 and 3. Evaluator, then the Investigator judges (first victim only) ----------
for judge in $JUDGES; do
    tag=$(basename "$judge" | tr 'A-Z.' 'a-z_')
    done_step "judge_$tag" && { [ "$judge" != "$EVALUATOR" ] || done_step evaluator; } && continue
    serve "$judge" 0.9 "--guided-decoding-backend outlines"
    if [ "$judge" = "$EVALUATOR" ] && ! done_step evaluator; then
        log "evaluator"
        python scripts/run_evaluator.py --input "${EVAL_FILES[@]}" --judge-model "$EVALUATOR" \
            --workers 64 --resume --output-csv "$OUT/evaluator.csv"
        alive
        mark_done evaluator
    fi
    # Phi-4-reasoning writes its reasoning before the answer; 640 tokens cut
    # it off before the verdict (see docs/CHANGES.md).
    max_tokens=640
    case "$judge" in *reasoning*) max_tokens=4096 ;; esac
    for pool in "${POOLS[@]}"; do
        read -r mode n <<< "$pool"
        log "investigator $tag on $mode x$n"
        python scripts/run_investigator.py --replays "$OUT/replays_${mode}_r$n.jsonl" --judge "$judge" \
            --scenario-root "$ROOT" --max-tokens "$max_tokens" --workers 32 --resume \
            --output-csv "$OUT/investigator_${mode}_r${n}_$tag.csv"
        alive
    done
    stop
    forget "$judge"
    mark_done "judge_$tag"
done

# ---- 4. Pack the results ---------------------------------------------------------------
tar -czf "results/v2/shard$SHARD.tar.gz" -C results/v2 "shard$SHARD" plan
log "finished: results/v2/shard$SHARD.tar.gz"
if [ -n "${BACKUP_PID:-}" ]; then
    kill "$BACKUP_PID" 2> /dev/null || true
    BACKUP_PID=
    backup && log "final backup done" || log "final backup FAILED, see $OUT/backup.log"
fi
