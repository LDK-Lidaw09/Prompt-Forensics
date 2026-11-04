#!/usr/bin/env bash
# Sensitivity runs on one GPU: temperature test, exact reproduction with the
# same seed, and the Investigator without the victim's answer.
#
# 1. Victim Mistral-7B, temperature sample (1,000 prompts by default):
#    20 replays at each temperature of TEMPERATURES (0 0.2 0.7 1.0), seeds
#    1234 + i, every other setting as in the main runs; then a second pass at
#    0.2 with the same seeds (SEED_CHECK_REPLAYS, 10), to count the replays
#    that give back exactly the same text.
# 2. Evaluator Qwen3-14B (blind) on every replay of step 1.
# 3. Investigator Qwen3-14B, same prompt and settings as the main runs:
#    a. without the victim's answer on the ablation sample (2,000 prompts);
#    b. with the answer (first non-empty replay at 0.2) on the temperature
#       sample, a baseline from the same run.
#    The main v2 runs give the baseline with the answer on the whole
#    ablation sample (pair by scenario_id).
#
#   export HF_TOKEN=hf_...      # write access, for the copy to the Hub
#   nohup bash scripts/run_sensitivity.sh > sensitivity.log 2>&1 &
#   tail -f results/v2/sensitivity/steps.log
#
# One A100 80 GB, about 6 to 7 hours. Resumable: run it again after a stop.
# Settings (environment): ABLATION (2000), TEMPERATURE_PROMPTS (1000),
# TEMPERATURES ("0 0.2 0.7 1.0"), REPLAYS (20), SEED_CHECK_REPLAYS (10),
# BACKUP (1), HF_BACKUP_REPO.
set -euo pipefail
ABLATION=${ABLATION:-2000}
TEMPERATURE_PROMPTS=${TEMPERATURE_PROMPTS:-1000}
TEMPERATURES=${TEMPERATURES:-"0 0.2 0.7 1.0"}
REPLAYS=${REPLAYS:-20}
SEED_CHECK_REPLAYS=${SEED_CHECK_REPLAYS:-10}
VICTIM=mistralai/Mistral-7B-Instruct-v0.3
JUDGE=Qwen/Qwen3-14B
OUT=results/v2/sensitivity
PLAN=$OUT/plan
VLLM_VERSION=${VLLM_VERSION:-0.10.1.1}
export LIVE_API_BASE=http://127.0.0.1:8000/v1
export EMBED_DEVICE=cuda EMBED_BATCH_SIZE=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/steps.log"; }
done_step() { [ -f "$OUT/.done_$1" ]; }
mark_done() { touch "$OUT/.done_$1"; log "done: $1"; }
# Needed even with BACKUP=0: Mistral-7B-Instruct-v0.3 is a gated model.
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN is not set (export HF_TOKEN=hf_..., no space after =)"; exit 1; }

# ---- Environments (same as run_gpu_shard.sh) ---------------------------------------
make_venv() {
    [ -d "$1" ] && return
    python3 -m venv "$1" || { rm -rf "$1"; apt-get update -qq && apt-get install -y -qq python3-venv && python3 -m venv "$1"; }
}
make_venv .venv-vllm
make_venv .venv
if ! done_step install; then
    .venv-vllm/bin/pip install -q --upgrade pip
    .venv-vllm/bin/pip install -q "vllm==$VLLM_VERSION" "transformers<5" hf_transfer
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q -e ".[ml]" hf_transfer
    mark_done install
fi
# Same detokenizer patch as scripts/run_gpu_shard.sh (vLLM 0.10.1.1 crash).
detok=$(echo .venv-vllm/lib/python3*/site-packages/vllm/v1/engine/detokenizer.py)
if [ -f "$detok" ]; then
    sed -i -e 's/if str(e) != INVALID_PREFIX_ERR_MSG:/if not str(e).startswith(INVALID_PREFIX_ERR_MSG):/' \
        -e 's/DecodeStream(self.skip_special_tokens)/DecodeStream(skip_special_tokens=self.skip_special_tokens)/' "$detok"
fi
# shellcheck disable=SC1091
. .venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1

python - $VICTIM $JUDGE <<'PY'
import sys
from huggingface_hub import hf_hub_download
for model in sys.argv[1:]:
    try:
        hf_hub_download(model, "config.json")
    except Exception as exc:
        sys.exit(f"no access to {model} ({type(exc).__name__}): accept the licence on huggingface.co with the account of HF_TOKEN")
PY

serve() {  # model, GPU memory share, extra vllm options
    .venv-vllm/bin/vllm serve "$1" --host 127.0.0.1 --port 8000 --gpu-memory-utilization "$2" --max-model-len 8192 ${3:-} \
        > "$OUT/vllm_${1##*/}.log" 2>&1 &
    VLLM_PID=$!
    SERVE_ARGS=("$@")
    VLLM_LOG="$OUT/vllm_${1##*/}.log"
    until curl -sf "$LIVE_API_BASE/models" > /dev/null; do
        kill -0 "$VLLM_PID" 2> /dev/null || { echo "vLLM stopped, see $VLLM_LOG"; exit 1; }
        sleep 5
    done
    log "serving $1"
}
trap '[ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2> /dev/null; [ -n "${BACKUP_PID:-}" ] && kill "$BACKUP_PID" 2> /dev/null' EXIT
retry() {  # run a step; if vLLM died during it, restart vLLM and resume the step
    local try
    for try in 1 2 3 4 5; do
        if "$@"; then
            kill -0 "$VLLM_PID" 2> /dev/null && return 0
        elif kill -0 "$VLLM_PID" 2> /dev/null; then
            return 1
        fi
        log "vLLM stopped during the step (see $VLLM_LOG.crash$try), restart $try/5"
        mv "$VLLM_LOG" "$VLLM_LOG.crash$try"
        wait "$VLLM_PID" 2> /dev/null || true
        sleep 10
        serve "${SERVE_ARGS[@]}"
    done
    log "vLLM keeps stopping, giving up"
    exit 1
}
stop() {
    kill "$VLLM_PID"
    wait "$VLLM_PID" 2> /dev/null || true
    VLLM_PID=
    sleep 10
}

{ nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; echo "vllm $VLLM_VERSION"; git rev-parse HEAD; } \
    > "$OUT/environment.txt"

# ---- 0. Plan ---------------------------------------------------------------------------
if ! done_step plan; then
    python scripts/plan_sensitivity.py --out "$PLAN" --overwrite \
        --ablation "$ABLATION" --temperature "$TEMPERATURE_PROMPTS"
    mark_done plan
fi

backup() {
    [ "${BACKUP:-1}" = 1 ] || return 0
    python scripts/backup_to_hf.py "$OUT" --path-in-repo sensitivity ${HF_BACKUP_REPO:+--repo "$HF_BACKUP_REPO"} \
        --message "sensitivity: $(tail -n 1 "$OUT/steps.log" 2> /dev/null | cut -c1-80)" >> "$OUT/backup.log" 2>&1
}
if [ "${BACKUP:-1}" = 1 ]; then
    log "backup check"
    backup || { echo "backup to the Hugging Face Hub failed (see $OUT/backup.log): HF_TOKEN needs the write permission, or set BACKUP=0"; exit 1; }
    ( while sleep 1800; do backup || true; done ) &
    BACKUP_PID=$!
fi

ttag() { echo "t$(echo "$1" | tr -d '.')"; }  # 0.2 -> t02

# ---- 1. Replays at each temperature, then the same-seed pass -----------------------------
if ! done_step replays; then
    serve "$VICTIM" 0.45
    for t in $TEMPERATURES; do
        for mode in attacks safe; do
            log "replays $mode x$REPLAYS at temperature $t"
            retry python scripts/run_replays.py --run-mode "$mode" --scenario-root "$PLAN/temperature_$mode" \
                --temperature "$t" --num-replays "$REPLAYS" --cosine-mapping raw \
                --scenario-workers 8 --replay-workers 20 --max-inflight-calls 128 --resume \
                --output-csv "$OUT/replays_${mode}_$(ttag "$t").csv" --output-jsonl "$OUT/replays_${mode}_$(ttag "$t").jsonl"
        done
    done
    for mode in attacks safe; do
        log "same-seed pass $mode x$SEED_CHECK_REPLAYS at temperature 0.2"
        retry python scripts/run_replays.py --run-mode "$mode" --scenario-root "$PLAN/temperature_$mode" \
            --temperature 0.2 --num-replays "$SEED_CHECK_REPLAYS" --cosine-mapping raw --no-bertscore \
            --scenario-workers 8 --replay-workers 10 --max-inflight-calls 128 --resume \
            --output-csv "$OUT/seedcheck_${mode}_t02.csv" --output-jsonl "$OUT/seedcheck_${mode}_t02.jsonl"
    done
    stop
    mark_done replays
fi

EVAL_FILES=()
for f in "$OUT"/replays_*_t*.jsonl "$OUT"/seedcheck_*_t02.jsonl; do
    if [ -f "$f" ] && [[ "$f" != *_pairs.jsonl ]]; then EVAL_FILES+=("$f"); fi
done

# ---- 2 and 3. Evaluator, then the Investigator (one model served once) -------------------
if ! done_step judge; then
    serve "$JUDGE" 0.9 "--guided-decoding-backend xgrammar"
    if ! done_step evaluator; then
        log "evaluator on ${#EVAL_FILES[@]} files"
        retry python scripts/run_evaluator.py --input "${EVAL_FILES[@]}" --judge-model "$JUDGE" \
            --workers 64 --resume --output-csv "$OUT/evaluator.csv"
        mark_done evaluator
    fi
    for mode in attacks safe; do
        log "investigator without the answer, $mode"
        retry python scripts/run_investigator.py --no-output --run-mode "$mode" \
            --scenario-root "$PLAN/ablation_$mode" --judge "$JUDGE" --workers 32 --resume \
            --output-csv "$OUT/investigator_noout_$mode.csv"
        log "investigator with the answer (temperature 0.2), $mode"
        retry python scripts/run_investigator.py --replays "$OUT/replays_${mode}_t02.jsonl" \
            --scenario-root "$PLAN/temperature_$mode" --judge "$JUDGE" --workers 32 --resume \
            --output-csv "$OUT/investigator_withanswer_$mode.csv"
    done
    stop
    mark_done judge
fi

# ---- 4. Pack and copy ------------------------------------------------------------------------
tar -czf results/v2/sensitivity.tar.gz -C results/v2 sensitivity
log "finished: results/v2/sensitivity.tar.gz"
if [ -n "${BACKUP_PID:-}" ]; then
    kill "$BACKUP_PID" 2> /dev/null || true
    BACKUP_PID=
    backup && log "final backup done" || log "final backup FAILED, see $OUT/backup.log"
fi
log "finished: stop the pod"
