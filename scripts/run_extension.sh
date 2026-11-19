#!/usr/bin/env bash
# Runs of docs/PROTOCOL_policies_verdict_chain_fidelity.md on one GPU.
#
# Experiment B2, complete path with chained and signed annotations: the 50
# prompts of the first complete path, alterations T1 to T9.
# Experiment C, contrasted policies: 70 requests under two policies (140
# items), one answer of the victim per pair, then the Investigator Qwen3-14B.
# Experiment D, Fidelity against an answer recorded apart: 200 prompts, the
# recorded answer in a first vLLM process, 20 replays in a second one, the
# Evaluator on all of them, then cosine and BERTScore.
#
#   export HF_TOKEN=hf_...      # write access, for the copy to the Hub
#   nohup bash scripts/run_extension.sh > extension.log 2>&1 &
#   tail -f results/v2/extension/steps.log
#
# One A100 80 GB. Resumable: run it again after a stop. The sample files are
# checked against data/repetition/plan/SHA256SUMS before anything runs.
# Settings (environment): BACKUP (1), HF_BACKUP_REPO, VLLM_VERSION.
set -euo pipefail
VICTIM=mistralai/Mistral-7B-Instruct-v0.3
JUDGE=Qwen/Qwen3-14B
PLAN=data/extension/plan
OUT=results/v2/extension
KEY=results/v2/extension_keys/signing_key.pem   # outside $OUT: never packed nor copied
VLLM_VERSION=${VLLM_VERSION:-0.10.1.1}
export LIVE_API_BASE=http://127.0.0.1:8000/v1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/B2" "$OUT/C" "$OUT/D"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/steps.log"; }
done_step() { [ -f "$OUT/.done_$1" ]; }
mark_done() { touch "$OUT/.done_$1"; log "done: $1"; }
# Needed even with BACKUP=0: Mistral-7B-Instruct-v0.3 is a gated model.
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN is not set (export HF_TOKEN=hf_..., no space after =)"; exit 1; }
( cd "$PLAN" && sha256sum -c --quiet SHA256SUMS ) || { echo "the sample files differ from $PLAN/SHA256SUMS"; exit 1; }
( cd data/repetition/plan && sha256sum -c --quiet SHA256SUMS ) || { echo "the sample of the complete path differs from data/repetition/plan/SHA256SUMS"; exit 1; }

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
    .venv/bin/pip install -q -e ".[ml]" huggingface_hub hf_transfer
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
    VLLM_LOG="$OUT/vllm_${1##*/}_$(date +%H%M%S).log"
    .venv-vllm/bin/vllm serve "$1" --host 127.0.0.1 --port 8000 --gpu-memory-utilization "$2" --max-model-len 8192 ${3:-} \
        > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    SERVE_ARGS=("$@")
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
        log "vLLM stopped during the step (see $VLLM_LOG), restart $try/5"
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

backup() {
    [ "${BACKUP:-1}" = 1 ] || return 0
    python scripts/backup_to_hf.py "$OUT" --path-in-repo extension ${HF_BACKUP_REPO:+--repo "$HF_BACKUP_REPO"} \
        --message "extension: $(tail -n 1 "$OUT/steps.log" 2> /dev/null | cut -c1-80)" >> "$OUT/backup.log" 2>&1
}
if [ "${BACKUP:-1}" = 1 ]; then
    log "backup check"
    backup || { echo "backup to the Hugging Face Hub failed (see $OUT/backup.log): HF_TOKEN needs the write permission, or set BACKUP=0"; exit 1; }
    ( while sleep 1800; do backup || true; done ) &
    BACKUP_PID=$!
fi

# ---- Victim, first server process: the recorded answers of D ---------------------------------
if ! done_step D_record; then
    serve "$VICTIM" 0.9
    log "D: recorded answers ($VLLM_LOG)"
    retry python scripts/run_extension.py fidelity-record --out "$OUT" --workers 32 --server-process "${VLLM_LOG##*/}"
    stop
    mark_done D_record
fi

# ---- Victim, second server process: replays of D, answers of C, capture of B2 ------------------
if ! done_step victim2; then
    serve "$VICTIM" 0.9
    log "D: 20 replays per prompt ($VLLM_LOG)"
    retry python scripts/run_extension.py fidelity-replay --out "$OUT" --workers 32 --server-process "${VLLM_LOG##*/}"
    log "C: answers of the victim"
    retry python scripts/run_extension.py policies-victim --out "$OUT" --workers 32
    log "B2: capture and replays"
    retry python scripts/run_integrated_path.py capture --out "$OUT/B2" --key "$KEY" --workers 32
    stop
    mark_done victim2
fi

# ---- Judge ---------------------------------------------------------------------------------------
if ! done_step judge; then
    serve "$JUDGE" 0.9 "--guided-decoding-backend xgrammar"
    log "D: evaluator"
    retry python scripts/run_extension.py fidelity-judge --out "$OUT" --workers 64
    log "C: investigator"
    retry python scripts/run_extension.py policies-judge --out "$OUT" --workers 32
    log "B2: evaluator and investigator"
    retry python scripts/run_integrated_path.py judge --out "$OUT/B2" --key "$KEY" --workers 32
    stop
    mark_done judge
fi

# ---- No model served -----------------------------------------------------------------------------
if ! done_step B2_finish; then
    log "B2: reports, anchor, verification, alterations"
    python scripts/run_integrated_path.py finish --out "$OUT/B2" --key "$KEY" | tee "$OUT/B2/finish.txt"
    mark_done B2_finish
fi
if ! done_step D_score; then
    log "D: similarity scores"
    python scripts/run_extension.py fidelity-score --out "$OUT"
    mark_done D_score
fi

# ---- Tables, pack and copy -------------------------------------------------------------------------
python scripts/analysis_v2/extension.py --plan "$PLAN" --run "$OUT" --out "$OUT/tables" || log "extension tables FAILED"
python scripts/analysis_v2/integrated_path.py --run "$OUT/B2" --out "$OUT/tables" || log "integrated path tables FAILED"
# The private signing key ($KEY) stays on the pod: it is not packed nor copied to the Hub.
tar -czf results/v2/extension.tar.gz -C results/v2 extension
log "finished: results/v2/extension.tar.gz"
if [ -n "${BACKUP_PID:-}" ]; then
    kill "$BACKUP_PID" 2> /dev/null || true
    BACKUP_PID=
    backup && log "final backup done" || log "final backup FAILED, see $OUT/backup.log"
fi
log "finished: stop the pod"
