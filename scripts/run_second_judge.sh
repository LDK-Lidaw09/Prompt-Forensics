#!/usr/bin/env bash
# Second Evaluator on a sample of 20,203 replays (thesis 6, agreement between judges).
#
# The sample (data/second_judge/sample_20k.jsonl.gz, drawn by
# scripts/analysis_v2/sample_second_judge.py) holds 10,000 attack and 10,000
# safe replays of Mistral-7B, stratified by family x domain (x safe-prompt kind),
# plus the 193 replays of the human validation sheet. They are judged with the
# same blind prompt, guided JSON (xgrammar) and settings as the main Evaluator,
# by a judge of another family (default google/gemma-3-27b-it).
#
#   export HF_TOKEN=hf_...      # write access, for the copy to the Hub
#   nohup bash scripts/run_second_judge.sh > second_judge.log 2>&1 &
#   tail -f results/v2/second_judge/steps.log
#
# One A100 80 GB, about 1 hour. Resumable: run it again after a stop.
set -euo pipefail
JUDGE=${JUDGE:-google/gemma-3-27b-it}
TAG=$(basename "$JUDGE" | tr 'A-Z.' 'a-z_')
OUT=results/v2/second_judge
VLLM_VERSION=${VLLM_VERSION:-0.10.1.1}
export LIVE_API_BASE=http://127.0.0.1:8000/v1
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/steps.log"; }
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN is not set (export HF_TOKEN=hf_..., no space after =)"; exit 1; }

make_venv() {
    [ -d "$1" ] && return
    python3 -m venv "$1" || { rm -rf "$1"; apt-get update -qq && apt-get install -y -qq python3-venv && python3 -m venv "$1"; }
}
make_venv .venv-vllm
make_venv .venv
if [ ! -f "$OUT/.done_install" ]; then
    .venv-vllm/bin/pip install -q --upgrade pip
    .venv-vllm/bin/pip install -q "vllm==$VLLM_VERSION" "transformers<5" hf_transfer
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q -e . huggingface_hub hf_transfer
    touch "$OUT/.done_install"; log "done: install"
fi
# Same detokenizer patch as scripts/run_gpu_shard.sh (vLLM 0.10.1.1 crash).
detok=$(echo .venv-vllm/lib/python3*/site-packages/vllm/v1/engine/detokenizer.py)
if [ -f "$detok" ]; then
    sed -i -e 's/if str(e) != INVALID_PREFIX_ERR_MSG:/if not str(e).startswith(INVALID_PREFIX_ERR_MSG):/' \
        -e 's/DecodeStream(self.skip_special_tokens)/DecodeStream(skip_special_tokens=self.skip_special_tokens)/' "$detok"
fi
. .venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1

python - "$JUDGE" <<'PY'
import sys
from huggingface_hub import hf_hub_download
try:
    hf_hub_download(sys.argv[1], "config.json")
except Exception as exc:
    sys.exit(f"no access to {sys.argv[1]} ({type(exc).__name__}): accept the licence on huggingface.co with the account of HF_TOKEN")
PY

INPUT="$OUT/sample_20k.jsonl"
[ -f "$INPUT" ] || gunzip -c data/second_judge/sample_20k.jsonl.gz > "$INPUT"
{ nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; echo "vllm $VLLM_VERSION"; echo "judge $JUDGE"; git rev-parse HEAD; } > "$OUT/environment.txt"

.venv-vllm/bin/vllm serve "$JUDGE" --port 8000 --gpu-memory-utilization 0.92 --max-model-len 8192 \
    --guided-decoding-backend xgrammar > "$OUT/vllm_$TAG.log" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2> /dev/null || true' EXIT
until curl -sf "$LIVE_API_BASE/models" > /dev/null; do
    kill -0 "$VLLM_PID" 2> /dev/null || { echo "vLLM stopped, see $OUT/vllm_$TAG.log"; exit 1; }
    sleep 5
done
log "serving $JUDGE"

for attempt in 1 2 3; do
    log "evaluator $TAG, attempt $attempt"
    if python scripts/run_evaluator.py --input "$INPUT" --judge-model "$JUDGE" --workers 64 --resume \
        --output-csv "$OUT/evaluator_$TAG.csv"; then
        break
    fi
done
log "done: evaluator $(($(wc -l < "$OUT/evaluator_$TAG.csv") - 1)) rows"
rm -f "$INPUT"
python scripts/backup_to_hf.py "$OUT" --path-in-repo second_judge --message "second judge $TAG" \
    && log "copied to the Hub (second_judge/)"
log "finished: stop the pod"
