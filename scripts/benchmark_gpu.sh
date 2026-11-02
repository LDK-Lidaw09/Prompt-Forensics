#!/usr/bin/env bash
# Speed test on one GPU, before paying for the full version 2 runs.
#
# Replays N attack and N safe scenarios R times with the victim model, scores
# them (cosine + BERTScore), judges every replay with the Evaluator, times each
# step and extrapolates the GPU hours and the cost of the full runs.
#
# Run from the repository root on a pod with one GPU (80 GB or more) and
# 150 GB of disk (two models of about 30 GB each, two Python environments):
#
#     export HF_TOKEN=hf_...          # Mistral-7B-Instruct-v0.3 is gated
#     PRICE=1.79 bash scripts/benchmark_gpu.sh
#
# Settings (environment): N (25 per run mode), R (100 replays), PRICE (USD per
# GPU hour, for the estimate), OUT (results/benchmark), SKIP_INSTALL=1.
set -euo pipefail

N=${N:-25}
R=${R:-100}
PRICE=${PRICE:-1.79}
OUT=${OUT:-results/benchmark}
ROOT=data/scenarios_v2
VICTIM=mistralai/Mistral-7B-Instruct-v0.3
JUDGE=Qwen/Qwen3-14B

export LIVE_API_BASE=http://127.0.0.1:8000/v1
# Embeddings and BERTScore share the GPU with vLLM (which takes 45 % of it
# while the victim runs). 100 replays of up to 512 tokens, scored by several
# scenarios at once, do not fit in larger batches.
export EMBED_DEVICE=cuda EMBED_BATCH_SIZE=16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

rm -rf "$OUT" && mkdir -p "$OUT"  # timings are only valid for a run from scratch

# hf_transfer: RunPod images set HF_HUB_ENABLE_HF_TRANSFER=1, and every
# model download fails without the package.
# Two virtual environments. pip cannot replace the Debian packages of the pod
# image (PyJWT, ...), and vLLM needs other transformers versions than the
# project: gte-large-en-v1.5 breaks under transformers 5.
# VLLM_VERSION default: the last series that still takes the guided_json
# request field used by the judges.
VLLM_VERSION=${VLLM_VERSION:-0.10.1.1}
make_venv() {
    [ -d "$1" ] && return
    python3 -m venv "$1" || { rm -rf "$1"; apt-get update -qq && apt-get install -y -qq python3-venv && python3 -m venv "$1"; }
}
make_venv .venv-vllm
make_venv .venv
if [ "${SKIP_INSTALL:-0}" != 1 ]; then
    .venv-vllm/bin/pip install -q --upgrade pip
    .venv-vllm/bin/pip install -q "vllm==$VLLM_VERSION" "transformers<5" hf_transfer
    .venv/bin/pip install -q --upgrade pip
    .venv/bin/pip install -q -e ".[ml]" hf_transfer
fi
# shellcheck disable=SC1091
. .venv/bin/activate
if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is not set (export HF_TOKEN=hf_..., no space after =)"; exit 1
fi

serve() {  # model, GPU memory share, extra vllm options
    .venv-vllm/bin/vllm serve "$1" --port 8000 --gpu-memory-utilization "$2" --max-model-len 8192 ${3:-} \
        > "$OUT/vllm_${1##*/}.log" 2>&1 &
    VLLM_PID=$!
    until curl -sf "$LIVE_API_BASE/models" > /dev/null; do
        kill -0 "$VLLM_PID" 2> /dev/null || { echo "vLLM stopped, see $OUT/vllm_${1##*/}.log"; exit 1; }
        sleep 5
    done
}

# On any error, do not leave a server holding the GPU.
trap '[ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2> /dev/null' EXIT

stop() {
    kill "$VLLM_PID"
    wait "$VLLM_PID" 2> /dev/null || true
    sleep 10  # let the GPU memory go
}

{ nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; echo "vllm $VLLM_VERSION"; } | tee "$OUT/gpu.txt"

# ---- 1. Replays + Consistency / Fidelity -------------------------------------
serve "$VICTIM" 0.45
start=$(date +%s)
for mode in attacks safe; do
    python scripts/run_replays.py --run-mode "$mode" --scenario-root "$ROOT" --split dev \
        --limit "$N" --num-replays "$R" \
        --scenario-workers 4 --replay-workers 32 --max-inflight-calls 128 \
        --output-csv "$OUT/replays_$mode.csv" --output-jsonl "$OUT/replays_$mode.jsonl"
done
echo "replays_s=$(( $(date +%s) - start ))" > "$OUT/timings.env"
stop

# BERTScore rate alone, to separate it from the generation time above.
python - "$OUT" <<'EOF' >> "$OUT/timings.env"
import json, sys, time
from prompt_forensics.metrics import similarity
rows = [json.loads(line) for line in open(f"{sys.argv[1]}/replays_attacks.jsonl")]
texts = [r["text"] for r in rows if (r.get("text") or "").strip()][:100]
pairs = [(a, b) for i, a in enumerate(texts) for b in texts[i + 1:]]
start = time.time()
similarity.bertscore([a for a, _ in pairs], [b for _, b in pairs])
print(f"bert_pairs_per_s={len(pairs) / (time.time() - start):.1f}")
EOF

# ---- 2. Evaluator on every replay ----------------------------------------------
# The judges ask for JSON with guided_json and the outlines backend; the
# engine must use the same backend.
serve "$JUDGE" 0.9 "--guided-decoding-backend outlines"
start=$(date +%s)
python scripts/run_evaluator.py --input "$OUT/replays_attacks.jsonl" "$OUT/replays_safe.jsonl" \
    --judge-model "$JUDGE" --workers 64 --output-csv "$OUT/evaluator.csv"
echo "evaluator_s=$(( $(date +%s) - start ))" >> "$OUT/timings.env"
stop

# ---- 3. Extrapolation ------------------------------------------------------------
python - "$OUT" "$N" "$R" "$PRICE" <<'EOF' | tee "$OUT/estimate.txt"
import csv, sys
out, n, r, price = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
t = dict(line.strip().split("=") for line in open(f"{out}/timings.env") if "=" in line)
replays_s, evaluator_s, bert_rate = float(t["replays_s"]), float(t["evaluator_s"]), float(t["bert_pairs_per_s"])

judged = list(csv.DictReader(open(f"{out}/evaluator.csv")))
failed = sum(row.get("judge_ok") not in ("True", "true", "1") for row in judged)
replays = 2 * n * r
pairs = lambda k: k * (k - 1) // 2 + (k - 1)          # Consistency pairs + Fidelity
gen_s = max(replays_s - 2 * n * pairs(r) / bert_rate, 0) / replays
judge_s = evaluator_s / max(len(judged), 1)

print(f"GPU: {open(f'{out}/gpu.txt').read().strip()}")
print(f"generation {1 / gen_s:.1f} replays/s, judge {1 / judge_s:.1f} replays/s, BERTScore {bert_rate:.0f} pairs/s")
print(f"judge failures: {failed} of {len(judged)}")

prompts = 21_600 + 19_981  # attack + safe prompts of version 2
options = {  # name: [(prompts, replays per prompt, BERTScore pairs per prompt)]
    "100 replays on 2,000 scenarios, 10 elsewhere": [(2_000, 100, pairs(100)), (prompts - 2_000, 10, pairs(10))],
    "30 replays everywhere, BERTScore vs reference only": [(prompts, 30, 29)],
    "100 replays everywhere": [(prompts, 100, pairs(100))],
}
for name, parts in options.items():
    seconds = sum(p * k * (gen_s + judge_s) + p * b / bert_rate for p, k, b in parts)
    hours = seconds / 3600
    print(f"{name}: {hours:,.0f} GPU hours, about USD {hours * price:,.0f} at {price}/h")
EOF
