#!/usr/bin/env bash
# Run v2 ablations sequentially on full 1169 to avoid rate-limit interference.
# Each run takes ~8 min at concurrency=24.

set -e
cd "$(dirname "$0")/../.."

PY=/opt/anaconda3/bin/python3
SCRIPT=scripts/v3/eval_akcr_v2.py
COMMON="--reader-model glm-5.1 --judge-model glm-4.7 --char-budget 24000 --K-atoms 200 --concurrency 24"
OUTDIR=data/eval/baselines

mkdir -p logs

run() {
  local method="$1"
  local extra="$2"
  local tag="$3"
  echo "=== $tag ==="
  $PY $SCRIPT --method $method $extra $COMMON \
    --out $OUTDIR/v2seq_${tag}_glm51_b24k.jsonl \
    --summary-out $OUTDIR/v2seq_${tag}_glm51_b24k.summary.json \
    2>&1 | tee logs/v2seq_${tag}.log
}

# Order: most likely to do well first (HACS α=0.8, closest to v1) → MAR variants
run hacs "--alpha 0.8" hacs_a08
run hacs "--alpha 0.7" hacs_a07
run mar_sum "" mar_sum
run mar_lse "--tau 5.0" mar_lse_t5

echo "=== sequential v2 ablation done ==="
