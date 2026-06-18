#!/usr/bin/env bash
# Run ACRP + 2 more v1 reruns sequentially for variance estimation.
set -e
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
COMMON="--reader-model glm-5.1 --judge-model glm-4.7 --char-budget 24000 --K-atoms 200 --concurrency 24"
B=${OUTDIR:-results}

mkdir -p logs

echo "=== ACRP (atoms-as-pointer + chunks-as-content, K_keyfacts=15) ==="
$PY scripts/eval_akcr_acrp.py $COMMON --K-keyfacts 15 \
  --out $B/v2seq_acrp_k15_glm51_b24k.jsonl \
  --summary-out $B/v2seq_acrp_k15_glm51_b24k.summary.json \
  2>&1 | tee logs/v2seq_acrp.log

echo "=== v1 rerun #3 (variance estimation) ==="
$PY scripts/eval_akcr_v2.py --method v1 $COMMON \
  --out $B/v2seq_v1_run3_glm51_b24k.jsonl \
  --summary-out $B/v2seq_v1_run3_glm51_b24k.summary.json \
  2>&1 | tee logs/v2seq_v1_run3.log

echo "=== v1 rerun #4 (variance estimation) ==="
$PY scripts/eval_akcr_v2.py --method v1 $COMMON \
  --out $B/v2seq_v1_run4_glm51_b24k.jsonl \
  --summary-out $B/v2seq_v1_run4_glm51_b24k.summary.json \
  2>&1 | tee logs/v2seq_v1_run4.log

echo "=== done ==="
