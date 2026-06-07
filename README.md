# Atom-Keyed Chunk Retrieval (AKCR) — long-document QA retrieval (ICMLW)

**What this code does.** AKCR is a retrieval pipeline for **long-document
question answering** (e.g. answering questions over a *whole book*). Its one
idea: **rank on a different unit than you read from.**

1. Split the document into chunks.
2. Extract one **atomic proposition** (a short Davidsonian fact) per chunk and
   embed it — atoms are the *ranking* unit.
3. At query time, rank atoms by cosine to the query, then follow a deterministic
   `atom → source-chunk` map back to the chunks.
4. Pack the top **chunks** (not the atoms) into a character budget and feed them
   to the reader LLM.

Ranking on atoms but reading from chunks closes a substantial fraction of the
gap to the same-reader long-context oracle at a fixed retrieval budget, and beats
a plain chunk-cosine baseline at the same budget. Full numbers and the position
behind them: *"Tokenization Beats Propagation: A Position on Graph Foundation
Models for LLM Retrieval Interfaces"* (GFM @ ICML 2026 workshop).

## When to use this

Reach for AKCR when you have a **single long document** and a reader LLM with a
limited context budget, and you want retrieval that lands on the *readable* span
rather than on isolated facts. The scoring variants below let you trade recall
vs. precision of the packed context.

## Install

```bash
pip install numpy httpx
```

The reader and judge are called over an HTTP API (GLM-4.7 by default; an
OpenAI-style endpoint is also supported for the cross-reader slice). Set
credentials via environment variables or a local `.env` (git-ignored):

```
GLM_URL=...
GLM_API_KEY=...
# optional OpenAI-style endpoint for cross-reader runs:
OPENAI_URL=...
OPENAI_API_KEY=...
```

## Run

`scripts/eval_akcr_v2.py` is the entry point. It supports four chunk-scoring
variants that share the *same* atom/chunk/query embeddings, candidate pool,
packing, reader prompt, and judge — only the scoring step differs:

| `--method` | chunk score |
|---|---|
| `v1`      | atom-best (chunk ranked by its best atom) |
| `mar_sum` | sum of top-K atom scores in the chunk |
| `mar_lse` | log-sum-exp (`--tau`) of top-K atom scores |
| `hacs`    | `alpha·atom-best + (1-alpha)·chunk-cosine` (`--alpha`, default 0.7) |

```bash
python scripts/eval_akcr_v2.py \
  --method hacs \
  --reader-model glm-4.7 \
  --char-budget 24000 \
  --out results/my_run.jsonl \
  --summary-out results/my_run.summary.json
```

Key flags: `--char-budget` (retrieval budget in chars), `--K-atoms` (candidate
atom pool, default 200), `--reader-model` / `--judge-model`, `--concurrency`,
`--max-questions` (smoke-test on a few items first).

### Inputs it expects

Atom/chunk/query embeddings and the NarrativeQA QA file (BGE-M3, V010 atoms).
Defaults point at `data/narrativeqa/processed_v010_full/` and
`data/embeddings/full_book/`; override with the corresponding flags or set
`MEMORYNET_REPO` to a sibling layout. NarrativeQA source text is **not**
redistributed here.

## What's in here

```
scripts/   the pipeline + baselines + table/significance tooling
results/   released per-question predictions (.jsonl) and summaries (.summary.json)
```

| Script | Purpose |
|---|---|
| `eval_akcr_v2.py` | main reader-eval (v1 / mar_sum / mar_lse / hacs) |
| `eval_akcr_acrp.py` | Atom-Conditioned Reader Prompt variant |
| `eval_akcr_v2_permissive.py` | permissive-matching ablation |
| `eval_akcr_e2e.py` | end-to-end AKCR v1 |
| `eval_dosrag_chunks_reader_sweep.py` | chunk-cosine baseline (`DOS+chunks`) |
| `eval_gar_e2e.py` | atoms-as-content baseline (GAR) |
| `compile_v2_table.py`, `compile_wave1_table.py` | build tables from `results/*.summary.json` |
| `compare_v2_to_v1.py`, `significance_test.py` | paired-bootstrap significance |
| `run_v2_sequential.sh`, `run_acrp_and_variance.sh` | batch runners |

Result files are named `full_book_{method}_{reader}_b{budget}.jsonl` with a
matching `.summary.json`.
