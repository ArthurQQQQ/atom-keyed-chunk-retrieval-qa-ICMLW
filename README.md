# AKCR — Tokenization Beats Propagation

Code, results, and paper source for the GFM @ ICML 2026 workshop position paper
**"Tokenization Beats Propagation: A Position on Graph Foundation Models for LLM
Retrieval Interfaces."**

This is a self-contained archive. The headline finding (Table `tab:headline`,
full-book NarrativeQA, `b = 24,000` chars): a cross-reader cost–quality trade —
HACS-AKCR + Opus 4.7 on 24K matches the GLM-4.7 long-context oracle on 78K
(3-run mean **0.6721 ± 0.007**). The same-reader gap (~9pp below the oracle at
matched reader) is reported honestly in §3.2.

## Layout

```
akcr_gfm/
├── main.tex / main.pdf       # the submission (4-page ICML 2026 position paper)
├── references.bib            # bibliography
├── *.sty / *.bst             # ICML 2026 template files
├── scripts/                  # code that produces the paper's tables
└── results/                  # all NarrativeQA result files (.jsonl + .summary.json)
```

## Scripts (what produces the paper)

| Script | Produces |
|---|---|
| `eval_akcr_v2.py` | Main reader-eval: AKCR v1, MAR-sum, MAR-lse, HACS (headline + ablation) |
| `eval_akcr_acrp.py` | Atom-Conditioned Reader Prompt variant |
| `eval_akcr_v2_permissive.py` | Permissive-matching variant for the ablation |
| `eval_akcr_e2e.py` | AKCR v1 (the `AKCR v1` rows) |
| `eval_dosrag_chunks_reader_sweep.py` | `DOS+chunks` chunk-cosine baseline |
| `eval_gar_e2e.py` | `Atoms-as-content` (GAR) baseline |
| `compile_v2_table.py`, `compile_wave1_table.py` | Compile the ablation / headline tables from `results/*.summary.json` |
| `compare_v2_to_v1.py`, `significance_test.py` | Paired-bootstrap significance (ablation table) |
| `run_v2_sequential.sh`, `run_acrp_and_variance.sh` | Batch runners |

Result files in `results/` are named
`full_book_{method}_{reader}_b{budget}.jsonl` with matching `.summary.json`.

> Status: archival snapshot — not under active development.
