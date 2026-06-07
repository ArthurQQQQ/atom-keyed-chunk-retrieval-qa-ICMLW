#!/usr/bin/env python3
"""compile_wave1_table.py
Auto-load every NarrativeQA full_book .summary.json under data/eval/baselines/,
classify by (method, reader, char_budget), and produce a unified table for the
Wave 1 checkpoint and the paper.

Usage:
    /opt/anaconda3/bin/python3 scripts/compile_wave1_table.py [--output table.md]
"""
from __future__ import annotations
import argparse, json, re, statistics
from pathlib import Path

REPO = Path("/Users/arthurqiu/MemoryNet")
BASELINES = REPO / "data/eval/baselines"

# Patterns to identify methods; first match wins
METHOD_PATTERNS = [
    (r"v2seq_hacs_a08_([\w.\-]+)_b24k_run\d+", "AKCR HACS α=0.8"),
    (r"v2seq_hacs_a08_([\w.\-]+)_b24k", "AKCR HACS α=0.8"),
    (r"v2seq_hacs_a07_([\w.\-]+)_b24k", "AKCR HACS α=0.7"),
    (r"v2seq_v1.*", "AKCR v1 (atom-best)"),
    (r"v2seq_mar_sum.*", "AKCR MAR-sum"),
    (r"v2seq_mar_lse.*", "AKCR MAR-lse"),
    (r"v2seq_acrp.*", "AKCR ACRP"),
    (r"full_book_akcr_v010_([\w.\-]+)_b\d+", "AKCR v1"),
    (r"full_book_akmr_llmrerank_([\w.\-]+)_b\d+", "AKMR + LLM Rerank"),
    (r"full_book_dosrag_chunks_([\w.\-]+)_b\d+", "DOS+chunks"),
    (r"full_book_bm25_chunks_([\w.\-]+)_b\d+", "BM25+chunks"),
    (r"full_book_colbertv2_chunks_([\w.\-]+)_b\d+", "ColBERT-v2 chunks"),
    (r"full_book_hipporag_e2e_([\w.\-]+)_b\d+", "HippoRAG e2e"),
    (r"full_book_graphrag_e2e_([\w.\-]+)_b\d+", "GraphRAG e2e"),
    (r"full_book_propositionizer_akcr_([\w.\-]+)_b\d+", "Propositionizer + AKCR"),
    (r"full_book_gar_(?:v\d+_)?v?\d+_([\w.\-]+)_b\d+", "Atoms-as-content (GAR)"),
    (r"full_book_raptor_e2e_b\d+", "RAPTOR"),
    (r"full_book_hyde_([\w.\-]+)_b\d+", "HyDE"),
    (r"long_context_oracle_full_book(?:_([\w.\-]+))?", "Long-context oracle"),
    (r"full_book_atoms_summ_([\w.\-]+)_b\d+", "Atoms-summary"),
    (r"full_book_nodes_([\w.\-]+)_b\d+", "Nodes (atom retrieve)"),
    (r"full_book_chunks_([\w.\-]+)_b\d+", "Chunks (chunk retrieve)"),
]

READER_NORMALIZE = {
    "glm51": "GLM-5.1", "glm-5.1": "GLM-5.1",
    "glm47": "GLM-4.7", "glm-4.7": "GLM-4.7", "glm4": "GLM-4.7",
    "opus": "Opus-4.7-EXCLUDED", "claude-opus-4-7": "Opus-4.7-EXCLUDED",
    "gpt4o": "GPT-4o", "gpt-4o": "GPT-4o",
    "glm-5": "GLM-5",
}

def classify(stem):
    for pat, method in METHOD_PATTERNS:
        m = re.match(pat, stem)
        if m:
            reader = m.group(1) if m.lastindex else None
            return method, READER_NORMALIZE.get(reader, reader or "")
    return None, None

def parse_budget(stem):
    for pat in [r"_b(\d+)", r"b(\d+)k"]:
        m = re.search(pat, stem)
        if m:
            v = int(m.group(1))
            if v < 100: v *= 1000  # b24k -> 24000
            return v
    return None

def load_summary(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=None, help="path to write markdown table; default stdout")
    ap.add_argument("--budget", type=int, default=24000, help="filter to one budget; 0=all")
    args = ap.parse_args()

    rows = []  # list of dicts: method, reader, budget, n, judge_acc, runs, file
    by_key = {}  # (method, reader, budget) -> list of (judge_acc, n, file)
    for p in sorted(BASELINES.glob("*.summary.json")):
        if p.stat().st_size == 0: continue
        # Skip smoke / pilot / debug files
        if any(s in p.stem.lower() for s in ["smoke", "pilot", "_debug", "_test", "cross_judge"]):
            continue
        # Skip very partial runs (smoke80 etc) — keep n>=400
        s = load_summary(p)
        if not s: continue
        if "judge_acc" not in s: continue
        if s.get("n_total", 0) < 400: continue
        # Always prefer reader from the summary JSON over filename parsing
        reader_raw = s.get("reader", "")
        reader = READER_NORMALIZE.get(reader_raw, reader_raw)
        if reader == "Opus-4.7-EXCLUDED": continue  # user directive 2026-05-09
        if any(x in reader_raw.lower() for x in ["opus", "claude-opus"]): continue
        method, _filename_reader = classify(p.stem)
        if not method: continue
        budget = s.get("char_budget") or parse_budget(p.stem)
        if args.budget and budget != args.budget:
            continue
        n = s.get("n_total", 0)
        acc = s["judge_acc"]
        key = (method, reader, budget)
        by_key.setdefault(key, []).append((acc, n, p.name))

    # Aggregate
    print(f"# NarrativeQA full_book — Wave 1 unified table (b={args.budget}, Opus excluded)\n")
    print("| Method | Reader | n | judge_acc | runs | files |")
    print("|---|---|---:|---:|---:|---|")
    summary_rows = []
    for (method, reader, budget), runs in sorted(by_key.items(), key=lambda kv: (kv[0][1] or "", kv[0][0])):
        accs = [a for a, _, _ in runs]
        ns = [n for _, n, _ in runs]
        if len(accs) == 1:
            acc_str = f"{accs[0]:.4f}"
        else:
            acc_str = f"{statistics.mean(accs):.4f} ± {statistics.stdev(accs):.4f}"
        n_str = f"{max(ns)}"
        files = ", ".join(r[2].replace(".summary.json","") for r in runs)
        print(f"| {method} | {reader} | {n_str} | {acc_str} | {len(runs)} | {files[:80]} |")
        summary_rows.append({
            "method": method, "reader": reader, "budget": budget,
            "n_max": max(ns), "judge_acc_mean": statistics.mean(accs),
            "judge_acc_sd": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "runs": len(accs), "files": [r[2] for r in runs],
        })

    # Sort summary by reader, then by judge_acc descending
    print(f"\n## Per-reader rankings (b={args.budget})\n")
    by_reader = {}
    for r in summary_rows:
        by_reader.setdefault(r["reader"], []).append(r)
    for reader in sorted(by_reader.keys()):
        rows = sorted(by_reader[reader], key=lambda r: -r["judge_acc_mean"])
        print(f"### {reader}\n")
        print("| Rank | Method | judge_acc | runs |")
        print("|---:|---|---:|---:|")
        for i, r in enumerate(rows, 1):
            acc = f"{r['judge_acc_mean']:.4f}"
            if r["judge_acc_sd"] > 0:
                acc += f" ± {r['judge_acc_sd']:.4f}"
            print(f"| {i} | {r['method']} | {acc} | {r['runs']} |")
        print()

if __name__ == "__main__":
    main()
