#!/usr/bin/env python3
"""Compile the final v2 ablation table for the revised paper.

Reads all summary.json files in data/eval/baselines/ and produces a single
markdown table with paired bootstrap vs new-script v1 baseline.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
B = REPO / "data" / "eval" / "baselines"


def load_yes(path):
    out = {}
    if not path.exists() or path.stat().st_size == 0: return out
    for line in open(path):
        try:
            r = json.loads(line)
        except: continue
        out[(r["qa_idx"], r.get("story_id", ""))] = bool(r.get("yes", False))
    return out


def paired_bootstrap_pp(yes_a, yes_b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    yes_a = yes_a.astype(np.int8); yes_b = yes_b.astype(np.int8)
    diffs = (yes_a - yes_b).astype(np.float32)
    n = len(diffs)
    boot = np.zeros(n_boot, dtype=np.float32)
    for i in range(n_boot):
        ix = rng.integers(0, n, size=n)
        boot[i] = diffs[ix].mean()
    return {
        "delta_pp": float(diffs.mean()) * 100,
        "ci_lo": float(np.percentile(boot, 2.5)) * 100,
        "ci_hi": float(np.percentile(boot, 97.5)) * 100,
        "p_a_beats_b": float((boot > 0).mean()),
    }


def main():
    conditions = [
        ("AKCR v1 (orig, May 2)",         "full_book_akcr_v010_glm51_b24000.jsonl"),
        ("AKCR v1 (rerun #2, May 8)",     "v2script_v1_glm51_b24k.jsonl"),
        ("AKCR v1 (rerun #3, May 8)",     "v2seq_v1_run3_glm51_b24k.jsonl"),
        ("AKCR v1 (rerun #4, May 8)",     "v2seq_v1_run4_glm51_b24k.jsonl"),
        ("AKCR v2 HACS α=0.8",            "v2seq_hacs_a08_glm51_b24k.jsonl"),
        ("AKCR v2 HACS α=0.7",            "v2seq_hacs_a07_glm51_b24k.jsonl"),
        ("AKCR v2 MAR-sum",               "v2seq_mar_sum_glm51_b24k.jsonl"),
        ("AKCR v2 MAR-lse τ=5",            "v2seq_mar_lse_t5_glm51_b24k.jsonl"),
        ("AKCR v2 ACRP K_kf=15",           "v2seq_acrp_k15_glm51_b24k.jsonl"),
        ("DOS+chunks (May 2)",            "full_book_dosrag_chunks_glm-5.1_b24000.jsonl"),
        ("Long-context oracle GLM-5.1",   "long_context_oracle_full_book_glm51.jsonl"),
        ("Atoms-as-content GAR V013",     "full_book_gar_v013_v012_glm51_b24000.jsonl"),
    ]
    yes_data = {}
    for name, fname in conditions:
        d = load_yes(B / fname)
        if d:
            yes_data[name] = d
            print(f"[ok ] {name:40s} n={len(d):4d} judge_acc={sum(d.values())/len(d):.4f}")
        else:
            print(f"[mis] {name:40s} (path: {fname})")

    v1_run_names = ["AKCR v1 (orig, May 2)", "AKCR v1 (rerun #2, May 8)",
                     "AKCR v1 (rerun #3, May 8)", "AKCR v1 (rerun #4, May 8)"]
    v1_runs = [yes_data.get(n) for n in v1_run_names if yes_data.get(n)]
    print()
    if len(v1_runs) >= 2:
        v1_accs = [sum(d.values())/len(d) for d in v1_runs]
        print(f"v1 reruns ({len(v1_runs)}): accs={[f'{a:.4f}' for a in v1_accs]}")
        print(f"  mean={np.mean(v1_accs):.4f}, sd={np.std(v1_accs, ddof=1):.4f}, range={max(v1_accs)-min(v1_accs):.4f}")

    base_name = "AKCR v1 (rerun #2, May 8)"
    if base_name not in yes_data:
        print(f"[fatal] no baseline {base_name}")
        return
    base_yes = yes_data[base_name]

    rows = []
    for name, _ in conditions:
        if name not in yes_data: continue
        d = yes_data[name]
        common = sorted(set(d) & set(base_yes))
        ya = np.array([d[k] for k in common])
        yb = np.array([base_yes[k] for k in common])
        if name == base_name:
            delta_str = "(baseline)"; ci_str = "—"; p_str = "—"
        else:
            r = paired_bootstrap_pp(ya, yb)
            delta_str = f"{r['delta_pp']:+.2f}"
            ci_str = f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
            p_str = f"{r['p_a_beats_b']:.3f}"
        acc = sum(d.values())/len(d)
        rows.append((name, len(d), acc, delta_str, ci_str, p_str))

    print()
    print("=" * 100)
    print("v2 ablation table (b=24K chars, GLM-5.1 reader; paired-bootstrap vs v1 rerun #2):")
    print("=" * 100)
    print(f"| {'Method':<32s} | {'n':>4} | {'acc':>6} | {'Δpp':>7} | {'95% CI':<18s} | {'P(>v1)':<7} |")
    print(f"|{'-'*34}|{'-'*6}|{'-'*8}|{'-'*9}|{'-'*20}|{'-'*9}|")
    for name, n, acc, dpp, ci, p in rows:
        print(f"| {name:<32s} | {n:>4d} | {acc:.4f} | {dpp:>7s} | {ci:<18s} | {p:<7s} |")


if __name__ == "__main__":
    main()
