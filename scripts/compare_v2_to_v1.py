#!/usr/bin/env python3
"""Compare v2 variants to new-script v1 baseline using paired bootstrap.

All conditions: GLM-5.1 reader, GLM-4.7 judge, b=24K, K=200, n≈1169.
Comparison baseline: v2script_v1 (the same script that produced the v2 variants),
NOT the original eval_akcr_e2e.py (which gave 0.5997, vs new-script v1's 0.5594).
The 4pp gap is API non-determinism + tie-break order; the apples-to-apples
v2 vs v1 comparison must use new-script v1 to control for both.
"""
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
B = REPO / "data" / "eval" / "baselines"


def load_yes(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[(r["qa_idx"], r["story_id"])] = bool(r.get("yes", False))
    return out


def paired_bootstrap(yes_a, yes_b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    yes_a = yes_a.astype(np.int8); yes_b = yes_b.astype(np.int8)
    diffs = (yes_a - yes_b).astype(np.float32)
    n = len(diffs)
    boot = np.zeros(n_boot, dtype=np.float32)
    for i in range(n_boot):
        ix = rng.integers(0, n, size=n)
        boot[i] = diffs[ix].mean()
    return {
        "mean_a": float(yes_a.mean()),
        "mean_b": float(yes_b.mean()),
        "delta_pp": float(diffs.mean()) * 100,
        "ci95_lo_pp": float(np.percentile(boot, 2.5)) * 100,
        "ci95_hi_pp": float(np.percentile(boot, 97.5)) * 100,
        "p_a_beats_b": float((boot > 0).mean()),
        "n": int(n),
        "n_a_only": int(((yes_a > 0) & (yes_b == 0)).sum()),
        "n_b_only": int(((yes_a == 0) & (yes_b > 0)).sum()),
        "mcnemar_disagreement": int(((yes_a > 0) & (yes_b == 0)).sum() +
                                     ((yes_a == 0) & (yes_b > 0)).sum()),
    }


def main():
    paths = {
        "v1_orig":           B / "full_book_akcr_v010_glm51_b24000.jsonl",
        "v1_new_script":     B / "v2script_v1_glm51_b24k.jsonl",
        "hacs_a08_seq":      B / "v2seq_hacs_a08_glm51_b24k.jsonl",
        "hacs_a07_seq":      B / "v2seq_hacs_a07_glm51_b24k.jsonl",
        "mar_sum_seq":       B / "v2seq_mar_sum_glm51_b24k.jsonl",
        "mar_lse_seq":       B / "v2seq_mar_lse_t5_glm51_b24k.jsonl",
        "hacs_a08_par":      B / "v2_hacs_a08_glm51_b24k.jsonl",
        "hacs_a07_par":      B / "v2_hacs_a07_glm51_b24k.jsonl",
        "mar_sum_par":       B / "v2_mar_sum_glm51_b24k.jsonl",
        "mar_lse_par":       B / "v2_mar_lse_t5_glm51_b24k.jsonl",
        "dos_chunks":        B / "full_book_dosrag_chunks_glm-5.1_b24000.jsonl",
    }
    yes_data = {}
    for name, p in paths.items():
        if p.exists() and p.stat().st_size > 0:
            yes_data[name] = load_yes(p)
            print(f"[load] {name}: n={len(yes_data[name])}, judge_acc={sum(yes_data[name].values())/len(yes_data[name]):.4f}")
        else:
            print(f"[miss] {name}: {p}")

    # Paired bootstrap of every v2 variant against v1_new_script
    if "v1_new_script" in yes_data:
        print()
        print("=" * 92)
        print("PAIRED BOOTSTRAP vs v1_new_script (same script, controls for API non-determinism)")
        print("=" * 92)
        v1_yes = yes_data["v1_new_script"]
        for name in ["hacs_a08_seq", "hacs_a07_seq", "mar_sum_seq", "mar_lse_seq",
                     "v1_orig", "dos_chunks"]:
            if name not in yes_data: continue
            common = sorted(set(yes_data[name]) & set(v1_yes))
            yes_a = np.array([yes_data[name][k] for k in common])
            yes_b = np.array([v1_yes[k] for k in common])
            res = paired_bootstrap(yes_a, yes_b)
            print(f"\n{name} vs v1_new_script  (n_common={res['n']})")
            print(f"  {name:25s} judge_acc = {res['mean_a']:.4f}")
            print(f"  v1_new_script           judge_acc = {res['mean_b']:.4f}")
            print(f"  Δ = {res['delta_pp']:+6.2f}pp   95% CI [{res['ci95_lo_pp']:+6.2f}, {res['ci95_hi_pp']:+6.2f}]   "
                  f"P({name} > v1_new) = {res['p_a_beats_b']:.4f}")
            print(f"  McNemar: {name}_only={res['n_a_only']}, v1_only={res['n_b_only']}")


if __name__ == "__main__":
    main()
