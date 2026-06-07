#!/usr/bin/env python3
"""
Paired bootstrap significance test for E2E judge_acc results.

Compares two methods on the same set of queries; reports:
  - mean judge_acc per method
  - mean difference Δ = m_A - m_B
  - 95% CI on Δ via 10000 paired bootstraps
  - one-sided p(Δ > 0) for "A beats B"
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np


def load_per_query(path):
    """Return dict[(qa_idx, story_id)] -> int{0,1} for judge_acc per query."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        key = (r["qa_idx"], r.get("story_id", ""))
        out[key] = int(r.get("yes", False))
    return out


def paired_bootstrap(yes_a, yes_b, n_boot=10000, seed=0):
    """yes_a, yes_b are aligned 0/1 arrays over the same query set."""
    rng = np.random.default_rng(seed)
    diffs = (yes_a - yes_b).astype(np.float32)
    n = len(diffs)
    boot = np.zeros(n_boot, dtype=np.float32)
    for i in range(n_boot):
        ix = rng.integers(0, n, size=n)
        boot[i] = diffs[ix].mean()
    return {
        "mean_a": float(yes_a.mean()),
        "mean_b": float(yes_b.mean()),
        "delta": float(diffs.mean()),
        "ci95_lo": float(np.percentile(boot, 2.5)),
        "ci95_hi": float(np.percentile(boot, 97.5)),
        "p_a_beats_b": float((boot > 0).mean()),
        "n": int(n),
        "n_a_yes": int(yes_a.sum()),
        "n_b_yes": int(yes_b.sum()),
        "n_both_yes": int((yes_a & yes_b).sum()),
        "n_a_only": int((yes_a & ~yes_b.astype(bool)).sum()),
        "n_b_only": int((~yes_a.astype(bool) & yes_b).sum()),
        "n_both_no": int((~yes_a.astype(bool) & ~yes_b.astype(bool)).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Path to method A per-query jsonl")
    ap.add_argument("--b", required=True, help="Path to method B per-query jsonl")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    a_by_q = load_per_query(args.a)
    b_by_q = load_per_query(args.b)
    common = sorted(set(a_by_q) & set(b_by_q))
    print(f"[load] {args.name_a}={len(a_by_q)}  {args.name_b}={len(b_by_q)}  common={len(common)}")
    yes_a = np.array([a_by_q[k] for k in common])
    yes_b = np.array([b_by_q[k] for k in common])

    res = paired_bootstrap(yes_a, yes_b, n_boot=args.n_boot)
    print(f"\n=== Paired bootstrap test ({args.n_boot} resamples, n={res['n']}) ===")
    print(f"  {args.name_a} judge_acc = {res['mean_a']:.4f}  ({res['n_a_yes']}/{res['n']})")
    print(f"  {args.name_b} judge_acc = {res['mean_b']:.4f}  ({res['n_b_yes']}/{res['n']})")
    print(f"  Δ ({args.name_a} - {args.name_b}) = {res['delta']:+.4f}")
    print(f"  95% CI on Δ: [{res['ci95_lo']:+.4f}, {res['ci95_hi']:+.4f}]")
    print(f"  P({args.name_a} > {args.name_b}) = {res['p_a_beats_b']:.4f}")
    print(f"\nContingency:")
    print(f"  both yes:        {res['n_both_yes']}")
    print(f"  {args.name_a} only:    {res['n_a_only']}")
    print(f"  {args.name_b} only:    {res['n_b_only']}")
    print(f"  both no:         {res['n_both_no']}")
    print(f"  McNemar disagreement: {res['n_a_only'] + res['n_b_only']}")

    res["name_a"] = args.name_a
    res["name_b"] = args.name_b
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
