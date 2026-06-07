#!/usr/bin/env python3
"""
AKCR v2 — full reader eval supporting multiple chunk-scoring variants.

Variants:
  v1            atom-best score, chunks ranked by best-atom rank (= existing AKCR)
  mar_sum       chunk score = sum of top-K atom scores in chunk
  mar_lse       chunk score = logsumexp(tau) of top-K atom scores
  hacs          chunk score = alpha * atom-best + (1-alpha) * chunk-cosine

All variants use:
  - same atom embeddings (bge-m3, V010 atoms)
  - same chunk embeddings (bge-m3, full-book chunks)
  - same query embeddings
  - same K_atoms = 200 candidate atom pool
  - same DOS chunk-packing into char_budget
  - same reader prompt
  - same judge

Only the chunk-scoring step differs. This isolates the philosophy axis
(what makes a chunk relevant given a query, when atoms are the ranking unit).
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, string, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import httpx

REPO = Path(__file__).resolve().parent.parent.parent
# Allow overriding via env or fall back to MemoryNet sibling layout
if not (REPO / "data/narrativeqa/processed_v010_full/qa_full.jsonl").exists():
    _candidates = [Path(os.environ.get("MEMORYNET_REPO", "")),
                   Path("/Users/arthurqiu/MemoryNet"),
                   Path.home() / "MemoryNet"]
    for _c in _candidates:
        if _c and (_c / "data/narrativeqa/processed_v010_full/qa_full.jsonl").exists():
            REPO = _c
            break


def load_dotenv():
    for p in (REPO / ".env", REPO / ".env.local"):
        if not p.exists(): continue
        for line in open(p):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()
load_dotenv()


READER_PROMPT = """\
Answer the question using ONLY the provided story passages (in document order). Reply with one short sentence.
If the passages don't contain the answer, reply: I don't know.

Passages:
{passages}

Question: {question}
"""

JUDGE_PROMPT = """\
Question: {question}
Reference answer 1: {gold1}
Reference answer 2: {gold2}
Model's answer: {pred}
Is the model's answer consistent with EITHER reference (paraphrase OK)? Reply YES or NO.
"""


def normalize_text(s):
    s = (s or "").lower()
    s = re.sub(r"["+re.escape(string.punctuation)+"]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def f1_token(pred, gold):
    pt, gt = normalize_text(pred).split(), normalize_text(gold).split()
    if not pt or not gt: return 0.0
    common = set(pt) & set(gt)
    if not common: return 0.0
    p = sum(min(pt.count(w), gt.count(w)) for w in common) / len(pt)
    r = sum(min(pt.count(w), gt.count(w)) for w in common) / len(gt)
    return 2*p*r/(p+r) if (p+r) else 0.0


def normalize_vec(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def select_v1(atom_scores, atom_chunk_idx, chunk_chars, chunk_texts,
               char_budget, k_atoms):
    order = np.argsort(-atom_scores)
    seen = set()
    ranked = []
    for ix in order[:k_atoms]:
        ci = int(atom_chunk_idx[ix])
        if ci in seen or ci < 0: continue
        seen.add(ci)
        ranked.append(ci)
    selected, chars = [], 0
    for ci in ranked:
        c = chunk_chars[ci] + 3
        if chars + c > char_budget: continue
        selected.append(ci)
        chars += c
    return selected, chars


def select_mar_sum(atom_scores, atom_chunk_idx, chunk_chars, chunk_texts,
                    char_budget, k_atoms):
    topk = np.argsort(-atom_scores)[:k_atoms]
    chunk_score = defaultdict(float)
    for ix in topk:
        ci = int(atom_chunk_idx[ix])
        if ci < 0: continue
        chunk_score[ci] += float(atom_scores[ix])
    ranked = sorted(chunk_score, key=lambda ci: -chunk_score[ci])
    selected, chars = [], 0
    for ci in ranked:
        c = chunk_chars[ci] + 3
        if chars + c > char_budget: continue
        selected.append(ci)
        chars += c
    return selected, chars


def select_mar_lse(atom_scores, atom_chunk_idx, chunk_chars, chunk_texts,
                    char_budget, k_atoms, tau=5.0):
    topk = np.argsort(-atom_scores)[:k_atoms]
    chunk_lse_buckets = defaultdict(list)
    for ix in topk:
        ci = int(atom_chunk_idx[ix])
        if ci < 0: continue
        chunk_lse_buckets[ci].append(float(atom_scores[ix]))
    chunk_score = {ci: float(np.log(np.sum(np.exp(tau * np.array(s))) + 1e-12) / tau)
                   for ci, s in chunk_lse_buckets.items()}
    ranked = sorted(chunk_score, key=lambda ci: -chunk_score[ci])
    selected, chars = [], 0
    for ci in ranked:
        c = chunk_chars[ci] + 3
        if chars + c > char_budget: continue
        selected.append(ci)
        chars += c
    return selected, chars


def select_hacs(atom_scores, atom_chunk_idx, chunk_scores_full,
                 chunk_chars, chunk_texts, char_budget, k_atoms, alpha):
    """HACS: alpha * (chunk's best top-K atom) + (1-alpha) * chunk cosine.
    Chunks not touching any top-K atom get atom-best=0 (only chunk-cosine carries)."""
    topk = np.argsort(-atom_scores)[:k_atoms]
    chunk_atom_best = defaultdict(float)
    for ix in topk:
        ci = int(atom_chunk_idx[ix])
        if ci < 0: continue
        s = float(atom_scores[ix])
        if s > chunk_atom_best[ci]:
            chunk_atom_best[ci] = s
    n_chunks = len(chunk_scores_full)
    composite = np.zeros(n_chunks, dtype=np.float32)
    for ci in range(n_chunks):
        composite[ci] = alpha * chunk_atom_best.get(ci, 0.0) + \
                         (1 - alpha) * float(chunk_scores_full[ci])
    order = np.argsort(-composite)
    selected, chars = [], 0
    for ci in order:
        ci = int(ci)
        c = chunk_chars[ci] + 3
        if chars + c > char_budget: continue
        selected.append(ci)
        chars += c
    return selected, chars


SELECT_FNS = {
    "v1":      lambda **kw: select_v1(kw["atom_scores"], kw["atom_chunk_idx"],
                                       kw["chunk_chars"], kw["chunk_texts"],
                                       kw["char_budget"], kw["k_atoms"]),
    "mar_sum": lambda **kw: select_mar_sum(kw["atom_scores"], kw["atom_chunk_idx"],
                                            kw["chunk_chars"], kw["chunk_texts"],
                                            kw["char_budget"], kw["k_atoms"]),
    "mar_lse": lambda **kw: select_mar_lse(kw["atom_scores"], kw["atom_chunk_idx"],
                                            kw["chunk_chars"], kw["chunk_texts"],
                                            kw["char_budget"], kw["k_atoms"],
                                            tau=kw.get("tau", 5.0)),
    "hacs":    lambda **kw: select_hacs(kw["atom_scores"], kw["atom_chunk_idx"],
                                         kw["chunk_scores_full"],
                                         kw["chunk_chars"], kw["chunk_texts"],
                                         kw["char_budget"], kw["k_atoms"],
                                         alpha=kw.get("alpha", 0.7)),
}


def endpoint_for_model(model):
    """Return (url, key) for the given reader/judge model name.
    GPT/OpenAI models go through OPENAI_URL/OPENAI_API_KEY; everything else
    (GLM, Claude via gateway, etc.) goes through GLM_URL/GLM_API_KEY."""
    m = (model or "").lower()
    if m.startswith("gpt") or "openai" in m:
        return os.environ["OPENAI_URL"], os.environ["OPENAI_API_KEY"]
    return os.environ["GLM_URL"], os.environ["GLM_API_KEY"]


async def call_model(client, model, messages, max_tokens=200, retries=4, timeout=180,
                     url=None, key=None):
    if url is None or key is None:
        url, key = endpoint_for_model(model)
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if "glm" in model.lower(): body["enable_thinking"] = False
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for a in range(retries):
        try:
            r = await client.post(url, json=body, headers=headers, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            return j["choices"][0]["message"]["content"], j.get("usage", {})
        except Exception:
            if a == retries - 1: raise
            await asyncio.sleep(2 ** a)
    return None, None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(SELECT_FNS.keys()))
    ap.add_argument("--alpha", type=float, default=0.7)  # for hacs
    ap.add_argument("--tau", type=float, default=5.0)    # for mar_lse
    ap.add_argument("--char-budget", type=int, default=24000)
    ap.add_argument("--K-atoms", type=int, default=200)
    ap.add_argument("--reader-model", default="glm-5.1")
    ap.add_argument("--judge-model", default="glm-4.7")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--n-limit", type=int, default=0)
    ap.add_argument("--max-questions", type=int, default=0,
                     help="Alias for --n-limit; if both set, the smaller (non-zero) wins.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()

    print("[load] data")
    nodes = [json.loads(l) for l in open(REPO / "data/narrativeqa/processed_v010_full/nodes_v010_full_book.jsonl")]
    chunks = [json.loads(l) for l in open(REPO / "data/narrativeqa/processed_v010_full/chunks_full.jsonl")]
    qa = [json.loads(l) for l in open(REPO / "data/narrativeqa/processed_v010_full/qa_full.jsonl")]
    _limits = [v for v in (args.n_limit, args.max_questions) if v > 0]
    if _limits:
        qa = qa[:min(_limits)]
    node_emb = np.load(REPO / "data/embeddings/full_book/nodes.npy").astype(np.float32)
    chunk_emb = np.load(REPO / "data/embeddings/full_book/chunks.npy").astype(np.float32)
    q_emb = np.load(REPO / "data/embeddings/full_book/queries.npy").astype(np.float32)

    by_story_a = defaultdict(list)
    for i, n in enumerate(nodes):
        by_story_a[n["story_id"]].append(i)
    by_story_c = defaultdict(list)
    for i, c in enumerate(chunks):
        by_story_c[c["story_id"]].append(i)
    chunk_idx_by_story_pos = {}
    for i, c in enumerate(chunks):
        chunk_idx_by_story_pos[(c["story_id"], c["chunk_index"])] = i

    print(f"[load] {len(nodes)} atoms, {len(chunks)} chunks, {len(qa)} queries; method={args.method}")

    # Endpoints resolved per-model inside call_model (GPT -> OpenAI, else GLM)
    sem = asyncio.Semaphore(args.concurrency)
    n_yes = n_total = 0
    f1_total = 0.0
    chars_total = 0
    n_chunks_total = 0
    results = []
    t0 = time.time()
    select_fn = SELECT_FNS[args.method]

    async def process(qa_obj, client):
        nonlocal n_yes, n_total, f1_total, chars_total, n_chunks_total
        async with sem:
            qi = qa_obj["qa_idx"]; sid = qa_obj["story_id"]
            atom_global_ix = by_story_a.get(sid, [])
            chunk_global_ix = by_story_c.get(sid, [])
            if not atom_global_ix or not chunk_global_ix: return

            sa_emb = normalize_vec(node_emb[atom_global_ix])
            sc_emb = normalize_vec(chunk_emb[chunk_global_ix])
            qv = normalize_vec(q_emb[qi])
            atom_scores = sa_emb @ qv
            chunk_scores_full = sc_emb @ qv

            chunk_global_to_local = {g: i for i, g in enumerate(chunk_global_ix)}
            atom_chunk_local = []
            for ai_global in atom_global_ix:
                ci_doc = nodes[ai_global]["chunk_index"]
                ci_global = chunk_idx_by_story_pos.get((sid, ci_doc))
                atom_chunk_local.append(chunk_global_to_local.get(ci_global, -1))
            atom_chunk_local = np.array(atom_chunk_local, dtype=np.int32)
            chunk_chars = np.array([len(chunks[g]["text"]) for g in chunk_global_ix],
                                    dtype=np.int32)
            chunk_texts = [chunks[g]["text"] for g in chunk_global_ix]
            chunk_doc_idx = [chunks[g]["chunk_index"] for g in chunk_global_ix]

            selected_local, chars = select_fn(
                atom_scores=atom_scores,
                atom_chunk_idx=atom_chunk_local,
                chunk_scores_full=chunk_scores_full,
                chunk_chars=chunk_chars,
                chunk_texts=chunk_texts,
                char_budget=args.char_budget,
                k_atoms=args.K_atoms,
                alpha=args.alpha,
                tau=args.tau,
            )
            if not selected_local: return
            # DOS sort by chunk_index in source doc
            sel_with_doc = [(chunk_doc_idx[ci], chunk_texts[ci]) for ci in selected_local]
            sel_with_doc.sort(key=lambda x: x[0])
            passages = "\n\n".join(f"[Passage {ci}]: {text}" for ci, text in sel_with_doc)

            try:
                pred, _ = await call_model(client, args.reader_model,
                    [{"role": "user", "content": READER_PROMPT.format(passages=passages,
                                                                       question=qa_obj["question"])}],
                    max_tokens=200, timeout=180)
                judge, _ = await call_model(client, args.judge_model,
                    [{"role": "user", "content": JUDGE_PROMPT.format(question=qa_obj["question"],
                        gold1=qa_obj.get("answer1", ""),
                        gold2=qa_obj.get("answer2", ""),
                        pred=pred or "")}],
                    max_tokens=10, timeout=60)
            except Exception:
                return
            yes = bool(judge and "YES" in judge.upper())
            f1 = max(f1_token(pred, qa_obj.get("answer1", "")),
                     f1_token(pred, qa_obj.get("answer2", "")))
            n_total += 1
            if yes: n_yes += 1
            f1_total += f1
            chars_total += chars
            n_chunks_total += len(sel_with_doc)
            results.append({"qa_idx": int(qi), "story_id": sid,
                             "n_chunks": int(len(sel_with_doc)),
                             "passages_chars": int(chars),
                             "pred": pred, "judge": judge, "yes": bool(yes),
                             "f1": float(f1)})

    limits = httpx.Limits(max_connections=max(args.concurrency * 2, 50),
                           max_keepalive_connections=args.concurrency)
    timeout = httpx.Timeout(180.0, connect=30.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        tasks = [process(q, client) for q in qa]
        for i, fut in enumerate(asyncio.as_completed(tasks)):
            await fut
            if (i + 1) % 100 == 0:
                acc = n_yes / max(1, n_total)
                print(f"  [{i+1}/{len(qa)}] judge_acc={acc:.4f} elapsed={time.time()-t0:.0f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "method": f"AKCR_v2_{args.method}",
        "method_args": {"alpha": args.alpha if args.method == "hacs" else None,
                         "tau": args.tau if args.method == "mar_lse" else None,
                         "K_atoms": args.K_atoms},
        "reader": args.reader_model, "judge": args.judge_model,
        "char_budget": args.char_budget,
        "n_total": n_total, "n_yes": n_yes,
        "judge_acc": n_yes / max(1, n_total),
        "avg_f1": f1_total / max(1, n_total),
        "avg_passages_chars": chars_total / max(1, n_total),
        "avg_n_chunks": n_chunks_total / max(1, n_total),
        "duration_sec": time.time() - t0,
    }
    Path(args.summary_out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
