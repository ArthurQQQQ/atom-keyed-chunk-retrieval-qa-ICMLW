#!/usr/bin/env python3
"""
AKCR + ACRP — Atom-Conditioned Reader Prompt.

Same v1 retrieval (atom-best, dedup, DOS pack) AND same chunks.
Difference: prompt prepends top-K_keyfacts atoms as "Key facts:" before passages.

Hypothesis: the atoms steer the reader's attention without replacing chunk content.
Atoms are scaffolding, chunks are the source. This is NOT atoms-as-content
(atoms alone caps at 0.46) — it's atoms-as-pointer + chunks-as-content.

If ACRP > v1, it means the atom info is useful BEYOND retrieval. If ACRP < v1
or ≈ v1, the atoms add nothing once retrieval has used them.
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, string, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import httpx

REPO = Path(__file__).resolve().parent.parent.parent


def load_dotenv():
    for p in (REPO / ".env", REPO / ".env.local"):
        if not p.exists(): continue
        for line in open(p):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()
load_dotenv()


READER_PROMPT_ACRP = """\
Use the key facts as a guide to find the answer in the story passages. Answer with one short sentence using ONLY the passages (the key facts may help you find the right passage). If the passages don't contain the answer, reply: I don't know.

Key facts (in priority order):
{keyfacts}

Story passages (in document order):
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


async def call_model(client, url, key, model, messages, max_tokens=200, retries=4, timeout=180):
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
    ap.add_argument("--char-budget", type=int, default=24000)
    ap.add_argument("--K-atoms", type=int, default=200)
    ap.add_argument("--K-keyfacts", type=int, default=15)
    ap.add_argument("--reader-model", default="glm-5.1")
    ap.add_argument("--judge-model", default="glm-4.7")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--n-limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()

    nodes = [json.loads(l) for l in open(REPO / "data/narrativeqa/processed_v010_full/nodes_v010_full_book.jsonl")]
    chunks = [json.loads(l) for l in open(REPO / "data/narrativeqa/processed_v010_full/chunks_full.jsonl")]
    qa = [json.loads(l) for l in open(REPO / "data/narrativeqa/processed_v010_full/qa_full.jsonl")]
    if args.n_limit > 0:
        qa = qa[:args.n_limit]
    node_emb = np.load(REPO / "data/embeddings/full_book/nodes.npy").astype(np.float32)
    q_emb = np.load(REPO / "data/embeddings/full_book/queries.npy").astype(np.float32)

    by_story_a = defaultdict(list)
    for i, n in enumerate(nodes): by_story_a[n["story_id"]].append(i)
    by_story_c = defaultdict(list)
    for i, c in enumerate(chunks): by_story_c[c["story_id"]].append(i)
    chunk_idx_by_story_pos = {(c["story_id"], c["chunk_index"]): i for i, c in enumerate(chunks)}

    print(f"[load] {len(nodes)} atoms, {len(chunks)} chunks, {len(qa)} queries")

    KEY = os.environ["GLM_API_KEY"]; URL = os.environ["GLM_URL"]
    sem = asyncio.Semaphore(args.concurrency)
    n_yes = n_total = 0; f1_total = 0.0
    chars_total = 0; n_chunks_total = 0
    results = []
    t0 = time.time()

    async def process(qa_obj, client):
        nonlocal n_yes, n_total, f1_total, chars_total, n_chunks_total
        async with sem:
            qi = qa_obj["qa_idx"]; sid = qa_obj["story_id"]
            atom_global = by_story_a.get(sid, [])
            chunk_global = by_story_c.get(sid, [])
            if not atom_global or not chunk_global: return

            sa = normalize_vec(node_emb[atom_global])
            qv = normalize_vec(q_emb[qi])
            atom_scores = sa @ qv
            order = np.argsort(-atom_scores)

            # Top-K_keyfacts atoms — text content
            keyfacts = []
            seen_text = set()
            for ix in order[:args.K_atoms]:
                ag = atom_global[ix]
                txt = nodes[ag].get("text", "").strip()
                if not txt or txt in seen_text: continue
                seen_text.add(txt)
                keyfacts.append(txt)
                if len(keyfacts) >= args.K_keyfacts: break

            # Same chunk selection as v1 (atom-best dedup, DOS pack)
            seen_ci = set(); ranked_chunks = []
            for ix in order[:args.K_atoms]:
                ag = atom_global[ix]
                ci_doc = nodes[ag]["chunk_index"]
                ci_g = chunk_idx_by_story_pos.get((sid, ci_doc))
                if ci_g is None: continue
                if ci_doc in seen_ci: continue
                seen_ci.add(ci_doc)
                ranked_chunks.append((ci_doc, chunks[ci_g]["text"]))

            selected = []; chars = 0
            for ci_doc, text in ranked_chunks:
                c = len(text) + 3
                if chars + c > args.char_budget: continue
                selected.append((ci_doc, text)); chars += c
            if not selected: return
            selected.sort(key=lambda x: x[0])

            keyfacts_text = "\n".join(f"- {kf}" for kf in keyfacts)
            passages = "\n\n".join(f"[Passage {ci}]: {txt}" for ci, txt in selected)
            prompt = READER_PROMPT_ACRP.format(keyfacts=keyfacts_text,
                                                 passages=passages,
                                                 question=qa_obj["question"])

            try:
                pred, _ = await call_model(client, URL, KEY, args.reader_model,
                    [{"role": "user", "content": prompt}], max_tokens=200, timeout=180)
                judge, _ = await call_model(client, URL, KEY, args.judge_model,
                    [{"role": "user", "content": JUDGE_PROMPT.format(question=qa_obj["question"],
                        gold1=qa_obj.get("answer1", ""), gold2=qa_obj.get("answer2", ""),
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
            n_chunks_total += len(selected)
            results.append({"qa_idx": int(qi), "story_id": sid,
                             "n_chunks": int(len(selected)),
                             "n_keyfacts": int(len(keyfacts)),
                             "passages_chars": int(chars),
                             "pred": pred, "judge": judge, "yes": bool(yes), "f1": float(f1)})

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
        "method": "AKCR_v2_acrp",
        "method_args": {"K_atoms": args.K_atoms, "K_keyfacts": args.K_keyfacts},
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
