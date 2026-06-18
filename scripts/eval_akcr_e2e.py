#!/usr/bin/env python3
"""
AKCR — Atom-Keyed Chunk Retrieval.

Hypothesis: atoms provide BETTER retrieval signal (more granular, propositional),
but chunks provide BETTER reader content (narrative coherence).

Pipeline:
  1. Retrieve top-K atoms by dense cosine on query
  2. Map each atom → its source chunk
  3. Take unique source chunks (preserve atom rank order)
  4. DOS sort chunks by document position
  5. Reader reads chunks (NOT atoms)

This combines fine-grained atomic ranking + coherent chunk content.
"""
import argparse, asyncio, json, os, re, string, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import httpx

REPO = Path(__file__).resolve().parents[1]
if not (REPO / "data/narrativeqa/processed_v010_full/qa_full.jsonl").exists():
    _candidates = []
    if os.environ.get("MEMORYNET_REPO"):
        _candidates.append(Path(os.environ["MEMORYNET_REPO"]))
    _candidates.append(Path.home() / "MemoryNet")
    for _c in _candidates:
        if (_c / "data/narrativeqa/processed_v010_full/qa_full.jsonl").exists():
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


def normalize(s):
    s = (s or "").lower()
    s = re.sub(r"["+re.escape(string.punctuation)+"]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def f1_token(pred, gold):
    pt, gt = normalize(pred).split(), normalize(gold).split()
    if not pt or not gt: return 0.0
    common = set(pt) & set(gt)
    if not common: return 0.0
    p = sum(min(pt.count(w), gt.count(w)) for w in common) / len(pt)
    r = sum(min(pt.count(w), gt.count(w)) for w in common) / len(gt)
    return 2*p*r/(p+r) if (p+r) else 0.0


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
            if a == retries-1: raise
            await asyncio.sleep(2 ** a)
    return None, None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes-jsonl", required=True)
    ap.add_argument("--nodes-emb", required=True)
    ap.add_argument("--chunks-jsonl", default="data/narrativeqa/processed_v010_full/chunks_full.jsonl")
    ap.add_argument("--queries-emb", default="data/embeddings/full_book/queries.npy")
    ap.add_argument("--qa-jsonl", default="data/narrativeqa/processed_v010_full/qa_full.jsonl")
    ap.add_argument("--char-budget", type=int, default=24000)
    ap.add_argument("--K-atoms", type=int, default=200)
    ap.add_argument("--reader-model", default="glm-4.7")
    ap.add_argument("--judge-model", default="glm-4.7")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()

    nodes = [json.loads(l) for l in open(args.nodes_jsonl)]
    chunks = [json.loads(l) for l in open(args.chunks_jsonl)]
    qa = [json.loads(l) for l in open(args.qa_jsonl)]
    node_emb = np.load(args.nodes_emb).astype(np.float32)
    q_emb = np.load(args.queries_emb).astype(np.float32)
    print(f"  {len(nodes)} atoms, {len(chunks)} chunks, {len(qa)} queries")

    by_story_a = defaultdict(list)
    for i, n in enumerate(nodes): by_story_a[n["story_id"]].append(i)
    chunk_by_key = {(c["story_id"], c.get("chunk_index", 0)): c for c in chunks}

    KEY = os.environ["GLM_API_KEY"]
    URL = os.environ["GLM_URL"]
    sem = asyncio.Semaphore(args.concurrency)
    n_yes = n_total = 0
    f1_total = 0.0
    chars_total = 0
    n_chunks_total = 0
    results = []
    t0 = time.time()

    async def process(qa_i, client):
        nonlocal n_yes, n_total, f1_total, chars_total, n_chunks_total
        async with sem:
            qi = qa_i["qa_idx"]; sid = qa_i["story_id"]
            story_atoms = by_story_a.get(sid, [])
            if not story_atoms: return
            qv = q_emb[qi] / (np.linalg.norm(q_emb[qi]) + 1e-9)
            cand = node_emb[story_atoms]
            cand = cand / (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-9)
            sims = cand @ qv
            order = np.argsort(-sims)[:args.K_atoms]

            # Map atoms → unique source chunks (preserve atom rank order)
            seen_chunks = set()
            ranked_chunks = []  # list of (chunk_index, chunk_text)
            for ix in order:
                gi = story_atoms[ix]
                ci = nodes[gi].get("chunk_index", 0)
                key = (sid, ci)
                if key in seen_chunks: continue
                seen_chunks.add(key)
                chunk = chunk_by_key.get(key)
                if not chunk: continue
                ranked_chunks.append((ci, chunk["text"]))

            # Pack chunks within budget
            selected = []
            chars = 0
            for ci, text in ranked_chunks:
                if chars + len(text) + 3 > args.char_budget: continue
                selected.append((ci, text))
                chars += len(text) + 3

            if not selected: return
            selected.sort(key=lambda x: x[0])  # DOS
            passages = "\n\n".join(f"[Passage {ci}]: {text}" for ci, text in selected)

            try:
                pred, _ = await call_model(client, URL, KEY, args.reader_model,
                    [{"role":"user","content":READER_PROMPT.format(passages=passages, question=qa_i["question"])}],
                    max_tokens=200, timeout=180)
                judge, _ = await call_model(client, URL, KEY, args.judge_model,
                    [{"role":"user","content":JUDGE_PROMPT.format(question=qa_i["question"],
                        gold1=qa_i.get("answer1",""), gold2=qa_i.get("answer2",""), pred=pred or "")}],
                    max_tokens=10, timeout=60)
            except Exception:
                return
            yes = bool(judge and "YES" in judge.upper())
            f1 = max(f1_token(pred, qa_i.get("answer1","")), f1_token(pred, qa_i.get("answer2","")))
            n_total += 1
            if yes: n_yes += 1
            f1_total += f1
            chars_total += chars
            n_chunks_total += len(selected)
            results.append({"qa_idx":qi, "story_id":sid, "n_chunks":len(selected),
                            "passages_chars":chars, "pred":pred, "judge":judge, "yes":yes, "f1":f1})

    limits = httpx.Limits(max_connections=max(args.concurrency*2, 50),
                           max_keepalive_connections=args.concurrency)
    timeout = httpx.Timeout(180.0, connect=30.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        tasks = [process(q, client) for q in qa]
        for i, fut in enumerate(asyncio.as_completed(tasks)):
            await fut
            if (i+1) % 100 == 0:
                acc = n_yes / max(1, n_total)
                print(f"  [{i+1}/{len(qa)}] judge_acc={acc:.4f} elapsed={time.time()-t0:.0f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"method":f"AKCR-reader_{args.reader_model}",
               "atom_source": args.nodes_jsonl,
               "reader":args.reader_model, "judge":args.judge_model,
               "char_budget":args.char_budget, "K_atoms":args.K_atoms,
               "n_total":n_total, "n_yes":n_yes,
               "judge_acc":n_yes/max(1,n_total), "avg_f1":f1_total/max(1,n_total),
               "avg_passages_chars":chars_total/max(1,n_total),
               "avg_n_chunks":n_chunks_total/max(1,n_total),
               "duration_sec":time.time()-t0}
    Path(args.summary_out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
