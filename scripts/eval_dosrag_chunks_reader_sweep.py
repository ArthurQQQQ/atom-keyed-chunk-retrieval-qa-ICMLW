#!/usr/bin/env python3
"""
Reader sweep: run identical DOS+chunks retrieval at b=24000 but vary the reader model.

Goal: isolate whether reader is the bottleneck. If GLM-5.1 / Claude Opus give
significant judge_acc jump over GLM-4.7 (baseline 0.523), the path to oracle 0.645
is via better reader, not better retrieval.
"""
from __future__ import annotations
import argparse, asyncio, json, os, re, string, sys, time
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
            REPO = _c; break


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
Answer the question using ONLY the provided facts (in document order). Reply with one short sentence.
If the facts don't contain the answer, reply: I don't know.

Facts:
{facts}

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
    s = re.sub(r"[" + re.escape(string.punctuation) + "]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def f1_token(pred, gold):
    pt, gt = normalize(pred).split(), normalize(gold).split()
    if not pt or not gt: return 0.0
    common = set(pt) & set(gt)
    if not common: return 0.0
    p = sum(min(pt.count(w), gt.count(w)) for w in common) / len(pt)
    r = sum(min(pt.count(w), gt.count(w)) for w in common) / len(gt)
    return 2 * p * r / (p + r) if (p + r) else 0.0


async def call_model(client, url, key, model, messages, max_tokens=200, max_retries=4, timeout=180):
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if "glm" in model.lower():
        body["enable_thinking"] = False
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(max_retries):
        try:
            r = await client.post(url, json=body, headers=headers, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            content = j["choices"][0]["message"]["content"]
            return content, j.get("usage", {})
        except Exception:
            if attempt == max_retries - 1: raise
            await asyncio.sleep(2 ** attempt)
    return None, None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader-model", required=True,
                     help="reader model: glm-4.7, glm-5.1, claude-opus-4-7, gpt-4o, etc")
    ap.add_argument("--judge-model", default="glm-4.7",
                     help="judge model (kept GLM-4.7 by default for consistent comparison)")
    ap.add_argument("--char-budget", type=int, default=24000)
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()

    DATA = REPO / "data" / "narrativeqa" / "processed_v010_full"
    EMB = REPO / "data" / "embeddings" / "full_book"

    print(f"[reader] {args.reader_model}  judge={args.judge_model}  budget={args.char_budget}")
    chunks = [json.loads(l) for l in open(DATA / "chunks_full.jsonl")]
    qa = [json.loads(l) for l in open(DATA / "qa_full.jsonl")]
    chunk_emb = np.load(EMB / "chunks.npy")
    q_emb = np.load(EMB / "queries.npy")

    by_story_c = defaultdict(list)
    for i, c in enumerate(chunks): by_story_c[c["story_id"]].append(i)
    print(f"  {len(chunks)} chunks, {len(qa)} queries")

    KEY = os.environ["GLM_API_KEY"]
    URL = os.environ["GLM_URL"]
    sem = asyncio.Semaphore(args.concurrency)
    n_yes = n_total = 0
    f1_total = 0.0
    char_total = 0
    tok_r = tok_j = 0
    results = []
    t0 = time.time()

    async def process(qa_i, client):
        nonlocal n_yes, n_total, f1_total, tok_r, tok_j, char_total
        async with sem:
            qi = qa_i["qa_idx"]
            sid = qa_i["story_id"]
            cixs = by_story_c.get(sid, [])
            if not cixs: return
            qv = q_emb[qi] / (np.linalg.norm(q_emb[qi]) + 1e-9)
            cand = chunk_emb[cixs]
            cand = cand / (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-9)
            sims = cand @ qv
            order = np.argsort(-sims)
            selected = []
            chars = 0
            for ix in order[:args.K]:
                gi = cixs[ix]
                text = chunks[gi]["text"]
                if chars + len(text) + 3 > args.char_budget: continue
                selected.append((chunks[gi].get("chunk_index", 0), text))
                chars += len(text) + 3
            if not selected: return
            selected.sort(key=lambda x: x[0])  # DOS sort
            facts_text = "\n".join(f"- {x[1]}" for x in selected)

            try:
                pred, ur = await call_model(client, URL, KEY, args.reader_model,
                                              [{"role": "user", "content": READER_PROMPT.format(
                                                  facts=facts_text, question=qa_i["question"])}],
                                              max_tokens=200, timeout=args.timeout)
                judge, uj = await call_model(client, URL, KEY, args.judge_model,
                                               [{"role": "user", "content": JUDGE_PROMPT.format(
                                                   question=qa_i["question"], gold1=qa_i.get("answer1", ""),
                                                   gold2=qa_i.get("answer2", ""), pred=pred or "")}],
                                               max_tokens=10, timeout=60)
            except Exception as e:
                return
            yes = bool(judge and "YES" in judge.upper())
            f1 = max(f1_token(pred, qa_i.get("answer1", "")),
                     f1_token(pred, qa_i.get("answer2", "")))
            n_total += 1
            if yes: n_yes += 1
            f1_total += f1
            char_total += chars
            tok_r += (ur or {}).get("total_tokens", 0)
            tok_j += (uj or {}).get("total_tokens", 0)
            results.append({
                "qa_idx": qi, "story_id": sid, "n_facts": len(selected),
                "facts_chars": chars, "pred": pred,
                "judge": judge, "yes": yes, "f1": f1,
                "tokens_reader": (ur or {}).get("total_tokens", 0),
                "tokens_judge": (uj or {}).get("total_tokens", 0),
            })

    limits = httpx.Limits(max_connections=max(args.concurrency*2, 50),
                           max_keepalive_connections=args.concurrency)
    timeout = httpx.Timeout(float(args.timeout), connect=30.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        tasks = [process(q, client) for q in qa]
        for i, fut in enumerate(asyncio.as_completed(tasks)):
            await fut
            if (i+1) % 100 == 0:
                acc = n_yes / max(1, n_total)
                print(f"  [{i+1}/{len(qa)}] judge_acc={acc:.4f}  elapsed={time.time()-t0:.0f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "method": f"DOS+chunks_reader-{args.reader_model}",
        "reader": args.reader_model, "judge": args.judge_model,
        "char_budget": args.char_budget,
        "n_total": n_total, "n_yes": n_yes,
        "judge_acc": n_yes / max(1, n_total),
        "avg_f1": f1_total / max(1, n_total),
        "avg_facts_chars": char_total / max(1, n_total),
        "tokens_total_reader": tok_r, "tokens_total_judge": tok_j,
        "duration_sec": time.time() - t0,
    }
    Path(args.summary_out).write_text(json.dumps(summary, indent=2))
    print(f"\n=== reader={args.reader_model} summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
