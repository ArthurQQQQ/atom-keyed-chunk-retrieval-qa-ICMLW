#!/usr/bin/env python3
"""
Graph-Aware Reader (GAR) — feed atoms TOGETHER WITH their typed edges (causal,
temporal, contradicting) to the reader, formatted as a structured chain.

This is the user's vision: graph structure carries inference scaffolding that
individual atoms lose.

Pipeline:
  1. Retrieve top-K atoms (dense cosine on bge-m3)
  2. For each retrieved atom, also include its DIRECT NEIGHBORS (atoms in same
     chunk linked by edges)
  3. Format as graph: each atom + its outgoing edges with narrative_glue
  4. Reader sees the graph and can do causal/temporal inference

Compatible with V010, V011, V012 atoms (uses edge_hints / edges fields if present).
"""
import argparse, asyncio, json, os, re, string, sys, time
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


READER_PROMPT = """\
Answer the question using ONLY the provided graph of facts and their relations.
The facts are arranged in document order. Edges show causal/temporal links.
Reason across the chain when needed. Reply with one short sentence. If the
graph doesn't contain the answer, reply: I don't know.

Graph (document order, with relations):
{graph_text}

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


def get_edges(node):
    """Return list of (relation, target_index, glue) tuples, V010/V011/V012 compat."""
    out = []
    # V012 schema: edges with narrative_glue
    for e in (node.get("edges") or []):
        if isinstance(e, dict):
            tgt = e.get("to_atom_index")
            rel = e.get("relation", "")
            glue = e.get("narrative_glue", "")
            if tgt is not None:
                out.append((rel, int(tgt), glue))
    # V010/V011 schema: edge_hints with relation + to_node_root or text
    for h in (node.get("edge_hints") or []):
        if isinstance(h, dict):
            rel = h.get("relation", "")
            tgt = h.get("to_node_root", h.get("to_atom_index", h.get("to_node_id", "")))
            out.append((rel, tgt, ""))
    return out


def format_graph(retrieved_atoms, all_atoms_in_story, by_chunk_index, prefer_verbatim=True):
    """Format retrieved atoms + their edges as a structured graph for reader.

    V013 change: use verbatim_source if available (preserves narrative voice).
    Fallback to decontextualized text if no verbatim.
    Annotate modality (quoted dialogue / inner thought / actual) so reader
    distinguishes asserted facts from character claims.
    """
    retrieved_atoms.sort(key=lambda a: (a.get("chunk_index", 0), a.get("atom_index_in_chunk", 0)))
    out_lines = []
    seen_chunks = set()
    for atom in retrieved_atoms:
        ci = atom.get("chunk_index", 0)
        if ci not in seen_chunks:
            out_lines.append(f"\n[Scene {ci}]")
            seen_chunks.add(ci)

        # Choose text: verbatim source preserves voice; fallback to atomic decontextualized
        verbatim = atom.get("verbatim_source") or ""
        text = atom.get("text", "") or ""
        if prefer_verbatim and verbatim and len(verbatim) <= 250:
            display_text = verbatim
        else:
            display_text = text

        # Modality annotation: tag quoted/hypothetical claims so reader doesn't treat as asserted
        modality = atom.get("modality", "actual")
        modality_prefix = ""
        if modality == "quoted_dialogue":
            modality_prefix = "[CHARACTER SAYS] "
        elif modality == "inner_thought":
            modality_prefix = "[CHARACTER THINKS] "
        elif modality in ("hypothetical", "counterfactual"):
            modality_prefix = "[HYPOTHETICAL] "

        # Salience hint
        salience = atom.get("salience_tier", "")
        sal_suffix = " (atmospheric)" if salience == "atmospheric" else ""

        out_lines.append(f"  • {modality_prefix}{display_text}{sal_suffix}")

        # Edges with narrative glue
        edges = get_edges(atom)
        for rel, tgt, glue in edges[:2]:
            target_text = ""
            if isinstance(tgt, int):
                key = (atom["chunk_index"], tgt)
                tgt_atom = by_chunk_index.get(key)
                if tgt_atom:
                    tgt_v = tgt_atom.get("verbatim_source") or tgt_atom.get("text", "")
                    target_text = tgt_v[:100]
            elif isinstance(tgt, str) and tgt:
                target_text = tgt[:100]
            if target_text:
                glue_str = f" \"{glue}\"" if glue else ""
                out_lines.append(f"      ↳ [{rel}{glue_str}] {target_text}")
    return "\n".join(out_lines).strip()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes-jsonl", required=True)
    ap.add_argument("--nodes-emb", required=True)
    ap.add_argument("--queries-emb", required=True)
    ap.add_argument("--qa-jsonl", default="data/narrativeqa/processed_v010_full/qa_full.jsonl")
    ap.add_argument("--char-budget", type=int, default=24000)
    ap.add_argument("--K", type=int, default=400)
    ap.add_argument("--reader-model", default="glm-5.1")
    ap.add_argument("--judge-model", default="glm-4.7")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()

    print(f"[GAR] reader={args.reader_model} judge={args.judge_model} budget={args.char_budget}")
    nodes = [json.loads(l) for l in open(args.nodes_jsonl)]
    qa = [json.loads(l) for l in open(args.qa_jsonl)]
    node_emb = np.load(args.nodes_emb).astype(np.float32)
    q_emb = np.load(args.queries_emb).astype(np.float32)
    print(f"  {len(nodes)} atoms, {len(qa)} queries")

    by_story = defaultdict(list)
    for i, n in enumerate(nodes):
        by_story[n["story_id"]].append(i)

    # Index atoms by (chunk_index, atom_index_in_chunk) per story for edge resolution
    by_chunk_atomidx = defaultdict(dict)
    for n in nodes:
        sid = n["story_id"]
        ci = n.get("chunk_index", 0)
        ai = n.get("atom_index_in_chunk", -1)
        by_chunk_atomidx[sid][(ci, ai)] = n

    KEY = os.environ["GLM_API_KEY"]
    URL = os.environ["GLM_URL"]
    sem = asyncio.Semaphore(args.concurrency)
    n_yes = n_total = 0
    f1_total = 0.0
    chars_total = 0
    results = []
    t0 = time.time()

    async def process(qa_i, client):
        nonlocal n_yes, n_total, f1_total, chars_total
        async with sem:
            qi = qa_i["qa_idx"]; sid = qa_i["story_id"]
            story_atoms = by_story.get(sid, [])
            if not story_atoms: return
            qv = q_emb[qi] / (np.linalg.norm(q_emb[qi]) + 1e-9)
            cand = node_emb[story_atoms]
            cand = cand / (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-9)
            sims = cand @ qv
            order = np.argsort(-sims)

            # Pack atoms within budget (account for graph format overhead)
            chosen = []
            chars = 0
            for ix in order[:args.K]:
                gi = story_atoms[ix]
                atom = nodes[gi]
                # estimate: text + edges ≈ text * 1.5
                est = int(len(atom.get("text", "")) * 1.5) + 30
                if chars + est > args.char_budget: continue
                chosen.append(atom)
                chars += est
            if not chosen: return

            # Format as graph
            graph_text = format_graph(chosen, nodes, by_chunk_atomidx[sid])
            chars_actual = len(graph_text)

            try:
                pred, _ = await call_model(client, URL, KEY, args.reader_model,
                    [{"role":"user","content":READER_PROMPT.format(graph_text=graph_text, question=qa_i["question"])}],
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
            chars_total += chars_actual
            results.append({"qa_idx":qi, "story_id":sid, "n_atoms":len(chosen),
                            "graph_chars":chars_actual, "pred":pred, "judge":judge, "yes":yes, "f1":f1})

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
    summary = {"method":f"GAR-reader_{args.reader_model}", "reader":args.reader_model, "judge":args.judge_model,
               "char_budget":args.char_budget, "n_total":n_total, "n_yes":n_yes,
               "judge_acc":n_yes/max(1,n_total), "avg_f1":f1_total/max(1,n_total),
               "avg_graph_chars":chars_total/max(1,n_total), "duration_sec":time.time()-t0}
    Path(args.summary_out).write_text(json.dumps(summary, indent=2))
    print("\n=== GAR summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
