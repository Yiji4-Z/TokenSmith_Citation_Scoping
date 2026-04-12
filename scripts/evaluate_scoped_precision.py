#!/usr/bin/env python3
"""
evaluate_scoped_precision.py

Evaluates precision@k and recall@k for scoped vs. unscoped retrieval using a
curated ground-truth benchmark whose chunk indices are verified against the
current index build.

Ground truth was built by:
  1. Enumerating all chunks with enumerate(meta) to get list-position indices.
  2. Reading the actual chunk text to confirm each chunk answers its question.
  3. Recording the list-position index (which matches what FAISS returns) and
     the correct chapter number.

This avoids the stale-ID problem in benchmarks.yaml, whose ideal_retrieved_chunks
were generated from a different index build.

Usage (from project root, with conda env active):
    python scripts/evaluate_scoped_precision.py
"""

import sys
import json
import pathlib
import re

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.retriever import FAISSRetriever, BM25Retriever, load_artifacts, apply_post_filter
from src.ranking.ranker import EnsembleRanker
from src.metadata_store import MetadataStore
from src.config import RAGConfig

# ---------------------------------------------------------------------------
# Curated ground-truth benchmark
# Each entry:
#   id            – short identifier
#   question      – the query sent to the retriever
#   chapter       – correct chapter number (used for precision)
#   ideal_chunks  – list-position indices verified against current index text
# ---------------------------------------------------------------------------
BENCHMARK = [
    # ---- Chapter 18: Two-Phase Locking & Deadlock -------------------------
    {
        "id": "2pl_protocol",
        "question": "What is the two-phase locking protocol and how does it guarantee conflict-serializability?",
        "chapter": 18,
        "ideal_chunks": [1094, 1095],   # 2PL definition + cascading rollback intro
    },
    {
        "id": "strict_2pl",
        "question": "What is strict two-phase locking and how does it prevent cascading rollbacks?",
        "chapter": 18,
        "ideal_chunks": [1095, 1097],   # cascading rollback + strict/rigorous 2PL
    },
    {
        "id": "deadlock_definition",
        "question": "What is a deadlock and what are the two main strategies for handling it?",
        "chapter": 18,
        "ideal_chunks": [1103, 1104],   # deadlock state + prevention approaches
    },
    {
        "id": "wait_for_graph",
        "question": "How does the wait-for graph detect deadlocks and when should detection be invoked?",
        "chapter": 18,
        "ideal_chunks": [1108, 1109],   # wait-for graph + invocation timing
    },
    {
        "id": "deadlock_recovery",
        "question": "How does a database system choose a victim transaction to break a deadlock, and what is starvation?",
        "chapter": 18,
        "ideal_chunks": [1110],         # victim selection + starvation
    },

    # ---- Chapter 19: ARIES Recovery ---------------------------------------
    {
        "id": "aries_overview",
        "question": "How does ARIES recover from a system crash? What are the three passes?",
        "chapter": 19,
        "ideal_chunks": [1247, 1248],   # 3-pass overview + analysis pass detail
    },
    {
        "id": "aries_lsn",
        "question": "What is a log sequence number in ARIES and what information does each log record store?",
        "chapter": 19,
        "ideal_chunks": [1244, 1245],   # LSN definition + pageLSN/dirty-page table
    },
    {
        "id": "aries_redo",
        "question": "How does the ARIES redo pass work and what is RedoLSN?",
        "chapter": 19,
        "ideal_chunks": [1249],         # redo pass repeats history from RedoLSN
    },
    {
        "id": "aries_undo",
        "question": "How does the ARIES undo pass roll back incomplete transactions?",
        "chapter": 19,
        "ideal_chunks": [1250, 1251],   # undo pass backward scan
    },

    # ---- Chapter 14: B+ Tree Indexing ------------------------------------
    {
        "id": "bptree_search",
        "question": "How does a B+ tree process a point query and a range query?",
        "chapter": 14,
        "ideal_chunks": [824, 825],     # find(v) pseudocode + range search
    },
    {
        "id": "bptree_insert",
        "question": "How is a new key inserted into a B+ tree when a leaf node is full?",
        "chapter": 14,
        "ideal_chunks": [829, 830, 831], # insert overview + Adams example + nonleaf split
    },

    # ---- Chapter 15: Query Processing ------------------------------------
    {
        "id": "external_sort_merge",
        "question": "How does the external sort-merge algorithm sort a relation that does not fit in memory?",
        "chapter": 15,
        "ideal_chunks": [909, 910],     # sort-merge algorithm description
    },
    {
        "id": "block_nested_loop",
        "question": "How does the block nested-loop join algorithm work and what is its I/O cost?",
        "chapter": 15,
        "ideal_chunks": [916, 917],     # block NLJ description + cost analysis
    },
]

TOP_K = 10
ARTIFACTS_DIR = pathlib.Path("index/sections")
INDEX_PREFIX  = "textbook_index"

cfg = RAGConfig.from_yaml(pathlib.Path("config/config.yaml"))

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
print("Loading artifacts...")
faiss_idx, bm25_idx, chunks, sources, meta = load_artifacts(ARTIFACTS_DIR, INDEX_PREFIX)

faiss_retriever = FAISSRetriever(faiss_idx, cfg.embed_model)
bm25_retriever  = BM25Retriever(bm25_idx)
ranker = EnsembleRanker(
    ensemble_method=cfg.ensemble_method,
    weights=cfg.ranker_weights,
    rrf_k=int(cfg.rrf_k),
)

meta_store = MetadataStore(db_path="index/metadata.db")
if meta_store.is_empty():
    meta_store.populate_from_metadata(meta)

# Build list-index → chapter mapping using enumerate (matches FAISS output)
chunk_to_chapter: dict[int, int] = {}
for list_idx, m in enumerate(meta):
    sp = m.get("section_path", "")
    match = re.match(r"Chapter\s+(\d+)", sp)
    if match:
        chunk_to_chapter[list_idx] = int(match.group(1))

print(f"Loaded {len(chunks)} chunks across {len(set(chunk_to_chapter.values()))} chapters.\n")


# ---------------------------------------------------------------------------
# Retrieval helper
# ---------------------------------------------------------------------------
def retrieve(question: str, valid_ids=None) -> list[int]:
    pool_n = max(cfg.num_candidates, TOP_K + 10)
    raw_scores = {
        "faiss": faiss_retriever.get_scores(question, pool_n, chunks),
        "bm25":  bm25_retriever.get_scores(question, pool_n, chunks),
    }
    ordered, _ = ranker.rank(raw_scores)
    if valid_ids is not None:
        ordered = apply_post_filter(ordered, valid_ids)
    return ordered[:TOP_K]


def precision_at_k(topk: list[int], chapter: int) -> float:
    if not topk:
        return 0.0
    return sum(1 for c in topk if chunk_to_chapter.get(c) == chapter) / len(topk)


def recall_at_k(topk: list[int], ideal: list[int]) -> float:
    if not ideal:
        return 0.0
    return sum(1 for c in topk if c in set(ideal)) / len(ideal)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
W = 100
print("=" * W)
print(f"{'ID':<24} {'Ch':>3} {'Unscp P@k':>10} {'Scpd P@k':>10} {'Unscp R@k':>10} {'Scpd R@k':>10} {'Gain':>7}")
print("=" * W)

results = []
for bm in BENCHMARK:
    chapter_ids = meta_store.get_chunk_ids_by_chapters([bm["chapter"]])
    unscoped    = retrieve(bm["question"])
    scoped      = retrieve(bm["question"], valid_ids=chapter_ids)

    up = precision_at_k(unscoped, bm["chapter"])
    sp = precision_at_k(scoped,   bm["chapter"])
    ur = recall_at_k(unscoped, bm["ideal_chunks"])
    sr = recall_at_k(scoped,   bm["ideal_chunks"])

    gain = sp - up
    print(f"{bm['id']:<24} {bm['chapter']:>3} {up:>10.2f} {sp:>10.2f} {ur:>10.2f} {sr:>10.2f} {gain:>+7.2f}")
    results.append({
        "id": bm["id"],
        "chapter": bm["chapter"],
        "ideal_chunks": bm["ideal_chunks"],
        "unscoped_precision": round(up, 4),
        "scoped_precision":   round(sp, 4),
        "unscoped_recall":    round(ur, 4),
        "scoped_recall":      round(sr, 4),
        "precision_gain":     round(sp - up, 4),
        "recall_gain":        round(sr - ur, 4),
        "top_k": TOP_K,
    })

print("=" * W)

# Averages
avg_up = sum(r["unscoped_precision"] for r in results) / len(results)
avg_sp = sum(r["scoped_precision"]   for r in results) / len(results)
avg_ur = sum(r["unscoped_recall"]    for r in results) / len(results)
avg_sr = sum(r["scoped_recall"]      for r in results) / len(results)

print(f"{'AVERAGE':<24} {'':>3} {avg_up:>10.2f} {avg_sp:>10.2f} {avg_ur:>10.2f} {avg_sr:>10.2f} {avg_sp-avg_up:>+7.2f}")
print("=" * W)

print(f"\nSummary:")
print(f"  Precision@{TOP_K}: {avg_up:.2f} (unscoped) → {avg_sp:.2f} (scoped)  [{avg_sp-avg_up:+.2f} pp]")
print(f"  Recall@{TOP_K}:    {avg_ur:.2f} (unscoped) → {avg_sr:.2f} (scoped)  [{avg_sr-avg_ur:+.2f}]")
note = "(scoped P@k = 1.0 because all returned chunks are from the correct chapter)" if avg_sp >= 0.999 else ""
if note:
    print(f"  Note: {note}")

# Save
out = pathlib.Path("logs/scoped_precision_results.json")
out.parent.mkdir(exist_ok=True)
with open(out, "w") as f:
    json.dump({
        "top_k": TOP_K,
        "n_benchmarks": len(results),
        "averages": {
            "unscoped_precision": round(avg_up, 4),
            "scoped_precision":   round(avg_sp, 4),
            "unscoped_recall":    round(avg_ur, 4),
            "scoped_recall":      round(avg_sr, 4),
            "precision_gain_pp":  round((avg_sp - avg_up) * 100, 1),
            "recall_gain":        round(avg_sr - avg_ur, 4),
        },
        "results": results,
    }, f, indent=2)
print(f"\nResults saved to {out}")
meta_store.close()
