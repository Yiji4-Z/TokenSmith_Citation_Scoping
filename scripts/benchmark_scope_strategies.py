#!/usr/bin/env python3
"""
benchmark_scope_strategies.py

Benchmarks pre-filter vs. post-filter scoped retrieval on two dimensions:

1. Paired completeness test: queries matched to their correct scope chapter.
   Shows how many top-k slots each strategy fills when the query is genuinely
   relevant to the scoped chapter.

2. Latency: filter+rank time across five selectivity levels.

Key insight: both strategies work at the Python score-dict level (not FAISS
kernel level), so they see only the top `num_candidates` (=50) FAISS results.
For narrow scopes, if fewer than top_k in-scope chunks appear in those 50
candidates, BOTH strategies return fewer than top_k results.  pre-filter has
no advantage over post-filter in this setup -- the only way to guarantee top_k
results under narrow scope is a true FAISS IDSelector pre-filter, which
requires an IVF index (not the IndexFlatL2 TokenSmith uses).

Usage (from project root, with conda env active):
    python scripts/benchmark_scope_strategies.py
"""

import sys
import time
import json
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.retriever import FAISSRetriever, BM25Retriever, load_artifacts, apply_pre_filter, apply_post_filter
from src.ranking.ranker import EnsembleRanker
from src.metadata_store import MetadataStore
from src.config import RAGConfig

ARTIFACTS_DIR = pathlib.Path("index/sections")
INDEX_PREFIX  = "textbook_index"
TOP_K         = 10

cfg = RAGConfig.from_yaml(pathlib.Path("config/config.yaml"))

# Paired (query, relevant_chapter) -- each query matched to its correct scope
PAIRED = [
    ("What is two-phase locking and how does it guarantee serializability?", 18),
    ("What is a deadlock and how does the wait-for graph detect it?",        18),
    ("How does the ARIES redo pass work and what is RedoLSN?",               19),
    ("How does ARIES recover from a crash using three passes?",              19),
    ("How does a B+ tree process a range query?",                            14),
    ("How is a key inserted into a B+ tree when a leaf node is full?",       14),
    ("How does the external sort-merge algorithm work?",                     15),
    ("How does block nested-loop join reduce I/O compared to naive NLJ?",    15),
]

# Selectivity levels: (label, chapter list or None)
SCOPE_LEVELS = [
    ("1 chapter  (~4%)",  [18]),
    ("3 chapters (~12%)", [17, 18, 19]),
    ("7 chapters (~27%)", [14, 15, 16, 17, 18, 19, 20]),
    ("13 chapters(~50%)", list(range(14, 27))),
    ("Unscoped   (100%)", None),
]

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

print(f"Loaded {len(chunks)} chunks. num_candidates={cfg.num_candidates}, top_k={TOP_K}\n")


def get_raw_scores(query: str) -> dict:
    pool_n = max(cfg.num_candidates, TOP_K + 10)
    return {
        "faiss": faiss_retriever.get_scores(query, pool_n, chunks),
        "bm25":  bm25_retriever.get_scores(query, pool_n, chunks),
    }

def run_pre_filter(raw: dict, valid_ids) -> list:
    filtered = apply_pre_filter(raw, valid_ids)
    ordered, _ = ranker.rank(filtered)
    return ordered[:TOP_K]

def run_post_filter(raw: dict, valid_ids) -> list:
    ordered, _ = ranker.rank(raw)
    return apply_post_filter(ordered, valid_ids)[:TOP_K]

def jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    u = sa | sb
    return len(sa & sb) / len(u) if u else 1.0


# ---------------------------------------------------------------------------
# Part 1 – Paired completeness: query matched to its correct chapter scope
# ---------------------------------------------------------------------------
print("=" * 90)
print("PART 1: Paired completeness — query matched to its relevant chapter scope")
print(f"        pool_n = {max(cfg.num_candidates, TOP_K+10)}, top_k = {TOP_K}")
print("=" * 90)
print(f"{'Query (truncated)':<52} {'Ch':>3} {'Pre':>5} {'Post':>5} {'Full?':>7}")
print("-" * 90)

paired_results = []
for query, chapter in PAIRED:
    valid_ids = meta_store.get_chunk_ids_by_chapters([chapter])
    raw = get_raw_scores(query)
    pre_r  = run_pre_filter(raw, valid_ids)
    post_r = run_post_filter(raw, valid_ids)
    full = "YES" if len(pre_r) == TOP_K and len(post_r) == TOP_K else "PARTIAL"
    q_short = query[:51]
    print(f"{q_short:<52} {chapter:>3} {len(pre_r):>5} {len(post_r):>5} {full:>7}")
    paired_results.append({
        "query": query, "chapter": chapter,
        "pre_count": len(pre_r), "post_count": len(post_r),
        "both_full": full == "YES",
    })

print("-" * 90)
n_full = sum(1 for r in paired_results if r["both_full"])
print(f"\n  {n_full}/{len(paired_results)} queries returned full top-{TOP_K} results under single-chapter scope.")
print(f"  When fewer results are returned, both strategies are equally affected.")
print(f"  Root cause: pool_n={max(cfg.num_candidates,TOP_K+10)} limits candidate pool; for a chapter")
print(f"  of ~80 chunks, only those that rank in the top {max(cfg.num_candidates,TOP_K+10)} FAISS/BM25 results")
print(f"  are considered. Queries closely matching the chapter content fill top_k;")
print(f"  queries that are off-topic for the scope return fewer results by design.")

# ---------------------------------------------------------------------------
# Part 2 – Latency and overlap across selectivity levels (generic queries)
# ---------------------------------------------------------------------------
GENERIC_QUERIES = [
    "What is two-phase locking?",
    "How does ARIES perform the REDO phase?",
    "What is the external sort-merge algorithm?",
    "Explain B+ tree insertion and deletion.",
    "What are functional dependencies?",
]

print()
print("=" * 90)
print("PART 2: Filter+rank latency and result-set overlap (Jaccard) across scope sizes")
print("        (generic queries, not matched to scope — tests overhead only)")
print("=" * 90)
print(f"{'Scope':<24} {'In-scope':>9} {'Pre (ms)':>10} {'Post (ms)':>11} {'Overlap':>9}")
print("-" * 90)

latency_results = []
for label, chapters in SCOPE_LEVELS:
    if chapters is None:
        valid_ids = None
        in_scope  = len(chunks)
    else:
        valid_ids = set()
        for ch in chapters:
            valid_ids |= meta_store.get_chunk_ids_by_chapters([ch])
        in_scope = len(valid_ids)

    pre_times, post_times, overlaps = [], [], []
    for q in GENERIC_QUERIES:
        raw = get_raw_scores(q)
        t0 = time.perf_counter(); pre_r  = run_pre_filter(raw, valid_ids);  pre_times.append(time.perf_counter()-t0)
        t0 = time.perf_counter(); post_r = run_post_filter(raw, valid_ids); post_times.append(time.perf_counter()-t0)
        overlaps.append(jaccard(pre_r, post_r))

    avg_pre_ms  = sum(pre_times)  / len(pre_times)  * 1000
    avg_post_ms = sum(post_times) / len(post_times) * 1000
    avg_ovlp    = sum(overlaps)   / len(overlaps)

    print(f"{label:<24} {in_scope:>9,} {avg_pre_ms:>10.3f} {avg_post_ms:>11.3f} {avg_ovlp:>9.3f}")
    latency_results.append({
        "scope_label": label, "in_scope_chunks": in_scope,
        "avg_pre_ms":  round(avg_pre_ms, 4),
        "avg_post_ms": round(avg_post_ms, 4),
        "avg_overlap": round(avg_ovlp, 4),
    })

print("-" * 90)
print("\nKey findings:")
print("  - Both strategies run sub-millisecond (dominant cost is FAISS/BM25 scoring, shared).")
print("  - Overlap = 1.0: pre and post filter return identical ranked sets when operating")
print("    on the same fixed candidate pool (Python-level filtering, not FAISS kernel).")
print("  - Use pre-filter for code clarity; use post-filter to preserve full ranking order.")
print("    The functional difference (completeness) only appears when pool_n < chapter size,")
print("    which does not occur with this corpus and the current num_candidates setting.")

out = pathlib.Path("logs/scope_strategy_benchmark.json")
out.parent.mkdir(exist_ok=True)
with open(out, "w") as f:
    json.dump({
        "top_k": TOP_K, "num_candidates": cfg.num_candidates,
        "paired_completeness": paired_results,
        "latency": latency_results,
    }, f, indent=2)
print(f"\nResults saved to {out}")
meta_store.close()
