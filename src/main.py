# noinspection PyUnresolvedReferences
import faiss  # force single OpenMP init

import argparse
import json
import pathlib
import sys
from typing import Dict, Optional, List, Tuple, Union, Any

from rich.live import Live
from rich.console import Console
from rich.markdown import Markdown

from src.config import RAGConfig
from src.generator import answer, double_answer, dedupe_generated_text
from src.index_builder import build_index
from src.index_updater import add_to_index
from src.instrumentation.logging import get_logger
from src.ranking.ranker import EnsembleRanker
from src.preprocessing.chunking import DocumentChunker
from src.query_enhancement import generate_hypothetical_document, contextualize_query
from src.retriever import (
    filter_retrieved_chunks,
    apply_pre_filter,
    apply_post_filter,
    format_citations,
    compute_trust_score,
    BM25Retriever,
    FAISSRetriever,
    IndexKeywordRetriever,
    get_page_numbers,
    load_artifacts,
)
from src.ranking.reranker import rerank
from src.cache import get_cache
from src.metadata_store import MetadataStore, ProvenanceStore
from src.utils import parse_chapter_arg, detect_scope_from_query

ANSWER_NOT_FOUND = "I'm sorry, but I don't have enough information to answer that question."

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Welcome to TokenSmith!")
    parser.add_argument("mode", choices=["index", "chat", "add-chapters"], help="operation mode")
    parser.add_argument("--pdf_dir", default="data/chapters/", help="directory containing PDF files")
    parser.add_argument("--index_prefix", default="textbook_index", help="prefix for generated index files")
    parser.add_argument("--partial", action="store_true",
        help="use a partial index stored in 'index/partial_sections' instead of 'index/sections'"
    )
    parser.add_argument("--model_path", help="path to generation model")
    parser.add_argument("--system_prompt_mode", choices=["baseline", "tutor", "concise", "detailed"], default="baseline")

    indexing_group = parser.add_argument_group("indexing options")
    indexing_group.add_argument("--keep_tables", action="store_true")
    indexing_group.add_argument("--multiproc_indexing", action="store_true")
    indexing_group.add_argument("--embed_with_headings", action="store_true")
    indexing_group.add_argument(
        "--chapters",
        nargs='+',
        type=int,
        help="a list of chapter numbers to index (e.g., --chapters 3 4 5)"
    )
    parser.add_argument(
        "--double_prompt",
        action="store_true",
        help="enable double prompting for higher quality answers",
    )

    # ---- scoped retrieval ----
    scope_group = parser.add_argument_group("scoped retrieval")
    scope_group.add_argument(
        "--scope_source",
        default=None,
        metavar="FILENAME",
        help="restrict retrieval to chunks whose source filename contains this substring",
    )
    scope_group.add_argument(
        "--scope_chapter",
        type=str,
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "restrict retrieval to one or more chapters. "
            "Examples: --scope_chapter 5  "
            "--scope_chapter 1 2 3  "
            "--scope_chapter 1-5"
        ),
    )
    scope_group.add_argument(
        "--scope_pages",
        type=int,
        nargs=2,
        default=None,
        metavar=("FROM", "TO"),
        help="restrict retrieval to chunks that touch any page in [FROM, TO]",
    )
    scope_group.add_argument(
        "--scope_strategy",
        choices=["pre", "post"],
        default="post",
        help="'pre'  – filter candidates before ranking; "
             "'post' – filter after ranking (default: post)",
    )

    return parser.parse_args()


def run_index_mode(args: argparse.Namespace, cfg: RAGConfig):
    strategy = cfg.get_chunk_strategy()
    chunker = DocumentChunker(strategy=strategy, keep_tables=args.keep_tables)
    artifacts_dir = cfg.get_artifacts_directory(partial=args.partial)

    data_dir = pathlib.Path("data")
    print(f"Looking for markdown files in {data_dir.resolve()}...")
    md_files = sorted(data_dir.glob("*.md"))
    print(f"Found {len(md_files)} markdown files.")
    print(f"First 5 markdown files: {[str(f) for f in md_files[:5]]}")

    if not md_files:
        print("ERROR: No markdown files found in data/.", file=sys.stderr)
        sys.exit(1)

    build_index(
        markdown_file=str(md_files[0]),
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        embedding_model_path=cfg.embed_model,
        embedding_model_context_window=cfg.embedding_model_context_window,
        artifacts_dir=artifacts_dir,
        index_prefix=args.index_prefix,
        use_multiprocessing=args.multiproc_indexing,
        use_headings=args.embed_with_headings,
        chapters_to_index=args.chapters,
    )

def run_add_chapters_mode(args: argparse.Namespace, cfg: RAGConfig):
    """Handles the logic for adding chapters to an existing index."""
    if not args.chapters:
        print("Please provide a list of chapters to add using the --chapters argument.")
        return

    strategy = cfg.get_chunk_strategy()
    chunker = DocumentChunker(strategy=strategy, keep_tables=args.keep_tables)
    artifacts_dir = cfg.get_artifacts_directory(partial=True)

    data_dir = pathlib.Path("data")
    print(f"Looking for markdown files in {data_dir.resolve()}...")
    md_files = sorted(data_dir.glob("*.md"))
    print(f"Found {len(md_files)} markdown files.")
    print(f"First 5 markdown files: {[str(f) for f in md_files[:5]]}")

    if not md_files:
        print("ERROR: No markdown files found in data/.", file=sys.stderr)
        sys.exit(1)

    add_to_index(
        markdown_file=str(md_files[0]),
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        embedding_model_path=cfg.embed_model,
        embedding_model_context_window=cfg.embedding_model_context_window,
        artifacts_dir=artifacts_dir,
        index_prefix=args.index_prefix,
        chapters_to_add=args.chapters,
        use_headings=args.embed_with_headings,
    )
    print("Successfully added chapters to the index.")

def use_indexed_chunks(question: str, chunks: list, cfg: RAGConfig, args: argparse.Namespace) -> list:
    # Logic for keyword matching from textbook index
    try:
        artifacts_dir = cfg.get_artifacts_directory(partial=args.partial)
        map_path = cfg.get_page_to_chunk_map_path(artifacts_dir, args.index_prefix)
        with open(map_path, 'r') as f:
            page_to_chunk_map = json.load(f)
        with open('data/extracted_index.json', 'r') as f:
            extracted_index = json.load(f)
    except FileNotFoundError:
        return []

    keywords = get_keywords(question)
    chunk_ids = {
        chunk_id
        for word in keywords
        if word in extracted_index
        for page_no in extracted_index[word]
        for chunk_id in page_to_chunk_map.get(str(page_no), [])
    }
    return [chunks[cid] for cid in chunk_ids], list(chunk_ids)

def get_answer(
    question: str,
    cfg: RAGConfig,
    args: argparse.Namespace,
    logger: Any,
    console: Optional["Console"],
    artifacts: Optional[Dict] = None,
    golden_chunks: Optional[list] = None,
    is_test_mode: bool = False,
    additional_log_info: Optional[Dict[str, Any]] = None,
    valid_ids: Optional[set] = None,
    retrieval_question: Optional[str] = None,
    meta_store=None,
    prov_store=None,
) -> Union[str, Tuple[str, List[Dict[str, Any]], Optional[str]]]:
    """
    Run a single query through the pipeline.

    Parameters
    ----------
    valid_ids : optional set of chunk IDs produced by MetadataStore.build_valid_ids().
                When provided, scoped retrieval is applied according to
                args.scope_strategy ('pre' or 'post').
    retrieval_question : if provided, used for FAISS/BM25 retrieval instead of
                         `question`. Useful when the original query contains scope
                         cues (e.g. "chapters 18-21") that improve embedding
                         alignment, while `question` is the cleaner form used for
                         generation and reranking.
    """
    chunks = artifacts["chunks"]
    sources = artifacts["sources"]
    retrievers = artifacts["retrievers"]
    ranker = artifacts["ranker"]

    # Ensure these locals exist for all control flows to avoid UnboundLocalError
    ranked_chunks: List[str] = []
    topk_idxs: List[int] = []
    scores = []

    # ---- resolve scope strategy ----
    scope_strategy = getattr(args, "scope_strategy", "post")

    cache = get_cache(cfg)
    normalized_question = cache.normalize_question(question)
    config_cache_key = cache.make_config_key(cfg, args, golden_chunks)
    question_embedding = cache.compute_embedding(normalized_question, retrievers, cfg.embed_model)

    semantic_hit = cache.lookup(config_cache_key, question_embedding, normalized_question)

    # Return cached answer if found
    if semantic_hit is not None:
        ans = semantic_hit.get("answer", "")
        if is_test_mode:
            return ans, semantic_hit.get("chunks_info"), semantic_hit.get("hyde_query")
        console.print("Using cached answer")
        render_final_answer(console, ans)
        return ans

    # Step 1: Get chunks (golden, retrieved, or none)
    chunks_info = None
    hyde_query = None
    if golden_chunks and cfg.use_golden_chunks:
        ranked_chunks = golden_chunks
    elif cfg.disable_chunks:
        ranked_chunks = []
    elif cfg.use_indexed_chunks:
        ranked_chunks, topk_idxs = use_indexed_chunks(question, chunks, cfg, args)
    else:
        retrieval_query = retrieval_question if retrieval_question is not None else question
        if cfg.use_hyde:
            retrieval_query = generate_hypothetical_document(retrieval_query, cfg.gen_model, max_tokens=cfg.hyde_max_tokens)

        pool_n = max(cfg.num_candidates, cfg.top_k + 10)
        # When post-filtering by scope, widen the candidate pool so that
        # in-scope chunks are not crowded out by out-of-scope results.
        if valid_ids is not None and scope_strategy == "post":
            pool_n = max(pool_n, len(valid_ids))
        raw_scores: Dict[str, Dict[int, float]] = {}
        for retriever in retrievers:
            raw_scores[retriever.name] = retriever.get_scores(retrieval_query, pool_n, chunks)

        # ---- Scoped retrieval: pre-filter strategy ----
        # Remove out-of-scope candidates before the ranker sees them so that
        # only in-scope chunks compete for the top-k slots.
        if valid_ids is not None and scope_strategy == "pre":
            raw_scores = apply_pre_filter(raw_scores, valid_ids)

        # Step 2: Ranking
        ordered, scores = ranker.rank(raw_scores=raw_scores)

        # ---- Scoped retrieval: post-filter strategy ----
        # Re-order after ranking but before taking top-k; out-of-scope chunks
        # are removed from the ranked list regardless of their score.
        if valid_ids is not None and scope_strategy == "post":
            ordered = apply_post_filter(ordered, valid_ids)

        topk_idxs = filter_retrieved_chunks(cfg, chunks, ordered)
        ranked_chunks = [chunks[i] for i in topk_idxs]

        # Capture chunk info if in test mode
        if is_test_mode:
            faiss_scores = raw_scores.get("faiss", {})
            bm25_scores = raw_scores.get("bm25", {})
            index_scores = raw_scores.get("index_keywords", {})

            faiss_ranked = sorted(faiss_scores.keys(), key=lambda i: faiss_scores[i], reverse=True)
            bm25_ranked = sorted(bm25_scores.keys(), key=lambda i: bm25_scores[i], reverse=True)
            index_ranked = sorted(index_scores.keys(), key=lambda i: index_scores[i], reverse=True)

            faiss_ranks = {idx: rank + 1 for rank, idx in enumerate(faiss_ranked)}
            bm25_ranks = {idx: rank + 1 for rank, idx in enumerate(bm25_ranked)}
            index_ranks = {idx: rank + 1 for rank, idx in enumerate(index_ranked)}

            chunks_info = []
            for rank, idx in enumerate(topk_idxs, 1):
                chunks_info.append({
                    "rank": rank,
                    "chunk_id": idx,
                    "content": chunks[idx],
                    "faiss_score": faiss_scores.get(idx, 0),
                    "faiss_rank": faiss_ranks.get(idx, 0),
                    "bm25_score": bm25_scores.get(idx, 0),
                    "bm25_rank": bm25_ranks.get(idx, 0),
                    "index_score": index_scores.get(idx, 0),
                    "index_rank": index_ranks.get(idx, 0),
                })

        # Step 3: Final re-ranking
        ranked_chunks = rerank(question, ranked_chunks, mode=cfg.rerank_mode, top_n=cfg.rerank_top_k)

    if not ranked_chunks and not cfg.disable_chunks:
        if console:
            console.print(f"\n{ANSWER_NOT_FOUND}\n")
        return ANSWER_NOT_FOUND

    # Step 4: Generation
    model_path = cfg.gen_model
    system_prompt = args.system_prompt_mode or cfg.system_prompt_mode

    use_double = getattr(args, "double_prompt", False) or cfg.use_double_prompt
    if use_double:
        stream_iter = double_answer(
            question,
            ranked_chunks,
            model_path,
            max_tokens=cfg.max_gen_tokens,
            system_prompt_mode=system_prompt,
        )
    else:
        stream_iter = answer(
            question,
            ranked_chunks,
            model_path,
            max_tokens=cfg.max_gen_tokens,
            system_prompt_mode=system_prompt,
        )

    if is_test_mode:
        ans = ""
        for delta in stream_iter:
            ans += delta
        ans = dedupe_generated_text(ans)
        return ans, chunks_info, hyde_query
    else:
        ans = render_streaming_ans(console, stream_iter)

        # ---- Source citations display ----
        meta = artifacts.get("meta", [])
        if topk_idxs and meta and console:
            citation_text = format_citations(topk_idxs, meta)
            if citation_text:
                console.print("\n[bold yellow]Sources:[/bold yellow]")
                console.print(citation_text)

        # ---- Trust score display ----
        try:
            faiss_retriever = next(
                (r for r in artifacts.get("retrievers", []) if isinstance(r, FAISSRetriever)),
                None,
            )
            if faiss_retriever and topk_idxs and console:
                trust, low_conf = compute_trust_score(
                    topk_idxs,
                    chunks,
                    faiss_retriever.embedder,
                    faiss_index=faiss_retriever.index,  # use stored vectors, no re-encoding
                    low_confidence_threshold=cfg.trust_score_threshold,
                )
                if low_conf:
                    console.print(
                        f"\n[bold red]⚠ Low confidence[/bold red] "
                        f"(chunk agreement: {trust:.2f}). "
                        "Retrieved sources cover different topics — verify this answer in the source material."
                    )
                else:
                    console.print(
                        f"\n[dim]Confidence: {trust:.2f} (chunk agreement)[/dim]"
                    )
        except Exception:
            pass  # trust score is best-effort; never block the answer

        # Logging (JSON file)
        page_nums = get_page_numbers(topk_idxs, meta)
        logger.save_chat_log(
            query=question,
            config_state=cfg.get_config_state(),
            ordered_scores=scores[:len(topk_idxs)] if scores else [],
            chat_request_params={
                "system_prompt": system_prompt,
                "max_tokens": cfg.max_gen_tokens,
            },
            top_idxs=topk_idxs,
            chunks=[chunks[i] for i in topk_idxs],
            sources=[sources[i] for i in topk_idxs],
            page_map=page_nums,
            full_response=ans,
            top_k=len(topk_idxs),
            additional_log_info=additional_log_info,
        )

    # Step 5: Store in semantic cache
    cache_payload = {
        "answer": ans,
        "chunks_info": chunks_info,
        "hyde_query": hyde_query,
        "chunk_indices": topk_idxs,
    }
    if question_embedding is None:
        question_embedding = cache.compute_embedding(normalized_question, retrievers, cfg.embed_model)
    cache.store(
        config_cache_key,
        normalized_question,
        question_embedding,
        cache_payload
    )

    # Provenance logging — only when stores are supplied by the caller
    if meta_store is not None and prov_store is not None and topk_idxs:
        try:
            prov_chunks = meta_store.get_metadata_for_chunks(topk_idxs)
            prov_store.log_query(
                query=question,
                retrieved_chunks=prov_chunks,
                answer=ans,
                config_state=cfg.get_config_state(),
            )
        except Exception:
            pass

    if is_test_mode:
        return ans, chunks_info, hyde_query

    return ans

def render_streaming_ans(console, stream_iter):
    ans = ""
    is_first = True
    with Live(console=console, refresh_per_second=8) as live:
        for delta in stream_iter:
            if is_first:
                console.print("\n[bold cyan]=== START OF ANSWER ===[/bold cyan]\n")
                is_first = False
            ans += delta
            live.update(Markdown(ans))
    ans = dedupe_generated_text(ans)
    live.update(Markdown(ans))
    console.print("\n[bold cyan]=== END OF ANSWER ===[/bold cyan]\n")
    return ans

# Fully generated answer without streaming (Usage: cache hits)
def render_final_answer(console, ans):
    if not console:
        raise ValueError("Console must be non null for rendering.")
    console.print(
        "\n[bold cyan]==================== START OF ANSWER ===================[/bold cyan]\n"
    )
    console.print(Markdown(ans))
    console.print(
        "\n[bold cyan]===================== END OF ANSWER ====================[/bold cyan]\n"
    )

def get_keywords(question: str) -> list:
    """
    Simple keyword extraction from the question.
    """
    stopwords = set([
        "the", "is", "at", "which", "on", "for", "a", "an", "and", "or", "in", 
        "to", "of", "by", "with", "that", "this", "it", "as", "are", "was", "what"
    ])
    words = question.lower().split()
    keywords = [word.strip('.,!?()[]') for word in words if word not in stopwords]
    return keywords

def run_chat_session(args: argparse.Namespace, cfg: RAGConfig):
    logger = get_logger()
    console = Console()

    print("Initializing TokenSmith Chat...")
    try:
        artifacts_dir = cfg.get_artifacts_directory(partial=args.partial)
        cfg.page_to_chunk_map_path = cfg.get_page_to_chunk_map_path(artifacts_dir, args.index_prefix)
        faiss_idx, bm25_idx, chunks, sources, meta = load_artifacts(artifacts_dir, args.index_prefix)
        print(f"Loaded {len(chunks)} chunks and {len(sources)} sources from artifacts.")
        retrievers = [FAISSRetriever(faiss_idx, cfg.embed_model), BM25Retriever(bm25_idx)]
        if cfg.ranker_weights.get("index_keywords", 0) > 0:
            retrievers.append(IndexKeywordRetriever(cfg.extracted_index_path, cfg.page_to_chunk_map_path))

        ranker = EnsembleRanker(ensemble_method=cfg.ensemble_method, weights=cfg.ranker_weights, rrf_k=int(cfg.rrf_k))
        print("Loaded retrievers and initialized ranker.")
        artifacts = {"chunks": chunks, "sources": sources, "retrievers": retrievers, "ranker": ranker, "meta": meta}
    except Exception as e:
        print(f"ERROR: {e}. Run 'index' mode first.")
        sys.exit(1)

    # ---- Initialize metadata and provenance stores ----
    meta_store = MetadataStore(db_path="index/metadata.db")
    prov_store = ProvenanceStore(db_path="index/provenance.db")

    # Populate SQLite from pkl on first run (idempotent – skips existing rows)
    if meta_store.is_empty() and meta:
        print("Populating SQLite metadata store from index artifacts...")
        meta_store.populate_from_metadata(meta)
        print(f"  Done ({len(meta)} chunks registered).")

    # ---- Build valid_ids for scoped retrieval (once per session) ----
    scope_source = getattr(args, "scope_source", None)
    scope_chapter_raw = getattr(args, "scope_chapter", None)
    scope_chapters = parse_chapter_arg(scope_chapter_raw) if scope_chapter_raw else None
    scope_pages = getattr(args, "scope_pages", None)   # [from, to] or None
    from_page = scope_pages[0] if scope_pages else None
    to_page   = scope_pages[1] if scope_pages else None

    valid_ids = meta_store.build_valid_ids(
        source=scope_source,
        chapters=scope_chapters,
        from_page=from_page,
        to_page=to_page,
    )

    # Print active scope summary
    if valid_ids is not None:
        scope_parts = []
        if scope_source:
            scope_parts.append(f"source='{scope_source}'")
        if scope_chapters is not None:
            scope_parts.append(f"chapters={scope_chapters}")
        if scope_pages:
            scope_parts.append(f"pages={scope_pages[0]}-{scope_pages[1]}")
        strategy = getattr(args, "scope_strategy", "post")
        console.print(
            f"[bold green]Scope active[/bold green]: {', '.join(scope_parts)} "
            f"| {len(valid_ids)} eligible chunks | strategy={strategy}"
        )
    else:
        console.print("[dim]No scope filter active – searching all chunks.[/dim]")

    chat_history = []
    additional_log_info = {}
    print("Initialization complete. You can start asking questions!")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'history' to review your past queries and their sources.")
    print("Tip: mention 'chapter 5', 'first half of the book', or 'pages 100-200' in your question to auto-scope retrieval.")

    while True:
        try:
            q = input("\nAsk > ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            # ---- history command ----
            if q.lower() == "history":
                _print_history(prov_store, console)
                continue

            effective_q = q
            if cfg.enable_history and chat_history:
                try:
                    effective_q = contextualize_query(q, chat_history, cfg.gen_model)
                    additional_log_info["is_contextualizing_query"] = True
                    additional_log_info["contextualized_query"] = effective_q
                    additional_log_info["original_query"] = q
                    additional_log_info["chat_history"] = chat_history
                    print(f"Contextualized Query: {effective_q}")
                except Exception as e:
                    print(f"Warning: Failed to contextualize query: {e}. Using original query.")
                    effective_q = q

            # ---- Auto-scope detection (125% goal) ----
            # Parse the raw user query for explicit chapter/page mentions and
            # build a per-query valid_ids that narrows the session scope.
            query_valid_ids = valid_ids  # default: use session-level scope
            scope_applied = False        # tracks whether per-query scope was set
            try:
                detected = detect_scope_from_query(q, max_chapter=meta_store.get_max_chapter())
                det_chapters = detected.get("chapters")
                det_pages    = detected.get("pages")
                if det_chapters or det_pages:
                    det_from   = det_pages[0] if det_pages else None
                    det_to     = det_pages[1] if det_pages else None
                    auto_ids   = meta_store.build_valid_ids(
                        chapters=det_chapters,
                        from_page=det_from,
                        to_page=det_to,
                    )
                    if auto_ids is not None:
                        # Intersect with any session scope already active
                        candidate = (
                            auto_ids if valid_ids is None else valid_ids & auto_ids
                        )
                        scope_desc = []
                        if det_chapters:
                            scope_desc.append(f"chapters={det_chapters}")
                        if det_pages:
                            scope_desc.append(f"pages={det_pages[0]}-{det_pages[1]}")
                        if candidate:
                            query_valid_ids = candidate
                            scope_applied = True
                            console.print(
                                f"[bold green]Auto-scope detected[/bold green]: "
                                f"{', '.join(scope_desc)} "
                                f"| {len(query_valid_ids)} eligible chunks"
                            )
                            additional_log_info["auto_scope"] = detected
                        else:
                            # Intersection is empty — the detected scope has no
                            # indexed chunks (e.g. a chapter number not in the
                            # textbook).  Fall back to the session scope so the
                            # query still gets an answer.
                            console.print(
                                f"[yellow]Auto-scope detected ({', '.join(scope_desc)}) "
                                f"but no indexed chunks match — falling back to session scope.[/yellow]"
                            )
            except Exception as e:
                print(f"Warning: auto-scope detection failed: {e}")

            # When the user's query contains an explicit scope cue (e.g.
            # "chapters 18-21"), use the original query for FAISS retrieval.
            # Contextualization strips those cue words, producing a shorter
            # query whose embedding aligns poorly with the scoped chapters.
            # For unscoped follow-ups, use the contextualized query so that
            # pronoun/reference resolution still helps retrieval.
            retrieval_q = q if scope_applied else effective_q

            ans = get_answer(
                effective_q,
                cfg,
                args,
                logger,
                console,
                artifacts=artifacts,
                additional_log_info=additional_log_info,
                valid_ids=query_valid_ids,
                retrieval_question=retrieval_q,
                meta_store=meta_store,
                prov_store=prov_store,
            )

            # When scope is active and nothing was found, give a targeted hint
            # depending on whether the scope came from auto-detection or CLI flags.
            if ans == ANSWER_NOT_FOUND and query_valid_ids is not None:
                if scope_applied:
                    console.print(
                        "\n[yellow]Tip: no chunks matched the chapter/page scope "
                        "detected in your question. Try rephrasing without mentioning "
                        "a specific chapter or page number to search the full document.[/yellow]"
                    )
                else:
                    console.print(
                        "\n[yellow]Tip: no chunks matched within the active scope. "
                        "Try asking without a scope filter (omit --scope_chapter / "
                        "--scope_pages / --scope_source) to search the full document.[/yellow]"
                    )

            # Update Chat history (atomic for user + assistant turn)
            try:
                user_turn = {"role": "user", "content": q}
                assistant_turn = {"role": "assistant", "content": ans}
                chat_history += [user_turn, assistant_turn]
            except Exception as e:
                print(f"Warning: Failed to update chat history: {e}")

            # Trim chat history to avoid exceeding context window
            if len(chat_history) > cfg.max_history_turns * 2:
                chat_history = chat_history[-cfg.max_history_turns * 2:]

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            break

    meta_store.close()
    prov_store.close()


def _print_history(prov_store: "ProvenanceStore", console: "Console") -> None:
    """Display the last 10 queries with their retrieved sources."""
    rows = prov_store.get_history(limit=10)
    if not rows:
        console.print("[dim]No query history yet.[/dim]")
        return
    console.print("\n[bold cyan]=== Query History (newest first) ===[/bold cyan]\n")
    for row in rows:
        console.print(f"[bold]#{row['id']}[/bold] [{row['timestamp']}]")
        console.print(f"  Q: {row['query']}")
        if row["retrieved_chunks"]:
            src_lines = []
            for c in row["retrieved_chunks"][:3]:
                pages = c.get("page_numbers", [])
                page_str = f"pp. {pages[0]}–{pages[-1]}" if pages else "?"
                src_lines.append(f"    • {c.get('section_path', c.get('section','?'))} ({page_str})")
            console.print("  Sources:\n" + "\n".join(src_lines))
        console.print(f"  A: {row['answer']}\n")



def main():
    args = parse_args()
    config_path = pathlib.Path("config/config.yaml")
    if not config_path.exists(): raise FileNotFoundError("config/config.yaml not found.")
    cfg = RAGConfig.from_yaml(config_path)
    print(f"Loaded configuration from {config_path.resolve()}.")
    if args.mode == "index":
        run_index_mode(args, cfg)
    elif args.mode == "chat":
        run_chat_session(args, cfg)
    elif args.mode == "add-chapters":
        run_add_chapters_mode(args, cfg)

if __name__ == "__main__":
    main()
