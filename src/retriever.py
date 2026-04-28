"""
retriever.py

Stores core retrieval logic using FAISS and BM25 scoring.
It also contains helpers for loading artifacts and filtering chunks.
"""

from __future__ import annotations

import pathlib
import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import nltk
from nltk.stem import WordNetLemmatizer

import faiss
import numpy as np
from src.embedder import CachedEmbedder

from src.config import RAGConfig
from src.index_builder import preprocess_for_bm25


# -------------------------- Embedder cache ------------------------------

_EMBED_CACHE: Dict[str, CachedEmbedder] = {}

def _get_embedder(model_name: str) -> CachedEmbedder:
    if model_name not in _EMBED_CACHE:
        # Use the cached embedding model to avoid reloading it on every call
        _EMBED_CACHE[model_name] = CachedEmbedder(model_name)
    return _EMBED_CACHE[model_name]


# -------------------------- Read artifacts -------------------------------

def load_artifacts(artifacts_dir: os.PathLike, index_prefix: str) -> Tuple[faiss.Index, List[str], List[str], Any]:
    """
    Loads:
      - FAISS index: {index_prefix}.faiss
      - chunks:      {index_prefix}_chunks.pkl
      - sources:     {index_prefix}_sources.pkl
    """
    artifacts_dir = pathlib.Path(artifacts_dir)
    faiss_index = faiss.read_index(str(artifacts_dir / f"{index_prefix}.faiss"))
    bm25_index  = pickle.load(open(artifacts_dir / f"{index_prefix}_bm25.pkl", "rb"))
    chunks      = pickle.load(open(artifacts_dir / f"{index_prefix}_chunks.pkl", "rb"))
    sources     = pickle.load(open(artifacts_dir / f"{index_prefix}_sources.pkl", "rb"))
    metadata = pickle.load(open(artifacts_dir / f"{index_prefix}_meta.pkl", "rb"))

    return faiss_index, bm25_index, chunks, sources, metadata


# -------------------------- Helper to get page nums for chunks -------------------------------

def get_page_numbers(chunk_indices: list[int], metadata: list[dict]) -> dict[int, List[int]]:
    if not metadata or not chunk_indices:
        return {}

    page_map: dict[int, List[int]] = {}

    for chunk_idx in chunk_indices:
        chunk_idx = int(chunk_idx)
        if 0 <= chunk_idx < len(metadata):
            chunk_pages = metadata[chunk_idx].get("page_numbers")
            if chunk_pages is None:
                continue  # don't store None; callers can default to [1]
            page_map[chunk_idx] = chunk_pages

    return page_map

# -------------------------- Filtering logic -----------------------------

def filter_retrieved_chunks(cfg: RAGConfig, chunks, ordered):
    topk_idxs = ordered[:cfg.top_k]
    return topk_idxs


# -------------------------- Scoped retrieval helpers --------------------

def apply_pre_filter(
    raw_scores: Dict[str, Dict[int, float]],
    valid_ids: Optional[set],
) -> Dict[str, Dict[int, float]]:
    """
    Pre-filter strategy: remove candidates outside *valid_ids* from every
    retriever's score dict *before* they reach the ranker.

    When *valid_ids* is None (no scope active), raw_scores is returned as-is.
    """
    if valid_ids is None:
        return raw_scores
    return {
        name: {k: v for k, v in scores.items() if k in valid_ids}
        for name, scores in raw_scores.items()
    }


def apply_post_filter(ordered: List[int], valid_ids: Optional[set]) -> List[int]:
    """
    Post-filter strategy: remove chunk indices outside *valid_ids* from the
    ranked list *after* the ranker has ordered them.

    When *valid_ids* is None (no scope active), ordered is returned as-is.
    """
    if valid_ids is None:
        return ordered
    return [i for i in ordered if i in valid_ids]


def compute_trust_score(
    topk_idxs: List[int],
    chunks: List[str],
    embedder,
    low_confidence_threshold: float = 0.30,
    faiss_index=None,
) -> Tuple[float, bool]:
    """Compute a trust score for a set of retrieved chunks.

    The score is the mean pairwise cosine similarity among the top-k chunk
    embeddings.  High agreement (high score) means the retrieved chunks are
    topically coherent and the answer is likely well-supported.  Low
    agreement flags that the retriever pulled from scattered topics, so the
    student should verify the answer in the textbook.

    Parameters
    ----------
    topk_idxs : list of chunk indices (as returned by filter_retrieved_chunks)
    chunks    : full chunk text list
    embedder  : a CachedEmbedder (or any object with .encode(texts) -> ndarray)
    low_confidence_threshold : mean similarity below this → low confidence
    faiss_index : optional FAISS index; when provided, pre-stored embeddings
                  are retrieved via index.reconstruct() instead of re-encoding
                  the chunk texts.  This is faster and more consistent since
                  these are the exact same vectors used for retrieval.

    Returns
    -------
    (mean_pairwise_similarity, is_low_confidence)
    """
    if len(topk_idxs) < 2:
        # With fewer than 2 chunks there is nothing to compare; trust fully.
        return 1.0, False

    valid_idxs = [i for i in topk_idxs if 0 <= i < len(chunks)]
    if len(valid_idxs) < 2:
        return 1.0, False

    if faiss_index is not None:
        # Preferred path: retrieve pre-computed embeddings from the FAISS index.
        # reconstruct(i) returns the exact vector stored at position i, which is
        # the same vector that drove retrieval — no model inference needed.
        try:
            vecs = np.array(
                [faiss_index.reconstruct(i) for i in valid_idxs],
                dtype="float32",
            )
        except Exception:
            # Fall back to re-encoding if reconstruct fails (e.g. IVF index
            # after a reconstruct_n call without an explicit mapping).
            texts = [chunks[i] for i in valid_idxs]
            vecs = embedder.encode(texts).astype("float32")
    else:
        texts = [chunks[i] for i in valid_idxs]
        vecs = embedder.encode(texts).astype("float32")

    # L2-normalise so dot product == cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    vecs = vecs / norms

    # Compute upper-triangle pairwise similarities
    sim_matrix = vecs @ vecs.T
    n = len(vecs)
    pairs = [(sim_matrix[i, j]) for i in range(n) for j in range(i + 1, n)]
    mean_sim = float(np.mean(pairs)) if pairs else 1.0

    return mean_sim, mean_sim < low_confidence_threshold


def format_citations(topk_idxs: List[int], meta: List[dict]) -> str:
    """
    Build a human-readable citation block for the top retrieved chunks.

    Each line shows: rank | section path | pages.
    Returns an empty string when metadata is unavailable.
    """
    if not meta or not topk_idxs:
        return ""
    lines = []
    for rank, idx in enumerate(topk_idxs, 1):
        if 0 <= idx < len(meta):
            m = meta[idx]
            section = m.get("section_path", m.get("section", "Unknown"))
            pages = m.get("page_numbers", [])
            page_str = (
                f"p. {pages[0]}" if len(pages) == 1
                else f"pp. {pages[0]}–{pages[-1]}" if pages
                else "page unknown"
            )
            lines.append(f"  [{rank}] {section}  ({page_str})")
    return "\n".join(lines)

# -------------------------- Retrieval core ------------------------------

class Retriever(ABC):
    @abstractmethod
    def get_scores(self, query: str, pool_size: int, chunks: List[str]):
        """Retrieves the top 'pool_size' chunks cores for a given query."""
        pass


class FAISSRetriever(Retriever):
    name = "faiss"

    def __init__(self, index, embed_model: str):
        self.index = index
        self.embedder = _get_embedder(embed_model)

    def get_scores(self,
                query: str,
                pool_size: int,
                chunks: List[str]) -> Dict[int, float]:
        """
        Returns FAISS scores for top 'pool_size' keyed by global chunk index.
        """
        # FAISS expects a 2D array
        q_vec = self.embedder.encode([query]).astype("float32")
        
        # Safety check on vector dimensions
        if q_vec.shape[1] !=  self.index.d:
            raise ValueError(
                f"Embedding dim mismatch: index={ self.index.d} vs query={q_vec.shape[1]}"
            )

        # Perform the search
        distances, indices =  self.index.search(q_vec, pool_size)

        # Remove invalid indices and ensure they are within bounds
        cand_idxs = [i for i in indices[0] if 0 <= i < len(chunks)]

        # Create the distance dictionary, ensuring we only include valid candidates
        dists = {idx: float(dist) for idx, dist in zip(cand_idxs, distances[0][:len(cand_idxs)])}

        # Invert distance to score: 1 / (1 + distance). Adding 1 avoids division by zero.
        return {
            idx: 1.0 / (1.0 + dist)
            for idx, dist in dists.items()
        }


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, index):
        self.index = index

    def get_scores(self,
                 query: str,
                 pool_size: int,
                 chunks: List[str]) -> Dict[int, float]:
        """
        Returns BM25 scores for top 'pool_size' keyed by global chunk index.
        """
        # Tokenize the query in the same way the index was built
        tokenized_query = preprocess_for_bm25(query)

        # Get scores for all documents in the corpus
        all_scores = self.index.get_scores(tokenized_query)

        # Find the indices of the top 'pool_size' scores
        num_candidates = min(pool_size, len(all_scores))
        top_k_indices = np.argpartition(-all_scores, kth=num_candidates-1)[:num_candidates]

        # Remove invalid indices and ensure they are within bounds
        top_k_indices = [i for i in top_k_indices if 0 <= i < len(chunks)]
        
        # Get the corresponding scores for the top indices
        top_scores = all_scores[top_k_indices]

        # Format the output as a dictionary of scores
        scores = {int(idx): float(score) for idx, score in zip(top_k_indices, top_scores)}

        return scores


class IndexKeywordRetriever(Retriever):
    name = "index_keywords"
    
    def __init__(self, extracted_index_path: os.PathLike, page_to_chunk_map_path: os.PathLike):
        """
        Retriever that uses textbook index keywords to boost chunks on relevant pages.
        
        Args:
            extracted_index_path: Path to extracted_index.json (keyword -> page numbers)
            page_to_chunk_map_path: Path to page_to_chunk_map.json (page -> chunk IDs)
        """
        import json
        nltk.download('wordnet', quiet=True)
        self.page_to_chunk_map = {}
        
        # Load and normalize index: lemmatize phrases as units
        # Build token->phrase mapping for fast lookup
        if os.path.exists(extracted_index_path):
            lemmatizer = WordNetLemmatizer()
            
            with open(extracted_index_path, 'r') as f:
                raw_index = json.load(f)
                self.phrase_to_pages = {}  # phrase -> pages
                self.token_to_phrases = {}  # token -> [phrases]
                
                for key, pages in raw_index.items():
                    # Lemmatize each word in the phrase but keep phrase together
                    key_lower = key.lower()
                    words = key_lower.split()
                    lemmatized_words = []
                    
                    for word in words:
                        cleaned = word.strip('.,!?()[]:"\'')
                        if not cleaned:
                            continue
                        lemmatized_words.append(self._lemmatize_word(cleaned, lemmatizer))
                    
                    lemmatized_phrase = ' '.join(lemmatized_words)
                    self.phrase_to_pages[lemmatized_phrase] = pages
                    
                    # Build reverse index: each token points to phrases containing it
                    for token in lemmatized_words:
                        if token not in self.token_to_phrases:
                            self.token_to_phrases[token] = []
                        self.token_to_phrases[token].append(lemmatized_phrase)
        else:
            self.phrase_to_pages = {}
            self.token_to_phrases = {}
        
        if os.path.exists(page_to_chunk_map_path):
            with open(page_to_chunk_map_path, 'r') as f:
                self.page_to_chunk_map = json.load(f)
    
    def get_scores(self, query: str, pool_size: int, chunks: List[str]) -> Dict[int, float]:
        """
        Returns scores for chunks that match index keywords.
        Score is proportional to the number of keyword hits.
        """
        keywords = self._extract_keywords(query)
        # chunk_id -> hit count
        chunk_hit_counts: Dict[int, int] = {} 
        
        # Match query keywords against index phrases (token overlap)
        for keyword in keywords:
            if keyword not in self.token_to_phrases:
                continue
            
            # Get all phrases containing this keyword token
            matching_phrases = self.token_to_phrases[keyword]
            
            for phrase in matching_phrases:
                page_numbers = self.phrase_to_pages[phrase]
                
                # Map pages to chunks
                for page_no in page_numbers:
                    chunk_ids = self.page_to_chunk_map.get(str(page_no), [])
                    for chunk_id in chunk_ids:
                        if chunk_id >= 0 and chunk_id < len(chunks):
                            chunk_hit_counts[chunk_id] = chunk_hit_counts.get(chunk_id, 0) + 1
        
        if not chunk_hit_counts:
            return {}
        
        # Normalize scores: more keyword hits = higher score
        max_hits = max(chunk_hit_counts.values())
        scores = {
            chunk_id: float(hit_count) / max_hits
            for chunk_id, hit_count in chunk_hit_counts.items()
        }
        
        return scores
    
    @staticmethod
    def _lemmatize_word(word: str, lemmatizer) -> str:
        """Lemmatize a word, trying noun then verb."""
        lemma = lemmatizer.lemmatize(word, pos='n')
        if lemma == word:
            lemma = lemmatizer.lemmatize(word, pos='v')
        return lemma
    
    @staticmethod
    def _extract_keywords(query: str) -> List[str]:
        """Extract keywords from query by removing stopwords and lemmatizing."""
        
        stopwords = {
            "the", "is", "at", "which", "on", "for", "a", "an", "and", "or", "in",
            "to", "of", "by", "with", "that", "this", "it", "as", "are", "was", 
            "what", "how", "why", "when", "where", "who", "does", "do", "be"
        }
        
        lemmatizer = WordNetLemmatizer()
        words = query.lower().split()
        keywords = []
        for word in words:
            cleaned = word.strip('.,!?()[]:"\'')
            if not cleaned or cleaned in stopwords:
                continue
            keywords.append(IndexKeywordRetriever._lemmatize_word(cleaned, lemmatizer))
        return keywords