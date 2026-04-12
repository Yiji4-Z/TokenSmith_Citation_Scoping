"""
tests/test_metadata_store.py

Unit tests for MetadataStore and ProvenanceStore.

Run with: conda run -n tokensmith python -m pytest tests/test_metadata_store.py -v
"""

import json
import os
import tempfile

import pytest

from src.metadata_store import MetadataStore, ProvenanceStore, _extract_chapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_METADATA = [
    {
        "chunk_id": 0,
        "filename": "data/textbook.md",
        "section": "Section 1.1 Introduction",
        "section_path": "Chapter 1 Section 1.1 Introduction",
        "page_numbers": [1, 2],
        "char_len": 500,
        "word_len": 80,
    },
    {
        "chunk_id": 1,
        "filename": "data/textbook.md",
        "section": "Section 1.2 Overview",
        "section_path": "Chapter 1 Section 1.2 Overview",
        "page_numbers": [3],
        "char_len": 400,
        "word_len": 60,
    },
    {
        "chunk_id": 2,
        "filename": "data/textbook.md",
        "section": "Section 18.1 Lock-Based Protocols",
        "section_path": "Chapter 18 Section 18.1 Lock-Based Protocols",
        "page_numbers": [1295, 1296],
        "char_len": 1800,
        "word_len": 300,
    },
    {
        "chunk_id": 3,
        "filename": "data/textbook.md",
        "section": "Section 18.1.3 Two-Phase Locking",
        "section_path": "Chapter 18 Section 18.1.3 The Two-Phase Locking Protocol",
        "page_numbers": [1296, 1297],
        "char_len": 1600,
        "word_len": 250,
    },
    {
        "chunk_id": 4,
        "filename": "data/slides.md",
        "section": "Overview",
        "section_path": "Chapter 5 Overview",
        "page_numbers": [50, 51],
        "char_len": 300,
        "word_len": 40,
    },
]


@pytest.fixture
def store(tmp_path):
    db = str(tmp_path / "test_meta.db")
    s = MetadataStore(db_path=db)
    s.populate_from_metadata(SAMPLE_METADATA)
    yield s
    s.close()


@pytest.fixture
def prov_store(tmp_path):
    db = str(tmp_path / "test_prov.db")
    s = ProvenanceStore(db_path=db)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# _extract_chapter helper
# ---------------------------------------------------------------------------

class TestExtractChapter:
    def test_standard_path(self):
        assert _extract_chapter("Chapter 18 Section 18.1 Two-Phase Locking") == 18

    def test_chapter_1(self):
        assert _extract_chapter("Chapter 1 Introduction") == 1

    def test_none_on_empty(self):
        assert _extract_chapter("") is None

    def test_none_on_no_chapter(self):
        assert _extract_chapter("Some section without chapter") is None

    def test_large_chapter_number(self):
        assert _extract_chapter("Chapter 26 Advanced Topics") == 26


# ---------------------------------------------------------------------------
# MetadataStore – population
# ---------------------------------------------------------------------------

class TestMetadataStorePopulation:
    def test_populate_inserts_all_rows(self, store):
        ids = store.get_all_chunk_ids()
        assert ids == {0, 1, 2, 3, 4}

    def test_idempotent_second_populate(self, store):
        # Second populate with same data should not raise or duplicate rows
        store.populate_from_metadata(SAMPLE_METADATA)
        ids = store.get_all_chunk_ids()
        assert len(ids) == len(SAMPLE_METADATA)

    def test_is_empty_false_after_populate(self, tmp_path):
        db = str(tmp_path / "empty.db")
        s = MetadataStore(db_path=db)
        assert s.is_empty()
        s.populate_from_metadata(SAMPLE_METADATA)
        assert not s.is_empty()
        s.close()


# ---------------------------------------------------------------------------
# MetadataStore – scoped retrieval
# ---------------------------------------------------------------------------

class TestScopedRetrieval:
    def test_filter_by_chapter(self, store):
        ids = store.get_chunk_ids_by_chapter(18)
        assert ids == {2, 3}

    def test_filter_by_chapter_1(self, store):
        ids = store.get_chunk_ids_by_chapter(1)
        assert ids == {0, 1}

    def test_filter_by_nonexistent_chapter(self, store):
        ids = store.get_chunk_ids_by_chapter(99)
        assert ids == set()

    def test_filter_by_source_filename(self, store):
        ids = store.get_chunk_ids_by_source("slides")
        assert ids == {4}

    def test_filter_by_source_case_insensitive(self, store):
        ids = store.get_chunk_ids_by_source("TEXTBOOK")
        assert ids == {0, 1, 2, 3}

    def test_filter_by_source_no_match(self, store):
        ids = store.get_chunk_ids_by_source("nonexistent_file")
        assert ids == set()

    def test_filter_by_pages_exact(self, store):
        ids = store.get_chunk_ids_by_pages(1295, 1296)
        # chunks 2 (pages [1295,1296]) and 3 (pages [1296,1297]) both touch this range
        assert {2, 3}.issubset(ids)

    def test_filter_by_pages_no_match(self, store):
        ids = store.get_chunk_ids_by_pages(9000, 9999)
        assert ids == set()

    def test_filter_by_pages_single(self, store):
        ids = store.get_chunk_ids_by_pages(3, 3)
        assert 1 in ids
        assert 0 not in ids

    def test_build_valid_ids_none_when_no_filter(self, store):
        result = store.build_valid_ids()
        assert result is None

    def test_build_valid_ids_single_chapter(self, store):
        result = store.build_valid_ids(chapters=[18])
        assert result == {2, 3}

    def test_build_valid_ids_multiple_chapters(self, store):
        # chapters 1 and 18 together should return {0,1,2,3}
        result = store.build_valid_ids(chapters=[1, 18])
        assert result == {0, 1, 2, 3}

    def test_build_valid_ids_chapter_range(self, store):
        # chapters 1-5 covers ch1 ({0,1}) and ch5 ({4})
        from src.utils import parse_chapter_arg
        chapters = parse_chapter_arg(["1-5"])
        result = store.build_valid_ids(chapters=chapters)
        assert {0, 1, 4}.issubset(result)

    def test_build_valid_ids_source_and_chapter(self, store):
        # textbook.md has chunks in ch1 and ch18; slides.md has ch5
        result = store.build_valid_ids(source="textbook", chapters=[18])
        assert result == {2, 3}

    def test_build_valid_ids_chapter_and_pages(self, store):
        # ch18 chunks are {2,3}; pages 1296-1296 touches both chunk 2 and 3
        result = store.build_valid_ids(chapters=[18], from_page=1296, to_page=1296)
        assert result == {2, 3}

    def test_build_valid_ids_empty_intersection(self, store):
        # ch1 chunks are {0,1}; pages 1295-1296 are only in ch18 chunks
        result = store.build_valid_ids(chapters=[1], from_page=1295, to_page=1296)
        assert result == set()

    def test_get_chunk_ids_by_chapters_multiple(self, store):
        # chapters [1, 18] → {0,1,2,3}; chapter 5 → {4}
        ids = store.get_chunk_ids_by_chapters([1, 18])
        assert ids == {0, 1, 2, 3}

    def test_get_chunk_ids_by_chapters_empty_list(self, store):
        assert store.get_chunk_ids_by_chapters([]) == set()

    def test_get_chunk_ids_by_chapters_single(self, store):
        assert store.get_chunk_ids_by_chapters([18]) == {2, 3}


# ---------------------------------------------------------------------------
# MetadataStore – get_metadata_for_chunks
# ---------------------------------------------------------------------------

class TestGetMetadataForChunks:
    def test_returns_correct_fields(self, store):
        rows = store.get_metadata_for_chunks([2])
        assert len(rows) == 1
        row = rows[0]
        assert row["chunk_id"] == 2
        assert row["chapter"] == 18
        assert 1295 in row["page_numbers"]

    def test_empty_input(self, store):
        assert store.get_metadata_for_chunks([]) == []

    def test_multiple_chunks(self, store):
        rows = store.get_metadata_for_chunks([0, 1, 4])
        chunk_ids = {r["chunk_id"] for r in rows}
        assert chunk_ids == {0, 1, 4}


# ---------------------------------------------------------------------------
# ProvenanceStore
# ---------------------------------------------------------------------------

class TestProvenanceStore:
    def test_log_and_retrieve(self, prov_store):
        prov_store.log_query(
            query="What is two-phase locking?",
            retrieved_chunks=[{"chunk_id": 2, "section": "Section 18.1", "page_numbers": [1295]}],
            answer="Two-phase locking ensures serializability.",
        )
        history = prov_store.get_history(limit=10)
        assert len(history) == 1
        assert history[0]["query"] == "What is two-phase locking?"

    def test_history_newest_first(self, prov_store):
        for i in range(3):
            prov_store.log_query(query=f"q{i}", retrieved_chunks=[], answer=f"a{i}")
        history = prov_store.get_history(limit=10)
        assert history[0]["query"] == "q2"
        assert history[1]["query"] == "q1"

    def test_history_limit(self, prov_store):
        for i in range(5):
            prov_store.log_query(query=f"q{i}", retrieved_chunks=[], answer=f"a{i}")
        history = prov_store.get_history(limit=2)
        assert len(history) == 2

    def test_empty_history(self, prov_store):
        assert prov_store.get_history() == []

    def test_answer_truncated_in_history(self, prov_store):
        long_answer = "x" * 400
        prov_store.log_query(query="q", retrieved_chunks=[], answer=long_answer)
        history = prov_store.get_history()
        assert len(history[0]["answer"]) <= 305  # 300 chars + ellipsis


# ---------------------------------------------------------------------------
# Integration: apply_pre_filter / apply_post_filter
# ---------------------------------------------------------------------------

class TestScopeFilters:
    """Integration tests for the retriever-level filter helpers."""

    def test_apply_pre_filter_removes_out_of_scope(self):
        from src.retriever import apply_pre_filter
        raw_scores = {
            "faiss": {0: 0.9, 1: 0.8, 2: 0.7, 3: 0.6},
            "bm25":  {0: 0.5, 2: 0.4},
        }
        valid_ids = {0, 2}
        filtered = apply_pre_filter(raw_scores, valid_ids)
        assert set(filtered["faiss"].keys()) == {0, 2}
        assert set(filtered["bm25"].keys()) == {0, 2}

    def test_apply_pre_filter_no_op_when_none(self):
        from src.retriever import apply_pre_filter
        raw_scores = {"faiss": {0: 0.9, 1: 0.8}}
        result = apply_pre_filter(raw_scores, None)
        assert result is raw_scores

    def test_apply_post_filter_removes_out_of_scope(self):
        from src.retriever import apply_post_filter
        ordered = [3, 1, 0, 2]
        valid_ids = {0, 2}
        result = apply_post_filter(ordered, valid_ids)
        assert result == [0, 2]

    def test_apply_post_filter_preserves_order(self):
        from src.retriever import apply_post_filter
        ordered = [5, 3, 1]
        valid_ids = {1, 3, 5}
        result = apply_post_filter(ordered, valid_ids)
        assert result == [5, 3, 1]

    def test_apply_post_filter_no_op_when_none(self):
        from src.retriever import apply_post_filter
        ordered = [0, 1, 2]
        result = apply_post_filter(ordered, None)
        assert result is ordered


# ---------------------------------------------------------------------------
# parse_chapter_arg
# ---------------------------------------------------------------------------

class TestParseChapterArg:
    """Unit tests for the CLI chapter argument parser."""

    def test_single_chapter(self):
        from src.utils import parse_chapter_arg
        assert parse_chapter_arg(["18"]) == [18]

    def test_multiple_chapters(self):
        from src.utils import parse_chapter_arg
        assert parse_chapter_arg(["1", "2", "3"]) == [1, 2, 3]

    def test_range(self):
        from src.utils import parse_chapter_arg
        assert parse_chapter_arg(["1-5"]) == [1, 2, 3, 4, 5]

    def test_range_and_individual(self):
        from src.utils import parse_chapter_arg
        assert parse_chapter_arg(["1-3", "7"]) == [1, 2, 3, 7]

    def test_single_element_range(self):
        from src.utils import parse_chapter_arg
        assert parse_chapter_arg(["5-5"]) == [5]

    def test_invalid_range_raises(self):
        from src.utils import parse_chapter_arg
        import pytest
        with pytest.raises(ValueError, match="Invalid chapter range"):
            parse_chapter_arg(["1-abc"])

    def test_invalid_integer_raises(self):
        from src.utils import parse_chapter_arg
        import pytest
        with pytest.raises(ValueError, match="Invalid chapter number"):
            parse_chapter_arg(["foo"])
