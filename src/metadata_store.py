"""
metadata_store.py

SQLite-backed stores for TokenSmith.

Two stores:
  1. MetadataStore   – persists per-chunk metadata (chapter, section, pages,
                       source) and exposes scoped-retrieval helpers that return
                       the set of valid chunk IDs for pre- or post-filtering.
  2. ProvenanceStore – logs every query + retrieved chunks + answer so students
                       can review their study history via the 'history' command.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_chapter(section_path: str) -> Optional[int]:
    """Extract a chapter/unit/part/module number from a section_path string.

    Handles common structural prefixes used across different books:
      "Chapter 18 …"   "Unit 3 …"   "Part 2 …"
      "Module 5 …"     "Ch. 14 …"   "Lecture 7 …"
    Returns the integer that follows the prefix, or None if none is found.
    """
    m = re.match(
        r"(?:Chapter|Unit|Part|Module|Lecture|Ch\.?|Lesson)\s+(\d+)",
        section_path or "",
        re.IGNORECASE,
    )
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# MetadataStore
# ---------------------------------------------------------------------------

class MetadataStore:
    """
    Persists chunk metadata in a SQLite database.

    The table is populated once at index-build time (or lazily on first use)
    via ``populate_from_metadata``, and then queried at retrieval time to
    obtain the set of valid chunk IDs for scoped retrieval.

    Schema
    ------
    chunk_metadata(chunk_id PK, filename, chapter, section, section_path,
                   page_numbers JSON, char_len, word_len)
    """

    _DDL = """
        CREATE TABLE IF NOT EXISTS chunk_metadata (
            chunk_id     INTEGER PRIMARY KEY,
            filename     TEXT,
            chapter      INTEGER,
            section      TEXT,
            section_path TEXT,
            page_numbers TEXT,
            char_len     INTEGER,
            word_len     INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_chapter  ON chunk_metadata(chapter);
        CREATE INDEX IF NOT EXISTS idx_filename ON chunk_metadata(filename);
    """

    def __init__(self, db_path: str = "index/metadata.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._con.executescript(self._DDL)
        self._con.commit()

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        row = self._con.execute("SELECT COUNT(*) FROM chunk_metadata").fetchone()
        return row[0] == 0

    def populate_from_metadata(self, metadata: List[Dict]) -> int:
        """
        Insert chunk metadata records, skipping rows that already exist.

        chunk_id is stored as the *list-position index* (0 to N-1) so that it
        matches the integer indices returned by FAISS and BM25 at retrieval
        time.  The stored ``m["chunk_id"]`` field is NOT used because it can
        differ from the list position (e.g. when the index was built with
        --embed_with_headings the stored IDs are offset by the number of
        heading-only chunks prepended to the list).

        Parameters
        ----------
        metadata : list of dicts produced by ``index_builder.build_index``

        Returns
        -------
        int – number of newly inserted rows
        """
        cur = self._con.cursor()
        cur.executemany(
            """
            INSERT OR IGNORE INTO chunk_metadata
                (chunk_id, filename, chapter, section, section_path,
                 page_numbers, char_len, word_len)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    list_idx,                                          # list-position index
                    m.get("filename", ""),
                    _extract_chapter(m.get("section_path", "")),
                    m.get("section", ""),
                    m.get("section_path", ""),
                    json.dumps(m.get("page_numbers", [])),
                    m.get("char_len", 0),
                    m.get("word_len", 0),
                )
                for list_idx, m in enumerate(metadata)
            ],
        )
        self._con.commit()
        return cur.rowcount

    # ------------------------------------------------------------------
    # Scoped-retrieval helpers
    # ------------------------------------------------------------------

    def get_all_chunk_ids(self) -> Set[int]:
        """Return the set of every indexed chunk ID."""
        cur = self._con.execute("SELECT chunk_id FROM chunk_metadata")
        return {row[0] for row in cur.fetchall()}

    def get_max_chapter(self) -> Optional[int]:
        """Return the highest chapter number present in the metadata store.

        Returns None when the store is empty or no chapter numbers were parsed.
        """
        cur = self._con.execute(
            "SELECT MAX(chapter) FROM chunk_metadata WHERE chapter IS NOT NULL"
        )
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else None

    def get_chunk_ids_by_source(self, source: str) -> Set[int]:
        """
        Return chunk IDs whose filename contains *source* (case-insensitive
        substring match).
        """
        cur = self._con.execute(
            "SELECT chunk_id FROM chunk_metadata WHERE LOWER(filename) LIKE ?",
            (f"%{source.lower()}%",),
        )
        return {row[0] for row in cur.fetchall()}

    def get_chunk_ids_by_chapter(self, chapter: int) -> Set[int]:
        """Return chunk IDs that belong to *chapter*."""
        cur = self._con.execute(
            "SELECT chunk_id FROM chunk_metadata WHERE chapter = ?",
            (chapter,),
        )
        return {row[0] for row in cur.fetchall()}

    def get_chunk_ids_by_chapters(self, chapters: List[int]) -> Set[int]:
        """Return chunk IDs that belong to any chapter in *chapters*.

        Accepts a list of chapter numbers (e.g. [1, 2, 3]) or a range
        produced by ``parse_chapter_arg``.  Uses a single SQL IN query.
        """
        if not chapters:
            return set()
        placeholders = ",".join("?" * len(chapters))
        cur = self._con.execute(
            f"SELECT chunk_id FROM chunk_metadata WHERE chapter IN ({placeholders})",
            chapters,
        )
        return {row[0] for row in cur.fetchall()}

    def get_chunk_ids_by_pages(self, from_page: int, to_page: int) -> Set[int]:
        """
        Return chunk IDs that touch any page in [from_page, to_page].

        ``page_numbers`` is stored as a JSON array; intersection is done in
        Python since SQLite cannot index into a JSON array without extensions.
        """
        cur = self._con.execute(
            "SELECT chunk_id, page_numbers FROM chunk_metadata"
        )
        page_range = set(range(from_page, to_page + 1))
        result: Set[int] = set()
        for chunk_id, pages_json in cur.fetchall():
            try:
                pages = set(json.loads(pages_json))
                if pages & page_range:
                    result.add(chunk_id)
            except (json.JSONDecodeError, TypeError):
                continue
        return result

    def build_valid_ids(
        self,
        source: Optional[str] = None,
        chapters: Optional[List[int]] = None,
        from_page: Optional[int] = None,
        to_page: Optional[int] = None,
    ) -> Optional[Set[int]]:
        """
        Combine scope constraints with set intersection.

        ``chapters`` accepts a list of chapter numbers (e.g. [1, 2, 3, 4, 5]).
        Pass a single-element list for a single chapter.

        Returns ``None`` when no filter is active (caller should skip
        filtering entirely), or a (possibly empty) set of chunk IDs.
        """
        active: Optional[Set[int]] = None

        def _intersect(new: Set[int]) -> Set[int]:
            return new if active is None else active & new

        if source is not None:
            active = _intersect(self.get_chunk_ids_by_source(source))
        if chapters is not None:
            active = _intersect(self.get_chunk_ids_by_chapters(chapters))
        if from_page is not None and to_page is not None:
            active = _intersect(self.get_chunk_ids_by_pages(from_page, to_page))

        return active  # None means "no filter applied"

    # ------------------------------------------------------------------
    # Provenance metadata lookup
    # ------------------------------------------------------------------

    def get_metadata_for_chunks(self, chunk_ids: List[int]) -> List[Dict]:
        """Return metadata rows for the given chunk IDs (order not guaranteed)."""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        cur = self._con.execute(
            f"SELECT chunk_id, filename, chapter, section, section_path, "
            f"page_numbers FROM chunk_metadata WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        rows = []
        for row in cur.fetchall():
            rows.append(
                {
                    "chunk_id": row[0],
                    "filename": row[1],
                    "chapter": row[2],
                    "section": row[3],
                    "section_path": row[4],
                    "page_numbers": json.loads(row[5]) if row[5] else [],
                }
            )
        return rows

    def close(self) -> None:
        self._con.close()


# ---------------------------------------------------------------------------
# ProvenanceStore
# ---------------------------------------------------------------------------

class ProvenanceStore:
    """
    Logs every query + retrieved chunk metadata + answer in SQLite so students
    can review their study history with the ``history`` chat command.

    Schema
    ------
    query_log(id PK, timestamp, query, retrieved_chunks JSON, answer,
              config_state JSON)
    """

    _DDL = """
        CREATE TABLE IF NOT EXISTS query_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT,
            query            TEXT,
            retrieved_chunks TEXT,
            answer           TEXT,
            config_state     TEXT
        );
    """

    def __init__(self, db_path: str = "index/provenance.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._con.executescript(self._DDL)
        self._con.commit()

    def log_query(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        answer: str,
        config_state: Optional[Dict] = None,
    ) -> None:
        """
        Persist one query-answer pair.

        Parameters
        ----------
        query             : the user's question
        retrieved_chunks  : list of dicts with keys chunk_id, section,
                            page_numbers, filename (subset of MetadataStore rows)
        answer            : full model response text
        config_state      : snapshot of RAGConfig (optional)
        """
        self._con.execute(
            """INSERT INTO query_log
               (timestamp, query, retrieved_chunks, answer, config_state)
               VALUES (?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                query,
                json.dumps(retrieved_chunks),
                answer,
                json.dumps(config_state or {}),
            ),
        )
        self._con.commit()

    def get_history(self, limit: int = 10) -> List[Dict]:
        """
        Return the *limit* most recent query-log entries, newest first.

        The ``answer`` field is truncated to 300 characters for display.
        """
        cur = self._con.execute(
            """SELECT id, timestamp, query, retrieved_chunks, answer
               FROM query_log
               ORDER BY id DESC
               LIMIT ?""",
            (limit,),
        )
        rows = []
        for row in cur.fetchall():
            answer_preview = row[4][:300] + "…" if len(row[4]) > 300 else row[4]
            try:
                chunks = json.loads(row[3])
            except (json.JSONDecodeError, TypeError):
                chunks = []
            rows.append(
                {
                    "id": row[0],
                    "timestamp": row[1],
                    "query": row[2],
                    "retrieved_chunks": chunks,
                    "answer": answer_preview,
                }
            )
        return rows

    def close(self) -> None:
        self._con.close()
