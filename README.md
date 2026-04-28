# TokenSmith

**TokenSmith** is a local-first RAG system for students to query textbooks, lecture slides, and notes and get fast, cited answers on their own machines using local LLMs. It combines FAISS dense retrieval with BM25 sparse scoring under an ensemble ranker, and applies database-inspired principles like indexing, caching, and incremental builds to optimize the ingestion → retrieval → generation pipeline.

<img width="1255" height="843" alt="tokensmith" src="https://github.com/user-attachments/assets/b36d6227-8cec-4f71-aacc-fccdd1285378" />

## Features

- Parse and index PDF documents into searchable chunks
- Hybrid retrieval: FAISS dense search + BM25 sparse scoring + ensemble ranking
- Local inference via `llama.cpp` (GGUF models) — no data leaves your machine
- **Scoped retrieval** — restrict answers to specific chapters, pages, or source files
- **Auto-scope detection** — automatically detects chapter/page references in your query and applies scope without extra flags
- **Source citations** — every answer shows which sections and pages were used
- **Session history** — replay past queries and their cited sources with the `history` command
- **Trust score** — warns when retrieved chunks disagree topically, signalling a low-confidence answer
- **Embedding cache** — repeated queries are served from disk without re-encoding
- Configurable chunking, Metal/CUDA/CPU acceleration, optional table preservation

---

## Requirements

- Python 3.9+
- Conda / Miniconda
- macOS: Xcode Command Line Tools — Linux: GCC, make, CMake — Windows: Visual Studio Build Tools

---

## Quick Start

### 1. Clone and download models

```shell
git clone https://github.com/georgia-tech-db/TokenSmith.git
cd TokenSmith
mkdir models && cd models
```

Create the model directories and put in the appropriate models in them.
```shell
mkdir -p models/generators models/embedders
```

```yaml
embed_model: "models/embedders/Qwen3-Embedding-4B-Q5_K_M.gguf"
gen_model:   "models/generators/qwen2.5-3b-instruct-q8_0.gguf"
```
Download the appropriate files and put them in the `models/embedders/` and `models/generators/` folders:
- https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF/tree/main
- https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/tree/main

- Embedding model: [Qwen3-Embedding-4B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF/tree/main)
- Generation model: [Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/tree/main)

### 2. Build

```shell
make build
```

Creates the `tokensmith` conda environment, installs dependencies, and builds `llama.cpp`.

> **Troubleshooting — NumPy conflict:** If you see `NumPy 1.x cannot be run in NumPy 2.x` errors:
> ```shell
> conda activate tokensmith
> conda uninstall faiss-cpu -y
> conda install -c conda-forge faiss-cpu
> ```

### 3. Activate the environment

```shell
conda activate tokensmith
```

### 4. Add your documents

```shell
mkdir -p data/chapters
cp your-textbook.pdf data/chapters/
```

### 5. Extract PDF to markdown

```shell
make run-extract
```

### 6. Index documents

```shell
make run-index
```

With options:

```shell
make run-index ARGS="--chunk_mode chars --visualize"
```

If you want to index a portion of the textbook:

```shell
make run-index-partial CHAPTERS="1 2"
```

If you want to add chapters to the index later:

```shell
make run-add-chapters-partial CHAPTERS="3"
```

### 7. Start chatting

```shell
python -m src.main chat
```

Note: if you only indexed a portion of your documents, use

```shell
python -m src.main chat --partial
```

or

```shell
make run-chat-partial
```

### 8. Deactivate

```shell
conda deactivate
```

---

## Scoped Retrieval

Scoped retrieval limits search to a subset of your document, so answers stay relevant to the material you're studying.

### Scope by chapter

```shell
# Single chapter
python -m src.main chat --scope_chapter 18

# Multiple chapters (space-separated)
python -m src.main chat --scope_chapter 14 15 18

# Range
python -m src.main chat --scope_chapter 14-18

# Range with extras
python -m src.main chat --scope_chapter 14-18 20
```

### Scope by page range

```shell
python -m src.main chat --scope_pages 978 1050
```

### Scope by source file

```shell
python -m src.main chat --scope_source "chapter14"
```

### Combine scopes

All three flags can be combined — the system intersects them so only chunks matching **all** specified criteria are eligible:

```shell
python -m src.main chat --scope_chapter 14-18 --scope_source "silberschatz"
```

### Filter strategy

Two strategies are available (default: `post`):

```shell
# pre-filter: removes out-of-scope candidates before ranking (faster dict passed to ranker)
python -m src.main chat --scope_chapter 18 --scope_strategy pre

# post-filter: ranks all candidates then removes out-of-scope results
python -m src.main chat --scope_chapter 18 --scope_strategy post
```

Both strategies produce identical results on the current `IndexFlatL2` index.

---

## Auto-Scope Detection

TokenSmith automatically detects chapter and page references in your query and applies scope without extra CLI flags. Just ask naturally:

```
Ask > I have a midterm on chapters 14 to 18. What is a clustered index?
# Auto-scope detected: chapters=[14, 15, 16, 17, 18] | 323 eligible chunks

Ask > Explain the ARIES redo pass in chapter 19
# Auto-scope detected: chapters=[19] | 95 eligible chunks

Ask > Walk me through the first half of the book
# Auto-scope detected: chapters=[1..13] | 862 eligible chunks
```

**Supported patterns:**

| Pattern | Example |
|---|---|
| Explicit chapter/section | `chapter 18`, `ch. fourteen`, `section 18.3` |
| Range | `chapters 14–18`, `chapter 1 to 3` |
| Comma list | `chapters 14, 15, and 18` |
| Count-based | `first 5 chapters`, `last 3 chapters` |
| Boundary phrase | `up to chapter 10`, `from chapter 14 onwards` |
| Relative fraction | `first half`, `last quarter`, `middle third` |

If a query-level scope is detected, it is **intersected** with any session-level `--scope_chapter` flag, so both sources narrow the candidate pool together. If the intersection is empty (e.g., a chapter not in the index), the system warns and falls back to the session scope.

Auto-scope operates on your **original** query text, not the rewritten form, so explicit chapter cues are never stripped before detection.

---

## Source Citations

After every answer, TokenSmith prints the sections and page numbers that were used:

```
Sources:
  [1] Chapter 18 Section 18.1.3 The Two-Phase Locking Protocol  (pp. 1295–1296)
  [2] Chapter 18 Section 18.1.3 The Two-Phase Locking Protocol  (pp. 1296–1297)
  [3] Chapter 18 Section 18.1.1 Locks  (pp. 1275–1288)
```

Citations are formatted from the SQLite metadata store (`index/metadata.db`) and reflect the actual chunks used for the answer.

---

## Session History

Type `history` at the prompt to replay your ten most recent queries with their timestamps, cited sources, and answer snippets:

```
Ask > history

=== Query History (newest first) ===

#12 [2026-04-27T19:32:28]
  Q: What is the difference between a clustered and unclustered index?
  Sources:
    • Chapter 14 Section 14.2.4 Secondary Indices (pp. 969–970)
    • Chapter 14 Section 14.2.5 Indices on Multiple Keys (pp. 971–971)
  A: A clustered index organizes data in the table in the order of the index key...
```

History persists across sessions in `index/provenance.db`.

---

## Trust Score

After each answer, TokenSmith computes the mean pairwise cosine similarity among the top-k retrieved chunks. If the score falls below the configured threshold, a warning is printed before you act on the answer:

```
[Low confidence] (chunk agreement: 0.21).
Retrieved sources cover different topics — verify this answer in the source material.
```

A high score (e.g., 0.79) means the retrieved chunks are topically coherent and the answer is well-supported. A low score means the retriever pulled from scattered topics — useful signal to check the textbook directly.

Configure the threshold in `config/config.yaml`:

```yaml
trust_score_threshold: 0.30   # default; lower = fewer warnings
```

---

## Configuration

`config/config.yaml` controls all runtime behaviour. Key options:

```yaml
# Models
embed_model: "models/Qwen3-Embedding-4B-Q5_K_M.gguf"
gen_model:   "models/qwen2.5-3b-instruct-q8_0.gguf"

# Retrieval
top_k: 10                  # chunks returned per query
num_candidates: 50         # candidate pool size before ranking
ensemble_method: "rrf"     # reciprocal rank fusion
ranker_weights: {"faiss": 0.6, "bm25": 0.4, "index_keywords": 0}

# Generation
max_gen_tokens: 400

# Chunking
chunk_mode: "recursive_sections"
chunk_size: 2000
chunk_overlap: 200

# Trust score
trust_score_threshold: 0.30   # warn when chunk agreement falls below this

# History
enable_history: true          # log queries and sources to provenance.db
max_history_turns: 3          # conversation context turns kept in memory
```

Config priority (highest → lowest):
1. `--config` CLI argument
2. `~/.config/tokensmith/config.yaml`
3. `config/config.yaml`

---

## Running Tests

```shell
conda run -n tokensmith pytest tests/
```

The test suite has **135 tests** across four files:

| File | Tests | What it covers |
|---|---|---|
| `test_metadata_store.py` | 47 | SQLite metadata/provenance store, filter axes, citation formatting, chapter-arg parsing |
| `test_e2e_scoped_retrieval.py` | 25 | Full pipeline with real FAISS/BM25 artifacts; scoped precision, keyword coverage, boundary cases |
| `test_auto_scope.py` | 50 | All regex pattern families, word/digit forms, ranges, fractions, false-positive guards |
| `test_trust_score.py` | 13 | Deterministic mock vectors; identical/orthogonal cases, fallback path, edge inputs |

---

## Development

```shell
make help        # list all targets
make build       # create env + build llama.cpp
make run-extract # extract PDFs to markdown
make run-index   # build FAISS/BM25 index
make test        # run test suite
make update-env  # sync conda env from environment.yml
make clean       # remove build artifacts
```
