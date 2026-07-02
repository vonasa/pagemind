# Pagemind

Pagemind turns a single book into a conversational knowledge base. Ask questions and get
concise, **citation-backed** answers — verbatim quotes, chapter summaries, character and
timeline lookups, and contextual "why did this happen" Q&A — with every claim traceable to
an exact passage in the source text.

It is a retrieval-augmented system built around one idea: **compile the book once, then keep
every query cheap.** Each book is "compiled" into a queryable substrate up front, so that at
conversation time each model call carries only a small, targeted slice of context. That token
discipline is what lets Pagemind run end-to-end on a modest self-hosted model — and the same
pipeline can be pointed at a commercial API purely through configuration.

---

## How it works

### Compilation — the offline pass

`just compile book.epub` transforms an EPUB into the structures the runtime queries. It is
idempotent and checkpointed: an interrupted compile resumes where it stopped instead of
restarting.

1. **Parse & structure.** The EPUB is reduced to plain text and split into chapters via a
   fallback ladder (TOC → spine items → heading regex → synthetic split), with front matter
   (cover, copyright, table of contents) flagged non-body.
2. **Segment.** Each chapter is split into *sections* — semantic units bounded by a token cap
   on paragraph/sentence boundaries — and overlapping *chunks* (~384 tokens, the embedding
   unit). A validation gate rejects pathological splits (oversized sections, length drift from
   the source).
3. **Precompute** — the model-driven pass, run stage by stage and recorded in a checkpoint
   ledger:
   - **Two-tier summaries** per chapter: a ~20-word micro-summary for routing plus a
     300–500-word full summary for orientation.
   - **Structured extraction**: per-section NER for characters and locations, dates, and
     events, followed by a global alias-clustering pass that resolves, e.g., "Eleanor" /
     "Miss Vance" to one canonical entity. Character offsets are validated against the source
     so every mention is anchorable back to the text.
   - **Embeddings**: each chunk is embedded into a 2048-dim vector and indexed (HNSW) for
     semantic search.
   - **Full-text index**: a Postgres `tsvector` index for lexical search.

Everything lands in Postgres — pgvector for vectors, `tsvector` for lexical search, and
relational tables for chapters, sections, entities, dates, and events. The compiled
artifacts, not the raw book, are what the runtime reads.

### Conversation — the query path

A question is routed to a recipe (chapter summary, fact lookup, verbatim quote,
contextual-why, entity location, …). Retrieval is **hybrid** — lexical (full-text) and
semantic (vector) candidates fused with Reciprocal Rank Fusion — and narrowed to a handful of
relevant sections. Each section is handed to a **context-quarantined reader** sub-call that
sees only that one passage and must answer from it; returned quotes are validated as exact
substrings of the stored text, so citations are grounded by construction. A final synthesizer
composes the answer from those distilled results.

Because the expensive reading happens once at compile time and each runtime call is given only
the slice it needs, tokens per query stay small — the property that keeps Pagemind practical
on a self-hosted model and economical on a commercial one. The compile-time and query-time
backends are configured independently (`INDEX_BACKEND` / `QUERY_BACKEND`).

---

## Requirements

- macOS, Apple Silicon (M1 or later)
- [Homebrew](https://brew.sh)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Postgres)
- [just](https://github.com/casey/just) — task runner
- [uv](https://docs.astral.sh/uv/) — Python package manager

Install the last two with Homebrew:

```bash
brew install just uv
```

---

## 1. Install omlx

omlx is an OpenAI-compatible LLM inference server for Apple Silicon.

```bash
brew tap jundot/omlx https://github.com/jundot/omlx
brew install omlx
```

---

## 2. Download a chat model

Pagemind uses any model served by omlx. Gemma 3 is the recommended default.

First install the HuggingFace CLI if you don't have it:

```bash
uv tool install "huggingface_hub[cli]"
```

Then download a model. Pick one based on your RAM:

**32 GB — Gemma 3 12B (4-bit, ~7 GB):**

```bash
huggingface-cli download mlx-community/gemma-3-12b-it-4bit \
  --local-dir ~/.omlx/models/gemma-3-12b-it
```

**64 GB — Gemma 3 27B (4-bit, ~14 GB):**

```bash
huggingface-cli download mlx-community/gemma-3-27b-it-4bit \
  --local-dir ~/.omlx/models/gemma-3-27b-it
```

The directory name (`gemma-3-12b-it` or `gemma-3-27b-it`) becomes the model ID
used in the next step.

---

## 3. Configure

Create a `.env` file at the project root:

```bash
# Chat backend — points at omlx (default port 8000)
LOCAL_BASE_URL=http://localhost:8000

# Must match the directory name you used in step 2
LOCAL_MODEL=gemma-3-12b-it
```

Everything else (`DATABASE_URL`, `EMBEDDING_URL`) has working defaults for the
local Docker setup and doesn't need to be set unless you change ports.

---

## 4. Start the services

**Start omlx** (keep this running in a terminal):

```bash
omlx serve --port 8000
```

**Start Postgres** (Docker) and apply migrations:

```bash
just up
```

This brings up **Postgres 17** with pgvector on port `5432` and runs the database
migrations.

**Start the embedding server** — [Infinity](https://github.com/michaelf34/infinity),
run **natively** on the host (not Docker):

```bash
just embed
```

This serves `infgrad/Jasper-Token-Compression-600M` on port `7997`. The first run
creates a dedicated Python venv (via `uv`) and downloads the model (~2.2 GB), so
expect a few minutes; it then runs in the background and the command returns once
the server is healthy. Logs go to `.infinity/run.log`.

> Infinity runs natively rather than in Docker: on Apple Silicon the published
> image runs under QEMU emulation and needs library versions the upstream image
> doesn't ship, so the native path is both faster and more reliable. `just embed`
> handles the venv and pinned dependencies for you.

---

## 5. Install Python dependencies

```bash
uv sync
```

---

## 6. Add a book

Download any EPUB from [Project Gutenberg](https://www.gutenberg.org) and compile it:

```bash
just compile path/to/book.epub
```

This runs the full compilation described above — parse, segment, validate, and the
model-driven precompute pass (summaries, entity/date/event extraction, embeddings,
full-text index) — with per-stage progress. The `book_meta.status` column moves
`ingesting` → `indexing` → `ready`. Re-running is safe: a completed book is skipped,
and an interrupted one **resumes from its last checkpoint** rather than recompiling.
Pass `--reindex` to force a clean rebuild from scratch.

---

## 7. Run the tests

```bash
just test
```

The smoke tests (model round-trips) skip automatically when omlx or Infinity
isn't reachable. The ingestion/segmentation tests always run and don't require
any running services.

---

## Stopping everything

```bash
omlx stop          # or Ctrl-C in the omlx terminal
just down          # stop the embedding server + Postgres (data volumes are preserved)
```

---

## Configuration reference

All settings are in [`pagemind/config.py`](pagemind/config.py) and can be
overridden in `.env`:

| Variable | Default | Description |
|---|---|---|
| `LOCAL_BASE_URL` | `http://localhost:11434` | omlx (or Ollama) base URL |
| `LOCAL_MODEL` | `gemma3` | model ID served by omlx |
| `EMBEDDING_URL` | `http://localhost:7997` | Infinity embedding server |
| `DATABASE_URL` | `postgresql://pagemind:pagemind@localhost:5432/pagemind` | Postgres connection |
| `INDEX_BACKEND` | `local` | backend for precompute (`local` \| `commercial` \| `anthropic`) |
| `QUERY_BACKEND` | `local` | backend for runtime queries |
