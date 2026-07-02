# Pagemind — architecture

A book is **compiled once** (offline) into a queryable substrate in Postgres, then
each **query** reads only a small, targeted slice of that substrate. Compile-time and
query-time model backends are configured independently.

## Entrypoints & shared

- `pagemind/cli.py` — CLI (`pagemind add <book>`, etc.); the compile driver.
- `pagemind/config.py` — env-driven settings (backends, URLs, models).
- `pagemind/db.py` — Postgres (pgvector) connection helpers.
- `migrations/` (SQL) + `pagemind/migrations/` (runner) — schema.

## Compile pipeline (offline)

- `pagemind/ingest/` — parse EPUB into plain text and chapters.
- `pagemind/segment/` — split chapters into sections and embedding-sized chunks; token counting and split validation.
- `pagemind/precompute/` — model-driven pass: two-tier chapter summaries, NER + alias clustering, chunk embeddings; checkpointed so an interrupted compile resumes.
- `pagemind/models/` — backend adapters for LLM chat and embeddings (local / commercial / Anthropic).

## Query path (runtime)

- `pagemind/retrieval/` — lexical (FTS) and semantic (vector) search fused with RRF, plus expand and structured (entity/date/event) lookups.
- `pagemind/recipes/` — one handler per query type (chapter summary, fact lookup, verbatim quote, contextual-why, locate-entity, structured view, generic fallback).
- `pagemind/runtime/` — query orchestration: router picks a recipe, context-quarantined reader sub-calls answer from single passages, synthesizer composes the final answer; also quote validation, history, and SSE streaming.

## Web

- `pagemind/api/` — FastAPI app exposing the runtime over HTTP with SSE streaming.
- `web/` — React/Vite frontend.

## Infra

- `docker/` — Postgres (pgvector) and the Infinity embedding server.
- `scripts/` — native Infinity server launcher and test-fixture generation.

@CLAUDE.local.md
