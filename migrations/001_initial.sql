-- Enable pgvector (halfvec requires pgvector >= 0.7)
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Book metadata ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS book_meta (
    book_id       UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    title         TEXT        NOT NULL,
    author        TEXT,
    -- pending | indexing | ready | error
    status        TEXT        NOT NULL DEFAULT 'pending',
    embed_model   TEXT,
    embed_dim     INTEGER,
    embed_version INTEGER     NOT NULL DEFAULT 1,
    cover_mime    TEXT,
    cover_data    BYTEA,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Chapters ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chapters (
    chapter_id UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    book_id    UUID        NOT NULL REFERENCES book_meta(book_id) ON DELETE CASCADE,
    ordinal    INTEGER     NOT NULL,
    title      TEXT,
    summary    TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chapters_book_id_idx ON chapters (book_id);

-- ── Sections (logical paragraph/block units within chapters) ───────────────────
CREATE TABLE IF NOT EXISTS sections (
    section_id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    book_id            UUID        NOT NULL REFERENCES book_meta(book_id) ON DELETE CASCADE,
    chapter_id         UUID        NOT NULL REFERENCES chapters(chapter_id) ON DELETE CASCADE,
    ordinal            INTEGER     NOT NULL,
    content            TEXT        NOT NULL,
    char_offset_start  INTEGER,
    char_offset_end    INTEGER,
    is_body            BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS sections_book_id_idx     ON sections (book_id);
CREATE INDEX IF NOT EXISTS sections_chapter_id_idx  ON sections (chapter_id);

-- ── Chunks (embedding units; halfvec stores 2048-d at fp16) ───────────────────
-- halfvec is used because Jasper-Token-Compression-600M has native dim 2048,
-- which exceeds pgvector's HNSW limit of 2000 for the standard vector type.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id          UUID           PRIMARY KEY DEFAULT gen_random_uuid(),
    book_id           UUID           NOT NULL REFERENCES book_meta(book_id) ON DELETE CASCADE,
    chapter_id        UUID           REFERENCES chapters(chapter_id) ON DELETE CASCADE,
    section_id        UUID           REFERENCES sections(section_id) ON DELETE CASCADE,
    ordinal           INTEGER        NOT NULL,
    content           TEXT           NOT NULL,
    char_offset_start INTEGER,
    char_offset_end   INTEGER,
    is_body           BOOLEAN        NOT NULL DEFAULT TRUE,
    embedding         halfvec(2048),
    created_at        TIMESTAMPTZ    NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_book_id_idx ON chunks (book_id);

-- HNSW cosine index; halfvec supports up to 4000 dims.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ── Entities ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS entities (
    entity_id   UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    book_id     UUID        NOT NULL REFERENCES book_meta(book_id) ON DELETE CASCADE,
    name        TEXT        NOT NULL,
    -- PERSON | PLACE | ORG | CONCEPT | EVENT | OTHER
    entity_type TEXT        NOT NULL,
    aliases     TEXT[],
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (book_id, name, entity_type)
);

CREATE INDEX IF NOT EXISTS entities_book_id_idx ON entities (book_id);

-- ── Occurrences (entity mentions anchored to sections) ────────────────────────
CREATE TABLE IF NOT EXISTS occurrences (
    occurrence_id     UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    book_id           UUID        NOT NULL REFERENCES book_meta(book_id) ON DELETE CASCADE,
    entity_id         UUID        NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    chapter_id        UUID        REFERENCES chapters(chapter_id) ON DELETE CASCADE,
    section_id        UUID        REFERENCES sections(section_id) ON DELETE CASCADE,
    char_offset_start INTEGER,
    char_offset_end   INTEGER,
    context           TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS occurrences_entity_id_idx ON occurrences (entity_id);
CREATE INDEX IF NOT EXISTS occurrences_book_id_idx   ON occurrences (book_id);

-- ── Events ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS events (
    event_id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    book_id           UUID        NOT NULL REFERENCES book_meta(book_id) ON DELETE CASCADE,
    chapter_id        UUID        REFERENCES chapters(chapter_id) ON DELETE CASCADE,
    section_id        UUID        REFERENCES sections(section_id) ON DELETE CASCADE,
    description       TEXT        NOT NULL,
    event_date        TEXT,
    entity_ids        UUID[],
    char_offset_start INTEGER,
    char_offset_end   INTEGER,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS events_book_id_idx ON events (book_id);

-- ── Dates (extracted temporal references) ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS dates (
    date_id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    book_id           UUID        NOT NULL REFERENCES book_meta(book_id) ON DELETE CASCADE,
    chapter_id        UUID        REFERENCES chapters(chapter_id) ON DELETE CASCADE,
    section_id        UUID        REFERENCES sections(section_id) ON DELETE CASCADE,
    raw_text          TEXT        NOT NULL,
    normalized_date   DATE,
    context           TEXT,
    char_offset_start INTEGER,
    char_offset_end   INTEGER,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS dates_book_id_idx ON dates (book_id);

-- ── Full-text search index (denormalized tsvectors) ───────────────────────────
CREATE TABLE IF NOT EXISTS sections_fts (
    section_id UUID      PRIMARY KEY REFERENCES sections(section_id) ON DELETE CASCADE,
    book_id    UUID      NOT NULL REFERENCES book_meta(book_id) ON DELETE CASCADE,
    chapter_id UUID      NOT NULL REFERENCES chapters(chapter_id) ON DELETE CASCADE,
    fts_vector TSVECTOR
);

CREATE INDEX IF NOT EXISTS sections_fts_idx ON sections_fts USING GIN (fts_vector);
