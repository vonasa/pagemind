-- Phase 1: ingestion/segmentation additions

-- Detect duplicate ingestion attempts
ALTER TABLE book_meta ADD COLUMN IF NOT EXISTS source_hash TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS book_meta_source_hash_idx ON book_meta (source_hash);

-- Non-body tagging at chapter level (front/back matter)
ALTER TABLE chapters ADD COLUMN IF NOT EXISTS is_body BOOLEAN NOT NULL DEFAULT TRUE;
