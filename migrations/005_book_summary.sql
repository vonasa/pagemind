-- Phase 7: book-level summary

-- Whole-book overview, generated offline (precompute stage / backfill) by reducing
-- over the per-chapter full summaries. Nullable until generated.
ALTER TABLE book_meta ADD COLUMN IF NOT EXISTS summary TEXT;
