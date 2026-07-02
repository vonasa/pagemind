-- Phase 2: precompute additions

-- Two-tier chapter summaries; existing summary column serves as full summary.
ALTER TABLE chapters ADD COLUMN IF NOT EXISTS micro_summary TEXT;

-- FTS: populate from sections content (idempotent batch run at precompute time)
