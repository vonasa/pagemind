-- Phase 6: precompute checkpoint ledger (per-unit auto-resume)

-- One row per completed unit of work; unit_key='*' marks a whole stage complete.
-- payload holds per-section raw extraction for the entities stage (else NULL).
CREATE TABLE IF NOT EXISTS precompute_checkpoints (
    book_id    UUID        NOT NULL REFERENCES book_meta(book_id) ON DELETE CASCADE,
    stage      TEXT        NOT NULL,
    unit_key   TEXT        NOT NULL,
    payload    JSONB,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (book_id, stage, unit_key)
);

CREATE INDEX IF NOT EXISTS precompute_checkpoints_stage_idx
    ON precompute_checkpoints (book_id, stage);
