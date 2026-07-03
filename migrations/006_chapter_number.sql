-- Phase 8: human-facing chapter number

-- `ordinal` is a raw 0-based detection index over ALL units (incl. non-body front/back
-- matter), so the body chapters shown in the UI have gaps ("Chapter 4", "Chapter 7").
-- `number` is the 1-based position among BODY chapters only — the number users see and
-- reference in chat. NULL for non-body chapters. Backfill is pure SQL (no re-compile).
ALTER TABLE chapters ADD COLUMN IF NOT EXISTS number INTEGER;

WITH numbered AS (
    SELECT chapter_id,
           ROW_NUMBER() OVER (PARTITION BY book_id ORDER BY ordinal) AS n
    FROM chapters
    WHERE is_body
)
UPDATE chapters c
SET number = numbered.n
FROM numbered
WHERE c.chapter_id = numbered.chapter_id;

CREATE INDEX IF NOT EXISTS chapters_book_number_idx ON chapters (book_id, number);
