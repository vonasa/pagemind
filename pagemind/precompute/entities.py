"""Per-section NER, global alias clustering, dates/events, offset-validation prune.

Resume model: per-section extraction is the bulk of the LLM cost, so each section's
raw extraction is stored durably in the checkpoint ledger (`payload`) and committed
per section — a resumed run skips sections that already have a payload. The back half
(global clustering + occurrence/date/event materialisation) cannot decompose per-unit
because clustering needs every section's names at once and is a fresh non-deterministic
LLM call; it therefore clears and rebuilds all derived entity data atomically.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any

import psycopg

from pagemind.models.chat import ChatClient
from pagemind.precompute.checkpoint import load_payloads, mark_stage_done, mark_unit

_STAGE = "entities"

# ── Prompts ───────────────────────────────────────────────────────────────────

_SECTION_SYS = (
    "You are a precise literary information extractor. "
    "Respond only with valid JSON matching the specified schema — no extra text."
)

_SECTION_PROMPT = """\
Extract structured information from the following passage.

Return a single JSON object with this exact shape:
{{
  "entities": [
    {{"name": "<exact substring>", "type": "character" | "location", "offset_start": <int>, "offset_end": <int>}}
  ],
  "dates": [
    {{"raw_text": "<exact substring>", "normalized": "<YYYY-MM-DD or null>", "offset_start": <int>, "offset_end": <int>}}
  ],
  "events": [
    {{"description": "<1-sentence summary>", "anchor_offset": <int>}}
  ]
}}

Rules:
- "name" and "raw_text" MUST be exact substrings of the passage (used for validation).
- Offsets are 0-indexed character positions within this passage.
- For dates: "normalized" is ISO format only for real-world dates, otherwise null.
- Extract at most 3 significant events; omit trivial ones.
- If nothing to extract in a category, return an empty list.

PASSAGE ({length} chars):
{content}"""

_CLUSTER_SYS = (
    "You are a literary alias resolver. "
    "Respond only with valid JSON matching the specified schema — no extra text."
)

_CLUSTER_PROMPT = """\
The following names were extracted from a single book. Cluster names that refer to the same character or place.

Return a single JSON object:
{{
  "entities": [
    {{"canonical": "<primary name>", "type": "character" | "location", "aliases": ["<alt name>", ...]}}
  ]
}}

Rules:
- Every input name must appear exactly once — either as a canonical or an alias.
- "canonical" should be the most complete name used in the text.
- Keep "character" and "location" separate; never merge cross-type.
- If a name has no aliases, use "aliases": [].

NAMES ({count} total):
{names_json}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    # Strip markdown code fences if the model wraps its output
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner)
    return json.loads(text)


def _offset_valid(content: str, name: str, start: int, end: int) -> bool:
    """True iff content[start:end] is a plausible match for name."""
    if not isinstance(start, int) or not isinstance(end, int):
        return False
    if start < 0 or end > len(content) or start >= end:
        return False
    extracted = content[start:end]
    nl = name.lower()
    el = extracted.lower()
    return nl in el or el in nl


def _context_snippet(content: str, start: int, end: int, window: int = 40) -> str:
    return content[max(0, start - window) : end + window]


# ── Main ──────────────────────────────────────────────────────────────────────

async def extract_entities(
    conn: psycopg.Connection,
    book_id: uuid.UUID,
    *,
    progress: Callable[[str], None] | None = None,
) -> None:
    client = ChatClient.from_config(axis="index")

    sections = conn.execute(
        """
        SELECT section_id, chapter_id, content, char_offset_start
        FROM sections
        WHERE book_id = %s AND is_body
        ORDER BY ordinal
        """,
        (book_id,),
    ).fetchall()

    # ── Front half: per-section extraction (durable, committed per section) ──
    # payloads keyed by section_id::text; reused across resume to skip LLM calls.
    payloads = load_payloads(conn, book_id, _STAGE)
    total = len(sections)
    base = sum(1 for sid, *_ in sections if str(sid) in payloads)
    i = base
    for section_id, _chapter_id, content, _global_offset in sections:
        if str(section_id) in payloads:
            continue
        i += 1
        if progress:
            progress(f"Extracting section {i}/{total} …")
        prompt = _SECTION_PROMPT.format(length=len(content), content=content)
        response = await client.complete(
            [
                {"role": "system", "content": _SECTION_SYS},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1500,
        )
        try:
            data = _parse_json_response(response)
        except (json.JSONDecodeError, ValueError, KeyError):
            data = {}
        parsed = {
            "entities": data.get("entities", []),
            "dates": data.get("dates", []),
            "events": data.get("events", []),
        }
        payloads[str(section_id)] = parsed
        mark_unit(conn, book_id, _STAGE, str(section_id), payload=parsed)
        conn.commit()

    # ── Back half: cluster + materialise, atomically rebuilt from payloads ───
    # Reconstruct the per-section raw dict (content/chapter/offset from the DB row,
    # extraction arrays from the durable payloads).
    raw: dict[uuid.UUID, dict] = {}
    for section_id, chapter_id, content, global_offset in sections:
        p = payloads.get(str(section_id), {})
        raw[section_id] = {
            "chapter_id": chapter_id,
            "content": content,
            "global_offset": global_offset or 0,
            "entities": p.get("entities", []),
            "dates": p.get("dates", []),
            "events": p.get("events", []),
        }

    # ── Step 2: collect all candidate names for clustering ───────────────────
    all_names: dict[str, str] = {}  # name → type
    for info in raw.values():
        for ent in info["entities"]:
            name = str(ent.get("name", "")).strip()
            etype = str(ent.get("type", "")).strip()
            if name and etype in ("character", "location"):
                # Later occurrence wins if type conflicts (rare)
                all_names[name] = etype

    # ── Step 3: global alias clustering ─────────────────────────────────────
    # canonical_map: lower(alias_or_name) → (canonical, type, aliases[])
    canonical_map: dict[str, tuple[str, str, list[str]]] = {}

    if all_names:
        if progress:
            progress(f"Clustering {len(all_names)} entities …")
        names_payload = [{"name": n, "type": t} for n, t in all_names.items()]
        cluster_resp = await client.complete(
            [
                {"role": "system", "content": _CLUSTER_SYS},
                {
                    "role": "user",
                    "content": _CLUSTER_PROMPT.format(
                        count=len(names_payload),
                        names_json=json.dumps(names_payload, ensure_ascii=False),
                    ),
                },
            ],
            max_tokens=3000,
        )
        try:
            cluster_data = _parse_json_response(cluster_resp)
        except (json.JSONDecodeError, ValueError, KeyError):
            cluster_data = {}

        for entry in cluster_data.get("entities", []):
            canonical = str(entry.get("canonical", "")).strip()
            etype = str(entry.get("type", "")).strip()
            aliases = [str(a).strip() for a in entry.get("aliases", []) if str(a).strip()]
            if not canonical or etype not in ("character", "location"):
                continue
            info_tuple = (canonical, etype, aliases)
            canonical_map[canonical.lower()] = info_tuple
            for alias in aliases:
                canonical_map[alias.lower()] = info_tuple

    # ── Steps 4–5: rebuild all derived entity data atomically ───────────────
    # Clustering is a fresh, non-deterministic LLM call, so a resumed back half can
    # produce different canonical names. To avoid orphaned entity rows we clear and
    # rebuild everything for this book in one transaction. occurrences/dates/events
    # have no usable conflict key (events.entity_ids is a bare UUID[]), so they are
    # deleted explicitly; book-scoped consistency, not FK ordering, is the point.
    with conn.transaction():
        conn.execute("DELETE FROM occurrences WHERE book_id = %s", (book_id,))
        conn.execute("DELETE FROM events WHERE book_id = %s", (book_id,))
        conn.execute("DELETE FROM dates WHERE book_id = %s", (book_id,))
        conn.execute("DELETE FROM entities WHERE book_id = %s", (book_id,))

        # Step 4: upsert canonical entities → (canonical, db_type) → entity_id
        entity_ids: dict[tuple[str, str], uuid.UUID] = {}
        seen: set[tuple[str, str]] = set()

        for _key, (canonical, etype, aliases) in canonical_map.items():
            db_type = "PERSON" if etype == "character" else "PLACE"
            key = (canonical, db_type)
            if key in seen:
                continue
            seen.add(key)
            row = conn.execute(
                """
                INSERT INTO entities (book_id, name, entity_type, aliases)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (book_id, name, entity_type) DO UPDATE
                  SET aliases = EXCLUDED.aliases
                RETURNING entity_id
                """,
                (book_id, canonical, db_type, aliases or None),
            ).fetchone()
            entity_ids[key] = row[0]

        def _resolve(name: str) -> uuid.UUID | None:
            entry = canonical_map.get(name.lower())
            if not entry:
                return None
            canonical, etype, _ = entry
            db_type = "PERSON" if etype == "character" else "PLACE"
            return entity_ids.get((canonical, db_type))

        # Step 5: store occurrences / dates / events with offset pruning
        for section_id, info in raw.items():
            chapter_id = info["chapter_id"]
            content = info["content"]
            global_off = info["global_offset"]

            for ent in info["entities"]:
                name = str(ent.get("name", "")).strip()
                start = ent.get("offset_start")
                end = ent.get("offset_end")
                if not name or start is None or end is None:
                    continue
                if not _offset_valid(content, name, int(start), int(end)):
                    continue
                entity_id = _resolve(name)
                if entity_id is None:
                    continue
                conn.execute(
                    """
                    INSERT INTO occurrences
                      (book_id, entity_id, chapter_id, section_id,
                       char_offset_start, char_offset_end, context)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        book_id, entity_id, chapter_id, section_id,
                        global_off + int(start),
                        global_off + int(end),
                        _context_snippet(content, int(start), int(end)),
                    ),
                )

            for d in info["dates"]:
                raw_text = str(d.get("raw_text", "")).strip()
                start = d.get("offset_start")
                end = d.get("offset_end")
                if not raw_text or start is None or end is None:
                    continue
                if not _offset_valid(content, raw_text, int(start), int(end)):
                    continue
                normalized = d.get("normalized") or None
                conn.execute(
                    """
                    INSERT INTO dates
                      (book_id, chapter_id, section_id, raw_text, normalized_date,
                       char_offset_start, char_offset_end, context)
                    VALUES (%s, %s, %s, %s, %s::date, %s, %s, %s)
                    """,
                    (
                        book_id, chapter_id, section_id, raw_text,
                        normalized,
                        global_off + int(start),
                        global_off + int(end),
                        _context_snippet(content, int(start), int(end)),
                    ),
                )

            for ev in info["events"]:
                description = str(ev.get("description", "")).strip()
                anchor = ev.get("anchor_offset")
                if not description or anchor is None:
                    continue
                anchor = int(anchor)
                if not (0 <= anchor < len(content)):
                    continue
                # Best-effort entity resolution: find canonical names in description
                ev_entity_ids = [
                    eid
                    for (canonical, _), eid in entity_ids.items()
                    if canonical.lower() in description.lower()
                ]
                conn.execute(
                    """
                    INSERT INTO events
                      (book_id, chapter_id, section_id, description, entity_ids,
                       char_offset_start, char_offset_end)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        book_id, chapter_id, section_id, description,
                        ev_entity_ids or None,
                        global_off + anchor,
                        global_off + anchor,
                    ),
                )

        mark_stage_done(conn, book_id, _STAGE)
    # transaction commits on block exit
