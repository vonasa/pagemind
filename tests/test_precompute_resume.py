"""Tests for the precompute checkpoint ledger and per-stage auto-resume.

DB tests skip automatically when Postgres (with migration 004 applied) is not
reachable, mirroring tests/test_retrieval.py. Model clients are mocked, so these
tests never touch oMLX or Infinity.
"""
from __future__ import annotations

import uuid

import psycopg
import pytest

from pagemind.config import settings
from pagemind.precompute import checkpoint as cp


# ── DB connection / skip guard ────────────────────────────────────────────────

def _try_connect() -> psycopg.Connection | None:
    try:
        conn = psycopg.connect(settings.database_url, connect_timeout=3)
    except Exception:
        return None
    # Skip if migration 004 hasn't been applied.
    try:
        conn.execute("SELECT 1 FROM precompute_checkpoints LIMIT 0")
    except Exception:
        conn.close()
        return None
    return conn


def _vec_to_pg(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.8g}" for v in vec) + "]"


@pytest.fixture
def conn():
    c = _try_connect()
    if c is None:
        pytest.skip("Postgres with migration 004 not reachable")
    c._created_books = []  # populated by _make_book
    yield c
    # Teardown: remove committed fixture books (cascades to ledger + derived rows).
    try:
        c.rollback()
        for bid in c._created_books:
            c.execute("DELETE FROM book_meta WHERE book_id = %s", (bid,))
        c.commit()
    except Exception:
        c.rollback()
    finally:
        c.close()


def _make_book(conn, *, n_chunks_per_section: int = 1) -> dict:
    """Insert a minimal 2-chapter / 2-section book. Returns ids; caller commits."""
    book_id = uuid.uuid4()
    conn._created_books.append(book_id)
    conn.execute(
        """
        INSERT INTO book_meta (book_id, title, status)
        VALUES (%s, 'Resume Fixture', 'indexing')
        """,
        (book_id,),
    )
    ids = {"book_id": book_id, "chapters": [], "sections": [], "chunks": []}
    # Section content embeds a known entity ("Alice") at offset 0 for the entities test.
    contents = [
        "Alice walked to London in the morning light and did not look back once.",
        "Alice returned from London at dusk, weary but certain of her purpose now.",
    ]
    for ordinal in range(2):
        ch = conn.execute(
            """
            INSERT INTO chapters (book_id, ordinal, title, is_body)
            VALUES (%s, %s, %s, TRUE) RETURNING chapter_id
            """,
            (book_id, ordinal, f"Chapter {ordinal}"),
        ).fetchone()[0]
        ids["chapters"].append(ch)
        content = contents[ordinal]
        sec = conn.execute(
            """
            INSERT INTO sections (book_id, chapter_id, ordinal, content,
                                  char_offset_start, char_offset_end, is_body)
            VALUES (%s, %s, 0, %s, 0, %s, TRUE) RETURNING section_id
            """,
            (book_id, ch, content, len(content)),
        ).fetchone()[0]
        ids["sections"].append(sec)
        for k in range(n_chunks_per_section):
            chunk = conn.execute(
                """
                INSERT INTO chunks (book_id, chapter_id, section_id, ordinal, content,
                                    char_offset_start, char_offset_end, is_body)
                VALUES (%s, %s, %s, %s, %s, 0, %s, TRUE) RETURNING chunk_id
                """,
                (book_id, ch, sec, k, f"chunk {ordinal}-{k}", 10),
            ).fetchone()[0]
            ids["chunks"].append(chunk)
    conn.commit()
    return ids


# ── Checkpoint ledger round-trip ──────────────────────────────────────────────

def test_checkpoint_roundtrip(conn):
    ids = _make_book(conn)
    book_id = ids["book_id"]

    assert cp.stage_done(conn, book_id, "summaries") is False
    assert cp.done_units(conn, book_id, "summaries") == set()

    cp.mark_unit(conn, book_id, "summaries", "u1")
    cp.mark_unit(conn, book_id, "entities", "s1", payload={"entities": [{"name": "Alice"}]})
    conn.commit()

    assert cp.done_units(conn, book_id, "summaries") == {"u1"}
    payloads = cp.load_payloads(conn, book_id, "entities")
    assert payloads == {"s1": {"entities": [{"name": "Alice"}]}}  # Jsonb round-trip

    # '*' sentinel is excluded from done_units/load_payloads.
    cp.mark_stage_done(conn, book_id, "summaries")
    conn.commit()
    assert cp.stage_done(conn, book_id, "summaries") is True
    assert cp.done_units(conn, book_id, "summaries") == {"u1"}

    cp.clear(conn, book_id)
    conn.commit()
    assert cp.stage_done(conn, book_id, "summaries") is False
    assert cp.done_units(conn, book_id, "entities") == set()


# ── Mock model clients ────────────────────────────────────────────────────────

class _FakeChat:
    """Async chat client returning schema-valid JSON; counts section vs cluster calls."""

    def __init__(self):
        self.section_calls = 0
        self.cluster_calls = 0
        self.summary_calls = 0

    @classmethod
    def install(cls, monkeypatch):
        inst = cls()
        from pagemind.models.chat import ChatClient

        monkeypatch.setattr(ChatClient, "from_config", classmethod(lambda c, axis="query": inst))
        return inst

    async def complete(self, messages, max_tokens=1024, **kwargs):
        user = messages[-1]["content"]
        if "Cluster names" in user:
            self.cluster_calls += 1
            # Canonical name changes between runs — exercises orphan prevention.
            if self.cluster_calls == 1:
                return '{"entities": [{"canonical": "Alice", "type": "character", "aliases": []}]}'
            return '{"entities": [{"canonical": "Alice Liddell", "type": "character", "aliases": ["Alice"]}]}'
        if "Extract structured information" in user:
            self.section_calls += 1
            passage = user.split("chars):\n", 1)[1]
            start = passage.find("Alice")
            return (
                '{"entities": [{"name": "Alice", "type": "character",'
                f' "offset_start": {start}, "offset_end": {start + 5}}}],'
                ' "dates": [], "events": []}'
            )
        # summary prompts
        self.summary_calls += 1
        return "A concise summary."


class _FakeEmbed:
    def __init__(self):
        self.batches = 0
        self.embedded_texts: list[str] = []

    @classmethod
    def install(cls, monkeypatch):
        inst = cls()
        from pagemind.models.embeddings import EmbeddingsClient

        monkeypatch.setattr(EmbeddingsClient, "from_config", classmethod(lambda c: inst))
        return inst

    async def embed(self, texts):
        self.batches += 1
        self.embedded_texts.extend(texts)
        return [[0.0] * 2048 for _ in texts]


# ── Per-stage resume ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_summaries_resume_skips_done_chapters(conn, monkeypatch):
    from pagemind.precompute.summaries import generate_summaries

    ids = _make_book(conn)
    book_id = ids["book_id"]

    # Pretend chapter 0 was already summarised before an interrupt.
    conn.execute(
        "UPDATE chapters SET summary = 'done', micro_summary = 'done' WHERE chapter_id = %s",
        (ids["chapters"][0],),
    )
    cp.mark_unit(conn, book_id, "summaries", str(ids["chapters"][0]))
    conn.commit()

    fake = _FakeChat.install(monkeypatch)
    await generate_summaries(conn, book_id, progress=None)

    # Only chapter 1 should be processed → 2 calls (micro + full), not 4.
    assert fake.summary_calls == 2
    assert cp.stage_done(conn, book_id, "summaries") is True
    assert cp.done_units(conn, book_id, "summaries") == {str(c) for c in ids["chapters"]}


@pytest.mark.asyncio
async def test_embeddings_resume_skips_embedded_chunks(conn, monkeypatch):
    from pagemind.precompute.embeddings import embed_chunks

    ids = _make_book(conn, n_chunks_per_section=2)  # 4 body chunks
    book_id = ids["book_id"]

    # Pre-embed 2 chunks (simulate a prior partial run).
    for chunk_id in ids["chunks"][:2]:
        conn.execute(
            "UPDATE chunks SET embedding = %s::halfvec(2048) WHERE chunk_id = %s",
            (_vec_to_pg([0.0] * 2048), chunk_id),
        )
    conn.commit()

    fake = _FakeEmbed.install(monkeypatch)
    await embed_chunks(conn, book_id, progress=None)

    # Only the 2 remaining NULL chunks get embedded.
    assert len(fake.embedded_texts) == 2
    n_null = conn.execute(
        "SELECT count(*) FROM chunks WHERE book_id = %s AND embedding IS NULL",
        (book_id,),
    ).fetchone()[0]
    assert n_null == 0
    assert cp.stage_done(conn, book_id, "embeddings") is True


@pytest.mark.asyncio
async def test_entities_resume_reuses_payloads_and_no_orphans(conn, monkeypatch):
    from pagemind.precompute.entities import extract_entities

    ids = _make_book(conn)
    book_id = ids["book_id"]

    fake = _FakeChat.install(monkeypatch)

    # First full run.
    await extract_entities(conn, book_id, progress=None)
    assert fake.section_calls == 2          # one per section
    assert fake.cluster_calls == 1
    n_entities = conn.execute(
        "SELECT count(*) FROM entities WHERE book_id = %s", (book_id,)
    ).fetchone()[0]
    n_occ = conn.execute(
        "SELECT count(*) FROM occurrences WHERE book_id = %s", (book_id,)
    ).fetchone()[0]
    assert n_entities == 1
    assert n_occ == 2                        # Alice mentioned in both sections

    # Simulate a back-half crash (sentinel missing) and resume.
    conn.execute(
        "DELETE FROM precompute_checkpoints WHERE book_id = %s AND stage = 'entities' AND unit_key = '*'",
        (book_id,),
    )
    conn.commit()

    await extract_entities(conn, book_id, progress=None)

    # Front half is NOT re-called (payloads reused); only clustering re-runs.
    assert fake.section_calls == 2
    assert fake.cluster_calls == 2
    # Clustering returned a *different* canonical, but the atomic rebuild clears
    # entities first, so there is exactly one entity row — no orphan accumulation.
    n_entities = conn.execute(
        "SELECT count(*) FROM entities WHERE book_id = %s", (book_id,)
    ).fetchone()[0]
    n_occ = conn.execute(
        "SELECT count(*) FROM occurrences WHERE book_id = %s", (book_id,)
    ).fetchone()[0]
    assert n_entities == 1
    assert n_occ == 2
    assert conn.execute(
        "SELECT name FROM entities WHERE book_id = %s", (book_id,)
    ).fetchone()[0] == "Alice Liddell"
