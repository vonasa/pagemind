"""Unit tests for the Phase-3 retrieval substrate.

DB tests skip automatically when Postgres is not reachable, so `just test`
passes in a cold environment. Pure-logic tests (RRF, nudge detection) always run.

Fixture design (2 chapters, 4 sections):
  Chapter 0 "Beginnings":
    section 0 — Alice explores Wonderland; embedding dim-0 unit vector
    section 1 — The rabbit hole leads Alice further into the dream
  Chapter 1 "Endings":
    section 2 — Bob discovers the future city; embedding dim-1 unit vector
    section 3 — The year 1984 marks a turning point for Bob

Embeddings are synthetic halfvec(2048) unit vectors so the semantic test
does not require Infinity.
"""
from __future__ import annotations

import uuid

import psycopg
import pytest

from pagemind.config import settings
from pagemind.retrieval.expand import expand_to_section
from pagemind.retrieval.lexical import detect_nudges, lexical_search
from pagemind.retrieval.rrf import rrf_fuse
from pagemind.retrieval.semantic import semantic_search
from pagemind.retrieval.structured import (
    get_chapter,
    lookup_dates,
    lookup_entities,
    lookup_events,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vec_to_pg(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.8g}" for v in vec) + "]"


def _unit_vec(dim: int, size: int = 2048) -> list[float]:
    v = [0.0] * size
    v[dim] = 1.0
    return v


def _try_connect() -> psycopg.Connection | None:
    try:
        return psycopg.connect(settings.database_url, connect_timeout=3)
    except Exception:
        return None


# ── DB fixture ────────────────────────────────────────────────────────────────

@pytest.fixture
def db_book(request):
    """Create an isolated fixture book; teardown removes it via CASCADE.

    Uses explicit commit() instead of `with conn:` because psycopg3's connection
    context manager closes the connection on exit.
    """
    conn = _try_connect()
    if conn is None:
        pytest.skip("Postgres not reachable")

    book_id = uuid.uuid4()

    # book
    conn.execute(
        """
        INSERT INTO book_meta (book_id, title, author, status, embed_model, embed_dim)
        VALUES (%s, 'Fixture Book', 'Test Author', 'ready',
                'Jasper-Token-Compression-600M', 2048)
        """,
        (book_id,),
    )

    # chapters
    ch0_id = conn.execute(
        """
        INSERT INTO chapters (book_id, ordinal, title, is_body)
        VALUES (%s, 0, 'Beginnings', TRUE) RETURNING chapter_id
        """,
        (book_id,),
    ).fetchone()[0]

    ch1_id = conn.execute(
        """
        INSERT INTO chapters (book_id, ordinal, title, is_body)
        VALUES (%s, 1, 'Endings', TRUE) RETURNING chapter_id
        """,
        (book_id,),
    ).fetchone()[0]

    # sections
    def _ins_section(chapter_id, ordinal, content):
        return conn.execute(
            """
            INSERT INTO sections (book_id, chapter_id, ordinal, content,
                                  char_offset_start, char_offset_end, is_body)
            VALUES (%s, %s, %s, %s, 0, %s, TRUE) RETURNING section_id
            """,
            (book_id, chapter_id, ordinal, content, len(content)),
        ).fetchone()[0]

    s0 = _ins_section(ch0_id, 0, "Alice explores Wonderland with curiosity and delight.")
    s1 = _ins_section(ch0_id, 1, "The rabbit hole leads Alice further into the dream world.")
    s2 = _ins_section(ch1_id, 0, "Bob discovers the future city glowing at night.")
    s3 = _ins_section(ch1_id, 1, "The year 1984 marks a turning point for Bob in the city.")

    # FTS (same as precompute._populate_fts)
    conn.execute(
        """
        INSERT INTO sections_fts (section_id, book_id, chapter_id, fts_vector)
        SELECT s.section_id, s.book_id, s.chapter_id,
               to_tsvector('english', s.content)
        FROM sections s
        WHERE s.book_id = %s
        ON CONFLICT (section_id) DO UPDATE SET fts_vector = EXCLUDED.fts_vector
        """,
        (book_id,),
    )

    # chunks with synthetic embeddings (unit vectors in dim 0 and 1)
    def _ins_chunk(chapter_id, section_id, ordinal, content, vec):
        conn.execute(
            """
            INSERT INTO chunks (book_id, chapter_id, section_id, ordinal, content,
                                char_offset_start, char_offset_end, is_body, embedding)
            VALUES (%s, %s, %s, %s, %s, 0, %s, TRUE, %s::halfvec(2048))
            """,
            (book_id, chapter_id, section_id, ordinal, content,
             len(content), _vec_to_pg(vec)),
        )

    _ins_chunk(ch0_id, s0, 0, "Alice explores Wonderland.", _unit_vec(0))
    _ins_chunk(ch0_id, s1, 0, "The rabbit hole leads Alice.", _unit_vec(0))
    _ins_chunk(ch1_id, s2, 0, "Bob discovers the future city.", _unit_vec(1))
    _ins_chunk(ch1_id, s3, 0, "The year 1984 marks Bob.", _unit_vec(1))

    # entities
    alice_id = conn.execute(
        """
        INSERT INTO entities (book_id, name, entity_type, aliases)
        VALUES (%s, 'Alice', 'PERSON', ARRAY['Alice Liddell'])
        ON CONFLICT (book_id, name, entity_type) DO UPDATE SET aliases = EXCLUDED.aliases
        RETURNING entity_id
        """,
        (book_id,),
    ).fetchone()[0]

    conn.execute(
        """
        INSERT INTO occurrences (book_id, entity_id, chapter_id, section_id,
                                 char_offset_start, char_offset_end, context)
        VALUES (%s, %s, %s, %s, 0, 5, 'Alice explores')
        """,
        (book_id, alice_id, ch0_id, s0),
    )

    # dates
    conn.execute(
        """
        INSERT INTO dates (book_id, chapter_id, section_id, raw_text,
                           normalized_date, char_offset_start, char_offset_end, context)
        VALUES (%s, %s, %s, '1984', '1984-01-01'::date, 4, 8,
                'The year 1984 marks')
        """,
        (book_id, ch1_id, s3),
    )

    # events
    conn.execute(
        """
        INSERT INTO events (book_id, chapter_id, section_id, description,
                            entity_ids, char_offset_start, char_offset_end)
        VALUES (%s, %s, %s, 'Alice falls down the rabbit hole', NULL, 0, 10)
        """,
        (book_id, ch0_id, s1),
    )

    conn.commit()

    ids = {
        "ch0": ch0_id, "ch1": ch1_id,
        "s0": s0, "s1": s1, "s2": s2, "s3": s3,
        "alice_id": alice_id,
    }

    def _cleanup():
        conn.rollback()  # discard any dirty transaction from the test
        conn.execute("DELETE FROM book_meta WHERE book_id = %s", (book_id,))
        conn.commit()
        conn.close()

    request.addfinalizer(_cleanup)
    return conn, book_id, ids


# ── Pure-logic tests (no DB required) ─────────────────────────────────────────

class TestRRF:
    def test_single_list(self):
        a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        result = rrf_fuse([[a, b, c]])
        ids = [r[0] for r in result]
        assert ids == [a, b, c]

    def test_agreement_raises_rank(self):
        a, b = uuid.uuid4(), uuid.uuid4()
        # Both lists agree: a first, b second
        result = rrf_fuse([[a, b], [a, b]])
        assert result[0][0] == a
        assert result[1][0] == b
        # a should have higher score than b
        assert result[0][1] > result[1][1]

    def test_disagreement_promotes_agreement(self):
        # a is first in both → should beat b even if b appears in only one list
        a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        result = rrf_fuse([[a, b], [a, c]])
        assert result[0][0] == a

    def test_weights_applied(self):
        a, b = uuid.uuid4(), uuid.uuid4()
        # a only in list 0 (weight 2.0); b only in list 1 (weight 0.5)
        result = rrf_fuse([[a], [b]], weights=[2.0, 0.5])
        assert result[0][0] == a

    def test_empty_lists(self):
        assert rrf_fuse([]) == []
        assert rrf_fuse([[]]) == []

    def test_scores_positive(self):
        ids = [uuid.uuid4() for _ in range(5)]
        result = rrf_fuse([ids])
        assert all(s > 0 for _, s in result)

    def test_deduplication(self):
        a = uuid.uuid4()
        b = uuid.uuid4()
        # Same id in different positions in two lists
        result = rrf_fuse([[a, b], [b, a]])
        assert len(result) == 2


class TestDetectNudges:
    def test_no_nudges(self):
        n = detect_nudges("what happened at the end")
        assert not n["boost_lexical"]
        assert not n["boost_date"]

    def test_quoted_phrase_boosts_lexical(self):
        n = detect_nudges('find "Mr. Darcy" in the text')
        assert n["boost_lexical"]

    def test_capitalized_mid_word_boosts_lexical(self):
        n = detect_nudges("what did Elizabeth think")
        assert n["boost_lexical"]

    def test_year_boosts_date(self):
        n = detect_nudges("what happened in 1984")
        assert n["boost_date"]

    def test_month_name_boosts_date(self):
        n = detect_nudges("events in January")
        assert n["boost_date"]

    def test_first_word_capital_no_boost(self):
        # First word capitalized is just the start of a sentence
        n = detect_nudges("What was the rabbit doing")
        assert not n["boost_lexical"]


# ── DB tests ──────────────────────────────────────────────────────────────────

class TestLexicalSearch:
    def test_exact_name_finds_section(self, db_book):
        conn, book_id, ids = db_book
        results = lexical_search(conn, book_id, "Alice")
        assert ids["s0"] in results or ids["s1"] in results, (
            "Alice sections not found in lexical results"
        )

    def test_query_excludes_unrelated_section(self, db_book):
        conn, book_id, ids = db_book
        results = lexical_search(conn, book_id, "Alice")
        # Bob-only sections should not appear
        assert ids["s2"] not in results
        assert ids["s3"] not in results

    def test_up_to_chapter_filters_scope(self, db_book):
        conn, book_id, ids = db_book
        # Searching for "Bob" limited to chapter 0 should return nothing
        results = lexical_search(conn, book_id, "Bob", up_to_chapter=0)
        assert ids["s2"] not in results
        assert ids["s3"] not in results

    def test_up_to_chapter_includes_correct_chapter(self, db_book):
        conn, book_id, ids = db_book
        results = lexical_search(conn, book_id, "Bob", up_to_chapter=1)
        assert ids["s2"] in results or ids["s3"] in results

    def test_no_match_returns_empty(self, db_book):
        conn, book_id, ids = db_book
        results = lexical_search(conn, book_id, "xyzzy_nonexistent_token_q9q")
        assert results == []

    def test_top_k_respected(self, db_book):
        conn, book_id, ids = db_book
        results = lexical_search(conn, book_id, "the", top_k=1)
        assert len(results) <= 1


class TestSemanticSearch:
    def test_dim0_query_returns_alice_sections(self, db_book):
        conn, book_id, ids = db_book
        # Query vector aligned with Alice sections (dim 0 unit vector)
        results = semantic_search(conn, book_id, _unit_vec(0))
        assert len(results) >= 1
        assert results[0] in (ids["s0"], ids["s1"]), (
            f"Expected Alice section first, got {results[0]}"
        )

    def test_dim1_query_returns_bob_sections(self, db_book):
        conn, book_id, ids = db_book
        results = semantic_search(conn, book_id, _unit_vec(1))
        assert len(results) >= 1
        assert results[0] in (ids["s2"], ids["s3"]), (
            f"Expected Bob section first, got {results[0]}"
        )

    def test_up_to_chapter_scope(self, db_book):
        conn, book_id, ids = db_book
        # Bob sections are in chapter 1; limiting to chapter 0 must exclude them
        results = semantic_search(conn, book_id, _unit_vec(1), up_to_chapter=0)
        assert ids["s2"] not in results
        assert ids["s3"] not in results

    def test_top_k_respected(self, db_book):
        conn, book_id, ids = db_book
        results = semantic_search(conn, book_id, _unit_vec(0), top_k=1)
        assert len(results) <= 1


class TestExpandToSection:
    def test_returns_section_content(self, db_book):
        conn, book_id, ids = db_book
        text = expand_to_section(conn, ids["s0"])
        assert text is not None
        assert "Alice" in text

    def test_cap_truncates_long_content(self, db_book):
        conn, book_id, ids = db_book
        text = expand_to_section(conn, ids["s0"], char_cap=5)
        assert text is not None
        assert len(text) == 5

    def test_short_content_not_truncated(self, db_book):
        conn, book_id, ids = db_book
        text = expand_to_section(conn, ids["s0"])
        original = conn.execute(
            "SELECT content FROM sections WHERE section_id = %s", (ids["s0"],)
        ).fetchone()[0]
        assert text == original

    def test_unknown_section_returns_none(self, db_book):
        conn, book_id, ids = db_book
        assert expand_to_section(conn, uuid.uuid4()) is None


class TestGetChapter:
    def test_returns_chapter_metadata(self, db_book):
        conn, book_id, ids = db_book
        ch = get_chapter(conn, book_id, 0)
        assert ch is not None
        assert ch["title"] == "Beginnings"
        assert ch["ordinal"] == 0

    def test_wrong_ordinal_returns_none(self, db_book):
        conn, book_id, ids = db_book
        assert get_chapter(conn, book_id, 99) is None

    def test_chapter_1_accessible(self, db_book):
        conn, book_id, ids = db_book
        ch = get_chapter(conn, book_id, 1)
        assert ch is not None
        assert ch["title"] == "Endings"


class TestLookupEntities:
    def test_finds_alice_by_name(self, db_book):
        conn, book_id, ids = db_book
        results = lookup_entities(conn, book_id, "Alice")
        assert len(results) >= 1
        assert results[0]["name"] == "Alice"

    def test_finds_by_alias(self, db_book):
        conn, book_id, ids = db_book
        results = lookup_entities(conn, book_id, "Liddell")
        assert len(results) >= 1

    def test_occurrences_included(self, db_book):
        conn, book_id, ids = db_book
        results = lookup_entities(conn, book_id, "Alice")
        assert len(results[0]["occurrences"]) >= 1
        assert results[0]["occurrences"][0]["section_id"] == ids["s0"]

    def test_up_to_chapter_scope_on_occurrences(self, db_book):
        conn, book_id, ids = db_book
        # Alice is only in chapter 0, so up_to_chapter=0 should still find her
        results = lookup_entities(conn, book_id, "Alice", up_to_chapter=0)
        assert len(results) >= 1

    def test_unknown_entity_returns_empty(self, db_book):
        conn, book_id, ids = db_book
        assert lookup_entities(conn, book_id, "Xyzzy_NoSuchPerson") == []


class TestLookupDates:
    def test_finds_year_1984(self, db_book):
        conn, book_id, ids = db_book
        results = lookup_dates(conn, book_id, "1984")
        assert len(results) >= 1
        assert results[0]["raw_text"] == "1984"

    def test_up_to_chapter_excludes_future_chapter(self, db_book):
        conn, book_id, ids = db_book
        # 1984 date is in chapter 1; restricting to chapter 0 must return nothing
        results = lookup_dates(conn, book_id, "1984", up_to_chapter=0)
        assert results == []

    def test_up_to_chapter_includes_correct_chapter(self, db_book):
        conn, book_id, ids = db_book
        results = lookup_dates(conn, book_id, "1984", up_to_chapter=1)
        assert len(results) >= 1

    def test_no_match_returns_empty(self, db_book):
        conn, book_id, ids = db_book
        assert lookup_dates(conn, book_id, "2099") == []


class TestLookupEvents:
    def test_finds_rabbit_hole_event(self, db_book):
        conn, book_id, ids = db_book
        results = lookup_events(conn, book_id, "rabbit hole")
        assert len(results) >= 1
        assert "rabbit hole" in results[0]["description"].lower()

    def test_up_to_chapter_scope(self, db_book):
        conn, book_id, ids = db_book
        # rabbit-hole event is in chapter 0; chapter 0 scope should include it
        results = lookup_events(conn, book_id, "rabbit", up_to_chapter=0)
        assert len(results) >= 1

    def test_up_to_chapter_excludes_later_chapters(self, db_book):
        conn, book_id, ids = db_book
        # If we restrict to chapter -1 (before any chapter), nothing should return
        results = lookup_events(conn, book_id, "rabbit", up_to_chapter=-1)
        assert results == []

    def test_no_match_returns_empty(self, db_book):
        conn, book_id, ids = db_book
        assert lookup_events(conn, book_id, "xyzzy_nonexistent") == []


class TestScopeCrossCheck:
    """Verify up_to_chapter=N consistently excludes N+1 across all query types."""

    def test_lexical_semantic_same_scope(self, db_book):
        conn, book_id, ids = db_book
        bob_sections = {ids["s2"], ids["s3"]}

        lex = set(lexical_search(conn, book_id, "Bob", up_to_chapter=0))
        sem = set(semantic_search(conn, book_id, _unit_vec(1), up_to_chapter=0))

        assert lex.isdisjoint(bob_sections), "Lexical leaked chapter-1 section"
        assert sem.isdisjoint(bob_sections), "Semantic leaked chapter-1 section"
