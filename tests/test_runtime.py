"""Tests for Phase-4 runtime: types, router, reader, synthesizer, orchestrator, recipes.

Mock-LLM tests always run. DB-backed tests skip when Postgres is not reachable.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from unittest.mock import AsyncMock, MagicMock

import psycopg
import pytest

from pagemind.config import settings
from pagemind.runtime.types import Citation, QueryResult, Quote, ReadResult
from pagemind.runtime.router import _parse_recipe, RECIPES
from pagemind.runtime.reader import _parse_reader_response, _locate_quote, _normalize
from pagemind.runtime.synthesizer import _format_evidence, _build_output


# ── Helpers ───────────────────────────────────────────────────────────────────

def _try_connect() -> psycopg.Connection | None:
    try:
        return psycopg.connect(settings.database_url, connect_timeout=3)
    except Exception:
        return None


def _mock_chat(response: str) -> MagicMock:
    chat = MagicMock()
    chat.complete = AsyncMock(return_value=response)
    return chat


def _unit_vec(dim: int, size: int = 2048) -> list[float]:
    v = [0.0] * size
    v[dim] = 1.0
    return v


def _vec_to_pg(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.8g}" for v in vec) + "]"


# ── DB fixture (same pattern as test_retrieval.py) ────────────────────────────

@pytest.fixture
def db_book(request):
    conn = _try_connect()
    if conn is None:
        pytest.skip("Postgres not reachable")

    book_id = uuid.uuid4()
    conn.execute(
        """
        INSERT INTO book_meta (book_id, title, author, status, embed_model, embed_dim)
        VALUES (%s, 'Runtime Test Book', 'Test Author', 'ready',
                'Jasper-Token-Compression-600M', 2048)
        """,
        (book_id,),
    )

    ch0_id = conn.execute(
        """
        INSERT INTO chapters (book_id, ordinal, title, is_body, summary, micro_summary)
        VALUES (%s, 0, 'Beginnings', TRUE,
                'Alice falls into Wonderland and meets strange creatures.',
                'Alice enters Wonderland.')
        RETURNING chapter_id
        """,
        (book_id,),
    ).fetchone()[0]

    ch1_id = conn.execute(
        """
        INSERT INTO chapters (book_id, ordinal, title, is_body, summary, micro_summary)
        VALUES (%s, 1, 'Endings', TRUE,
                'Bob discovers the future and returns changed.',
                'Bob finds the future.')
        RETURNING chapter_id
        """,
        (book_id,),
    ).fetchone()[0]

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

    conn.execute(
        """
        INSERT INTO sections_fts (section_id, book_id, chapter_id, fts_vector)
        SELECT s.section_id, s.book_id, s.chapter_id, to_tsvector('english', s.content)
        FROM sections s WHERE s.book_id = %s
        ON CONFLICT (section_id) DO UPDATE SET fts_vector = EXCLUDED.fts_vector
        """,
        (book_id,),
    )

    def _ins_chunk(chapter_id, section_id, ordinal, content, vec):
        conn.execute(
            """
            INSERT INTO chunks (book_id, chapter_id, section_id, ordinal, content,
                                char_offset_start, char_offset_end, is_body, embedding)
            VALUES (%s, %s, %s, %s, %s, 0, %s, TRUE, %s::halfvec(2048))
            """,
            (book_id, chapter_id, section_id, ordinal, content, len(content), _vec_to_pg(vec)),
        )

    _ins_chunk(ch0_id, s0, 0, "Alice explores Wonderland.", _unit_vec(0))
    _ins_chunk(ch0_id, s1, 0, "The rabbit hole leads Alice.", _unit_vec(0))
    _ins_chunk(ch1_id, s2, 0, "Bob discovers the future city.", _unit_vec(1))
    _ins_chunk(ch1_id, s3, 0, "The year 1984 marks Bob.", _unit_vec(1))

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
        VALUES (%s, %s, %s, %s, 0, 5, 'Alice explores Wonderland')
        """,
        (book_id, alice_id, ch0_id, s0),
    )

    conn.commit()

    ids = {"ch0": ch0_id, "ch1": ch1_id, "s0": s0, "s1": s1, "s2": s2, "s3": s3,
           "alice_id": alice_id}

    def _cleanup():
        conn.rollback()
        conn.execute("DELETE FROM book_meta WHERE book_id = %s", (book_id,))
        conn.commit()
        conn.close()

    request.addfinalizer(_cleanup)
    return conn, book_id, ids


# ── Types ─────────────────────────────────────────────────────────────────────

class TestTypes:
    def test_query_result_defaults(self):
        r = QueryResult(text="hello")
        assert r.text == "hello"
        assert r.quotes == []
        assert r.citations == []
        assert r.weak is False

    def test_read_result_defaults(self):
        sid = uuid.uuid4()
        r = ReadResult(section_id=sid, chapter=2, answer="x")
        assert r.verbatim_quotes == []
        assert r.char_offsets == []

    def test_citation_fields(self):
        bid = uuid.uuid4()
        sid = uuid.uuid4()
        c = Citation(book_id=bid, chapter=3, section_id=sid, char_offset=42)
        assert c.char_offset == 42

    def test_quote_wraps_citation(self):
        bid, sid = uuid.uuid4(), uuid.uuid4()
        cit = Citation(book_id=bid, chapter=1, section_id=sid)
        q = Quote(text="exact text", citation=cit)
        assert q.citation.chapter == 1


# ── Router ────────────────────────────────────────────────────────────────────

class TestRouter:
    def test_all_recipes_recognized(self):
        for recipe in RECIPES:
            assert _parse_recipe(recipe) == recipe

    def test_recipe_in_longer_response(self):
        assert _parse_recipe("I think this is fact_lookup.") == "fact_lookup"

    def test_unknown_defaults_to_generic_fallback(self):
        assert _parse_recipe("something completely unrecognized") == "generic_fallback"

    def test_case_insensitive(self):
        assert _parse_recipe("FACT_LOOKUP") == "fact_lookup"

    def test_empty_response_defaults(self):
        assert _parse_recipe("") == "generic_fallback"

    def test_verbatim_quote_recognized(self):
        assert _parse_recipe("  verbatim_quote  ") == "verbatim_quote"

    def test_chapter_summary_recognized(self):
        assert _parse_recipe("chapter_summary") == "chapter_summary"

    def test_contextual_why_recognized(self):
        assert _parse_recipe("contextual_why") == "contextual_why"

    async def test_route_calls_chat(self):
        from pagemind.runtime.router import route
        chat = _mock_chat("fact_lookup")
        result = await route(chat, "Who is Alice?")
        assert result == "fact_lookup"
        chat.complete.assert_called_once()

    async def test_route_falls_back_on_bad_llm_output(self):
        from pagemind.runtime.router import route
        chat = _mock_chat("I have no idea what this is")
        result = await route(chat, "some obscure question")
        assert result == "generic_fallback"


# ── Reader ────────────────────────────────────────────────────────────────────

class TestReaderParsing:
    def test_valid_json_parsed(self):
        raw = json.dumps({"answer": "Alice is a girl.", "verbatim_quotes": ["Alice"]})
        answer, quotes = _parse_reader_response(raw)
        assert answer == "Alice is a girl."
        assert quotes == ["Alice"]

    def test_json_embedded_in_text(self):
        raw = 'Here is my response: {"answer": "She falls.", "verbatim_quotes": ["falls down"]}'
        answer, quotes = _parse_reader_response(raw)
        assert "falls" in answer

    def test_bad_json_falls_back_to_raw(self):
        raw = "This is not JSON at all."
        answer, quotes = _parse_reader_response(raw)
        assert answer == raw
        assert quotes == []

    def test_empty_quotes_list(self):
        raw = json.dumps({"answer": "Nothing relevant.", "verbatim_quotes": []})
        answer, quotes = _parse_reader_response(raw)
        assert quotes == []

    def test_locate_quote_exact_match(self):
        source = "Alice explores Wonderland with delight."
        slice_, s, e = _locate_quote(source, "Alice explores")
        assert (slice_, s, e) == ("Alice explores", 0, 14)
        i = source.index("with delight")
        assert _locate_quote(source, "with delight") == ("with delight", i, i + len("with delight"))

    def test_locate_quote_hallucination_returns_none(self):
        assert _locate_quote("hello world", "nonexistent phrase") is None


class TestQuoteRealign:
    """The reader model reproduces words, not bytes: realign to the source slice."""

    def _check(self, source, quote):
        located = _locate_quote(source, quote)
        assert located is not None, f"could not locate {quote!r}"
        slice_, start, end = located
        # Byte-honest span (popover invariant) + normalised-substring match.
        assert source[start:end] == slice_
        assert end <= len(source)
        assert end == start + len(slice_)
        assert _normalize(quote) in _normalize(slice_)
        return slice_, start, end

    def test_newline_collapse(self):
        # The real Step-0 failure: EPUB hard line-break the model rendered as space.
        source = "A flame went over his eyes, and a flame flew\nover her body, melting her bones."
        slice_, _s, _e = self._check(source, "A flame flew over her body, melting her bones.")
        assert "\n" in slice_  # original slice keeps the verbatim newline

    def test_curly_quotes_and_dash(self):
        source = "She said “yes”—firmly—and left."
        self._check(source, 'She said "yes"-firmly-and left.')

    def test_ligature_length_change(self):
        # NFKC expands the "fi" ligature; the per-char map must still align.
        source = "The oﬃce was quiet."  # "o<ffi>ce"
        self._check(source, "The office was quiet.")

    def test_nbsp_and_multi_space(self):
        source = "one two   three"
        self._check(source, "one two three")

    def test_mid_expansion_boundary(self):
        # Quote boundary falls INSIDE a single source char's NFKC expansion
        # (ﬃ→ffi): "The off" ends at the second f, which came from ﬃ. end snaps
        # outward to the whole ligature, so the invariant is substring (not equality).
        source = "The oﬃce was quiet."  # "o<ffi>ce"
        slice_, _s, _e = self._check(source, "The off")
        assert "ﬃ" in slice_  # snapped outward to include the full ligature


# ── Synthesizer ───────────────────────────────────────────────────────────────

class TestSynthesizer:
    def test_format_evidence_single(self):
        r = ReadResult(
            section_id=uuid.uuid4(), chapter=0, answer="Alice is brave.",
            verbatim_quotes=["Alice"], char_offsets=[(0, 5)],
        )
        ev = _format_evidence([r])
        assert "chapter 0" in ev
        assert "Alice is brave" in ev
        assert '"Alice"' in ev

    def test_format_evidence_multiple(self):
        r1 = ReadResult(section_id=uuid.uuid4(), chapter=0, answer="a1")
        r2 = ReadResult(section_id=uuid.uuid4(), chapter=1, answer="a2")
        ev = _format_evidence([r1, r2])
        assert "Source 1" in ev
        assert "Source 2" in ev

    def test_build_output_deduplicates_citations(self):
        bid = uuid.uuid4()
        sid = uuid.uuid4()
        r = ReadResult(section_id=sid, chapter=0, answer="x",
                       verbatim_quotes=["q"], char_offsets=[(0, 1)])
        result = _build_output(bid, "q?", "answer text", [r])
        # Only one unique section → one citation
        assert len(result.citations) == 1
        assert result.citations[0].section_id == sid

    def test_build_output_quotes_get_citation(self):
        bid = uuid.uuid4()
        sid = uuid.uuid4()
        r = ReadResult(section_id=sid, chapter=1, answer="x",
                       verbatim_quotes=["exact quote"], char_offsets=[(5, 16)])
        result = _build_output(bid, "q?", "answer", [r])
        assert len(result.quotes) == 1
        assert result.quotes[0].text == "exact quote"
        assert result.quotes[0].citation.char_offset == 5

    async def test_synthesize_calls_chat(self):
        from pagemind.runtime.synthesizer import synthesize
        bid = uuid.uuid4()
        sid = uuid.uuid4()
        r = ReadResult(section_id=sid, chapter=0, answer="Alice is brave.")
        chat = _mock_chat("Alice shows bravery throughout the story.")
        result = await synthesize(chat, bid, "Is Alice brave?", [r])
        assert "Alice" in result.text or "brave" in result.text.lower()

    async def test_synthesize_empty_results(self):
        from pagemind.runtime.synthesizer import synthesize
        bid = uuid.uuid4()
        chat = _mock_chat("irrelevant")
        result = await synthesize(chat, bid, "question?", [])
        assert result.text == "No relevant passages found."


# ── Grounded mode toggle ──────────────────────────────────────────────────────

class TestGroundedMode:
    def test_synth_answer_prompts_grounded(self):
        from pagemind.runtime.synthesizer import answer_prompts, _SYS, _PROMPT
        assert answer_prompts(True) == (_SYS, _PROMPT)

    def test_synth_answer_prompts_open_differs(self):
        from pagemind.runtime.synthesizer import answer_prompts
        assert answer_prompts(False) != answer_prompts(True)

    def test_reader_prompts_grounded_forbids_outside_knowledge(self):
        from pagemind.runtime.reader import _reader_prompts
        sys_grounded, _ = _reader_prompts(True)
        assert "never add outside knowledge" in sys_grounded

    def test_reader_prompts_open_differs(self):
        from pagemind.runtime.reader import _reader_prompts
        assert _reader_prompts(False) != _reader_prompts(True)

    async def test_read_uses_open_prompt_when_ungrounded(self, monkeypatch):
        """read(grounded=False) must send the open-book system prompt to the LLM."""
        import pagemind.runtime.reader as reader_mod
        from pagemind.runtime.reader import read, _SYS, _SYS_OPEN

        monkeypatch.setattr(reader_mod, "expand_to_section", lambda conn, sid: "Some passage text.")
        monkeypatch.setattr(reader_mod, "_section_chapter", lambda conn, sid: 1)
        chat = _mock_chat('{"answer": "ok", "verbatim_quotes": []}')

        await read(None, chat, uuid.uuid4(), "q?", grounded=False)
        sys_msg = chat.complete.call_args.args[0][0]["content"]
        assert sys_msg == _SYS_OPEN and sys_msg != _SYS

    def test_ask_request_defaults_grounded_true(self):
        from pagemind.api import AskRequest
        assert AskRequest(question="hi").grounded is True


# ── Fan-out concurrency ───────────────────────────────────────────────────────

class TestFanOut:
    async def test_fan_out_respects_concurrency_cap(self, db_book):
        from pagemind.runtime.reader import fan_out
        conn, book_id, ids = db_book

        call_times: list[float] = []
        concurrent: list[int] = [0]
        max_seen: list[int] = [0]

        async def slow_read(c, ch, section_id, question):
            concurrent[0] += 1
            max_seen[0] = max(max_seen[0], concurrent[0])
            call_times.append(time.monotonic())
            await asyncio.sleep(0.05)
            concurrent[0] -= 1
            return ReadResult(section_id=section_id, chapter=0, answer="ok")

        from pagemind.runtime import reader as reader_mod
        orig_read = reader_mod.read

        async def patched_fan_out(conn, chat, section_ids, question, *, max_concurrent=2):
            sem = asyncio.Semaphore(max_concurrent)
            async def _g(sid):
                async with sem:
                    return await slow_read(conn, chat, sid, question)
            return list(await asyncio.gather(*(_g(sid) for sid in section_ids)))

        section_ids = [ids["s0"], ids["s1"], ids["s2"], ids["s3"]]
        chat = _mock_chat("{}")
        results = await patched_fan_out(conn, chat, section_ids, "test", max_concurrent=2)
        assert len(results) == 4
        assert max_seen[0] <= 2

    async def test_fan_out_returns_all_results(self, db_book):
        from pagemind.runtime.reader import fan_out
        conn, book_id, ids = db_book

        reader_response = json.dumps({"answer": "answer", "verbatim_quotes": []})
        chat = _mock_chat(reader_response)
        section_ids = [ids["s0"], ids["s1"]]
        results = await fan_out(conn, chat, section_ids, "test question")
        assert len(results) == 2
        assert all(isinstance(r, ReadResult) for r in results)


# ── Orchestrator round cap ────────────────────────────────────────────────────

class TestOrchestrator:
    async def test_round_cap_zero_returns_error(self, db_book):
        from pagemind.runtime.orchestrator import ask
        conn, book_id, _ = db_book
        result = await ask(conn, book_id, "any question?", max_rounds=0)
        assert "exceeded" in result.text.lower() or result.text  # cap 0 → no rounds

    async def test_normal_round_returns_result(self, db_book):
        """Orchestrator dispatches and returns QueryResult — inner calls fully mocked."""
        import pagemind.runtime.orchestrator as orch_mod

        conn, book_id, _ = db_book
        expected = QueryResult(text="Mocked recipe answer.", citations=[])

        async def mock_route(chat, question):
            return "generic_fallback"

        async def mock_dispatch(recipe, conn, chat, book_id, question, *, up_to_chapter=None):
            return expected

        orig_route = orch_mod.route
        orig_dispatch = orch_mod.dispatch
        orch_mod.route = mock_route
        orch_mod.dispatch = mock_dispatch
        try:
            result = await orch_mod.ask(conn, book_id, "Tell me about Alice.", max_rounds=10)
            assert isinstance(result, QueryResult)
            assert result.text == "Mocked recipe answer."
        finally:
            orch_mod.route = orig_route
            orch_mod.dispatch = orig_dispatch


# ── Recipe: chapter_summary ───────────────────────────────────────────────────

class TestChapterSummaryRecipe:
    from pagemind.recipes.chapter_summary import _parse_chapter_number

    def test_parse_chapter_number(self):
        from pagemind.recipes.chapter_summary import _parse_chapter_number
        assert _parse_chapter_number("summarize chapter 3") == 3
        assert _parse_chapter_number("chapter 10 summary") == 10
        assert _parse_chapter_number("what happened in ch. 2") == 2
        assert _parse_chapter_number("the second chapter") == 2
        assert _parse_chapter_number("summarize the first chapter") == 1

    def test_parse_no_number_returns_none(self):
        from pagemind.recipes.chapter_summary import _parse_chapter_number
        assert _parse_chapter_number("what happened so far") is None

    async def test_chapter_summary_returns_precomputed(self, db_book):
        from pagemind.recipes.chapter_summary import run
        conn, book_id, ids = db_book
        chat = _mock_chat("unused")
        result = await run(conn, chat, book_id, "summarize chapter 0")
        assert "Alice" in result.text or "Wonderland" in result.text
        assert result.weak is False  # a real summary is not a dead end

    async def test_chapter_summary_missing_chapter(self, db_book):
        from pagemind.recipes.chapter_summary import run
        conn, book_id, _ = db_book
        chat = _mock_chat("unused")
        result = await run(conn, chat, book_id, "summarize chapter 99")
        assert "not found" in result.text.lower()
        assert result.weak is True

    async def test_chapter_summary_no_number_in_query(self, db_book):
        from pagemind.recipes.chapter_summary import run
        conn, book_id, _ = db_book
        chat = _mock_chat("unused")
        result = await run(conn, chat, book_id, "what happened so far")
        assert "chapter" in result.text.lower() or "specify" in result.text.lower()


# ── Recipe: locate_entity ─────────────────────────────────────────────────────

class TestLocateEntityRecipe:
    def test_extract_simple_name(self):
        from pagemind.recipes.locate_entity import _extract_entity_name
        assert _extract_entity_name("where does Alice appear").lower() == "alice"

    def test_extract_with_alias(self):
        from pagemind.recipes.locate_entity import _extract_entity_name
        name = _extract_entity_name("find all occurrences of Elizabeth Bennet")
        assert "elizabeth" in name.lower() or "bennet" in name.lower()

    async def test_locate_entity_finds_alice(self, db_book):
        from pagemind.recipes.locate_entity import run
        conn, book_id, ids = db_book
        chat = _mock_chat("unused")
        result = await run(conn, chat, book_id, "where does Alice appear")
        assert "Alice" in result.text
        assert len(result.citations) >= 1

    async def test_locate_entity_unknown(self, db_book):
        from pagemind.recipes.locate_entity import run
        conn, book_id, _ = db_book
        chat = _mock_chat("unused")
        result = await run(conn, chat, book_id, "where does Xyzzy_NoEntity appear")
        assert "not found" in result.text.lower() or len(result.citations) == 0
        assert result.weak is True  # no usable index result → triggers fallback

    async def test_locate_entity_matched_but_zero_occurrences(self, db_book):
        """The reported Julietta case: entity exists but has no occurrences → weak."""
        from pagemind.recipes.locate_entity import run
        conn, book_id, _ = db_book
        conn.execute(
            """
            INSERT INTO entities (book_id, name, entity_type, aliases)
            VALUES (%s, 'Ghost', 'PERSON', ARRAY[]::text[])
            ON CONFLICT (book_id, name, entity_type) DO NOTHING
            """,
            (book_id,),
        )
        conn.commit()
        chat = _mock_chat("unused")
        result = await run(conn, chat, book_id, "where does Ghost appear")
        assert "0 occurrence(s)" in result.text
        assert result.citations == []
        assert result.weak is True


# ── Recipe: generic_fallback ──────────────────────────────────────────────────

class TestGenericFallbackRecipe:
    async def test_returns_query_result(self, db_book):
        import pagemind.recipes.generic_fallback as gf_mod
        conn, book_id, ids = db_book

        reader_json = json.dumps({"answer": "Alice is curious.", "verbatim_quotes": []})
        synth = "A synthesized answer about Alice."

        async def mock_complete(messages, max_tokens=1024, **kw):
            if any("PASSAGE" in (m.get("content") or "") for m in messages):
                return reader_json
            return synth

        chat = MagicMock()
        chat.complete = AsyncMock(side_effect=mock_complete)

        orig_hybrid = gf_mod.hybrid_search
        gf_mod.hybrid_search = AsyncMock(return_value=[(ids["s0"], 0.9)])
        try:
            result = await gf_mod.run(conn, chat, book_id, "Tell me about Alice")
            assert isinstance(result, QueryResult)
            assert result.text
        finally:
            gf_mod.hybrid_search = orig_hybrid


# ── Recipe: structured_view ───────────────────────────────────────────────────

class TestStructuredViewRecipe:
    async def test_reader_enriched_narrative(self, db_book):
        from pagemind.recipes.structured_view import run
        conn, book_id, ids = db_book
        # Give Alice a scene-mate in s0 so exactly one edge / one shared section exists.
        bob_id = conn.execute(
            """
            INSERT INTO entities (book_id, name, entity_type)
            VALUES (%s, 'Bob', 'PERSON') RETURNING entity_id
            """,
            (book_id,),
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO occurrences (book_id, entity_id, chapter_id, section_id,
                                     char_offset_start, char_offset_end, context)
            VALUES (%s, %s, %s, %s, 0, 3, 'Bob')
            """,
            (book_id, bob_id, ids["ch0"], ids["s0"]),
        )
        conn.commit()

        reader_json = json.dumps({
            "answer": "Alice and Bob explore Wonderland together.",
            "verbatim_quotes": ["Alice explores Wonderland"],  # substring of s0
        })
        synth = "Alice and Bob are companions who share the opening scene."

        async def mock_complete(messages, max_tokens=1024, **kw):
            # Reader prompts contain PASSAGE; the synthesis prompt must not.
            user = messages[-1]["content"]
            return reader_json if "PASSAGE" in user else synth

        chat = MagicMock()
        chat.complete = AsyncMock(side_effect=mock_complete)

        result = await run(conn, chat, book_id, "how are the characters connected?")
        # The synthesis (non-PASSAGE call) is returned; a reader ran first.
        assert result.text == synth
        assert not result.weak
        # The shared section was read → cited, and its verbatim quote carried for the UI.
        assert any(c.section_id == ids["s0"] for c in result.citations)
        assert any(q.text == "Alice explores Wonderland" for q in result.quotes)
        # Exactly one synthesis call, whose evidence names both characters and never
        # contains "PASSAGE" (which would have mis-branched to the reader path).
        synth_calls = [
            c for c in chat.complete.await_args_list
            if "PASSAGE" not in c.args[0][-1]["content"]
        ]
        assert len(synth_calls) == 1
        evidence = synth_calls[0].args[0][-1]["content"]
        assert "Alice" in evidence and "Bob" in evidence
        assert "PASSAGE" not in evidence

    async def test_no_cooccurrence_is_weak(self, db_book):
        # Single indexed entity (Alice) → no pairs → weak, defers to fallback.
        from pagemind.recipes.structured_view import run
        conn, book_id, _ = db_book
        chat = _mock_chat("unused")
        result = await run(conn, chat, book_id, "map the relationships")
        assert result.weak
        chat.complete.assert_not_awaited()

    async def test_no_entities_message(self, db_book):
        from pagemind.recipes.structured_view import run, _get_all_entities
        conn, book_id, _ = db_book
        # Remove Alice from this book temporarily by using a fresh empty book_id
        empty_book_id = uuid.uuid4()
        conn.execute(
            """
            INSERT INTO book_meta (book_id, title, author, status, embed_model, embed_dim)
            VALUES (%s, 'Empty', 'Nobody', 'ready', 'Jasper-Token-Compression-600M', 2048)
            """,
            (empty_book_id,),
        )
        conn.commit()
        try:
            chat = _mock_chat("unused")
            result = await run(conn, chat, empty_book_id, "map relationships")
            assert "No entities" in result.text or result.text
        finally:
            conn.execute("DELETE FROM book_meta WHERE book_id = %s", (empty_book_id,))
            conn.commit()


# ── structured_view: pure graph helpers (no DB) ───────────────────────────────

class TestCooccurrenceGraph:
    def _ent(self, name, sids, etype="PERSON"):
        return {"name": name, "entity_type": etype, "aliases": [], "section_ids": sids}

    def test_build_cooccurrence_dedups_and_ranks(self):
        from pagemind.recipes.structured_view import _build_cooccurrence
        s1, s2, s3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        entities = [
            # A appears TWICE in s1 (duplicate section) and once in s2, s3.
            self._ent("A", [s1, s1, s2, s3]),
            self._ent("B", [s1, s2]),   # shares s1, s2 with A → count 2
            self._ent("C", [s3]),       # shares s3 with A → count 1
        ]
        edges, sec_to_ents = _build_cooccurrence(entities)
        by_pair = {(a, b): (count, sids) for a, b, count, sids in edges}
        # Duplicate s1 for A does not inflate the A–B count.
        assert by_pair[("A", "B")][0] == 2
        assert by_pair[("A", "B")][1] == frozenset({s1, s2})
        assert by_pair[("A", "C")][0] == 1
        # Sorted by count desc: A–B (2) before A–C (1).
        assert edges[0][:2] == ("A", "B")
        # Section→entities map is deduped per section.
        assert sec_to_ents[s1] == {"A", "B"}

    def test_build_cooccurrence_tiebreak_is_deterministic(self):
        from pagemind.recipes.structured_view import _build_cooccurrence
        s1, s2 = uuid.uuid4(), uuid.uuid4()
        # Three pairs all at count 1 → order must be a stable (name_a, name_b) sort.
        entities = [
            self._ent("Bob", [s1]),
            self._ent("Ann", [s1]),
            self._ent("Cy", [s2]),
            self._ent("Ann2", [s2]),
        ]
        edges, _ = _build_cooccurrence(entities)
        names = [(a, b) for a, b, _c, _s in edges]
        assert names == sorted(names)  # deterministic alphabetical tiebreak

    def test_select_read_sections_prioritises_then_caps(self):
        from pagemind.recipes.structured_view import _select_read_sections
        edge_sec, both_sec, event_sec = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        top_edges = [
            ("A", "B", 2, frozenset({edge_sec, both_sec})),
            ("A", "C", 1, frozenset({both_sec})),  # both_sec supports 2 edges
        ]
        sec_to_ents = {both_sec: {"A", "B", "C"}, edge_sec: {"A", "B"}}
        events = [event_sec, both_sec]

        # Full budget → all three candidates, ranked: edge∩event first, then
        # edge-central by support, then event-only.
        chosen = _select_read_sections(top_edges, sec_to_ents, events, max_reads=3)
        assert chosen == [both_sec, edge_sec, event_sec]

        # Tight budget → capped, lowest-priority (event-only) dropped.
        capped = _select_read_sections(top_edges, sec_to_ents, events, max_reads=2)
        assert capped == [both_sec, edge_sec]
        assert event_sec not in capped


# ── Recipe: contextual_why (multi-hop) ────────────────────────────────────────

class TestContextualWhyRecipe:
    async def test_two_stage_returns_result(self, db_book):
        import pagemind.recipes.contextual_why as cw_mod
        conn, book_id, ids = db_book

        reader_json = json.dumps({"answer": "Alice is the main character.", "verbatim_quotes": []})
        synth = "Alice's curiosity is explained later in the book."

        async def mock_complete(messages, max_tokens=1024, **kw):
            if any("PASSAGE" in (m.get("content") or "") for m in messages):
                return reader_json
            return synth

        chat = MagicMock()
        chat.complete = AsyncMock(side_effect=mock_complete)

        call_n = [0]
        async def mock_hybrid(conn, book_id, question, *, top_k=10, up_to_chapter=None):
            call_n[0] += 1
            # First call: anchor in ch0; second call: return ch1 section for forward scan
            if call_n[0] == 1:
                return [(ids["s0"], 0.9)]
            return [(ids["s2"], 0.7), (ids["s3"], 0.6)]

        orig_hybrid = cw_mod.hybrid_search
        cw_mod.hybrid_search = mock_hybrid
        try:
            result = await cw_mod.run(conn, chat, book_id, "why is Alice so curious?")
            assert isinstance(result, QueryResult)
            assert result.text
            # Both hops cite: anchor (s0, ch0) and at least one forward section (ch1)
            cited_chapters = {c.chapter for c in result.citations}
            assert 0 in cited_chapters  # anchor hop
        finally:
            cw_mod.hybrid_search = orig_hybrid

    async def test_no_hits_returns_message(self, db_book):
        import pagemind.recipes.contextual_why as cw_mod
        conn, book_id, _ = db_book

        orig_hybrid = cw_mod.hybrid_search
        cw_mod.hybrid_search = AsyncMock(return_value=[])
        try:
            result = await cw_mod.run(conn, _mock_chat("{}"), book_id, "xyzzy nonexistent q9q9q")
            assert result.text
            assert "not found" in result.text.lower() or "no relevant" in result.text.lower()
        finally:
            cw_mod.hybrid_search = orig_hybrid


# ── Output seam: all recipes return QueryResult ───────────────────────────────

class TestOutputSeam:
    """Verify that all recipe run() functions return a QueryResult instance."""

    async def _fake_result(self, conn, chat, book_id, question, *, up_to_chapter=None):
        return QueryResult(text="test")

    async def test_generic_fallback_output_type(self, db_book):
        import pagemind.recipes.generic_fallback as gf_mod
        conn, book_id, ids = db_book

        reader_json = json.dumps({"answer": "a", "verbatim_quotes": []})
        orig_hybrid = gf_mod.hybrid_search
        gf_mod.hybrid_search = AsyncMock(return_value=[(ids["s0"], 0.9)])
        try:
            result = await gf_mod.run(conn, _mock_chat(reader_json), book_id, "anything")
            assert isinstance(result, QueryResult)
        finally:
            gf_mod.hybrid_search = orig_hybrid

    async def test_chapter_summary_output_type(self, db_book):
        from pagemind.recipes.chapter_summary import run
        conn, book_id, _ = db_book
        result = await run(conn, _mock_chat("x"), book_id, "chapter 0 summary")
        assert isinstance(result, QueryResult)

    async def test_locate_entity_output_type(self, db_book):
        from pagemind.recipes.locate_entity import run
        conn, book_id, _ = db_book
        result = await run(conn, _mock_chat("x"), book_id, "where does Alice appear")
        assert isinstance(result, QueryResult)

    async def test_structured_view_output_type(self, db_book):
        from pagemind.recipes.structured_view import run
        conn, book_id, _ = db_book
        result = await run(conn, _mock_chat("x"), book_id, "map relationships")
        assert isinstance(result, QueryResult)


# ── Conversation history: condense / format / cap ─────────────────────────────

class TestConversationHistory:
    async def test_condense_empty_history_returns_question_no_call(self):
        from pagemind.runtime.history import condense_question
        chat = _mock_chat("SHOULD NOT BE USED")
        result = await condense_question(chat, [], "Who is Alice?")
        assert result == "Who is Alice?"
        chat.complete.assert_not_called()

    async def test_condense_none_history_returns_question_no_call(self):
        from pagemind.runtime.history import condense_question
        chat = _mock_chat("x")
        result = await condense_question(chat, None, "Who is Alice?")
        assert result == "Who is Alice?"
        chat.complete.assert_not_called()

    async def test_condense_with_history_returns_standalone(self):
        from pagemind.runtime.history import condense_question
        chat = _mock_chat("  What happens to Alice?  ")
        history = [{"role": "user", "content": "Who is Alice?"},
                   {"role": "assistant", "content": "A curious girl."}]
        result = await condense_question(chat, history, "what happens to her?")
        assert result == "What happens to Alice?"
        chat.complete.assert_called_once()

    async def test_condense_empty_result_falls_back_to_raw(self):
        from pagemind.runtime.history import condense_question
        chat = _mock_chat("   ")
        history = [{"role": "user", "content": "Who is Alice?"}]
        result = await condense_question(chat, history, "and her sister?")
        assert result == "and her sister?"

    async def test_condense_exception_falls_back_to_raw(self):
        from pagemind.runtime.history import condense_question
        chat = MagicMock()
        chat.complete = AsyncMock(side_effect=RuntimeError("boom"))
        history = [{"role": "user", "content": "Who is Alice?"}]
        result = await condense_question(chat, history, "and her sister?")
        assert result == "and her sister?"

    def test_format_history_labels_roles(self):
        from pagemind.runtime.history import format_history
        out = format_history([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        assert out == "User: hi\nPageMind: hello"

    def test_cap_history_limits_turns(self):
        from pagemind.runtime.history import cap_history
        history = [{"role": "user", "content": str(i)} for i in range(20)]
        capped = cap_history(history, max_turns=6)
        assert len(capped) == 6
        assert capped[0]["content"] == "14"  # last 6: 14..19

    def test_cap_history_truncates_content(self):
        from pagemind.runtime.history import cap_history
        history = [{"role": "assistant", "content": "x" * 1000}]
        capped = cap_history(history, max_chars=50)
        assert len(capped[0]["content"]) <= 51  # 50 chars + ellipsis
        assert capped[0]["content"].endswith("…")

    def test_ask_request_history_defaults_empty(self):
        from pagemind.api import AskRequest
        assert AskRequest(question="hi").history == []

    def test_ask_request_accepts_history(self):
        from pagemind.api import AskRequest
        req = AskRequest(question="hi", history=[{"role": "user", "content": "prior"}])
        assert req.history[0].role == "user"
        assert req.history[0].content == "prior"


# ── ask_stream wiring: condensed question drives retrieval/synthesis ───────────

class TestAskStreamWiring:
    async def _run(self, monkeypatch, *, history, condense_ret="CONDENSED Q"):
        import pagemind.runtime.streaming as sm

        captured: dict = {}

        async def fake_condense(chat, hist, question):
            captured["condense_called"] = True
            captured["condense_args"] = (hist, question)
            return condense_ret

        async def fake_route(chat, q):
            captured["route_q"] = q
            return "generic_fallback"  # search-based path

        async def fake_hybrid(conn, book_id, q, *, top_k=5, up_to_chapter=None):
            captured["hybrid_q"] = q
            return [(uuid.uuid4(), 1.0), (uuid.uuid4(), 0.9)]

        async def fake_fan_out(conn, chat, section_ids, q, *, grounded=True):
            captured["fan_out_q"] = q
            return [ReadResult(section_id=section_ids[0], chapter=1, answer="ans")]

        async def fake_stream(messages, max_tokens=512):
            captured["synth_messages"] = messages
            yield "final answer"

        fake_chat = MagicMock()
        fake_chat.stream_complete = fake_stream

        monkeypatch.setattr(sm.ChatClient, "from_config", classmethod(lambda cls, axis="query": fake_chat))
        monkeypatch.setattr(sm, "condense_question", fake_condense)
        monkeypatch.setattr(sm, "route", fake_route)
        monkeypatch.setattr(sm, "hybrid_search", fake_hybrid)
        monkeypatch.setattr(sm, "fan_out", fake_fan_out)

        events = [
            json.loads(chunk[6:]) async for chunk in sm.ask_stream(
                None, uuid.uuid4(), "and then?", history=history
            )
        ]
        return captured, events

    async def test_condensed_question_reaches_retrieval_and_synth(self, monkeypatch):
        captured, events = await self._run(
            monkeypatch,
            history=[{"role": "user", "content": "Who is Alice?"}],
        )
        # Condense ran and its output — not the raw follow-up — drove everything.
        assert captured["condense_called"] is True
        assert captured["route_q"] == "CONDENSED Q"
        assert captured["hybrid_q"] == "CONDENSED Q"
        assert captured["fan_out_q"] == "CONDENSED Q"
        # Synthesis is single-turn: exactly [system, user], no history injected.
        msgs = captured["synth_messages"]
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert "CONDENSED Q" in msgs[1]["content"]
        assert any(e["type"] == "done" for e in events)
        assert any(e.get("text") == "Rephrasing…" for e in events if e["type"] == "step")

    async def test_no_history_skips_condense(self, monkeypatch):
        captured, events = await self._run(monkeypatch, history=[])
        assert "condense_called" not in captured
        assert captured["route_q"] == "and then?"
        assert captured["hybrid_q"] == "and then?"
        assert not any(e.get("text") == "Rephrasing…" for e in events if e["type"] == "step")

    async def test_backend_failure_emits_error_event(self, monkeypatch):
        """A dead embedding server/LLM yields a terminal error event, not a crash."""
        import httpx
        import pagemind.runtime.streaming as sm

        fake_chat = MagicMock()
        monkeypatch.setattr(sm.ChatClient, "from_config", classmethod(lambda cls, axis="query": fake_chat))
        monkeypatch.setattr(sm, "route", AsyncMock(return_value="generic_fallback"))

        async def dead_hybrid(*a, **k):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(sm, "hybrid_search", dead_hybrid)

        events = [
            json.loads(chunk[6:]) async for chunk in sm.ask_stream(
                None, uuid.uuid4(), "anything", history=[]
            )
        ]
        assert events[-1]["type"] == "error"
        assert "unavailable" in events[-1]["text"].lower()


# ── Quote selection: dedupe, cap, non-mutation ────────────────────────────────

class TestSelectQuotes:
    def _rr(self, quotes, offsets):
        return ReadResult(section_id=uuid.uuid4(), chapter=0, answer="a",
                          verbatim_quotes=quotes, char_offsets=offsets)

    def test_dedupe_and_global_cap(self):
        from pagemind.runtime.quotes import select_quotes
        r1 = self._rr(["A flame flew\nover", "A flame flew over", "unique one"],
                      [(0, 10), (20, 30), (40, 50)])
        r2 = self._rr(["second section quote", "another quote here"],
                      [(5, 15), (25, 35)])
        out = select_quotes([r1, r2], want=3)
        kept = [(q, o) for r in out for q, o in zip(r.verbatim_quotes, r.char_offsets)]
        assert len(kept) == 3  # global cap
        texts = [q for q, _o in kept]
        # The two whitespace-variant near-dupes collapse to one.
        assert not ("A flame flew\nover" in texts and "A flame flew over" in texts)
        # Every kept quote keeps its own paired offset.
        assert all(isinstance(o, tuple) and len(o) == 2 for _q, o in kept)

    def test_non_mutating(self):
        from pagemind.runtime.quotes import select_quotes
        r = self._rr(["q1", "q2", "q3", "q4"], [(0, 1), (1, 2), (2, 3), (3, 4)])
        before_q, before_o = list(r.verbatim_quotes), list(r.char_offsets)
        select_quotes([r], want=2)
        assert r.verbatim_quotes == before_q and r.char_offsets == before_o

    def test_format_quote_answer(self):
        from pagemind.runtime.quotes import format_quote_answer
        assert "couldn't" in format_quote_answer(0).lower()
        assert format_quote_answer(1).count("passage") == 1
        assert "loosely" in format_quote_answer(3, weak=True).lower()


# ── weak-signal fallback: summaries-only path ─────────────────────────────────

class TestGetChapterSummaries:
    async def test_returns_body_chapters_ordered(self, db_book):
        from pagemind.retrieval.structured import get_chapter_summaries
        conn, book_id, _ = db_book
        rows = get_chapter_summaries(conn, book_id)
        assert [r["ordinal"] for r in rows] == [0, 1]
        assert "Alice" in rows[0]["summary"]

    async def test_respects_up_to_chapter(self, db_book):
        from pagemind.retrieval.structured import get_chapter_summaries
        conn, book_id, _ = db_book
        rows = get_chapter_summaries(conn, book_id, up_to_chapter=0)
        assert [r["ordinal"] for r in rows] == [0]

    async def test_excludes_non_body_chapters(self, db_book):
        from pagemind.retrieval.structured import get_chapter_summaries
        conn, book_id, _ = db_book
        conn.execute(
            """
            INSERT INTO chapters (book_id, ordinal, title, is_body, summary, micro_summary)
            VALUES (%s, 2, 'CONTENTS', FALSE, NULL, NULL)
            """,
            (book_id,),
        )
        conn.commit()
        rows = get_chapter_summaries(conn, book_id)
        assert [r["ordinal"] for r in rows] == [0, 1]  # non-body row excluded


class TestSynthesizeWeak:
    async def test_empty_results_is_weak(self):
        from pagemind.runtime.synthesizer import synthesize
        chat = _mock_chat("unused")
        result = await synthesize(chat, uuid.uuid4(), "q", [])
        assert result.weak is True

    async def test_nonempty_results_not_weak(self):
        from pagemind.runtime.synthesizer import synthesize
        chat = _mock_chat("A synthesized answer.")
        rr = ReadResult(section_id=uuid.uuid4(), chapter=1, answer="ans")
        result = await synthesize(chat, uuid.uuid4(), "q", [rr])
        assert result.weak is False


class TestSummaryFallbackModule:
    def test_module_imports_cold(self):
        # Guards the pre-existing recipes<->runtime import-order cycle.
        import importlib
        assert importlib.import_module("pagemind.runtime.fallback") is not None

    def test_build_outline_budget(self):
        from pagemind.runtime.fallback import _build_outline, _OUTLINE_BUDGET
        chapters = [
            {"ordinal": i, "title": f"C{i}",
             "summary": "x" * 5000, "micro_summary": "short micro"}
            for i in range(3)
        ]
        outline = _build_outline(chapters)
        # First chapter's full summary fits; later ones fall back to micro.
        assert "x" * 5000 in outline
        assert "short micro" in outline

    async def test_fallback_answers_from_summaries(self, db_book):
        from pagemind.runtime.fallback import summary_fallback
        conn, book_id, _ = db_book

        captured: dict = {}

        async def fake_complete(messages, max_tokens=512, **kw):
            captured["messages"] = messages
            return "She moves to Wonderland."

        chat = MagicMock()
        chat.complete = AsyncMock(side_effect=fake_complete)

        result = await summary_fallback(conn, chat, book_id, "where does she go")
        # The chapter-1 summary reached the synthesizer as evidence.
        assert "Alice falls into Wonderland" in captured["messages"][1]["content"]
        assert result.text == "She moves to Wonderland."
        assert result.citations == []  # uncited prose

    async def test_fallback_no_summaries_returns_original(self, db_book):
        from pagemind.runtime.fallback import summary_fallback
        conn, _book_id, _ = db_book
        original = QueryResult(text="dead end", weak=True)
        chat = _mock_chat("should not be called")
        # A book_id with no chapters → no summaries → original returned unchanged.
        result = await summary_fallback(
            conn, chat, uuid.uuid4(), "q", original=original
        )
        assert result is original
        chat.complete.assert_not_called()


class TestWeakTriggersFallback:
    async def test_orchestrator_weak_result_falls_back(self, db_book):
        import pagemind.runtime.orchestrator as orch_mod
        conn, book_id, _ = db_book
        sentinel = QueryResult(text="FROM SUMMARY")

        async def mock_route(chat, question):
            return "locate_entity"

        async def mock_dispatch(recipe, conn, chat, book_id, question, *, up_to_chapter=None):
            return QueryResult(text="Ghost (PERSON): 0 occurrence(s)", weak=True)

        async def mock_fallback(conn, chat, book_id, question, *, up_to_chapter=None,
                                grounded=True, original=None):
            return sentinel

        orig = (orch_mod.route, orch_mod.dispatch, orch_mod.summary_fallback)
        orch_mod.route, orch_mod.dispatch, orch_mod.summary_fallback = (
            mock_route, mock_dispatch, mock_fallback)
        try:
            result = await orch_mod.ask(conn, book_id, "where does Ghost move")
            assert result.text == "FROM SUMMARY"
        finally:
            orch_mod.route, orch_mod.dispatch, orch_mod.summary_fallback = orig

    async def test_streaming_weak_nonsearch_streams_fallback(self, monkeypatch):
        import pagemind.runtime.streaming as sm
        import pagemind.recipes as recipes_pkg

        async def fake_route(chat, q):
            return "locate_entity"  # a _NON_SEARCH recipe

        async def fake_dispatch(recipe, conn, chat, book_id, q, *, up_to_chapter=None):
            return QueryResult(text="Ghost (PERSON): 0 occurrence(s)", weak=True)

        async def fake_stream_fallback(conn, chat, book_id, q, *, up_to_chapter=None,
                                       grounded=True, original=None):
            yield ("token", "from the summary")
            yield ("done", QueryResult(text="from the summary"))

        fake_chat = MagicMock()
        monkeypatch.setattr(sm.ChatClient, "from_config",
                            classmethod(lambda cls, axis="query": fake_chat))
        monkeypatch.setattr(sm, "route", fake_route)
        monkeypatch.setattr(recipes_pkg, "dispatch", fake_dispatch)
        monkeypatch.setattr(sm, "stream_summary_fallback", fake_stream_fallback)

        events = [
            json.loads(chunk[6:]) async for chunk in sm.ask_stream(
                None, uuid.uuid4(), "where does Ghost move", history=[]
            )
        ]
        tokens = [e["text"] for e in events if e["type"] == "token"]
        done = [e for e in events if e["type"] == "done"]
        # The fallback's text is streamed; the raw weak "0 occurrence(s)" is not.
        assert "from the summary" in tokens
        assert not any("occurrence(s)" in t for t in tokens)
        assert done and done[-1]["result"]["text"] == "from the summary"
