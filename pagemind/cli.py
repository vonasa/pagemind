import asyncio
import sys
import uuid
from pathlib import Path


def _make_progress():
    """Build (on_stage, on_tick, finish) printers for run_precompute.

    Stage headlines print on their own line; per-unit ticks overwrite a single line
    in place when stdout is a TTY (and fall back to plain lines when piped, to avoid
    carriage-return litter in logs/CI).
    """
    interactive = sys.stdout.isatty()
    state = {"tick_open": False}

    def on_stage(msg: str) -> None:
        if state["tick_open"]:
            print()
            state["tick_open"] = False
        print(f"  {msg}")

    def on_tick(msg: str) -> None:
        if interactive:
            print(f"\r    {msg}", end="", flush=True)
            state["tick_open"] = True
        else:
            print(f"    {msg}")

    def finish() -> None:
        if state["tick_open"]:
            print()
            state["tick_open"] = False

    return on_stage, on_tick, finish


def _add(book_path_str: str, *, reindex: bool = False) -> None:
    from pagemind.ingest import parse_epub
    from pagemind.segment import segment_book, validate
    from pagemind.db import (
        get_conn,
        find_book_by_hash,
        create_book,
        clear_book_content,
        update_status,
        store_segments,
    )
    from pagemind.precompute import run_precompute

    path = Path(book_path_str)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    if path.suffix.lower() != ".epub":
        print(f"error: only .epub files are supported (got {path.suffix!r})", file=sys.stderr)
        sys.exit(1)

    # ── Parse ──────────────────────────────────────────────────────────────────
    print(f"Parsing {path.name} …")
    try:
        parsed = parse_epub(path)
    except Exception as exc:
        print(f"error: failed to parse EPUB: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  title:    {parsed.title}")
    print(f"  author:   {parsed.author or '(unknown)'}")
    print(f"  chapters: {len(parsed.chapters)} detected")

    with get_conn() as conn:
        # ── Idempotency / resume decision ────────────────────────────────────────
        resume = False
        existing = find_book_by_hash(conn, parsed.source_hash)
        if existing:
            book_id, status = existing
            if status == "ready" and not reindex:
                print(f"Book already compiled (id={book_id}). Use --reindex to force.")
                return
            if not reindex:
                # Segments commit before precompute, so any post-store interruption
                # (hard kill → 'indexing', or exception → 'failed') leaves chapters
                # intact. Their presence — not the status string — decides resume.
                has_chapters = conn.execute(
                    "SELECT EXISTS (SELECT 1 FROM chapters WHERE book_id = %s)",
                    (book_id,),
                ).fetchone()[0]
                if has_chapters:
                    print(f"Found incomplete ingestion (id={book_id}, status={status}); resuming …")
                    resume = True
            if not resume:
                # reindex, or an incomplete run with no stored segments → rebuild.
                if reindex:
                    print(f"Reindexing (id={book_id}); clearing existing content …")
                else:
                    print(f"Found incomplete ingestion (id={book_id}, status={status}); restarting …")
                clear_book_content(conn, book_id)
                update_status(conn, book_id, "ingesting")
                conn.commit()
        else:
            book_id = create_book(conn, parsed)
            conn.commit()

        # ── Segment / Validate / Store (skipped on resume) ───────────────────────
        if not resume:
            print("Segmenting …")
            try:
                result = segment_book(parsed)
            except Exception as exc:
                update_status(conn, book_id, "failed")
                conn.commit()
                print(f"error: segmentation failed: {exc}", file=sys.stderr)
                sys.exit(1)

            print(f"  sections: {len(result.sections)}")
            print(f"  chunks:   {len(result.chunks)}")

            print("Validating …")
            passed, issues = validate(result, parsed.raw_length)
            if not passed:
                update_status(conn, book_id, "failed")
                conn.commit()
                print("Validation failed:", file=sys.stderr)
                for issue in issues:
                    print(f"  • {issue}", file=sys.stderr)
                sys.exit(1)

            print("Storing …")
            try:
                with conn.transaction():
                    store_segments(conn, book_id, result)
                    update_status(conn, book_id, "indexing")
            except Exception as exc:
                update_status(conn, book_id, "failed")
                conn.commit()
                print(f"error: database write failed: {exc}", file=sys.stderr)
                sys.exit(1)
        else:
            update_status(conn, book_id, "indexing")
            conn.commit()

        # ── Precompute (resumes from the checkpoint ledger) ──────────────────────
        print("Precomputing …")
        on_stage, on_tick, finish = _make_progress()
        try:
            asyncio.run(
                run_precompute(conn, book_id, on_stage=on_stage, on_tick=on_tick)
            )
        except Exception as exc:
            finish()
            update_status(conn, book_id, "failed")
            conn.commit()
            print(f"error: precompute failed: {exc}", file=sys.stderr)
            sys.exit(1)
        finish()

        update_status(conn, book_id, "ready")
        conn.commit()

        # ── Done — counts from the DB (works for both fresh and resumed runs) ────
        n_ch, n_sec, n_chunk = conn.execute(
            """
            SELECT
              (SELECT count(*) FROM chapters WHERE book_id = %(b)s),
              (SELECT count(*) FROM sections WHERE book_id = %(b)s),
              (SELECT count(*) FROM chunks   WHERE book_id = %(b)s)
            """,
            {"b": book_id},
        ).fetchone()

    print(f"\nDone. book_id={book_id}")
    print(f"  {n_ch} chapters  |  {n_sec} sections  |  {n_chunk} chunks")


def _ask(book_id_str: str, question: str) -> None:
    from pagemind.db import get_conn
    from pagemind.runtime import ask

    try:
        book_id = uuid.UUID(book_id_str)
    except ValueError:
        print(f"error: invalid book_id {book_id_str!r} (must be a UUID)", file=sys.stderr)
        sys.exit(1)

    async def _run():
        with get_conn() as conn:
            return await ask(conn, book_id, question)

    result = asyncio.run(_run())
    print(result.text)
    if result.quotes:
        print()
        for q in result.quotes:
            cit = q.citation
            print(f'  "{q.text}"')
            print(f"  — ch.{cit.chapter}, section {cit.section_id}")


def _backfill_summaries() -> None:
    """Generate the whole-book summary for ready books that lack one.

    For already-ingested books (compiled before the book_summary stage existed):
    reuses the same offline generation the precompute pipeline runs, so no re-ingest
    is needed.
    """
    from pagemind.db import get_conn
    from pagemind.precompute.summaries import generate_book_summary

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT book_id, title FROM book_meta "
            "WHERE status = 'ready' AND summary IS NULL "
            "ORDER BY created_at"
        ).fetchall()

        if not rows:
            print("No ready books are missing a summary.")
            return

        print(f"Generating book summaries for {len(rows)} book(s) …")
        for book_id, title in rows:
            print(f"  {title} ({book_id}) …", flush=True)
            try:
                asyncio.run(generate_book_summary(conn, book_id))
            except Exception as exc:
                print(f"    error: {exc}", file=sys.stderr)
        print("Done.")


def _reindex_entities(book_id_str: str | None = None) -> None:
    """Re-run only the entities stage, reusing the cached per-section payloads.

    The entities stage does per-section NER (the expensive LLM pass) once and stores
    each section's extraction durably in the checkpoint ledger. Re-materialising just
    the back half (alias clustering + occurrence/date/event build) therefore costs a
    single clustering call, not a full re-ingest. Used to apply materialisation-side
    fixes — e.g. offset-recovery for occurrences/dates — to already-compiled books.

    Deleting the entities '*' sentinel makes run_precompute re-run that stage; it
    reloads the payloads (front half skipped) and extract_entities atomically rebuilds
    entities/occurrences/dates/events. Other stages keep their sentinels and are skipped.
    """
    from pagemind.db import get_conn
    from pagemind.precompute import run_precompute

    with get_conn() as conn:
        if book_id_str is not None:
            try:
                targets = [(uuid.UUID(book_id_str), None)]
            except ValueError:
                print(f"error: invalid book_id {book_id_str!r} (must be a UUID)", file=sys.stderr)
                sys.exit(1)
        else:
            targets = conn.execute(
                "SELECT book_id, title FROM book_meta WHERE status = 'ready' ORDER BY created_at"
            ).fetchall()

        if not targets:
            print("No ready books to reindex.")
            return

        for book_id, title in targets:
            # Guard: payloads must exist, or this would silently produce nothing
            # (the front half would have no cached extractions to reuse).
            n_payloads = conn.execute(
                "SELECT count(*) FROM precompute_checkpoints "
                "WHERE book_id = %s AND stage = 'entities' AND unit_key <> '*'",
                (book_id,),
            ).fetchone()[0]
            if n_payloads == 0:
                print(f"  {title or book_id}: no cached entity payloads — use `add --reindex` instead. Skipping.")
                continue

            print(f"Reindexing entities: {title or book_id} ({book_id}) …")
            conn.execute(
                "DELETE FROM precompute_checkpoints "
                "WHERE book_id = %s AND stage = 'entities' AND unit_key = '*'",
                (book_id,),
            )
            conn.commit()

            on_stage, on_tick, finish = _make_progress()
            try:
                asyncio.run(run_precompute(conn, book_id, on_stage=on_stage, on_tick=on_tick))
            except Exception as exc:
                finish()
                print(f"    error: {exc}", file=sys.stderr)
                continue
            finish()

            n_occ, n_dates = conn.execute(
                """
                SELECT (SELECT count(*) FROM occurrences WHERE book_id = %(b)s),
                       (SELECT count(*) FROM dates       WHERE book_id = %(b)s)
                """,
                {"b": book_id},
            ).fetchone()
            print(f"    {n_occ} occurrences  |  {n_dates} dates")
        print("Done.")


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: pagemind <command> [args...]")
        print("commands: add, ask, backfill-summaries, reindex-entities")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "add":
        args = sys.argv[2:]
        reindex = "--reindex" in args
        positionals = [a for a in args if not a.startswith("--")]
        if not positionals:
            print("usage: pagemind add <path-to-book.epub> [--reindex]")
            sys.exit(1)
        _add(positionals[0], reindex=reindex)
    elif cmd == "ask":
        if len(sys.argv) < 4:
            print('usage: pagemind ask <book-id> "question"')
            sys.exit(1)
        _ask(sys.argv[2], " ".join(sys.argv[3:]))
    elif cmd == "backfill-summaries":
        _backfill_summaries()
    elif cmd == "reindex-entities":
        positionals = [a for a in sys.argv[2:] if not a.startswith("--")]
        _reindex_entities(positionals[0] if positionals else None)
    else:
        print(f"unknown command: {cmd!r}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
