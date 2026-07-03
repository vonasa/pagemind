"""FastAPI application — Phase 5 web API."""
from __future__ import annotations

import base64
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from pagemind.config import settings
from pagemind.runtime.streaming import ask_stream

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="PageMind", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# DB dependency (sync psycopg, one connection per request)
# ---------------------------------------------------------------------------

@contextmanager
def _db() -> Generator[psycopg.Connection, None, None]:
    with psycopg.connect(settings.database_url) as conn:
        yield conn


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class HistoryTurn(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    up_to_chapter: int | None = None
    grounded: bool = True
    history: list[HistoryTurn] = []


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /books
# ---------------------------------------------------------------------------

@app.get("/books")
async def list_books() -> list[dict]:
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT
                b.book_id,
                b.title,
                b.author,
                b.status,
                b.cover_mime,
                b.cover_data,
                COUNT(c.chapter_id) AS chapter_count
            FROM book_meta b
            LEFT JOIN chapters c ON c.book_id = b.book_id
            GROUP BY b.book_id
            ORDER BY b.created_at DESC
            """
        ).fetchall()

    books = []
    for row in rows:
        book_id, title, author, status, cover_mime, cover_data, chapter_count = row
        cover_url: str | None = None
        if cover_data and cover_mime:
            b64 = base64.b64encode(bytes(cover_data)).decode()
            cover_url = f"data:{cover_mime};base64,{b64}"
        books.append({
            "id": str(book_id),
            "title": title,
            "author": author,
            "status": status,
            "cover_url": cover_url,
            "chapter_count": chapter_count,
        })
    return books


# ---------------------------------------------------------------------------
# GET /books/{book_id}/summary
#
# Path is /summary (not /overview) on purpose: the client-side SPA route is
# /books/:id/overview, so an API route at that same path would shadow it on a hard
# refresh / direct navigation (the server would return this JSON instead of the SPA).
# ---------------------------------------------------------------------------

@app.get("/books/{book_id}/summary")
async def get_book_overview(book_id: uuid.UUID) -> dict:
    """Whole-book overview. Pure read — the summary is generated offline (precompute
    stage or `pagemind backfill-summaries`), never in the request path. `summary` is
    null until that has run."""
    with _db() as conn:
        row = conn.execute(
            "SELECT title, author, summary FROM book_meta WHERE book_id = %s",
            (book_id,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Book not found")

    title, author, summary = row
    return {
        "id": str(book_id),
        "title": title,
        "author": author,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# GET /books/{book_id}/chapters
# ---------------------------------------------------------------------------

@app.get("/books/{book_id}/chapters")
async def list_chapters(book_id: uuid.UUID) -> list[dict]:
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT chapter_id, ordinal, number, title, micro_summary, summary
            FROM chapters
            WHERE book_id = %s AND is_body
            ORDER BY ordinal
            """,
            (book_id,),
        ).fetchall()

    if not rows:
        # Check whether book exists at all
        with _db() as conn:
            exists = conn.execute(
                "SELECT 1 FROM book_meta WHERE book_id = %s", (book_id,)
            ).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="Book not found")

    return [
        {
            "id": str(r[0]),
            "ordinal": r[1],
            "number": r[2],
            "title": r[3] or f"Chapter {r[2]}",
            "micro_summary": r[4],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# GET /chapters/{chapter_id}/summary
# ---------------------------------------------------------------------------

@app.get("/chapters/{chapter_id}/summary")
async def get_chapter_summary(chapter_id: uuid.UUID) -> dict:
    with _db() as conn:
        row = conn.execute(
            "SELECT ordinal, number, title, summary, micro_summary FROM chapters WHERE chapter_id = %s",
            (chapter_id,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Chapter not found")

    ordinal, number, title, summary, micro_summary = row
    text = summary or micro_summary or "(No summary available for this chapter.)"
    return {
        "id": str(chapter_id),
        "ordinal": ordinal,
        "number": number,
        "title": title or f"Chapter {number}",
        "summary": text,
    }


# ---------------------------------------------------------------------------
# GET /sections/{section_id}
# ---------------------------------------------------------------------------

@app.get("/sections/{section_id}")
async def get_section(section_id: uuid.UUID) -> dict:
    with _db() as conn:
        row = conn.execute(
            """
            SELECT s.section_id, s.content, s.char_offset_start, s.char_offset_end,
                   c.number AS chapter_number, c.title AS chapter_title
            FROM sections s
            JOIN chapters c ON c.chapter_id = s.chapter_id
            WHERE s.section_id = %s
            """,
            (section_id,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Section not found")

    sid, content, offset_start, offset_end, ch_number, ch_title = row
    return {
        "id": str(sid),
        "content": content,
        "char_offset_start": offset_start,
        "char_offset_end": offset_end,
        "chapter": ch_number,
        "chapter_title": ch_title or f"Chapter {ch_number}",
    }


# ---------------------------------------------------------------------------
# POST /books/{book_id}/ask  →  SSE stream
# ---------------------------------------------------------------------------

@app.post("/books/{book_id}/ask")
async def ask(book_id: uuid.UUID, body: AskRequest):
    with _db() as conn:
        row = conn.execute(
            "SELECT status FROM book_meta WHERE book_id = %s", (book_id,)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Book not found")
    if row[0] != "ready":
        raise HTTPException(status_code=409, detail=f"Book is not ready (status={row[0]})")

    async def event_generator():
        with _db() as conn:
            async for chunk in ask_stream(
                conn,
                book_id,
                body.question,
                up_to_chapter=body.up_to_chapter,
                grounded=body.grounded,
                history=[t.model_dump() for t in body.history],
            ):
                # ask_stream yields "data: {...}\n\n" strings;
                # EventSourceResponse expects plain data values or dicts.
                # We strip the SSE framing and yield just the data line content
                # so sse_starlette can re-frame it.
                if chunk.startswith("data: "):
                    yield chunk[6:].strip()

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# Serve React SPA (Phase 5)
# ---------------------------------------------------------------------------

_DIST = Path(__file__).parent.parent.parent / "web" / "dist"

if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        # API routes take priority (mounted above); this catches everything else.
        index = _DIST / "index.html"
        if index.exists():
            return FileResponse(index)
        return {"error": "Frontend not built. Run: just build-web"}
else:
    from fastapi.responses import HTMLResponse

    _PLACEHOLDER = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PageMind</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 540px; margin: 5rem auto; color: #222; }
    h1   { font-size: 2rem; }
    a    { color: #0070f3; }
  </style>
</head>
<body>
  <h1>PageMind</h1>
  <p>Run <code>just build-web</code> to build the frontend, then <code>just serve</code>.</p>
  <p><a href="/docs">API docs →</a></p>
</body>
</html>
"""

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root() -> str:
        return _PLACEHOLDER
