// API client for PageMind backend

export interface Book {
  id: string;
  title: string;
  author: string | null;
  status: "ingesting" | "embedding" | "ready" | "failed" | string;
  cover_url: string | null;
  chapter_count: number;
}

export interface Chapter {
  id: string;
  ordinal: number;
  // 1-based position among body chapters — the number shown to the user and used
  // when referencing a chapter in chat. `ordinal` is the internal storage index.
  number: number;
  title: string;
  micro_summary: string | null;
}

export interface ChapterSummary {
  id: string;
  ordinal: number;
  number: number;
  title: string;
  summary: string;
}

export interface BookOverview {
  id: string;
  title: string;
  author: string | null;
  // Null until the book-level summary has been generated (precompute stage or
  // `pagemind backfill-summaries`).
  summary: string | null;
}

export interface Section {
  id: string;
  content: string;
  char_offset_start: number | null;
  char_offset_end: number | null;
  chapter: number;
  chapter_title: string;
}

export interface Citation {
  book_id: string;
  chapter: number;
  section_id: string;
  char_offset: number | null;
}

export interface Quote {
  text: string;
  citation: Citation;
}

export interface QueryResult {
  text: string;
  quotes: Quote[];
  citations: Citation[];
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export type SseEvent =
  | { type: "step"; text: string }
  | { type: "token"; text: string }
  | { type: "done"; result: QueryResult }
  | { type: "error"; text: string };

// ── REST calls ──────────────────────────────────────────────────────────────

export async function fetchBooks(): Promise<Book[]> {
  const r = await fetch("/books");
  if (!r.ok) throw new Error(`GET /books failed: ${r.status}`);
  return r.json();
}

export async function fetchChapters(bookId: string): Promise<Chapter[]> {
  const r = await fetch(`/books/${bookId}/chapters`);
  if (!r.ok) throw new Error(`GET /books/${bookId}/chapters failed: ${r.status}`);
  return r.json();
}

export async function fetchChapterSummary(chapterId: string): Promise<ChapterSummary> {
  const r = await fetch(`/chapters/${chapterId}/summary`);
  if (!r.ok) throw new Error(`GET /chapters/${chapterId}/summary failed: ${r.status}`);
  return r.json();
}

export async function fetchBookOverview(bookId: string): Promise<BookOverview> {
  // Endpoint is /summary, not /overview: /overview is the client-side SPA route, so an
  // API path there would shadow it on a hard refresh.
  const r = await fetch(`/books/${bookId}/summary`);
  if (!r.ok) throw new Error(`GET /books/${bookId}/summary failed: ${r.status}`);
  return r.json();
}

export async function fetchSection(sectionId: string): Promise<Section> {
  const r = await fetch(`/sections/${sectionId}`);
  if (!r.ok) throw new Error(`GET /sections/${sectionId} failed: ${r.status}`);
  return r.json();
}

// ── SSE streaming ask ───────────────────────────────────────────────────────

export function askStream(
  bookId: string,
  question: string,
  onEvent: (ev: SseEvent) => void,
  upToChapter?: number,
  grounded: boolean = true,
  history: ChatTurn[] = []
): () => void {
  let cancelled = false;
  const controller = new AbortController();

  (async () => {
    try {
      const resp = await fetch(`/books/${bookId}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, up_to_chapter: upToChapter ?? null, grounded, history }),
        signal: controller.signal,
      });
      if (!resp.ok) {
        const err = await resp.text();
        onEvent({ type: "error", text: `Server error ${resp.status}: ${err}` });
        return;
      }
      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (!cancelled) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed === "data:") continue;
          const data = trimmed.startsWith("data: ") ? trimmed.slice(6) : trimmed;
          try {
            onEvent(JSON.parse(data) as SseEvent);
          } catch {
            // ignore malformed lines
          }
        }
      }
    } catch (err: unknown) {
      if (!cancelled) {
        onEvent({ type: "error", text: String(err) });
      }
    }
  })();

  return () => {
    cancelled = true;
    controller.abort();
  };
}
