import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { ChevronRight } from "lucide-react";
import {
  fetchBooks, fetchChapters, fetchChapterSummary, fetchBookOverview,
  type Book, type Chapter, type ChapterSummary, type BookOverview,
} from "../api";
import BookHeader from "../components/BookHeader";

// Per-chapter fetch state for the accordion.
type SummaryState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "ready"; summary: ChapterSummary };

export default function Overview() {
  const { bookId } = useParams<{ bookId: string }>();
  const navigate = useNavigate();

  const [book, setBook] = useState<Book | null>(null);
  const [overview, setOverview] = useState<BookOverview | null>(null);
  const [overviewError, setOverviewError] = useState<string | null>(null);
  const [overviewLoading, setOverviewLoading] = useState(true);

  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [chaptersLoading, setChaptersLoading] = useState(true);

  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [summaries, setSummaries] = useState<Record<string, SummaryState>>({});

  // Load book meta, overview, and chapters.
  useEffect(() => {
    if (!bookId) return;
    fetchBooks().then(books => {
      const found = books.find(b => b.id === bookId);
      if (!found) { navigate("/"); return; }
      setBook(found);
    });
    fetchBookOverview(bookId)
      .then(o => { setOverview(o); setOverviewLoading(false); })
      .catch(e => { setOverviewError(String(e)); setOverviewLoading(false); });
    fetchChapters(bookId)
      .then(chs => { setChapters(chs); setChaptersLoading(false); })
      .catch(() => setChaptersLoading(false));
  }, [bookId]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggleChapter = (ch: Chapter) => {
    setExpanded(prev => {
      const next = new Set(prev);
      if (next.has(ch.id)) { next.delete(ch.id); return next; }
      next.add(ch.id);
      // Fetch the full summary on first expand.
      if (!summaries[ch.id]) {
        setSummaries(s => ({ ...s, [ch.id]: { status: "loading" } }));
        fetchChapterSummary(ch.id)
          .then(summary => setSummaries(s => ({ ...s, [ch.id]: { status: "ready", summary } })))
          .catch(e => setSummaries(s => ({ ...s, [ch.id]: { status: "error", message: String(e) } })));
      }
      return next;
    });
  };

  if (!book) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100dvh", gap: 10, color: "var(--text-muted)" }}>
        <span className="spinner" /> Loading book…
      </div>
    );
  }

  const notReady = book.status !== "ready";

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100dvh" }}>
      <BookHeader bookId={book.id} title={book.title} author={book.author} active="overview" />

      <div style={{ flex: 1, overflowY: "auto" }}>
        <div style={{ maxWidth: 760, margin: "0 auto", padding: "24px 20px 48px" }}>
          {/* ── Book summary ─────────────────────────────────────────── */}
          <section style={{ marginBottom: 36 }}>
            <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 10 }}>
              Book summary
            </div>

            {notReady && (
              <p style={{ color: "var(--text-muted)", fontSize: 14 }}>
                This book is still being prepared ({book.status}). Its summary will be available once it's ready.
              </p>
            )}

            {!notReady && overviewLoading && (
              <div style={{ display: "flex", gap: 8, color: "var(--text-muted)", fontSize: 14 }}>
                <span className="spinner" /> Loading summary…
              </div>
            )}

            {!notReady && overviewError && (
              <p style={{ color: "var(--danger)", fontSize: 14 }}>{overviewError}</p>
            )}

            {!notReady && !overviewLoading && !overviewError && overview && (
              overview.summary
                ? <p style={{ fontSize: 15, lineHeight: 1.75, whiteSpace: "pre-wrap" }}>{overview.summary}</p>
                : (
                  <p style={{ color: "var(--text-muted)", fontSize: 14, lineHeight: 1.7 }}>
                    Summary not generated yet. Run <code style={{ background: "var(--surface2)", padding: "1px 6px", borderRadius: 4 }}>just backfill-summaries</code> to generate it.
                  </p>
                )
            )}
          </section>

          {/* ── Chapters ─────────────────────────────────────────────── */}
          <section>
            <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--text-muted)", marginBottom: 10 }}>
              Chapters
            </div>

            {chaptersLoading && (
              <div style={{ display: "flex", gap: 8, color: "var(--text-muted)", fontSize: 14 }}>
                <span className="spinner" /> Loading chapters…
              </div>
            )}

            {!chaptersLoading && chapters.length === 0 && (
              <p style={{ color: "var(--text-muted)", fontSize: 14 }}>No chapters found.</p>
            )}

            <div style={{ border: "1px solid var(--border-strong)", borderRadius: "var(--radius)", overflow: "hidden" }}>
              {chapters.map((ch, i) => {
                const isOpen = expanded.has(ch.id);
                const state = summaries[ch.id];
                return (
                  <div key={ch.id} style={{ borderTop: i === 0 ? "none" : "1px solid var(--border-strong)" }}>
                    <button
                      onClick={() => toggleChapter(ch)}
                      aria-expanded={isOpen}
                      style={{
                        width: "100%",
                        textAlign: "left",
                        padding: "12px 16px",
                        display: "flex",
                        alignItems: "flex-start",
                        gap: 12,
                        background: isOpen ? "var(--surface2)" : "var(--surface)",
                        transition: "background 0.12s",
                      }}
                      onMouseEnter={e => { if (!isOpen) (e.currentTarget).style.background = "var(--surface2)"; }}
                      onMouseLeave={e => { if (!isOpen) (e.currentTarget).style.background = "var(--surface)"; }}
                    >
                      <span style={{ color: "var(--text-muted)", display: "inline-flex", transform: isOpen ? "rotate(90deg)" : "none", transition: "transform 0.12s", marginTop: 2 }}>
                        <ChevronRight size={16} />
                      </span>
                      <span style={{ flex: 1, minWidth: 0 }}>
                        <span style={{ display: "block", fontFamily: "var(--font-display)", fontSize: 15, fontWeight: 600, color: "var(--text)" }}>
                          Ch. {ch.ordinal} — {ch.title}
                        </span>
                        {ch.micro_summary && (
                          <span style={{ display: "block", fontSize: 13, color: "var(--text-muted)", marginTop: 3, lineHeight: 1.5 }}>
                            {ch.micro_summary}
                          </span>
                        )}
                      </span>
                    </button>

                    {isOpen && (
                      <div style={{ padding: "4px 16px 16px 40px", fontSize: 14, lineHeight: 1.7 }}>
                        {(!state || state.status === "loading") && (
                          <div style={{ display: "flex", gap: 8, color: "var(--text-muted)" }}>
                            <span className="spinner" style={{ width: 14, height: 14 }} /> Loading…
                          </div>
                        )}
                        {state?.status === "error" && (
                          <p style={{ color: "var(--danger)" }}>{state.message}</p>
                        )}
                        {state?.status === "ready" && (
                          <p style={{ whiteSpace: "pre-wrap" }}>{state.summary.summary}</p>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
