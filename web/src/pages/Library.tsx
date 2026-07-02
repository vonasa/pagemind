import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Loader2, Library as LibraryIcon, AlertCircle } from "lucide-react";
import { fetchBooks, type Book } from "../api";
import AppShell from "../components/ui/AppShell";
import Card from "../components/ui/Card";
import Badge, { statusBadge } from "../components/ui/Badge";

const ACTIVE_STATUSES = new Set(["ingesting", "embedding", "indexing"]);

// Deterministic warm gradient from the title, for books without a cover image.
// Hue is constrained to a warm band (20–60°) and lightness kept high so the
// dark-ink serif initial overlaid on top always stays readable (~5:1+).
function coverGradient(title: string): string {
  let hash = 0;
  for (let i = 0; i < title.length; i++) hash = (hash * 31 + title.charCodeAt(i)) >>> 0;
  const hue = 20 + (hash % 40);
  return `linear-gradient(145deg, hsl(${hue} 45% 82%), hsl(${(hue + 348) % 360} 40% 72%))`;
}

function BookCard({ book, onClick }: { book: Book; onClick?: () => void }) {
  const isReady = book.status === "ready";
  const isActive = ACTIVE_STATUSES.has(book.status);
  const { label, variant } = statusBadge(book.status);
  const initial = book.title.trim().charAt(0).toUpperCase() || "?";

  return (
    <Card
      interactive={isReady}
      onClick={isReady ? onClick : undefined}
      className={cnCard(isReady)}
    >
      {/* Cover */}
      <div
        className="relative h-44 flex items-center justify-center shrink-0"
        style={
          book.cover_url
            ? { background: `url(${book.cover_url}) center/cover` }
            : { background: coverGradient(book.title) }
        }
      >
        {!book.cover_url && (
          <span
            className="font-display font-semibold text-ink/70 select-none"
            style={{ fontSize: 72 }}
          >
            {initial}
          </span>
        )}
        {isActive && (
          <div className="absolute inset-x-0 bottom-0 h-1 bg-black/10">
            <div className="pulse h-full w-3/5 rounded-sm bg-warning" />
          </div>
        )}
      </div>

      {/* Info */}
      <div className="p-4 flex-1 flex flex-col gap-1.5">
        <div className="flex justify-between items-start gap-2">
          <h3 className="font-display text-[17px] font-semibold leading-snug text-ink flex-1">
            {book.title}
          </h3>
          <Badge variant={variant}>{label}</Badge>
        </div>
        {book.author && <p className="text-[13px] text-muted">{book.author}</p>}
        <p className="text-xs text-muted mt-auto pt-1">
          {book.chapter_count} chapter{book.chapter_count !== 1 ? "s" : ""}
        </p>
      </div>
    </Card>
  );
}

// Keep dimmed, non-interactive look for books that aren't ready yet.
function cnCard(isReady: boolean): string {
  return isReady ? "flex flex-col" : "flex flex-col opacity-75";
}

export default function Library() {
  const [books, setBooks] = useState<Book[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const data = await fetchBooks();
        if (!cancelled) { setBooks(data); setLoading(false); }
      } catch (e) {
        if (!cancelled) { setError(String(e)); setLoading(false); }
      }
    };
    load();

    // Poll while any book is still compiling
    const interval = setInterval(async () => {
      if (cancelled) return;
      const hasActive = books.some(b => ACTIVE_STATUSES.has(b.status));
      if (!hasActive) return;
      try {
        const data = await fetchBooks();
        if (!cancelled) setBooks(data);
      } catch { /* ignore */ }
    }, 3000);

    return () => { cancelled = true; clearInterval(interval); };
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <AppShell>
      <div className="mb-7">
        <h1 className="font-display text-3xl font-semibold text-ink tracking-tight">Library</h1>
        <p className="text-muted mt-1">Everything you've compiled, ready to explore.</p>
      </div>

      {loading && (
        <div className="flex items-center gap-2.5 text-muted">
          <Loader2 size={18} className="animate-spin" /> Loading library…
        </div>
      )}

      {error && (
        <div className="flex items-start gap-2 text-danger bg-danger/10 border border-danger/40 rounded-[var(--radius)] p-4">
          <AlertCircle size={18} className="mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {!loading && !error && books.length === 0 && (
        <div className="text-muted text-center py-20 flex flex-col items-center gap-4">
          <LibraryIcon size={44} className="text-edge" />
          <p className="font-display text-lg text-ink">No books yet</p>
          <p>Add one with:</p>
          <code className="block px-4 py-2 bg-sand border border-line rounded-[var(--radius)] text-[13px] text-ink">
            just compile path/to/book.epub
          </code>
        </div>
      )}

      {!loading && !error && books.length > 0 && (
        <div className="grid gap-5" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))" }}>
          {books.map(book => (
            <BookCard
              key={book.id}
              book={book}
              onClick={() => navigate(`/books/${book.id}/chat`)}
            />
          ))}
        </div>
      )}
    </AppShell>
  );
}
