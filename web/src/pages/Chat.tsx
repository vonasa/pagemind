import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { MessageSquare, Info } from "lucide-react";
import {
  fetchBooks,
  askStream,
  type Book, type QueryResult, type SseEvent, type ChatTurn,
} from "../api";
import QuoteCard from "../components/QuoteCard";
import BookHeader from "../components/BookHeader";

// ── Per-book conversation persistence (localStorage) ──────────────────────────

const STORAGE_PREFIX = "pagemind:chat:";
const HISTORY_MAX = 6; // turns of context sent to the backend

// Slimmed shape we persist — drop bulky quote payloads and transient fields.
interface StoredMessage {
  id: number;
  role: MessageRole;
  text: string;
  grounded?: boolean;
}

function loadStored(bookId: string): Message[] {
  try {
    const raw = localStorage.getItem(STORAGE_PREFIX + bookId);
    if (!raw) return [];
    const arr = JSON.parse(raw) as StoredMessage[];
    if (!Array.isArray(arr)) return [];
    return arr
      // Conversations persisted by an earlier version may contain injected "chapter"
      // messages; that role no longer exists, so drop them.
      .filter(m => (m.role as string) !== "chapter")
      .map(m => ({ id: m.id, role: m.role, text: m.text, grounded: m.grounded }));
  } catch {
    return [];
  }
}

function saveStored(bookId: string, msgs: Message[]) {
  try {
    const slim: StoredMessage[] = msgs
      .filter(m => !m.streaming)
      .map(m => ({ id: m.id, role: m.role, text: m.text, grounded: m.grounded }));
    localStorage.setItem(STORAGE_PREFIX + bookId, JSON.stringify(slim));
  } catch {
    // Quota or serialization error — persistence is best-effort, never fatal.
  }
}

// ── Types ────────────────────────────────────────────────────────────────────

type MessageRole = "user" | "assistant";

interface Message {
  id: number;
  role: MessageRole;
  text: string;
  result?: QueryResult;
  // For streaming: partial text being built
  streaming?: boolean;
  status?: string;
  // False when this answer was produced with "Grounded mode" off (may include
  // the model's own knowledge of the book). Undefined for grounded/other roles.
  grounded?: boolean;
}

let _msgId = 0;
function nextId() { return ++_msgId; }

// ── Quick action chips ────────────────────────────────────────────────────────

const QUICK_ACTIONS = [
  "Who are the main characters?",
  "Find quotes about love and betrayal",
  "How are the characters connected?",
  "How does the protagonist change over the story?",
];

// ── Message bubble ────────────────────────────────────────────────────────────

function MessageBubble({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";

  return (
    <div
      style={{
        padding: "16px 20px",
        display: "flex",
        flexDirection: "column",
        alignItems: isUser ? "flex-end" : "flex-start",
        gap: 10,
      }}
    >
      {/* Role label */}
      <span style={{ fontSize: 11, color: "var(--text-muted)", fontWeight: 600, letterSpacing: "0.05em" }}>
        {isUser ? "You" : "PageMind"}
      </span>

      {/* Bubble */}
      <div
        style={{
          maxWidth: "80%",
          background: isUser ? "var(--accent)" : "var(--surface2)",
          color: isUser ? "#fff" : "var(--text)",
          borderRadius: "var(--radius)",
          padding: "10px 14px",
          fontSize: 14,
          lineHeight: 1.7,
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}
      >
        {/* Status line while streaming */}
        {msg.streaming && msg.status && (
          <div style={{ color: isUser ? "#ffffffaa" : "var(--text-muted)", fontSize: 12, marginBottom: msg.text ? 8 : 0, display: "flex", alignItems: "center", gap: 6 }}>
            <span className="spinner" style={{ width: 12, height: 12, borderWidth: 1.5 }} />
            {msg.status}
          </div>
        )}
        {msg.text}
        {msg.streaming && !msg.status && !msg.text && (
          <span className="spinner" style={{ width: 12, height: 12, borderWidth: 1.5 }} />
        )}
      </div>

      {/* Provenance disclaimer for ungrounded answers */}
      {!isUser && msg.grounded === false && !msg.streaming && (
        <div
          style={{
            maxWidth: "80%",
            fontSize: 12,
            color: "var(--text-muted)",
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <Info size={13} aria-hidden style={{ flexShrink: 0 }} />
          May include the model's own knowledge of this book, beyond the cited passages.
        </div>
      )}

      {/* Quote cards */}
      {msg.result && msg.result.quotes.length > 0 && (
        <div style={{ maxWidth: "80%", display: "flex", flexDirection: "column", gap: 8, width: "100%" }}>
          {msg.result.quotes.map((q, i) => (
            <QuoteCard key={i} quote={q} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Main Chat page ───────────────────────────────────────────────────────────

export default function Chat() {
  const { bookId } = useParams<{ bookId: string }>();
  const navigate = useNavigate();

  const [book, setBook] = useState<Book | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [grounded, setGrounded] = useState(true);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const cancelRef = useRef<(() => void) | null>(null);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Load book, and hydrate any persisted conversation for this book.
  useEffect(() => {
    if (!bookId) return;
    const stored = loadStored(bookId);
    if (stored.length) {
      // _msgId is a shared SPA-lifetime singleton — only ever advance it, so
      // restored ids can't collide with new ones after a book switch.
      _msgId = Math.max(_msgId, ...stored.map(m => m.id));
    }
    setMessages(stored);
    fetchBooks().then(books => {
      const found = books.find(b => b.id === bookId);
      if (!found) { navigate("/"); return; }
      setBook(found);
    });
  }, [bookId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Submit question
  const handleSubmit = useCallback((question: string) => {
    if (!question.trim() || streaming || !bookId) return;

    // Prior conversation → history for the backend condense step. Keep only real
    // user/assistant turns, dropping empties and error turns — an "Error: …" string
    // would otherwise pollute the rewritten question.
    const history: ChatTurn[] = messages
      .filter(m =>
        (m.role === "user" || m.role === "assistant") &&
        m.text.trim() !== "" &&
        !m.text.startsWith("Error:")
      )
      .map(m => ({ role: m.role as "user" | "assistant", content: m.text }))
      .slice(-HISTORY_MAX);

    setInput("");
    const userMsg: Message = { id: nextId(), role: "user", text: question.trim() };
    const assistantMsg: Message = { id: nextId(), role: "assistant", text: "", streaming: true, status: "", grounded };

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setStreaming(true);

    const assistantId = assistantMsg.id;

    const cancel = askStream(
      bookId,
      question.trim(),
      (ev: SseEvent) => {
        if (ev.type === "step") {
          setMessages(prev => prev.map(m =>
            m.id === assistantId ? { ...m, status: ev.text } : m
          ));
        } else if (ev.type === "token") {
          setMessages(prev => prev.map(m =>
            m.id === assistantId ? { ...m, text: m.text + ev.text, status: "" } : m
          ));
        } else if (ev.type === "done") {
          setMessages(prev => {
            const next = prev.map(m =>
              m.id === assistantId
                ? { ...m, text: ev.result.text, result: ev.result, streaming: false, status: undefined }
                : m
            );
            saveStored(bookId, next);
            return next;
          });
          setStreaming(false);
          cancelRef.current = null;
        } else if (ev.type === "error") {
          setMessages(prev => {
            const next = prev.map(m =>
              m.id === assistantId
                ? { ...m, text: `Error: ${ev.text}`, streaming: false, status: undefined }
                : m
            );
            saveStored(bookId, next);
            return next;
          });
          setStreaming(false);
          cancelRef.current = null;
        }
      },
      undefined,
      grounded,
      history,
    );
    cancelRef.current = cancel;
  }, [bookId, streaming, grounded, messages]);

  // Clear the conversation and start fresh
  const handleNewChat = useCallback(() => {
    cancelRef.current?.();
    cancelRef.current = null;
    setStreaming(false); // aborted stream won't fire "done", so re-enable input
    setMessages([]);
    setInput("");
    if (bookId) {
      try { localStorage.removeItem(STORAGE_PREFIX + bookId); } catch { /* ignore */ }
    }
  }, [bookId]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(input);
    }
  };

  if (!book) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100dvh", gap: 10, color: "var(--text-muted)" }}>
        <span className="spinner" /> Loading book…
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100dvh" }}>
      <BookHeader bookId={book.id} title={book.title} author={book.author} active="chat">
        <button
          onClick={handleNewChat}
          disabled={messages.length === 0}
          title="Clear the conversation and start fresh"
          style={{
            border: "1px solid var(--border)",
            borderRadius: "var(--radius)",
            padding: "6px 12px",
            fontSize: 13,
            fontWeight: 600,
            color: messages.length === 0 ? "var(--text-muted)" : "var(--text)",
            background: "var(--surface2)",
            cursor: messages.length === 0 ? "not-allowed" : "pointer",
            opacity: messages.length === 0 ? 0.5 : 1,
          }}
        >
          New chat
        </button>
      </BookHeader>

      {/* Body: full-width chat */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {/* Chat area */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          {/* Messages */}
          <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column" }}>
            {messages.length === 0 && (
              <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16, color: "var(--text-muted)", padding: 32 }}>
                <MessageSquare size={40} strokeWidth={1.5} style={{ color: "var(--border-strong)" }} />
                <p style={{ textAlign: "center", maxWidth: 340 }}>
                  Ask anything about <strong style={{ color: "var(--text)" }}>{book.title}</strong>.
                  Open the <strong style={{ color: "var(--text)" }}>Overview</strong> tab for the book and chapter summaries.
                </p>
              </div>
            )}
            {messages.map(msg => (
              <MessageBubble key={msg.id} msg={msg} />
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Quick action chips */}
          {messages.length === 0 && (
            <div style={{ padding: "0 16px 12px", display: "flex", flexWrap: "wrap", gap: 8 }}>
              {QUICK_ACTIONS.map(action => (
                <button
                  key={action}
                  onClick={() => handleSubmit(action)}
                  disabled={streaming}
                  style={{
                    background: "var(--surface2)",
                    border: "1px solid var(--border)",
                    borderRadius: 20,
                    padding: "6px 14px",
                    fontSize: 13,
                    color: "var(--text)",
                    transition: "border-color 0.15s, background 0.15s",
                    cursor: streaming ? "not-allowed" : "pointer",
                    opacity: streaming ? 0.5 : 1,
                  }}
                  onMouseEnter={e => { if (!streaming) { (e.currentTarget).style.borderColor = "var(--accent)"; (e.currentTarget).style.background = "var(--accent-weak)"; }}}
                  onMouseLeave={e => { (e.currentTarget).style.borderColor = "var(--border)"; (e.currentTarget).style.background = "var(--surface2)"; }}
                >
                  {action}
                </button>
              ))}
            </div>
          )}

          {/* Grounded-mode toggle */}
          <div style={{
            borderTop: "1px solid var(--border)",
            padding: "8px 16px 0",
            flexShrink: 0,
            background: "var(--surface)",
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}>
            <label
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                fontSize: 13,
                color: "var(--text-muted)",
                cursor: streaming ? "not-allowed" : "pointer",
                userSelect: "none",
              }}
            >
              <input
                type="checkbox"
                checked={grounded}
                onChange={e => setGrounded(e.target.checked)}
                disabled={streaming}
                style={{
                  width: 15,
                  height: 15,
                  accentColor: "var(--accent)",
                  cursor: streaming ? "not-allowed" : "pointer",
                  padding: 0,
                }}
              />
              Grounded mode
            </label>
            <span
              className="info"
              tabIndex={0}
              role="button"
              aria-label="About grounded mode"
              style={{ color: "var(--text-muted)", opacity: 0.7 }}
            >
              <Info size={14} />
              <span className="info-bubble" role="tooltip">
                When on, answers use only this book's retrieved passages and ignore the
                model's built-in knowledge. Uncheck to let the model also draw on what it
                already knows about this book.
              </span>
            </span>
          </div>

          {/* Input */}
          <div style={{
            padding: "10px 16px 12px",
            display: "flex",
            gap: 10,
            flexShrink: 0,
            background: "var(--surface)",
          }}>
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={streaming}
              placeholder="Ask about the book… (Enter to send, Shift+Enter for newline)"
              rows={2}
              style={{
                flex: 1,
                resize: "none",
                fontSize: 14,
                opacity: streaming ? 0.7 : 1,
              }}
            />
            <button
              onClick={() => handleSubmit(input)}
              disabled={streaming || !input.trim()}
              style={{
                background: streaming || !input.trim() ? "var(--surface2)" : "var(--accent)",
                color: streaming || !input.trim() ? "var(--text-muted)" : "#fff",
                borderRadius: "var(--radius)",
                padding: "0 18px",
                fontWeight: 600,
                fontSize: 14,
                transition: "background 0.15s",
                cursor: streaming || !input.trim() ? "not-allowed" : "pointer",
                alignSelf: "stretch",
              }}
            >
              {streaming ? <span className="spinner" /> : "Send"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
