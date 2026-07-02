import type { ReactNode } from "react";
import { ArrowLeft } from "lucide-react";
import { useNavigate } from "react-router-dom";

type Mode = "overview" | "chat";

interface Props {
  bookId: string;
  title: string;
  author?: string | null;
  /** Which mode is currently active — highlights the matching segment. */
  active: Mode;
  /** Page-specific action rendered at the right edge (e.g. Chat's "New chat"). */
  children?: ReactNode;
}

// Shared header for the two book modes: back-to-library, title/author, an
// Overview | Chat segmented toggle, and an optional page action slot.
export default function BookHeader({ bookId, title, author, active, children }: Props) {
  const navigate = useNavigate();

  return (
    <header
      style={{
        borderBottom: "1px solid var(--border)",
        padding: "12px 20px",
        display: "flex",
        alignItems: "center",
        gap: 16,
        background: "var(--surface)",
        flexShrink: 0,
      }}
    >
      <button
        onClick={() => navigate("/")}
        style={{ color: "var(--text-muted)", display: "inline-flex", padding: "4px" }}
        title="Back to library"
      >
        <ArrowLeft size={18} />
      </button>

      <div style={{ minWidth: 0 }}>
        <h2 style={{ fontFamily: "var(--font-display)", fontSize: 17, fontWeight: 600, lineHeight: 1.2, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          {title}
        </h2>
        {author && <p style={{ fontSize: 13, color: "var(--text-muted)" }}>{author}</p>}
      </div>

      {/* Overview | Chat segmented control */}
      <div
        role="tablist"
        aria-label="Book mode"
        style={{
          display: "inline-flex",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius)",
          overflow: "hidden",
          background: "var(--surface2)",
        }}
      >
        <Segment label="Overview" isActive={active === "overview"} onClick={() => navigate(`/books/${bookId}/overview`)} />
        <Segment label="Chat" isActive={active === "chat"} onClick={() => navigate(`/books/${bookId}/chat`)} />
      </div>

      <div style={{ marginLeft: "auto" }}>{children}</div>
    </header>
  );
}

function Segment({ label, isActive, onClick }: { label: string; isActive: boolean; onClick: () => void }) {
  return (
    <button
      role="tab"
      aria-selected={isActive}
      onClick={onClick}
      style={{
        padding: "6px 16px",
        fontSize: 13,
        fontWeight: 600,
        color: isActive ? "#fff" : "var(--text-muted)",
        background: isActive ? "var(--accent)" : "transparent",
        transition: "background 0.15s, color 0.15s",
        cursor: isActive ? "default" : "pointer",
      }}
    >
      {label}
    </button>
  );
}
