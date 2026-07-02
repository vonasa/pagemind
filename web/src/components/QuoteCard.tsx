import { useState } from "react";
import { ArrowUpRight } from "lucide-react";
import type { Quote } from "../api";
import PassagePopover from "./PassagePopover";

interface Props {
  quote: Quote;
}

export default function QuoteCard({ quote }: Props) {
  const [showPopover, setShowPopover] = useState(false);

  return (
    <>
      <div
        style={{
          background: "var(--surface2)",
          border: "1px solid var(--border)",
          borderLeft: "3px solid var(--accent)",
          borderRadius: "var(--radius)",
          padding: "10px 12px",
          fontSize: 13,
          lineHeight: 1.6,
          display: "flex",
          flexDirection: "column",
          gap: 6,
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}>
          <blockquote style={{ fontStyle: "italic", color: "var(--text)", flex: 1 }}>
            "{quote.text}"
          </blockquote>
          <button
            title="View source passage"
            onClick={() => setShowPopover(true)}
            style={{
              color: "var(--accent)",
              display: "inline-flex",
              flexShrink: 0,
              padding: "4px",
              borderRadius: 4,
              background: "transparent",
              transition: "background 0.15s",
            }}
            onMouseEnter={e => ((e.currentTarget as HTMLButtonElement).style.background = "var(--accent-weak)")}
            onMouseLeave={e => ((e.currentTarget as HTMLButtonElement).style.background = "transparent")}
          >
            <ArrowUpRight size={15} />
          </button>
        </div>
        <span style={{ fontSize: 11, color: "var(--text-muted)", fontWeight: 500 }}>
          Chapter {quote.citation.chapter}
        </span>
      </div>

      {showPopover && (
        <PassagePopover
          sectionId={quote.citation.section_id}
          charOffset={quote.citation.char_offset}
          quoteText={quote.text}
          onClose={() => setShowPopover(false)}
        />
      )}
    </>
  );
}
