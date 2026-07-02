import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import { fetchSection, type Section } from "../api";

interface Props {
  sectionId: string;
  charOffset: number | null;
  quoteText: string;
  onClose: () => void;
}

export default function PassagePopover({ sectionId, charOffset, quoteText, onClose }: Props) {
  const [section, setSection] = useState<Section | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const markRef = useRef<HTMLElement>(null);

  useEffect(() => {
    fetchSection(sectionId)
      .then(s => { setSection(s); setLoading(false); })
      .catch(e => { setError(String(e)); setLoading(false); });
  }, [sectionId]);

  // Once the section renders, scroll the highlighted quote into view. When the
  // quote can't be located there is no <mark>, so the popover stays at the top.
  useEffect(() => {
    if (section && markRef.current) {
      markRef.current.scrollIntoView({ block: "center" });
    }
  }, [section]);

  // Build highlighted content
  const renderContent = () => {
    if (!section) return null;
    const text = section.content;

    // Find the quote in the text
    const start = charOffset != null && charOffset >= 0
      ? charOffset
      : text.indexOf(quoteText);
    const end = start >= 0 ? start + quoteText.length : -1;

    if (start < 0 || end < 0 || end > text.length) {
      return <p style={{ whiteSpace: "pre-wrap" }}>{text}</p>;
    }

    return (
      <p style={{ whiteSpace: "pre-wrap" }}>
        {text.slice(0, start)}
        <mark ref={markRef} style={{ background: "var(--highlight)", color: "var(--text)", borderRadius: 2, padding: "1px 2px" }}>
          {text.slice(start, end)}
        </mark>
        {text.slice(end)}
      </p>
    );
  };

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        style={{
          position: "fixed", inset: 0, background: "#0009", zIndex: 100,
        }}
      />

      {/* Panel */}
      <div
        style={{
          position: "fixed",
          top: "50%", left: "50%",
          transform: "translate(-50%, -50%)",
          width: "min(640px, 90vw)",
          maxHeight: "70vh",
          background: "var(--surface)",
          border: "1px solid var(--border-strong)",
          borderRadius: "var(--radius-lg)",
          boxShadow: "var(--shadow-pop)",
          display: "flex",
          flexDirection: "column",
          zIndex: 101,
          overflow: "hidden",
        }}
      >
        {/* Header */}
        <div style={{
          padding: "12px 16px",
          borderBottom: "1px solid var(--border-strong)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexShrink: 0,
        }}>
          <span style={{ fontFamily: "var(--font-display)", fontWeight: 600, fontSize: 15 }}>
            {section ? `${section.chapter_title}` : "Source passage"}
          </span>
          <button
            onClick={onClose}
            title="Close"
            style={{ color: "var(--text-muted)", display: "inline-flex", lineHeight: 1, padding: 2 }}
          >
            <X size={18} />
          </button>
        </div>

        {/* Body */}
        <div style={{ padding: "16px", overflowY: "auto", fontSize: 14, lineHeight: 1.7 }}>
          {loading && (
            <div style={{ display: "flex", gap: 8, color: "var(--text-muted)" }}>
              <span className="spinner" /> Loading passage…
            </div>
          )}
          {error && <p style={{ color: "var(--danger)" }}>{error}</p>}
          {section && renderContent()}
        </div>
      </div>
    </>
  );
}
