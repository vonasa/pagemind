import { BookOpen } from "lucide-react";
import { Link } from "react-router-dom";

/** Editorial brand bar: book-open mark + Fraunces wordmark + muted tagline. */
export default function Masthead() {
  return (
    <header className="border-b border-line bg-bg/80 backdrop-blur-sm sticky top-0 z-10">
      <div className="mx-auto max-w-6xl px-6 py-4 flex items-center gap-3">
        <Link to="/" className="flex items-center gap-3 group" aria-label="PageMind home">
          <span className="grid place-items-center size-9 rounded-lg bg-accent/10 text-accent">
            <BookOpen size={20} strokeWidth={2} />
          </span>
          <span className="flex flex-col leading-none">
            <span className="font-display text-[22px] font-semibold text-ink tracking-tight">
              PageMind
            </span>
            <span className="font-ui text-[12px] text-muted mt-0.5">
              Your reading companion
            </span>
          </span>
        </Link>
      </div>
    </header>
  );
}
