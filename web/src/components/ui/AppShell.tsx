import { type ReactNode } from "react";
import Masthead from "./Masthead";

/** Full-page editorial shell: brand masthead + centered content column. */
export default function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-dvh bg-bg">
      <Masthead />
      <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
    </div>
  );
}
