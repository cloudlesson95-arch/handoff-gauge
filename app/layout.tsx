import type { Metadata } from "next";
import Nav from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "Handoff Gauge",
  description:
    "A ticket-triage agent that uses calibrated confidence to decide when to hand off to a human.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">
        <header className="border-b border-line bg-surface">
          <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
            <span className="font-semibold tracking-tight">Handoff Gauge</span>
            <Nav />
          </div>
        </header>
        <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-6">{children}</main>
        <footer className="border-t border-line px-4 py-4 text-xs text-muted">
          <p className="mx-auto max-w-5xl">
            Messages from Banking77 (PolyAI, CC BY 4.0), a public dataset of customer-banking
            questions. Demo only: no data is stored on the server, and the review queue lives in
            this browser.
          </p>
        </footer>
      </body>
    </html>
  );
}
