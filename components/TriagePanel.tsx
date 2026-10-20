"use client";

import { useEffect, useState } from "react";
import { addToQueue } from "@/lib/queue";
import type { Example, TriageResult } from "@/lib/types";
import DecisionCard from "./DecisionCard";

const MAX_CHARS = 500;
const STEPS = [
  "Running the input guardrails",
  "Sampling the model 3 times in parallel",
  "Checking the answers against the schema",
  "Calibrating the confidence and deciding",
];

type State =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "done"; result: TriageResult; queued: boolean };

export default function TriagePanel() {
  const [text, setText] = useState("");
  const [examples, setExamples] = useState<Example[]>([]);
  const [state, setState] = useState<State>({ kind: "idle" });

  useEffect(() => {
    fetch("/api/py/examples")
      .then((r) => (r.ok ? r.json() : []))
      .then(setExamples)
      .catch(() => setExamples([]));
  }, []);

  async function run(message: string) {
    setState({ kind: "loading" });
    try {
      const response = await fetch("/api/py/triage", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ text: message }),
      });
      if (!response.ok) {
        const message =
          response.status === 422
            ? `Messages are limited to ${MAX_CHARS} characters.`
            : `The API answered ${response.status}.`;
        setState({ kind: "error", message });
        return;
      }
      const result: TriageResult = await response.json();
      const queued = result.decision === "handoff";
      if (queued) {
        addToQueue({
          text: result.source === "cached_example" ? (result.cached_text ?? "") : result.redacted_text,
          predicted: result.intent,
          confidence: result.calibrated_confidence,
          reasons: result.reasons,
          cachedLabel: result.cached_label,
        });
      }
      setState({ kind: "done", result, queued });
    } catch {
      setState({ kind: "error", message: "Could not reach the API." });
    }
  }

  const loading = state.kind === "loading";

  return (
    <div className="space-y-6">
      <form
        className="space-y-3"
        onSubmit={(event) => {
          event.preventDefault();
          if (text.trim()) run(text);
        }}
      >
        <label htmlFor="message" className="block text-sm font-medium">
          Customer message
        </label>
        <textarea
          id="message"
          value={text}
          maxLength={MAX_CHARS}
          rows={3}
          onChange={(event) => setText(event.target.value)}
          placeholder="e.g. I was charged twice for my card"
          className="w-full rounded-lg border border-line bg-surface px-3 py-2 text-base outline-none focus:border-s1"
        />
        <div className="flex flex-wrap items-center justify-between gap-3">
          <button
            type="submit"
            disabled={loading || !text.trim()}
            className="rounded-lg bg-ink px-4 py-2 text-sm font-medium text-page disabled:opacity-40"
          >
            {loading ? "Triaging…" : "Triage"}
          </button>
          <span className="tabular text-xs text-muted">
            {text.length} / {MAX_CHARS}
          </span>
        </div>
      </form>

      {examples.length > 0 && (
        <div>
          <p className="mb-2 text-xs font-medium text-ink-2">Try an example</p>
          <div className="flex flex-wrap gap-2">
            {examples.map((example) => (
              <button
                key={example.id}
                type="button"
                disabled={loading}
                onClick={() => {
                  setText(example.text);
                  run(example.text);
                }}
                className="rounded-full border border-line px-3 py-1 text-xs hover:bg-wash disabled:opacity-40"
              >
                {example.title}
              </button>
            ))}
          </div>
        </div>
      )}

      {state.kind === "loading" && (
        <div
          className="rounded-xl border border-line bg-surface p-5"
          role="status"
          aria-live="polite"
          aria-busy="true"
        >
          <p className="flex items-center gap-2 text-sm font-semibold">
            <span
              aria-hidden
              className="h-4 w-4 animate-spin rounded-full border-2 border-grid border-t-s1"
            />
            Triaging the message
          </p>
          <ol className="mt-3 space-y-1.5 text-sm text-ink-2">
            {STEPS.map((step, i) => (
              <li
                key={step}
                className="animate-pulse"
                style={{ animationDelay: `${i * 250}ms` }}
              >
                {step}
              </li>
            ))}
          </ol>
        </div>
      )}

      {state.kind === "error" && (
        <p className="rounded-lg border border-critical px-3 py-2 text-sm" role="alert">
          {state.message}
        </p>
      )}

      {state.kind === "done" && <DecisionCard result={state.result} queued={state.queued} />}
    </div>
  );
}
