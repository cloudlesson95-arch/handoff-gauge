import Link from "next/link";
import { intentLabel } from "@/lib/intents";
import type { TriageResult } from "@/lib/types";
import TraceWaterfall from "./TraceWaterfall";

function parseSample(content: string): { intent: string; confidence: number } | null {
  try {
    const data = JSON.parse(content);
    if (typeof data.intent === "string" && typeof data.confidence === "number") return data;
  } catch {}
  return null;
}

export default function DecisionCard({ result, queued }: { result: TriageResult; queued: boolean }) {
  const auto = result.decision === "auto";
  const confidence = result.calibrated_confidence;
  const threshold = result.threshold;
  const tripped = result.guardrails.filter((g) => g.tripped);

  return (
    <article className="space-y-5 rounded-xl border border-line bg-surface p-4 sm:p-5" aria-live="polite">
      {result.source === "cached_example" && (
        <p className="rounded-md border border-dashed border-axis px-3 py-2 text-sm">
          <span className="font-semibold">{result.cached_label}</span>
          <span className="text-ink-2">
            {" "}
            · served instead of a live call. The decision below is for this example message:
            “{result.cached_text}”
          </span>
        </p>
      )}

      <header className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <span
          className="inline-flex items-center gap-2 rounded-full border border-line px-3 py-1 text-sm font-semibold"
          role="status"
        >
          <span
            aria-hidden
            className="flex h-5 w-5 items-center justify-center rounded-full text-xs text-white"
            style={{ background: auto ? "var(--good)" : "var(--serious)" }}
          >
            {auto ? "✓" : "↗"}
          </span>
          {auto ? "AUTO" : "HANDED OFF"}
        </span>
        <span className="text-lg font-semibold">{intentLabel(result.intent)}</span>
        {result.intent && <code className="text-xs text-muted">{result.intent}</code>}
      </header>

      <div className="space-y-1.5">
        <p className="tabular text-sm">
          Calibrated confidence <strong>{confidence.toFixed(3)}</strong>
          {threshold !== null && (
            <span className="text-ink-2"> · threshold {threshold.toFixed(3)}</span>
          )}
        </p>
        <div className="relative h-2.5 rounded-full bg-grid" aria-hidden>
          <div
            className="h-full rounded-full"
            style={{ width: `${confidence * 100}%`, background: "var(--s1)" }}
          />
          {threshold !== null && (
            <div
              className="absolute -top-1 h-[18px] w-0.5 bg-ink"
              style={{ left: `calc(${threshold * 100}% - 1px)` }}
            />
          )}
        </div>
      </div>

      <div>
        <h3 className="text-sm font-semibold">{auto ? "Why it was handled automatically" : "Why it went to a human"}</h3>
        <ul className="mt-1 list-disc space-y-0.5 pl-5 text-sm text-ink-2">
          {result.reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
          {result.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
        {!auto && queued && (
          <p className="mt-2 text-sm">
            Added to the{" "}
            <Link href="/review" className="underline underline-offset-2">
              review queue
            </Link>
            .
          </p>
        )}
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <h3 className="text-sm font-semibold">Guardrails</h3>
          <ul className="mt-1 flex flex-wrap gap-1.5 text-xs">
            {result.guardrails.map((g) => (
              <li
                key={g.name}
                className={`rounded-md border px-2 py-0.5 ${g.tripped ? "border-critical font-semibold" : "border-line text-ink-2"}`}
              >
                {g.tripped ? "✕" : "✓"} {g.name}
              </li>
            ))}
          </ul>
          {tripped.length === 0 && result.guardrails.length < 6 && (
            <p className="mt-1 text-xs text-muted">The chain stopped early.</p>
          )}
        </div>
        <div>
          <h3 className="text-sm font-semibold">Samples ({result.samples.length})</h3>
          {result.samples.length === 0 ? (
            <p className="mt-1 text-xs text-muted">The model was not called.</p>
          ) : (
            <ul className="tabular mt-1 space-y-0.5 text-xs text-ink-2">
              {result.samples.map((s) => {
                const parsed = parseSample(s.content);
                return (
                  <li key={s.seed}>
                    seed {s.seed}:{" "}
                    {parsed ? (
                      <>
                        <code>{parsed.intent}</code> at {parsed.confidence.toFixed(2)}
                      </>
                    ) : (
                      <span className="text-critical">{s.error ?? "invalid answer"}</span>
                    )}
                    {s.cache_hit && <span className="text-muted"> · cached</span>}
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>

      <div>
        <h3 className="text-sm font-semibold">What the model saw</h3>
        <p className="mt-1 rounded-md bg-wash px-3 py-2 text-sm">{result.redacted_text || "(empty)"}</p>
      </div>

      <TraceWaterfall key={result.spans[0]?.span_id} spans={result.spans} />
    </article>
  );
}
