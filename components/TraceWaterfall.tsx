"use client";

import { useState } from "react";
import type { Span } from "@/lib/types";

const KINDS = [
  { key: "llm", label: "LLM call", color: "var(--s1)" },
  { key: "guardrail", label: "Guardrail", color: "var(--s2)" },
  { key: "agent", label: "Agent", color: "var(--s3)" },
  { key: "decision", label: "Estimate and decision", color: "var(--s4)" },
] as const;

function kindOf(span: Span) {
  if (span.name.startsWith("chat ")) return KINDS[0];
  if (span.name.startsWith("guardrail ")) return KINDS[1];
  if (span.name.startsWith("invoke_agent")) return KINDS[2];
  return KINDS[3];
}

function formatMs(ms: number) {
  return ms >= 100 ? `${ms.toFixed(0)} ms` : `${ms.toFixed(1)} ms`;
}

function formatValue(value: Span["attributes"][string]) {
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "number" && !Number.isInteger(value)) return value.toFixed(4);
  return String(value);
}

export default function TraceWaterfall({ spans }: { spans: Span[] }) {
  const [selected, setSelected] = useState<string | null>(null);
  if (spans.length === 0) return null;
  const total = Math.max(...spans.map((s) => s.start_ms + s.duration_ms), 1);
  const depth = (span: Span) => (span.parent_id ? 1 : 0);
  const current = spans.find((s) => s.span_id === selected);

  return (
    <section aria-labelledby="trace-title" className="space-y-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 id="trace-title" className="text-sm font-semibold">
          Trace <span className="font-normal text-muted">· {spans.length} OpenTelemetry spans</span>
        </h3>
        <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2" aria-label="Span kinds">
          {KINDS.map((k) => (
            <li key={k.key} className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: k.color }} />
              {k.label}
            </li>
          ))}
        </ul>
      </div>

      <ol className="space-y-px">
        {spans.map((span) => {
          const kind = kindOf(span);
          const left = (span.start_ms / total) * 100;
          const width = Math.max((span.duration_ms / total) * 100, 0.4);
          const tripped = span.attributes["guardrail.tripped"] === true;
          const active = selected === span.span_id;
          return (
            <li key={span.span_id}>
              <button
                type="button"
                onClick={() => setSelected(active ? null : span.span_id)}
                aria-expanded={active}
                title={`${span.name}: starts at ${formatMs(span.start_ms)}, takes ${formatMs(span.duration_ms)}`}
                className={`grid w-full grid-cols-1 items-center gap-x-3 rounded px-1.5 py-1 text-left text-xs sm:grid-cols-[minmax(0,15rem)_1fr] ${
                  active ? "bg-wash" : "hover:bg-wash"
                }`}
              >
                <span
                  className="truncate text-ink-2"
                  style={{ paddingLeft: `${depth(span) * 0.9}rem` }}
                >
                  {span.name}
                  {tripped && <span className="ml-1 font-semibold text-critical">✕ tripped</span>}
                </span>
                <span className="relative block h-4">
                  <span className="absolute inset-y-[7px] left-0 right-0 bg-grid" />
                  <span
                    className="absolute inset-y-0.5 rounded-[3px]"
                    style={{
                      left: `${left}%`,
                      width: `${width}%`,
                      minWidth: 2,
                      background: kind.color,
                    }}
                  />
                  <span
                    className="tabular absolute top-0 whitespace-nowrap rounded bg-surface px-1 text-[11px] leading-4 text-ink-2"
                    style={
                      left + width <= 78
                        ? { left: `calc(${left + width}% + 4px)` }
                        : left >= 15
                          ? { right: `calc(${100 - left}% + 4px)` }
                          : { right: 4 }
                    }
                  >
                    {formatMs(span.duration_ms)}
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ol>
      <p className="tabular flex justify-between text-[11px] text-muted sm:ml-[15.75rem]">
        <span>0 ms</span>
        <span>{formatMs(total)}</span>
      </p>

      {current ? (
        <div className="rounded-lg border border-line bg-surface p-3 text-xs">
          <p className="mb-2 font-semibold">
            {current.name}{" "}
            <span className="tabular font-normal text-muted">
              · starts {formatMs(current.start_ms)} · {formatMs(current.duration_ms)} ·{" "}
              {current.status}
            </span>
          </p>
          <dl className="grid grid-cols-1 gap-x-4 gap-y-1 sm:grid-cols-[auto_1fr]">
            {Object.entries(current.attributes).map(([key, value]) => (
              <div key={key} className="contents">
                <dt className="font-mono text-muted">{key}</dt>
                <dd className="break-all font-mono">{formatValue(value)}</dd>
              </div>
            ))}
          </dl>
        </div>
      ) : (
        <p className="text-xs text-muted">Select a span to see its attributes.</p>
      )}
    </section>
  );
}
