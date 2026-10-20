"use client";

import { useState } from "react";
import { INTENTS, intentLabel } from "@/lib/intents";
import { clearResolved, resolveItem, useQueue, type QueueItem } from "@/lib/queue";

function Item({ item }: { item: QueueItem }) {
  const [choice, setChoice] = useState(item.predicted ?? "");
  const selectId = `intent-${item.id}`;
  return (
    <li className="space-y-3 rounded-xl border border-line bg-surface p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2 text-xs text-muted">
        <span className="tabular">{new Date(item.createdAt).toLocaleString()}</span>
        {item.cachedLabel && <span>{item.cachedLabel}</span>}
      </div>
      <p className="text-sm">“{item.text || "(empty message)"}”</p>
      <p className="tabular text-xs text-ink-2">
        {item.predicted ? (
          <>
            Agent suggested <strong>{intentLabel(item.predicted)}</strong> at calibrated
            confidence {item.confidence.toFixed(3)}
          </>
        ) : (
          "The agent made no suggestion."
        )}
      </p>
      <ul className="list-disc pl-5 text-xs text-ink-2">
        {item.reasons.map((reason) => (
          <li key={reason}>{reason}</li>
        ))}
      </ul>

      {item.resolved ? (
        <p className="flex items-center gap-2 text-sm" role="status">
          <span
            aria-hidden
            className="flex h-5 w-5 items-center justify-center rounded-full text-xs text-white"
            style={{ background: "var(--good)" }}
          >
            ✓
          </span>
          <span>
            Reviewed as <strong>{intentLabel(item.resolved.intent)}</strong>
            <span className="text-ink-2">
              {" "}
              · {item.resolved.agreed ? "agreed with the agent" : "corrected the agent"}
            </span>
          </span>
        </p>
      ) : (
        <form
          className="flex flex-wrap items-end gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            if (choice) resolveItem(item.id, choice);
          }}
        >
          <div className="flex min-w-0 flex-1 flex-col gap-1">
            <label htmlFor={selectId} className="text-xs font-medium">
              Right intent
            </label>
            <select
              id={selectId}
              value={choice}
              onChange={(event) => setChoice(event.target.value)}
              className="rounded-lg border border-line bg-surface px-2 py-1.5 text-sm"
            >
              <option value="" disabled>
                Choose an intent
              </option>
              {Object.entries(INTENTS).map(([key, label]) => (
                <option key={key} value={key}>
                  {label}
                </option>
              ))}
              <option value="not_a_banking_request">Not a banking request</option>
            </select>
          </div>
          <button
            type="submit"
            disabled={!choice}
            className="rounded-lg bg-ink px-3 py-1.5 text-sm font-medium text-page disabled:opacity-40"
          >
            Confirm
          </button>
        </form>
      )}
    </li>
  );
}

export default function ReviewQueue() {
  const items = useQueue();
  const open = items.filter((item) => !item.resolved);
  const done = items.filter((item) => item.resolved);

  if (items.length === 0) {
    return (
      <p className="rounded-xl border border-dashed border-axis p-6 text-sm text-ink-2">
        The queue is empty. Messages the agent hands off on the Triage page show up here.
      </p>
    );
  }

  return (
    <div className="space-y-8">
      <section className="space-y-3">
        <h2 className="text-sm font-semibold">Waiting for review ({open.length})</h2>
        {open.length === 0 ? (
          <p className="text-sm text-muted">Nothing waiting.</p>
        ) : (
          <ul className="space-y-3">
            {open.map((item) => (
              <Item key={item.id} item={item} />
            ))}
          </ul>
        )}
      </section>
      {done.length > 0 && (
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">Reviewed ({done.length})</h2>
            <button type="button" onClick={clearResolved} className="text-xs text-ink-2 underline">
              Clear reviewed
            </button>
          </div>
          <ul className="space-y-3">
            {done.map((item) => (
              <Item key={item.id} item={item} />
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
