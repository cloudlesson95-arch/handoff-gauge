"use client";

import { useSyncExternalStore } from "react";

// The human review queue lives only in this browser's localStorage. Items hold the redacted
// text, never the raw message.
export type QueueItem = {
  id: string;
  createdAt: string;
  text: string;
  predicted: string | null;
  confidence: number;
  reasons: string[];
  cachedLabel: string | null;
  resolved?: { intent: string; at: string; agreed: boolean };
};

const KEY = "handoff-gauge.queue.v1";
const EMPTY: QueueItem[] = [];
const listeners = new Set<() => void>();
// Last value read, so the snapshot stays referentially stable; also the in-memory fallback
// when storage is blocked.
let snapshot: { raw: string | null; items: QueueItem[] } = { raw: null, items: EMPTY };

function read(): QueueItem[] {
  let raw: string | null = snapshot.raw;
  try {
    raw = window.localStorage.getItem(KEY);
  } catch {
    return snapshot.items;
  }
  if (raw === snapshot.raw) return snapshot.items;
  let items: QueueItem[] = EMPTY;
  try {
    const parsed = raw ? JSON.parse(raw) : [];
    items = Array.isArray(parsed) ? parsed : EMPTY;
  } catch {
    items = EMPTY;
  }
  snapshot = { raw, items };
  return items;
}

function write(items: QueueItem[]) {
  const raw = JSON.stringify(items);
  try {
    window.localStorage.setItem(KEY, raw);
  } catch {
    // storage blocked: keep the queue in memory for this tab
  }
  snapshot = { raw, items };
  listeners.forEach((notify) => notify());
}

function subscribe(notify: () => void) {
  listeners.add(notify);
  const onStorage = (event: StorageEvent) => {
    if (event.key === KEY) notify();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(notify);
    window.removeEventListener("storage", onStorage);
  };
}

export function useQueue(): QueueItem[] {
  return useSyncExternalStore(subscribe, read, () => EMPTY);
}

export function addToQueue(item: Omit<QueueItem, "id" | "createdAt">) {
  const entry: QueueItem = { ...item, id: crypto.randomUUID(), createdAt: new Date().toISOString() };
  write([entry, ...read()].slice(0, 100));
}

export function resolveItem(id: string, intent: string) {
  write(
    read().map((item) =>
      item.id === id
        ? { ...item, resolved: { intent, at: new Date().toISOString(), agreed: intent === item.predicted } }
        : item,
    ),
  );
}

export function clearResolved() {
  write(read().filter((item) => !item.resolved));
}
