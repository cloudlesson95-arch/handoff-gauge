export type AttributeValue = string | number | boolean | string[] | number[];

export type Span = {
  name: string;
  span_id: string;
  parent_id: string | null;
  start_ms: number;
  duration_ms: number;
  status: string;
  attributes: Record<string, AttributeValue>;
};

export type GuardrailResult = {
  name: string;
  stage: string;
  tripped: boolean;
  reason: string;
};

export type Sample = {
  seed: number;
  content: string;
  cache_hit: boolean;
  error: string | null;
};

export type TriageCore = {
  decision: "auto" | "handoff";
  intent: string | null;
  calibrated_confidence: number;
  raw_score: number;
  threshold: number | null;
  reasons: string[];
  notes: string[];
  redacted_text: string;
  guardrails: GuardrailResult[];
  samples: Sample[];
  spans: Span[];
};

export type TriageResult = TriageCore & {
  source: "live" | "cached_example";
  cached_reason: string | null;
  cached_label: string | null;
  cached_text?: string;
};

export type Example = {
  id: string;
  title: string;
  text: string;
  result: TriageCore;
};
