"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

const HEIGHT = 300;
const M = { top: 16, right: 20, bottom: 44, left: 48 };

function useWidth() {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) =>
      setWidth(Math.floor(entry.contentRect.width)),
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

type Tip = { x: number; y: number; value: string; label: string };

function Tooltip({ tip }: { tip: Tip | null }) {
  if (!tip) return null;
  return (
    <div
      className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full rounded-md border border-line bg-surface px-2.5 py-1.5 text-xs shadow-sm"
      style={{ left: tip.x, top: tip.y - 10 }}
    >
      <div className="tabular font-semibold">{tip.value}</div>
      <div className="text-ink-2">{tip.label}</div>
    </div>
  );
}

function scale(d0: number, d1: number, r0: number, r1: number) {
  return (v: number) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
}

function Axes({
  width,
  x,
  y,
  xTicks,
  yTicks,
  xLabel,
  yLabel,
}: {
  width: number;
  x: (v: number) => number;
  y: (v: number) => number;
  xTicks: number[];
  yTicks: number[];
  xLabel: string;
  yLabel: string;
}) {
  const bottom = HEIGHT - M.bottom;
  return (
    <g className="tabular" fontSize={11}>
      {yTicks.map((t) => (
        <g key={`y${t}`}>
          <line x1={M.left} x2={width - M.right} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
          <text x={M.left - 8} y={y(t)} dy="0.32em" textAnchor="end" fill="var(--muted)">
            {t.toFixed(t % 1 ? 2 : 1)}
          </text>
        </g>
      ))}
      {xTicks.map((t) => (
        <text key={`x${t}`} x={x(t)} y={bottom + 16} textAnchor="middle" fill="var(--muted)">
          {t.toFixed(t % 1 ? 2 : 1)}
        </text>
      ))}
      <line x1={M.left} x2={width - M.right} y1={bottom} y2={bottom} stroke="var(--axis)" />
      <text x={(M.left + width - M.right) / 2} y={HEIGHT - 6} textAnchor="middle" fill="var(--ink-2)">
        {xLabel}
      </text>
      <text
        transform={`translate(12 ${(M.top + bottom) / 2}) rotate(-90)`}
        textAnchor="middle"
        fill="var(--ink-2)"
      >
        {yLabel}
      </text>
    </g>
  );
}

function Frame({ title, legend, children }: { title: string; legend?: ReactNode; children: ReactNode }) {
  return (
    <figure className="min-w-0 space-y-2 rounded-xl border border-line bg-surface p-4">
      <figcaption className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-sm font-semibold">{title}</span>
        {legend}
      </figcaption>
      {children}
    </figure>
  );
}

function LegendLine({ color, label, dashed }: { color: string; label: string; dashed?: boolean }) {
  return (
    <span className="flex items-center gap-1.5 text-xs text-ink-2">
      <svg width="18" height="8" aria-hidden>
        <line x1="1" x2="17" y1="4" y2="4" stroke={color} strokeWidth="2" strokeDasharray={dashed ? "4 3" : undefined} strokeLinecap="round" />
      </svg>
      {label}
    </span>
  );
}

export type Bin = { lo: number; hi: number; count: number; conf: number; acc: number };

export function ReliabilityChart({
  before,
  after,
  eceBefore,
  eceAfter,
}: {
  before: Bin[];
  after: Bin[];
  eceBefore: number;
  eceAfter: number;
}) {
  const [ref, width] = useWidth();
  const [tip, setTip] = useState<Tip | null>(null);
  const x = scale(0, 1, M.left, width - M.right);
  const y = scale(0, 1, HEIGHT - M.bottom, M.top);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const series = [
    { key: "before", label: `Stated confidence · ECE ${eceBefore.toFixed(3)}`, short: "stated", color: "var(--s2)", bins: before.filter((b) => b.count) },
    { key: "after", label: `Calibrated · ECE ${eceAfter.toFixed(3)}`, short: "calibrated", color: "var(--s1)", bins: after.filter((b) => b.count) },
  ];

  return (
    <Frame
      title="Reliability on the test split"
      legend={
        <span className="flex flex-wrap gap-3">
          {series.map((s) => (
            <LegendLine key={s.key} color={s.color} label={s.label} />
          ))}
          <LegendLine color="var(--muted)" label="Perfect calibration" dashed />
        </span>
      }
    >
      <div ref={ref} className="relative w-full" style={{ height: HEIGHT }} onPointerLeave={() => setTip(null)}>
        {width > 0 && <svg width={width} height={HEIGHT} role="img" aria-label="Reliability diagram: accuracy against confidence, before and after calibration">
          <Axes width={width} x={x} y={y} xTicks={ticks} yTicks={ticks} xLabel="Confidence" yLabel="Accuracy" />
          <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="var(--muted)" strokeDasharray="4 4" />
          {series.map((s) => (
            <g key={s.key}>
              <polyline
                points={s.bins.map((b) => `${x(b.conf)},${y(b.acc)}`).join(" ")}
                fill="none"
                stroke={s.color}
                strokeWidth={2}
                strokeLinejoin="round"
              />
              {s.bins.map((b) => (
                <g key={b.lo}>
                  <circle cx={x(b.conf)} cy={y(b.acc)} r={5} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
                  <circle
                    cx={x(b.conf)}
                    cy={y(b.acc)}
                    r={12}
                    fill="transparent"
                    tabIndex={0}
                    aria-label={`${s.short}, bin ${b.lo.toFixed(1)} to ${b.hi.toFixed(1)}: ${b.count} messages, confidence ${b.conf.toFixed(2)}, accuracy ${b.acc.toFixed(2)}`}
                    onPointerEnter={() =>
                      setTip({ x: x(b.conf), y: y(b.acc), value: `accuracy ${b.acc.toFixed(2)} · confidence ${b.conf.toFixed(2)}`, label: `${s.short}, ${b.count} messages in ${b.lo.toFixed(1)}–${b.hi.toFixed(1)}` })
                    }
                    onFocus={() =>
                      setTip({ x: x(b.conf), y: y(b.acc), value: `accuracy ${b.acc.toFixed(2)} · confidence ${b.conf.toFixed(2)}`, label: `${s.short}, ${b.count} messages in ${b.lo.toFixed(1)}–${b.hi.toFixed(1)}` })
                    }
                    onBlur={() => setTip(null)}
                    className="outline-none"
                  />
                </g>
              ))}
            </g>
          ))}
        </svg>}
        <Tooltip tip={tip} />
      </div>
      <p className="text-xs text-muted">
        Each point is a confidence bin; points below the diagonal are over-confident. Bins with
        few messages move a lot: the test split has 60.
      </p>
    </Frame>
  );
}

export type CurvePoint = { threshold: number; precision: number; coverage: number };

export function CoverageChart({
  curve,
  target,
  chosenThreshold,
}: {
  curve: CurvePoint[];
  target: number;
  chosenThreshold: number | null;
}) {
  const [ref, width] = useWidth();
  const [index, setIndex] = useState<number | null>(null);
  const points = [...curve].sort((a, b) => a.coverage - b.coverage);
  const yMin = Math.min(0.5, Math.floor(Math.min(...points.map((p) => p.precision)) * 10) / 10);
  const x = scale(0, 1, M.left, width - M.right);
  const y = scale(yMin, 1, HEIGHT - M.bottom, M.top);
  const yTicks = [yMin, (yMin + 1) / 2, target, 1].filter((t, i, a) => a.indexOf(t) === i).sort();
  const chosen =
    chosenThreshold === null
      ? null
      : points.filter((p) => p.threshold >= chosenThreshold).sort((a, b) => a.threshold - b.threshold)[0];
  const active = index === null ? null : points[index];

  function nearest(px: number) {
    let best = 0;
    points.forEach((p, i) => {
      if (Math.abs(x(p.coverage) - px) < Math.abs(x(points[best].coverage) - px)) best = i;
    });
    return best;
  }

  return (
    <Frame
      title="Precision against coverage on the test split"
      legend={
        <span className="flex flex-wrap gap-3">
          <LegendLine color="var(--s1)" label="Auto-handled precision" />
          <LegendLine color="var(--muted)" label={`Target ${target.toFixed(2)}`} dashed />
        </span>
      }
    >
      <div ref={ref} className="relative w-full" style={{ height: HEIGHT }}>
        {width > 0 && <svg
          width={width}
          height={HEIGHT}
          role="img"
          tabIndex={0}
          aria-label="Precision of auto-handled messages against the share handled automatically, one point per threshold. Use the arrow keys to step through thresholds."
          className="outline-none focus-visible:ring-2 focus-visible:ring-s1"
          onPointerMove={(event) => {
            const box = event.currentTarget.getBoundingClientRect();
            setIndex(nearest(event.clientX - box.left));
          }}
          onPointerLeave={() => setIndex(null)}
          onBlur={() => setIndex(null)}
          onKeyDown={(event) => {
            if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
            event.preventDefault();
            const step = event.key === "ArrowRight" ? 1 : -1;
            setIndex((i) => Math.min(points.length - 1, Math.max(0, (i ?? points.length - 1) + step)));
          }}
        >
          <Axes width={width} x={x} y={y} xTicks={[0, 0.25, 0.5, 0.75, 1]} yTicks={yTicks} xLabel="Coverage (share handled automatically)" yLabel="Precision" />
          <line x1={x(0)} x2={x(1)} y1={y(target)} y2={y(target)} stroke="var(--muted)" strokeDasharray="4 4" />
          <polyline
            points={points.map((p) => `${x(p.coverage)},${y(p.precision)}`).join(" ")}
            fill="none"
            stroke="var(--s1)"
            strokeWidth={2}
            strokeLinejoin="round"
          />
          {chosen && (
            <g>
              <line
                x1={x(chosen.coverage)}
                x2={x(chosen.coverage)}
                y1={y(chosen.precision)}
                y2={HEIGHT - M.bottom}
                stroke="var(--axis)"
                strokeDasharray="2 3"
              />
              <circle cx={x(chosen.coverage)} cy={y(chosen.precision)} r={5} fill="var(--s1)" stroke="var(--surface)" strokeWidth={2} />
              <text
                x={x(chosen.coverage) - 6}
                y={HEIGHT - M.bottom - 8}
                textAnchor="end"
                fontSize={11}
                fill="var(--ink)"
                className="tabular"
              >
                chosen threshold {chosenThreshold?.toFixed(3)}
              </text>
            </g>
          )}
          {active && (
            <g pointerEvents="none">
              <line x1={x(active.coverage)} x2={x(active.coverage)} y1={M.top} y2={HEIGHT - M.bottom} stroke="var(--axis)" />
              <circle cx={x(active.coverage)} cy={y(active.precision)} r={5} fill="var(--s1)" stroke="var(--surface)" strokeWidth={2} />
            </g>
          )}
        </svg>}
        <Tooltip
          tip={
            active && {
              x: x(active.coverage),
              y: y(active.precision),
              value: `precision ${active.precision.toFixed(3)} · coverage ${active.coverage.toFixed(3)}`,
              label: `threshold ${active.threshold.toFixed(3)}`,
            }
          }
        />
      </div>
      <p className="text-xs text-muted">
        Moving the threshold down handles more messages automatically at lower precision. The
        threshold was chosen on the calibration split, not on this curve.
      </p>
    </Frame>
  );
}
