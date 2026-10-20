import { CoverageChart, ReliabilityChart } from "@/components/Charts";
import report from "@/reports/calibration.json";

const fmt = (v: number | null, digits = 3) => (v === null ? "none" : v.toFixed(digits));

function Tile({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div className="rounded-xl border border-line bg-surface p-4">
      <p className="text-xs text-ink-2">{label}</p>
      <p className="mt-1 text-2xl font-semibold">{value}</p>
      {detail && <p className="mt-1 text-xs text-muted">{detail}</p>}
    </div>
  );
}

export default function CalibrationPage() {
  const h = report.headline;
  const chosen = report.chosen;
  return (
    <div className="space-y-6">
      <div className="max-w-3xl space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">Calibration</h1>
        <p className="text-sm text-ink-2">
          {report.splits.calibration} Banking77 messages fit the calibration maps and the threshold;{" "}
          {report.splits.test} held-out messages are reported here. The model ({report.model}) is
          sampled {report.n_samples} times per message at temperature {report.temperature}. The
          threshold is the lowest one whose auto-handled precision reaches{" "}
          {report.target_precision.toFixed(2)} at the 90% lower confidence bound.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 md:grid-cols-4">
        <Tile label="ECE, stated → calibrated" value={`${fmt(h.ece_before)} → ${fmt(h.ece_after)}`} detail="expected calibration error, lower is better" />
        <Tile label="Precision on auto-handled" value={fmt(h.precision_auto)} detail={`${h.n_auto} of ${report.splits.test} test messages, target ${report.target_precision.toFixed(2)}`} />
        <Tile label="Coverage" value={`${(h.coverage * 100).toFixed(0)}%`} detail={`handoff rate ${(h.handoff_rate * 100).toFixed(0)}%`} />
        <Tile label="Threshold" value={fmt(chosen.threshold)} detail={`${chosen.estimator} + ${chosen.method}`} />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ReliabilityChart
          before={report.reliability.before}
          after={report.reliability.after}
          eceBefore={h.ece_before}
          eceAfter={h.ece_after}
        />
        <CoverageChart curve={report.curve} target={report.target_precision} chosenThreshold={chosen.threshold} />
      </div>

      <section className="space-y-2">
        <h2 className="text-sm font-semibold">Every estimator and calibration method</h2>
        <p className="max-w-3xl text-xs text-ink-2">
          The chosen pair has the most cross-validated coverage among those whose cross-validated
          precision meets the target. Self-consistency alone has only three possible scores with
          three samples, so no threshold on it reaches the target with a margin. Accuracy of the
          majority answer on the test split is {fmt(h.accuracy)}.
        </p>
        <div className="overflow-x-auto rounded-xl border border-line bg-surface">
          <table className="tabular w-full min-w-[720px] text-left text-xs">
            <thead className="text-ink-2">
              <tr className="border-b border-line">
                <th className="px-3 py-2 font-medium">Estimator</th>
                <th className="px-3 py-2 font-medium">Method</th>
                <th className="px-3 py-2 text-right font-medium">ECE raw → cal</th>
                <th className="px-3 py-2 text-right font-medium">Brier raw → cal</th>
                <th className="px-3 py-2 text-right font-medium">CV precision</th>
                <th className="px-3 py-2 text-right font-medium">CV coverage</th>
                <th className="px-3 py-2 text-right font-medium">Test precision</th>
                <th className="px-3 py-2 text-right font-medium">Test coverage</th>
                <th className="px-3 py-2 text-right font-medium">Threshold</th>
              </tr>
            </thead>
            <tbody>
              {report.runs.map((run) => {
                const isChosen = run.estimator === chosen.estimator && run.method === chosen.method;
                return (
                  <tr key={`${run.estimator}-${run.method}`} className={`border-b border-line last:border-0 ${isChosen ? "bg-wash font-semibold" : ""}`}>
                    <td className="px-3 py-1.5">
                      {run.estimator}
                      {isChosen && <span className="ml-1.5 text-good-text">✓ chosen</span>}
                    </td>
                    <td className="px-3 py-1.5">{run.method}</td>
                    <td className="px-3 py-1.5 text-right">{fmt(run.ece_raw)} → {fmt(run.ece)}</td>
                    <td className="px-3 py-1.5 text-right">{fmt(run.brier_raw)} → {fmt(run.brier)}</td>
                    <td className="px-3 py-1.5 text-right">{run.threshold === null ? "—" : fmt(run.cv_precision, 2)}</td>
                    <td className="px-3 py-1.5 text-right">{run.threshold === null ? "—" : fmt(run.cv_coverage, 2)}</td>
                    <td className="px-3 py-1.5 text-right">{run.threshold === null ? "—" : fmt(run.test_precision, 2)}</td>
                    <td className="px-3 py-1.5 text-right">{run.threshold === null ? "—" : fmt(run.test_coverage, 2)}</td>
                    <td className="px-3 py-1.5 text-right">{fmt(run.threshold)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-muted">
          Each row is one MLflow run in the <code>calibration</code> experiment. CV columns are
          5-fold cross-validation on the calibration split; test columns use the held-out split.
        </p>
      </section>
    </div>
  );
}
