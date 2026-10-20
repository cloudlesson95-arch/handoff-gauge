import TriagePanel from "@/components/TriagePanel";

export default function TriagePage() {
  return (
    <div className="space-y-6">
      <div className="max-w-3xl space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">Triage a customer message</h1>
        <p className="text-sm text-ink-2">
          The agent runs the message through guardrails, asks the model three times in parallel,
          and turns the answers into a calibrated confidence. Above the threshold it handles the
          message itself; below it, or when any guardrail trips, a human reviews it.
        </p>
      </div>
      <TriagePanel />
    </div>
  );
}
