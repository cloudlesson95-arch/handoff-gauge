import ReviewQueue from "@/components/ReviewQueue";

export default function ReviewPage() {
  return (
    <div className="space-y-6">
      <div className="max-w-3xl space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">Review queue</h1>
        <p className="text-sm text-ink-2">
          Messages the agent handed off, with its reasons. Pick the right intent to close an item.
          The queue is kept in this browser only and holds the redacted text.
        </p>
      </div>
      <ReviewQueue />
    </div>
  );
}
