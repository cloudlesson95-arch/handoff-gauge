// Mirrors handoff_gauge/config.py INTENTS.
export const INTENTS: Record<string, string> = {
  card_arrival: "Card has not arrived",
  card_delivery_estimate: "Card delivery time",
  lost_or_stolen_card: "Lost or stolen card",
  compromised_card: "Compromised card",
  pending_transfer: "Pending transfer",
  transfer_not_received_by_recipient: "Transfer not received",
  transaction_charged_twice: "Charged twice",
  card_payment_not_recognised: "Payment not recognised",
  request_refund: "Request a refund",
  refund_not_showing_up: "Refund not showing up",
};

export function intentLabel(intent: string | null): string {
  if (!intent) return "No intent";
  return INTENTS[intent] ?? intent;
}
