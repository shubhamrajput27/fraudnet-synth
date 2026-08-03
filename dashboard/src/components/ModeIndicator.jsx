// Per-bank mode badge (CLAUDE.md Tier 1: "mode indicators (Augment/Schema/Real-only)"). If an
// arm is "augmented" but Phase 3 validation rejected that bank's synthetic batch, the pipeline
// silently falls back to that bank's real data (ml/common/data.py) — the badge says so honestly
// instead of showing "Augment"/"Schema" for a bank that's actually running on real data only.
const BANK_MODE = { A: "Augment (CTGAN)", B: "Augment (CTGAN)", C: "Schema (LLM)", D: "Schema (LLM)" };

export default function ModeIndicator({ bank, augmented, verdict }) {
  if (!augmented) {
    return <span className="badge badge-real">Real-only</span>;
  }
  const rejected = verdict && verdict !== "PASS";
  return (
    <span className={`badge ${rejected ? "badge-rejected" : "badge-augment"}`}>
      {BANK_MODE[bank]}
      {rejected ? " (rejected -> real-only)" : ""}
    </span>
  );
}
