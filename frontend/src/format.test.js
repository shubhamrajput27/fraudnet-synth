import { describe, expect, it } from "vitest";
import { armName, bankName, errorText, fmt, parsePm, parseRow, pct } from "./format.js";
import { REASON_TERM, TERMS } from "./glossary.js";

const FIELDS = ["Time", ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`), "Amount"];

describe("formatters", () => {
  it("shows a dash for missing values instead of 'NaN' or '0'", () => {
    expect(fmt(null)).toBe("—");
    expect(fmt(Number.NaN)).toBe("—");
    expect(pct(undefined)).toBe("—");
    expect(fmt(0.81234, 3)).toBe("0.812");
  });
  it("names banks", () => expect(bankName("bank_c")).toBe("Bank C"));
  it("parses 'mean ± std' strings from the results CSV", () => {
    expect(parsePm("0.8093 ± 0.0048")).toEqual({ mean: 0.8093, std: 0.0048 });
    expect(parsePm("0.5")).toEqual({ mean: 0.5, std: 0 });
  });
});

describe("parseRow (paste a CSV line on the predict page)", () => {
  const row = Array.from({ length: 30 }, (_, i) => i + 0.5);
  it("maps 30 values to Time, V1..V28, Amount", () => {
    const tx = parseRow(row.join(","), FIELDS);
    expect(tx.Time).toBe("0.5");
    expect(tx.V28).toBe("28.5");
    expect(tx.Amount).toBe("29.5");
  });
  it("drops a trailing Class column so the model never sees the answer", () => {
    expect(parseRow([...row, 1].join(","), FIELDS).Amount).toBe("29.5");
  });
  it("rejects the wrong number of values or non-numbers", () => {
    expect(() => parseRow("1,2,3", FIELDS)).toThrow(/expected 30/);
    expect(() => parseRow([...row.slice(1), "abc"].join(","), FIELDS)).toThrow(/not a number/);
  });
});

describe("glossary", () => {
  it("explains every rejection reason the Overview table shows", () => {
    for (const key of Object.values(REASON_TERM)) expect(TERMS[key]).toBeDefined();
    expect(Object.keys(REASON_TERM)).toHaveLength(6);
  });
});

describe("armName (one naming scheme on every page)", () => {
  it("turns runner keys and numbers into the on-screen names", () => {
    expect(armName("arm3_federated_real")).toBe("Arm 3 · Federated, real");
    expect(armName("arm6_central_aug")).toBe("Arm 6 · Centralized, augmented");
    expect(armName(4)).toBe("Arm 4 · Federated, augmented");
  });
  it("leaves unknown keys alone instead of guessing", () => expect(armName("something")).toBe("something"));
});

describe("errorText (readable API errors, never raw JSON)", () => {
  it("summarises FastAPI's 422 validation list", () => {
    const body = { detail: [{ loc: ["body", "seeds"], msg: "List should have at least 1 item after validation, not 0" }] };
    expect(errorText(body)).toBe("seeds: List should have at least 1 item after validation, not 0");
  });
  it("passes plain messages through", () => {
    expect(errorText({ detail: "unknown job" })).toBe("unknown job");
    expect(errorText({ error: "requires role: admin" })).toBe("requires role: admin");
    expect(errorText(null)).toBeNull();
  });
});
