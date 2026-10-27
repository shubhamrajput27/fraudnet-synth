import { describe, expect, it } from "vitest";
import { empty, reducer } from "./runState.js";

const round = (n) => ({ type: "round", arm: "arm3_federated_real", seed: 42, round: n, globalTestPrAuc: 0.7 });

describe("live run state", () => {
  it("ignores a round that arrives twice (catch-up replay + live stream overlap)", () => {
    let s = reducer(empty, round(1));
    s = reducer(s, round(1));
    s = reducer(s, round(2));
    expect(s.rounds.map((r) => r.round)).toEqual([1, 2]);
  });
  it("replaces a model's result instead of listing it twice", () => {
    const m = { arm: "arm1", seed: 42, model: "bank_a", f1: 0.5 };
    let s = reducer(empty, { type: "models_loaded", models: [m] });
    s = reducer(s, { type: "model_done", ...m, f1: 0.6 });
    expect(s.models).toHaveLength(1);
    expect(s.models[0].f1).toBe(0.6);
  });
  it("keeps only the last 200 progress messages", () => {
    let s = empty;
    for (let i = 0; i < 250; i++) s = reducer(s, { type: "message", message: `m${i}` });
    expect(s.messages).toHaveLength(200);
    expect(s.messages[0]).toBe("m50");
  });
  it("marks the run ended with its final status", () => {
    expect(reducer(empty, { type: "end", status: "succeeded" })).toMatchObject({ ended: true, status: "succeeded" });
  });
});
