// ============================================================================
// Policy Manager Tests
// ============================================================================
import { describe, it, expect } from "vitest";
import {
  resolveEffectivePolicy,
  applyMutation,
  getParamValue,
  clonePolicy,
  normalizeDimWeights,
} from "../policy/policy-manager.js";
import { DEFAULT_POLICY } from "../constants.js";
import type { PolicyState, PolicyOverrides } from "../types.js";

function makeBasePolicy(): PolicyState {
  return structuredClone(DEFAULT_POLICY);
}

describe("resolveEffectivePolicy", () => {
  it("returns base when no overrides provided", () => {
    const base = makeBasePolicy();
    const result = resolveEffectivePolicy(base);
    expect(result.compression.alpha).toBe(base.compression.alpha);
    expect(result.quality.threshold).toBe(base.quality.threshold);
  });

  it("returns base when overrides is undefined", () => {
    const base = makeBasePolicy();
    const result = resolveEffectivePolicy(base, undefined);
    expect(result).toBe(base); // same reference when no overrides
  });

  it("merges compression override at leaf level", () => {
    const base = makeBasePolicy();
    const overrides: Partial<PolicyOverrides> = {
      compression: { alpha: 0.80 },
    };
    const result = resolveEffectivePolicy(base, overrides);
    expect(result.compression.alpha).toBe(0.80);
    // other compression fields unchanged
    expect(result.compression.beta).toBe(base.compression.beta);
    expect(result.compression.gamma).toBe(base.compression.gamma);
  });

  it("merges quality override", () => {
    const base = makeBasePolicy();
    const overrides: Partial<PolicyOverrides> = {
      quality: { threshold: 0.90, maxAttempts: 5 },
    };
    const result = resolveEffectivePolicy(base, overrides);
    expect(result.quality.threshold).toBe(0.90);
    expect(result.quality.maxAttempts).toBe(5);
  });

  it("merges temperature override", () => {
    const base = makeBasePolicy();
    const overrides: Partial<PolicyOverrides> = {
      temperature: { t0: 0.80 },
    };
    const result = resolveEffectivePolicy(base, overrides);
    expect(result.temperature.t0).toBe(0.80);
    expect(result.temperature.t1).toBe(base.temperature.t1);
  });

  it("merges multiple overrides at once", () => {
    const base = makeBasePolicy();
    const overrides: Partial<PolicyOverrides> = {
      compression: { alpha: 0.40 },
      retrieval: { topK: 10 },
    };
    const result = resolveEffectivePolicy(base, overrides);
    expect(result.compression.alpha).toBe(0.40);
    expect(result.retrieval.topK).toBe(10);
  });

  it("does NOT mutate the original base", () => {
    const base = makeBasePolicy();
    const originalAlpha = base.compression.alpha;
    resolveEffectivePolicy(base, { compression: { alpha: 0.99 } });
    expect(base.compression.alpha).toBe(originalAlpha);
  });
});

describe("applyMutation", () => {
  it("sets a top-level path correctly", () => {
    const policy = makeBasePolicy();
    const result = applyMutation(policy, "compression.alpha", 0.75);
    expect(result.compression.alpha).toBe(0.75);
    expect(policy.compression.alpha).not.toBe(0.75); // original unchanged
  });

  it("sets a two-level nested path", () => {
    const policy = makeBasePolicy();
    const result = applyMutation(policy, "retrieval.mmrLambda", 0.55);
    expect(result.retrieval.mmrLambda).toBe(0.55);
  });

  it("sets a three-level nested path", () => {
    const policy = makeBasePolicy();
    const result = applyMutation(policy, "retrieval.dimWeights.idfOverlap", 0.50);
    expect(result.retrieval.dimWeights.idfOverlap).toBe(0.50);
  });

  it("throws on nonexistent path", () => {
    const policy = makeBasePolicy();
    expect(() => applyMutation(policy, "nonexistent.field" as any, 1))
      .toThrow("Cannot set path");
  });

  it("throws on non-numeric target path", () => {
    const policy = makeBasePolicy();
    expect(() => applyMutation(policy, "retrieval.dimWeights" as any, 1))
      .toThrow(/is not a number/);
  });
});

describe("getParamValue", () => {
  it("reads a simple path", () => {
    const policy = makeBasePolicy();
    expect(getParamValue(policy, "compression.alpha")).toBe(policy.compression.alpha);
  });

  it("reads a nested path", () => {
    const policy = makeBasePolicy();
    expect(getParamValue(policy, "quality.threshold")).toBe(policy.quality.threshold);
  });

  it("throws on nonexistent path", () => {
    const policy = makeBasePolicy();
    expect(() => getParamValue(policy, "nonexistent.field" as any))
      .toThrow();
  });
});

describe("clonePolicy", () => {
  it("produces a deep copy", () => {
    const policy = makeBasePolicy();
    const cloned = clonePolicy(policy);
    expect(cloned).toEqual(policy);
    expect(cloned).not.toBe(policy);

    cloned.compression.alpha = 0.99;
    expect(policy.compression.alpha).not.toBe(0.99);
  });
});

describe("normalizeDimWeights", () => {
  it("normalizes weights to sum to 1", () => {
    const weights = { a: 1, b: 2, c: 3 };
    const result = normalizeDimWeights(weights);
    const sum = Object.values(result).reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 10);
  });

  it("returns uniform weights when sum is 0", () => {
    const weights = { a: 0, b: 0, c: 0 };
    const result = normalizeDimWeights(weights);
    expect(result.a).toBeCloseTo(1 / 3, 10);
    expect(result.b).toBeCloseTo(1 / 3, 10);
    expect(result.c).toBeCloseTo(1 / 3, 10);
  });

  it("does not mutate the input", () => {
    const weights = { a: 1, b: 1 };
    normalizeDimWeights(weights);
    expect(weights.a).toBe(1); // unchanged
  });
});
