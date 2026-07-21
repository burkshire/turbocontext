// ============================================================================
// Thompson Sampling Tests
// ============================================================================
import { describe, it, expect } from "vitest";
import { sampleBeta, sampleGamma, gaussianRandom } from "../rl/thompson.js";

describe("gaussianRandom", () => {
  it("returns a finite number", () => {
    const x = gaussianRandom();
    expect(Number.isFinite(x)).toBe(true);
  });

  it("produces values with both signs over many samples", () => {
    let positive = 0, negative = 0;
    for (let i = 0; i < 1000; i++) {
      const x = gaussianRandom();
      if (x > 0) positive++;
      if (x < 0) negative++;
    }
    expect(positive).toBeGreaterThan(200);
    expect(negative).toBeGreaterThan(200);
  });
});

describe("sampleGamma", () => {
  it("returns finite positive values for shape >= 1", () => {
    for (let i = 0; i < 100; i++) {
      const x = sampleGamma(2.0);
      expect(Number.isFinite(x)).toBe(true);
      expect(x).toBeGreaterThan(0);
    }
  });

  it("returns finite positive values for shape < 1", () => {
    for (let i = 0; i < 100; i++) {
      const x = sampleGamma(0.5);
      expect(Number.isFinite(x)).toBe(true);
      expect(x).toBeGreaterThan(0);
    }
  });

  it("produces values with reasonable mean for Gamma(5,1)", () => {
    const n = 1000;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += sampleGamma(5);
    }
    const mean = sum / n;
    // Gamma(5,1) mean = 5, std = sqrt(5) ≈ 2.24
    expect(mean).toBeGreaterThan(4.0);
    expect(mean).toBeLessThan(6.0);
  });
});

describe("sampleBeta", () => {
  it("always returns values in [0, 1]", () => {
    for (let i = 0; i < 200; i++) {
      const x = sampleBeta(1, 1);
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(1);
    }
  });

  it("Beta(1,1) is approximately uniform with mean ~0.5", () => {
    const n = 1000;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += sampleBeta(1, 1);
    }
    const mean = sum / n;
    expect(mean).toBeGreaterThan(0.45);
    expect(mean).toBeLessThan(0.55);
  });

  it("Beta(10,2) is peaked near 0.83", () => {
    const n = 500;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += sampleBeta(10, 2);
    }
    const mean = sum / n;
    // True mean: 10/(10+2) = 0.833
    expect(mean).toBeGreaterThan(0.78);
    expect(mean).toBeLessThan(0.88);
  });

  it("Beta(2,10) is peaked near 0.17", () => {
    const n = 500;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += sampleBeta(2, 10);
    }
    const mean = sum / n;
    // True mean: 2/(2+10) = 0.167
    expect(mean).toBeGreaterThan(0.12);
    expect(mean).toBeLessThan(0.22);
  });

  it("handles edge case alpha=0, beta=0 (clamped to 0.1)", () => {
    for (let i = 0; i < 50; i++) {
      const x = sampleBeta(0, 0);
      expect(Number.isFinite(x)).toBe(true);
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(1);
    }
  });

  it("handles alpha=100, beta=1 (high confidence success)", () => {
    const n = 200;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += sampleBeta(100, 1);
    }
    const mean = sum / n;
    expect(mean).toBeGreaterThan(0.95);
  });
});
