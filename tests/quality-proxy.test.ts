// ============================================================================
// V6 Quality Proxy Tests — PACE-inspired signal extraction + regression
// ============================================================================

import { describe, it, expect, beforeEach } from "vitest";
import { extractSignals, signalVectorToArray, SIGNAL_DIMENSION } from "../src/core/signal-extractor.js";
import { QualityProxy } from "../src/core/quality-proxy.js";

// ============================================================================
// Signal Extractor Tests
// ============================================================================

describe("Signal Extractor", () => {
  it("extracts all 8 signal dimensions from output text", () => {
    const output = "```typescript\nexport function hello() { return 'world'; }\n```\n\nThe implementation uses proper error handling.";
    const taskDesc = "implement hello function with error handling";
    const signals = extractSignals(output, taskDesc);

    expect(signals.codeBlockCount).toBe(1);
    expect(signals.hasErrorPatterns).toBe(1); // clean output
    expect(signals.keywordCoverage).toBeGreaterThan(0);
    expect(signals.structuralScore).toBeGreaterThanOrEqual(0.7);
  });

  it("detects error patterns in output", () => {
    const output = "Sorry, I cannot implement this. TODO: FIXME placeholder. This is incomplete.";
    const signals = extractSignals(output, "implement feature");

    expect(signals.hasErrorPatterns).toBe(0); // error patterns detected
  });

  it("detects compilation failure from execution metrics", () => {
    const signals = extractSignals("code here", "task", {
      compiled: false,
      compilerErrors: 3,
      smokeTestPassed: false,
    });

    expect(signals.compilationSuccess).toBe(0);
    expect(signals.testPassRate).toBe(0);
  });

  it("detects compilation success from execution metrics", () => {
    const signals = extractSignals("code here", "task", {
      compiled: true,
      smokeTestPassed: true,
    });

    expect(signals.compilationSuccess).toBe(1);
    expect(signals.testPassRate).toBe(1);
  });

  it("signalVectorToArray returns correct dimension", () => {
    const signals = extractSignals("output text", "task description");
    const arr = signalVectorToArray(signals);

    expect(arr).toHaveLength(SIGNAL_DIMENSION);
    arr.forEach(v => expect(v).toBeGreaterThanOrEqual(0));
  });

  it("computes attempt efficiency correctly", () => {
    const s1 = extractSignals("out", "task", undefined, 1);
    const s2 = extractSignals("out", "task", undefined, 3);

    expect(s1.attemptEfficiency).toBe(1);
    expect(s2.attemptEfficiency).toBeCloseTo(1 / 3, 1);
  });

  it("detects unbalanced code blocks", () => {
    const output = "```typescript\ncode here\n"; // unclosed block
    const signals = extractSignals(output, "task");

    expect(signals.structuralScore).toBeLessThan(0.8);
  });
});

// ============================================================================
// Quality Proxy Tests
// ============================================================================

describe("Quality Proxy", () => {
  let proxy: QualityProxy;

  beforeEach(() => {
    proxy = new QualityProxy({ minSamplesForFit: 5, bootstrapSamples: 50 });
  });

  it("starts with no calibration data", () => {
    expect(proxy.getCalibrationSize()).toBe(0);
    expect(proxy.getWeights()).toBeNull();
  });

  it("predicts with heuristic fallback when uncalibrated", () => {
    const result = proxy.predict(
      "function hello() { return 'world'; }",
      "implement hello function",
      "code_generation",
    );

    expect(result.predictedQuality).toBeGreaterThan(0);
    expect(result.predictedQuality).toBeLessThanOrEqual(1);
    expect(result.isReliable).toBe(false); // no calibration yet
  });

  it("fits regression weights after sufficient calibration", () => {
    // Add calibration points with clear signal→quality pattern:
    // high compilation + clean output → high quality
    for (let i = 0; i < 10; i++) {
      const compiled = i >= 3; // first 3 fail, rest succeed
      const hasErrors = i < 2; // first 2 have error patterns
      const hardQuality = compiled ? (hasErrors ? 0.4 : 0.9) : 0.1;

      proxy.calibrate(
        compiled ? "function hello() { return true; }" : "function hello() { return }",
        "implement hello function",
        "code_generation",
        hardQuality,
        { compiled, smokeTestPassed: compiled && !hasErrors, compilerErrors: compiled ? 0 : 1 },
      );
    }

    expect(proxy.getCalibrationSize()).toBe(10);
    expect(proxy.getWeights()).not.toBeNull();

    const weights = proxy.getWeights()!;
    expect(weights.weights).toHaveLength(SIGNAL_DIMENSION);
    expect(weights.sampleCount).toBe(10);

    // Compilation success signal should have strongest relevance
    // (it was the best discriminator in our calibration data)
    expect(weights.relevance[0]).toBeDefined(); // compilation signal exists
  });

  it("predicts with learned weights after calibration", () => {
    // Train on simple pattern
    for (let i = 0; i < 10; i++) {
      proxy.calibrate(
        "function x() { return 1; }",
        "task",
        "code_generation",
        i >= 5 ? 0.9 : 0.2,
        { compiled: i >= 5, smokeTestPassed: i >= 5 },
      );
    }

    // Predict on a "good" output
    const goodResult = proxy.predict(
      "function y() { return 42; }",
      "task",
      "code_generation",
      { compiled: true, smokeTestPassed: true },
    );

    expect(goodResult.isReliable).toBe(true);
    expect(goodResult.predictedQuality).toBeGreaterThan(0.5);
    expect(goodResult.confidenceLow).toBeLessThanOrEqual(goodResult.predictedQuality);
    expect(goodResult.confidenceHigh).toBeGreaterThanOrEqual(goodResult.predictedQuality);

    // Predict on a "bad" output
    const badResult = proxy.predict(
      "sorry I don't know TODO",
      "task",
      "code_generation",
      { compiled: false, smokeTestPassed: false },
    );

    // Bad output should score lower than good output
    expect(badResult.predictedQuality).toBeLessThan(goodResult.predictedQuality);
  });

  it("provides signal relevance profile (PACE Fig 3 equivalent)", () => {
    for (let i = 0; i < 10; i++) {
      proxy.calibrate(
        "output",
        "task",
        "code_generation",
        i >= 5 ? 0.9 : 0.3,
        { compiled: i >= 5, smokeTestPassed: i >= 5 },
      );
    }

    const profile = proxy.getSignalProfile("code_generation");
    expect(profile.length).toBeGreaterThan(0);
    // Profile should be sorted by relevance (descending)
    for (let i = 1; i < profile.length; i++) {
      expect(profile[i - 1].relevance).toBeGreaterThanOrEqual(profile[i].relevance);
    }

    // Execution signals should dominate for code_generation
    const topSignals = profile.slice(0, 3).map(s => s.signal);
    expect(topSignals).toContain("Compilation");
  });

  it("evicts old calibration points when over capacity", () => {
    const smallProxy = new QualityProxy({ maxCalibrationPoints: 15, minSamplesForFit: 5 });

    for (let i = 0; i < 30; i++) {
      smallProxy.calibrate(
        "output",
        "task",
        "code_review",
        0.5,
        undefined,
      );
    }

    expect(smallProxy.getCalibrationSize()).toBeLessThanOrEqual(15);
  });

  it("handles zero calibration gracefully", () => {
    const result = proxy.predict("", "empty", "general");
    expect(result.predictedQuality).toBeGreaterThanOrEqual(0);
    expect(result.isReliable).toBe(false);
  });
});
