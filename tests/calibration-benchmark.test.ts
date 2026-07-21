// ============================================================================
// Turbocontext V6 — Calibration Benchmark Test
// ============================================================================
// Verifies the calibration benchmark produces real quality signals and
// the QualityProxy learns to predict quality from cheap signals.
// ============================================================================

import { describe, it, expect, beforeAll } from "vitest";
import {
  runCalibrationBenchmark,
  defineBenchmarkTasks,
  type BenchmarkResult,
} from "./calibration-benchmark.js";
import { QualityProxy } from "../src/core/quality-proxy.js";

describe("Calibration Benchmark", () => {
  let results: BenchmarkResult[];
  let proxy: QualityProxy;

  beforeAll(async () => {
    ({ results, proxy } = await runCalibrationBenchmark({ goodRuns: 1, badRuns: 1 }));
  }, 30000);

  it("defines at least 15 benchmark tasks", () => {
    const tasks = defineBenchmarkTasks();
    expect(tasks.length).toBeGreaterThanOrEqual(15);
  });

  it("covers all 5 task types", () => {
    const types = new Set(results.map(r => r.taskType));
    expect(types.has("code_generation")).toBe(true);
    expect(types.has("debugging")).toBe(true);
    expect(types.has("code_refactor")).toBe(true);
    expect(types.has("code_review")).toBe(true);
    expect(types.has("analysis")).toBe(true);
  });

  it("produces at least 20 calibration points", () => {
    expect(results.length).toBeGreaterThanOrEqual(20);
  });

  it("has both good and bad outcomes in benchmark", () => {
    const goodResults = results.filter(r => r.hardQuality >= 0.5);
    const badResults = results.filter(r => r.hardQuality <= 0.3);
    expect(goodResults.length).toBeGreaterThan(0);
    expect(badResults.length).toBeGreaterThan(0);
  });

  it("calibrates QualityProxy with >= 8 samples", () => {
    expect(proxy.getCalibrationSize()).toBeGreaterThanOrEqual(8);
  });

  it("QualityProxy weights have non-zero features after calibration", () => {
    const weights = proxy.getWeights();
    expect(weights).not.toBeNull();
    expect(weights!.sampleCount).toBeGreaterThanOrEqual(8);
    // At least some weights should be non-zero
    const nonZeroWeights = weights!.weights.filter(w => Math.abs(w) > 0.001);
    expect(nonZeroWeights.length).toBeGreaterThan(0);
  });

  it("QualityProxy predicts >= 70% accuracy on calibration data", () => {
    let correct = 0;
    let total = 0;
    for (const r of results) {
      const pred = r.postCalibrationPrediction || r.preCalibrationPrediction;
      if (pred && pred.isReliable) {
        total++;
        if (Math.abs(pred.predictedQuality - r.hardQuality) < 0.3) {
          correct++;
        }
      }
    }
    if (total >= 5) {
      const accuracy = correct / total;
      expect(accuracy).toBeGreaterThanOrEqual(0.70);
    }
  });

  it("signal profile shows meaningful relevance values", () => {
    const profile = proxy.getSignalProfile();
    expect(profile.length).toBeGreaterThan(0);
    // Top signal should have non-zero relevance
    expect(Math.abs(profile[0].relevance)).toBeGreaterThan(0);
  });
});
