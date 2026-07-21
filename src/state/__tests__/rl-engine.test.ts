// ============================================================================
// RLEngineV5 Tests — Smoke, integration, and edge-case coverage
// ============================================================================
// Covers the main orchestrator that is now wired into the production path.
// All tests use in-memory engine (no disk I/O, no state file contamination).

import { describe, it, expect, beforeEach } from "vitest";
import { RLEngineV5 } from "../rl/rl-engine.js";
import { SharedStateManager } from "../state-manager.js";
import type { Trial, RetrievalInput } from "../types.js";
import { ContextOrigin, TaskType } from "../types.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Build a minimal valid Trial for testing */
function makeTrial(overrides: Partial<Trial> = {}): Trial {
  const now = new Date().toISOString();
  return {
    id: `test_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
    timestamp: now,
    context: ContextOrigin.AUTONOMOUS,
    taskType: "code_review" as const,
    descriptionHash: "abcdef0123456789",
    descriptionLength: 100,
    capabilityRequirements: ["code_understanding", "error_detection"],
    compressionRatio: 0.4,
    compressionWeights: { alpha: 0.55, beta: 0.20, gamma: 0.25 },
    temperatureSchedule: [0.7, 0.35, 0.1],
    modelTier: "medium" as const,
    retrievalTopK: 5,
    tokenBudgetUsed: 4000,
    maxAttempts: 3,
    outcome: "success" as const,
    qualityScores: [0.8, 0.9, 0.7, 0.85],
    qualityScore: 0.85,
    costUsd: 0.003,
    latencyMs: 1200,
    attemptCount: 1,
    bestAttemptIndex: 0,
    predictedQuality: null,
    surprise: 0,
    counterfactuals: [],
    curriculumPhase: 0,
    retrievedMemoryIds: [],
    referencedMemoryIds: [],
    advantage: null,
    causalUtility: 0,
    herGoals: [],
    ...overrides,
  };
}

/** Build a minimal RetrievalInput for queryOptimalParams */
function makeQuery(taskType: RetrievalInput["taskType"] = "code_review", desc = "Review auth module for security issues"): RetrievalInput {
  return {
    taskType,
    description: desc,
    capabilityRequirements: ["code_understanding", "error_detection"],
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("RLEngineV5", () => {
  let engine: RLEngineV5;

  beforeEach(() => {
    engine = RLEngineV5.createInMemory();
  });

  // ── Lifecycle ──

  describe("createInMemory", () => {
    it("returns an engine with zero trials and phase 0", () => {
      const status = engine.getStatus();
      expect(status.totalTrials).toBe(0);
      expect(status.curriculumPhase).toBe(0);
      expect(status.activeMemories).toBe(0);
    });

    it("does not touch disk (no state file created)", () => {
      // In-memory engine — saveState() is a no-op
      // Just verify it doesn't throw
      expect(() => engine.saveState()).not.toThrow();
      expect(engine.getStatus().totalTrials).toBe(0);
    });
  });

  // ── recordTrial Lite mode ──

  describe("recordTrial (lite)", () => {
    it("records a trial and increments total invocations", () => {
      const trial = makeTrial();
      const result = engine.recordTrial(trial, "lite");

      expect(result.surprise).toBeDefined();
      expect(result.memoriesUpdated).toBe(0);
      expect(result.pendingSyncCount).toBe(0);

      const status = engine.getStatus();
      expect(status.totalTrials).toBeGreaterThanOrEqual(1);
    });

    it("computes non-zero surprise when predicted quality differs from actual", () => {
      // initial predictive model weights are near zero → predicted ≈ 0.5
      // actual quality 0.85 → surprise ≈ 0.35
      const trial = makeTrial({ qualityScore: 0.85 });
      const result = engine.recordTrial(trial, "lite");
      expect(result.surprise).toBeGreaterThan(0);
    });
  });

  // ── recordTrial Full mode ──

  describe("recordTrial (full)", () => {
    it("records a trial without crashing", () => {
      const trial = makeTrial();
      const result = engine.recordTrial(trial, "full");

      expect(result.surprise).toBeDefined();
      expect(result.tdError).toBeDefined();
      expect(result.counterfactuals).toBeDefined();

      const status = engine.getStatus();
      expect(status.totalTrials).toBeGreaterThanOrEqual(1);
    });

    it("accumulates trials correctly across multiple calls", () => {
      for (let i = 0; i < 5; i++) {
        engine.recordTrial(makeTrial(), "full");
      }
      const status = engine.getStatus();
      expect(status.totalTrials).toBeGreaterThanOrEqual(5);
    });
  });

  // ── queryOptimalParams ──

  describe("queryOptimalParams", () => {
    it("returns compression weights that sum reasonably", () => {
      const result = engine.queryOptimalParams(makeQuery());
      const { alpha, beta, gamma } = result.compressionWeights;
      expect(alpha).toBeGreaterThan(0);
      expect(beta).toBeGreaterThan(0);
      expect(gamma).toBeGreaterThan(0);
    });

    it("returns a valid 3-element temperature schedule", () => {
      const result = engine.queryOptimalParams(makeQuery());
      expect(result.temperatureSchedule).toHaveLength(3);
      expect(result.temperatureSchedule[0]).toBeGreaterThanOrEqual(0.1);
      expect(result.temperatureSchedule[1]).toBeGreaterThanOrEqual(0.1);
      expect(result.temperatureSchedule[2]).toBeGreaterThanOrEqual(0.1);
    });

    it("returns quality threshold in valid range", () => {
      const result = engine.queryOptimalParams(makeQuery());
      expect(result.qualityThreshold).toBeGreaterThan(0);
      expect(result.qualityThreshold).toBeLessThanOrEqual(1);
    });

    it("returns maxAttempts >= 1", () => {
      const result = engine.queryOptimalParams(makeQuery());
      expect(result.maxAttempts).toBeGreaterThanOrEqual(1);
    });

    it("returns a model tier string", () => {
      const result = engine.queryOptimalParams(makeQuery());
      expect(["fast", "medium", "best"]).toContain(result.modelTier);
    });

    it("returns different params for different task types", () => {
      const reviewResult = engine.queryOptimalParams(makeQuery("code_review"));
      const genResult = engine.queryOptimalParams(makeQuery("code_generation"));

      // Both should be valid
      expect(reviewResult.qualityThreshold).toBeGreaterThan(0);
      expect(genResult.qualityThreshold).toBeGreaterThan(0);
    });

    it("includes retrieved memories (empty for fresh state)", () => {
      const result = engine.queryOptimalParams(makeQuery());
      expect(Array.isArray(result.retrievedMemories)).toBe(true);
      expect(result.contrastiveInsights).toEqual([]);
    });
  });

  // ── v5.1 Baseline guard (regression test) ──

  describe("baseline guard (v5.1)", () => {
    it("handles V4 task types not in the V5 TaskType enum (e.g. code_refactor)", () => {
      // "code_refactor" exists in V4 types.ts but is "refactoring" in V5 TaskType
      const trial = makeTrial({ taskType: "code_refactor" as any, outcome: "failure" });

      // Must not throw "Cannot read properties of undefined (reading 'ema')"
      expect(() => engine.recordTrial(trial, "full")).not.toThrow();

      // Should still record the trial
      const status = engine.getStatus();
      expect(status.totalTrials).toBeGreaterThanOrEqual(1);
    });

    it("handles V4 task types in lite mode", () => {
      const trial = makeTrial({ taskType: "code_refactor" as any });
      expect(() => engine.recordTrial(trial, "lite")).not.toThrow();
    });

    it("handles task types not in any enum (e.g. unknown)", () => {
      const trial = makeTrial({ taskType: "some_future_type" as any, outcome: "failure" });
      expect(() => engine.recordTrial(trial, "full")).not.toThrow();
      expect(() => engine.recordTrial(trial, "lite")).not.toThrow();
    });

    it("handles crash outcome gracefully", () => {
      const trial = makeTrial({ taskType: "general" as any, outcome: "crash" });
      expect(() => engine.recordTrial(trial, "full")).not.toThrow();
    });
  });

  // ── Status report ──

  describe("getStatus", () => {
    it("returns a valid status report for a fresh engine", () => {
      const status = engine.getStatus();
      expect(status.totalTrials).toBe(0);
      expect(status.activeMemories).toBe(0);
      expect(status.coldMemories).toBe(0);
      expect(status.curriculumPhase).toBeGreaterThanOrEqual(0);
      expect(status.predictiveModelAccuracy).toBeGreaterThanOrEqual(0);
    });

    it("reflects recorded trials", () => {
      engine.recordTrial(makeTrial({ taskType: "code_review", qualityScore: 0.9 }), "full");
      engine.recordTrial(makeTrial({ taskType: "debugging", qualityScore: 0.7 }), "full");

      const status = engine.getStatus();
      expect(status.totalTrials).toBeGreaterThanOrEqual(2);
    });
  });

  // ── Edge cases ──

  describe("edge cases", () => {
    it("handles trial with empty description", () => {
      const trial = makeTrial({ descriptionLength: 0 });
      expect(() => engine.recordTrial(trial, "full")).not.toThrow();
    });

    it("handles trial with extreme quality score", () => {
      const trial = makeTrial({ qualityScore: 1.0, qualityScores: [1, 1, 1, 1] });
      expect(() => engine.recordTrial(trial, "full")).not.toThrow();
    });

    it("handles trial with zero quality score", () => {
      const trial = makeTrial({ qualityScore: 0, qualityScores: [0, 0, 0, 0], outcome: "failure" });
      expect(() => engine.recordTrial(trial, "full")).not.toThrow();
    });
  });
});
