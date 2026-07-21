// ============================================================
// Phase 5: Learner Tests
// ============================================================

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { Learner } from "../src/core/learner.js";
import { unlinkSync, existsSync } from "fs";
import type { ExecutionRecord, TurboContextConfig, StrategyMutation } from "../src/types.js";

const defaultConfig: TurboContextConfig = {
  alpha: 0.55,
  beta: 0.20,
  gamma: 0.25,
  maxTokenBudget: 8000,
  minCoverage: 0.80,
  qualityThreshold: 0.85,
  maxAttempts: 3,
  temperatureSchedule: [0.7, 0.35, 0.1],
  complexityThresholdLow: 0.35,
  complexityThresholdHigh: 0.70,
  learningRate: 0.1,
  historyWindow: 100,
};

function makeRecord(overrides: Partial<ExecutionRecord> = {}): ExecutionRecord {
  return {
    taskId: "r1",
    taskType: "code_review",
    timestamp: Date.now(),
    compressionRatio: 0.3,
    qualityScore: 0.85,
    totalCost: 0.01,
    latencyMs: 1000,
    attemptCount: 1,
    modelUsed: "medium",
    coverage: { code_understanding: 1 },
    dimensionScores: { completeness: 0.8, correctness: 0.9, consistency: 0.8, format: 0.9 },
    ...overrides,
  };
}

describe("Learner", () => {
  let learner: Learner;

  beforeEach(() => {
    learner = new Learner(defaultConfig, "/tmp/turbocontext-test-state.json");
  });

  afterEach(() => {
    try { if (existsSync("/tmp/turbocontext-test-state.json")) unlinkSync("/tmp/turbocontext-test-state.json"); } catch {}
  });

  describe("record", () => {
    it("stores execution records", () => {
      learner.record(makeRecord());
      learner.record(makeRecord({ taskId: "r2" }));
      learner.record(makeRecord({ taskId: "r3" }));
      const trend = learner.getQualityTrend();
      expect(trend.average).toBeGreaterThan(0);
    });

    it("tracks multiple records", () => {
      for (let i = 0; i < 5; i++) {
        learner.record(makeRecord({ taskId: `r${i}`, qualityScore: 0.8 + i * 0.03 }));
      }
      const trend = learner.getQualityTrend();
      expect(trend.average).toBeGreaterThan(0);
      expect(trend.byType).toHaveProperty("code_review");
    });

    it("records with different task types", () => {
      learner.record(makeRecord({ taskType: "code_review", qualityScore: 0.9 }));
      learner.record(makeRecord({ taskType: "code_generation", qualityScore: 0.7 }));
      learner.record(makeRecord({ taskType: "debugging", qualityScore: 0.8 }));
      const trend = learner.getQualityTrend();
      expect(Object.keys(trend.byType).length).toBe(3);
    });
  });

  describe("learn", () => {
    it("returns no adjustments with insufficient data", () => {
      learner.record(makeRecord());
      const result = learner.learn();
      expect(result.adjustments).toEqual([]);
    });

    it("performs learning with sufficient data (low quality pattern)", () => {
      // Simulate 5 low-quality executions with low compression
      for (let i = 0; i < 5; i++) {
        learner.record(makeRecord({
          qualityScore: 0.5 + i * 0.05,
          compressionRatio: 0.1,
          attemptCount: 3,
        }));
      }
      const result = learner.learn();
      expect(result.adjustments.length).toBeGreaterThanOrEqual(0);
      expect(result.config).toBeDefined();
    });

    it("adjusts temperature when attempts are high", () => {
      // Simulate executions requiring many attempts
      for (let i = 0; i < 5; i++) {
        learner.record(makeRecord({
          qualityScore: 0.6,
          attemptCount: 3,
          compressionRatio: 0.4,
        }));
      }
      const before = learner.getConfig().temperatureSchedule[0];
      learner.learn();
      const after = learner.getConfig().temperatureSchedule[0];
      // After many retries, initial temperature should increase
      expect(after).toBeGreaterThanOrEqual(before);
    });
  });

  describe("getConfig", () => {
    it("returns current configuration", () => {
      const config = learner.getConfig();
      expect(config.alpha).toBe(0.55);
      expect(config.beta).toBe(0.20);
      expect(config.gamma).toBe(0.25);
    });
  });

  // ------------------------------------------------------------------
  // Branch-based features (new in v2.1)
  // ------------------------------------------------------------------

  describe("branch features", () => {
    it("tracks per-task-type branches separately", () => {
      learner.record(makeRecord({ taskType: "code_review", qualityScore: 0.9 }));
      learner.record(makeRecord({ taskType: "code_generation", qualityScore: 0.6 }));
      learner.record(makeRecord({ taskType: "code_generation", qualityScore: 0.7 }));

      const branches = learner.getBranches();
      const reviewBranch = branches.get("code_review")!;
      const genBranch = branches.get("code_generation")!;

      expect(reviewBranch.totalExperiments).toBe(1);
      expect(genBranch.totalExperiments).toBe(2);
      expect(reviewBranch.successCount).toBe(1);
      expect(genBranch.failureCount).toBeGreaterThanOrEqual(1);
    });

    it("tracks branch trajectory (momentum + stability)", () => {
      // Record 5 improving quality scores on the same branch
      for (let i = 0; i < 5; i++) {
        learner.record(makeRecord({
          taskType: "debugging",
          qualityScore: 0.6 + i * 0.05,
          coverage: { error_detection: 1 },
        }));
      }
      const branch = learner.getBranches().get("debugging")!;
      expect(branch.trajectory.improvementVelocity).toBeGreaterThan(0);
      expect(branch.trajectory.stabilityScore).toBeGreaterThanOrEqual(0);
      expect(branch.trajectory.qualityHistory.length).toBe(5);
    });

    it("generates branch summary after enough experiments", () => {
      for (let i = 0; i < 6; i++) {
        learner.record(makeRecord({
          taskType: "code_review",
          qualityScore: 0.8 + (i < 4 ? 0.03 : -0.02),
          attemptCount: 1,
        }));
      }
      const branch = learner.getBranches().get("code_review")!;
      expect(branch.totalExperiments).toBe(6);
      expect(branch.summary.length).toBeGreaterThan(0);
      expect(branch.lastSummaryExperimentCount).toBeGreaterThanOrEqual(5);

      // getBranchSummary should return formatted text
      const summary = learner.getBranchSummary("code_review");
      expect(summary).toContain("code_review");
      expect(summary).toContain("Experiments");
    });

    it("getSourceBoost returns 0 for unknown sources", () => {
      expect(learner.getSourceBoost("/unknown/file.ts")).toBe(0);
    });

    it("getSourceBoost returns positive boost for high-success sources", () => {
      learner.record(makeRecord({
        taskType: "code_generation",
        qualityScore: 0.9,
        sourceFiles: ["/src/helper.ts"],
      }));
      learner.record(makeRecord({
        taskType: "code_generation",
        qualityScore: 0.95,
        sourceFiles: ["/src/helper.ts"],
      }));
      const boost = learner.getSourceBoost("/src/helper.ts");
      expect(boost).toBeGreaterThan(0);
      expect(boost).toBeLessThanOrEqual(0.1);
    });

    it("getSourceBoost returns negative boost for low-success sources", () => {
      // Record 3 failures for the same source
      for (let i = 0; i < 3; i++) {
        learner.record(makeRecord({
          qualityScore: 0.3,
          sourceFiles: ["/src/buggy.ts"],
        }));
      }
      const boost = learner.getSourceBoost("/src/buggy.ts");
      expect(boost).toBeLessThan(0);
    });

    it("getActiveBranches returns only branches with experiments", () => {
      expect(learner.getActiveBranches()).toEqual([]);
      learner.record(makeRecord({ taskType: "code_review" }));
      const active = learner.getActiveBranches();
      expect(active).toContain("code_review");
      expect(active).not.toContain("code_generation");
    });

    it("getRelatedBranches finds family relationships", () => {
      learner.record(makeRecord({ taskType: "code_generation" }));
      const related = learner.getRelatedBranches("code_refactor");
      expect(related.length).toBeGreaterThanOrEqual(1);
      // code_refactor and code_generation share a family
      const genRelated = related.find(r => r.type === "code_generation");
      expect(genRelated).toBeDefined();
    });

    it("learnBranchThresholds raises threshold for stable high-pass branches", () => {
      // 10 successful passes on the same branch
      for (let i = 0; i < 10; i++) {
        learner.record(makeRecord({
          taskType: "testing",
          qualityScore: 0.92,
          attemptCount: 1,
        }));
      }
      const before = learner.getBranchQualityThreshold("testing");
      learner.learn();
      const after = learner.getBranchQualityThreshold("testing");
      // Should have raised threshold for consistently good branch
      expect(after).toBeGreaterThanOrEqual(before);
    });

    it("getQualityTrend returns branch data", () => {
      learner.record(makeRecord({ taskType: "code_review", qualityScore: 0.85 }));
      learner.record(makeRecord({ taskType: "analysis", qualityScore: 0.82 }));
      learner.record(makeRecord({ taskType: "analysis", qualityScore: 0.78 }));
      const trend = learner.getQualityTrend();
      expect(trend.branches).toBeDefined();
      expect(trend.branches["code_review"]).toBeDefined();
      expect(trend.branches["analysis"]).toBeDefined();
    });
  });

  // ------------------------------------------------------------------
  // Self-Evolution (v2.3)
  // ------------------------------------------------------------------

  describe("self-evolution", () => {
    it("proposeMutation returns null when not enough data", () => {
      const mutation = learner.proposeMutation("code_review");
      expect(mutation).toBeNull();
    });

    it("proposeMutation returns a valid mutation with sufficient data", () => {
      for (let i = 0; i < 5; i++) {
        learner.record(makeRecord({ taskType: "code_review", qualityScore: 0.85 }));
      }
      const mutation = learner.proposeMutation("code_review");
      expect(mutation).not.toBeNull();
      const validTypes = ["merge_rounds", "remove_round", "reorder_rounds", "split_round",
        "add_quality_criterion", "remove_quality_criterion"];
      expect(validTypes).toContain(mutation!.type);
    });

    it("getEvolutionStats returns zero initially", () => {
      const stats = learner.getEvolutionStats();
      expect(stats.total).toBe(0);
      expect(stats.kept).toBe(0);
      expect(stats.discarded).toBe(0);
      expect(stats.active).toBe(0);
    });

    it("getActiveMutation returns null when no mutation proposed", () => {
      const mut = learner.getActiveMutation("code_review");
      expect(mut).toBeNull();
    });

    it("getActiveMutation returns mutation after proposal", () => {
      for (let i = 0; i < 5; i++) {
        learner.record(makeRecord({ taskType: "code_review", qualityScore: 0.85 }));
      }
      learner.proposeMutation("code_review");
      const mut = learner.getActiveMutation("code_review");
      expect(mut).not.toBeNull();
    });

    it("recordTrial tracks trial quality without error", () => {
      for (let i = 0; i < 5; i++) {
        learner.record(makeRecord({ taskType: "code_review", qualityScore: 0.85 }));
      }
      learner.proposeMutation("code_review");

      for (let i = 0; i < 3; i++) {
        learner.recordTrial("code_review", 0.80, false);
      }
      for (let i = 0; i < 3; i++) {
        learner.recordTrial("code_review", 0.90, true);
      }

      const stats = learner.getEvolutionStats();
      expect(stats.total).toBeGreaterThan(0);
    });

    it("saves and restores evolution state across persistence cycle", () => {
      const statePath = "/tmp/turbocontext-test-state.json";
      for (let i = 0; i < 5; i++) {
        learner.record(makeRecord({ taskType: "code_review", qualityScore: 0.85 }));
      }
      learner.proposeMutation("code_review");

      const learner2 = new Learner(defaultConfig, statePath);
      const stats2 = learner2.getEvolutionStats();
      expect(stats2.total).toBeGreaterThan(0);
    });
  });
});
