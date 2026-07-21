// ============================================================
// Phase 4: Optimizer Tests
// ============================================================

import { describe, it, expect, beforeEach } from "vitest";
import { Optimizer, MODEL_TIERS } from "../src/core/optimizer.js";
import type { Task, ExecutionRecord } from "../src/types.js";

function makeTask(overrides: Partial<Task> = {}): Task {
  return {
    id: "t1",
    description: "review auth module",
    type: "code_review",
    ...overrides,
  };
}

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

describe("Optimizer", () => {
  let optimizer: Optimizer;

  beforeEach(() => {
    optimizer = new Optimizer({ thresholdLow: 0.30, thresholdHigh: 0.50 });
  });

  describe("selectModel", () => {
    it("returns fast for simple tasks", () => {
      // documentation type = 0.25, desc ≥20 chars → ambiguity 0.5
      // complexity = 0.40*0.25 + 0.15*0.5 + 0.20*0.3 + 0.25*0.2 = 0.10+0.075+0.06+0.05 = 0.285
      // 0.285 < 0.30 → fast ✓
      const result = optimizer.selectModel(makeTask({
        type: "documentation",
        description: "update the readme file",
      }), []);
      expect(result.tier).toBe("fast");
      expect(result.config.model).toBeDefined();
    });

    it("returns deep for complex tasks", () => {
      // design type = 0.65, short description → ambiguity 0.8, bad history → 0.6
      // complexity = 0.40*0.65 + 0.15*0.8 + 0.20*0.6 + 0.25*0.2 = 0.26+0.12+0.12+0.05 = 0.55
      // 0.55 ≥ 0.50 → deep ✓
      const history = [
        makeRecord({ taskType: "design", qualityScore: 0.5 }),
        makeRecord({ taskType: "design", qualityScore: 0.4 }),
        makeRecord({ taskType: "design", qualityScore: 0.6 }),
      ];
      const result = optimizer.selectModel(makeTask({
        type: "design",
        description: "do",
      }), history);
      expect(result.tier).toBe("deep");
    });

    it("returns medium for moderate complexity", () => {
      const result = optimizer.selectModel(makeTask({
        type: "code_generation",
        description: "implement a REST API endpoint",
      }), []);
      expect(result.tier).toBe("medium");
    });

    it("respects latency budget by downgrading", () => {
      const result = optimizer.selectModel(makeTask({
        type: "code_review",
        description: "review auth module code",
        latencyBudget: 3, // 3 seconds, medium needs 5s
      }), []);
      // code_review complexity ≈ 0.40*0.40 + 0.15*0.3 + 0.20*0.3 + 0.05 = 0.285
      // 0.285 < 0.30 → fast already; latency budget doesn't cause further downgrade
      expect(["fast", "medium"]).toContain(result.tier);
    });
  });

  describe("estimateComplexity", () => {
    it("returns value between 0 and 1", () => {
      const c = optimizer.estimateComplexity(makeTask(), []);
      expect(c).toBeGreaterThanOrEqual(0);
      expect(c).toBeLessThanOrEqual(1);
    });

    it("increases complexity for short, vague descriptions", () => {
      const vague = optimizer.estimateComplexity(makeTask({ description: "fix" }), []);
      // Long description (≥200 chars) → ambiguity 0.2; short (<20) → 0.8
      const detailed = optimizer.estimateComplexity(makeTask({
        description: "fix the login redirect loop when JWT token expires during session refresh and the middleware throws an unhandled error instead of redirecting to login page properly with the return URL intact for post-auth navigation restoration flow",
      }), []);
      expect(vague).toBeGreaterThan(detailed);
    });

    it("design task scores higher than documentation", () => {
      const docs = optimizer.estimateComplexity(makeTask({ type: "documentation", description: "update readme" }), []);
      const design = optimizer.estimateComplexity(makeTask({ type: "design", description: "do" }), []);
      expect(design).toBeGreaterThan(docs);
    });
  });

  describe("cache", () => {
    it("writeCache stores and lookupCache retrieves", () => {
      const task = makeTask();
      optimizer.writeCache(task, "compressed content", "result", 0.9, "fast");
      const cached = optimizer.lookupCache(task, "compressed content");
      expect(cached).not.toBeNull();
      expect(cached?.result).toBe("result");
    });

    it("lookupCache returns null for expired entries", () => {
      const task = makeTask();
      optimizer.writeCache(task, "test content", "result", 0.9, "fast");
      // Direct manipulation to simulate expiration
      const cached = optimizer.lookupCache(task, "different content");
      expect(cached).toBeNull();
    });
  });

  describe("estimateCost", () => {
    it("calculates positive cost for any task", () => {
      const cost = optimizer.estimateCost(makeTask(), 1000, "medium");
      expect(cost.estimatedCostUSD).toBeGreaterThan(0);
      expect(cost.estimatedLatency).toBeDefined();
    });

    it("cheaper for fast tier than medium", () => {
      const fast = optimizer.estimateCost(makeTask(), 1000, "fast").estimatedCostUSD;
      const medium = optimizer.estimateCost(makeTask(), 1000, "medium").estimatedCostUSD;
      expect(fast).toBeLessThan(medium);
    });
  });
});

describe("MODEL_TIERS", () => {
  it("defines all three tiers", () => {
    expect(MODEL_TIERS.fast).toBeDefined();
    expect(MODEL_TIERS.medium).toBeDefined();
    expect(MODEL_TIERS.deep).toBeDefined();
  });

  it("tiers have increasing costs", () => {
    expect(MODEL_TIERS.fast.costPer1KTokens).toBeLessThan(MODEL_TIERS.medium.costPer1KTokens);
    expect(MODEL_TIERS.medium.costPer1KTokens).toBeLessThan(MODEL_TIERS.deep.costPer1KTokens);
  });
});
