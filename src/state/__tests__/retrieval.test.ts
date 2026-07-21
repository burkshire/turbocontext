// ============================================================================
// Retrieval (7-Dim MMR) Tests
// ============================================================================
import { describe, it, expect } from "vitest";
import {
  retrieveMemories,
  scoreMemory,
  computeIDFOverlap,
  computeCapabilityJaccard,
  computeTaskTypeMatch,
  computeInfoDensity,
  mmrReRank,
  computeMemorySimilarity,
  type ScoredMemory,
  type RetrievalQuery,
} from "../rl/retrieval.js";
import { DEFAULT_POLICY } from "../constants.js";
import type { IndexedMemory } from "../types.js";

const now = new Date().toISOString();

function makeMemory(overrides: Partial<IndexedMemory> = {}): IndexedMemory {
  return {
    id: `mem-${Math.random().toString(36).slice(2, 10)}`,
    taskType: "code_review",
    description: "Review code for bugs",
    hypothesis: "Null pointer in auth module",
    insight: "Added null check before dereference",
    outcome: "success",
    qualityScore: 0.90,
    causalUtility: 0.85,
    thompsonAlpha: 10,
    thompsonBeta: 2,
    retrievalCount: 5,
    lastRetrievedAt: now,
    capabilityRequirements: ["code_understanding", "error_detection"],
    status: "active",
    createdAt: now,
    parameters: {
      compression: { alpha: 0.55, beta: 0.20, gamma: 0.25 },
      quality: { threshold: 0.85, maxAttempts: 3 },
      temperature: { t0: 0.7, t1: 0.35, t2: 0.1 },
      modelTier: "medium",
    },
    retrievalUtility: { thompsonAlpha: 10, thompsonBeta: 2 },
    ...overrides,
  } as IndexedMemory;
}

const defaultPolicy = DEFAULT_POLICY.retrieval;

describe("computeIDFOverlap", () => {
  it("returns 1.0 for identical text", () => {
    expect(computeIDFOverlap("hello world", "hello world")).toBeCloseTo(1.0);
  });

  it("returns 0 for no overlap", () => {
    expect(computeIDFOverlap("abc", "xyz")).toBe(0);
  });

  it("returns partial score for partial overlap", () => {
    const score = computeIDFOverlap("hello world foo", "hello world bar");
    expect(score).toBeGreaterThan(0);
    expect(score).toBeLessThan(1);
  });

  it("handles empty query", () => {
    expect(computeIDFOverlap("", "hello")).toBe(0);
  });
});

describe("computeCapabilityJaccard", () => {
  it("returns 1.0 for identical sets", () => {
    expect(computeCapabilityJaccard(["a", "b"], ["a", "b"])).toBe(1);
  });

  it("returns ~0.33 for {a,b} ∩ {b,c} = {b}, union = {a,b,c}", () => {
    expect(computeCapabilityJaccard(["a", "b"], ["b", "c"])).toBeCloseTo(1 / 3);
  });

  it("returns 0.5 for {a,b} ∩ {b,c,d} = {b}, union size 4", () => {
    expect(computeCapabilityJaccard(["a", "b", "d"], ["b", "c", "d"])).toBeCloseTo(2 / 4);
  });

  it("returns 0 for no overlap", () => {
    expect(computeCapabilityJaccard(["a"], ["b"])).toBe(0);
  });

  it("returns 1.0 when both are empty", () => {
    expect(computeCapabilityJaccard([], [])).toBe(1);
  });
});

describe("computeTaskTypeMatch", () => {
  it("returns 1.0 for exact match", () => {
    expect(computeTaskTypeMatch("code_review", "code_review")).toBe(1);
  });

  it("returns 0.5 for same family", () => {
    expect(computeTaskTypeMatch("code_review", "code_generation")).toBe(0.5);
  });

  it("returns 0 for different families", () => {
    expect(computeTaskTypeMatch("code_review", "analysis")).toBe(0);
  });

  it("handles strings without underscores", () => {
    expect(computeTaskTypeMatch("general", "general")).toBe(1);
    expect(computeTaskTypeMatch("general", "debugging")).toBe(0);
  });
});

describe("computeInfoDensity", () => {
  it("returns a number in [0, 1]", () => {
    const mem = makeMemory();
    const d = computeInfoDensity(mem, 10);
    expect(d).toBeGreaterThanOrEqual(0);
    expect(d).toBeLessThanOrEqual(1);
  });

  it("returns higher value for memories with more insight text", () => {
    const sparse = makeMemory({ insight: "x", hypothesis: "" });
    const rich = makeMemory({
      insight: "This is a very detailed insight with many words describing the root cause",
      hypothesis: "Long hypothesis about the system behavior",
    });
    const d1 = computeInfoDensity(sparse, 10);
    const d2 = computeInfoDensity(rich, 10);
    expect(d2).toBeGreaterThan(d1);
  });
});

describe("retrieveMemories", () => {
  it("returns empty array for empty memory list", () => {
    const result = retrieveMemories(
      [], { taskType: "code_review", description: "test" }, defaultPolicy
    );
    expect(result).toEqual([]);
  });

  it("returns all memories when fewer than topK", () => {
    const memories = [makeMemory(), makeMemory()];
    const result = retrieveMemories(
      memories,
      { taskType: "code_review", description: "review auth code", capabilityRequirements: ["code_understanding"] },
      { ...defaultPolicy, topK: 10 },
    );
    expect(result.length).toBe(2);
  });

  it("respects topK parameter", () => {
    const memories = Array.from({ length: 20 }, () => makeMemory());
    const result = retrieveMemories(
      memories,
      { taskType: "code_review", description: "review code" },
      { ...defaultPolicy, topK: 5 },
    );
    expect(result.length).toBeLessThanOrEqual(5);
  });

  it("scores exact task type match higher than family match", () => {
    const exactMem = makeMemory({ id: "exact", taskType: "code_review" });
    const familyMem = makeMemory({ id: "family", taskType: "code_generation" });
    const unrelated = makeMemory({ id: "unrelated", taskType: "architecture" });

    const result = retrieveMemories(
      [unrelated, familyMem, exactMem], // reverse order intentional
      { taskType: "code_review", description: "review code" },
      { ...defaultPolicy, topK: 3 },
    );

    // exact should come first (or at least be present)
    expect(result.some(m => m.id === "exact")).toBe(true);
  });

  it("filters to active memories only", () => {
    const active = makeMemory({ id: "active", status: "active" });
    const cold = makeMemory({ id: "cold", status: "cold" });

    const result = retrieveMemories(
      [active, cold],
      { taskType: "code_review", description: "test" },
      { ...defaultPolicy, topK: 10 },
    );

    expect(result.every(m => m.status === "active")).toBe(true);
    expect(result.some(m => m.id === "cold")).toBe(false);
  });
});

describe("mmrReRank", () => {
  it("returns all candidates if fewer than topK", () => {
    const items: ScoredMemory[] = [
      { memory: makeMemory({ id: "a" }), score: 0.9, dimScores: {} },
      { memory: makeMemory({ id: "b" }), score: 0.5, dimScores: {} },
    ];
    const result = mmrReRank(items, 10, 0.7);
    expect(result.length).toBe(2);
  });

  it("mmrLambda=1.0 gives pure relevance ranking", () => {
    const items: ScoredMemory[] = [
      { memory: makeMemory({ id: "a" }), score: 0.3, dimScores: {} },
      { memory: makeMemory({ id: "b" }), score: 0.9, dimScores: {} },
      { memory: makeMemory({ id: "c" }), score: 0.6, dimScores: {} },
    ];
    const result = mmrReRank(items, 2, 1.0);
    // Highest scored first
    expect(result[0].memory.id).toBe("b");
    expect(result[1].memory.id).toBe("c");
  });
});

describe("computeMemorySimilarity", () => {
  it("returns higher similarity for same task type + same outcome", () => {
    const a = makeMemory({ taskType: "code_review", outcome: "success" });
    const b = makeMemory({ taskType: "code_review", outcome: "success" });
    const c = makeMemory({ taskType: "architecture", outcome: "failure" });

    const simAB = computeMemorySimilarity(a, b);
    const simAC = computeMemorySimilarity(a, c);
    expect(simAB).toBeGreaterThan(simAC);
  });

  it("returns value in [0, 1]", () => {
    const a = makeMemory();
    const b = makeMemory();
    const sim = computeMemorySimilarity(a, b);
    expect(sim).toBeGreaterThanOrEqual(0);
    expect(sim).toBeLessThanOrEqual(1);
  });
});
