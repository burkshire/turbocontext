// ============================================================
// Phase 2: Composer Tests
// ============================================================

import { describe, it, expect } from "vitest";
import { composePromptArchitecture } from "../src/core/composer.js";
import type { Task, CompressedContext, StrategyMutation } from "../src/types.js";

function makeTask(overrides: Partial<Task> = {}): Task {
  return {
    id: "t1",
    description: "review auth module for security issues",
    type: "code_review",
    ...overrides,
  };
}

function makeCompressedContext(): CompressedContext {
  return {
    originalTokens: 500,
    compressedTokens: 200,
    compressionRatio: 0.6,
    fragments: [
      {
        original: {
          id: "1",
          source: "src/auth/login.ts",
          contentType: "source",
          content: "function login() {}",
          lastModified: Date.now(),
          length: 20,
        },
        score: 0.9,
        preservedSections: ["function login() {}"],
      },
    ],
    coverage: { code_understanding: 1 },
  };
}

describe("composePromptArchitecture", () => {
  it("returns 3 rounds for code_review task", () => {
    const result = composePromptArchitecture(makeTask(), makeCompressedContext(), []);
    expect(result.rounds.length).toBe(3);
  });

  it("returns 3 rounds for code_generation task", () => {
    const result = composePromptArchitecture(
      makeTask({ type: "code_generation", description: "generate login form" }),
      makeCompressedContext(), []
    );
    expect(result.rounds.length).toBe(3);
    expect(result.rounds[0].goal).toContain("分析");
    expect(result.rounds[1].goal).toContain("生成");
    expect(result.rounds[2].goal).toContain("检查");
  });

  it("correctly identifies code_refactor task type", () => {
    const result = composePromptArchitecture(
      makeTask({ type: "code_refactor", description: "refactor the user service module" }),
      makeCompressedContext(), []
    );
    expect(result.rounds.length).toBe(3);
    expect(result.rounds[0].goal).toContain("分析");
    expect(result.rounds[1].goal).toContain("重构");
  });

  it("falls back to general strategy for unknown types", () => {
    const result = composePromptArchitecture(
      makeTask({ type: "general" as any, description: "do something" }),
      makeCompressedContext(), []
    );
    expect(result.rounds.length).toBe(3);
  });

  it("estimates token count", () => {
    const result = composePromptArchitecture(makeTask(), makeCompressedContext(), []);
    expect(result.estimatedTokens).toBeGreaterThan(0);
  });

  // ------------------------------------------------------------------
  // Mutation tests (v2.3 — self-evolution)
  // ------------------------------------------------------------------

  describe("mutations", () => {
    it("merge_rounds reduces round count by 1", () => {
      const base = composePromptArchitecture(makeTask(), makeCompressedContext(), []);
      const mergeMutation: StrategyMutation = {
        type: "merge_rounds",
        roundIndices: [0, 1],
        newGoal: "理解并审查",
      };
      const mutated = composePromptArchitecture(makeTask(), makeCompressedContext(), [], mergeMutation);
      expect(mutated.rounds.length).toBe(base.rounds.length - 1);
    });

    it("remove_round reduces round count by 1", () => {
      const base = composePromptArchitecture(makeTask(), makeCompressedContext(), []);
      const removeMutation: StrategyMutation = {
        type: "remove_round",
        roundIndex: 1,
      };
      const mutated = composePromptArchitecture(makeTask(), makeCompressedContext(), [], removeMutation);
      expect(mutated.rounds.length).toBe(base.rounds.length - 1);
    });

    it("reorder_rounds changes round order", () => {
      const base = composePromptArchitecture(makeTask({ type: "code_generation" }), makeCompressedContext(), []);
      const reorderMutation: StrategyMutation = {
        type: "reorder_rounds",
        newOrder: [2, 0, 1],
      };
      const mutated = composePromptArchitecture(
        makeTask({ type: "code_generation" }), makeCompressedContext(), [], reorderMutation
      );
      expect(mutated.rounds.length).toBe(3);
      expect(mutated.rounds[0].goal).not.toBe(base.rounds[0].goal);
    });

    it("add_quality_criterion adds a criterion", () => {
      const addMutation: StrategyMutation = {
        type: "add_quality_criterion",
        roundIndex: 0,
        criterion: "代码必须通过 lint",
      };
      const mutated = composePromptArchitecture(makeTask(), makeCompressedContext(), [], addMutation);
      const round0 = mutated.rounds.find(r => r.sequence === 1)!;
      expect(round0.qualityCriteria).toContain("代码必须通过 lint");
    });

    it("remove_quality_criterion removes a criterion", () => {
      const base = composePromptArchitecture(makeTask(), makeCompressedContext(), []);
      const originalCount = base.rounds[0].qualityCriteria.length;
      const removeMutation: StrategyMutation = {
        type: "remove_quality_criterion",
        roundIndex: 0,
        criterionIndex: 0,
      };
      const mutated = composePromptArchitecture(makeTask(), makeCompressedContext(), [], removeMutation);
      expect(mutated.rounds[0].qualityCriteria.length).toBe(originalCount - 1);
    });

    it("invalid mutation indices are safe (no crash)", () => {
      const badMutations: StrategyMutation[] = [
        { type: "remove_round", roundIndex: 999 },
        { type: "reorder_rounds", newOrder: [42] },
        { type: "merge_rounds", roundIndices: [0, 999], newGoal: "x" },
        { type: "add_quality_criterion", roundIndex: 999, criterion: "x" },
        { type: "remove_quality_criterion", roundIndex: 999, criterionIndex: 0 },
      ];
      for (const mutation of badMutations) {
        // Should not throw
        const result = composePromptArchitecture(makeTask(), makeCompressedContext(), [], mutation);
        expect(result.rounds.length).toBeGreaterThan(0);
      }
    });
  });
});
