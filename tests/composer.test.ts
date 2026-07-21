// ============================================================
// Phase 2: Composer Tests
// ============================================================

import { describe, it, expect } from "vitest";
import { composePromptArchitecture } from "../src/core/composer.js";
import type { Task, CompressedContext } from "../src/types.js";

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

});
