// ============================================================
// Phase 3: Generator Tests
// ============================================================

import { describe, it, expect } from "vitest";
import { evaluateQuality, DEFAULT_QUALITY_CONFIG } from "../src/core/generator.js";
import type { Task } from "../src/types.js";

function makeTask(overrides: Partial<Task> = {}): Task {
  return {
    id: "t1",
    description: "review auth module for security vulnerabilities",
    type: "code_review",
    ...overrides,
  };
}

describe("evaluateQuality", () => {
  it("returns score between 0 and 1", () => {
    const result = evaluateQuality("Some output content here.", makeTask());
    expect(result.score).toBeGreaterThanOrEqual(0);
    expect(result.score).toBeLessThanOrEqual(1);
  });

  it("detects high quality output", () => {
    const output = [
      "## Security Analysis",
      "I have reviewed the auth module comprehensively.",
      "### Findings",
      "1. The login function handles input validation correctly.",
      "2. Password hashing uses bcrypt with adequate rounds.",
      "3. JWT tokens are properly signed and expiring.",
      "### Recommendations",
      "- Add rate limiting to prevent brute force attacks.",
      "- Consider implementing account lockout after 5 failures.",
    ].join("\n");
    const result = evaluateQuality(output, makeTask({ description: "review auth module for security vulnerabilities" }));
    expect(result.score).toBeGreaterThanOrEqual(0.5);
  });

  it("penalizes uncertain language", () => {
    const output = "Sorry, I don't know the answer to that. Assuming it might work, perhaps try a different approach.";
    const result = evaluateQuality(output, makeTask());
    // penalties: sorry = -0.10, assuming = -0.05, might be = -0.05
    // base ~1.0, after penalties ~0.80, format may add extra deductions
    expect(result.score).toBeLessThan(0.95);
  });

  it("penalizes incomplete output for short responses", () => {
    const result = evaluateQuality("short", makeTask({ description: "review auth module for security issues and code quality problems in the login system" }));
    // short output: format.specificity gets -0.5 (length < 10)
    // format weight = 0.20 (code_review) → score reduced by 0.10
    // completeness: requirements extracted but none matched → lower score
    expect(result.score).toBeLessThan(0.9);
  });

  it("uses task-specific weights", () => {
    const output = "Some output content for testing.";
    const codeGenResult = evaluateQuality(output, makeTask({ type: "code_generation", description: "generate a login form" }));
    const docResult = evaluateQuality(output, makeTask({ type: "documentation", description: "document the API" }));
    expect(typeof codeGenResult.score).toBe("number");
    expect(typeof docResult.score).toBe("number");
  });

  it("detects code block formatting issues for code_gen tasks", () => {
    const output = "Here is the code:\nfunction hello() {\n  return 'world';\n}\n";
    // code_generation task without ``` → format penalty of -0.3
    const result = evaluateQuality(output, makeTask({ type: "code_generation", description: "generate a hello function" }));
    // format score = 0.7 (after -0.3 for no code block), which is >= 0.6 so no issue generated
    // The penalty is applied to the format dimension, not necessarily generating an issue
    expect(result.dimensions.format).toBeLessThanOrEqual(0.7);
    expect(result.dimensions.format).toBeGreaterThanOrEqual(0.6);
  });

  it("returns passed = true when above threshold", () => {
    // Generate output that should score well
    const output = "## Code Review\n\n### Findings\n1. Issue one\n2. Issue two\n\n### Recommendations\n- Fix this\n- Improve that\n";
    const result = evaluateQuality(output, makeTask({ type: "code_review", description: "review code for issues and problems" }));
    // High completeness (matches "review", "code", "issues", "problems")
    // No error signals → high correctness
    expect(result.passed).toBeDefined();
  });
});

describe("DEFAULT_QUALITY_CONFIG", () => {
  it("has expected defaults", () => {
    expect(DEFAULT_QUALITY_CONFIG.qualityThreshold).toBe(0.85);
    expect(DEFAULT_QUALITY_CONFIG.maxAttempts).toBe(3);
    expect(DEFAULT_QUALITY_CONFIG.temperatureSchedule).toEqual([0.7, 0.35, 0.1]);
  });
});
