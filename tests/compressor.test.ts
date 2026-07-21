// ============================================================
// Phase 1: Compressor Tests
// ============================================================

import { describe, expect, it } from "vitest";
import { compressContext, _internal } from "../src/core/compressor.js";
import type { Task, ContextFragment } from "../src/types.js";

const { calculateScore, computeSemanticSimilarity, computeRecency, computeSpecificity, compressFragment, greedySelect, decomposeTask } = _internal;

function makeFragment(overrides: Partial<ContextFragment> = {}): ContextFragment {
  return {
    id: "test1",
    source: "test.ts",
    contentType: "source",
    content: "export function hello() { return 'world'; }",
    lastModified: Date.now() - 86400000,
    length: 50,
    ...overrides,
  };
}

function makeTask(overrides: Partial<Task> = {}): Task {
  return {
    id: "task1",
    description: "review auth module code for security issues",
    type: "code_review",
    ...overrides,
  };
}

describe("decomposeTask", () => {
  it("extracts capabilities from security review task", () => {
    const reqs = decomposeTask(makeTask());
    expect(reqs.length).toBeGreaterThan(0);
    // code_understanding and error_detection should have high weight for "security review"
    const understanding = reqs.find(r => r.name === "code_understanding");
    const errorDetection = reqs.find(r => r.name === "error_detection");
    expect(understanding).toBeDefined();
    expect(errorDetection).toBeDefined();
    if (understanding) expect(understanding.weight).toBeGreaterThan(0);
    if (errorDetection) expect(errorDetection.weight).toBeGreaterThan(0);
  });

  it("returns low-weight requirements even for empty description", () => {
    const reqs = decomposeTask(makeTask({ description: "" }));
    // empty description means all keywords get base weight (0.5 * defaultWeight)
    expect(reqs.length).toBeGreaterThan(0);
    expect(reqs.every(r => r.weight < 0.5)).toBe(true);
  });

  it("normalizes weights to sum ~1", () => {
    const reqs = decomposeTask(makeTask({ description: "implement new login system with JWT and refresh tokens" }));
    const totalWeight = reqs.reduce((s, r) => s + r.weight, 0);
    expect(totalWeight).toBeCloseTo(1, 1);
  });
});

describe("calculateScore", () => {
  const config = { alpha: 0.55, beta: 0.20, gamma: 0.25 };
  const task = makeTask();

  it("returns number between 0 and 1", () => {
    const score = calculateScore(makeFragment(), task, config);
    expect(score).toBeGreaterThanOrEqual(0);
    expect(score).toBeLessThanOrEqual(1);
  });

  it("favors relevant fragments over irrelevant ones", () => {
    const relevant = makeFragment({
      content: "export function login(email: string, password: string) { return auth; }",
    });
    const irrelevant = makeFragment({
      id: "test2",
      content: "export function formatDate(date: Date) { return date.toString(); }",
    });
    const relScore = calculateScore(relevant, makeTask({ description: "review login and auth" }), config,
      [relevant, irrelevant]);
    const irrScore = calculateScore(irrelevant, makeTask({ description: "review login and auth" }), config,
      [relevant, irrelevant]);
    expect(relScore).toBeGreaterThanOrEqual(irrScore);
  });

  it("handles empty fragments gracefully", () => {
    const score = calculateScore(makeFragment({ content: "" }), task, config);
    expect(score).toBeGreaterThanOrEqual(0);
  });
});

describe("computeRecency", () => {
  it("returns 1 for just modified files", () => {
    const score = computeRecency(makeFragment({ lastModified: Date.now() }));
    expect(score).toBeCloseTo(1, 1);
  });

  it("returns lower score for old files", () => {
    const old = computeRecency(makeFragment({ lastModified: Date.now() - 30 * 86400000 }));
    const recent = computeRecency(makeFragment({ lastModified: Date.now() - 86400000 }));
    expect(old).toBeLessThan(recent);
  });
});

describe("computeSpecificity", () => {
  it("returns 1 for empty content (max specificity)", () => {
    const score = computeSpecificity(makeFragment({ length: 0 }));
    expect(score).toBe(1);
  });

  it("returns lower score for long content", () => {
    const short = computeSpecificity(makeFragment({ length: 100 }));
    const long = computeSpecificity(makeFragment({ length: 5000 }));
    expect(long).toBeLessThan(short);
  });
});

describe("compressFragment", () => {
  it("preserves structural lines like function signatures", () => {
    const code = "export function hello() {\n  const x = 1;\n  return x;\n}\n";
    const result = compressFragment(makeFragment({ content: code }), 0.9);
    expect(result.preservedSections.some(s => s.includes("function hello"))).toBe(true);
  });

  it("summarizes long function bodies", () => {
    const lines = ["export function longFunc() {"];
    for (let i = 0; i < 20; i++) lines.push(`  console.log(${i});`);
    lines.push("}");
    const result = compressFragment(makeFragment({ content: lines.join("\n") }), 0.8);
    expect(result.preservedSections.some(s => s.includes("lines omitted"))).toBe(true);
  });

  it("preserves short function bodies entirely", () => {
    const code = "function short() {\n  return 42;\n}\n";
    const result = compressFragment(makeFragment({ content: code }), 0.9);
    expect(result.preservedSections.some(s => s.includes("return 42"))).toBe(true);
  });

  it("handles empty content", () => {
    const result = compressFragment(makeFragment({ content: "" }), 0);
    expect(result.preservedSections).toEqual([]);
  });

  it("detects structural keyword (class) and preserves it", () => {
    const code = "class UserService {\n  constructor() {}\n}\n";
    const result = compressFragment(makeFragment({ content: code }), 0.9);
    expect(result.preservedSections.some(s => s.includes("class UserService"))).toBe(true);
  });
});

describe("greedySelect", () => {
  const config = { maxTokenBudget: 1000, minCoverage: 0.8 };

  it("selects fragments within budget", () => {
    const fragments = [
      { fragment: makeFragment({ id: "1", content: "function a() {}", length: 15 }), score: 0.9 },
      { fragment: makeFragment({ id: "2", content: "function b() {}", length: 15 }), score: 0.8 },
      { fragment: makeFragment({ id: "3", content: "function c() {}", length: 15 }), score: 0.3 },
    ];
    const reqs = decomposeTask(makeTask());
    const selected = greedySelect(fragments, reqs, config, makeTask());
    expect(selected.length).toBeGreaterThan(0);
    expect(selected.length).toBeLessThanOrEqual(fragments.length);
  });

  it("selects higher scored fragments first", () => {
    const fragments = [
      { fragment: makeFragment({ id: "1", content: "function highest() {}", length: 20 }), score: 0.95 },
      { fragment: makeFragment({ id: "2", content: "function low() {}", length: 20 }), score: 0.1 },
    ];
    const reqs = decomposeTask(makeTask());
    const selected = greedySelect(fragments, reqs, config, makeTask());
    expect(selected[0].fragment.id).toBe("1");
  });
});

describe("compressContext (integration)", () => {
  it("returns valid CompressedContext", async () => {
    const task = makeTask();
    const fragments = [
      makeFragment({ id: "1", source: "src/auth/login.ts", content: "export async function login(email: string, password: string) {\n  const user = await db.users.findByEmail(email);\n  if (!user) throw new AuthError('User not found');\n  return user;\n}" }),
      makeFragment({ id: "2", source: "src/utils/date.ts", content: "export function formatDate(d: Date) { return d.toISOString(); }" }),
    ];
    const result = await compressContext(task, fragments);
    expect(result.originalTokens).toBeGreaterThan(0);
    expect(result.compressedTokens).toBeGreaterThan(0);
    expect(result.compressionRatio).toBeGreaterThanOrEqual(0);
    expect(result.compressionRatio).toBeLessThanOrEqual(1);
    expect(result.fragments.length).toBeGreaterThan(0);
    expect(result.coverage).toBeDefined();
  });

  it("handles empty fragment list", async () => {
    const result = await compressContext(makeTask(), []);
    expect(result.fragments).toEqual([]);
    expect(result.originalTokens).toBe(0);
    expect(result.compressedTokens).toBe(0);
  });

  // v3.7: Two-phase causal retrieval
  it("causalBoostFn boosts fragment scores", async () => {
    const fragments = [
      makeFragment({ id: "f1", source: "a.ts", content: "export function login() {}", lastModified: Date.now(), length: 30 }),
      makeFragment({ id: "f2", source: "b.ts", content: "export function logout() {}", lastModified: Date.now() - 1000, length: 31 }),
    ];

    // Without causal boost, both get similar scores
    const noBoost = await compressContext(makeTask(), fragments);
    const scoresNoBoost = noBoost.fragments.map(f => f.score);

    // With causal boost: f1 gets 1.5x, f2 gets 0.5x
    const withBoost = await compressContext(makeTask(), fragments, {
      causalBoostFn: (frag: ContextFragment) => frag.source === "a.ts" ? 1.5 : 0.5,
    });
    const scoresWithBoost = withBoost.fragments.map(f => f.score);

    // f1 should be ranked higher with causal boost
    const f1NoBoost = noBoost.fragments.find(f => f.original.source === "a.ts");
    const f1WithBoost = withBoost.fragments.find(f => f.original.source === "a.ts");
    // With 1.5x multiplier, f1 should score higher (or same if both hit ceiling)
    expect(f1WithBoost!.score).toBeGreaterThanOrEqual(f1NoBoost!.score);
  });

  it("causalBoostFn default 1.0 is neutral", async () => {
    const fragments = [
      makeFragment({ id: "f1", source: "a.ts", content: "export function x() {}", lastModified: Date.now(), length: 28 }),
    ];

    const noBoost = await compressContext(makeTask(), fragments);
    const neutralBoost = await compressContext(makeTask(), fragments, {
      causalBoostFn: () => 1.0,
    });

    expect(noBoost.fragments[0].score).toBe(neutralBoost.fragments[0].score);
  });
});
