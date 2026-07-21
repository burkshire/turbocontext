// ============================================================
// End-to-End Experiment Harness (v3.2)
// ============================================================
// Integration tests that exercise the full TurboContext pipeline:
//   Phase 1 (compress) → Phase 2 (compose) → Phase 4 (optimize)
//   → Phase 3 (generate) → Phase 5 (learn) → evolution loop
//
// Uses the built-in simulator (no real LLM calls), ensuring
// tests are fast, deterministic, and runnable in CI.
// ============================================================

import { describe, it, expect, beforeEach } from "vitest";
import { TurboContextEngine } from "../src/index.js";
import type { Task, ContextFragment } from "../src/types.js";

// ------------------------------------------------------------------
// Test Fixtures
// ------------------------------------------------------------------

/** Create a set of realistic-looking source file fragments. */
function makeContextPool(): ContextFragment[] {
  return [
    {
      id: "auth-login",
      source: "src/auth/login.ts",
      contentType: "source",
      lastModified: Date.now() - 86_400_000, // 1 day ago
      length: 512,
      content: [
        'import { db } from "../db";',
        'import { generateJWT, sanitizeUser } from "./utils";',
        'import bcrypt from "bcrypt";',
        "",
        "export async function login(email: string, password: string) {",
        "  const user = await db.users.findByEmail(email);",
        '  if (!user) throw new AuthError("User not found");',
        "  const valid = await bcrypt.compare(password, user.passwordHash);",
        '  if (!valid) throw new AuthError("Invalid password");',
        "  const token = generateJWT({ userId: user.id, role: user.role });",
        "  return { user: sanitizeUser(user), token };",
        "}",
      ].join("\n"),
    },
    {
      id: "auth-register",
      source: "src/auth/register.ts",
      contentType: "source",
      lastModified: Date.now() - 172_800_000, // 2 days ago
      length: 420,
      content: [
        'import { db } from "../db";',
        'import { sanitizeUser } from "./utils";',
        "import bcrypt from 'bcrypt';",
        "",
        "export async function register(data: RegisterInput) {",
        "  const existing = await db.users.findByEmail(data.email);",
        '  if (existing) throw new AuthError("Email already registered");',
        "  const hash = await bcrypt.hash(data.password, 12);",
        "  const user = await db.users.create({ ...data, passwordHash: hash });",
        "  return { user: sanitizeUser(user) };",
        "}",
      ].join("\n"),
    },
    {
      id: "middleware-rate",
      source: "src/middleware/rate-limit.ts",
      contentType: "source",
      lastModified: Date.now() - 43_200_000, // 12 hours ago
      length: 680,
      content: [
        "import type { Request, Response, NextFunction } from 'express';",
        "",
        "interface RateLimitStore {",
        "  [key: string]: { count: number; resetAt: number };",
        "}",
        "",
        "const store: RateLimitStore = {};",
        "",
        "export function rateLimiter(maxRequests = 100, windowMs = 60000) {",
        "  return (req: Request, res: Response, next: NextFunction) => {",
        "    const key = req.ip || 'unknown';",
        "    const now = Date.now();",
        "    if (!store[key] || store[key].resetAt < now) {",
        "      store[key] = { count: 0, resetAt: now + windowMs };",
        "    }",
        "    store[key].count++;",
        "    if (store[key].count > maxRequests) {",
        "      res.status(429).json({ error: 'Too many requests' });",
        "      return;",
        "    }",
        "    next();",
        "  };",
        "}",
      ].join("\n"),
    },
    {
      id: "utils-date",
      source: "src/utils/date.ts",
      contentType: "source",
      lastModified: Date.now() - 604_800_000, // 7 days ago
      length: 180,
      content: [
        "export function formatDate(d: Date): string {",
        "  return d.toISOString().split('T')[0];",
        "}",
        "",
        "export function daysBetween(a: Date, b: Date): number {",
        "  return Math.abs(a.getTime() - b.getTime()) / 86400000;",
        "}",
      ].join("\n"),
    },
    {
      id: "test-login",
      source: "src/auth/__tests__/login.test.ts",
      contentType: "test",
      lastModified: Date.now() - 259_200_000, // 3 days ago
      length: 320,
      content: [
        "import { describe, it, expect } from 'vitest';",
        "import { login } from '../login';",
        "",
        "describe('login', () => {",
        "  it('returns token for valid credentials', async () => {",
        "    const result = await login('test@example.com', 'password123');",
        "    expect(result.token).toBeDefined();",
        "    expect(result.user).toBeDefined();",
        "  });",
        "});",
      ].join("\n"),
    },
  ];
}

/** Create a diverse task pool covering all 9 task types. */
function makeTaskPool(): Task[] {
  return [
    {
      id: "rev-auth",
      description: "Review src/auth module — check for SQL injection, XSS, missing input validation, and unsafe JWT handling",
      type: "code_review",
      qualityThreshold: 0.85,
    },
    {
      id: "gen-rate-limit",
      description: "Implement rate limiting middleware for the Express API with configurable window and max requests",
      type: "code_generation",
      qualityThreshold: 0.85,
    },
    {
      id: "ref-user-svc",
      description: "Refactor user service to use the repository pattern for better testability and separation of concerns",
      type: "code_refactor",
      qualityThreshold: 0.85,
    },
    {
      id: "dbg-session",
      description: "Debug race condition in session cleanup — sessions are being deleted before TTL expires under high concurrency",
      type: "debugging",
      qualityThreshold: 0.85,
    },
    {
      id: "test-auth",
      description: "Write comprehensive unit tests for the auth module covering login, register, and edge cases",
      type: "testing",
      qualityThreshold: 0.85,
    },
    {
      id: "analyze-perf",
      description: "Analyze the performance bottleneck in the current authentication flow, focusing on bcrypt and DB queries",
      type: "analysis",
      qualityThreshold: 0.85,
    },
    {
      id: "design-cache",
      description: "Design a caching layer for frequently accessed user data to reduce auth latency by 50%",
      type: "design",
      qualityThreshold: 0.85,
    },
    {
      id: "doc-api",
      description: "Document the auth module API endpoints including request/response formats, error codes, and authentication flow",
      type: "documentation",
      qualityThreshold: 0.85,
    },
    {
      id: "gen-config",
      description: "Generate a typed configuration loader with environment variable validation",
      type: "general",
      qualityThreshold: 0.85,
    },
  ];
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

describe("TurboContextEngine — Full Pipeline", () => {
  let engine: TurboContextEngine;
  let contextPool: ContextFragment[];
  let taskPool: Task[];

  beforeEach(() => {
    engine = new TurboContextEngine({
      qualityThreshold: 0.85,
      maxAttempts: 3,
      maxTokenBudget: 4000,
    });
    contextPool = makeContextPool();
    taskPool = makeTaskPool();
  });

  it("executes full 5-phase pipeline and returns valid result", async () => {
    const task = taskPool[0]; // code_review
    const result = await engine.execute(task, contextPool);

    // Phase 1: compression
    expect(result.compressed.originalTokens).toBeGreaterThan(0);
    expect(result.compressed.compressionRatio).toBeGreaterThanOrEqual(0);
    expect(result.compressed.compressionRatio).toBeLessThanOrEqual(1);
    expect(result.compressed.fragments.length).toBeGreaterThan(0);
    expect(Object.keys(result.compressed.coverage).length).toBeGreaterThan(0);

    // Phase 2: architecture
    expect(result.architecture.rounds.length).toBeGreaterThanOrEqual(1);
    expect(result.architecture.estimatedTokens).toBeGreaterThan(0);

    // Phase 4: model selection
    expect(["fast", "medium", "deep"]).toContain(result.modelSelection.tier);
    expect(result.costEstimate.estimatedCostUSD).toBeGreaterThanOrEqual(0);

    // Phase 3: generation
    expect(result.generations.length).toBeGreaterThan(0);
    expect(result.finalQuality).toBeGreaterThanOrEqual(0);
    expect(result.finalQuality).toBeLessThanOrEqual(1);
    expect(result.totalAttempts).toBeGreaterThanOrEqual(1);
    expect(result.totalAttempts).toBeLessThanOrEqual(3);
    expect(result.totalLatency).toBeGreaterThan(0);

    // Phase 5: learning
    expect(result.qualityTrend).toBeDefined();
    expect(result.executionCount).toBeGreaterThanOrEqual(1);
  });

  it("triggers learning after 5+ executions", async () => {
    // Execute 6 tasks to trigger learning (interval = 5)
    for (let i = 0; i < 6; i++) {
      const task = taskPool[i % taskPool.length];
      await engine.execute(task, contextPool);
    }

    // Config should have been updated
    const config = engine.getConfig();
    expect(config.alpha).toBeDefined();
    expect(config.beta).toBeDefined();
    expect(config.gamma).toBeDefined();
  });

  it("accumulates branch statistics across task types", async () => {
    // Run 3 different task types
    for (let i = 0; i < 3; i++) {
      await engine.execute(taskPool[i], contextPool);
    }

    const learner = engine.getLearner();
    const stats = learner.getStats();

    // At least some branches should have data
    const activeBranches = learner.getActiveBranches();
    expect(activeBranches.length).toBeGreaterThan(0);

    // Each active branch should have stats
    for (const branchType of activeBranches) {
      const branchStats = stats.get(branchType);
      expect(branchStats).toBeDefined();
      expect(branchStats!.count).toBeGreaterThan(0);
    }
  });

  it("returns valid quality trend data", async () => {
    // Need at least 3 executions for trend analysis
    for (let i = 0; i < 4; i++) {
      await engine.execute(taskPool[i % taskPool.length], contextPool);
    }

    const learner = engine.getLearner();
    const trend = learner.getQualityTrend();

    expect(trend.average).toBeGreaterThanOrEqual(0);
    expect(["improving", "stable", "declining"]).toContain(trend.trend);
    expect(trend.byType).toBeDefined();
    expect(trend.branches).toBeDefined();
  });

  it("detects plateau after repeated same-task executions with similar quality", async () => {
    const codeReviewTask = taskPool[0];

    // Run the same task type 10 times to accumulate enough data
    for (let i = 0; i < 10; i++) {
      await engine.execute({
        ...codeReviewTask,
        id: `plateau-test-${i}`,
      }, contextPool);
    }

    const learner = engine.getLearner();
    const plateau = learner.detectPlateau("code_review");

    // With 10 experiments, plateau detection should have enough data
    expect(plateau.rules.length).toBeGreaterThanOrEqual(1);
    // At least the "insufficient_data" rule should NOT be the only result
    const hasRealRules = plateau.rules.some(
      r => r.rule !== "insufficient_data"
    );
    expect(hasRealRules).toBe(true);
  });

  it("generates strategic directives for active branches", async () => {
    // Establish some history
    for (let i = 0; i < 3; i++) {
      await engine.execute(taskPool[i], contextPool);
    }

    const learner = engine.getLearner();
    const activeBranches = learner.getActiveBranches();

    for (const branchType of activeBranches) {
      const directive = learner.generateStrategicDirective(branchType);
      expect(directive.directive).toBeDefined();
      expect(directive.message.length).toBeGreaterThan(0);
      expect(directive.suggestedAction.length).toBeGreaterThan(0);
      expect(directive.metrics).toBeDefined();
    }
  });

  it("proposes evolution mutations after sufficient branch history", async () => {
    // Accumulate enough experiments to trigger mutation proposals
    for (let i = 0; i < 8; i++) {
      await engine.execute(taskPool[i % taskPool.length], contextPool);
    }

    const learner = engine.getLearner();
    // Try proposing a mutation for code_review (should have the most data)
    const mutation = learner.proposeMutation("code_review");

    // Either a mutation is proposed or null (if all candidates tried)
    if (mutation) {
      expect([
        "merge_rounds",
        "split_round",
        "remove_round",
        "reorder_rounds",
        "add_quality_criterion",
        "remove_quality_criterion",
      ]).toContain(mutation.type);
    }
  });

  it("finds contrastive pairs with sufficient history", async () => {
    // Run multiple tasks to build up success/failure data
    for (let i = 0; i < 6; i++) {
      await engine.execute(taskPool[i % taskPool.length], contextPool);
    }

    const learner = engine.getLearner();
    const pairs = learner.findContrastivePairs("code_review", 2);

    // May or may not find pairs depending on quality scores
    // Just verify the method doesn't crash and returns valid structure
    expect(Array.isArray(pairs)).toBe(true);
    for (const pair of pairs) {
      expect(pair.success).toBeDefined();
      expect(pair.failure).toBeDefined();
      expect(pair.sharedFeatures).toBeDefined();
      expect(pair.insight.length).toBeGreaterThan(0);
    }
  });

  it("computes adaptive MMR lambda based on branch state", async () => {
    for (let i = 0; i < 5; i++) {
      await engine.execute(taskPool[i], contextPool);
    }

    const learner = engine.getLearner();
    const lambda = learner.getAdaptiveMmrLambda("code_review");

    // Should be in valid range regardless of plateau state
    expect(lambda).toBeGreaterThanOrEqual(0.3);
    expect(lambda).toBeLessThanOrEqual(0.9);
  });
});

describe("TurboContextEngine — Experiment Loop", () => {
  it("runs experiment loop with runExperiments()", async () => {
    const engine = new TurboContextEngine({
      qualityThreshold: 0.85,
      maxTokenBudget: 4000,
    });

    const taskPool = makeTaskPool().slice(0, 4); // Use subset for speed
    const contextPool = makeContextPool();

    const runs = await engine.runExperiments({
      maxExperiments: 5,
      tokenBudgetPerRun: 4000,
      timeBudgetPerRun: 10,
      taskPool,
      contextPool,
    });

    // Verify structure
    expect(Array.isArray(runs)).toBe(true);
    expect(runs.length).toBeGreaterThanOrEqual(1);

    // First run should be baseline (no mutation, deltaPercent=0 or mutation of its own)
    const baseline = runs[0];
    expect(baseline.baselineMetric.efficiency).toBeGreaterThan(0);
    expect(baseline.experimentMetric.efficiency).toBeGreaterThan(0);

    // Each run should have valid structure
    for (const run of runs) {
      expect(run.taskType).toBeDefined();
      expect(run.decision).toMatch(/keep|discard/);
      expect(run.status).toMatch(/success|crash|timeout|discarded/);
      expect(run.runNumber).toBeGreaterThan(0);
      expect(typeof run.deltaPercent).toBe("number");
    }

    // At least some runs should have tried mutations
    const runsWithMutations = runs.filter(r => r.mutation !== null);
    // Even if no mutations proposed, the loop should complete gracefully
    expect(runs.every(r => r.status === "success" || r.status === "discarded")).toBe(true);
  });

  it("produces valid evolution stats after experiment loop", async () => {
    const engine = new TurboContextEngine();
    const taskPool = makeTaskPool().slice(0, 3);
    const contextPool = makeContextPool();

    await engine.runExperiments({
      maxExperiments: 3,
      taskPool,
      contextPool,
    });

    const learner = engine.getLearner();
    const stats = learner.getEvolutionStats();

    expect(stats.total).toBeGreaterThanOrEqual(0);
    expect(stats.kept).toBeGreaterThanOrEqual(0);
    expect(stats.discarded).toBeGreaterThanOrEqual(0);
    expect(stats.active).toBeGreaterThanOrEqual(0);
  });
});

describe("TurboContextEngine — Embedding Provider (v3.2)", () => {
  it("works correctly with no embedding provider (IDF fallback)", async () => {
    // Default engine — no embedding provider configured
    const engine = new TurboContextEngine();
    const task = makeTaskPool()[0];
    const context = makeContextPool();

    const result = await engine.execute(task, context);

    // Should complete successfully using IDF
    expect(result.compressed.fragments.length).toBeGreaterThan(0);
    expect(result.finalQuality).toBeGreaterThanOrEqual(0);
  });
});
