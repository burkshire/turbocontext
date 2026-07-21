// ============================================================
// Session Memory Tests
// ============================================================
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { SessionMemory } from "../src/core/session-memory.js";
import { writeFileSync, unlinkSync, existsSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";

function tmpPath(name: string): string {
  return join(tmpdir(), `turbocontext-test-${name}-${Date.now()}.json`);
}

describe("SessionMemory", () => {
  let sm: SessionMemory;
  let sessionsPath: string;
  let indexPath: string;

  beforeEach(() => {
    sessionsPath = tmpPath("sessions");
    indexPath = tmpPath("index");
    sm = new SessionMemory(sessionsPath, indexPath);
  });

  afterEach(() => {
    try { unlinkSync(sessionsPath); } catch { /* ok */ }
    try { unlinkSync(indexPath); } catch { /* ok */ }
  });

  // ── Recording ──

  it("records a session and assigns id/timestamp/keywords", () => {
    const record = sm.record({
      taskDescription: "Review the auth module for security issues",
      taskType: "code_review",
      workingDirectory: "/Users/test/project",
      filesRead: ["/Users/test/project/src/auth/login.ts"],
      filesModified: [],
      strategy: "Check token validation first, then authorization logic",
      outcome: "success",
      selfAssessment: 0.85,
      notes: "Found 2 issues",
      roundCount: 3,
    });

    expect(record.id).toBeDefined();
    expect(record.id.length).toBe(12);
    expect(record.timestamp).toBeDefined();
    expect(record.keywords.length).toBeGreaterThan(0);
    expect(record.keywords.length).toBeLessThanOrEqual(15);
  });

  it("records multiple sessions and increments count", () => {
    sm.record(makeRecord({ taskDescription: "Task A" }));
    sm.record(makeRecord({ taskDescription: "Task B" }));
    sm.record(makeRecord({ taskDescription: "Task C" }));
    expect(sm.getSessionCount()).toBe(3);
  });

  // ── Keyword extraction ──

  it("splits camelCase tokens", () => {
    const record = sm.record({
      taskDescription: "Refactor authMiddleware for better tokenValidation",
      taskType: "code_refactor",
      workingDirectory: "/tmp/test",
      filesRead: [],
      filesModified: [],
      strategy: "",
      outcome: "success",
      selfAssessment: 0.8,
      notes: "",
      roundCount: 1,
    });

    // "authMiddleware" should split into "auth" and "middleware"
    // "tokenValidation" should split into "token" and "validation"
    const kw = record.keywords;
    expect(kw).toContain("auth");
    expect(kw).toContain("middleware");
    expect(kw).toContain("token");
    expect(kw).toContain("validation");
  });

  it("removes stopwords", () => {
    const record = sm.record({
      taskDescription: "the and for with this is a the of in to",
      taskType: "general",
      workingDirectory: "/tmp/test",
      filesRead: [],
      filesModified: [],
      strategy: "",
      outcome: "success",
      selfAssessment: 0.5,
      notes: "",
      roundCount: 1,
    });

    // All stopwords — keywords should be empty or minimal
    const hasStopwords = record.keywords.some(k =>
      ["the", "and", "for", "with", "this", "is"].includes(k)
    );
    expect(hasStopwords).toBe(false);
  });

  // ── Similarity ──

  it("returns high similarity for identical task descriptions", () => {
    sm.record(makeRecord({ taskDescription: "Fix the login bug in auth module" }));
    const result = sm.recall({
      taskDescription: "Fix the login bug in auth module",
      workingDirectory: "/Users/test/project",
    });
    expect(result.similarSessions.length).toBeGreaterThanOrEqual(1);
    if (result.similarSessions.length > 0) {
      expect(result.similarSessions[0].similarity).toBeGreaterThan(0.7);
    }
  });

  it("returns low similarity for unrelated tasks", () => {
    sm.record(makeRecord({ taskDescription: "Write unit tests for the payment processor" }));
    const result = sm.recall({
      taskDescription: "Add a dark mode toggle to the settings UI",
      workingDirectory: "/Users/test/project",
    });
    // Should still return the session (it's the only one) but with low similarity
    if (result.similarSessions.length > 0) {
      expect(result.similarSessions[0].similarity).toBeLessThan(0.5);
    }
  });

  // ── File recommendations ──

  it("recommends files that were useful in similar sessions", () => {
    sm.record(makeRecord({
      taskDescription: "Review auth module security",
      filesRead: ["/Users/test/project/src/auth/login.ts"],
      filesModified: ["/Users/test/project/src/auth/login.ts"],
      taskType: "code_review",
    }));
    sm.record(makeRecord({
      taskDescription: "Add input validation to login",
      filesRead: ["/Users/test/project/src/auth/login.ts"],
      filesModified: ["/Users/test/project/src/auth/validate.ts"],
      taskType: "code_generation",
    }));

    const result = sm.recall({
      taskDescription: "Review the authentication system",
      workingDirectory: "/Users/test/project",
      taskType: "code_review",
    });

    expect(result.recommendedFiles.length).toBeGreaterThan(0);
    const loginFile = result.recommendedFiles.find(f => f.path.includes("login.ts"));
    expect(loginFile).toBeDefined();
    if (loginFile) {
      expect(loginFile.sessionCount).toBeGreaterThanOrEqual(2);
    }
  });

  it("scopes file recommendations to working directory", () => {
    sm.record(makeRecord({
      filesRead: ["/Users/test/project-b/src/other.ts"],
      workingDirectory: "/Users/test/project-b",
    }));
    sm.record(makeRecord({
      filesRead: ["/Users/test/project-a/src/main.ts"],
      workingDirectory: "/Users/test/project-a",
    }));

    const result = sm.recall({
      taskDescription: "review code",
      workingDirectory: "/Users/test/project-a",
    });

    // Should only recommend files from project-a
    const hasProjectB = result.recommendedFiles.some(f =>
      f.path.includes("project-b")
    );
    expect(hasProjectB).toBe(false);
  });

  // ── Persistence ──

  it("persists and reloads sessions across instances", () => {
    sm.record(makeRecord({ taskDescription: "Task that should persist" }));
    sm.save();

    const sm2 = SessionMemory.load(sessionsPath, indexPath);
    expect(sm2.getSessionCount()).toBe(1);

    const result = sm2.recall({
      taskDescription: "Task that should persist",
      workingDirectory: "/Users/test/project",
    });
    expect(result.similarSessions.length).toBeGreaterThanOrEqual(1);
  });

  // ── Pruning ──

  it("prunes sessions beyond max count", () => {
    for (let i = 0; i < 25; i++) {
      sm.record(makeRecord({ taskDescription: `Task ${i}` }));
    }
    expect(sm.getSessionCount()).toBe(25);
    const removed = sm.prune({ maxSessions: 10 });
    expect(removed).toBe(15);
    expect(sm.getSessionCount()).toBe(10);
  });

  it("prunes old sessions by age", () => {
    // Record a session with an old timestamp by accessing internals
    const record = sm.record(makeRecord({ taskDescription: "Old task" }));
    // Manually set old timestamp (access private field for testing)
    (sm as any).sessions[0].timestamp = "2020-01-01T00:00:00.000Z";
    sm.save();

    const removed = sm.prune({ maxAgeDays: 365 });
    expect(removed).toBeGreaterThanOrEqual(1);
  });

  // ── Empty corpus ──

  it("returns empty results for empty corpus", () => {
    const result = sm.recall({
      taskDescription: "Any task",
      workingDirectory: "/tmp",
    });
    expect(result.similarSessions).toHaveLength(0);
    expect(result.recommendedFiles).toHaveLength(0);
    expect(result.summary).toContain("cold start");
  });

  // ── Outcome bonus ──

  it("ranks successful sessions higher than failures for same keywords", () => {
    sm.record(makeRecord({
      taskDescription: "Fix the login rate limiter",
      outcome: "failure",
      selfAssessment: 0.3,
    }));
    sm.record(makeRecord({
      taskDescription: "Fix the login rate limiter",
      outcome: "success",
      selfAssessment: 0.9,
    }));

    const result = sm.recall({
      taskDescription: "Fix the login rate limiter",
      workingDirectory: "/Users/test/project",
    });

    expect(result.similarSessions.length).toBeGreaterThanOrEqual(2);
    // The successful session should rank first
    const first = result.similarSessions[0];
    expect(first.session.outcome).toBe("success");
  });

  // ── Stats ──

  it("returns accurate corpus statistics", () => {
    sm.record(makeRecord({ taskType: "code_review", filesRead: ["/tmp/proj/a.ts"] }));
    sm.record(makeRecord({ taskType: "code_generation", filesRead: ["/tmp/proj/b.ts"] }));
    sm.record(makeRecord({ taskType: "code_review", filesRead: ["/tmp/proj/c.ts"] }));

    const stats = sm.stats();
    expect(stats.totalSessions).toBe(3);
    expect(stats.perTaskType["code_review"]).toBe(2);
    expect(stats.perTaskType["code_generation"]).toBe(1);
    expect(stats.totalUniqueFiles).toBe(3);
    expect(stats.oldestSession).toBeDefined();
    expect(stats.newestSession).toBeDefined();
  });
});

// ── Helper ──

function makeRecord(overrides: Partial<{
  taskDescription: string;
  taskType: string;
  workingDirectory: string;
  filesRead: string[];
  filesModified: string[];
  strategy: string;
  outcome: "success" | "partial" | "failure";
  selfAssessment: number;
  notes: string;
  roundCount: number;
}> = {}) {
  return {
    taskDescription: overrides.taskDescription ?? "Review code for issues",
    taskType: (overrides.taskType ?? "code_review") as any,
    workingDirectory: overrides.workingDirectory ?? "/Users/test/project",
    filesRead: overrides.filesRead ?? [],
    filesModified: overrides.filesModified ?? [],
    strategy: overrides.strategy ?? "Standard review approach",
    outcome: overrides.outcome ?? "success",
    selfAssessment: overrides.selfAssessment ?? 0.8,
    notes: overrides.notes ?? "",
    roundCount: overrides.roundCount ?? 2,
  };
}
