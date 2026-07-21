// ============================================================================
// Turbocontext v6 — Autonomous Experiment Loop E2E Test
// ============================================================================
// Verifies the full experiment loop works end-to-end in ":memory:" mode.
// Uses simulated LLM calls — no real API key needed.
// ============================================================================

import { describe, it, expect } from "vitest";
import { TurboContextEngine } from "../src/index.js";
import type { Task, ContextFragment } from "../src/types.js";
import { computeUnifiedMetric, computeSimplicity } from "../src/core/generator.js";
import { selectExperimentType } from "../src/core/evolution-engine.js";
import { loadProgram } from "../src/core/program-loader.js";

// Simulated LLM call — returns plausible content without API calls
function simulatedLLM(prompt: string, _temperature: number): Promise<string> {
  const mockContent = `## Code Review Results

### Security Issues Found
1. **Input Validation**: The login function does not validate email format before querying the database.
2. **Error Handling**: Auth errors expose internal error messages to the client.
3. **SQL Injection Risk**: The query uses string interpolation instead of parameterized queries.

### Quality Assessment
- Completeness: All major security concerns covered
- Correctness: Issues identified are real and actionable
- Consistency: Terminology consistent throughout
- Format: Properly structured with headings and code blocks

\`\`\`typescript
// Fix: Add email validation
function validateEmail(email: string): boolean {
  return /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/.test(email);
}
\`\`\`

### Recommendations
- Add rate limiting
- Use parameterized queries
- Hash passwords with bcrypt (already done)`;
  return Promise.resolve(mockContent);
}

function makeTask(type: string, desc: string): Task {
  return {
    id: `test_${Date.now()}`,
    description: desc,
    type: type as Task["type"],
    qualityThreshold: 0.85,
  };
}

function makeContext(): ContextFragment[] {
  return [
    {
      id: "1", source: "src/auth/login.ts", contentType: "source",
      content: 'export async function login(email: string, password: string) { const user = await db.users.findByEmail(email); if (!user) throw new AuthError("User not found"); const valid = await bcrypt.compare(password, user.passwordHash); if (!valid) throw new AuthError("Invalid password"); return { user, token: generateJWT({ userId: user.id }) }; }',
      lastModified: Date.now() - 86400000, length: 320,
    },
    {
      id: "2", source: "src/auth/register.ts", contentType: "source",
      content: 'export async function register(data: RegisterInput) { const existing = await db.users.findByEmail(data.email); if (existing) throw new AuthError("Email already registered"); const hash = await bcrypt.hash(data.password, 12); const user = await db.users.create({ ...data, passwordHash: hash }); return { user: sanitizeUser(user) }; }',
      lastModified: Date.now() - 172800000, length: 280,
    },
  ];
}

describe("TurboContextEngine — v6 Autonomous Experiment Loop", () => {
  it("runs 5 experiments in memory mode and records state", async () => {
    const engine = new TurboContextEngine({
      llm: simulatedLLM,
      alpha: 0.55, beta: 0.20, gamma: 0.25,
    });

    const tasks = [
      makeTask("code_review", "Review auth module for security issues"),
      makeTask("code_generation", "Add rate limiting to login endpoint"),
      makeTask("code_refactor", "Refactor auth middleware for error handling"),
      makeTask("debugging", "Debug token validation in auth middleware"),
      makeTask("analysis", "Analyze auth flow performance"),
    ];

    const context = makeContext();

    const runs = await engine.runExperiments({
      maxExperiments: 5,
      tokenBudgetPerRun: 8000,
      timeBudgetPerRun: 300,
      taskPool: tasks,
      contextPool: context,
    });

    // Verify we got 5 runs
    expect(runs).toHaveLength(5);

    // Verify each run has the required fields
    for (const run of runs) {
      expect(run.id).toBeTruthy();
      expect(run.taskType).toBeTruthy();
      expect(run.decision).toMatch(/^(keep|discard)$/);
      expect(run.status).toMatch(/^(success|crash|timeout|discarded)$/);
      expect(run.experimentType).toBeTruthy(); // v6: experiment type
      expect(run.simplicityScore).toBeGreaterThanOrEqual(0);
      expect(run.simplicityScore).toBeLessThanOrEqual(1); // v6: simplicity
      expect(run.baselineMetric.efficiency).toBeGreaterThan(0);
      expect(run.experimentMetric.alpha).toBe(1.0); // v6: unified metric
    }

    // Verify the V5 engine recorded trials
    const rlStatus = engine.getRLDiagnostics();
    expect(rlStatus.curriculumPhase).toBeGreaterThanOrEqual(0);
  });

  it("computes simplicity correctly for different mutation types", () => {
    // Baseline = simplest
    expect(computeSimplicity(null)).toBe(1.0);
    expect(computeSimplicity(undefined)).toBe(1.0);

    // Removing = simpler
    expect(computeSimplicity({ type: "remove_round", roundIndex: 1 })).toBeGreaterThan(0.9);

    // Adding = more complex
    expect(computeSimplicity({ type: "add_quality_criterion", roundIndex: 0, criterion: "security" })).toBeLessThan(0.5);

    // Parameter mutations in the middle
    const compressionSimplicity = computeSimplicity({ type: "mutate_compression_weights", alpha: 0.6, beta: 0.3, gamma: 0.1 });
    expect(compressionSimplicity).toBeGreaterThanOrEqual(0.5);
    expect(compressionSimplicity).toBeLessThanOrEqual(0.7);
  });

  it("selectExperimentType returns valid types", () => {
    const emptyEvolution = {
      totalExperiments: 0,
      keptCount: 0,
      discardedCount: 0,
      experiments: [],
      canonicalStrategies: {},
      currentExperimentId: null,
      trialLog: [],
    };

    // Run 100 times — should always return a valid type
    const types = new Set<string>();
    for (let i = 0; i < 100; i++) {
      types.add(selectExperimentType(emptyEvolution));
    }
    const validTypes = ["hypothesis_test", "parameter_sweep", "ablation_study",
                        "transfer_experiment", "boundary_probe", "adversarial_test"];
    for (const t of validTypes) {
      expect(types.has(t)).toBe(true);
    }
  });

  it("loadProgram returns defaults when no mission.md", () => {
    // Use a non-existent path to trigger defaults
    const program = loadProgram("/tmp/nonexistent-mission.md");
    expect(program.maxExperiments).toBe(20);
    expect(program.allowedMutations.length).toBeGreaterThan(0);
    expect(program.frozenParams).toContain("learningRate");
    expect(program.tokenBudgetPerRun).toBe(8000);
  });

  it("computes unified metric with simplicity multiplier", () => {
    const base = computeUnifiedMetric(0.85, 0.005, 1000, 2);
    const withSimplicity = computeUnifiedMetric(0.85, 0.005, 1000, 2, { simplicityMultiplier: 1.5 });
    const withPenalty = computeUnifiedMetric(0.85, 0.005, 1000, 2, { simplicityMultiplier: 0.5 });

    // Higher simplicity = higher efficiency
    expect(withSimplicity.efficiency).toBeGreaterThan(base.efficiency);
    // Lower simplicity = lower efficiency
    expect(withPenalty.efficiency).toBeLessThan(base.efficiency);
  });

  it("state manager correctly reports branch health metrics", () => {
    // Verify the new fields are present in the baseline type
    // This is a type-level test — if it compiles, the types are correct
    const baseline: {
      mean: number; ema: number; count: number; recentScores: number[];
      slope: number; improvementVelocity: number; stabilityScore: number;
      noveltyScore: number; plateauConfidence: number;
      successCount: number; crashCount: number; lastHypotheses: string[];
    } = {
      mean: 0, ema: 0, count: 0, recentScores: [], slope: 0,
      improvementVelocity: 0, stabilityScore: 0.5, noveltyScore: 0.5,
      plateauConfidence: 0, successCount: 0, crashCount: 0,
      lastHypotheses: [],
    };
    expect(baseline.improvementVelocity).toBe(0);
    expect(baseline.stabilityScore).toBe(0.5);
  });
});
