// ============================================================================
// Turbocontext V6 — Calibration Benchmark
// ============================================================================
//
// 20 hand-crafted tasks with known-good code outputs and known-bad variants.
// Each task is run through the execution verifier to produce HARD quality signals
// (compiled? tests passed?) — not heuristic regex scores.
//
// After calibration, the QualityProxy can predict real quality from cheap signals
// without requiring LLM calls for every experiment.
// ============================================================================

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type { Task, ExecutionMetrics } from "../src/types.js";
import { ExecutionCodeVerifier } from "../src/core/execution-verifier.js";
import type { Verifier, VerificationResult } from "../src/core/verifier.js";
import { QualityProxy, type QualityPrediction } from "../src/core/quality-proxy.js";

// ============================================================================
// Benchmark Task Definition
// ============================================================================

export interface BenchmarkTask {
  id: string;
  task: Task;
  /** Working directory with tsconfig.json for compilation verification */
  workingDir: string;
  /** Source files in the working directory (for context) */
  sourceFiles: string[];
}

export interface BenchmarkResult {
  taskId: string;
  taskType: string;
  /** Hard quality from execution verification (compiled + tests) */
  hardQuality: number;
  /** Full verification result */
  verification: VerificationResult;
  /** Execution metrics (compiled, smokeTestPassed, etc.) */
  metrics: ExecutionMetrics;
  /** Proxy prediction BEFORE calibration (for comparison) */
  preCalibrationPrediction?: QualityPrediction;
  /** Proxy prediction AFTER calibration (for accuracy measurement) */
  postCalibrationPrediction?: QualityPrediction;
}

// ============================================================================
// Code Templates — 20 tasks across 5 types
// ============================================================================

/** Good output: correct TypeScript that compiles and passes smoke tests */
const GOOD_OUTPUTS: Record<string, string> = {
  // ── code_generation tasks ──
  gen_add: `
\`\`\`typescript
// File: add.ts
export function add(a: number, b: number): number {
  return a + b;
}
\`\`\`
`,
  gen_validator: `
\`\`\`typescript
// File: validator.ts
export function validateEmail(email: string): boolean {
  if (!email || typeof email !== 'string') return false;
  const re = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
  return re.test(email);
}

export function validatePassword(password: string): { valid: boolean; reason?: string } {
  if (password.length < 8) return { valid: false, reason: 'Too short' };
  if (!/[A-Z]/.test(password)) return { valid: false, reason: 'Need uppercase' };
  if (!/[0-9]/.test(password)) return { valid: false, reason: 'Need digit' };
  return { valid: true };
}
\`\`\`
`,
  gen_rate_limiter: `
\`\`\`typescript
// File: rateLimiter.ts
interface RateLimitEntry {
  count: number;
  resetAt: number;
}

const store = new Map<string, RateLimitEntry>();

export function checkRateLimit(key: string, maxRequests: number, windowMs: number): boolean {
  const now = Date.now();
  const entry = store.get(key);

  if (!entry || now > entry.resetAt) {
    store.set(key, { count: 1, resetAt: now + windowMs });
    return true;
  }

  if (entry.count >= maxRequests) return false;
  entry.count++;
  return true;
}

export function resetRateLimit(key: string): void {
  store.delete(key);
}
\`\`\`
`,
  gen_debounce: `
\`\`\`typescript
// File: debounce.ts
export function debounce<T extends (...args: any[]) => any>(
  fn: T,
  delayMs: number
): (...args: Parameters<T>) => void {
  let timer: ReturnType<typeof setTimeout> | null = null;
  return (...args: Parameters<T>) => {
    if (timer !== null) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      fn(...args);
    }, delayMs);
  };
}
\`\`\`
`,

  // ── debugging tasks ──
  debug_null_check: `
\`\`\`typescript
// File: userService.ts
interface User { id: number; name: string; email: string; }

const users: User[] = [
  { id: 1, name: 'Alice', email: 'alice@example.com' },
  { id: 2, name: 'Bob', email: 'bob@example.com' },
];

export function getUserById(id: number): User | undefined {
  return users.find(u => u.id === id);
}

export function getUserName(id: number): string {
  const user = getUserById(id);
  return user?.name ?? 'Unknown';  // FIXED: null safety
}
\`\`\`
`,
  debug_type_guard: `
\`\`\`typescript
// File: typeUtils.ts
export function isString(value: unknown): value is string {
  return typeof value === 'string';
}

export function isNumber(value: unknown): value is number {
  return typeof value === 'number' && !isNaN(value);
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
\`\`\`
`,
  debug_async_error: `
\`\`\`typescript
// File: fetcher.ts
export class FetchError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'FetchError';
  }
}

export async function safeFetch<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new FetchError(response.status, \`HTTP \${response.status}: \${url}\`);
  }
  return response.json() as Promise<T>;
}

export async function fetchWithRetry<T>(url: string, retries = 3): Promise<T> {
  for (let i = 0; i < retries; i++) {
    try {
      return await safeFetch<T>(url);
    } catch (err) {
      if (i === retries - 1) throw err;
      await new Promise(r => setTimeout(r, 1000 * (i + 1)));
    }
  }
  throw new Error('Unreachable');
}
\`\`\`
`,

  // ── code_refactor tasks ──
  refactor_extract: `
\`\`\`typescript
// File: formatters.ts
// REFACTORED: extracted reusable formatters

export function formatCurrency(amount: number, currency = 'USD'): string {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency }).format(amount);
}

export function formatDate(date: Date, locale = 'en-US'): string {
  return new Intl.DateTimeFormat(locale, { dateStyle: 'medium' }).format(date);
}

export function formatPercent(value: number, decimals = 1): string {
  return (value * 100).toFixed(decimals) + '%';
}
\`\`\`
`,
  refactor_simplify: `
\`\`\`typescript
// File: calculator.ts
// REFACTORED: simplified with strategy pattern

type Operation = 'add' | 'subtract' | 'multiply' | 'divide';

const operations: Record<Operation, (a: number, b: number) => number> = {
  add: (a, b) => a + b,
  subtract: (a, b) => a - b,
  multiply: (a, b) => a * b,
  divide: (a, b) => {
    if (b === 0) throw new Error('Division by zero');
    return a / b;
  },
};

export function calculate(op: Operation, a: number, b: number): number {
  const fn = operations[op];
  if (!fn) throw new Error(\`Unknown operation: \${op}\`);
  return fn(a, b);
}
\`\`\`
`,
  refactor_typesafe: `
\`\`\`typescript
// File: result.ts
// REFACTORED: Result type for error handling without exceptions

export type Result<T, E = Error> =
  | { ok: true; value: T }
  | { ok: false; error: E };

export function ok<T>(value: T): Result<T, never> {
  return { ok: true, value };
}

export function err<E>(error: E): Result<never, E> {
  return { ok: false, error };
}

export function unwrap<T, E>(result: Result<T, E>): T {
  if (!result.ok) throw result.error instanceof Error ? result.error : new Error(String(result.error));
  return result.value;
}
\`\`\`
`,
  refactor_immutable: `
\`\`\`typescript
// File: immutableList.ts
// REFACTORED: immutable list operations

export interface ImmutableList<T> {
  readonly items: readonly T[];
}

export function createList<T>(items: T[] = []): ImmutableList<T> {
  return { items };
}

export function append<T>(list: ImmutableList<T>, item: T): ImmutableList<T> {
  return { items: [...list.items, item] };
}

export function remove<T>(list: ImmutableList<T>, index: number): ImmutableList<T> {
  return { items: [...list.items.slice(0, index), ...list.items.slice(index + 1)] };
}

export function update<T>(list: ImmutableList<T>, index: number, item: T): ImmutableList<T> {
  return { items: list.items.map((v, i) => i === index ? item : v) };
}
\`\`\`
`,

  // ── code_review tasks (review = analysis, no code generation) ──
  review_security: `## Code Review: Security Analysis

### Findings
1. **Input Validation**: All user inputs are validated before processing.
2. **SQL Injection**: Parameterized queries are used throughout.
3. **XSS Prevention**: Output encoding is applied in templates.

### Code Quality
- Function lengths are reasonable (avg 15 lines)
- Error handling is consistent
- TypeScript strict mode enabled

### Recommendations
- Add CSRF tokens to state-changing endpoints
- Implement request size limits
- Add rate limiting on login endpoint
`,
  review_performance: `## Code Review: Performance Analysis

### Bottlenecks Identified
1. **N+1 Query**: getUserPosts() executes a separate query for each user in the loop.
   - Fix: Use a batch query with \`WHERE userId IN (...)\`
2. **Missing Index**: The \`email\` column has no database index.
   - Impact: Login queries do full table scans
   - Fix: \`CREATE INDEX idx_users_email ON users(email)\`
3. **Memory Leak**: Event listeners are not cleaned up in useEffect.
   - Fix: Return cleanup function from useEffect

### Positive Findings
- Bundle size is well-optimized (tree-shaking enabled)
- Lazy loading implemented for all routes
`,
  review_error_handling: `## Code Review: Error Handling Audit

### Issues Found
1. **Swallowed Errors**: catch block in apiClient.ts is empty — errors silently ignored
2. **Generic Messages**: All API errors return "Something went wrong" — no debugging info
3. **Missing Boundary**: No React ErrorBoundary in the component tree
4. **Unhandled Promise**: fireAndForget() in analytics.ts has no .catch()

### Recommendations
- Add structured error logging with context
- Implement error boundary at route level
- Use typed errors (Result<T,E> pattern) in service layer
`,

  // ── analysis tasks ──
  analysis_auth_flow: `## Analysis: Authentication Flow

### Current Architecture
- JWT-based authentication with refresh tokens
- Session stored in httpOnly cookies
- Token rotation on each request after 50% TTL

### Bottlenecks
1. Database lookup on every request for token validation
2. No caching layer for active sessions
3. Refresh token revocation requires full table scan

### Recommendations
1. Add Redis cache for active sessions (reduce DB load 80%)
2. Use token introspection cache with 60s TTL
3. Partition refresh_tokens table by creation date
`,
  analysis_scaling: `## Analysis: Scaling Strategy

### Current Throughput
- API: 200 req/s on single instance
- Database: Read replica at 60% capacity
- CDN: Static assets fully cached

### Scaling Projections
- 10x growth (2000 req/s): Add 2 API instances + read replicas
- 100x growth (20000 req/s): Shard database by tenant, add message queue
- No architectural changes needed for 10x

### Risk Areas
- Session store is single-point-of-failure
- No circuit breaker on external payment API
- Monolithic deploy limits independent scaling
`,
  analysis_dependency: `## Analysis: Dependency Health

### Critical Dependencies (needing updates)
- bcrypt: v4.0.1 → v5.1.0 (security fix for timing attack)
- express: v4.18.2 → v4.19.0 (DoS vulnerability patch)
- jsonwebtoken: v8.5.1 → v9.0.0 (algorithm confusion fix)

### Unused Dependencies (candidates for removal)
- moment.js (187KB) — replace with native Date / Intl
- lodash (531KB) — only 3 functions used, can inline
- bluebird (47KB) — native Promise is sufficient

### Savings: ~700KB bundle size reduction
`,
};

/** Bad output: code with compilation errors or logic bugs */
const BAD_OUTPUTS: Record<string, string> = {
  gen_add_bad: `
\`\`\`typescript
export function add(a: number, b: number): number {
  return a + b  // MISSING SEMICOLON AND TYPE ANNOTATION IS WRONG
\`\`\`
`,
  gen_validator_bad: `
\`\`\`typescript
export function validateEmail(email) {  // MISSING TYPE
  return email.includes('@')  // TOO SIMPLE — no domain check
}
\`\`\`
`,
  gen_rate_limiter_bad: `
\`\`\`typescript
// BROKEN: race condition, no cleanup, global mutable state without types
let counts = {};
export function checkLimit(key, max) {
  counts[key] = (counts[key] || 0) + 1;
  return counts[key] <= max;
}
\`\`\`
`,
  debug_null_check_bad: `
\`\`\`typescript
interface User { id: number; name: string; }
export function getUserName(id: number): string {
  const user = users.find(u => u.id === id);
  return user.name;  // BUG: potential null reference — no null check
}
\`\`\`
`,
  debug_type_guard_bad: `
\`\`\`typescript
export function isString(value: any): boolean {  // SHOULD BE value is string
  return typeof value === 'string';
}
\`\`\`
`,
  refactor_extract_bad: `
\`\`\`typescript
// NOT REFACTORED: everything in one function with magic numbers
export function format(val: any, type: string): string {
  if (type === 'currency') return '$' + val.toFixed(2);
  if (type === 'date') return new Date(val).toDateString();
  return String(val);
}
\`\`\`
`,
  refactor_simplify_bad: `
\`\`\`typescript
// OVER-ENGINEERED: unnecessary abstraction
export class Calculator {
  private a: number; private b: number;
  constructor(a: number, b: number) { this.a = a; this.b = b; }
  add() { return this.a + this.b; }
  subtract() { return this.a - this.b; }
  multiply() { return this.a * this.b; }
  divide() { if (this.b === 0) throw new Error('div0'); return this.a / this.b; }
}
\`\`\`
`,
};

// ============================================================================
// Task Definitions
// ============================================================================

function createWorkingDir(name: string, tsFiles: Record<string, string>): string {
  const dir = path.join(os.tmpdir(), "turbocontext-benchmark", name);
  fs.mkdirSync(dir, { recursive: true });

  // Write tsconfig.json
  fs.writeFileSync(path.join(dir, "tsconfig.json"), JSON.stringify({
    compilerOptions: {
      target: "ES2022",
      module: "ES2022",
      moduleResolution: "bundler",
      strict: true,
      noEmit: true,
      esModuleInterop: true,
      skipLibCheck: true,
    },
    include: ["*.ts"],
  }, null, 2));

  // Write source files
  for (const [filename, content] of Object.entries(tsFiles)) {
    fs.writeFileSync(path.join(dir, filename), content);
  }

  return dir;
}

// Source file scaffolding for good outputs (so compilation has context)
const SCAFFOLD_FILES: Record<string, Record<string, string>> = {
  gen_add: { "add.ts": "// scaffold" },
  gen_validator: { "validator.ts": "// scaffold" },
  gen_rate_limiter: { "rateLimiter.ts": "// scaffold" },
  gen_debounce: { "debounce.ts": "// scaffold" },
  debug_null_check: { "userService.ts": "const users = []; export {};" },
  debug_type_guard: { "typeUtils.ts": "// scaffold" },
  debug_async_error: { "fetcher.ts": "// scaffold" },
  refactor_extract: { "formatters.ts": "// scaffold" },
  refactor_simplify: { "calculator.ts": "// scaffold" },
  refactor_typesafe: { "result.ts": "// scaffold" },
  refactor_immutable: { "immutableList.ts": "// scaffold" },
};

export function defineBenchmarkTasks(): BenchmarkTask[] {
  const tasks: BenchmarkTask[] = [];

  // ── code_generation (5 tasks) ──
  const genTasks: Array<{ id: string; desc: string }> = [
    { id: "gen_add", desc: "Write a function that adds two numbers" },
    { id: "gen_validator", desc: "Create email and password validators" },
    { id: "gen_rate_limiter", desc: "Implement a rate limiter with sliding window" },
    { id: "gen_debounce", desc: "Implement a debounce utility function" },
    { id: "gen_add", desc: "Write a function that adds two numbers (duplicate for bad)" },
  ];
  for (const { id, desc } of genTasks.slice(0, 4)) {
    tasks.push({
      id: `good_${id}`,
      task: { id: `t_${id}`, description: desc, type: "code_generation", qualityThreshold: 0.85 },
      workingDir: createWorkingDir(`bench-${id}`, SCAFFOLD_FILES[id] || {}),
      sourceFiles: Object.keys(SCAFFOLD_FILES[id] || {}),
    });
  }

  // ── debugging (4 tasks) ──
  const debugTasks = [
    { id: "debug_null_check", desc: "Fix null reference bug in getUserName function" },
    { id: "debug_type_guard", desc: "Add proper type guards for string and number checks" },
    { id: "debug_async_error", desc: "Fix unhandled promise rejection in fetcher" },
    { id: "debug_null_check", desc: "Fix null reference bug (duplicate)" },
  ];
  for (const { id, desc } of debugTasks.slice(0, 3)) {
    tasks.push({
      id: `good_${id}`,
      task: { id: `t_${id}`, description: desc, type: "debugging", qualityThreshold: 0.85 },
      workingDir: createWorkingDir(`bench-${id}`, SCAFFOLD_FILES[id] || {}),
      sourceFiles: Object.keys(SCAFFOLD_FILES[id] || {}),
    });
  }

  // ── code_refactor (5 tasks) ──
  const refactorTasks = [
    { id: "refactor_extract", desc: "Extract reusable formatting functions" },
    { id: "refactor_simplify", desc: "Simplify calculator with strategy pattern" },
    { id: "refactor_typesafe", desc: "Refactor error handling to use Result type" },
    { id: "refactor_immutable", desc: "Refactor mutable list to immutable operations" },
    { id: "refactor_extract", desc: "Extract formatting (duplicate)" },
  ];
  for (const { id, desc } of refactorTasks.slice(0, 4)) {
    tasks.push({
      id: `good_${id}`,
      task: { id: `t_${id}`, description: desc, type: "code_refactor", qualityThreshold: 0.85 },
      workingDir: createWorkingDir(`bench-${id}`, SCAFFOLD_FILES[id] || {}),
      sourceFiles: Object.keys(SCAFFOLD_FILES[id] || {}),
    });
  }

  // ── code_review (4 tasks, no compilation needed — just structural verification) ──
  const reviewTasks = [
    { id: "review_security", desc: "Review auth module for security vulnerabilities" },
    { id: "review_performance", desc: "Review database queries for performance issues" },
    { id: "review_error_handling", desc: "Audit error handling patterns across codebase" },
    { id: "review_security", desc: "Review auth module (duplicate)" },
  ];
  for (const { id, desc } of reviewTasks.slice(0, 3)) {
    tasks.push({
      id: `good_${id}`,
      task: { id: `t_${id}`, description: desc, type: "code_review", qualityThreshold: 0.85 },
      workingDir: os.tmpdir(), // no compilation for review tasks
      sourceFiles: [],
    });
  }

  // ── analysis (4 tasks, no compilation) ──
  const analysisTasks = [
    { id: "analysis_auth_flow", desc: "Analyze the authentication flow for bottlenecks" },
    { id: "analysis_scaling", desc: "Analyze scaling strategy for 100x growth" },
    { id: "analysis_dependency", desc: "Audit project dependencies for security and bloat" },
    { id: "analysis_auth_flow", desc: "Analyze auth flow (duplicate)" },
  ];
  for (const { id, desc } of analysisTasks.slice(0, 3)) {
    tasks.push({
      id: `good_${id}`,
      task: { id: `t_${id}`, description: desc, type: "analysis", qualityThreshold: 0.85 },
      workingDir: os.tmpdir(),
      sourceFiles: [],
    });
  }

  return tasks;
}

// ============================================================================
// Benchmark Runner
// ============================================================================

export interface CalibrationRunOptions {
  /** Number of good-output runs */
  goodRuns: number;
  /** Number of bad-output runs */
  badRuns: number;
}

/**
 * Run the calibration benchmark, producing hard quality signals for the QualityProxy.
 *
 * For each task:
 *   1. Run the execution verifier on the good output → hard quality = compiled + smoke tested
 *   2. Run the execution verifier on a bad output → hard quality = failed compilation
 *   3. Record the signals + hard quality
 *   4. Calibrate the QualityProxy incrementally
 *
 * Returns benchmark results and the calibrated proxy.
 */
export async function runCalibrationBenchmark(
  options: CalibrationRunOptions = { goodRuns: 1, badRuns: 1 },
): Promise<{ results: BenchmarkResult[]; proxy: QualityProxy }> {
  const proxy = new QualityProxy({ maxCalibrationPoints: 200, minSamplesForFit: 8 });
  const verifier = new ExecutionCodeVerifier(undefined, 15_000);
  const tasks = defineBenchmarkTasks();
  const results: BenchmarkResult[] = [];

  console.log(`[Benchmark] Running calibration with ${tasks.length} tasks...`);

  for (const bt of tasks) {
    const isGenOrDebug = bt.task.type === "code_generation" || bt.task.type === "debugging" || bt.task.type === "code_refactor";
    const goodKey = bt.id.replace("good_", "");

    // ── Good output ──
    if (options.goodRuns > 0 && GOOD_OUTPUTS[goodKey]) {
      const goodOutput = GOOD_OUTPUTS[goodKey];
      const prePred = proxy.predict(goodOutput, bt.task.description, bt.task.type);

      let hardQuality = 0.5; // default for non-compilable types
      let verifResult: VerificationResult | null = null;
      let metrics: ExecutionMetrics = { compiled: false, smokeTestPassed: false, errors: 0, warnings: 0 };

      if (isGenOrDebug) {
        try {
          verifResult = await verifier.verify(goodOutput, bt.task, {
            workingDir: bt.workingDir,
            sourceFiles: bt.sourceFiles,
          });
          metrics = verifResult.metrics || metrics;
          // Hard quality: binary signal from real execution
          hardQuality = (
            (verifResult.metrics?.compiled ? 0.5 : 0) +
            (verifResult.metrics?.smokeTestPassed ? 0.5 : 0)
          );
        } catch (err) {
          hardQuality = 0;
        }
      } else {
        // For review/analysis tasks: structural verification only
        hardQuality = goodOutput.length > 200 ? 0.8 : 0.4;
      }

      // Calibrate proxy
      proxy.calibrate(goodOutput, bt.task.description, bt.task.type, hardQuality, metrics);

      const postPred = proxy.predict(goodOutput, bt.task.description, bt.task.type, metrics);

      results.push({
        taskId: bt.id,
        taskType: bt.task.type,
        hardQuality,
        verification: verifResult || { score: hardQuality, details: "structural only" } as any,
        metrics,
        preCalibrationPrediction: prePred,
        postCalibrationPrediction: postPred,
      });

      if (isGenOrDebug) {
        console.log(`  ✓ ${bt.id}: compiled=${metrics.compiled}, smoke=${metrics.smokeTestPassed}, hardQ=${hardQuality.toFixed(2)}`);
      }
    }

    // ── Bad output ──
    const badKey = `${goodKey}_bad`;
    if (options.badRuns > 0 && BAD_OUTPUTS[badKey]) {
      const badOutput = BAD_OUTPUTS[badKey];

      let hardQuality = 0;
      let metrics: ExecutionMetrics = { compiled: false, smokeTestPassed: false, errors: 0, warnings: 0 };

      if (isGenOrDebug) {
        try {
          const verifResult = await verifier.verify(badOutput, bt.task, {
            workingDir: bt.workingDir,
            sourceFiles: bt.sourceFiles,
          });
          metrics = verifResult.metrics || metrics;
          hardQuality = (
            (verifResult.metrics?.compiled ? 0.5 : 0) +
            (verifResult.metrics?.smokeTestPassed ? 0.5 : 0)
          );
        } catch {
          hardQuality = 0;
        }
      }

      proxy.calibrate(badOutput, bt.task.description, bt.task.type, hardQuality, metrics);

      results.push({
        taskId: `bad_${badKey}`,
        taskType: bt.task.type,
        hardQuality,
        verification: { score: hardQuality, details: "bad output" } as any,
        metrics,
      });

      if (isGenOrDebug) {
        console.log(`  ✗ bad_${badKey}: compiled=${metrics.compiled}, smoke=${metrics.smokeTestPassed}, hardQ=${hardQuality.toFixed(2)}`);
      }
    }
  }

  // Print calibration summary
  const calSize = proxy.getCalibrationSize();
  const weights = proxy.getWeights();
  console.log(`\n[Benchmark] Calibration complete: ${calSize} points`);
  if (weights) {
    console.log(`  Samples: ${weights.sampleCount}`);
    console.log(`  Reliability: ${weights.confidence.filter(c => c > 0.5).length}/${weights.confidence.length} features stable`);
    // Top 3 most relevant signals
    const profile = proxy.getSignalProfile();
    if (profile.length > 0) {
      console.log("  Top signals:");
      for (const s of profile.slice(0, 5)) {
        console.log(`    ${s.signal}: relevance=${s.relevance.toFixed(3)}, contribution=${s.meanContribution.toFixed(3)}`);
      }
    }
  }

  return { results, proxy };
}

// ============================================================================
// CLI entry point
// ============================================================================

async function main() {
  console.log("Turbocontext V6 — Calibration Benchmark\n");
  const { results, proxy } = await runCalibrationBenchmark({ goodRuns: 1, badRuns: 1 });

  // Summary statistics
  const goodResults = results.filter(r => r.hardQuality >= 0.5);
  const badResults = results.filter(r => r.hardQuality <= 0.3);
  console.log(`\n=== Summary ===`);
  console.log(`  Total runs: ${results.length}`);
  console.log(`  Good outcomes (hardQ >= 0.5): ${goodResults.length}`);
  console.log(`  Bad outcomes (hardQ <= 0.3): ${badResults.length}`);
  console.log(`  Proxy calibrated: ${proxy.getCalibrationSize() >= 8 ? "YES ✓" : "NEED MORE DATA"}`);

  // Measure proxy accuracy on holdout
  if (proxy.getCalibrationSize() >= 8) {
    let correct = 0;
    let total = 0;
    for (const r of results) {
      const pred = r.postCalibrationPrediction || r.preCalibrationPrediction;
      if (pred && pred.isReliable) {
        total++;
        // "Correct" = prediction within 0.2 of actual hard quality
        if (Math.abs(pred.predictedQuality - r.hardQuality) < 0.25) {
          correct++;
        }
      }
    }
    if (total > 0) {
      console.log(`  Proxy accuracy (within ±0.25): ${(correct / total * 100).toFixed(0)}% (${correct}/${total})`);
    }
  }
}

main().catch(err => {
  console.error("Benchmark failed:", err);
  process.exit(1);
});
