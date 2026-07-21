/**
 * TurboContext Calibration Benchmark Suite
 * =========================================
 * Multi-level benchmark to calibrate the RL learning pipeline.
 * Each level has KNOWN GROUND TRUTH — we know which files are required
 * and the system must learn to assign them higher causal utility.
 *
 * Principle: "Point the telescope at a known star before deep space."
 *
 * Usage:
 *   export DEEPSEEK_API_KEY=sk-...
 *   npx tsx scripts/benchmark-suites.ts <level>
 *
 * Levels:
 *   1 — Single required file (5 files, 1 needed)        → calibrate Thompson
 *   2 — Two required files (6 files, 2 needed)           → calibrate credit assignment
 *   3 — File reordering matters (4 files, order affects) → calibrate compression params
 *   4 — Real mini-project (unknown ground truth)          → full pipeline validation
 */

import { TurboContextEngine } from "../src/index.js";
import type { ContextFragment, Task } from "../src/types.js";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

// ═══════════════════════════════════════════════════════════════════════════
// Level 1: Single Required File + Distractors
// ═══════════════════════════════════════════════════════════════════════════
//
// Ground truth: user-types.ts is REQUIRED (defines User and Result).
// The other 4 files are distractions.
// System must learn: user-types.ts has high causal utility.
//
// Task: "Write a TypeScript function `validateUser(user: User): Result`"

const LEVEL_1 = {
  name: "Single Required File",
  description: "5 files, 1 contains required type definitions. System must identify it.",

  task: {
    id: "bench-l1",
    description: "Write a TypeScript function 'validateUser(user: User): Result' that validates a User object and returns a Result. Export it.",
    type: "code_generation" as const,
    complexity: 0.3,
    qualityThreshold: 0.6,
    latencyBudgetMs: 60000,
  },

  files: {
    required: [
      {
        source: "src/types/user-types.ts",
        content: `export interface User { id: number; name: string; email: string; age: number; }
export interface Result { valid: boolean; errors: string[]; }
export type ValidationRule = (user: User) => string | null;`,
      },
    ],
    distractor: [
      {
        source: "src/utils/string-utils.ts",
        content: `export function capitalize(s: string): string { return s.charAt(0).toUpperCase() + s.slice(1); }
export function slugify(s: string): string { return s.toLowerCase().replace(/\\s+/g, "-"); }
export function truncate(s: string, max: number): string { return s.length > max ? s.slice(0, max) + "..." : s; }`,
      },
      {
        source: "src/utils/math-utils.ts",
        content: `export function clamp(v: number, lo: number, hi: number): number { return Math.max(lo, Math.min(hi, v)); }
export function lerp(a: number, b: number, t: number): number { return a + (b - a) * t; }
export function roundTo(n: number, decimals: number): number { const m = 10 ** decimals; return Math.round(n * m) / m; }`,
      },
      {
        source: "src/services/logger.ts",
        content: `export enum LogLevel { DEBUG, INFO, WARN, ERROR }
export class Logger {
  private level: LogLevel;
  constructor(level: LogLevel = LogLevel.INFO) { this.level = level; }
  info(msg: string) { if (this.level <= LogLevel.INFO) console.log("[INFO]", msg); }
  error(msg: string) { if (this.level <= LogLevel.ERROR) console.error("[ERROR]", msg); }
}`,
      },
      {
        source: "src/config.ts",
        content: `export const APP_NAME = "TurboBench";
export const VERSION = "1.0.0";
export const MAX_RETRIES = 3;
export const TIMEOUT_MS = 5000;
export const DEFAULT_LOCALE = "en-US";`,
      },
    ],
  },

  /** What the system must learn */
  groundTruth: {
    requiredFiles: ["src/types/user-types.ts"],
    expectedLearning: "user-types.ts should have highest Thompson alpha and causal utility after 15+ trials",
  },

  trialsNeeded: 20,
};

// ═══════════════════════════════════════════════════════════════════════════
// Level 2: Two Required Files (Interaction)
// ═══════════════════════════════════════════════════════════════════════════
//
// Ground truth: api-types.ts AND http-client.ts are both required.
// api-types alone → can't make API calls
// http-client alone → doesn't know the types
// System must learn: BOTH files have high utility, neither alone suffices.
//
// Task: "Write a function `fetchUsers(): Promise<User[]>`"

const LEVEL_2 = {
  name: "Two Required Files (Interaction)",
  description: "6 files, 2 needed together. System must learn credit assignment across multiple dependencies.",

  task: {
    id: "bench-l2",
    description: "Write a TypeScript function 'fetchUsers(): Promise<User[]>' that fetches users from an API and returns them typed. Use the provided api types and http client. Export it.",
    type: "code_generation" as const,
    complexity: 0.45,
    qualityThreshold: 0.6,
    latencyBudgetMs: 60000,
  },

  files: {
    required: [
      {
        source: "src/api/api-types.ts",
        content: `export interface User { id: number; name: string; email: string; }
export interface ApiResponse<T> { data: T; status: number; error?: string; }
export type HttpMethod = "GET" | "POST" | "PUT" | "DELETE";
export interface RequestConfig { method: HttpMethod; body?: unknown; headers?: Record<string, string>; }`,
      },
      {
        source: "src/api/http-client.ts",
        content: `import type { ApiResponse, RequestConfig } from "./api-types";
export async function request<T>(url: string, config: RequestConfig): Promise<ApiResponse<T>> {
  const response = await fetch(url, {
    method: config.method,
    headers: { "Content-Type": "application/json", ...config.headers },
    body: config.body ? JSON.stringify(config.body) : undefined,
  });
  const data = await response.json();
  return { data: data as T, status: response.status };
}`,
      },
    ],
    distractor: [
      {
        source: "src/utils/date-utils.ts",
        content: `export function formatDate(d: Date): string { return d.toISOString().split("T")[0]; }
export function daysBetween(a: Date, b: Date): number { return Math.abs(b.getTime() - a.getTime()) / 86400000; }`,
      },
      {
        source: "src/utils/validation.ts",
        content: `export function isEmail(s: string): boolean { return /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/.test(s); }
export function isUrl(s: string): boolean { try { new URL(s); return true; } catch { return false; } }`,
      },
      {
        source: "src/services/cache.ts",
        content: `export class Cache<T> { private store = new Map<string, { value: T; expiry: number }>();
  set(key: string, value: T, ttlMs: number) { this.store.set(key, { value, expiry: Date.now() + ttlMs }); }
  get(key: string): T | null { const e = this.store.get(key); if (!e || e.expiry < Date.now()) { this.store.delete(key); return null; } return e.value; }
}`,
      },
      {
        source: "src/config.ts",
        content: `export const API_BASE_URL = "https://api.example.com";
export const DEFAULT_PAGE_SIZE = 20;
export const MAX_UPLOAD_SIZE_MB = 10;`,
      },
    ],
  },

  groundTruth: {
    requiredFiles: ["src/api/api-types.ts", "src/api/http-client.ts"],
    expectedLearning: "Both required files should have higher utility than distractors. Removing either should degrade quality.",
  },

  trialsNeeded: 30,
};

// ═══════════════════════════════════════════════════════════════════════════
// Level 3: File Ordering Matters
// ═══════════════════════════════════════════════════════════════════════════
//
// Ground truth: types.ts should be presented BEFORE implementation.ts
// The LLM generates better code when it sees types first.
// System must learn: compression.beta (recency weight) matters for ordering.
//
// Task: "Implement the interfaces defined in the types file"

const LEVEL_3 = {
  name: "File Ordering Matters",
  description: "4 files where presentation order affects output quality. System must learn recency weighting.",

  task: {
    id: "bench-l3",
    description: "Write a TypeScript class 'InMemoryStore<T>' that implements the Store<T> interface defined in the types file. Include all methods. Export it.",
    type: "code_generation" as const,
    complexity: 0.5,
    qualityThreshold: 0.6,
    latencyBudgetMs: 60000,
  },

  files: {
    required: [
      {
        source: "src/store/types.ts",
        content: `export interface Store<T> {
  get(id: string): T | undefined;
  set(id: string, value: T): void;
  delete(id: string): boolean;
  has(id: string): boolean;
  keys(): string[];
  size(): number;
  clear(): void;
}
export interface StoreEntry<T> { id: string; value: T; createdAt: Date; updatedAt: Date; }`,
      },
      {
        source: "src/store/base.ts",
        content: `import type { Store, StoreEntry } from "./types";
export abstract class BaseStore<T> implements Store<T> {
  abstract get(id: string): T | undefined;
  abstract set(id: string, value: T): void;
  abstract delete(id: string): boolean;
  has(id: string): boolean { return this.get(id) !== undefined; }
  abstract keys(): string[];
  size(): number { return this.keys().length; }
  clear(): void { for (const k of this.keys()) this.delete(k); }
}`,
      },
      {
        source: "src/store/utils.ts",
        content: `export function generateId(): string { return Math.random().toString(36).slice(2, 10); }
export function now(): Date { return new Date(); }`,
      },
    ],
    distractor: [
      {
        source: "src/services/auth.ts",
        content: `export interface AuthToken { token: string; expiresAt: number; }
export function parseToken(raw: string): AuthToken | null { try { const t = JSON.parse(atob(raw.split(".")[1])); return { token: raw, expiresAt: t.exp * 1000 }; } catch { return null; } }
export function isExpired(token: AuthToken): boolean { return Date.now() > token.expiresAt; }`,
      },
    ],
  },

  groundTruth: {
    requiredFiles: ["src/store/types.ts", "src/store/base.ts"],
    expectedLearning: "Higher beta (recency weight) should increase quality when types.ts is presented before base.ts. System should learn that beta ≈ 0.4-0.5 is optimal.",
  },

  trialsNeeded: 40,
  sweepParam: "compression.beta",
  sweepRange: [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50] as number[],
};

// ═══════════════════════════════════════════════════════════════════════════
// Level 4: Real Mini-Project
// ═══════════════════════════════════════════════════════════════════════════
//
// Ground truth: UNKNOWN. This is the "deep space" observation.
// A small but real codebase (Todo app). The system must learn which files
// matter for adding a new feature (due dates).
//
// This validates that learning from Levels 1-3 transfers to real tasks.

const LEVEL_4 = {
  name: "Real Mini-Project",
  description: "Add a feature to a small existing codebase. Ground truth unknown — real world conditions.",

  task: {
    id: "bench-l4",
    description: "Add a 'dueDate' field to the Todo type and implement 'getOverdue(): Todo[]' in the TodoService. Update all affected files.",
    type: "code_generation" as const,
    complexity: 0.55,
    qualityThreshold: 0.6,
    latencyBudgetMs: 90000,
  },

  files: {
    required: [
      {
        source: "src/todo/types.ts",
        content: `export interface Todo { id: string; title: string; completed: boolean; createdAt: Date; }
export type TodoFilter = "all" | "active" | "completed";
export interface TodoStats { total: number; active: number; completed: number; }`,
      },
      {
        source: "src/todo/service.ts",
        content: `import type { Todo, TodoFilter, TodoStats } from "./types";
export class TodoService {
  private todos: Todo[] = [];
  add(title: string): Todo { const t: Todo = { id: crypto.randomUUID(), title, completed: false, createdAt: new Date() }; this.todos.push(t); return t; }
  toggle(id: string): boolean { const t = this.todos.find(t => t.id === id); if (!t) return false; t.completed = !t.completed; return true; }
  remove(id: string): boolean { const i = this.todos.findIndex(t => t.id === id); if (i === -1) return false; this.todos.splice(i, 1); return true; }
  getAll(filter: TodoFilter = "all"): Todo[] { switch (filter) { case "active": return this.todos.filter(t => !t.completed); case "completed": return this.todos.filter(t => t.completed); default: return [...this.todos]; } }
  getStats(): TodoStats { const all = this.todos.length; const completed = this.todos.filter(t => t.completed).length; return { total: all, active: all - completed, completed }; }
}`,
      },
      {
        source: "src/todo/routes.ts",
        content: `import { TodoService } from "./service";
const svc = new TodoService();
export function handleAdd(title: string) { return svc.add(title); }
export function handleToggle(id: string) { return svc.toggle(id); }
export function handleRemove(id: string) { return svc.remove(id); }
export function handleList(filter?: "all" | "active" | "completed") { return svc.getAll(filter || "all"); }`,
      },
    ],
    distractor: [
      {
        source: "src/utils/formatters.ts",
        content: `export function formatBytes(bytes: number): string { const units = ["B","KB","MB","GB"]; let i = 0; let n = bytes; while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; } return n.toFixed(1) + " " + units[i]; }
export function pluralize(word: string, count: number): string { return count === 1 ? word : word + "s"; }`,
      },
      {
        source: "src/services/notifications.ts",
        content: `export type NotificationType = "email" | "push" | "sms";
export interface Notification { type: NotificationType; message: string; recipient: string; }
export function send(n: Notification): Promise<boolean> { console.log("[NOTIFY]", n.type, "→", n.recipient, ":", n.message); return Promise.resolve(true); }`,
      },
      { source: "src/config.ts", content: `export const APP_TITLE = "TurboTodo"; export const MAX_TODOS = 100; export const ENABLE_SYNC = false;` },
    ],
  },

  groundTruth: {
    requiredFiles: [], // UNKNOWN — this is the real test
    expectedLearning: "System should identify src/todo/types.ts and src/todo/service.ts as causally important after multiple trials.",
  },

  trialsNeeded: 50,
};

// ═══════════════════════════════════════════════════════════════════════════
// All levels
// ═══════════════════════════════════════════════════════════════════════════

export const BENCHMARKS = { level1: LEVEL_1, level2: LEVEL_2, level3: LEVEL_3, level4: LEVEL_4 } as const;
export type BenchmarkLevel = keyof typeof BENCHMARKS;

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark runner
// ═══════════════════════════════════════════════════════════════════════════

export interface TrialResult {
  trial: number;
  quality: number;
  compiled: boolean;
  cost: number;
  predictedQuality: number;
  surprise: number;
  // Per-file metrics (for diagnostics)
  thompsonAlphas: Record<string, number>;
  causalUtilities: Record<string, number>;
}

export interface BenchmarkResult {
  level: string;
  totalTrials: number;
  compileRate: number;
  avgQuality: number;
  firstHalfQuality: number;
  secondHalfQuality: number;
  qualityTrend: "improving" | "flat" | "declining";
  topFile: { source: string; thompsonAlpha: number; causalUtility: number };
  costTotal: number;
  trials: TrialResult[];
}

export function buildFragments(
  files: { required: Array<{ source: string; content: string }>; distractor: Array<{ source: string; content: string }> },
): ContextFragment[] {
  const all = [...files.required, ...files.distractor];
  return all.map((f) => ({
    source: f.source,
    content: f.content,
    type: "code" as const,
    timestamp: Date.now() - 86400000,
    length: f.content.length,
    contentType: (f.source.endsWith(".ts") ? "code" : "config") as "code" | "config",
  }));
}

/**
 * Run one benchmark level and return structured results.
 *
 * @param engine — fresh TurboContextEngine (state should be cleared before calling)
 * @param level — benchmark definition
 * @param workDir — temp directory with source files written
 * @param apiKey — DeepSeek API key
 */
export async function runBenchmark(
  engine: TurboContextEngine,
  level: typeof LEVEL_1 | typeof LEVEL_2 | typeof LEVEL_3 | typeof LEVEL_4,
  workDir: string,
): Promise<BenchmarkResult> {
  const fragments = buildFragments(level.files);
  const task: Task = { ...level.task, id: `${level.task.id}-${Date.now()}` };
  const allFiles = [...level.files.required, ...level.files.distractor];
  const nTrials = level.trialsNeeded;
  const trials: TrialResult[] = [];

  // Beta sweep for Level 3
  const betas = "sweepRange" in level ? (level as typeof LEVEL_3).sweepRange : [0.30];

  console.log(`\n══ ${level.name} (${nTrials} trials) ══`);
  console.log(`   Required: ${level.files.required.map((f) => f.source).join(", ")}`);
  console.log(`   Distractor: ${level.files.distractor.map((f) => f.source).join(", ")}`);
  console.log("");

  for (let i = 0; i < nTrials; i++) {
    const beta = betas[i % betas.length];
    // Access learner's internal config directly (getConfig() returns a copy)
    (engine as any).learner.config.beta = beta;

    const start = Date.now();
    let result;
    try {
      result = await engine.execute({ ...task, id: `${task.id}-${i}` }, fragments, { workingDir: workDir });
    } catch (err) {
      console.log(`  #${i + 1} ❌ crash: ${(err as Error).message.slice(0, 60)}`);
      continue;
    }

    // Hard-signal quality: compilation + structural check
    const genOutput = result.generations[result.generations.length - 1];
    const content = genOutput?.content || "";
    const hardQuality = measureHardQuality(content, level, workDir);

    const status = engine.rlEngineV5.getStatus();
    const cost = result.costEstimate.estimatedCostUSD;

    // Per-file diagnostics from V5 state
    const memories = engine.rlEngineV5 as any;
    const thompsonAlphas: Record<string, number> = {};
    const causalUtilities: Record<string, number> = {};

    // Collect per-file RL state
    for (const f of allFiles) {
      const boost = (engine.rlEngineV5 as any).getCausalBoost?.(f.source, task.type) || 0;
      causalUtilities[f.source] = boost;
    }

    trials.push({
      trial: i + 1,
      quality: hardQuality,
      compiled: hardQuality >= 0.5,
      cost,
      predictedQuality: status.globalBaseline || 0.5,
      surprise: status.surpriseGlobalMean || 0,
      thompsonAlphas,
      causalUtilities,
    });

    const bar = hardQuality >= 0.5 ? (hardQuality >= 0.8 ? "✅" : "⚠️") : "❌";
    console.log(
      `  #${String(i + 1).padStart(2)} β=${beta.toFixed(2)} ${bar} q=${hardQuality.toFixed(2)} ` +
      `$${cost.toFixed(4)} pred=${(status.globalBaseline || 0.5).toFixed(3)}`,
    );
  }

  // Analysis
  const compileRate = trials.filter((t) => t.compiled).length / trials.length;
  const avgQuality = trials.reduce((s, t) => s + t.quality, 0) / trials.length;
  const mid = Math.floor(trials.length / 2);
  const firstHalf = trials.slice(0, mid).reduce((s, t) => s + t.quality, 0) / mid;
  const secondHalf = trials.slice(mid).reduce((s, t) => s + t.quality, 0) / (trials.length - mid);
  const costTotal = trials.reduce((s, t) => s + t.cost, 0);

  // Find top file by causal utility
  let topFile = { source: "none", thompsonAlpha: 0, causalUtility: 0 };
  for (const f of allFiles) {
    const cu = causalUtilities[f.source] || 0;
    if (cu > topFile.causalUtility) {
      topFile = { source: f.source, thompsonAlpha: thompsonAlphas[f.source] || 0, causalUtility: cu };
    }
  }

  const trend = secondHalf > firstHalf + 0.05 ? "improving" : secondHalf < firstHalf - 0.05 ? "declining" : "flat";

  return {
    level: level.name,
    totalTrials: trials.length,
    compileRate,
    avgQuality,
    firstHalfQuality: firstHalf,
    secondHalfQuality: secondHalf,
    qualityTrend: trend,
    topFile,
    costTotal,
    trials,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Hard-signal quality measurement (per-level)
// ═══════════════════════════════════════════════════════════════════════════

function measureHardQuality(
  output: string,
  level: typeof LEVEL_1 | typeof LEVEL_2 | typeof LEVEL_3 | typeof LEVEL_4,
  workDir: string,
): number {
  const tsMatch = output.match(/```(?:typescript|ts)\n([\s\S]*?)```/);
  const code = tsMatch ? tsMatch[1] : output.replace(/```/g, "");

  // Write to temp
  const testFile = path.join(workDir, `output-${Date.now()}.ts`);
  fs.writeFileSync(testFile, code);

  // Level-specific structural checks
  const checks: Record<string, RegExp[]> = {
    "Single Required File": [/export\s+function\s+validateUser/, /User/, /Result/],
    "Two Required Files": [/export\s+function\s+fetchUsers/, /User/, /ApiResponse/, /request/],
    "File Ordering Matters": [/export\s+class\s+InMemoryStore/, /Store/, /get\(/, /set\(/],
    "Real Mini-Project": [/dueDate/, /getOverdue/, /Date/, /Todo/],
  };

  const levelChecks = checks[level.name] || [];
  const structuralPasses = levelChecks.filter((re) => re.test(code)).length;
  const structuralScore = levelChecks.length > 0 ? structuralPasses / levelChecks.length : 0.5;

  // Try compilation
  let compiled = false;
  try {
    const { execSync } = require("child_process");
    execSync(`npx tsc --noEmit ${testFile} 2>&1`, { cwd: workDir, timeout: 15000, stdio: "pipe" });
    compiled = true;
  } catch {
    compiled = false;
  }

  // Cleanup temp file
  try { fs.unlinkSync(testFile); } catch { /* ok */ }

  // Hard quality formula
  return compiled ? 0.5 + 0.5 * structuralScore : Math.min(0.4, 0.4 * structuralScore);
}

// ═══════════════════════════════════════════════════════════════════════════
// CLI entry point
// ═══════════════════════════════════════════════════════════════════════════

async function main() {
  const level = process.argv[2] || "1";

  if (!process.env.DEEPSEEK_API_KEY) {
    console.error("Set DEEPSEEK_API_KEY");
    process.exit(1);
  }

  const levels: Record<string, typeof LEVEL_1> = {
    "1": LEVEL_1, "2": LEVEL_2, "3": LEVEL_3, "4": LEVEL_4,
  };

  const bench = levels[level];
  if (!bench) {
    console.error(`Unknown level: ${level}. Use 1, 2, 3, or 4.`);
    process.exit(1);
  }

  // Setup working directory
  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), `tc-bench-l${level}-`));
  const allFiles = [...bench.files.required, ...bench.files.distractor];
  for (const f of allFiles) {
    const fp = path.join(workDir, f.source);
    fs.mkdirSync(path.dirname(fp), { recursive: true });
    fs.writeFileSync(fp, f.content);
  }
  fs.writeFileSync(path.join(workDir, "tsconfig.json"), JSON.stringify({
    compilerOptions: { target: "ES2022", module: "ESNext", moduleResolution: "node", strict: true, esModuleInterop: true, skipLibCheck: true },
  }));
  fs.writeFileSync(path.join(workDir, "package.json"), JSON.stringify({ name: "tc-bench", type: "module" }));

  console.log(`📁 ${workDir}`);

  // Fresh engine
  const engine = new TurboContextEngine({
    alpha: 0.55, beta: 0.20, gamma: 0.25,
    maxTokenBudget: 8000,
    temperatureSchedule: [0.7, 0.35, 0.1],
    learningRate: 0.01, historyWindow: 100,
    complexityThresholdLow: 0.30, complexityThresholdHigh: 0.65,
  });

  const result = await runBenchmark(engine, bench, workDir);

  // Print report
  console.log(`\n═══ ${bench.name} — Results ═══`);
  console.log(`  Trials:       ${result.totalTrials}`);
  console.log(`  Compile rate: ${(result.compileRate * 100).toFixed(0)}%`);
  console.log(`  Avg quality:  ${result.avgQuality.toFixed(3)}`);
  console.log(`  1st half:     ${result.firstHalfQuality.toFixed(3)}`);
  console.log(`  2nd half:     ${result.secondHalfQuality.toFixed(3)}`);
  console.log(`  Trend:        ${result.qualityTrend}`);
  console.log(`  Top file:     ${result.topFile.source} (causal=${result.topFile.causalUtility.toFixed(3)})`);
  console.log(`  Total cost:   $${result.costTotal.toFixed(4)}`);

  // Ground truth check
  if ("groundTruth" in bench && bench.groundTruth.requiredFiles.length > 0) {
    const gt = bench.groundTruth;
    const topMatch = gt.requiredFiles.includes(result.topFile.source);
    console.log(`\n  Ground truth check: ${topMatch ? "✅ IDENTIFIED CORRECTLY" : "❌ WRONG FILE"}`);
    console.log(`  Expected required: ${gt.requiredFiles.join(", ")}`);
    console.log(`  System picked:     ${result.topFile.source}`);
  }

  // Cleanup
  fs.rmSync(workDir, { recursive: true, force: true });
  engine.rlEngineV5.saveState();
}

main().catch((err) => { console.error("Fatal:", err); process.exit(1); });
