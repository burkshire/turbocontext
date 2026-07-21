// TurboContext: CLI Interface (Plan B — simplified)
import { TurboContextEngine } from "./index.js";
import type { Task, ContextFragment, TaskType } from "./types.js";
import { readFileSync, existsSync, readdirSync, statSync } from "fs";
import path from "path";

async function main() {
  const args = process.argv.slice(2);
  const command = args[0] || "help";

  if (command === "run") { await runCommand(args.slice(1)); }
  else if (command === "demo") { await demoCommand(); }
  else if (command === "formula") { showFormula(); }
  else { showHelp(); }
}

async function runCommand(args: string[]) {
  const task = getArg(args, "--task") || getArg(args, "-t");
  const dir = getArg(args, "--dir") || getArg(args, "-d") || ".";
  const type = (getArg(args, "--type") || "general") as TaskType;
  const threshold = parseFloat(getArg(args, "--threshold") || "0.85");
  const useLLM = args.includes("--llm") || !!process.env.DEEPSEEK_API_KEY;
  if (!task) { console.error("Error: --task is required"); process.exit(1); }

  console.log("");
  console.log("=== TurboContext Engine ===");
  console.log("");

  const fragments = collectContext(dir);
  console.log(`Context: ${fragments.length} fragments (${fragments.reduce((s, f) => s + f.length, 0)} chars)`);

  const turboTask: Task = {
    id: "task_" + Date.now(),
    description: task,
    type,
    qualityThreshold: threshold,
  };

  console.log("Running TurboContext pipeline...");
  console.log("LLM: " + (useLLM ? "Deepseek API" : "Simulated (use --llm or set DEEPSEEK_API_KEY)"));
  const engine = new TurboContextEngine({ qualityThreshold: threshold });
  const result = await engine.execute(turboTask, fragments);

  console.log("");
  printResult(result);
}

async function demoCommand() {
  console.log("");
  console.log("=== TurboContext Algorithm Demo ===");
  console.log("");

  const engine = new TurboContextEngine();

  // Demo task 1: code review on synthetic auth module
  const fragments: ContextFragment[] = [
    { id: "1", source: "src/auth/login.ts", contentType: "source",
      content: `export async function login(email: string, password: string) {
  const user = await db.users.findByEmail(email);
  if (!user) throw new AuthError("User not found");
  const valid = await bcrypt.compare(password, user.passwordHash);
  if (!valid) throw new AuthError("Invalid password");
  const token = generateJWT({ userId: user.id, role: user.role });
  return { user: sanitizeUser(user), token };
}`,
      lastModified: Date.now() - 86400000, length: 320 },
    { id: "2", source: "src/auth/register.ts", contentType: "source",
      content: `export async function register(data: RegisterInput) {
  const existing = await db.users.findByEmail(data.email);
  if (existing) throw new AuthError("Email already registered");
  const hash = await bcrypt.hash(data.password, 12);
  const user = await db.users.create({ ...data, passwordHash: hash });
  return { user: sanitizeUser(user) };
}`,
      lastModified: Date.now() - 172800000, length: 280 },
    { id: "3", source: "src/auth/middleware.ts", contentType: "source",
      content: `export function authMiddleware(req: Request) {
  const token = req.headers.authorization?.split(" ")[1];
  if (!token) throw new AuthError("Missing token");
  const payload = verifyJWT(token);
  req.user = payload;
}`,
      lastModified: Date.now() - 259200000, length: 200 },
  ];

  const task1: Task = {
    id: "demo_1",
    description: "Review the auth module for security issues and code quality",
    type: "code_review",
    qualityThreshold: 0.85,
  };

  console.log("Task 1: Review auth module for security");
  const result1 = await engine.execute(task1, fragments);
  console.log(`  Quality: ${(result1.finalQuality * 100).toFixed(0)}% | Attempts: ${result1.totalAttempts} | Cost: $${result1.costEstimate.estimatedCostUSD.toFixed(4)}`);
  console.log(`  Coverage: ${Object.entries(result1.coverage).map(([k, v]) => `${k}=${(v * 100).toFixed(0)}%`).join(", ")}`);

  console.log("");

  const task2: Task = {
    id: "demo_2",
    description: "Add rate limiting to the login endpoint",
    type: "code_generation",
    qualityThreshold: 0.85,
  };

  console.log("Task 2: Add rate limiting to login endpoint");
  const result2 = await engine.execute(task2, fragments);
  console.log(`  Quality: ${(result2.finalQuality * 100).toFixed(0)}% | Attempts: ${result2.totalAttempts} | Cost: $${result2.costEstimate.estimatedCostUSD.toFixed(4)}`);

  console.log("");
  console.log("=== Demo Complete ===");
  console.log("Try: npx tsx src/cli.ts run --task 'Your task' --dir ./src --type code_review");
}

function showFormula() {
  console.log("");
  console.log("=== TurboContext Core Formulas ===");
  console.log("");
  console.log("Phase 1 — Context Scoring:");
  console.log("  score(cᵢ) = α·sim(cᵢ,T) + β·recency(cᵢ) + γ·specificity(cᵢ)");
  console.log("  Defaults: α=0.55, β=0.20, γ=0.25");
  console.log("");
  console.log("Phase 2 — Prompt Architecture:");
  console.log("  3-round decomposition: understand → execute → verify");
  console.log("");
  console.log("Phase 3 — Quality-Gated Generation:");
  console.log("  Temperature annealing: [0.7, 0.35, 0.1]");
  console.log("  4-dim quality: completeness, correctness, consistency, format");
  console.log("  Q(o) ≥ 0.85 → accept, else inject feedback + retry");
  console.log("");
  console.log("Phase 4 — Cost Optimization:");
  console.log("  complexity < θ₁ → fast, θ₁ ≤ complexity < θ₂ → medium, ≥ θ₂ → deep");
  console.log("  Default: θ₁=0.30, θ₂=0.50");
  console.log("");
  console.log("Session Memory — Cross-Session Recall:");
  console.log("  IDF-weighted keyword similarity + recency decay + outcome bonus");
  console.log("  Data: ~/.turbocontext/sessions.json");
}

function showHelp() {
  console.log("");
  console.log("TurboContext — Adaptive Context Optimization + Session Memory");
  console.log("");
  console.log("Usage:");
  console.log("  turbocontext run         Run a single task");
  console.log("  turbocontext demo        Run demo with 2 tasks");
  console.log("  turbocontext formula     Show formulas");
  console.log("  turbocontext help        Show help");
  console.log("");
  console.log("Run options:");
  console.log("  --task, -t     Task description (required)");
  console.log("  --dir, -d      Source directory (default: current dir)");
  console.log("  --type         Task type (default: general)");
  console.log("  --threshold    Quality threshold 0-1 (default: 0.85)");
  console.log("  --llm          Use real Deepseek API");
  console.log("");
  console.log("Examples:");
  console.log("  turbocontext demo");
  console.log("  turbocontext run --task 'Review src/auth' --dir ./src --type code_review");
}

// ── Helpers ──

function getArg(args: string[], flag: string): string | undefined {
  const idx = args.indexOf(flag);
  if (idx >= 0 && idx + 1 < args.length) return args[idx + 1];
  return undefined;
}

function collectContext(dir: string): ContextFragment[] {
  const fragments: ContextFragment[] = [];
  if (!existsSync(dir)) return fragments;

  const files = walkDir(dir, [".ts", ".tsx", ".js", ".jsx", ".py", ".md", ".json"]);
  for (const file of files) {
    try {
      const content = readFileSync(file, "utf-8");
      if (content.length === 0) continue;
      const stat = statSync(file);
      const ext = path.extname(file).slice(1);
      fragments.push({
        id: `ctx_${fragments.length}`,
        source: file,
        content,
        contentType: ext === "md" ? "docs" : ext === "json" ? "config" : "source",
        lastModified: stat.mtimeMs,
        length: content.length,
      });
    } catch { /* skip unreadable files */ }
  }
  return fragments;
}

function walkDir(dir: string, extensions: string[]): string[] {
  const results: string[] = [];
  try {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (entry.name === "node_modules" || entry.name === ".git" || entry.name === "dist") continue;
        results.push(...walkDir(full, extensions));
      } else if (entry.isFile() && extensions.some(ext => entry.name.endsWith(ext))) {
        if (entry.name.endsWith(".test.ts") || entry.name.endsWith(".test.tsx")) continue;
        results.push(full);
      }
    }
  } catch { /* skip unreadable dirs */ }
  return results;
}

function printResult(result: Awaited<ReturnType<TurboContextEngine["execute"]>>) {
  console.log("Done. Executions: #" + result.totalAttempts);
  console.log(`Final quality: ${(result.finalQuality * 100).toFixed(1)}%`);
  console.log(`Attempts: ${result.totalAttempts}`);
  console.log(`Latency: ${result.totalLatency}ms`);
  console.log(`Coverage: ${Object.entries(result.coverage).map(([k, v]) => `${k}=${(v * 100).toFixed(0)}%`).join(", ")}`);
  console.log(`Model: ${result.costEstimate.tier} ($${result.costEstimate.estimatedCostUSD.toFixed(4)})`);
}

main().catch(console.error);
