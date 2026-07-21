// TurboContext: CLI Interface
import { TurboContextEngine } from "./index.js";
import type { Task, ContextFragment, TaskType, Mission } from "./types.js";
import { readFileSync, existsSync, readdirSync, statSync } from "fs";
import { Learner } from "./core/learner.js";
import path from "path";

async function main() {
  const args = process.argv.slice(2);
  const command = args[0] || "help";

  if (command === "run") { await runCommand(args.slice(1)); }
  else if (command === "demo") { await demoCommand(); }
  else if (command === "experiment") { await experimentCommand(args.slice(1)); }
  else if (command === "ablate") { await ablateCommand(args.slice(1)); }
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
  console.log("=== TurboContext Engine v2.0 ===");
  console.log("");

  const fragments = collectContext(dir);
  console.log("Context: " + fragments.length + " fragments (" +
    fragments.reduce((s, f) => s + f.length, 0) + " chars)");

  const turboTask: Task = {
    id: "task_" + Date.now(),
    description: task,
    type,
    qualityThreshold: threshold,
  };

  console.log("Running TurboContext pipeline...");
  console.log("LLM: " + (useLLM ? "Deepseek API" : "Simulated (use --llm or set DEEPSEEK_API_KEY)"));
  const engine = new TurboContextEngine({
    qualityThreshold: threshold,
    ...(useLLM ? {} : { llm: undefined }),
  });
  // workingDir for execution verification: use cwd (project root with tsconfig.json),
  // not --dir (which specifies source collection path).
  const workingDir = process.cwd();
  const result = await engine.execute(turboTask, fragments, { workingDir });

  console.log("");
  printResult(result);
}

async function demoCommand() {
  console.log("");
  console.log("=== TurboContext Algorithm Demo ===");
  console.log("");

  const demoContext: ContextFragment[] = [
    {
      id: "1", source: "src/auth/login.ts", contentType: "source",
      content: [
        'export async function login(email: string, password: string) {',
        '  const user = await db.users.findByEmail(email);',
        '  if (!user) throw new AuthError("User not found");',
        '  const valid = await bcrypt.compare(password, user.passwordHash);',
        '  if (!valid) throw new AuthError("Invalid password");',
        '  const token = generateJWT({ userId: user.id, role: user.role });',
        '  return { user: sanitizeUser(user), token };',
        '}'
      ].join("\n"),
      lastModified: Date.now() - 86400000, length: 320,
    },
    {
      id: "2", source: "src/auth/register.ts", contentType: "source",
      content: [
        'export async function register(data: RegisterInput) {',
        '  const existing = await db.users.findByEmail(data.email);',
        '  if (existing) throw new AuthError("Email already registered");',
        '  const hash = await bcrypt.hash(data.password, 12);',
        '  const user = await db.users.create({ ...data, passwordHash: hash });',
        '  return { user: sanitizeUser(user) };',
        '}'
      ].join("\n"),
      lastModified: Date.now() - 172800000, length: 310,
    },
    {
      id: "3", source: "src/auth/middleware.ts", contentType: "source",
      content: [
        'export async function authMiddleware(req, res, next) {',
        '  const token = req.headers.authorization?.replace("Bearer ", "");',
        '  if (!token) throw new AuthError("No token provided");',
        '  const payload = verifyJWT(token);',
        '  req.user = payload;',
        '  next();',
        '}'
      ].join("\n"),
      lastModified: Date.now() - 259200000, length: 250,
    },
    {
      id: "4", source: "src/utils/jwt.ts", contentType: "source",
      content: [
        'import jwt from "jsonwebtoken";',
        'const SECRET = process.env.JWT_SECRET || "dev-secret";',
        'export function generateJWT(payload: object): string {',
        '  return jwt.sign(payload, SECRET, { expiresIn: "24h" });',
        '}',
        'export function verifyJWT(token: string): object {',
        '  return jwt.verify(token, SECRET) as object;',
        '}'
      ].join("\n"),
      lastModified: Date.now() - 86400000, length: 260,
    },
    {
      id: "5", source: "README.md", contentType: "docs",
      content: "# Auth Service\n\nHandles user authentication with JWT tokens.\nSupports login, register, and token refresh.",
      lastModified: Date.now() - 604800000, length: 120,
    },
  ];

  const tasks: Array<{ label: string; task: Task }> = [
    {
      label: "Code Review",
      task: { id: "demo1", description: "Review auth module for security issues and code quality", type: "code_review" },
    },
    {
      label: "Code Generation",
      task: { id: "demo2", description: "Add forgot password feature with reset token and email sending", type: "code_generation" },
    },
  ];

  const useLLM = !!process.env.DEEPSEEK_API_KEY;
  const engine = new TurboContextEngine();
  if (useLLM) {
    console.log("(Deepseek API detected, using real LLM for generation)\n");
  }

  for (const { label, task } of tasks) {
    console.log("--- " + label + " ---");
    console.log("Task: " + task.description);
    console.log("");

    const result = await engine.execute(task, demoContext);

    console.log("Phase 1 - Context Compression:");
    console.log("  Original: " + result.compressed.originalTokens + " tokens");
    console.log("  Compressed: " + result.compressed.compressedTokens + " tokens");
    console.log("  Ratio: " + (result.compressed.compressionRatio * 100).toFixed(1) + "%");
    console.log("  Fragments: " + result.compressed.fragments.length + "/" + demoContext.length);
    console.log("");

    console.log("Phase 2 - Prompt Architecture:");
    console.log("  Rounds: " + result.architecture.rounds.length);
    for (const round of result.architecture.rounds) {
      console.log("  Round " + round.sequence + ": " + round.goal);
    }
    console.log("  Est. tokens: " + result.architecture.estimatedTokens);
    console.log("");

    console.log("Phase 4 - Cost Optimization:");
    console.log("  Model: " + result.modelSelection.tier + " (" + result.modelSelection.config.model + ")");
    console.log("  Rationale: " + result.modelSelection.rationale);
    console.log("  Est. cost: $" + result.costEstimate.estimatedCostUSD);
    console.log("");

    console.log("Phase 3 - Quality-Weighted Generation:");
    for (const gen of result.generations) {
      const bar = "#".repeat(Math.round(gen.qualityScore * 10)) +
                  "-".repeat(Math.round((1 - gen.qualityScore) * 10));
      console.log("  Attempt " + gen.attempt + ": [" + bar + "] " + (gen.qualityScore * 100).toFixed(1) + "%");
    }
    console.log("  Final quality: " + (result.finalQuality * 100).toFixed(1) + "%");
    console.log("  Attempts: " + result.totalAttempts);
    console.log("");
  }

  console.log("Phase 5 - Learning System:");
  const trend = engine.getLearner().getQualityTrend();
  console.log("  Quality trend: " + trend.trend + " (avg " + (trend.average * 100).toFixed(1) + "%)");
  const config = engine.getLearner().getConfig();
  console.log("  Weights: a=" + config.alpha.toFixed(3) + ", b=" + config.beta.toFixed(3) + ", g=" + config.gamma.toFixed(3));
  console.log("  Thresholds: t1=" + config.complexityThresholdLow.toFixed(2) + ", t2=" + config.complexityThresholdHigh.toFixed(2));
  console.log("  Temps: [" + config.temperatureSchedule.map(t => t.toFixed(2)).join(", ") + "]");
  console.log("  Quality threshold: " + (config.qualityThreshold * 100).toFixed(0) + "%");
  console.log("");
}

function showFormula() {
  console.log("");
  console.log("=== TurboContext: Complete Mathematical Formula ===");
  console.log("");
  console.log("Phase 1: Context Compression");
  console.log("  score(c_i) = a * sim(c_i, T) + b * recency(c_i) + g * specificity(c_i)");
  console.log("  recency(c_i) = 1 / (1 + days_since_modified)");
  console.log("  specificity(c_i) = 1 - len(c_i) / MAX_LEN");
  console.log("  where a + b + g = 1 (default: a=0.55, b=0.20, g=0.25)");
  console.log("");
  console.log("  Optimization constraint:");
  console.log("    max sum_i sum_j score(c_i) * cover(c_i, r_j)");
  console.log("    s.t. sum_i tokens(c_i) <= budget");
  console.log("         for all r_j: sum_i cover(c_i, r_j) >= 1");
  console.log("");
  console.log("Phase 2: Prompt Composition");
  console.log("  T -> S = {s1, s2, ..., sm}  (task decomposition)");
  console.log("  P = {prompt(s1), prompt(s2|o1), ...}");
  console.log("");
  console.log("Phase 3: Quality-Weighted Generation");
  console.log("  for k = 1..K:");
  console.log("    t_k = temperature_schedule[k]");
  console.log("    o_k ~ LLM(P_k, t_k)");
  console.log("    Q(o_k) = sum_i w_i * q_i(o_k)");
  console.log("    if Q(o_k) >= theta: return o_k");
  console.log("    else: inject feedback and retry");
  console.log("");
  console.log("Phase 4: Cost Optimization");
  console.log("  complexity = 0.40 * type + 0.15 * ambiguity + 0.20 * historical + 0.25 * base");
  console.log("  model = fast (if complexity < t1), medium (if t1 <= complexity < t2), deep (if >= t2)");
  console.log("");
  console.log("Phase 5: Continuous Learning");
  console.log("  Update a, b, g based on compression-quality correlation");
  console.log("  Update t1, t2 based on model pass rates");
  console.log("  Update temperature based on avg attempt count");
  console.log("");
}

async function experimentCommand(args: string[]) {
  const maxN = parseInt(getArg(args, "--max") || getArg(args, "-n") || "10");
  const useLLM = args.includes("--llm") || !!process.env.DEEPSEEK_API_KEY;
  const missionPath = getArg(args, "--mission") || getArg(args, "-m");

  console.log("");
  console.log("=== TurboContext Autonomous Experiment Loop ===");
  console.log(`Budget: ${maxN} experiments`);
  console.log(`LLM: ${useLLM ? "Deepseek API" : "Simulated"}`);

  // Load mission if available
  const mission = missionPath ? Learner.loadMission(missionPath) : Learner.loadMission();
  if (mission) {
    console.log(`Mission: ${mission.goal}`);
    if (mission.humanNotes) {
      console.log(`Notes: ${mission.humanNotes.slice(0, 120)}...`);
    }
  }

  const engine = new TurboContextEngine({
    qualityThreshold: 0.85,
    maxTokenBudget: mission?.tokenBudgetPerRun ?? 8000,
  });

  const runs = await engine.runExperiments({
    maxExperiments: maxN,
    tokenBudgetPerRun: mission?.tokenBudgetPerRun ?? 8000,
    timeBudgetPerRun: mission?.timeBudgetPerRun ?? 300,
    mission: mission ?? undefined,
    onProgress: (run, summary) => {
      console.log(`  ${summary}`);
    },
  });

  const kept = runs.filter(r => r.decision === "keep").length;
  const best = runs
    .filter(r => r.decision === "keep")
    .sort((a, b) => b.deltaPercent - a.deltaPercent)[0];

  console.log("");
  console.log(`=== Complete: ${runs.length} experiments ===`);
  console.log(`Kept: ${kept}/${runs.length} (${(kept/runs.length*100).toFixed(0)}%)`);
  if (best) {
    console.log(`Best: #${best.runNumber} ${best.mutation?.type} → +${best.deltaPercent.toFixed(2)}%`);
  }
  console.log(`Results saved to ~/.turbocontext/results.tsv`);
}

async function ablateCommand(args: string[]) {
  const taskDesc = getArg(args, "--task") || getArg(args, "-t") || "Review auth module for security";
  const dir = getArg(args, "--dir") || getArg(args, "-d") || ".";
  const type = (getArg(args, "--type") || "code_generation") as TaskType;
  console.log(`Task: "${taskDesc}" (${type})`);
  console.log(`Source: ${path.resolve(dir)}`);

  const fragments = collectContext(dir);
  if (fragments.length === 0) {
    console.error("No source files found in " + dir);
    process.exit(1);
  }
  console.log(`Context: ${fragments.length} fragments (${fragments.reduce((s, f) => s + f.length, 0)} chars)`);

  const engine = new TurboContextEngine({ qualityThreshold: 0.85 });

  const task: Task = {
    id: `ablate_${Date.now()}`,
    description: taskDesc,
    type: type as Task["type"],
    qualityThreshold: 0.85,
  };

  console.log("Running ablation (runs pipeline twice — with and without target file)...");
  const result = await engine.ablate(task, fragments, { workingDir: process.cwd() });

  if (!result) {
    console.log("Ablation skipped — no suitable target file found.");
    console.log("(Files need ≥3 previous executions to build uncertainty estimates.)");
  }
}

function showHelp() {
  console.log("");
  console.log("TurboContext v3.6 - Adaptive Context Optimization Algorithm");
  console.log("  (with per-file ablation for clean causal signals)");
  console.log("");
  console.log("Usage:");
  console.log("  turbocontext run         Run a single task");
  console.log("  turbocontext demo        Run demo with 2 tasks");
  console.log("  turbocontext experiment  Run autonomous experiment loop (autoresearch mode)");
  console.log("  turbocontext ablate      Run per-file ablation (causal counterfactual)");
  console.log("  turbocontext formula     Show formulas");
  console.log("  turbocontext help        Show help");
  console.log("");
  console.log("Experiment options:");
  console.log("  --max, -n      Max experiments (default: 10)");
  console.log("  --mission, -m  Path to mission.md");
  console.log("  --llm          Use real Deepseek API");
  console.log("");
  console.log("Ablation options (v3.6):");
  console.log("  --task, -t     Task description (required)");
  console.log("  --dir, -d      Source directory (default: current dir)");
  console.log("  --type         Task type (default: code_generation)");
  console.log("");
  console.log("Run options:");
  console.log("  --task, -t     Task description (required)");
  console.log("  --dir, -d      Source directory (default: current dir)");
  console.log("  --type         Task type");
  console.log("  --threshold    Quality threshold 0-1 (default: 0.85)");
  console.log("  --llm          Use real Deepseek API");
  console.log("");
  console.log("Examples:");
  console.log("  turbocontext experiment --max 20 --llm");
  console.log("  turbocontext ablate --task \"Review auth\" --dir ./src --type code_review");
  console.log("  turbocontext demo");
  console.log("  turbocontext run --task \"Review src/auth\" --dir ./src --type code_review");
  console.log("");
}

function getArg(args: string[], name: string): string | undefined {
  const idx = args.indexOf(name);
  return (idx >= 0 && idx < args.length - 1) ? args[idx + 1] : undefined;
}

function collectContext(dir: string): ContextFragment[] {
  const fragments: ContextFragment[] = [];
  const absDir = path.resolve(dir);
  if (!existsSync(absDir)) return fragments;

  function walk(d: string) {
    try {
      for (const entry of readdirSync(d)) {
        const fp = path.join(d, entry);
        const s = statSync(fp);
        if (s.isDirectory() && !entry.startsWith(".") && entry !== "node_modules") walk(fp);
        else if (s.isFile() && isSupported(entry)) {
          try {
            const c = readFileSync(fp, "utf-8");
            fragments.push({
              id: path.relative(absDir, fp),
              source: path.relative(absDir, fp),
              contentType: classify(entry),
              content: c,
              lastModified: s.mtimeMs,
              length: c.length,
            });
          } catch {}
        }
      }
    } catch {}
  }
  walk(absDir);
  return fragments;
}

function isSupported(name: string): boolean {
  const ext = path.extname(name).toLowerCase();
  return [".ts", ".tsx", ".js", ".jsx", ".py", ".go", ".rs", ".java",
          ".rb", ".swift", ".kt", ".json", ".yaml", ".yml", ".toml",
          ".md", ".sql", ".graphql", ".sh"].includes(ext);
}

function classify(name: string): "source" | "config" | "docs" | "test" | "other" {
  const ext = path.extname(name).toLowerCase();
  const base = name.toLowerCase();
  if (base.includes("test") || base.includes("spec")) return "test";
  if ([".json", ".yaml", ".yml", ".toml"].includes(ext)) return "config";
  if (ext === ".md") return "docs";
  return "source";
}

function printResult(result: any) {
  console.log("Done. Executions: #" + result.executionCount);
  console.log("Final quality: " + (result.finalQuality * 100).toFixed(1) + "%");
  console.log("Attempts: " + result.totalAttempts);
  console.log("Latency: " + result.totalLatency + "ms");
  console.log("Trend: " + result.qualityTrend.trend + " (avg " + (result.qualityTrend.average * 100).toFixed(1) + "%)");

  // v3.5: Show execution verification metrics if available
  const lastGen = result.generations?.[result.generations.length - 1];
  if (lastGen?.executionMetrics) {
    const em = lastGen.executionMetrics;
    console.log("");
    console.log("=== Execution Verification ===");
    console.log(`  Project: ${em.projectType}`);
    console.log(`  Compiled: ${em.compiled ? "✓ yes" : "✗ no"}`);
    if (em.compilerExitCode !== null && em.compilerExitCode !== undefined) {
      console.log(`  Compiler exit: ${em.compilerExitCode}`);
    }
    if (em.compilerErrors > 0) console.log(`  Errors: ${em.compilerErrors}`);
    if (em.compilerWarnings > 0) console.log(`  Warnings: ${em.compilerWarnings}`);
  }
}

main().catch(console.error);
