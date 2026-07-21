/**
 * Phase 1: Prove TurboContext can learn from real signals.
 *
 * Controlled experiment — 20 trials on code_generation, varying compression.beta.
 * Hypothesis: higher beta (more recency weight → more relevant context) improves
 * code generation quality measured by hard-signal compilation verification.
 *
 * Karpathy principle: "One metric, one variable, prove it learns."
 *
 * Usage:
 *   export DEEPSEEK_API_KEY=sk-...
 *   npx tsx scripts/phase1-learn.ts
 */
import { TurboContextEngine } from "../src/index.js";
import type { ContextFragment } from "../src/types.js";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

// ── Fixed task: generate a deterministic utility function ──
const TASK_DESC = "Write a TypeScript function 'chunkArray<T>(arr: T[], size: number): T[][]' that splits an array into chunks. Include input validation, generics, edge cases. Export it as a single self-contained .ts file.";

function makeTask(id: number) {
  return {
    id: `learn-${id}`,
    description: TASK_DESC,
    type: "code_generation" as const,
    complexity: 0.4,
    qualityThreshold: 0.6,
    latencyBudgetMs: 120000,
  };
}

// ── Context fragments ──
const FRAGMENTS: ContextFragment[] = [
  {
    source: "src/utils/array.ts", type: "code", timestamp: Date.now() - 86400000, length: 200, contentType: "code",
    content: `export function unique<T>(arr: T[]): T[] { return [...new Set(arr)]; }\nexport function flatten<T>(arr: T[][]): T[] { return arr.flat(); }`,
  },
  {
    source: "src/utils/string.ts", type: "code", timestamp: Date.now() - 86400000, length: 60, contentType: "code",
    content: `export function capitalize(s: string): string { return s.charAt(0).toUpperCase() + s.slice(1); }`,
  },
  {
    source: "tsconfig.json", type: "config", timestamp: Date.now() - 86400000, length: 200, contentType: "config",
    content: JSON.stringify({ compilerOptions: { target: "ES2022", module: "ESNext", moduleResolution: "node", strict: true, esModuleInterop: true, skipLibCheck: true } }),
  },
];

// ── Beta sweep values ──
const BETA_VALUES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60];

// ── Hard-signal quality measurement ──
function measureHardQuality(output: string, workDir: string): { compiled: boolean; smokeTestPassed: boolean; hardQuality: number } {
  // Extract TypeScript code block from LLM output
  const tsMatch = output.match(/```(?:typescript|ts)\n([\s\S]*?)```/);
  const code = tsMatch ? tsMatch[1] : output.replace(/```/g, "");

  // Write to temp file in workDir
  const testFile = path.join(workDir, "chunkArray.ts");
  fs.writeFileSync(testFile, code);

  // Check for expected function signature
  const hasExport = /export\s+function\s+chunkArray\s*</.test(code);
  const hasValidation = /throw|Error/.test(code);
  const hasGeneric = /<T>/.test(code);

  // Try to compile
  let compiled = false;
  try {
    const { execSync } = require("child_process");
    execSync(`npx tsc --noEmit ${testFile} 2>&1`, {
      cwd: workDir, timeout: 15000, stdio: "pipe",
    });
    compiled = true;
  } catch {
    compiled = false;
  }

  // Hard quality: compilation is gate, then structural checks
  const structuralScore = (hasExport ? 0.3 : 0) + (hasValidation ? 0.1 : 0) + (hasGeneric ? 0.1 : 0);
  const hardQuality = compiled ? 0.5 + structuralScore : Math.min(0.3, structuralScore);

  return { compiled, smokeTestPassed: compiled && hasExport, hardQuality };
}

// ── Main ──
async function main() {
  console.log("═".repeat(64));
  console.log("Phase 1: Proving TurboContext Can Learn");
  console.log("Variable: compression.beta | Metric: hard-signal quality");
  console.log("═".repeat(64));

  if (!process.env.DEEPSEEK_API_KEY) {
    console.error("❌ Set DEEPSEEK_API_KEY");
    process.exit(1);
  }

  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), "tc-learn-"));
  // Write context files for compiler
  for (const f of FRAGMENTS) {
    const fp = path.join(workDir, f.source);
    fs.mkdirSync(path.dirname(fp), { recursive: true });
    fs.writeFileSync(fp, f.content);
  }
  console.log(`📁 ${workDir}`);

  // ── Create engine with fresh state ──
  const engine = new TurboContextEngine({
    alpha: 0.55, beta: 0.20, gamma: 0.25,
    maxTokenBudget: 8000,
    temperatureSchedule: [0.7, 0.35, 0.1],
    learningRate: 0.01,
    historyWindow: 100,
    complexityThresholdLow: 0.30,
    complexityThresholdHigh: 0.65,
  });

  const results: Array<{
    run: number; beta: number; quality: number; compiled: boolean;
    cost: number; latencyMs: number; predictedQuality: number; surprise: number;
  }> = [];

  console.log("\n── Running 20 experiments ──\n");

  for (let i = 0; i < 20; i++) {
    const beta = BETA_VALUES[i % BETA_VALUES.length];
    const task = makeTask(i + 1);

    // Override beta for this run
    engine.config.beta = beta;

    const start = Date.now();
    let result;
    try {
      result = await engine.execute(task, FRAGMENTS, { workingDir: workDir });
    } catch (err) {
      console.log(`  #${i + 1} beta=${beta.toFixed(2)} ❌ crash: ${(err as Error).message.slice(0, 80)}`);
      continue;
    }

    const genOutput = result.generations[result.generations.length - 1];
    const { hardQuality, compiled } = measureHardQuality(
      genOutput?.content || "", workDir,
    );

    // Read V5 engine state for learning metrics
    const status = engine.rlEngineV5.getStatus();
    const cost = result.costEstimate.estimatedCostUSD;

    results.push({
      run: i + 1, beta, quality: hardQuality, compiled,
      cost, latencyMs: Date.now() - start,
      predictedQuality: status.surpriseGlobalMean > 0 ? status.globalBaseline : 0.5,
      surprise: status.surpriseGlobalMean,
    });

    const bar = compiled ? "✅" : "❌";
    console.log(
      `  #${String(i + 1).padStart(2)} beta=${beta.toFixed(2)} ` +
      `${bar} q=${hardQuality.toFixed(2)} ` +
      `$${cost.toFixed(4)} ${Date.now() - start}ms ` +
      `pred=${status.globalBaseline?.toFixed(3) || "N/A"} ` +
      `surprise=${status.surpriseGlobalMean?.toFixed(4) || "N/A"}`,
    );
  }

  // ── Analysis ──
  console.log("\n" + "═".repeat(64));
  console.log("Results Analysis");
  console.log("═".repeat(64));

  const compiled = results.filter(r => r.compiled);
  const goodQuality = results.filter(r => r.quality >= 0.7);

  console.log(`\n  Total: ${results.length} experiments, $${results.reduce((s, r) => s + r.cost, 0).toFixed(4)}`);
  console.log(`  Compiled: ${compiled.length}/${results.length} (${(compiled.length / results.length * 100).toFixed(0)}%)`);
  console.log(`  High quality (≥0.7): ${goodQuality.length}/${results.length}`);

  // Beta vs quality correlation
  console.log("\n  Beta → Quality:");
  const byBeta = new Map<number, number[]>();
  for (const r of results) {
    const arr = byBeta.get(r.beta) || [];
    arr.push(r.quality);
    byBeta.set(r.beta, arr);
  }
  const sorted = [...byBeta.entries()].sort((a, b) => a[0] - b[0]);
  for (const [beta, quals]) {
    const avg = quals.reduce((a, b) => a + b, 0) / quals.length;
    const bar = "█".repeat(Math.round(avg * 40));
    console.log(`    β=${beta.toFixed(2)}: ${bar} ${avg.toFixed(3)} (n=${quals.length})`);
  }

  // Learning trend: did quality improve in the second half?
  const firstHalf = results.slice(0, 10);
  const secondHalf = results.slice(10);
  const firstAvg = firstHalf.reduce((a, r) => a + r.quality, 0) / firstHalf.length;
  const secondAvg = secondHalf.reduce((a, r) => a + r.quality, 0) / secondHalf.length;
  console.log(`\n  First 10 avg quality:  ${firstAvg.toFixed(3)}`);
  console.log(`  Last 10 avg quality:   ${secondAvg.toFixed(3)}`);
  console.log(`  Δ:                     ${secondAvg > firstAvg ? "+" : ""}${(secondAvg - firstAvg).toFixed(3)}`);

  // V5 engine state
  const finalStatus = engine.rlEngineV5.getStatus();
  console.log(`\n  RL State:`);
  console.log(`    Trials recorded:  ${finalStatus.totalTrials}`);
  console.log(`    Active memories:  ${finalStatus.activeMemories}`);
  console.log(`    Global baseline:  ${finalStatus.globalBaseline?.toFixed(4) || "N/A"}`);
  console.log(`    Curriculum phase: ${finalStatus.curriculumPhase}`);

  // VERDICT
  console.log("\n" + "═".repeat(64));
  if (secondAvg > firstAvg && secondAvg > 0.5) {
    console.log("✅ SYSTEM LEARNS: quality improved in second half");
  } else if (compiled.length > 5) {
    console.log("⚠️  MIXED: compilation works but no clear learning trend");
  } else {
    console.log("❌ DOES NOT LEARN: needs architecture simplification");
  }
  console.log("═".repeat(64));

  // Cleanup
  fs.rmSync(workDir, { recursive: true, force: true });
  engine.rlEngineV5.saveState();
}

main().catch(err => { console.error("Fatal:", err); process.exit(1); });
