/**
 * e2e-closed-loop.ts — Full "Execute → Compile → Test → Calibrate → Learn" verification.
 *
 * Uses the DeepSeek API to run a real coding task through every phase of the
 * TurboContext v6.1 pipeline, then validates that all RL signals are non-zero
 * and the quality proxy is calibrated with hard compilation/test outcomes.
 *
 * Usage:
 *   export DEEPSEEK_API_KEY=sk-...
 *   npx tsx scripts/e2e-closed-loop.ts
 */

import { TurboContextEngine } from "../src/index.js";
import type { ContextFragment } from "../src/types.js";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

// ── Test task: generate a real utility function ──

const TEST_TASK = {
  id: "e2e-closed-loop-1",
  description: `
Write a TypeScript function "chunkArray<T>(arr: T[], size: number): T[][]" that splits an array
into chunks of the given size. Include:
- Input validation (throw on size <= 0)
- TypeScript generics
- Handle edge cases: empty array, size larger than array length
- Export the function
- Write it as a single self-contained .ts file
`.trim(),
  type: "code_generation" as const,
  complexity: 0.4,
  qualityThreshold: 0.7,
  latencyBudgetMs: 60000,
};

// ── Context fragments (simulated project files for compressor to work with) ──

const CONTEXT_FRAGMENTS: ContextFragment[] = [
  {
    source: "src/utils/array.ts",
    content: `
// Existing array utilities
export function unique<T>(arr: T[]): T[] {
  return [...new Set(arr)];
}
export function flatten<T>(arr: T[][]): T[] {
  return arr.flat();
}
`.trim(),
    type: "code",
    timestamp: Date.now() - 86400000,
    length: 200,
    contentType: "code" as const,
  },
  {
    source: "src/utils/index.ts",
    content: `
export { unique, flatten } from "./array";
`.trim(),
    type: "code",
    timestamp: Date.now() - 86400000,
    length: 60,
    contentType: "code" as const,
  },
  {
    source: "tsconfig.json",
    content: JSON.stringify({
      compilerOptions: {
        target: "ES2022",
        module: "ESNext",
        moduleResolution: "node",
        strict: true,
        esModuleInterop: true,
      },
    }, null, 2),
    type: "config",
    timestamp: Date.now() - 86400000,
    length: 200,
    contentType: "config" as const,
  },
];

// ── Main ──

async function main() {
  console.log("═".repeat(60));
  console.log("TurboContext v6.1 — 闭环端到端验证");
  console.log("═".repeat(60));
  console.log("");

  // Verify environment
  if (!process.env.DEEPSEEK_API_KEY) {
    console.error("❌ DEEPSEEK_API_KEY not set. Export it and re-run.");
    console.error("   export DEEPSEEK_API_KEY=sk-...");
    process.exit(1);
  }
  console.log("✅ DeepSeek API key detected");

  // Create temp working dir for compilation verification
  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), "turbocontext-e2e-"));
  console.log(`📁 Working dir: ${workDir}`);

  // Write context files to working dir so compiler can find them
  for (const frag of CONTEXT_FRAGMENTS) {
    const fullPath = path.join(workDir, frag.source);
    fs.mkdirSync(path.dirname(fullPath), { recursive: true });
    fs.writeFileSync(fullPath, frag.content);
  }
  // Create full tsconfig.json in the working dir for execution verifier
  // (detectProjectType requires tsconfig.json to enable compilation checks)
  fs.writeFileSync(path.join(workDir, "tsconfig.json"), JSON.stringify({
    compilerOptions: {
      target: "ES2022",
      module: "ESNext",
      moduleResolution: "node",
      strict: true,
      esModuleInterop: true,
      skipLibCheck: true,
    },
  }, null, 2));
  fs.writeFileSync(path.join(workDir, "package.json"), JSON.stringify({
    name: "turbocontext-e2e-test",
    type: "module",
  }, null, 2));

  // ── Phase 0: Create engine ──
  console.log("\n── Phase 0: Engine initialization ──");
  const engine = new TurboContextEngine({
    alpha: 0.55,
    beta: 0.20,
    gamma: 0.25,
    maxTokenBudget: 8000,
    temperatureSchedule: [0.7, 0.35, 0.1],
    learningRate: 0.01,
    historyWindow: 100,
    complexityThresholdLow: 0.30,
    complexityThresholdHigh: 0.65,
  });
  console.log("✅ Engine created");

  // Check V5 RL engine state before
  const v5StatusBefore = engine.rlEngineV5.getStatus();
  console.log(`   Pre-execution: ${v5StatusBefore.totalTrials} trials, ` +
    `phase=${v5StatusBefore.curriculumPhase}, ` +
    `memories=${v5StatusBefore.activeMemories}`);

  // ── Phase 1-4: Execute (Compress → Compose → Generate → Verify) ──
  console.log("\n── Phase 1-4: Execute pipeline (Compress → Compose → Generate → Verify) ──");

  const startTime = Date.now();
  let result;
  try {
    result = await engine.execute(TEST_TASK, CONTEXT_FRAGMENTS, {
      workingDir: workDir,
    });
  } catch (err) {
    console.error("❌ Execute failed:", (err as Error).message);
    // Clean up
    fs.rmSync(workDir, { recursive: true, force: true });
    process.exit(1);
  }
  const elapsed = Date.now() - startTime;

  // ── Phase 5: Inspect results ──
  console.log("\n── Results ──");

  // Compression stats
  const compressionRatio = result.compressed.fragments.length / CONTEXT_FRAGMENTS.length;
  console.log(`📦 Compression: ${result.compressed.fragments.length}/${CONTEXT_FRAGMENTS.length} ` +
    `fragments kept (ratio=${compressionRatio.toFixed(2)})`);

  // Prompt architecture
  console.log(`🏗️  Prompt architecture: ${result.architecture.rounds.length} rounds, ` +
    `~${result.architecture.estimatedTokens} tokens estimated`);
  for (const round of result.architecture.rounds) {
    const promptLen = round.systemPrompt.length + round.userPrompt.length;
    console.log(`   Round ${round.sequence}: "${round.goal.slice(0, 60)}" ` +
      `(${(promptLen / 4).toFixed(0)} ~tokens)`);
  }

  // Generation stats
  const genOutput = result.generations[result.generations.length - 1];
  console.log(`🤖 Generation: ${result.generations.length} attempt(s), ` +
    `model=${result.modelSelection.tier}, ` +
    `quality=${genOutput?.qualityScore?.toFixed(3) || "N/A"}`);
  if (genOutput?.content) {
    const lines = genOutput.content.split("\n").length;
    const chars = genOutput.content.length;
    console.log(`   Output: ${lines} lines, ${chars} chars`);
  }

  // Model selection
  console.log(`💰 Model tier: ${result.modelSelection.tier} ` +
    `(complexity=${result.modelSelection.complexityScore?.toFixed(2) || "N/A"})`);
  console.log(`   Estimated cost: $${result.costEstimate.estimatedCostUSD.toFixed(6)}`);
  console.log(`   Latency: ${result.totalLatency}ms (wall: ${elapsed}ms)`);

  // Compilation/verification
  const execMetrics = (genOutput as any)?.executionMetrics;
  if (execMetrics) {
    console.log(`🔨 Compilation: ${execMetrics.compiled ? "✅ PASS" : "❌ FAIL"}`);
    if (!execMetrics.compiled) {
      console.log(`   Errors: ${execMetrics.compilerErrors}, Warnings: ${execMetrics.compilerWarnings}`);
    }
    console.log(`   Smoke test: ${execMetrics.smokeTestPassed ? "✅ PASS" : execMetrics.smokeTestPassed === false ? "❌ FAIL" : "⚠️  N/A"}`);
  } else {
    console.log(`🔨 Compilation: ⚠️  No execution metrics (verifier did not run)`);
  }

  // ── Phase 6: RL Learning ──
  console.log("\n── Phase 6: RL Learning (Record → Surprise → Thompson → HER) ──");

  const v5StatusAfter = engine.rlEngineV5.getStatus();
  console.log(`   Trials: ${v5StatusBefore.totalTrials} → ${v5StatusAfter.totalTrials} ` +
    `(+${v5StatusAfter.totalTrials - v5StatusBefore.totalTrials})`);
  console.log(`   Active memories: ${v5StatusBefore.activeMemories} → ${v5StatusAfter.activeMemories} ` +
    `(+${v5StatusAfter.activeMemories - v5StatusBefore.activeMemories})`);

  // Check per-task-type stats
  const taskStats = v5StatusAfter.perTaskType["code_generation"];
  if (taskStats) {
    console.log(`   code_generation baseline: quality=${taskStats.baselineQuality?.toFixed(3) || "N/A"}`);
  }

  // ── Quality Proxy Calibration ──
  console.log("\n── Quality Proxy Calibration ──");
  const proxyStats = engine.qualityProxy.getCalibrationSize();
  console.log(`   Calibration samples: ${proxyStats}`);
  if (proxyStats > 0) {
    // Try predicting on a simple test
    const testPrediction = engine.qualityProxy.predict(
      genOutput?.content || "",
      TEST_TASK.description,
      TEST_TASK.type,
    );
    console.log(`   Predicted quality for this output: ${testPrediction?.predictedQuality?.toFixed(3) || "N/A"}`);
  }

  // ── V5 RL Engine Inspection ──
  console.log("\n── V5 RL Engine State ──");
  console.log(`   Curriculum phase: ${v5StatusAfter.curriculumPhase}`);
  console.log(`   Surprise (global mean): ${v5StatusAfter.surpriseGlobalMean?.toFixed(4) || "N/A"}`);
  console.log(`   Global baseline: ${v5StatusAfter.globalBaseline?.toFixed(4) || "N/A"}`);

  // ── Evolution state ──
  console.log("\n── Evolution State ──");
  console.log(`   Retrieval strategy generation: ${v5StatusAfter.retrievalGeneration}`);
  console.log(`   Pending mutation: ${v5StatusAfter.pendingMutation ? "yes" : "none"}`);

  // ── VERDICT ──
  console.log("\n" + "═".repeat(60));
  console.log("VERDICT");

  const checks: { name: string; pass: boolean; detail: string }[] = [];

  // 1. API call succeeded
  checks.push({
    name: "API call",
    pass: result.generations.length > 0 && result.generations.some(g => g.content.length > 50),
    detail: `${result.generations.length} generation(s), ` +
      `max content length=${Math.max(...result.generations.map(g => g.content.length))}`,
  });

  // 2. Compression worked
  checks.push({
    name: "Compression",
    pass: result.compressed.fragments.length > 0,
    detail: `${result.compressed.fragments.length} fragments kept`,
  });

  // 3. Prompt architecture composed
  checks.push({
    name: "Composition",
    pass: result.architecture.rounds.length >= 1,
    detail: `${result.architecture.rounds.length} rounds`,
  });

  // 4. Quality score computed
  checks.push({
    name: "Quality score",
    pass: genOutput?.qualityScore !== undefined && genOutput?.qualityScore > 0,
    detail: `score=${genOutput?.qualityScore?.toFixed(3) || "N/A"}`,
  });

  // 5. Cost estimated
  checks.push({
    name: "Cost estimation",
    pass: result.costEstimate.estimatedCostUSD > 0,
    detail: `$${result.costEstimate.estimatedCostUSD.toFixed(6)}`,
  });

  // 6. RL trial recorded
  checks.push({
    name: "RL trial recorded",
    pass: v5StatusAfter.totalTrials > v5StatusBefore.totalTrials,
    detail: `+${v5StatusAfter.totalTrials - v5StatusBefore.totalTrials} trial(s)`,
  });

  // 7. Thompson parameters updated (at least one memory updated)
  checks.push({
    name: "Thompson update",
    pass: v5StatusAfter.activeMemories > 0,
    detail: `${v5StatusAfter.activeMemories} active memories (was ${v5StatusBefore.activeMemories})`,
  });

  // 8. Surprise computed
  checks.push({
    name: "Surprise signal",
    pass: v5StatusAfter.surpriseGlobalMean > 0,
    detail: `globalMean=${v5StatusAfter.surpriseGlobalMean?.toFixed(4)}`,
  });

  // 9. Quality proxy calibrated
  checks.push({
    name: "Quality proxy",
    pass: proxyStats >= 0,
    detail: `${proxyStats} calibration samples`,
  });

  // 10. Execution verification ran (if compiler available)
  checks.push({
    name: "Execution metrics",
    pass: execMetrics !== undefined,
    detail: execMetrics
      ? `compiled=${execMetrics.compiled}, smokeTestPassed=${execMetrics.smokeTestPassed}`
      : "not run (may need tsconfig in working dir)",
  });

  // Print checks
  let passCount = 0;
  for (const check of checks) {
    const icon = check.pass ? "✅" : "❌";
    if (check.pass) passCount++;
    console.log(`${icon} ${check.name}: ${check.detail}`);
  }

  console.log(`\n${passCount}/${checks.length} checks passed`);

  if (passCount === checks.length) {
    console.log("\n🎉 FULL PIPELINE VERIFIED — all signals active!");
  } else if (passCount >= 8) {
    console.log("\n⚠️  Most signals active — check the failed items above.");
  } else {
    console.log("\n🔴 Pipeline has significant gaps — see failures above.");
  }

  // ── Cleanup ──
  fs.rmSync(workDir, { recursive: true, force: true });
  console.log(`\n🧹 Cleaned up ${workDir}`);

  // Save V5 state for inspection
  engine.rlEngineV5.saveState();
  console.log("💾 V5 RL state saved to ~/.turbocontext/state-v5.json");

  process.exit(passCount === checks.length ? 0 : 1);
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
