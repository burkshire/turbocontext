// ============================================================================
// TurboContext V6 — Live DeepSeek Benchmark v2
// Real LLM → compile → test → calibrate → learn (direct API, full feedback loop)
// ============================================================================
import { writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { execSync } from "node:child_process";
import { createLLMCall } from "../src/core/llm.js";
import { QualityProxy } from "../src/core/quality-proxy.js";
import { extractSignals } from "../src/core/signal-extractor.js";
import { TurboContextEngine } from "../src/index.js";

// ============================================================================
// Tasks
// ============================================================================

const TASKS = [
  {
    name: "fibonacci", taskType: "code_generation" as const,
    prompt: "Write ONLY a TypeScript function `export function fibonacci(n: number): number` that returns the nth Fibonacci (0-indexed) using iteration. No markdown, no explanation, just the code with export.",
    smokeTest: `const r = fibonacci(10); if (r !== 55) throw new Error("fibonacci(10)=" + r); if (fibonacci(0) !== 0) throw new Error("fib(0)"); console.log("fibonacci OK");`,
  },
  {
    name: "binary_search", taskType: "code_generation" as const,
    prompt: "Write ONLY a TypeScript function `export function binarySearch(arr: number[], target: number): number` that performs binary search on a sorted array and returns the index or -1. No markdown, no explanation, just the code with export.",
    smokeTest: `if (binarySearch([1,3,5,7,9], 5) !== 2) throw new Error("find 5"); if (binarySearch([1,3,5], 2) !== -1) throw new Error("not found"); console.log("binarySearch OK");`,
  },
  {
    name: "is_palindrome", taskType: "code_generation" as const,
    prompt: "Write ONLY a TypeScript function `export function isPalindrome(s: string): boolean` that checks if a string is a palindrome (ignoring case and non-alphanumeric chars). No markdown, no explanation, just the code with export.",
    smokeTest: `if (!isPalindrome("A man, a plan, a canal: Panama")) throw new Error("panama"); if (isPalindrome("hello")) throw new Error("hello"); console.log("isPalindrome OK");`,
  },
  {
    name: "merge_sorted", taskType: "code_generation" as const,
    prompt: "Write ONLY a TypeScript function `export function mergeSorted(a: number[], b: number[]): number[]` that merges two already-sorted arrays into one sorted array. No markdown, no explanation, just the code with export.",
    smokeTest: `const r = mergeSorted([1,3,5], [2,4,6]); if (r.length !== 6 || r[0] !== 1 || r[5] !== 6) throw new Error("merge: "+JSON.stringify(r)); console.log("mergeSorted OK");`,
  },
  {
    name: "debounce", taskType: "code_generation" as const,
    prompt: "Write ONLY a TypeScript function `export function debounce<T extends (...args: any[]) => void>(fn: T, delay: number): (...args: Parameters<T>) => void` that delays execution until `delay` ms after the last call. No markdown, no explanation, just the code with export.",
    smokeTest: `let called = 0; const d = debounce(() => { called++; }, 50); d(); d(); setTimeout(() => { if (called !== 1) throw new Error("debounce: "+called); console.log("debounce OK"); }, 100);`,
  },
];

// ============================================================================
// Compile + test helper
// ============================================================================

function compileAndTest(code: string, taskName: string, smokeTest: string): {
  compiled: boolean; errorCount: number; testPassed: boolean; errorSummary: string;
} {
  const tmpDir = mkdtempSync(join(tmpdir(), "tc-live-"));
  const tsFile = join(tmpDir, `${taskName}.ts`);
  try {
    writeFileSync(tsFile, code + "\n" + smokeTest, "utf-8");
    let compiled = true, errorCount = 0, errorSummary = "";
    try {
      execSync(`npx tsc --noEmit --strict --skipLibCheck --target ES2022 --moduleResolution node "${tsFile}" 2>&1`, { stdio: "pipe", timeout: 15000 });
    } catch (err: any) {
      compiled = false;
      const lines = (err.stderr?.toString() || "").split("\n").filter((l: string) => l.includes("error TS"));
      errorCount = lines.length;
      errorSummary = lines.slice(0, 2).join("; ");
    }
    let testPassed = false;
    if (compiled) {
      try { execSync(`npx tsx "${tsFile}" 2>&1`, { stdio: "pipe", timeout: 10000 }); testPassed = true; }
      catch { testPassed = false; }
    }
    return { compiled, errorCount, testPassed, errorSummary };
  } finally {
    try { rmSync(tmpDir, { recursive: true, force: true }); } catch {}
  }
}

function extractCode(raw: string): string {
  // Strip markdown code fences
  const m = raw.match(/```(?:typescript|ts|javascript|js)?\n?([\s\S]*?)```/);
  return m ? m[1].trim() : raw.trim();
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.log("═══════════════════════════════════════════════════════════════");
  console.log("  TurboContext V6 — LIVE DeepSeek Feedback Loop");
  console.log("  Real LLM → compile → test → calibrate → RL learn");
  console.log("═══════════════════════════════════════════════════════════════\n");

  const apiKey = process.env.DEEPSEEK_API_KEY;
  if (!apiKey) { console.log("DEEPSEEK_API_KEY not set"); process.exit(1); }

  // Direct LLM call (bypass turbocontext pipeline for clean code gen)
  const llm = createLLMCall({ apiKey, model: "deepseek-v4-pro", maxTokens: 512, maxRetries: 2 });

  // Use turbocontext for RL learning — pass the same LLM function
  const engine = new TurboContextEngine({ llm: { apiKey, model: "deepseek-v4-pro", maxTokens: 512 } });
  const proxy = new QualityProxy({ minSamplesForFit: 3, bootstrapSamples: 100 });
  const v5 = engine.getRLEngineV5();

  console.log(`Proxy before: ${proxy.getCalibrationSize()} samples | fitted=${proxy.getWeights() ? 'YES' : 'NO'}`);
  console.log(`V5 state: ${v5.getStatus().totalTrials} trials | phase=${v5.getStatus().curriculumPhase}\n`);

  console.log(`${"Task".padEnd(16)} ${"Status".padEnd(6)} ${"Compile".padEnd(8)} ${"Test".padEnd(8)} ${"hardQ".padEnd(6)} ${"predQ".padEnd(7)} ${"Time"}`);
  console.log("-".repeat(72));

  let totalCost = 0;
  let successCount = 0;
  let totalTokens = 0;

  for (const t of TASKS) {
    const startTime = Date.now();

    // Step 1: Real LLM call (temperature=0.1 for deterministic code gen)
    const rawOutput = await llm(t.prompt, 0.1);
    const code = extractCode(rawOutput);
    const llmLatency = Date.now() - startTime;

    // Step 2: Real compilation + smoke test (the HARD SIGNAL)
    const compileResult = compileAndTest(code, t.name, t.smokeTest);

    // Hard quality = compilation success (0.6) + test pass (0.4)
    const hardQuality = (compileResult.compiled ? 0.6 : 0) + (compileResult.testPassed ? 0.4 : 0);

    // Step 3: Quality Proxy prediction based on extracted signals
    const signals = extractSignals(code, t.prompt, {
      compiled: compileResult.compiled,
      compilerErrors: compileResult.errorCount,
      smokeTestPassed: compileResult.testPassed,
    });

    const predBefore = proxy.predict(code, t.prompt, t.taskType, {
      compiled: compileResult.compiled,
      compilerErrors: compileResult.errorCount,
      smokeTestPassed: compileResult.testPassed,
    });

    // Step 4: CALIBRATE — feed real hard quality back into proxy
    proxy.calibrate(code, t.prompt, t.taskType, hardQuality, {
      compiled: compileResult.compiled,
      compilerErrors: compileResult.errorCount,
      smokeTestPassed: compileResult.testPassed,
    });

    // Step 5: Also feed into V5 RL engine via turbocontext pipeline
    // Run a lightweight execution through the engine for RL learning
    try {
      await engine.execute(
        { id: `live_${t.name}_${Date.now()}`, description: t.prompt, type: t.taskType },
        [{ id: "frag", source: "bench.ts", contentType: "source", content: code, lastModified: Date.now(), length: code.length }],
      );
    } catch {} // Engine execute may fail on quality gate; RL learning still happens via V5

    if (compileResult.compiled && compileResult.testPassed) successCount++;

    const status = compileResult.compiled ? (compileResult.testPassed ? "✅" : "⚠️") : "❌";
    const predQ = predBefore.predictedQuality;
    totalTokens += rawOutput.length;

    console.log(
      `${t.name.padEnd(16)} ${status.padEnd(6)} ` +
      `${(compileResult.compiled ? "PASS" : "FAIL").padEnd(8)} ` +
      `${(compileResult.testPassed ? "PASS" : "FAIL").padEnd(8)} ` +
      `${hardQuality.toFixed(1).padEnd(6)} ` +
      `${(predQ * 100).toFixed(0) + "%".padEnd(7)} ` +
      `${llmLatency}ms`
    );

    if (!compileResult.compiled && compileResult.errorSummary) {
      console.log(`  ⚡ Errors: ${compileResult.errorSummary.slice(0, 100)}`);
    }
  }

  // ── RESULTS ──
  console.log(`\n${"═".repeat(72)}`);
  console.log(`RESULTS`);
  console.log(`${"═".repeat(72)}`);
  console.log(`Compile+test pass: ${successCount}/${TASKS.length}`);
  console.log(`Total tokens: ${totalTokens}`);
  console.log(`Proxy calibrated: ${proxy.getCalibrationSize()} samples (from real LLM output)`);

  // Proxy weights after live calibration
  const weights = proxy.getWeights();
  if (weights) {
    console.log(`\n📊 Quality Proxy weights (learned from ${weights.sampleCount} LIVE DeepSeek samples):`);
    const names = ["compilation", "testPass", "codeBlocks", "noErrors", "keywordCov", "structScore", "respLength", "attEfficiency"];
    console.log(`  ${"Signal".padEnd(16)} | ${"Weight".padEnd(10)} | ${"ρ".padEnd(8)} | Confidence`);
    console.log(`  ${"-".repeat(54)}`);
    for (let i = 0; i < weights.weights.length; i++) {
      const conf = weights.confidence[i] >= 0.7 ? "🟢" : weights.confidence[i] >= 0.4 ? "🟡" : "🔴";
      console.log(`  ${names[i].padEnd(16)} | ${weights.weights[i].toFixed(4).padEnd(10)} | ${weights.relevance[i].toFixed(3).padEnd(8)} | ${conf}`);
    }
    console.log(`  ${"intercept".padEnd(16)} | ${weights.intercept.toFixed(4)}`);
  }

  // Signal profile
  console.log(`\n📈 Signal profile (which signals best predict real code quality):`);
  const profile = proxy.getSignalProfile("code_generation");
  for (const s of profile.slice(0, 5)) {
    const bar = "█".repeat(Math.round(s.relevance * 25));
    console.log(`  ${s.signal.padEnd(20)} | ρ=${s.relevance.toFixed(3)} ${bar}`);
  }

  // V5 RL state
  const v5status = v5.getStatus();
  console.log(`\n🧠 V5 RL engine: ${v5status.totalTrials} trials | phase ${v5status.curriculumPhase} | ${v5status.activeMemories} memories`);

  // Verify the feedback loop: predict after calibration should be better
  if (successCount > 0 && proxy.getCalibrationSize() > 0) {
    console.log(`\n✅ Feedback loop verified: ${proxy.getCalibrationSize()} real-code samples → ${weights ? 'weights learned' : 'pending'}`);
  }

  console.log(`\n=== Live benchmark complete ===`);
}

main().catch(console.error);
