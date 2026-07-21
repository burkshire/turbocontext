// ============================================================================
// TurboContext V6 — Real-Compilation Benchmark (v2)
// ============================================================================
//
// Uses correct TypeScript reference implementations + buggy variants.
// Real tsc compilation → real hard quality → calibrate Quality Proxy.
//
// This tests the proxy's ability to distinguish good code from bad code
// using REAL compilation as ground truth — no LLM API needed.
// ============================================================================

import { writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { execSync } from "node:child_process";
import { QualityProxy } from "../src/core/quality-proxy.js";
import { extractSignals } from "../src/core/signal-extractor.js";

// ============================================================================
// Reference implementations (known-correct TypeScript)
// ============================================================================

const REFERENCE_IMPLS: Record<string, string> = {
  fibonacci: `
export function fibonacci(n: number): number {
  if (n <= 1) return n;
  let a = 0, b = 1;
  for (let i = 2; i <= n; i++) {
    const temp = a + b;
    a = b;
    b = temp;
  }
  return b;
}
`,
  is_palindrome: `
export function isPalindrome(s: string): boolean {
  const cleaned = s.toLowerCase().replace(/[^a-z0-9]/g, "");
  let left = 0, right = cleaned.length - 1;
  while (left < right) {
    if (cleaned[left] !== cleaned[right]) return false;
    left++;
    right--;
  }
  return true;
}
`,
  binary_search: `
export function binarySearch(arr: number[], target: number): number {
  let left = 0, right = arr.length - 1;
  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    if (arr[mid] === target) return mid;
    if (arr[mid] < target) left = mid + 1;
    else right = mid - 1;
  }
  return -1;
}
`,
  merge_sort: `
export function mergeSort(arr: number[]): number[] {
  if (arr.length <= 1) return arr;
  const mid = Math.floor(arr.length / 2);
  const left = mergeSort(arr.slice(0, mid));
  const right = mergeSort(arr.slice(mid));
  const result: number[] = [];
  let i = 0, j = 0;
  while (i < left.length && j < right.length) {
    if (left[i] < right[j]) result.push(left[i++]);
    else result.push(right[j++]);
  }
  return result.concat(left.slice(i)).concat(right.slice(j));
}
`,
  lru_cache: `
export class LRUCache<K, V> {
  private capacity: number;
  private map = new Map<K, V>();
  constructor(capacity: number) { this.capacity = capacity; }
  get(key: K): V | undefined {
    if (!this.map.has(key)) return undefined;
    const val = this.map.get(key)!;
    this.map.delete(key);
    this.map.set(key, val);
    return val;
  }
  put(key: K, value: V): void {
    if (this.map.has(key)) this.map.delete(key);
    else if (this.map.size >= this.capacity) {
      const first = this.map.keys().next().value;
      if (first !== undefined) this.map.delete(first);
    }
    this.map.set(key, value);
  }
}
`,
};

// ============================================================================
// Bug generator — creates realistic code defects
// ============================================================================

type BugType = "correct" | "syntax_error" | "logic_bug" | "missing_return" | "type_error" | "infinite_loop";

function generateVariant(original: string, bugType: BugType): { content: string; description: string } {
  switch (bugType) {
    case "correct":
      return { content: original, description: "Correct implementation" };

    case "syntax_error": {
      // Remove random closing braces/brackets
      const broken = original
        .replace(/return\s+\w+;/, "return ")           // missing return value
        .replace(/\)\s*{/, ") {");                       // keep structure
      return { content: broken.replace(/}/g, (_, i) => i % 3 === 0 ? "" : "}"), description: "Syntax error: missing closing braces" };
    }

    case "logic_bug": {
      // Introduce off-by-one or wrong operator
      const buggy = original
        .replace(/<=/g, "<")        // off-by-one
        .replace(/left < right/g, "left <= right")  // wrong comparison
        .replace(/===/g, "==");     // loose equality
      return { content: "// TODO: this might be wrong — I'm not sure about edge cases\n// FIXME: review logic\n" + buggy, description: "Logic bug + TODO/FIXME comments" };
    }

    case "missing_return": {
      // Remove a return statement
      return { content: original.replace(/return [^;]+;/, "// return value goes here\n  throw new Error('Not implemented');"), description: "Missing return: throws instead" };
    }

    case "type_error": {
      // Change types to be wrong
      const typed = original
        .replace(/: number/g, ": string")
        .replace(/: boolean/g, ": number")
        .replace(/: string\b(?!\[])/g, ": number");
      return { content: "// Sorry, I'm not sure about the types here — assuming they work\n// This is a rough draft, needs review\n" + typed, description: "Type errors + uncertain language" };
    }

    case "infinite_loop": {
      // Make while(true) or remove loop increment
      return { content: original.replace(/left\+\+;/g, "// left++;").replace(/right--;/g, "// right--;"), description: "Infinite loop: increment removed" };
    }

    default:
      return { content: original, description: "Unknown" };
  }
}

// ============================================================================
// Compilation + Smoke Test
// ============================================================================

interface CompileResult {
  compiled: boolean;
  errorCount: number;
  testPassed: boolean;
  errorSummary: string;
}

const SMOKE_TESTS: Record<string, string> = {
  fibonacci: `const r = fibonacci(10); if (r !== 55) throw new Error("Expected 55, got " + r); console.log("fibonacci(10)=" + r);`,
  is_palindrome: `if (!isPalindrome("racecar")) throw new Error("racecar should be true"); if (isPalindrome("hello")) throw new Error("hello should be false"); console.log("isPalindrome OK");`,
  binary_search: `const r = binarySearch([1,3,5,7,9], 5); if (r !== 2) throw new Error("Expected index 2, got " + r); console.log("binarySearch OK");`,
  merge_sort: `const r = mergeSort([3,1,4,1,5,9,2,6]); if (r[0] !== 1 || r[7] !== 9) throw new Error("Sort failed"); console.log("mergeSort OK");`,
  lru_cache: `const c = new LRUCache<string, number>(2); c.put("a",1); c.put("b",2); if (c.get("a") !== 1) throw new Error("LRU get failed"); c.put("c",3); if (c.get("b") !== undefined) throw new Error("LRU evict failed"); console.log("LRU OK");`,
};

function compileAndTest(source: string, taskName: string): CompileResult {
  const tmpDir = mkdtempSync(join(tmpdir(), "tc-bench-"));
  const tsFile = join(tmpDir, `${taskName}.ts`);

  try {
    const smokeTest = SMOKE_TESTS[taskName] || "";
    const wrapped = source + "\n" + smokeTest;
    writeFileSync(tsFile, wrapped, "utf-8");

    let compiled = true;
    let errorCount = 0;
    let errorSummary = "";

    try {
      execSync(`npx tsc --noEmit --strict --skipLibCheck --target ES2022 --moduleResolution node "${tsFile}" 2>&1`, {
        stdio: "pipe",
        timeout: 15000,
      });
    } catch (err: any) {
      compiled = false;
      const stderr = err.stderr?.toString() || err.message || "";
      const lines = stderr.split("\n").filter((l: string) => l.includes("error TS"));
      errorCount = lines.length;
      errorSummary = lines.slice(0, 2).join("; ");
    }

    let testPassed = false;
    if (compiled) {
      try {
        execSync(`npx tsx "${tsFile}" 2>&1`, { stdio: "pipe", timeout: 10000 });
        testPassed = true;
      } catch { testPassed = false; }
    }

    return { compiled, errorCount, testPassed, errorSummary };
  } finally {
    try { rmSync(tmpDir, { recursive: true, force: true }); } catch {}
  }
}

// ============================================================================
// Main
// ============================================================================

function main() {
  console.log("=== TurboContext V6 — Real-Compilation Benchmark ===\n");

  const BUG_TYPES: BugType[] = ["correct", "syntax_error", "logic_bug", "missing_return", "type_error", "infinite_loop"];
  const taskNames = Object.keys(REFERENCE_IMPLS);

  console.log(`Tasks: ${taskNames.length} × ${BUG_TYPES.length} variants = ${taskNames.length * BUG_TYPES.length} samples`);
  console.log(`Ground truth: real tsc compilation + smoke test\n`);

  const proxy = new QualityProxy({ minSamplesForFit: 5, bootstrapSamples: 100 });

  // Build evaluation matrix
  interface EvalRow {
    task: string;
    bugType: string;
    compiled: boolean;
    testPassed: boolean;
    hardQuality: number;
    predictedQuality: number;
    isReliable: boolean;
  }
  const rows: EvalRow[] = [];

  const startTime = Date.now();
  let totalCompiled = 0;
  let totalTestsPassed = 0;
  let totalSamples = 0;

  for (const taskName of taskNames) {
    const original = REFERENCE_IMPLS[taskName];

    for (const bugType of BUG_TYPES) {
      const { content, description } = generateVariant(original, bugType);
      const compileResult = compileAndTest(content, taskName);

      // Hard quality: compilation (0.6) + smoke test (0.4)
      const hardQuality = (compileResult.compiled ? 0.6 : 0) + (compileResult.testPassed ? 0.4 : 0);

      // Extract signals
      const signals = extractSignals(content, `Implement ${taskName}`, {
        compiled: compileResult.compiled,
        compilerErrors: compileResult.errorCount,
        smokeTestPassed: compileResult.testPassed,
      });

      // Predict before calibration
      const pred = proxy.predict(content, `Implement ${taskName}`, "code_generation", {
        compiled: compileResult.compiled,
        compilerErrors: compileResult.errorCount,
        smokeTestPassed: compileResult.testPassed,
      });

      // Calibrate
      proxy.calibrate(content, `Implement ${taskName}`, "code_generation", hardQuality, {
        compiled: compileResult.compiled,
        compilerErrors: compileResult.errorCount,
        smokeTestPassed: compileResult.testPassed,
      });

      rows.push({
        task: taskName,
        bugType,
        compiled: compileResult.compiled,
        testPassed: compileResult.testPassed,
        hardQuality,
        predictedQuality: pred.predictedQuality,
        isReliable: pred.isReliable,
      });

      if (compileResult.compiled) totalCompiled++;
      if (compileResult.testPassed) totalTestsPassed++;
      totalSamples++;

      const status = compileResult.compiled ? (compileResult.testPassed ? "✅" : "⚠️") : "❌";
      console.log(`  ${taskName.padEnd(16)} ${bugType.padEnd(16)} ${status} hardQ=${hardQuality.toFixed(1)} predQ=${(pred.predictedQuality * 100).toFixed(0)}%`);
    }
  }

  const elapsed = Date.now() - startTime;

  // ── Summary ──
  console.log(`\n${"═".repeat(70)}`);
  console.log(`RESULTS`);
  console.log(`${"═".repeat(70)}`);
  console.log(`Samples:       ${totalSamples}`);
  console.log(`Compiled:      ${totalCompiled}/${totalSamples} (${(totalCompiled / totalSamples * 100).toFixed(0)}%)`);
  console.log(`Tests passed:  ${totalTestsPassed}/${totalSamples} (${(totalTestsPassed / totalSamples * 100).toFixed(0)}%)`);
  console.log(`Time:          ${elapsed}ms`);

  // Per-bug-type breakdown
  console.log(`\nPer bug type:`);
  console.log(`  ${"Bug Type".padEnd(16)} | ${"Compiled".padEnd(10)} | ${"Tests OK".padEnd(10)} | ${"Avg hardQ"}`);
  console.log(`  ${"-".repeat(50)}`);
  for (const bt of BUG_TYPES) {
    const btRows = rows.filter(r => r.bugType === bt);
    const comp = btRows.filter(r => r.compiled).length;
    const test = btRows.filter(r => r.testPassed).length;
    const avgQ = btRows.reduce((s, r) => s + r.hardQuality, 0) / btRows.length;
    console.log(`  ${bt.padEnd(16)} | ${String(comp).padEnd(10)} | ${String(test).padEnd(10)} | ${(avgQ * 100).toFixed(0)}%`);
  }

  // Proxy weights
  const weights = proxy.getWeights();
  if (weights) {
    console.log(`\nLearned Quality Proxy weights (${weights.sampleCount} real-compilation samples):`);
    const names = ["compilation", "testPass", "codeBlocks", "noErrors", "keywordCov", "structScore", "respLength", "attEfficiency"];
    console.log(`  ${"Signal".padEnd(16)} | ${"Weight".padEnd(10)} | ${"Relevance ρ"}`);
    console.log(`  ${"-".repeat(42)}`);
    for (let i = 0; i < weights.weights.length; i++) {
      console.log(`  ${names[i].padEnd(16)} | ${weights.weights[i].toFixed(4).padEnd(10)} | ${weights.relevance[i].toFixed(3)}`);
    }
    console.log(`  ${"intercept".padEnd(16)} | ${weights.intercept.toFixed(4)}`);
  }

  // Test proxy discrimination
  const correctRows = rows.filter(r => r.bugType === "correct");
  const brokenRows = rows.filter(r => r.bugType === "syntax_error");
  if (correctRows.length > 0 && brokenRows.length > 0) {
    console.log(`\nDiscrimination test:`);
    for (const taskName of taskNames) {
      const good = rows.find(r => r.task === taskName && r.bugType === "correct");
      const bad = rows.find(r => r.task === taskName && r.bugType === "syntax_error");
      if (good && bad) {
        const ok = good.predictedQuality > bad.predictedQuality;
        console.log(`  ${taskName.padEnd(16)} correct=${(good.predictedQuality * 100).toFixed(0)}% broken=${(bad.predictedQuality * 100).toFixed(0)}% ${ok ? '✅' : '❌'}`);
      }
    }
  }

  console.log(`\n=== Benchmark complete ===`);
}

main();
