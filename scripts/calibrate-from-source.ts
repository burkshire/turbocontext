// ============================================================================
// Calibrate Quality Proxy from turbocontext's own TypeScript source
// ============================================================================
import { readdirSync, statSync } from "node:fs";
import { join, extname } from "node:path";
import { generateCalibrationBatch } from "../src/core/calibration-generator.js";
import { QualityProxy } from "../src/core/quality-proxy.js";

function walkDir(dir: string, exts: Set<string>, maxFiles: number): string[] {
  const results: string[] = [];
  function walk(d: string) {
    if (results.length >= maxFiles) return;
    let entries: string[];
    try { entries = readdirSync(d); } catch { return; }
    for (const entry of entries) {
      if (results.length >= maxFiles) return;
      const fullPath = join(d, entry);
      let stat;
      try { stat = statSync(fullPath); } catch { continue; }
      if (stat.isDirectory() && !entry.startsWith(".") && entry !== "node_modules" && entry !== "dist") {
        walk(fullPath);
      } else if (stat.isFile() && exts.has(extname(entry))) {
        results.push(fullPath);
      }
    }
  }
  walk(dir);
  return results;
}

async function main() {
  console.log("=== TurboContext V6 — Real-Code Calibration ===\n");

  const tsFiles = walkDir("src", new Set([".ts"]), 100);
  const srcFiles = tsFiles.filter(f => !f.includes("__tests__") && !f.includes(".test.ts"));
  console.log(`Found ${srcFiles.length} non-test TypeScript files\n`);

  console.log("Generating calibration variants (compile-checking each)...");
  const startTime = Date.now();
  const batch = generateCalibrationBatch(srcFiles, { maxFiles: 50 });
  const elapsed = Date.now() - startTime;

  console.log(`Generated ${batch.totalVariants} variants from ${batch.sourceFiles} files in ${elapsed}ms`);
  console.log(`Quality distribution: ${JSON.stringify(batch.qualityDistribution)}`);
  console.log(`Variant stats:`);
  for (const s of batch.variantStats) {
    console.log(`  ${s.variant.padEnd(16)} | count=${String(s.count).padEnd(4)} | avgCompiled=${(s.avgCompiled * 100).toFixed(0)}% | avgHardQ=${s.avgHardQuality.toFixed(1)}`);
  }

  // Calibrate
  console.log("\nCalibrating Quality Proxy...");
  const proxy = new QualityProxy({ minSamplesForFit: 5, bootstrapSamples: 100 });

  for (const v of batch.variants) {
    proxy.calibrate(v.content, `Implement ${v.fileName}`, "code_generation", v.hardQuality, {
      compiled: v.compilationResult.compiled,
      compilerErrors: v.compilationResult.errorCount,
      smokeTestPassed: v.compilationResult.compiled && v.hardQuality >= 0.5,
    });
  }
  console.log(`Calibrated: ${proxy.getCalibrationSize()} points`);

  // Learned weights
  const weights = proxy.getWeights();
  if (weights) {
    console.log("\nLearned regression weights (from REAL code):");
    const signalNames = ["compilation", "testPass", "codeBlocks", "noErrors", "keywordCov", "structScore", "respLength", "attEfficiency"];
    console.log(`  ${"Signal".padEnd(16)} | ${"Weight".padEnd(10)} | ${"Conf".padEnd(6)} | ${"ρ"}`);
    console.log(`  ${"-".repeat(48)}`);
    for (let i = 0; i < weights.weights.length; i++) {
      const dir = weights.weights[i] >= 0 ? "+" : "";
      const conf = weights.confidence[i] >= 0.7 ? "HIGH" : weights.confidence[i] >= 0.4 ? "MED" : "LOW";
      console.log(`  ${signalNames[i].padEnd(16)} | ${dir}${weights.weights[i].toFixed(4).padEnd(9)} | ${conf.padEnd(6)} | ${weights.relevance[i].toFixed(3)}`);
    }
    console.log(`  ${"intercept".padEnd(16)} | ${weights.intercept.toFixed(4)}`);
  }

  // Test predictions
  console.log("\n--- Prediction: Good vs Bad ---");
  const goodV = batch.variants.find(v => v.variant === "original" && v.hardQuality >= 1.0);
  const badV = batch.variants.find(v => v.variant === "syntax_broken");

  if (goodV && badV) {
    const goodPred = proxy.predict(goodV.content, `Implement ${goodV.fileName}`, "code_generation",
      { compiled: true, smokeTestPassed: true });
    const badPred = proxy.predict(badV.content, `Implement ${badV.fileName}`, "code_generation",
      { compiled: false, compilerErrors: badV.compilationResult.errorCount || 3, smokeTestPassed: false });

    console.log(`Good (${goodV.fileName}): ${(goodPred.predictedQuality * 100).toFixed(0)}% [${(goodPred.confidenceLow * 100).toFixed(0)}–${(goodPred.confidenceHigh * 100).toFixed(0)}%] reliable=${goodPred.isReliable}`);
    console.log(`Bad  (${badV.fileName}): ${(badPred.predictedQuality * 100).toFixed(0)}% [${(badPred.confidenceLow * 100).toFixed(0)}–${(badPred.confidenceHigh * 100).toFixed(0)}%] reliable=${badPred.isReliable}`);
    console.log(`Good > Bad: ${goodPred.predictedQuality > badPred.predictedQuality ? '✅ YES' : '❌ NO'}`);
  }

  // Signal profile
  console.log("\n--- Signal Profile (PACE Fig 3) ---");
  for (const s of proxy.getSignalProfile("code_generation").slice(0, 6)) {
    console.log(`  ${s.signal.padEnd(20)} | ${s.category.padEnd(12)} | ρ=${s.relevance.toFixed(3)} ${"█".repeat(Math.round(s.relevance * 25))}`);
  }

  console.log(`\n=== Done: ${proxy.getCalibrationSize()} real-code calibration points ===`);
}

main().catch(console.error);
