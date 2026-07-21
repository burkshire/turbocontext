// ============================================================================
// Turbocontext V6 — Ablation Runner (Lightweight)
// ============================================================================
//
// Re-enabled with real quality signals from the calibration benchmark.
// Original causal-graph + ablation-engine were deleted because they never
// ran on real LLM output. This version uses the V6 QualityProxy (calibrated
// on real compilation/test signals) to estimate per-file causal contribution
// WITHOUT requiring 2x LLM calls per file.
//
// Approach:
//   1. Extract code blocks from output
//   2. For each source file in the context, estimate quality WITH and WITHOUT
//      that file's code blocks (using the proxy, not an extra LLM call)
//   3. Delta in predicted quality = file's approximate causal contribution
//   4. Record in V5 state for downstream RL credit assignment
//
// This is "soft ablation" — uses the proxy instead of re-running the LLM.
// For "hard ablation" (2x LLM calls), use the original ablation-engine pattern.
// ============================================================================

import type { Task, ContextFragment, ExecutionMetrics } from "../types.js";
import { QualityProxy } from "./quality-proxy.js";
import { extractSignals, signalVectorToArray } from "./signal-extractor.js";

// ============================================================================
// Types
// ============================================================================

export interface AblationResult {
  /** Source file being ablated */
  source: string;
  /** Quality prediction WITH this file's code */
  qualityWithFile: number;
  /** Quality prediction WITHOUT this file's code */
  qualityWithoutFile: number;
  /** Causal delta: positive = file helps, negative = file hurts */
  delta: number;
  /** Confidence in the delta (0-1, higher = proxy more calibrated) */
  confidence: number;
  /** Whether the proxy was calibrated enough for reliable result */
  isReliable: boolean;
}

export interface AblationRun {
  taskId: string;
  taskType: string;
  timestamp: string;
  results: AblationResult[];
  /** Files ranked by causal contribution (most helpful first) */
  ranking: Array<{ source: string; delta: number }>;
}

// ============================================================================
// Ablation Runner
// ============================================================================

/**
 * runSoftAblation: estimate per-file causal contribution using the QualityProxy.
 *
 * For each source file in the context, we estimate quality with and without
 * that file's contribution to the output. The delta is the file's approximate
 * causal effect on output quality.
 *
 * This is O(n) in the number of files (n proxy predictions), not O(2n) LLM calls.
 */
export async function runSoftAblation(
  task: Task,
  output: string,
  contextFragments: ContextFragment[],
  proxy: QualityProxy,
  executionMetrics?: ExecutionMetrics,
): Promise<AblationRun> {
  if (proxy.getCalibrationSize() < 8) {
    return {
      taskId: task.id,
      taskType: task.type,
      timestamp: new Date().toISOString(),
      results: [],
      ranking: [],
    };
  }

  const results: AblationResult[] = [];

  // Full-output baseline prediction
  const fullPrediction = proxy.predict(output, task.description, task.type, executionMetrics);
  const baselineQuality = fullPrediction.predictedQuality;

  for (const frag of contextFragments) {
    // Estimate contribution: if this file's content appears in the output,
    // removing it would reduce quality proportional to its presence
    const contentInOutput = output.includes(frag.source) ||
      (frag.content.length > 50 && output.includes(frag.content.slice(0, 50)));

    if (!contentInOutput && frag.contentType !== "source") {
      results.push({
        source: frag.source,
        qualityWithFile: baselineQuality,
        qualityWithoutFile: baselineQuality,
        delta: 0,
        confidence: fullPrediction.isReliable ? 0.7 : 0.3,
        isReliable: fullPrediction.isReliable,
      });
      continue;
    }

    // Approximate WITHOUT this file: remove its content references from output
    const outputWithoutFile = removeFileContent(output, frag);

    const withoutPrediction = proxy.predict(
      outputWithoutFile,
      task.description,
      task.type,
      executionMetrics,
    );

    const delta = baselineQuality - withoutPrediction.predictedQuality;

    results.push({
      source: frag.source,
      qualityWithFile: baselineQuality,
      qualityWithoutFile: withoutPrediction.predictedQuality,
      delta: Math.round(delta * 1000) / 1000,
      confidence: fullPrediction.isReliable ? 0.8 : 0.3,
      isReliable: fullPrediction.isReliable,
    });
  }

  // Rank by causal contribution (most helpful first)
  const ranking = results
    .filter(r => Math.abs(r.delta) > 0.01)
    .sort((a, b) => b.delta - a.delta)
    .map(r => ({ source: r.source, delta: r.delta }));

  return {
    taskId: task.id,
    taskType: task.type,
    timestamp: new Date().toISOString(),
    results,
    ranking,
  };
}

/**
 * removeFileContent: strips code blocks and text references to a specific
 * source file from the output. Used for "without this file" estimation.
 */
function removeFileContent(output: string, frag: ContextFragment): string {
  let cleaned = output;

  // Remove code block markers referencing this file
  const filePattern = new RegExp(
    `\`\`\`[^\\n]*${escapeRegex(frag.source)}[^\\n]*\\n[\\s\\S]*?\`\`\``,
    "gi"
  );
  cleaned = cleaned.replace(filePattern, "```\n// [file content removed for ablation]\n```");

  // Remove inline references
  const namePattern = new RegExp(
    `\\b${escapeRegex(frag.source.replace(/\\.[^.]+$/, ""))}\\b`,
    "gi"
  );
  cleaned = cleaned.replace(namePattern, "[removed]");

  return cleaned;
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// ============================================================================
// V5 State Integration
// ============================================================================

/**
 * recordAblationToV5: stores ablation results in the V5 engine for
 * downstream causal credit assignment and retrieval boosting.
 *
 * Causal signal precedence (V5 design):
 *   Ablation delta (direct causal) > TD(λ) credit (temporal) > Heuristic score
 */
export function recordAblationToV5(
  run: AblationRun,
  engine: { recordAblation(entry: any): void },
): void {
  for (const r of run.results) {
    if (!r.isReliable || Math.abs(r.delta) < 0.02) continue;
    engine.recordAblation({
      timestamp: run.timestamp,
      taskId: run.taskId,
      taskType: run.taskType,
      source: r.source,
      delta: r.delta,
      confidence: r.confidence,
    });
  }
}
