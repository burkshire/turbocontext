// ============================================================================
// Turbocontext V6 — Learner Optimizer (extracted from learner.ts)
// ============================================================================
//
// Self-contained parameter optimization logic. Takes recent execution records
// and produces LearningAdjustment patches. Extracted from the Learner god module
// to reduce its surface area and enable independent testing.
// ============================================================================

import type { ExecutionRecord, TurboContextConfig } from "../types.js";

export interface LearningAdjustment {
  component: string;
  changes: Record<string, string>;
  reason: string;
}

/**
 * learnFromHistory: runs all parameter optimization rules against recent
 * execution records. Returns a list of adjustments to apply.
 *
 * Rules (ordered by impact):
 *   1. Compression weights (α, β, γ) — adapt to compression-quality correlation
 *   2. Complexity thresholds (θ₁, θ₂) — adapt model tier selection
 *   3. Temperature schedule — adapt generation temperature
 *   4. Branch thresholds — adapt per-task-type quality thresholds
 */
export function learnFromHistory(
  config: TurboContextConfig,
  recent: ExecutionRecord[],
  branchThresholds: Map<string, { mean: number; count: number }>,
): {
  adjustments: LearningAdjustment[];
  consolidationResult: { consolidatedCount: number; archivedCount: number };
} {
  if (recent.length < 5) {
    return { adjustments: [], consolidationResult: { consolidatedCount: 0, archivedCount: 0 } };
  }

  const adjustments: LearningAdjustment[] = [];

  // Rule 1: Compression weights
  const compAdj = optimizeCompressionWeights(config, recent);
  if (compAdj) adjustments.push(compAdj);

  // Rule 2: Complexity thresholds
  const thresholdAdj = optimizeComplexityThresholds(config, recent);
  if (thresholdAdj) adjustments.push(thresholdAdj);

  // Rule 3: Temperature schedule
  const tempAdj = optimizeTemperatureSchedule(config, recent);
  if (tempAdj) adjustments.push(tempAdj);

  // Rule 4: Per-task-type quality thresholds
  const branchAdj = optimizeBranchThresholds(config, recent, branchThresholds);
  adjustments.push(...branchAdj);

  return {
    adjustments,
    consolidationResult: { consolidatedCount: 0, archivedCount: 0 },
  };
}

// ── Rule 1: Compression Weights ──

function optimizeCompressionWeights(
  config: TurboContextConfig,
  recent: ExecutionRecord[],
): LearningAdjustment | null {
  const highQuality = recent.filter(r => r.qualityScore >= 0.85);
  const lowQuality = recent.filter(r => r.qualityScore < 0.70);
  if (highQuality.length < 3 || lowQuality.length < 3) return null;

  const avgHighCompression = highQuality.reduce((s, r) => s + r.compressionRatio, 0) / highQuality.length;
  const avgLowCompression = lowQuality.reduce((s, r) => s + r.compressionRatio, 0) / lowQuality.length;

  const oldAlpha = config.alpha;
  const oldBeta = config.beta;
  const oldGamma = config.gamma;

  // High compression + high quality → reduce semantic weight, increase specificity
  if (avgHighCompression - avgLowCompression > 0.2) {
    config.gamma = Math.min(0.40, config.gamma + 0.02);
    config.alpha = Math.max(0.35, config.alpha - 0.01);
  }
  // Low compression + low quality → increase semantic weight
  if (avgLowCompression < 0.3) {
    config.alpha = Math.min(0.70, config.alpha + 0.02);
    config.gamma = Math.max(0.15, config.gamma - 0.01);
  }

  // Renormalize
  const total = config.alpha + config.beta + config.gamma;
  config.alpha /= total;
  config.beta /= total;
  config.gamma /= total;

  return {
    component: "compression_weights",
    changes: {
      alpha: `${oldAlpha.toFixed(3)} → ${config.alpha.toFixed(3)}`,
      beta: `${oldBeta.toFixed(3)} → ${config.beta.toFixed(3)}`,
      gamma: `${oldGamma.toFixed(3)} → ${config.gamma.toFixed(3)}`,
    },
    reason: `highQ_compression=${(avgHighCompression * 100).toFixed(0)}%, lowQ_compression=${(avgLowCompression * 100).toFixed(0)}%`,
  };
}

// ── Rule 2: Complexity Thresholds ──

function optimizeComplexityThresholds(
  config: TurboContextConfig,
  recent: ExecutionRecord[],
): LearningAdjustment | null {
  if (recent.length < 10) return null;
  const oldLow = config.complexityThresholdLow;
  const oldHigh = config.complexityThresholdHigh;
  const byModel: Record<string, ExecutionRecord[]> = {};

  for (const r of recent) {
    (byModel[r.modelUsed] ||= []).push(r);
  }

  let changed = false;

  const fastRecords = byModel["fast"] || [];
  if (fastRecords.length >= 3) {
    const passRate = fastRecords.filter(r => r.qualityScore >= 0.8).length / fastRecords.length;
    if (passRate > 0.9) {
      config.complexityThresholdLow = Math.min(0.45, config.complexityThresholdLow + 0.03);
      changed = true;
    } else if (passRate < 0.7) {
      config.complexityThresholdLow = Math.max(0.20, config.complexityThresholdLow - 0.03);
      changed = true;
    }
  }

  const deepRecords = byModel["deep"] || [];
  if (deepRecords.length >= 3) {
    const failRate = deepRecords.filter(r => r.qualityScore < 0.8).length / deepRecords.length;
    if (failRate > 0.3) {
      config.complexityThresholdHigh = Math.min(0.85, config.complexityThresholdHigh + 0.03);
      changed = true;
    }
  }

  if (!changed) return null;

  return {
    component: "complexity_thresholds",
    changes: {
      theta1: `${oldLow.toFixed(3)} → ${config.complexityThresholdLow.toFixed(3)}`,
      theta2: `${oldHigh.toFixed(3)} → ${config.complexityThresholdHigh.toFixed(3)}`,
    },
    reason: `fast_pass_rate adjusted based on ${fastRecords.length} fast + ${deepRecords.length} deep records`,
  };
}

// ── Rule 3: Temperature Schedule ──

function optimizeTemperatureSchedule(
  config: TurboContextConfig,
  recent: ExecutionRecord[],
): LearningAdjustment | null {
  const avgAttempts = recent.reduce((s, r) => s + r.attemptCount, 0) / recent.length;
  const oldT0 = config.temperatureSchedule[0];

  if (avgAttempts <= 1.1) {
    config.temperatureSchedule = [
      Math.max(0.3, config.temperatureSchedule[0] - 0.05),
      config.temperatureSchedule[1],
      config.temperatureSchedule[2],
    ] as [number, number, number];
  } else if (avgAttempts >= 2.5) {
    config.temperatureSchedule = [
      Math.min(0.9, config.temperatureSchedule[0] + 0.05),
      config.temperatureSchedule[1],
      config.temperatureSchedule[2],
    ] as [number, number, number];
  } else {
    return null;
  }

  return {
    component: "temperature_schedule",
    changes: { t0: `${oldT0.toFixed(2)} → ${config.temperatureSchedule[0].toFixed(2)}` },
    reason: `avg_attempts=${avgAttempts.toFixed(2)} → ${avgAttempts <= 1.1 ? "lower t0" : "raise t0"}`,
  };
}

// ── Rule 4: Branch Thresholds ──

function optimizeBranchThresholds(
  config: TurboContextConfig,
  recent: ExecutionRecord[],
  branchThresholds: Map<string, { mean: number; count: number }>,
): LearningAdjustment[] {
  const adjustments: LearningAdjustment[] = [];

  for (const [taskType, stats] of branchThresholds) {
    if (stats.count < 5) continue;
    const taskRecent = recent.filter(r => r.taskType === taskType);
    if (taskRecent.length < 3) continue;

    const successRate = taskRecent.filter(r => r.qualityScore >= config.qualityThreshold).length / taskRecent.length;

    // If success rate is consistently high, consider lowering threshold for this type
    if (successRate > 0.9) {
      adjustments.push({
        component: `branch_threshold_${taskType}`,
        changes: { suggestion: "lower_quality_threshold" },
        reason: `${taskType}: ${(successRate * 100).toFixed(0)}% success rate — consider lowering threshold`,
      });
    }
  }

  return adjustments;
}
