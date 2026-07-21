// ============================================================
// v2.3-v2.4 — Strategy Evolution Engine (extracted from learner.ts)
// ============================================================
// proposeMutation → recordTrial → decideKeepDiscard loop.
// Inspired by Karpathy's autoresearch keep/discard cycle.
//
// All functions are stateful — they mutate the provided evolution
// and activeMutation objects. The Learner class owns these objects
// and delegates computation to these functions.
// ============================================================

import type {
  TaskType, BranchState,
  StrategyMutation, EvolutionExperiment,
  StrategyEvolutionData, TrialLogEntry, ExperimentType,
} from "../types.js";

// ------------------------------------------------------------------
// Active Mutation Query
// ------------------------------------------------------------------

/**
 * 获取当前活跃的变异策略（供 composer 使用）。
 */
export function getActiveMutation(
  evolution: StrategyEvolutionData,
  activeMutation: Map<TaskType, { experimentId: string; trialBaselines: number[] }>,
  taskType: TaskType,
): StrategyMutation | null {
  const entry = activeMutation.get(taskType);
  if (!entry) return null;
  const exp = evolution.experiments.find(e => e.id === entry.experimentId);
  if (!exp || exp.status !== "pending") return null;
  return exp.mutation;
}

// ------------------------------------------------------------------
// Mutation Proposal
// ------------------------------------------------------------------

/**
 * 提出一个策略变异提案（autoresearch-style: propose a change）。
 *
 * 在当前最佳策略基础上生成一个合理的变体。
 * 变异类型:
 *   - merge_rounds:  合并两个轮次 → 简化
 *   - remove_round:  移除一个轮次 → 简化
 *   - reorder_rounds: 调整顺序  → 中性
 */
export function proposeMutation(
  branches: Map<TaskType, BranchState>,
  evolution: StrategyEvolutionData,
  activeMutation: Map<TaskType, { experimentId: string; trialBaselines: number[] }>,
  taskType: TaskType,
  evolutionTrialSize: number = 5,
): StrategyMutation | null {
  const branch = branches.get(taskType);
  if (!branch || branch.totalExperiments < 3) return null;

  // 已有 pending 实验时不重复提案
  if (activeMutation.has(taskType)) return null;

  // 检查是否已尝试过相同变异
  const recentExps = evolution.experiments
    .filter(e => e.taskType === taskType)
    .slice(-2);
  const alreadyTriedMutations = new Set(
    recentExps.map(e => JSON.stringify(e.mutation))
  );

  const candidates: Array<StrategyMutation & { complexityDelta: number }> = [
    { type: "merge_rounds", roundIndices: [1, 2], newGoal: "执行并验证", complexityDelta: -1 },
    { type: "merge_rounds", roundIndices: [0, 1], newGoal: "理解并执行", complexityDelta: -1 },
    { type: "remove_round", roundIndex: 2, complexityDelta: -1 },
    { type: "reorder_rounds", newOrder: [2, 0, 1], complexityDelta: 0 },
    // v6: Multi-target parameter mutations (Karpathy-inspired experiment type diversity)
    // Compression weights: log-normal perturbation, clamp to valid ranges, normalize gamma
    (() => {
      const a = Math.max(0.10, Math.min(0.95, 0.55 * Math.exp((Math.random() - 0.5) * 0.3)));
      const b = Math.max(0.10, Math.min(0.95, 0.25 * Math.exp((Math.random() - 0.5) * 0.3)));
      const g = Math.max(0.05, 1.0 - a - b);
      return { type: "mutate_compression_weights" as const, alpha: a, beta: b, gamma: g, complexityDelta: 0 };
    })(),
    { type: "mutate_model_tiers",
      theta1: Math.max(0.05, Math.min(0.80, 0.30 * Math.exp((Math.random() - 0.5) * 0.2))),
      theta2: Math.max(0.10, Math.min(0.90, 0.50 * Math.exp((Math.random() - 0.5) * 0.2))),
      complexityDelta: 0 },
    { type: "mutate_temperature",
      schedule: [
        Math.max(0.0, Math.min(1.0, 0.7 * Math.exp((Math.random() - 0.5) * 0.2))),
        Math.max(0.0, Math.min(1.0, 0.35 * Math.exp((Math.random() - 0.5) * 0.2))),
        Math.max(0.0, Math.min(1.5, 0.1 * Math.exp((Math.random() - 0.5) * 0.3))),
      ] as [number, number, number],
      complexityDelta: 0 },
    { type: "mutate_quality_weights",
      dimWeights: [0.25, 0.35, 0.20, 0.20].map(w =>
        Math.max(0.05, Math.min(0.60, w * Math.exp((Math.random() - 0.5) * 0.3)))
      ) as [number, number, number, number],
      complexityDelta: 0 },
    { type: "mutate_retrieval",
      mmrLambda: Math.max(0.10, Math.min(0.95, 0.70 * Math.exp((Math.random() - 0.5) * 0.25))),
      topK: Math.round(Math.max(1, Math.min(20, 5 * Math.exp((Math.random() - 0.5) * 0.3)))),
      complexityDelta: 0 },
  ];

  for (const mutation of candidates) {
    const { complexityDelta, ...mutationOnly } = mutation;
    (mutationOnly as StrategyMutation).complexityDelta = complexityDelta;
    const key = JSON.stringify(mutationOnly);
    if (!alreadyTriedMutations.has(key)) {
      const exp: EvolutionExperiment = {
        id: `evo_${evolution.totalExperiments + 1}_${taskType}_${Date.now()}`,
        taskType,
        parentStrategyId: `baseline_${taskType}`,
        mutation: mutationOnly,
        status: "pending",
        trialCount: 0,
        trialQualitySum: 0,
        trialTokensSum: 0,
        baselineCount: 0,
        baselineQualitySum: 0,
        baselineTokensSum: 0,
        crashedEarly: false,
        startedAt: Date.now(),
        concludedAt: null,
      };
      evolution.experiments.push(exp);
      evolution.currentExperimentId = exp.id;
      evolution.totalExperiments++;
      activeMutation.set(taskType, {
        experimentId: exp.id,
        trialBaselines: [],
      });
      return mutationOnly;
    }
  }
  return null;
}

// ------------------------------------------------------------------
// Trial Recording
// ------------------------------------------------------------------

/**
 * 记录一次 trial 结果（autoresearch-style: run the experiment）。
 *
 * 根据 usingMutation 判断当前执行是 trial 还是 baseline。
 * v2.4: 记录完整试验日志 + token 效率追踪。
 */
export function recordTrial(
  evolution: StrategyEvolutionData,
  activeMutation: Map<TaskType, { experimentId: string; trialBaselines: number[] }>,
  taskType: TaskType,
  qualityScore: number,
  usingMutation: boolean,
  tokensUsed: number,
  evolutionTrialSize: number = 5,
): void {
  const entry = activeMutation.get(taskType);
  if (!entry) return;

  const exp = evolution.experiments.find(e => e.id === entry.experimentId);
  if (!exp || exp.status === "crashed") return;
  if (exp.status !== "pending") return;

  if (usingMutation) {
    exp.trialCount++;
    exp.trialQualitySum += qualityScore;
    exp.trialTokensSum += tokensUsed;
  } else {
    exp.baselineCount++;
    exp.baselineQualitySum += qualityScore;
    exp.baselineTokensSum += tokensUsed;
  }

  // 记录完整试验日志（autoresearch: results.tsv）
  evolution.trialLog.push({
    experimentId: exp.id,
    taskType,
    usingMutation,
    qualityScore,
    tokensUsed,
    timestamp: Date.now(),
    status: "success",
  });
  if (evolution.trialLog.length > 1000) {
    evolution.trialLog = evolution.trialLog.slice(-1000);
  }

  // 达到 trial size → 决策 keep/discard
  if (exp.trialCount >= evolutionTrialSize) {
    decideKeepDiscard(evolution, activeMutation, exp);
  }
}

/**
 * 记录一次崩溃的 trial（autoresearch-style: crash resilience）。
 *
 * 崩溃 → 立即标记为 crashed 并 auto-discard，
 * 不等 5 次 trial（autoresearch: "fundamentally broken → mark crash, skip"）。
 */
export function recordTrialCrash(
  evolution: StrategyEvolutionData,
  activeMutation: Map<TaskType, { experimentId: string; trialBaselines: number[] }>,
  taskType: TaskType,
): void {
  const entry = activeMutation.get(taskType);
  if (!entry) return;

  const exp = evolution.experiments.find(e => e.id === entry.experimentId);
  if (!exp || exp.status !== "pending") return;

  exp.trialCount++;
  exp.crashedEarly = true;

  evolution.trialLog.push({
    experimentId: exp.id,
    taskType,
    usingMutation: true,
    qualityScore: 0,
    tokensUsed: 0,
    timestamp: Date.now(),
    status: "crash",
  });

  exp.status = "crashed";
  exp.concludedAt = Date.now();
  evolution.currentExperimentId = null;
  activeMutation.delete(exp.taskType);
  evolution.discardedCount++;

  console.log(
    `[TurboContext Evolution] ${exp.taskType}: crashed — auto-discarded ` +
    `(mutation: ${exp.mutation.type})`
  );
}

// ------------------------------------------------------------------
// Keep / Discard Decision
// ------------------------------------------------------------------

/**
 * 决策 keep/discard（autoresearch-style: evaluate the result）。
 *
 * v2.4 增强:
 *   1. 简约性加权: 简化型变异降低 keep 门槛，复杂化变异提高门槛
 *   2. Token 效率: 质量/token 也必须不退化
 *   3. 保留后自动晋升为 canonical 策略
 */
export function decideKeepDiscard(
  evolution: StrategyEvolutionData,
  activeMutation: Map<TaskType, { experimentId: string; trialBaselines: number[] }>,
  exp: EvolutionExperiment,
): { status: string; delta: number } {
  const trialAvg = exp.trialQualitySum / exp.trialCount;
  const baselineAvg = exp.baselineCount > 0
    ? exp.baselineQualitySum / exp.baselineCount
    : trialAvg * 0.95;

  const delta = trialAvg - baselineAvg;

  // 简约性调整（autoresearch: simplicity criterion）
  // v6: Increased to 0.02 — stronger bonus for simpler mutations.
  // "A 0.001 improvement from deleting code? Definitely keep.
  //  A 0.001 improvement from adding 20 hacky lines? Probably not." — Karpathy
  const complexityDelta = exp.mutation.complexityDelta ?? 0;
  const simplicityAdjustment = -complexityDelta * 0.02;
  const adjustedDelta = delta + simplicityAdjustment;

  // Token 效率检查（autoresearch: fixed budget = fair comparison）
  let tokenEfficiencyPenalty = 0;
  if (exp.trialTokensSum > 0 && exp.baselineTokensSum > 0) {
    const trialEff = exp.trialQualitySum / Math.max(1, exp.trialTokensSum / 1000);
    const baselineEff = exp.baselineQualitySum / Math.max(1, exp.baselineTokensSum / 1000);
    const effRatio = trialEff / Math.max(0.001, baselineEff);
    if (effRatio < 0.95) {
      tokenEfficiencyPenalty = 0.015;
    }
  }

  const finalDelta = adjustedDelta - tokenEfficiencyPenalty;
  const baseQualityThreshold = 0.02;

  if (finalDelta >= baseQualityThreshold) {
    exp.status = "kept";
    evolution.keptCount++;

    // 晋升为 canonical 策略（autoresearch: branch tip = best config）
    const canonical = evolution.canonicalStrategies[exp.taskType] || [];
    canonical.push(exp.mutation);
    evolution.canonicalStrategies[exp.taskType] = canonical;
  } else {
    exp.status = "discarded";
    evolution.discardedCount++;
  }

  exp.concludedAt = Date.now();
  evolution.currentExperimentId = null;
  activeMutation.delete(exp.taskType);

  const simplicityLabel = complexityDelta < 0 ? "simpler" : complexityDelta > 0 ? "complex" : "neutral";
  console.log(
    `[TurboContext Evolution] ${exp.taskType}: ${exp.status} ` +
    `(trial avg=${(trialAvg * 100).toFixed(1)}%, ` +
    `baseline avg=${(baselineAvg * 100).toFixed(1)}%, ` +
    `delta=${(delta * 100).toFixed(2)}%, ` +
    `simplicity=${simplicityLabel}(${complexityDelta >= 0 ? "+" : ""}${complexityDelta}), ` +
    `tokenPenalty=${tokenEfficiencyPenalty > 0 ? "yes" : "no"}, ` +
    `finalDelta=${(finalDelta * 100).toFixed(2)}%)`
  );

  return { status: exp.status, delta: finalDelta };
}

// ------------------------------------------------------------------
// Canonical Strategy Management
// ------------------------------------------------------------------

/**
 * 获取指定任务类型的 canonical 策略栈。
 * autoresearch: the branch tip IS the best config.
 */
export function getCanonicalMutations(
  evolution: StrategyEvolutionData,
  taskType: TaskType,
): StrategyMutation[] {
  return evolution.canonicalStrategies[taskType] || [];
}

/**
 * 重置指定类型的 canonical 策略（回退到原始基线）。
 * autoresearch: git reset to undo bad ideas.
 */
export function resetCanonicalStrategy(
  evolution: StrategyEvolutionData,
  taskType: TaskType,
): void {
  delete evolution.canonicalStrategies[taskType];
}

/**
 * 获取进化统计。
 */
export function getEvolutionStats(
  evolution: StrategyEvolutionData,
  activeMutation: Map<TaskType, { experimentId: string; trialBaselines: number[] }>,
): { total: number; kept: number; discarded: number; active: number } {
  return {
    total: evolution.totalExperiments,
    kept: evolution.keptCount,
    discarded: evolution.discardedCount,
    active: activeMutation.size,
  };
}

/**
 * 获取完整试验日志（autoresearch: results.tsv）。
 */
export function getTrialLog(evolution: StrategyEvolutionData): TrialLogEntry[] {
  return [...evolution.trialLog];
}

// ============================================================
// v3.3 — Retrieval strategy evolution
// ============================================================
// Extends the same proposeMutation → recordTrial → decideKeepDiscard
// loop to retrieval hyperparameters (dim weights, MMR λ, top_k, token budgets).

import type {
  RetrievalStrategyState, ExperienceEntry,
} from "../types.js";
import {
  ucbSelectDimension as ucbSelectDim,
  recordUCBOutcome,
} from "./rl-core.js";

/**
 * Propose a mutation to retrieval strategy hyperparameters.
 *
 * Uses UCB to select which dimension to mutate. Covers:
 *   - dim_weight.* (scoring dimension multipliers)
 *   - mmr_lambda (diversity-vs-relevance balance)
 *   - top_k (number of retrieved items)
 *   - token_budget_tier.* (per-tier token allocations)
 */
export function proposeRetrievalStrategyMutation(
  strategy: RetrievalStrategyState,
): { newStrategy: RetrievalStrategyState; mutation: Record<string, unknown> } | null {
  if (strategy.pendingMutation) return null;

  const minTrials = Math.max(3, Math.floor(strategy.generation * 0.5 + 2));
  if (strategy.trialsInGeneration < minTrials) return null;

  const magnitude = strategy.mutationMagnitude;

  // Plateau burst
  const fh = strategy.fitnessHistory;
  if (fh.length >= 3) {
    const recent = fh.slice(-3).map(f => f.fitness);
    if (Math.max(...recent) <= (fh[fh.length - 3]?.fitness ?? 0.5) + 0.02) {
      strategy.mutationMagnitude = Math.min(0.40, magnitude * 1.5);
    }
  }

  const newStrategy = structuredClone(strategy);
  const allDims = Object.keys(strategy.dimWeights);
  const roll = Math.random();

  let mutation: Record<string, unknown>;
  const actualMagnitude = newStrategy.mutationMagnitude;

  if (roll < 0.70) {
    // UCB-selected dimension weight mutation
    const dim = ucbSelectDim(
      strategy.ucbDimCounts, strategy.ucbDimRewards,
      strategy.ucbTotalMutations, allDims,
    );
    const oldVal = strategy.dimWeights[dim] ?? 1.0;
    const multiplier = Math.exp(
      (Math.random() - 0.5) * 2 * 0.1 * actualMagnitude
    );
    const newVal = Math.round(
      Math.max(0.25, Math.min(4.0, oldVal * multiplier)) * 1000
    ) / 1000;
    newStrategy.dimWeights = { ...strategy.dimWeights, [dim]: newVal };
    mutation = { target: `dim_weight.${dim}`, old: oldVal, new: newVal, _ucb_dim: dim };
  } else if (roll < 0.85) {
    const oldVal = strategy.mmrLambda;
    const delta = (Math.random() - 0.5) * 2 * 0.05 * actualMagnitude;
    const newVal = Math.round(
      Math.max(0.20, Math.min(0.95, oldVal + delta)) * 1000
    ) / 1000;
    newStrategy.mmrLambda = newVal;
    mutation = { target: "mmr_lambda", old: oldVal, new: newVal };
  } else if (roll < 0.95) {
    const tierIdx = Math.floor(Math.random() * 3);
    const oldVal = strategy.tokenBudgetTiers[tierIdx];
    const delta = Math.round(
      (Math.random() - 0.5) * 2 * oldVal * actualMagnitude * 0.5
    );
    const newVal = Math.max(400, Math.min(4000, oldVal + delta));
    const newTiers = [...strategy.tokenBudgetTiers] as [number, number, number];
    newTiers[tierIdx] = newVal;
    newStrategy.tokenBudgetTiers = newTiers;
    mutation = { target: `token_budget_tier.${tierIdx}`, old: oldVal, new: newVal };
  } else {
    const oldVal = strategy.topK;
    const delta = Math.random() < 0.5 ? -1 : 1;
    const newVal = Math.max(3, Math.min(12, oldVal + delta));
    newStrategy.topK = newVal;
    mutation = { target: "top_k", old: oldVal, new: newVal };
  }

  newStrategy.pendingMutation = mutation;
  newStrategy.ancestorFitness = strategy.fitness;
  newStrategy.trialsInGeneration = 0;
  newStrategy.successesInGeneration = 0;
  newStrategy.generation++;

  return { newStrategy, mutation };
}

/**
 * Decide whether to keep or revert a retrieval strategy mutation.
 */
export function decideRetrievalStrategyMutation(
  strategy: RetrievalStrategyState,
): { decision: "keep" | "revert"; delta: number } | null {
  const pm = strategy.pendingMutation;
  if (!pm) return null;

  const minTrials =
    typeof pm.old === "number" && typeof pm.new === "number" &&
    Math.abs((pm.old as number) - (pm.new as number)) /
    Math.max(Math.abs(pm.old as number), 0.01) > 0.2
      ? 4 : 6;

  if (strategy.trialsInGeneration < minTrials) return null;

  const delta = strategy.fitness - strategy.ancestorFitness;

  let decision: "keep" | "revert";
  if (delta > 0.03) {
    decision = "keep";
    strategy.mutationMagnitude = Math.max(0.05, strategy.mutationMagnitude * 0.8);
  } else if (delta < -0.05) {
    decision = "revert";
    const target = pm.target as string;
    if (target.startsWith("dim_weight.")) {
      const dim = target.slice("dim_weight.".length);
      strategy.dimWeights[dim] = pm.old as number;
    } else if (target === "mmr_lambda") {
      strategy.mmrLambda = pm.old as number;
    } else if (target.startsWith("token_budget_tier.")) {
      const idx = parseInt(target.split(".").pop()!, 10);
      const tiers = [...strategy.tokenBudgetTiers] as [number, number, number];
      tiers[idx] = pm.old as number;
      strategy.tokenBudgetTiers = tiers;
    } else if (target === "top_k") {
      strategy.topK = pm.old as number;
    }
  } else {
    decision = "keep"; // neutral — no strong signal either way
  }

  // Record UCB outcome
  const ucbDim = pm._ucb_dim as string | undefined;
  if (ucbDim) {
    recordUCBOutcome(
      strategy.ucbDimCounts, strategy.ucbDimRewards, ucbDim, delta
    );
    strategy.ucbTotalMutations = Object.values(strategy.ucbDimCounts)
      .reduce((a, b) => a + b, 0);
  }

  strategy.fitnessHistory.push({
    generation: strategy.generation,
    fitness: strategy.fitness,
    delta: Math.round(delta * 10000) / 10000,
    decision,
    mutation: pm,
    trials: strategy.trialsInGeneration,
  });
  if (strategy.fitnessHistory.length > 50) {
    strategy.fitnessHistory = strategy.fitnessHistory.slice(-50);
  }

  strategy.pendingMutation = null;
  return { decision, delta };
}

/**
 * Record trial outcome for retrieval strategy fitness.
 */
export function recordRetrievalStrategyTrial(
  strategy: RetrievalStrategyState,
  outcome: "success" | "failure" | "crash",
  qualityScore: number,
  bestQuality: number,
): RetrievalStrategyState {
  const newStrategy = structuredClone(strategy);
  newStrategy.trialsInGeneration++;

  let reward: number;
  if (outcome === "success") {
    const improvement = Math.max(0, qualityScore - bestQuality);
    reward = 0.5 + 0.5 * Math.min(1.0, improvement / 0.05);
    newStrategy.successesInGeneration++;
  } else if (outcome === "crash") {
    reward = 0.0;
  } else {
    reward = 0.15;
  }

  const alpha = newStrategy.trialsInGeneration < 10 ? 0.20 : 0.08;
  newStrategy.fitness = Math.round(
    ((1 - alpha) * strategy.fitness + alpha * reward) * 10000
  ) / 10000;

  return newStrategy;
}

// ------------------------------------------------------------------
// Experience Library (meta-model)
// ------------------------------------------------------------------

/**
 * Extract a compact scenario signature from current evolution context.
 */
export function extractScenario(
  taskType: TaskType,
  totalExperiments: number,
  branchState: { successCount: number; failureCount: number } | null,
  fitnessHistory: Array<{ fitness: number; mutation?: Record<string, unknown> }>,
): Record<string, unknown> {
  const n = totalExperiments;

  // Stage classification
  let stage: string;
  if (n < 8) stage = "early";
  else if (n < 25) stage = "mid";
  else stage = "mature";

  // Trend from fitness history
  let trend = "insufficient_data";
  if (fitnessHistory.length >= 3) {
    const recent = fitnessHistory.slice(-3).map(f => f.fitness);
    const avgDelta = recent.length >= 2
      ? (recent[recent.length - 1] - recent[0]) / recent.length
      : 0;
    if (avgDelta > 0.02) trend = "improving";
    else if (avgDelta < -0.02) trend = "declining";
    else trend = "flat";
  }

  // Crash rate
  const crashRate = branchState
    ? Math.round((branchState.failureCount / Math.max(n, 1)) * 100) / 100
    : 0;

  return {
    taskType,
    stage,
    trend,
    crashRate,
    nExperiments: n,
  };
}

/**
 * Record an experience entry for future meta-model guidance.
 */
export function recordExperience(
  experienceLib: ExperienceEntry[],
  scenario: Record<string, unknown>,
  mutation: Record<string, unknown>,
  outcome: "keep" | "revert" | "crash",
  delta: number,
): ExperienceEntry[] {
  const entry: ExperienceEntry = {
    scenario: scenario as ExperienceEntry["scenario"],
    mutation: {
      target: String(mutation.target ?? ""),
      old: mutation.old,
      new: mutation.new,
    },
    outcome,
    delta: Math.round(delta * 10000) / 10000,
    timestamp: new Date().toISOString(),
  };

  const lib = [...experienceLib, entry];
  return lib.length > 200 ? lib.slice(-200) : lib;
}

/**
 * Predict best mutation target from experience library.
 * Matches by scenario (taskType + stage + trend) and returns
 * the target with highest average fitness delta.
 */
export function predictBestMutation(
  experienceLib: ExperienceEntry[],
  scenario: Record<string, unknown>,
): string | null {
  if (experienceLib.length < 5) return null;

  // Narrow match: same taskType, stage, trend
  const narrow = experienceLib.filter(e =>
    e.scenario.taskType === scenario.taskType &&
    e.scenario.stage === scenario.stage &&
    e.scenario.trend === scenario.trend
  );

  // Broad match: same stage + trend
  const broad = experienceLib.filter(e =>
    e.scenario.stage === scenario.stage &&
    e.scenario.trend === scenario.trend
  );

  const pool = narrow.length >= 4 ? narrow : broad.length >= 6 ? broad : null;
  if (!pool) return null;

  // Aggregate by mutation target
  const byTarget = new Map<string, number[]>();
  for (const e of pool) {
    const target = e.mutation.target;
    // Normalize dim_weight.xxx → dim_weight
    const base = target.startsWith("dim_weight.") ? "dim_weight" : target;
    const weightedDelta = e.outcome === "keep" ? e.delta : e.delta * 0.3;
    if (!byTarget.has(base)) byTarget.set(base, []);
    byTarget.get(base)!.push(weightedDelta);
  }

  // Score each target
  let bestTarget: string | null = null;
  let bestScore = -Infinity;
  for (const [target, deltas] of byTarget) {
    if (deltas.length < 2) continue;
    const avg = deltas.reduce((a, b) => a + b, 0) / deltas.length;
    const positiveRatio = deltas.filter(d => d > 0).length / deltas.length;
    const score = avg * (0.5 + 0.5 * positiveRatio);
    if (score > bestScore && score > 0.005) {
      bestScore = score;
      bestTarget = target;
    }
  }

  return bestTarget;
}

// ============================================================
// v6: Experiment Type Selection (Karpathy-inspired)
// ============================================================
/**
 * selectExperimentType: picks an experiment type using UCB-like exploration.
 *
 * Six types from autoresearch:
 *   hypothesis_test (30%) — test a specific hypothesis
 *   parameter_sweep (20%) — scan parameter space
 *   ablation_study (15%) — remove a feature, measure impact
 *   transfer_experiment (15%) — cross-task-type knowledge transfer
 *   boundary_probe (10%) — explore crash boundaries
 *   adversarial_test (10%) — challenge existing conclusions
 *
 * Probability distribution shifts based on evolution phase:
 *   Early (few exps): prioritize hypothesis_test + parameter_sweep
 *   Mid: balanced
 *   Late: prioritize adversarial_test + ablation_study
 */
export function selectExperimentType(
  evolution: StrategyEvolutionData,
): ExperimentType {
  const total = evolution.totalExperiments;
  // Shift weights based on maturity
  const phase = total < 10 ? "early" : total < 30 ? "mid" : "late";

  const weights: Record<ExperimentType, number> = phase === "early" ? {
    hypothesis_test: 0.40, parameter_sweep: 0.25, ablation_study: 0.10,
    transfer_experiment: 0.10, boundary_probe: 0.10, adversarial_test: 0.05,
  } : phase === "mid" ? {
    hypothesis_test: 0.30, parameter_sweep: 0.20, ablation_study: 0.15,
    transfer_experiment: 0.15, boundary_probe: 0.10, adversarial_test: 0.10,
  } : {
    hypothesis_test: 0.20, parameter_sweep: 0.15, ablation_study: 0.20,
    transfer_experiment: 0.15, boundary_probe: 0.10, adversarial_test: 0.20,
  };

  const r = Math.random();
  let cumulative = 0;
  for (const [type, weight] of Object.entries(weights) as [ExperimentType, number][]) {
    cumulative += weight;
    if (r < cumulative) return type;
  }
  return "hypothesis_test";
}
