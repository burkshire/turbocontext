// ============================================================
// v3.3 — RL Core: Thompson Sampling, TD(λ), Advantage, Predictive Model
// ============================================================
// Ported from Karpathy's autoresearch ResearchMemory RL mechanisms.
//
// Five interacting RL subsystems:
//   1. Thompson Sampling — Beta-distribution retrieval exploration
//   2. TD(λ) Eligibility Traces — credit assignment through retrieval chains
//   3. Advantage-weighted utility — removes "easy subsystem" bias
//   4. Online Predictive Model — SGD-updated linear outcome predictor
//   5. Entropy-regularized MMR — outcome diversity in retrieval re-ranking
// ============================================================

import type {
  ThompsonParams, EligibilityTrace, PredictiveModel,
  CurriculumPhase, CurriculumState,
  ExperienceEntry, VerificationRecord,
  RetrievalStrategyState, RLExecutionRecord, ExecutionRecord, TaskType,
} from "../types.js";
import { DEFAULT_CURRICULUM } from "../types.js";
import { createDefaultRetrievalStrategy } from "../types.js";

// ------------------------------------------------------------------
// 1. Thompson Sampling (Beta distribution)
// ------------------------------------------------------------------

/**
 * Sample from Beta(α, β) distribution using the gamma method.
 *
 * Beta(1,1) = uniform [0,1]; Beta(10,2) ≈ peaked near 0.83.
 * Returns a value in [0, 1] representing the sampled retrieval utility.
 *
 * Unlike ε-greedy, this is a proper Bayesian approach to explore/exploit:
 * uncertain memories (low α+β) produce high-variance samples and
 * occasionally get selected, gathering data to reduce uncertainty.
 */
export function thompsonSample(alpha: number, beta: number): number {
  const a = Math.max(0.1, alpha);
  const b = Math.max(0.1, beta);
  // Gamma method: Beta(a,b) = Gamma(a,1) / (Gamma(a,1) + Gamma(b,1))
  const x = gammaSample(a);
  const y = gammaSample(b);
  return x / (x + y);
}

/**
 * Marsaglia-Tsang gamma sampler (shape ≥ 1 case with rejection sampling).
 * For shape < 1, use the standard α → 1+α → shrink trick.
 */
function gammaSample(shape: number): number {
  if (shape < 1) {
    // Use: Gamma(α,1) = Gamma(1+α,1) * U^(1/α)
    const g = gammaSample(shape + 1);
    return g * Math.pow(Math.random(), 1 / shape);
  }

  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);

  for (;;) {
    let x: number;
    let v: number;
    do {
      x = normalRandom();
      v = 1 + c * x;
    } while (v <= 0);

    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

/** Box-Muller normal random */
function normalRandom(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/**
 * Update Beta parameters based on experiment outcome.
 *
 * Success → increase alpha (shift distribution right, higher expected utility)
 * Failure → increase beta (shift distribution left)
 * Crash   → strongly increase beta (strong negative signal)
 */
export function updateThompsonParams(
  params: ThompsonParams,
  outcome: "success" | "failure" | "crash",
  rewardMagnitude: number,
): ThompsonParams {
  const result = { ...params };
  if (outcome === "success") {
    result.alphaTs = Math.min(50, params.alphaTs + rewardMagnitude * 2.0);
  } else if (outcome === "crash") {
    result.betaTs = Math.min(50, params.betaTs + 2.0);
  } else {
    result.betaTs = Math.min(50, params.betaTs + 0.5);
  }
  return result;
}

// ------------------------------------------------------------------
// 2. TD(λ) Eligibility Traces — credit assignment through retrieval chains
// ------------------------------------------------------------------

/**
 * Decay all eligibility traces by γλ.
 *
 * trace[t+1](m) = γλ × trace[t](m)
 *
 * γ=0.90: future credit matters, but less than immediate
 * λ=0.70: blend of Monte Carlo (λ=1) and TD(0) (λ=0)
 */
export function decayEligibilityTraces(
  traces: Map<string, number>,
  gamma: number = 0.90,
  lambda: number = 0.70,
): void {
  const decay = gamma * lambda;
  const toDelete: string[] = [];
  for (const [key, trace] of traces) {
    const newTrace = trace * decay;
    if (newTrace < 0.001) {
      toDelete.push(key);
    } else {
      traces.set(key, newTrace);
    }
  }
  for (const key of toDelete) {
    traces.delete(key);
  }
}

/**
 * Bump eligibility traces for retrieved memory keys.
 * trace(m) += 1 for newly retrieved memories.
 */
export function bumpEligibilityTraces(
  traces: Map<string, number>,
  memoryKeys: string[],
): void {
  for (const key of memoryKeys) {
    traces.set(key, (traces.get(key) || 0) + 1.0);
  }
}

/**
 * Apply TD(λ) update to all memories with active eligibility traces.
 *
 * TD_error = reward - expected_value
 * For each memory with trace > 0:
 *   causal_utility += α × trace × TD_error
 *
 * This propagates credit backward through the retrieval chain.
 * Memories retrieved 3 iterations ago that influenced a good outcome
 * still get partial credit proportional to their decayed trace.
 *
 * Returns the number of memories updated.
 */
export function applyTDUpdate(
  traces: Map<string, number>,
  utilities: Map<string, number>,
  reward: number,
  learningRate: number = 0.10,
): number {
  if (traces.size === 0) return 0;

  // Expected value: average causal_utility of traced memories
  let expectedSum = 0;
  let expectedCount = 0;
  for (const [key, trace] of traces) {
    if (trace > 0.01) {
      expectedSum += utilities.get(key) ?? 0.5;
      expectedCount++;
    }
  }
  const expected = expectedCount > 0 ? expectedSum / expectedCount : 0.5;
  const tdError = reward - expected;

  let updated = 0;
  for (const [key, trace] of traces) {
    if (trace < 0.01) continue;
    const oldUtil = utilities.get(key) ?? 0.5;
    const delta = learningRate * trace * tdError;
    const newUtil = Math.max(-0.5, Math.min(1.5, oldUtil + delta));
    utilities.set(key, Math.round(newUtil * 10000) / 10000);
    updated++;
  }

  return updated;
}

// ------------------------------------------------------------------
// 3. Advantage-weighted utility
// ------------------------------------------------------------------

/**
 * Compute subsystem baselines: V(subsystem) = average causal_utility per family.
 *
 * This removes the "easy subsystem" bias — a memory from an "easy" task type
 * shouldn't get artificially high causal_utility just because its domain succeeds often.
 */
export function computeSubsystemBaselines(
  records: RLExecutionRecord[],
): Map<string, number> {
  const byFamily = new Map<string, number[]>();

  for (const rec of records) {
    const family = rec.taskType;
    const causal = rec.causalUtility ?? 0.5;
    if (!byFamily.has(family)) byFamily.set(family, []);
    byFamily.get(family)!.push(causal);
  }

  const baseline = new Map<string, number>();
  for (const [family, values] of byFamily) {
    baseline.set(family, values.reduce((a, b) => a + b, 0) / values.length);
  }
  baseline.set("general", 0.5); // default

  return baseline;
}

/**
 * Advantage = causal_utility - V(subsystem).
 *
 * Positive advantage → this memory is unusually helpful for its domain.
 * Negative → worse than typical.
 */
export function computeAdvantage(
  record: RLExecutionRecord,
  baseline: Map<string, number>,
): number {
  const causal = record.causalUtility ?? 0.5;
  const familyVal = baseline.get(record.taskType) ?? baseline.get("general") ?? 0.5;
  return causal - familyVal;
}

/**
 * v4.0: Stateless advantage computation for two-phase retrieval Phase 2.
 *
 * Unlike computeAdvantage() which takes an RLExecutionRecord, this variant
 * accepts raw values so the retrieval system can compute advantage without
 * needing full RLExecutionRecord objects.
 *
 * agent.py lines 723-736: _advantage() method.
 */
export function computeAdvantageForMemory(
  causalUtility: number,
  taskType: TaskType,
  baseline: Map<string, number>,
): number {
  const familyVal = baseline.get(taskType) ?? baseline.get("general") ?? 0.5;
  return causalUtility - familyVal;
}

// ------------------------------------------------------------------
// 4. Entropy bonus for MMR diversity
// ------------------------------------------------------------------

/**
 * Compute entropy bonus for outcome diversity in retrieval ranking.
 *
 * If all selected memories have the same outcome (e.g., all successes),
 * a memory with a different outcome gets a high entropy bonus.
 * This prevents "outcome monoculture" in retrieved context.
 */
export function entropyBonus(
  itemOutcome: string,
  selectedOutcomes: string[],
): number {
  if (selectedOutcomes.length === 0) return 0.0;

  // Count outcomes in selected set
  const counts = new Map<string, number>();
  for (const o of selectedOutcomes) {
    counts.set(o, (counts.get(o) || 0) + 1);
  }

  const n = selectedOutcomes.length;
  const pCurrent = (counts.get(itemOutcome) || 0) / n;
  const epsilon = 0.1;

  // Rare outcomes get large bonus: -log(p + ε)
  const bonus = -Math.log(pCurrent + epsilon) * 0.5;
  return Math.max(0.0, bonus);
}

// ------------------------------------------------------------------
// 5. Online Predictive Model (linear, SGD-updated)
// ------------------------------------------------------------------

/**
 * Create a default predictive model.
 */
export function createPredictiveModel(): PredictiveModel {
  return {
    featureWeights: {},
    intercept: 0.5,
    nUpdates: 0,
    learningRate: 0.05,
    recentAccuracy: 0.5,
  };
}

/**
 * Extract features from an execution record for prediction.
 * Features are all computable without extra LLM calls.
 *
 * v4.0: Added hypothesis_complexity and subsystem_family features
 * (agent.py _extract_prediction_features, lines 1818-1881).
 */
export function extractPredictionFeatures(
  record: ExecutionRecord,
  history: ExecutionRecord[],
  branchFamily?: string,
): Record<string, number> {
  const n = history.length;
  const taskType = record.taskType;

  // Feature 1: Task type success rate
  const typeSuccesses = history.filter(
    h => h.taskType === taskType && h.qualityScore >= 0.85
  ).length;
  const typeTotal = history.filter(h => h.taskType === taskType).length || 1;
  const typeSuccessRate = typeSuccesses / typeTotal;

  // Feature 2: Is novel task type? (never attempted before)
  const triedTypes = new Set(history.slice(0, -1).map(h => h.taskType));
  const isNovel = triedTypes.has(taskType) ? 0.0 : 1.0;

  // Feature 3: Log experiment count (knowledge maturity proxy)
  const logN = Math.log(Math.max(n, 1));

  // Feature 4: Recency-weighted momentum for this task type
  const typeRecords = history.filter(h => h.taskType === taskType);
  let momentum = 0.0;
  if (typeRecords.length >= 2) {
    const half = Math.floor(typeRecords.length / 2);
    const earlyAvg = typeRecords.slice(0, half)
      .reduce((s, r) => s + r.qualityScore, 0) / half;
    const lateAvg = typeRecords.slice(half)
      .reduce((s, r) => s + r.qualityScore, 0) / (typeRecords.length - half);
    momentum = (lateAvg - earlyAvg) / Math.max(Math.abs(earlyAvg), 0.001);
  }

  // Feature 5: Compression ratio (proxy for task complexity)
  const compressionRatio = record.compressionRatio;

  // v4.0 Feature 6: Hypothesis/task complexity (word count / 50, capped at 1.0)
  // agent.py line 1872-1873: hyp_complexity feature
  const taskDesc = record.taskId || "";
  const wordCount = taskDesc.split(/[\s_-]+/).filter(w => w.length > 0).length;
  const hypComplexity = Math.min(1.0, wordCount / 50.0);

  // v4.0 Feature 7: Subsystem family one-hot indicator
  // agent.py: family_success_rate feature, but here we encode the family itself
  // as a categorical proxy — 1.0 if branchFamily matches taskType family, else 0
  const subsystemFamily = branchFamily
    ? (taskType.startsWith(branchFamily) || branchFamily.includes(taskType) ? 1.0 : 0.0)
    : 0.5;  // unknown family → neutral

  const features: Record<string, number> = {
    type_success_rate: Math.round(typeSuccessRate * 10000) / 10000,
    is_novel: isNovel,
    log_n: Math.round(logN * 10000) / 10000,
    type_momentum: Math.round(momentum * 10000) / 10000,
    compression_ratio: Math.round(compressionRatio * 10000) / 10000,
    hypothesis_complexity: Math.round(hypComplexity * 10000) / 10000,
    subsystem_family: Math.round(subsystemFamily * 10000) / 10000,
  };

  return features;
}

/**
 * Predict experiment success probability from features.
 * Uses the learned linear model weights + sigmoid squash.
 * Higher = more likely to succeed.
 */
export function predictOutcome(
  features: Record<string, number>,
  model: PredictiveModel,
): number {
  let score = model.intercept;
  for (const [fname, fval] of Object.entries(features)) {
    const w = model.featureWeights[fname] ?? 0.0;
    score += w * fval;
  }

  // Sigmoid squash to [0, 1]
  try {
    return Math.round((1.0 / (1.0 + Math.exp(-score))) * 10000) / 10000;
  } catch {
    return score > 0 ? 1.0 : 0.0;
  }
}

/**
 * Online SGD update of the predictive model.
 *
 * actualOutcome: 1.0 = success, 0.5 = failure, 0.0 = crash.
 * Loss = (prediction - actual)²
 */
export function updatePredictiveModel(
  features: Record<string, number>,
  actualOutcome: number,
  model: PredictiveModel,
): { prediction: number; error: number } {
  const prediction = predictOutcome(features, model);
  const error = prediction - actualOutcome;
  const lr = model.learningRate;

  // Sigmoid derivative
  const sigmoidDeriv = prediction * (1.0 - prediction);
  const gradientBase = 2.0 * error * sigmoidDeriv;

  // Update weights
  for (const [fname, fval] of Object.entries(features)) {
    const gradient = gradientBase * fval;
    const oldW = model.featureWeights[fname] ?? 0.0;
    model.featureWeights[fname] = oldW - lr * gradient;
  }

  // Update intercept
  model.intercept = model.intercept - lr * gradientBase;

  // Update metadata
  model.nUpdates++;
  const acc = 1.0 - Math.abs(error);
  model.recentAccuracy = Math.round(
    (0.9 * model.recentAccuracy + 0.1 * acc) * 10000
  ) / 10000;

  return { prediction, error: Math.abs(error) };
}

// ------------------------------------------------------------------
// 6. Surprise = |predicted - actual|
// ------------------------------------------------------------------

/**
 * Compute surprise: |predicted_outcome - actual_outcome|.
 *
 * High surprise → the model's understanding was wrong → high learning value.
 * Surprise-weighted retrieval boosts memories that were surprising.
 */
export function computeSurprise(
  predicted: number,
  actualOutcome: "success" | "failure" | "crash",
): number {
  const actual = actualOutcome === "success" ? 1.0
    : actualOutcome === "crash" ? 0.0
    : 0.5;
  return Math.round(Math.abs(predicted - actual) * 10000) / 10000;
}

// ------------------------------------------------------------------
// 7. Counterfactual synthesis
// ------------------------------------------------------------------

/**
 * Synthesize counterfactual insight for an execution.
 *
 * Uses heuristics (no extra LLM call) to construct "what if" reasoning.
 * This gives the planner causal leverage — not just "X worked" but
 * "X worked BECAUSE... and here's what would happen if we didn't do it."
 */
export function synthesizeCounterfactual(
  record: ExecutionRecord,
  outcome: "success" | "failure" | "crash",
  qualityScore: number,
): string {
  const taskType = record.taskType;
  const coverage = record.coverage
    ? Object.entries(record.coverage).map(([k, v]) => k).slice(0, 2).join(", ")
    : "unknown domain";

  if (outcome === "crash") {
    return [
      `Counterfactual: If the approach for '${taskType}' had been applied`,
      `more conservatively (smaller scope, gradual rollout), it might have`,
      `avoided the failure. Consider a minimal version of the same idea`,
      `targeting ${coverage}.`,
    ].join(" ");
  }

  if (outcome === "success") {
    const compStr = record.compressionRatio > 0.3
      ? "high compression"
      : "broad context";
    return [
      `Counterfactual: Without the ${compStr} strategy for '${taskType}',`,
      `quality would likely be lower by an estimated margin.`,
      `If combined with a complementary change in an orthogonal quality`,
      `dimension, the gains might compound. Consider cross-dimension combinations.`,
    ].join(" ");
  }

  // failure
  return [
    `Counterfactual: The approach to '${taskType}' didn't achieve target quality`,
    `(${(qualityScore * 100).toFixed(0)}%), but a variant with different compression`,
    `weights or quality thresholds might. The negative result rules out this`,
    `SPECIFIC configuration, not the direction.`,
  ].join(" ");
}

// ------------------------------------------------------------------
// 8. Curriculum learning
// ------------------------------------------------------------------

/**
 * Determine current curriculum phase and return phase-specific parameters.
 *
 * Phase 0 (explore):   Exps  1-10  → High diversity, wide MMR, large mutations
 * Phase 1 (focus):     Exps 11-30  → Exploit promising branches, narrower MMR
 * Phase 2 (principled): Exps 31-60 → Fine-tune based on learned principles
 * Phase 3 (adversarial): Exps 61+  → Challenge assumptions, verify old results
 */
export function getCurriculumPhase(
  totalExperiments: number,
): { phase: CurriculumPhase; params: import("../types.js").CurriculumPhaseParams } {
  const boundaries = [10, 30, 60];
  let phase: CurriculumPhase = 0;
  if (totalExperiments >= boundaries[2]) phase = 3;
  else if (totalExperiments >= boundaries[1]) phase = 2;
  else if (totalExperiments >= boundaries[0]) phase = 1;

  const params = DEFAULT_CURRICULUM.phaseParams[phase];
  return { phase, params };
}

/**
 * v3.9: Get adaptive curriculum parameters, adjusted by runtime metrics.
 *
 * Karpathy principle: "Build abstractions from experience."
 * Rather than using static per-phase params, this adjusts based on
 * actual branch performance signals — velocity, novelty, success rate.
 *
 * Adjustments:
 *   - mmrLambda ↑ when improving (exploit), ↓ when declining (explore)
 *   - explorationBonus ↓ when successRate > 0.8 (don't fix what's working)
 *   - curiosityWeight ↑ when novelty < 0.2 (force diversification)
 *   - mutationMagnitude ↑ when plateaued, ↓ when improving
 *   - adversarialInterval ↓ in later phases (verify more often)
 */
export function getAdaptiveCurriculumParams(
  totalExperiments: number,
  metrics?: {
    velocity?: number;
    novelty?: number;
    successRate?: number;
    isPlateaued?: boolean;
  },
): ReturnType<typeof getCurriculumPhase> & { adjusted: Record<string, number> } {
  const base = getCurriculumPhase(totalExperiments);
  const p = { ...base.params };
  const adjustments: Record<string, number> = {};

  if (metrics) {
    // MMR lambda: exploit when improving, explore when declining
    if (metrics.velocity !== undefined) {
      if (metrics.velocity > 0.001) {
        p.mmrLambda = Math.min(0.90, p.mmrLambda + 0.15);
        adjustments.mmrLambda = +0.15;
      } else if (metrics.velocity < -0.0005) {
        p.mmrLambda = Math.max(0.20, p.mmrLambda - 0.20);
        adjustments.mmrLambda = -0.20;
      }
    }

    // Reduce exploration when consistently successful
    if (metrics.successRate !== undefined && metrics.successRate > 0.8) {
      p.explorationBonus = Math.max(0.3, p.explorationBonus * 0.7);
      adjustments.explorationBonus = Math.round((p.explorationBonus - base.params.explorationBonus) * 100) / 100;
    }

    // Force diversification when novelty collapses
    if (metrics.novelty !== undefined && metrics.novelty < 0.2) {
      p.curiosityWeight = Math.min(2.0, p.curiosityWeight * 1.5);
      adjustments.curiosityWeight = Math.round((p.curiosityWeight - base.params.curiosityWeight) * 100) / 100;
    }

    // Larger mutations when stuck, smaller when improving
    if (metrics.isPlateaued) {
      p.mutationMagnitude = Math.min(0.40, p.mutationMagnitude * 1.5);
      adjustments.mutationMagnitude = Math.round((p.mutationMagnitude - base.params.mutationMagnitude) * 100) / 100;
    } else if (metrics.velocity !== undefined && metrics.velocity > 0.001) {
      p.mutationMagnitude = Math.max(0.05, p.mutationMagnitude * 0.8);
      adjustments.mutationMagnitude = Math.round((p.mutationMagnitude - base.params.mutationMagnitude) * 100) / 100;
    }
  }

  return { phase: base.phase, params: p, adjusted: adjustments };
}

// ------------------------------------------------------------------
// 9. Adversarial memory verification
// ------------------------------------------------------------------

/**
 * Adversarially re-evaluate old "success" records against current knowledge.
 *
 * A memory deemed successful early on might be merely average by later standards.
 * This prevents "success inflation" where early lucky results are given
 * undue weight in retrieval.
 *
 * Returns the number of records verified.
 */
export function adversarialVerify(
  records: RLExecutionRecord[],
  maxToVerify: number = 3,
  minAge: number = 10,
): number {
  const n = records.length;
  if (n < minAge + 5) return 0;

  // Compute current baseline
  const successes = records.filter(
    r => (r.causalUtility ?? 0.5) > 0.5 && r.qualityScore >= 0.85
  );
  if (successes.length < 5) return 0;

  const currentAvg = successes.reduce((s, r) => s + r.qualityScore, 0) / successes.length;

  // Find old unverified successes
  let verified = 0;
  for (let i = 0; i < Math.min(records.length - minAge, records.length); i++) {
    const rec = records[i];
    if (rec.qualityScore < 0.85) continue;
    const age = n - 1 - i;
    if (age < minAge) continue;

    // Check if verified recently
    const lastV = rec.verificationHistory[rec.verificationHistory.length - 1];
    if (lastV && n - lastV.experimentCount < minAge) continue;

    // Adversarial scoring: how much worse than current avg?
    const avgGap = (rec.qualityScore - currentAvg) / Math.max(Math.abs(currentAvg), 0.001);

    if (avgGap < -0.02) {
      // Significantly worse than current average → downgrade all utility signals
      // agent.py v4 lines 2136-2144: confidence, retrieval_utility, alpha_ts all downgraded
      rec.causalUtility = Math.max(0.2, (rec.causalUtility ?? 0.5) * 0.8);
      rec.thompsonAlpha = Math.max(0.5, rec.thompsonAlpha * 0.7);
      // v4.0: Also downgrade retrieval utility for obsolete successes
      rec.retrievalUtility = Math.max(0.1, (rec.retrievalUtility ?? 0.5) * 0.75);
    } else if (avgGap < 0) {
      // Mildly below average — moderate downgrade
      rec.causalUtility = Math.max(0.3, (rec.causalUtility ?? 0.5) * 0.9);
      rec.retrievalUtility = Math.max(0.2, (rec.retrievalUtility ?? 0.5) * 0.85);
    } else {
      // Still competitive: boost (adversarial test PASSED)
      // agent.py v4 lines 2149-2152: confidence boosted for surviving memories
      rec.causalUtility = Math.min(0.95, (rec.causalUtility ?? 0.5) * 1.05);
      rec.thompsonAlpha = Math.min(10, rec.thompsonAlpha * 1.1);  // v4.0: boost Thompson α
      rec.retrievalUtility = Math.min(0.95, (rec.retrievalUtility ?? 0.5) * 1.05);  // v4.0: boost retrieval utility
    }

    rec.verificationHistory.push({
      experimentCount: n,
      currentBest: successes.reduce((min, r) => Math.min(min, r.qualityScore), 1),
      currentAvg,
      gapToBest: Math.round(avgGap * 10000) / 10000,
      newConfidence: rec.causalUtility ?? 0.5,
      timestamp: new Date().toISOString(),
    });
    if (rec.verificationHistory.length > 10) {
      rec.verificationHistory = rec.verificationHistory.slice(-10);
    }
    verified++;
    if (verified >= maxToVerify) break;
  }

  return verified;
}

// ------------------------------------------------------------------
// 10. Memory consolidation with attribution
// ------------------------------------------------------------------

/**
 * v4.0: Consolidate old low-utility records into summary entries with attribution.
 *
 * When the record pool exceeds maxRecords, merge old low-utility records
 * into a single insight entry. This frees memory while preserving
 * the key lesson from each cluster.
 *
 * v4 enhancement (agent.py _maybe_consolidate_memories_v4):
 *   - Tracks token savings per group
 *   - Records success/failure breakdown per group
 *   - Returns structured attribution for audit trail
 *
 * Returns { consolidatedCount, tokensSaved, groups }.
 */
export function consolidateMemories(
  records: RLExecutionRecord[],
  maxRecords: number = 60,
  consolidateOlderThan: number = 20,
  nextIdFn: () => number,
): {
  consolidatedCount: number;
  tokensSaved: number;
  groups: Array<{
    subsystem: string;
    sources: number;
    tokensSaved: number;
    successes: number;
    failures: number;
  }>;
} {
  const result = {
    consolidatedCount: 0,
    tokensSaved: 0,
    groups: [] as Array<{
      subsystem: string;
      sources: number;
      tokensSaved: number;
      successes: number;
      failures: number;
    }>,
  };

  if (records.length <= maxRecords) {
    return result;
  }

  const cutoff = records.length - consolidateOlderThan;
  const candidates: Array<{ idx: number; rec: RLExecutionRecord }> = [];

  for (let i = 0; i < cutoff; i++) {
    const rec = records[i];
    if ((rec.causalUtility ?? 0.5) < 0.4 && !rec.consolidated) {
      candidates.push({ idx: i, rec });
    }
  }

  if (candidates.length < 5) return result;

  // Group by task type for coherent summaries
  const groups = new Map<string, Array<{ idx: number; rec: RLExecutionRecord }>>();
  for (const c of candidates) {
    const key = c.rec.taskType;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(c);
  }

  let consolidatedCount = 0;
  let totalTokensSaved = 0;

  for (const [taskType, group] of groups) {
    if (group.length < 3) continue;

    // Estimate token savings
    const tokensBefore = group.reduce((sum, { rec }) =>
      sum + Math.ceil(rec.taskId.length / 4), 0
    );
    const tokensSaved = Math.floor(tokensBefore * 0.75);
    totalTokensSaved += tokensSaved;

    // Count outcomes in this group
    const successes = group.filter(({ rec }) =>
      rec.qualityScore >= 0.85
    ).length;
    const failures = group.length - successes;

    // Mark originals as consolidated
    const mergedId = `consolidated_${nextIdFn()}`;
    for (const { rec } of group) {
      rec.consolidated = true;
      rec.consolidatedInto = mergedId;
    }

    // v4.0: Record attribution for audit trail
    result.groups.push({
      subsystem: taskType,
      sources: group.length,
      tokensSaved,
      successes,
      failures,
    });

    consolidatedCount += group.length;
  }

  result.consolidatedCount = consolidatedCount;
  result.tokensSaved = totalTokensSaved;
  return result;
}

// ------------------------------------------------------------------
// 11. UCB-guided dimension selection
// ------------------------------------------------------------------

/**
 * Select a retrieval dimension to mutate using Upper Confidence Bound (UCB).
 *
 * UCB(dim) = avg_reward(dim) + c * sqrt(log(N) / n(dim))
 *
 * This naturally balances:
 *   - Exploit: pick dimensions with high historical reward
 *   - Explore: pick dimensions with few trials (wide confidence interval)
 *
 * Replaces random selection with principled bandit optimization.
 */
export function ucbSelectDimension(
  dimCounts: Record<string, number>,
  dimRewards: Record<string, number>,
  totalMutations: number,
  allDims: string[],
  c: number = 1.5,
): string {
  const N = Math.max(totalMutations, 1);
  let bestDim = allDims[0];
  let bestUcb = -Infinity;

  for (const dim of allDims) {
    const n = Math.max(dimCounts[dim] ?? 0, 0.1);
    const avgReward = (dimRewards[dim] ?? 0) / n;
    const explorationBonus = c * Math.sqrt(Math.log(N + 1) / n);
    const ucb = avgReward + explorationBonus;

    if (ucb > bestUcb) {
      bestUcb = ucb;
      bestDim = dim;
    }
  }

  return bestDim;
}

/**
 * Record the outcome of a dimension mutation for UCB tracking.
 */
export function recordUCBOutcome(
  dimCounts: Record<string, number>,
  dimRewards: Record<string, number>,
  dim: string,
  reward: number,
): { totalMutations: number } {
  dimCounts[dim] = (dimCounts[dim] || 0) + 1;
  dimRewards[dim] = (dimRewards[dim] || 0) + reward;
  const totalMutations = Object.values(dimCounts).reduce((a, b) => a + b, 0);
  return { totalMutations };
}

// ------------------------------------------------------------------
// 12. Outcome-to-reward mapping
// ------------------------------------------------------------------

/**
 * Map experiment outcome to a scalar reward signal.
 *
 * Success → positive reward scaled by improvement magnitude
 * Failure → mild negative
 * Crash   → strong negative
 */
export function outcomeToReward(
  outcome: "success" | "failure" | "crash",
  qualityScore: number,
  bestQuality: number,
): { signal: number; magnitude: number } {
  if (outcome === "crash") {
    return { signal: -0.5, magnitude: 0.0 };
  }
  if (outcome === "failure") {
    return { signal: -0.15, magnitude: 0.15 };
  }
  // success
  const improvement = Math.max(0, qualityScore - bestQuality);
  const magnitude = Math.min(1.0, improvement / 0.05); // scale: 5% improvement = full
  const signal = 0.2 + 0.8 * magnitude;
  return { signal, magnitude: 0.2 + 0.8 * magnitude };
}

// ------------------------------------------------------------------
// 13. Curiosity/EIG bonus
// ------------------------------------------------------------------

/**
 * Compute curiosity/Expected Information Gain bonus for a record.
 *
 * Rewards memories from task types where:
 *   1. The predictive model is uncertain (few examples → high variance)
 *   2. The task type has been under-explored relative to others
 *   3. Past experiments in this area had high surprise
 *
 * Returns bonus in [0, 5].
 */
export function curiosityBonus(
  record: RLExecutionRecord,
  allRecords: RLExecutionRecord[],
): number {
  const family = record.taskType;
  const n = allRecords.length;

  // Count experiments per task type
  const familyCounts = new Map<string, number>();
  for (const r of allRecords) {
    familyCounts.set(r.taskType, (familyCounts.get(r.taskType) || 0) + 1);
  }

  // Novelty: fewer experiments → higher curiosity
  const nFamily = familyCounts.get(family) || 0;
  const maxFamily = Math.max(...familyCounts.values(), 1);
  const noveltyBonus = Math.max(0, 1.0 - nFamily / maxFamily);

  // Uncertainty: average prediction error in this family
  const familyErrors: number[] = [];
  for (const r of allRecords) {
    if (r.taskType === family && r.predictionError != null) {
      familyErrors.push(r.predictionError);
    }
  }
  const avgError = familyErrors.length > 0
    ? familyErrors.reduce((a, b) => a + b, 0) / familyErrors.length
    : 0.5;

  // Surprise in this family
  const familySurprises: number[] = [];
  for (const r of allRecords) {
    if (r.taskType === family) {
      familySurprises.push(r.surpriseScore ?? 0.5);
    }
  }
  const avgSurprise = familySurprises.length > 0
    ? familySurprises.reduce((a, b) => a + b, 0) / familySurprises.length
    : 0.5;

  // Composite curiosity bonus [0, 5]
  const bonus = noveltyBonus * 2.0 + avgError * 1.5 + (avgSurprise - 0.5) * 1.5;
  return Math.round(Math.max(0.0, Math.min(5.0, bonus)) * 10000) / 10000;
}

// ------------------------------------------------------------------
// 14. Retrieval strategy mutation (self-evolution of retrieval)
// ------------------------------------------------------------------

/**
 * Propose a mutation to retrieval strategy hyperparameters.
 *
 * Uses UCB to select which dimension to mutate, then applies
 * a Gaussian perturbation to the current value.
 *
 * Returns the mutated strategy state and mutation descriptor.
 */
export function proposeRetrievalMutation(
  strategy: RetrievalStrategyState,
): { strategy: RetrievalStrategyState; mutation: Record<string, unknown> } | null {
  // Don't mutate if already pending
  if (strategy.pendingMutation) return null;

  // Need enough trials in current generation
  const minTrials = Math.max(3, Math.floor(strategy.generation * 0.5 + 2));
  if (strategy.trialsInGeneration < minTrials) return null;

  const magnitude = strategy.mutationMagnitude;

  // Exploration burst: if fitness plateaued, boost magnitude
  const fh = strategy.fitnessHistory;
  if (fh.length >= 3) {
    const recent = fh.slice(-3).map(f => f.fitness);
    if (Math.max(...recent) <= (fh[fh.length - 3].fitness + 0.02)) {
      strategy.mutationMagnitude = Math.min(0.40, magnitude * 1.5);
    }
  }

  const newStrategy = { ...strategy };
  const allDims = Object.keys(strategy.dimWeights);
  const roll = Math.random();

  let mutation: Record<string, unknown>;

  if (roll < 0.70) {
    // Mutate a dimension weight (UCB-selected)
    const dim = ucbSelectDimension(
      strategy.ucbDimCounts,
      strategy.ucbDimRewards,
      strategy.ucbTotalMutations,
      allDims,
    );
    const oldVal = strategy.dimWeights[dim];
    const multiplier = Math.exp(normalRandom() * 0.1 * magnitude);
    const newVal = Math.round(Math.max(0.25, Math.min(4.0, oldVal * multiplier)) * 1000) / 1000;
    newStrategy.dimWeights = { ...strategy.dimWeights, [dim]: newVal };
    mutation = { target: `dim_weight.${dim}`, old: oldVal, new: newVal, _ucb_dim: dim };
  } else if (roll < 0.85) {
    // Mutate MMR lambda
    const oldVal = strategy.mmrLambda;
    const delta = normalRandom() * 0.05 * magnitude;
    const newVal = Math.round(Math.max(0.20, Math.min(0.95, oldVal + delta)) * 1000) / 1000;
    newStrategy.mmrLambda = newVal;
    mutation = { target: "mmr_lambda", old: oldVal, new: newVal };
  } else if (roll < 0.95) {
    // Mutate token budget tier
    const tierIdx = Math.floor(Math.random() * 3);
    const oldVal = strategy.tokenBudgetTiers[tierIdx];
    const delta = Math.round(normalRandom() * oldVal * magnitude * 0.5);
    const newVal = Math.max(400, Math.min(4000, oldVal + delta));
    const newTiers = [...strategy.tokenBudgetTiers] as [number, number, number];
    newTiers[tierIdx] = newVal;
    newStrategy.tokenBudgetTiers = newTiers;
    mutation = { target: `token_budget_tier.${tierIdx}`, old: oldVal, new: newVal };
  } else {
    // Mutate top_k
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

  return { strategy: newStrategy, mutation };
}

/**
 * Decide whether to keep or revert a retrieval strategy mutation.
 */
export function decideRetrievalMutation(
  strategy: RetrievalStrategyState,
): { decision: "keep" | "revert"; delta: number } | null {
  const pm = strategy.pendingMutation;
  if (!pm) return null;

  // Need enough trials
  const minTrials = typeof pm.old === "number" && typeof pm.new === "number"
    && Math.abs((pm.old as number) - (pm.new as number)) / Math.max(Math.abs(pm.old as number), 0.01) > 0.2
    ? 4 : 6;
  if (strategy.trialsInGeneration < minTrials) return null;

  const delta = strategy.fitness - strategy.ancestorFitness;

  let decision: "keep" | "revert";
  if (delta > 0.03) {
    decision = "keep";
    strategy.mutationMagnitude = Math.max(0.05, strategy.mutationMagnitude * 0.8);
  } else if (delta < -0.05) {
    decision = "revert";
    // Revert the parameter
    const target = pm.target as string;
    if (target.startsWith("dim_weight.")) {
      const dim = target.split(".").slice(1).join(".");
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
    decision = "keep"; // neutral
  }

  // Record UCB outcome
  const ucbDim = pm._ucb_dim as string | undefined;
  if (ucbDim) {
    const { totalMutations } = recordUCBOutcome(
      strategy.ucbDimCounts, strategy.ucbDimRewards, ucbDim, delta
    );
    strategy.ucbTotalMutations = totalMutations;
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
 * Record a trial outcome for retrieval strategy fitness tracking.
 */
export function recordRetrievalTrial(
  strategy: RetrievalStrategyState,
  outcome: "success" | "failure" | "crash",
  qualityScore: number,
  bestQuality: number,
): RetrievalStrategyState {
  const newStrategy = { ...strategy };
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