// ============================================================================
// Turbocontext v5 — Value Function + TD(λ)
// ============================================================================
//
// Tracks expected quality per task type via EMA baselines.
// TD(λ) assigns credit to memories that contributed to outcomes
// via eligibility traces — memories retrieved recently get partial
// credit for the TD error, decayed exponentially by γ·λ.
import type {
  ValueFunctionState, IndexedMemory, TaskTypeBaseline,
  Trial, TaskType,
} from "../types.js";
import { CLAMP_ADVANTAGE, MIN_TRACE, EPSILON } from "../constants.js";

// ── Baseline updates ──

/**
 * updateBaseline: updates the EMA baseline for a single task type.
 *
 * Algorithm:
 *   baseline.ema += alpha * (qualityScore - baseline.ema)
 *   baseline.mean = (mean * count + score) / (count + 1)
 *   baseline.count += 1
 *   Append score to recentScores (ring buffer, max 10)
 *   Recompute slope via linear regression
 *
 * RL theory: the EMA baseline serves as the "critic" in actor-critic
 * methods. The advantage (quality - baseline) tells us whether an action
 * was better or worse than expected for this task type.
 *
 * @param alpha Learning rate for EMA update (default 0.1)
 */
export function updateBaseline(
  baseline: TaskTypeBaseline,
  qualityScore: number,
  alpha = 0.1,
): TaskTypeBaseline {
  const newEma = baseline.ema + alpha * (qualityScore - baseline.ema);
  const newCount = baseline.count + 1;
  const newMean = (baseline.mean * baseline.count + qualityScore) / newCount;

  const recentScores = [...baseline.recentScores, qualityScore];
  if (recentScores.length > 10) recentScores.shift();

  const slope = computeSlope(recentScores);

  return { mean: newMean, ema: newEma, count: newCount, recentScores, slope };
}

/**
 * updateAllBaselines: updates both the task-type-specific baseline and
 * the global baseline after a trial.
 */
export function updateAllBaselines(
  vf: ValueFunctionState,
  taskType: TaskType,
  qualityScore: number,
): ValueFunctionState {
  const baselines = { ...vf.baselines };
  // v5.1: Guard — lazily initialize baseline for task types not in the canonical enum
  let bl = baselines[taskType];
  if (!bl) {
    bl = { mean: 0, ema: 0.5, count: 0, recentScores: [], slope: 0 };
  }
  baselines[taskType] = updateBaseline(bl, qualityScore);

  const globalBaseline = vf.globalBaseline + 0.1 * (qualityScore - vf.globalBaseline);

  return { ...vf, baselines, globalBaseline };
}

/** Create a fresh baseline entry for a newly-seen task type */
export function ensureBaseline(): { mean: number; ema: number; count: number; recentScores: number[]; slope: number } {
  return { mean: 0, ema: 0.5, count: 0, recentScores: [], slope: 0 };
}

// ── Plateau detection ──

/**
 * detectPlateau: returns true if the task-type baseline appears stuck.
 *
 * Criteria: |slope| < 0.005 AND at least 5 recent scores exist.
 * This indicates the policy has stopped improving for this task type.
 */
export function detectPlateau(baseline: TaskTypeBaseline): boolean {
  return baseline.recentScores.length >= 5 && Math.abs(baseline.slope) < 0.005;
}

/**
 * escapePlateau: increases the effective learning rate by 50% to
 * help the baseline escape a local minimum.
 */
export function escapePlateau(baseline: TaskTypeBaseline): TaskTypeBaseline {
  // Signal is read by the curriculum system — the baseline itself doesn't change
  return { ...baseline, slope: baseline.slope * 1.5 };
}

// ── TD(λ) trace operations ──

/**
 * decayTraces: decays all eligibility traces by γ·λ.
 *
 * Theory (Sutton 1988): eligibility traces bridge the gap between
 * TD(0) (one-step lookahead) and Monte Carlo (full episode).
 * λ controls the decay rate — λ=0 is TD(0), λ=1 is Monte Carlo.
 *
 * Traces below MIN_TRACE (0.001) are pruned to prevent accumulation.
 */
export function decayTraces(vf: ValueFunctionState): ValueFunctionState {
  const traces: Record<string, number> = {};
  const { gamma, lambda } = vf.td;

  for (const [key, trace] of Object.entries(vf.traces)) {
    const decayed = trace * gamma * lambda;
    if (decayed >= MIN_TRACE) {
      traces[key] = decayed;
    }
  }

  return { ...vf, traces };
}

/**
 * bumpTraces: increments eligibility traces for memories involved in this trial.
 *
 * Referenced memories (planner actually used them): bump += 1.0 (full credit)
 * Retrieved-but-not-referenced: bump += 0.5 (partial credit — they were
 *   available but the planner chose not to use them significantly).
 */
export function bumpTraces(
  vf: ValueFunctionState,
  referencedIds: string[],
  retrievedIds: string[],
): ValueFunctionState {
  const traces = { ...vf.traces };
  const refSet = new Set(referencedIds);

  for (const id of referencedIds) {
    traces[id] = (traces[id] || 0) + 1.0;
  }
  for (const id of retrievedIds) {
    if (!refSet.has(id)) {
      traces[id] = (traces[id] || 0) + 0.5;
    }
  }

  return { ...vf, traces };
}

/**
 * applyTDUpdate: applies the TD(λ) update through the entire trace chain.
 *
 * Algorithm:
 *   For each memory with an active eligibility trace:
 *     update = td.alpha * trace * tdError
 *     memory.causalUtility += update
 *     memory.causalUtility = clamp(memory.causalUtility, 0, 1)
 *
 * RL theory: this is the "credit assignment" step. Memories that were
 * retrieved before a successful outcome get their causal utility
 * increased proportionally to how recently they were retrieved
 * (encoded in the trace value).
 *
 * @returns Updated value function AND a map of memory ID → partial patch
 *   to apply to each affected memory.
 */
export function applyTDUpdate(
  vf: ValueFunctionState,
  memories: IndexedMemory[],
  tdError: number,
): { vf: ValueFunctionState; memoryPatches: Map<string, Partial<IndexedMemory>> } {
  const memoryMap = new Map(memories.map(m => [m.id, m]));
  const patches = new Map<string, Partial<IndexedMemory>>();
  const traces = { ...vf.traces };
  const memoryPriorities = { ...vf.memoryPriorities };
  let maxPriority = vf.maxPriority;

  for (const [memId, trace] of Object.entries(vf.traces)) {
    const memory = memoryMap.get(memId);
    if (!memory) continue;

    const update = vf.td.alpha * trace * tdError;
    const newCausalUtility = clamp(memory.causalUtility + update, 0, 1);
    const newTdError = tdError;
    const priority = Math.abs(tdError) + EPSILON;

    patches.set(memId, { causalUtility: newCausalUtility, tdError: newTdError });
    memoryPriorities[memId] = priority;
    if (priority > maxPriority) maxPriority = priority;
  }

  return {
    vf: { ...vf, traces, memoryPriorities, maxPriority },
    memoryPatches: patches,
  };
}

// ── Composite reward ──

/**
 * compositeReward: converts trial outcome + quality into a scalar reward.
 *
 *   success → qualityScore              (0.75–1.0)
 *   failure → qualityScore * 0.3        (partial credit for effort)
 *   crash   → -0.1                      (small penalty — informative failure)
 *
 * RL theory: the reward function shapes what the policy optimizes. We reward
 * quality heavily for success, give partial credit for failures (HER handles
 * the rest), and apply a small penalty for crashes so the system learns
 * to avoid unsafe parameter combinations.
 */
export function compositeReward(outcome: string, qualityScore: number): number {
  switch (outcome) {
    case "success": return qualityScore;
    case "failure": return qualityScore * 0.3;
    case "crash":   return -0.1;
    default:        return 0;
  }
}

// ── Advantage ──

/**
 * computeAdvantage: advantage = qualityScore - baseline.ema
 *
 * Clamped to [-0.5, 0.5] to prevent extreme updates from outliers.
 * Positive advantage → action was better than expected.
 * Negative advantage → action was worse than expected.
 */
export function computeAdvantage(
  baseline: TaskTypeBaseline,
  qualityScore: number,
): number {
  return clamp(qualityScore - baseline.ema, CLAMP_ADVANTAGE[0], CLAMP_ADVANTAGE[1]);
}

/**
 * advantageMultiplier: 1.0 + advantage, range [0.5, 1.5].
 *
 * Maps advantage to a multiplicative factor for causal utility updates.
 * Highly positive advantage → strong boost to contributing memories.
 * Negative advantage → demotion of contributing memories.
 */
export function advantageMultiplier(advantage: number): number {
  return 1.0 + advantage; // range [0.5, 1.5] after clamping
}

// ── Slope computation ──

/**
 * computeSlope: linear regression slope of scores over index.
 *
 * y = a * x + b, returns a.
 * Positive → improving. Negative → declining. Near zero → plateaued.
 */
export function computeSlope(scores: number[]): number {
  const n = scores.length;
  if (n < 2) return 0;
  const xMean = (n - 1) / 2;
  const yMean = scores.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (i - xMean) * (scores[i] - yMean);
    den += (i - xMean) ** 2;
  }
  return den === 0 ? 0 : num / den;
}

// ── PER priorities ──

/**
 * updateMemoryPriorities: refreshes PER priorities from new TD errors.
 *
 * priority = |tdError| + EPSILON
 * maxPriority tracks the global maximum for importance-sampling normalization.
 */
export function updateMemoryPriorities(
  vf: ValueFunctionState,
  memoryPatches: Map<string, Partial<IndexedMemory>>,
): ValueFunctionState {
  const priorities = { ...vf.memoryPriorities };
  let maxP = vf.maxPriority;

  for (const [memId, patch] of memoryPatches) {
    const tdError = patch.tdError ?? 0;
    const priority = Math.abs(tdError) + EPSILON;
    priorities[memId] = priority;
    if (priority > maxP) maxP = priority;
  }

  return { ...vf, memoryPriorities: priorities, maxPriority: maxP };
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}
