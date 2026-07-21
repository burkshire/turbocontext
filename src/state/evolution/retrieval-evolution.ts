// ============================================================================
// Turbocontext v5 — Retrieval Strategy Self-Evolution
// ============================================================================
//
// Self-evolution of retrieval hyperparameters via UCB-guided log-normal
// mutation proposals. The same proposeMutation → testTrials → decideKeepDiscard
// loop used for task strategies is applied to the retrieval algorithm itself.
//
// THREE-BRANCH LOGIC:
//   1. No pending mutation: propose one via UCB, clone ancestor, start testing
//   2. Pending + not enough trials (< 4): keep testing (return no_mutation)
//   3. Pending + enough trials (>= 4): compute fitness delta, decide keep/revert
//
// RL theory: this is a form of meta-learning — the system learns how to
// learn by evolving the parameters that control experience retrieval.
import type {
  SharedStateV5, EvolutionResult,
  RetrievalStrategyState, StrategyExperience, TaskType,
} from "../types.js";
import { EvolutionDecision, MutationDirection } from "../types.js";
import {
  FITNESS_KEEP_THRESHOLD, FITNESS_REVERT_THRESHOLD,
  TUNABLE_PARAMS, PARAM_BOUNDS,
} from "../constants.js";

// ── Main evolution step ──

/**
 * evolveRetrievalStrategy: executes one evolution cycle.
 *
 * Returns { newState, result } — state mutated in-place plus a result
 * describing what happened and why.
 */
export function evolveRetrievalStrategy(
  state: SharedStateV5,
): { newState: SharedStateV5; result: EvolutionResult } {
  const rs = state.retrievalStrategy;

  // Branch 1: No pending mutation → propose one
  if (!rs.pendingMutation) {
    const param = selectParameterToTune(state);
    const oldValue = getParamValue(state.policy, param);
    const mutationMag = state.curriculum.phases[state.totalInvocations < 10 ? 0 : state.totalInvocations < 30 ? 1 : state.totalInvocations < 60 ? 2 : 3].mutationMagnitude;

    // Karpathy-inspired: plateau-adaptive mutation magnitude.
    // If fitness hasn't improved in 3 generations, boost exploration.
    // If fitness is improving steadily, reduce magnitude for fine-tuning.
    let adaptiveMag = mutationMag;
    const expLib = rs.experienceLibrary;
    if (expLib.length >= 3) {
      const recentDeltas = expLib.slice(-3).map(e => e.fitnessDelta);
      const maxRecentDelta = Math.max(...recentDeltas);
      if (maxRecentDelta <= 0.02) {
        // Plateaued — boost exploration to escape local optimum
        adaptiveMag = Math.min(0.40, mutationMag * 1.5);
      } else if (maxRecentDelta > 0.05) {
        // Improving — reduce magnitude for fine-tuning
        adaptiveMag = Math.max(0.05, mutationMag * 0.8);
      }
    }
    const actualNewValue = applyMutation(oldValue, param, adaptiveMag);

    // Clone current strategy as ancestor
    rs.ancestor = { ...rs.active };
    rs.ancestorFitness = computeStrategyFitness(state);
    rs.pendingMutation = { targetParam: param, oldValue, newValue: actualNewValue };
    rs.trialsInGeneration = 0;

    return {
      newState: state,
      result: {
        generation: rs.generation,
        mutation: { param, oldValue, newValue: actualNewValue },
        fitnessDelta: null,
        decision: EvolutionDecision.NO_MUTATION,
      },
    };
  }

  // Branch 2: Pending + still testing (fewer than 4 trials in this generation)
  if (rs.trialsInGeneration < 4) {
    return {
      newState: state,
      result: {
        generation: rs.generation,
        mutation: rs.pendingMutation ? {
          param: rs.pendingMutation.targetParam,
          oldValue: rs.pendingMutation.oldValue,
          newValue: rs.pendingMutation.newValue,
        } : null,
        fitnessDelta: null,
        decision: EvolutionDecision.NO_MUTATION,
      },
    };
  }

  // Branch 3: Pending + enough trials → decide
  const fitnessAfter = computeStrategyFitness(state);
  const fitnessBefore = rs.ancestorFitness;
  const delta = fitnessAfter - fitnessBefore;

  let decision: EvolutionDecision;
  if (delta > FITNESS_KEEP_THRESHOLD) {
    decision = EvolutionDecision.KEEP;
    // Apply the mutation to the active strategy
    setParamValue(state.policy, rs.pendingMutation.targetParam, rs.pendingMutation.newValue);
  } else if (delta < FITNESS_REVERT_THRESHOLD) {
    decision = EvolutionDecision.REVERT;
    // Restore ancestor (revert the mutation)
    if (rs.ancestor) {
      rs.active = { ...rs.ancestor };
    }
  } else {
    decision = EvolutionDecision.KEEP; // neutral: keep but not strongly beneficial
    setParamValue(state.policy, rs.pendingMutation.targetParam, rs.pendingMutation.newValue);
  }

  // Record experience
  const mutation = rs.pendingMutation;
  const scenario = generateScenario(state);
  const experience: StrategyExperience = {
    scenario,
    mutation: {
      param: mutation.targetParam,
      direction: mutation.newValue > mutation.oldValue ? MutationDirection.INCREASE : MutationDirection.DECREASE,
    },
    fitnessDelta: delta,
    decision,
  };
  rs.experienceLibrary.push(experience);

  // Log evolution entry
  const entry = {
    timestamp: new Date().toISOString(),
    generation: rs.generation,
    mutation: { param: mutation.targetParam, oldValue: mutation.oldValue, newValue: mutation.newValue },
    fitnessBefore,
    fitnessAfter,
    delta,
    decision,
    scenario,
  };
  state.evolutionLog.push(entry);

  // Reset for next generation
  rs.pendingMutation = null;
  rs.generation += 1;
  rs.trialsInGeneration = 0;

  return {
    newState: state,
    result: {
      generation: rs.generation - 1,
      mutation: { param: mutation.targetParam, oldValue: mutation.oldValue, newValue: mutation.newValue },
      fitnessDelta: delta,
      decision,
    },
  };
}

/**
 * incrementGenerationTrials: called after each trial when a mutation is pending.
 * Simply increments the counter.
 */
export function incrementGenerationTrials(state: SharedStateV5): void {
  if (state.retrievalStrategy.pendingMutation) {
    state.retrievalStrategy.trialsInGeneration += 1;
  }
}

// ── Fitness computation ──

/**
 * computeStrategyFitness: average quality over recent trials.
 *
 * If no trials in this generation, returns the global baseline EMA.
 */
export function computeStrategyFitness(state: SharedStateV5): number {
  if (state.totalInvocations === 0) return 0.5;

  // Use last N trials filtered by current generation to avoid conflation across strategies
  const gen = state.retrievalStrategy.generation;
  const recentTrials = state.trials
    .filter(t => t.generation === undefined || t.generation === gen)
    .slice(-10);
  if (recentTrials.length === 0) return state.valueFunction.globalBaseline;

  return recentTrials.reduce((sum, t) => sum + t.qualityScore, 0) / recentTrials.length;
}

// ── Mutation ──

/**
 * applyMutation: log-normal mutation around the current value.
 *
 * mutatedValue = currentValue * exp(N(0, σ))
 * where σ = mutationMagnitude, then clamp to PARAM_BOUNDS[param].
 *
 * Log-normal ensures multiplicative mutations (scale-invariant) rather
 * than additive (which would be sensitive to parameter scale).
 */
export function applyMutation(
  currentValue: number,
  paramPath: string,
  mutationMagnitude: number,
): number {
  // Box-Muller for N(0,1)
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  const gauss = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);

  const mutated = currentValue * Math.exp(gauss * mutationMagnitude);
  const bounds = PARAM_BOUNDS[paramPath];
  if (bounds) {
    return Math.max(bounds[0], Math.min(bounds[1], mutated));
  }
  return mutated;
}

// ── UCB parameter selection ──

/**
 * selectParameterToTune: UCB-guided parameter selection for the next mutation.
 *
 * For each tunable parameter:
 *   n = count of evolution entries targeting this param (from experience library)
 *   avgReward = mean fitness delta of "keep" decisions for this param
 *   ucb = avgReward + ucbC * sqrt(log(totalMutations) / max(n, 0.1))
 *
 * Returns the parameter with the highest UCB score.
 * If no evolution history exists, selects randomly among all tunable params.
 *
 * RL theory (Auer et al. 2002): UCB balances exploration of untested
 * parameters with exploitation of parameters that have yielded good results.
 */
export function selectParameterToTune(state: SharedStateV5): string {
  const ucbC = state.policy.exploration.ucbC;
  const library = state.retrievalStrategy.experienceLibrary;

  if (library.length === 0) {
    // No history: random selection
    return TUNABLE_PARAMS[Math.floor(Math.random() * TUNABLE_PARAMS.length)];
  }

  // Compute per-param statistics from experience library
  const paramStats: Record<string, { n: number; rewardSum: number }> = {};
  for (const entry of library) {
    const p = entry.mutation.param;
    if (!paramStats[p]) paramStats[p] = { n: 0, rewardSum: 0 };
    paramStats[p].n += 1;
    // Count ALL deltas in reward sum (not just KEEPs) — UCB needs the true average reward
    paramStats[p].rewardSum += entry.fitnessDelta;
  }

  const totalMutations = library.length;

  // Compute UCB for each tunable param
  let bestParam = TUNABLE_PARAMS[0];
  let bestUCB = -Infinity;

  for (const param of TUNABLE_PARAMS) {
    const stats = paramStats[param] || { n: 0, rewardSum: 0 };
    const avgReward = stats.n > 0 ? stats.rewardSum / stats.n : 0;
    const exploration = ucbC * Math.sqrt(Math.log(totalMutations + 1) / Math.max(stats.n, 0.1));
    const ucb = avgReward + exploration;

    if (ucb > bestUCB) {
      bestUCB = ucb;
      bestParam = param;
    }
  }

  return bestParam;
}

// ── Parameter accessors ──

/**
 * getParamValue: reads a policy parameter by dot-separated path.
 * e.g., "compression.alpha" → policy.compression.alpha
 */
export function getParamValue(policy: any, paramPath: string): number {
  const parts = paramPath.split(".");
  let obj = policy;
  for (const part of parts) {
    obj = obj[part];
    if (obj === undefined) return 0;
  }
  return obj as unknown as number;
}

/**
 * setParamValue: writes a policy parameter by dot-separated path.
 * Mutates the policy object in place.
 */
export function setParamValue(policy: any, paramPath: string, value: number): void {
  const parts = paramPath.split(".");
  let obj = policy;
  for (let i = 0; i < parts.length - 1; i++) {
    obj = obj[parts[i]];
  }
  obj[parts[parts.length - 1]] = value;
}

// ── Scenario generation ──

/**
 * generateScenario: creates a human-readable scenario string for experience recording.
 *
 * Format: "{dominantTaskType}_{phaseName}_{trendDirection}"
 * e.g., "code_generation_broad_exploration_improving"
 */
function generateScenario(state: SharedStateV5): string {
  const dominantTask = getDominantTaskType(state);
  const phase = state.curriculum.phases[state.totalInvocations < 10 ? 0 : state.totalInvocations < 30 ? 1 : state.totalInvocations < 60 ? 2 : 3];
  const phaseName = phase?.name || "unknown";

  // Determine trend
  const relevantTrials = state.trials.filter(t => t.taskType === dominantTask).slice(-5);
  let trend = "neutral";
  if (relevantTrials.length >= 3) {
    const scores = relevantTrials.map(t => t.qualityScore);
    const slope = scores.length >= 2
      ? (scores[scores.length - 1] - scores[0]) / scores.length
      : 0;
    trend = slope > 0.01 ? "improving" : slope < -0.01 ? "declining" : "stable";
  }

  return `${dominantTask}_${phaseName}_${trend}`;
}

/**
 * getDominantTaskType: returns the most frequent task type in recent trials.
 */
function getDominantTaskType(state: SharedStateV5): TaskType {
  const counts: Record<string, number> = {};
  for (const trial of state.trials) {
    counts[trial.taskType] = (counts[trial.taskType] || 0) + 1;
  }
  let best = "code_generation";
  let bestCount = 0;
  for (const [tt, c] of Object.entries(counts)) {
    if (c > bestCount) { best = tt; bestCount = c; }
  }
  return best as TaskType;
}
