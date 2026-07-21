// ============================================================================
// Turbocontext v5 — Curriculum Learning
// ============================================================================
//
// Phase-adaptive hyperparameter scheduling.
//
// RL theory (Bengio et al. 2009): curriculum learning organizes training
// from easy to hard. Here, "difficulty" is measured by total experience
// count — early phases explore broadly, later phases optimize deeply.
//
// Phase 0 (0-9 trials):   Broad Exploration    — high diversity, large mutations
// Phase 1 (10-29 trials): Focused Exploitation — deepen promising branches
// Phase 2 (30-59 trials): Principled Opt.      — fine-tune based on principles
// Phase 3 (60+ trials):   Adversarial Refinement — challenge assumptions
import type { SharedStateV5, CurriculumPhaseConfig, TaskType, ValueFunctionState } from "../types.js";
import { CurriculumPhase } from "../types.js";

/**
 * getCurrentPhase: determines the current curriculum phase from totalInvocations.
 *
 * Uses the configurable phase boundaries from state.curriculum.phaseBoundaries.
 * Phase boundaries are trial-count thresholds: [10, 30, 60] means:
 *   phase 0 at 0-9, phase 1 at 10-29, phase 2 at 30-59, phase 3 at 60+.
 */
export function getCurrentPhase(state: SharedStateV5): number {
  const n = state.totalInvocations;
  const [b1, b2, b3] = state.curriculum.phaseBoundaries;
  if (n < b1) return CurriculumPhase.BROAD_EXPLORATION;
  if (n < b2) return CurriculumPhase.FOCUSED_EXPLOITATION;
  if (n < b3) return CurriculumPhase.PRINCIPLED_OPTIMIZATION;
  return CurriculumPhase.ADVERRIAL_REFINEMENT;
}

/**
 * getPhaseConfig: returns the active phase's hyperparameter configuration.
 */
export function getPhaseConfig(state: SharedStateV5): CurriculumPhaseConfig {
  return state.curriculum.phases[getCurrentPhase(state)];
}

/**
 * isLearningStep: true when totalInvocations is a multiple of
 * the current phase's learningInterval. Controls evolution frequency.
 *
 * Phase 0: every 3 trials → aggressive exploration
 * Phase 1: every 5 trials
 * Phase 2: every 8 trials → more stable
 * Phase 3: every 10 trials → fine-tuning
 */
export function isLearningStep(state: SharedStateV5): boolean {
  const config = getPhaseConfig(state);
  return state.totalInvocations > 0 && state.totalInvocations % config.learningInterval === 0;
}

/**
 * isConsolidationStep: true when totalInvocations is a multiple of
 * the current phase's consolidationInterval. Controls memory pruning frequency.
 */
export function isConsolidationStep(state: SharedStateV5): boolean {
  const config = getPhaseConfig(state);
  return state.totalInvocations > 0 && state.totalInvocations % config.consolidationInterval === 0;
}

/**
 * effectiveExplorationRate: the phase's exploration rate, modulated by
 * plateau detection. If the task type baseline has plateaued, increase
 * exploration to escape the local minimum.
 *
 * plateaued → explorationRate * 1.5
 * normal   → explorationRate
 */
export function effectiveExplorationRate(
  state: SharedStateV5,
  taskType: TaskType,
): number {
  const base = getPhaseConfig(state).explorationRate;
  const baseline = state.valueFunction.baselines[taskType];
  if (baseline && baseline.recentScores.length >= 5 && Math.abs(baseline.slope) < 0.005) {
    return Math.min(base * 1.5, 1.0); // increase but cap at 1.0
  }
  return base;
}

/**
 * effectiveMutationMagnitude: the phase's mutation magnitude, scaled by
 * surprise anomaly detection.
 *
 * If recent surprises are anomalously high → increase mutation to explore more.
 *   multiplier = 1.0 + clamp((meanSurprise - anomalyThreshold) / anomalyThreshold, 0, 0.5)
 */
export function effectiveMutationMagnitude(state: SharedStateV5): number {
  const base = getPhaseConfig(state).mutationMagnitude;
  const ss = state.curiosity.surpriseStats;
  if (ss.anomalyThreshold > 0 && ss.globalMean > ss.anomalyThreshold) {
    const excess = Math.min((ss.globalMean - ss.anomalyThreshold) / Math.max(ss.anomalyThreshold, 0.01), 0.5);
    return base * (1.0 + excess);
  }
  return base;
}
