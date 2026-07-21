// ============================================================================
// Turbocontext v5 — Hindsight Experience Replay (HER)
// ============================================================================
//
// HER (Andrychowicz et al. 2017): relabels failures as successes by
// reinterpreting what "success" means. This dramatically increases the
// density of positive training signal from sparse rewards.
//
// THREE RELABELING STRATEGIES:
//   1. CRASH BOUNDARY — found where the system breaks      (reward: 0.7)
//   2. RULED OUT APPROACH — eliminated an ineffective config (reward: 0.5)
//   3. SIDE IMPROVEMENT — some dimension improved despite failure (reward: ≤0.7)
import type { Trial, HERGoal } from "../types.js";
import { DIM_NAMES } from "../constants.js";

/**
 * herRelabel: relabels a failure/crash trial into one or more "success" goals.
 *
 * Only called when trial.outcome !== "success".
 * Returns 1-3 HERGoal entries, each with outcome="success" and a reward
 * in [0.5, 0.7] — lower than a real success, but high enough to provide
 * a meaningful learning signal.
 *
 * @param trial The failed trial to relabel
 * @param baselineScores Optional per-dimension baseline scores for this task type.
 *   If omitted, side-improvement relabeling uses 0 as the baseline.
 * @returns Array of HERGoal entries (always at least 1 for crash, 1-3 for failure)
 */
export function herRelabel(
  trial: Trial,
  baselineScores?: [number, number, number, number],
): HERGoal[] {
  if (trial.outcome === "success") return [];

  const goals: HERGoal[] = [];

  // Strategy 1: Crash boundary (always generated for crash outcome)
  if (trial.outcome === "crash") {
    goals.push(...herRelabelCrashBoundary(trial));
  }

  // Strategy 2: Ruled out approach (always generated for non-success)
  goals.push(herRelabelRuledOut(trial));

  // Strategy 3: Side improvements (only if some dimension improved)
  goals.push(...herRelabelSideImprovements(trial, baselineScores));

  return goals;
}

/**
 * herRelabelCrashBoundary: generates crash-boundary goals.
 *
 * RL theory: discovering where the system breaks is valuable information.
 * It defines the safe operating envelope and prevents future crashes.
 * Reward = 0.7, reflecting high informational value of knowing a boundary.
 */
function herRelabelCrashBoundary(trial: Trial): HERGoal[] {
  const caps = trial.capabilityRequirements.slice(0, 3).join(", ");
  return [{
    goal: `Determine safe operating bounds for ${trial.taskType}`,
    outcome: "success",
    reward: 0.7,
    insight: `Established crash boundary for capabilities: ${caps || "unknown"}. ` +
      `Compression ratio ${trial.compressionRatio.toFixed(2)} with ` +
      `α=${trial.compressionWeights.alpha.toFixed(2)}, ` +
      `temperature ${trial.temperatureSchedule[0]} caused instability. ` +
      `Future attempts should use lower compression or higher temperature.`,
  }];
}

/**
 * herRelabelRuledOut: generates a ruled-out-approach goal.
 *
 * RL theory: negative results are essential for efficient exploration.
 * Knowing that a configuration DOESN'T work reduces the search space.
 * Reward = 0.5, lower than crash boundary (less diagnostic value).
 */
function herRelabelRuledOut(trial: Trial): HERGoal {
  const { alpha, beta, gamma } = trial.compressionWeights;
  return {
    goal: `Eliminate ineffective configuration for ${trial.taskType}`,
    outcome: "success",
    reward: 0.5,
    insight: `Ruled out compression (α=${alpha.toFixed(2)}, β=${beta.toFixed(2)}, ` +
      `γ=${gamma.toFixed(2)}) with model=${trial.modelTier}, ` +
      `temp=(${trial.temperatureSchedule.map(t => t.toFixed(2)).join(",")}) ` +
      `for ${trial.taskType}. Quality was ${trial.qualityScore.toFixed(2)}. ` +
      `This configuration should be avoided in future attempts.`,
  };
}

/**
 * herRelabelSideImprovements: generates goals for each dimension that improved.
 *
 * RL theory: even when overall quality is below threshold, individual
 * dimensions may have improved. These partial successes are learning
 * signals — they tell us which aspects of the policy are working.
 *
 * Reward = improvement magnitude, capped at 0.7.
 * Improvements < 0.01 are suppressed (noise).
 */
function herRelabelSideImprovements(
  trial: Trial,
  baselineScores?: [number, number, number, number],
): HERGoal[] {
  const goals: HERGoal[] = [];
  const baseline = baselineScores || [0, 0, 0, 0];

  // If all scores are zero, skip side-improvement (nothing to celebrate)
  const totalScore = trial.qualityScores.reduce((a, b) => a + b, 0);
  if (totalScore < 0.01) return goals;

  for (let dim = 0; dim < 4; dim++) {
    const improvement = trial.qualityScores[dim] - baseline[dim];
    if (improvement < 0.01) continue; // suppress negligible

    const reward = Math.min(improvement, 0.7);
    goals.push({
      goal: `Improve ${DIM_NAMES[dim]} for ${trial.taskType}`,
      outcome: "success",
      reward,
      insight: `Improved ${DIM_NAMES[dim]} from ${baseline[dim].toFixed(2)} ` +
        `to ${trial.qualityScores[dim].toFixed(2)} ` +
        `(+${improvement.toFixed(2)}) despite overall quality of ` +
        `${trial.qualityScore.toFixed(2)}. The parameter choices partially worked.`,
    });
  }
  return goals;
}
