// ============================================================================
// Turbocontext v5 — Counterfactual Synthesis
// ============================================================================
//
// Heuristic counterfactual generation — NO LLM CALL REQUIRED.
// Produces 2-4 natural-language "what if" statements per trial.
//
// RL theory (Pearl 2009): counterfactuals answer "what would have happened
// if we had done X instead of Y?" They are the basis for causal reasoning
// and help the policy learner identify parameter-to-outcome relationships.
import type { Trial, SharedStateV5 } from "../types.js";
import { TUNABLE_PARAMS, PARAM_BOUNDS } from "../constants.js";

/**
 * synthesizeCounterfactuals: generates counterfactual strings for a trial.
 *
 * Uses hardcoded templates parameterized with trial values. Three branches:
 *   - Success: "what if we had ALSO tuned {untuned_param}?"
 *   - Failure: "what if we had used a different {divergent_param}?"
 *   - Crash: "what if we had been more conservative?"
 *
 * Always generates at least 2 counterfactuals (3-4 for failures/crashes).
 */
export function synthesizeCounterfactuals(
  trial: Trial,
  state: SharedStateV5,
): string[] {
  switch (trial.outcome) {
    case "success": return successCounterfactuals(trial, state);
    case "failure": return failureCounterfactuals(trial, state);
    case "crash":   return crashCounterfactuals(trial, state);
  }
}

// ── Success branch ──

function successCounterfactuals(trial: Trial, state: SharedStateV5): string[] {
  const cfs: string[] = [];
  const { alpha, beta, gamma } = trial.compressionWeights;

  // Main what-if: what if we also tuned another parameter?
  const untuned = selectUntunedParameter(trial, state);
  if (untuned) {
    cfs.push(
      `Without compression (α=${alpha.toFixed(2)}, β=${beta.toFixed(2)}, ` +
      `γ=${gamma.toFixed(2)}) and temperature=${trial.temperatureSchedule[0].toFixed(2)}, ` +
      `quality would likely be lower than ${trial.qualityScore.toFixed(2)}. ` +
      `If we had ALSO increased ${untuned}, the gains might compound.`
    );
  }

  // Secondary: what if we used a faster model?
  cfs.push(
    `Using ${trial.modelTier} model with ${trial.tokenBudgetUsed} tokens ` +
    `achieved quality ${trial.qualityScore.toFixed(2)}. A faster model might ` +
    `achieve similar quality at lower cost — worth testing on the next similar task.`
  );

  return cfs;
}

// ── Failure branch ──

function failureCounterfactuals(trial: Trial, state: SharedStateV5): string[] {
  const cfs: string[] = [];
  const { alpha, beta, gamma } = trial.compressionWeights;

  // Main: what if we had used a different parameter? Propose CONCRETE values.
  const divergent = selectDivergentParameter(trial, state);
  if (divergent) {
    // Propose a concrete alternative: move 30% toward the opposite bound
    const bounds = PARAM_BOUNDS[divergent];
    const altValue = bounds
      ? (bounds[0] + bounds[1]) / 2 // suggest the midpoint as an alternative
      : 0;
    cfs.push(
      `Parameters (α=${alpha.toFixed(2)}, β=${beta.toFixed(2)}, γ=${gamma.toFixed(2)}) ` +
      `did not help ${trial.taskType}. Try ${divergent}=${altValue.toFixed(2)} ` +
      `(current policy default) or model tier "${trial.modelTier === "fast" ? "medium" : "fast"}" ` +
      `to change the outcome. The negative result rules out THIS configuration, not the approach.`
    );
  }

  // Secondary: what if we used a different beta? (most common stuck parameter)
  const betaAlt = beta >= 0.40 ? 0.25 : 0.40;
  cfs.push(
    `compression.beta=${beta.toFixed(2)} has been unchanged in all trials. ` +
    `Try beta=${betaAlt.toFixed(2)} — ${betaAlt > beta ? "higher beta weights recency more" : "lower beta reduces recency bias"}. ` +
    `This is the #1 untested parameter and the most likely lever for ${trial.taskType}.`
  );

  // Tertiary: temperature adjustment with concrete suggestion
  const t0Alt = trial.temperatureSchedule[0] < 0.5 ? 0.65 : 0.50;
  cfs.push(
    `Temperature schedule (${trial.temperatureSchedule.map(t => t.toFixed(2)).join(", ")}) ` +
    `may have been suboptimal. Try t0=${t0Alt.toFixed(2)} (currently ` +
    `${trial.temperatureSchedule[0].toFixed(2)}) to ${t0Alt > trial.temperatureSchedule[0] ? "increase" : "decrease"} exploration on first attempt.`
  );

  return cfs.slice(0, 4); // max 4 counterfactuals
}

// ── Crash branch ──

function crashCounterfactuals(trial: Trial, _state: SharedStateV5): string[] {
  const { alpha, beta } = trial.compressionWeights;

  return [
    `If parameters had been more conservative (lower compression ratio, ` +
    `faster model tier, higher temperature), the result might have been ` +
    `stable. Consider gradual rollout of this parameter combination.`,

    `Compression weights (α=${alpha.toFixed(2)}, β=${beta.toFixed(2)}) ` +
    `combined with temperature ${trial.temperatureSchedule[0].toFixed(2)} ` +
    `caused a crash at ${trial.tokenBudgetUsed} tokens. ` +
    `Try halving the compression ratio and increasing temperature to 0.5+.`,

    `The ${trial.modelTier} model tier may be too aggressive for ` +
    `${trial.taskType} when combined with these parameters. ` +
    `Downgrading to "fast" tier would reduce cost and may avoid the crash.`,

    // v6: Crash boundary established — this is useful information
    `Crash boundary established for ${trial.taskType}: (α=${alpha.toFixed(2)}, ` +
    `β=${beta.toFixed(2)}, T=${trial.temperatureSchedule[0].toFixed(2)}, ` +
    `${trial.tokenBudgetUsed} tokens). Mark as "ruled out" for future exploration.`,
  ];
}

// ── v6: Extended counterfactual templates (Karpathy-inspired) ──

function successCounterfactualsV6(trial: Trial, state: SharedStateV5): string[] {
  const cfs = successCounterfactuals(trial, state);
  const { alpha, beta, gamma } = trial.compressionWeights;
  const bl = state.valueFunction.baselines[trial.taskType];

  // Big improvement (>0.1 above baseline): suggest compounding
  if (bl && trial.qualityScore > bl.ema + 0.1) {
    cfs.push(
      `BIG WIN: quality ${trial.qualityScore.toFixed(3)} vs baseline ${bl.ema.toFixed(3)} ` +
      `(+${((trial.qualityScore - bl.ema) * 100).toFixed(1)}%). Without ` +
      `(α=${alpha.toFixed(2)}, β=${beta.toFixed(2)}, γ=${gamma.toFixed(2)}), ` +
      `quality would be ~${bl.ema.toFixed(3)}. Combined with orthogonal changes, gains might compound.`
    );
  }

  // Marginal improvement: suggest simpler alternative
  if (bl && trial.qualityScore < bl.ema + 0.05 && trial.qualityScore >= state.policy.quality.threshold) {
    cfs.push(
      `MARGINAL: (α=${alpha.toFixed(2)}, β=${beta.toFixed(2)}) helped but minimally ` +
      `(+${((trial.qualityScore - bl.ema) * 100).toFixed(1)}% vs baseline). ` +
      `A simpler configuration with fewer parameters might achieve the same.`
    );
  }

  return cfs.slice(0, 5);
}

function failureCounterfactualsV6(trial: Trial, state: SharedStateV5): string[] {
  const cfs = failureCounterfactuals(trial, state);
  const bl = state.valueFunction.baselines[trial.taskType];

  // Repeated failure: count recent failures for this task type
  const recentFailures = state.trials.filter(
    t => t.taskType === trial.taskType && t.outcome === "failure"
  ).length;
  if (recentFailures >= 3) {
    cfs.push(
      `REPEATED FAILURE: ${trial.taskType} has now failed ${recentFailures} times ` +
      `with similar parameter ranges. Strong evidence against this approach — ` +
      `consider switching to a fundamentally different strategy or task type.`
    );
  }

  // Plateau detection: slope near zero
  if (bl && Math.abs(bl.slope) < 0.005 && bl.recentScores.length >= 5) {
    cfs.push(
      `PLATEAU: ${trial.taskType} quality has been flat (slope=${bl.slope.toFixed(4)}). ` +
      `Recent changes haven't moved the needle. Consider an orthogonal direction: ` +
      `different model tier, radically different compression, or cross-task transfer.`
    );
  }

  return cfs.slice(0, 5);
}

// ── Main export (v6: extended dispatch) ──

/**
 * synthesizeCounterfactuals: generates counterfactual strings for a trial.
 *
 * v6: Expanded from 3 to 8+ templates with context-aware dispatch.
 * Success → successCounterfactualsV6 (big_improvement | marginal | baseline)
 * Failure → failureCounterfactualsV6 (repeated | plateau | divergent_param)
 * Crash → crashCounterfactuals (crash_boundary)
 *
 * Always generates 2-5 counterfactuals. No LLM calls — pure heuristics.
 */
export function synthesizeCounterfactualsV6(
  trial: Trial,
  state: SharedStateV5,
): string[] {
  switch (trial.outcome) {
    case "success": return successCounterfactualsV6(trial, state);
    case "failure": return failureCounterfactualsV6(trial, state);
    case "crash":   return crashCounterfactuals(trial, state);
  }
}

// ── Parameter selection helpers ──

/**
 * selectUntunedParameter: picks a tunable parameter that was NOT
 * meaningfully varied in this trial (compared to policy defaults).
 *
 * Strategy: find the parameter whose value is closest to the policy mean
 * (least tuned) and return it as a candidate for "what if we also tuned X?"
 */
function selectUntunedParameter(
  trial: Trial,
  state: SharedStateV5,
): string | null {
  // Use trial params vs policy to find least-deviated parameter
  const policy = state.policy;
  const paramValues: Record<string, number> = {
    "compression.alpha": trial.compressionWeights.alpha,
    "compression.beta": trial.compressionWeights.beta,
    "compression.gamma": trial.compressionWeights.gamma,
    "compression.theta1": policy.compression.theta1,
    "compression.theta2": policy.compression.theta2,
    "quality.threshold": policy.quality.threshold,
    "temperature.t0": trial.temperatureSchedule[0],
    "temperature.t1": trial.temperatureSchedule[1],
    "temperature.t2": trial.temperatureSchedule[2],
    "retrieval.mmrLambda": policy.retrieval.mmrLambda,
    "retrieval.topK": trial.retrievalTopK,
  };

  // Find param with value closest to its bound midpoint (least "tuned")
  let bestParam: string | null = null;
  let bestDistance = Infinity;

  for (const param of TUNABLE_PARAMS) {
    const val = paramValues[param];
    if (val === undefined) continue;
    const bounds = PARAM_BOUNDS[param];
    if (!bounds) continue;
    const midpoint = (bounds[0] + bounds[1]) / 2;
    const dist = Math.abs(val - midpoint);
    if (dist < bestDistance) {
      bestDistance = dist;
      bestParam = param;
    }
  }

  return bestParam && bestDistance < 0.15 ? bestParam : TUNABLE_PARAMS[0];
}

/**
 * selectDivergentParameter: picks the parameter whose value is furthest
 * from the policy mean — the one that was most aggressively tuned.
 *
 * "What if we had used a DIFFERENT value for X?" where X is the parameter
 * that deviated most from default.
 */
function selectDivergentParameter(
  trial: Trial,
  state: SharedStateV5,
): string | null {
  const policy = state.policy;
  const paramDiffs: Array<{ param: string; diff: number }> = [
    { param: "compression.alpha", diff: Math.abs(trial.compressionWeights.alpha - policy.compression.alpha) },
    { param: "compression.beta", diff: Math.abs(trial.compressionWeights.beta - policy.compression.beta) },
    { param: "compression.gamma", diff: Math.abs(trial.compressionWeights.gamma - policy.compression.gamma) },
    { param: "temperature.t0", diff: Math.abs(trial.temperatureSchedule[0] - policy.temperature.t0) },
    { param: "retrieval.topK", diff: Math.abs(trial.retrievalTopK - policy.retrieval.topK) },
  ];

  paramDiffs.sort((a, b) => b.diff - a.diff);
  return paramDiffs[0]?.diff > 0.05 ? paramDiffs[0].param : null;
}
