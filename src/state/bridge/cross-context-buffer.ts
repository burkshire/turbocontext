// ============================================================================
// Turbocontext v5 — Cross-Context Bridge
// ============================================================================
//
// Bidirectional bridge between Context A (skill invocation — Lite mode)
// and Context B (autonomous agent — Full mode).
//
// Flow:
//   Context A records trials in Lite mode → enqueues to crossContextBuffer
//   Context B periodically syncs → runs full RL pipeline on pending trials
//   Context B writes back refined insights, policy diffs, canonical strategies
//
// RL theory: this is a form of asynchronous distributed RL — one process
// generates experience (acting), another processes it (learning), and
// the learned policy improvements flow back to the actor.
import type {
  SharedStateV5, Trial, CrossContextBuffer,
  CanonicalStrategy, SyncResult, IndexedMemory,
} from "../types.js";
import { CANONICAL_TRIAL_THRESHOLD, CANONICAL_SUCCESS_RATE } from "../constants.js";

// ── Enqueue (Context A side) ──

/**
 * enqueueTrialForSync: adds a trial to the pending queue for later
 * full-RL processing by Context B.
 *
 * Called during Lite-mode recordTrial. The trial waits in the buffer
 * until the next Full-mode sync consumes it.
 */
export function enqueueTrialForSync(
  buffer: CrossContextBuffer,
  trial: Trial,
): CrossContextBuffer {
  const queue = buffer.pendingTrialsFromSkill;
  queue.trials.push(trial);
  queue.count = queue.trials.length;
  if (queue.trials.length === 1) {
    queue.oldestPending = trial.timestamp;
  }
  return buffer;
}

// ── Consume (Context B side) ──

/**
 * consumePendingTrialsFromSkill: processes all pending skill trials
 * through the full RL pipeline.
 *
 * For each pending trial:
 *   1. Decay + bump eligibility traces
 *   2. Compute predictedQuality + surprise
 *   3. Update Thompson params for retrieved memories
 *   4. Apply TD(λ) update
 *   5. Update value function baselines
 *   6. Update curiosity (RND + IDF)
 *   7. Synthesize counterfactuals + HER
 *   8. Append to state.trials
 *
 * This function delegates to the RL engine for the actual learning steps
 * — it primarily manages the buffer lifecycle.
 *
 * @param state The current state (mutated in place)
 * @param processTrialFn A callback that runs full RL on one trial
 * @returns Processed state and sync result
 */
export function consumePendingTrialsFromSkill(
  state: SharedStateV5,
  processTrialFn: (state: SharedStateV5, trial: Trial) => void,
): { newState: SharedStateV5; syncResult: SyncResult } {
  const buffer = state.crossContextBuffer;
  const pending = buffer.pendingTrialsFromSkill;
  const trials = [...pending.trials];

  if (trials.length === 0) {
    return {
      newState: state,
      syncResult: { trialsProcessed: 0, insightsGenerated: 0, policyDiffsApplied: 0 },
    };
  }

  // Process each trial through the full RL pipeline
  for (const trial of trials) {
    processTrialFn(state, trial);
  }

  // Detect canonical strategies from the processed batch
  const newCanonicals = detectCanonicalStrategies(trials, state);
  for (const cs of newCanonicals) {
    if (!buffer.canonicalStrategies.some(existing => existing.pattern === cs.pattern)) {
      buffer.canonicalStrategies.push(cs);
    }
  }

  // Write refined insights
  const insightsGenerated = newCanonicals.length;
  const updatedMemoryIds = new Set<string>();
  for (const trial of trials) {
    for (const mid of trial.referencedMemoryIds) {
      updatedMemoryIds.add(mid);
    }
  }

  const updatedUtils: Record<string, number> = {};
  for (const mid of updatedMemoryIds) {
    const mem = state.memories.find(m => m.id === mid);
    if (mem) updatedUtils[mid] = mem.causalUtility;
  }

  state.crossContextBuffer = writeRefinedInsights(
    state, trials.length, insightsGenerated, 0,
  );

  // Clear the pending queue
  buffer.pendingTrialsFromSkill.trials = [];
  buffer.pendingTrialsFromSkill.count = 0;

  return {
    newState: state,
    syncResult: {
      trialsProcessed: trials.length,
      insightsGenerated,
      policyDiffsApplied: 0,
    },
  };
}

// ── Write-back ──

/**
 * writeRefinedInsights: populates refinedInsights after processing.
 *
 * Updates:
 *   - updatedMemoryUtils: memoryId → new causalUtility map
 *   - discoveredPatterns: task-type-level insights
 *   - lastSyncTimestamp: now
 *   - agentIterationsProcessed: incremented
 */
export function writeRefinedInsights(
  state: SharedStateV5,
  processedCount: number,
  insightsGenerated: number,
  policyDiffsApplied: number,
): CrossContextBuffer {
  const buffer = state.crossContextBuffer;
  const now = new Date().toISOString();

  // Collect updated memory utilities
  const updatedMemoryUtils: Record<string, number> = {};
  for (const mem of state.memories.filter(m => m.status === "active")) {
    updatedMemoryUtils[mem.id] = mem.causalUtility;
  }

  return {
    ...buffer,
    refinedInsights: {
      ...buffer.refinedInsights,
      updatedMemoryUtils,
      discoveredPatterns: insightsGenerated > 0
        ? [...buffer.refinedInsights.discoveredPatterns, `Cross-context sync processed ${processedCount} trials, found ${insightsGenerated} canonical strategies`]
        : buffer.refinedInsights.discoveredPatterns,
      lastSyncTimestamp: now,
      agentIterationsProcessed: buffer.refinedInsights.agentIterationsProcessed + 1,
    },
  };
}

// ── Canonical strategy detection ──

/**
 * detectCanonicalStrategies: finds parameter patterns that appear across
 * >= CANONICAL_TRIAL_THRESHOLD trials with >= CANONICAL_SUCCESS_RATE success.
 *
 * A "pattern" is defined by: compressionWeights (α,β,γ within 0.05) +
 * modelTier + retrievalTopK + t0 (within 0.05).
 *
 * RL theory: canonical strategies are "proven recipes" — parameter
 * combinations that reliably produce good outcomes. They serve as
 * strong priors for future parameter selection.
 */
export function detectCanonicalStrategies(
  trials: Trial[],
  state: SharedStateV5,
): CanonicalStrategy[] {
  if (trials.length < CANONICAL_TRIAL_THRESHOLD) return [];

  const patterns = new Map<string, { successes: number; total: number; taskType: string; trial: Trial }>();

  for (const trial of trials) {
    const patternKey = buildPatternKey(trial);
    const existing = patterns.get(patternKey);
    if (existing) {
      existing.total += 1;
      if (trial.outcome === "success") existing.successes += 1;
    } else {
      patterns.set(patternKey, {
        successes: trial.outcome === "success" ? 1 : 0,
        total: 1,
        taskType: trial.taskType,
        trial,
      });
    }
  }

  const canonicals: CanonicalStrategy[] = [];
  const now = new Date().toISOString();

  for (const [, data] of patterns) {
    if (data.total < CANONICAL_TRIAL_THRESHOLD) continue;
    const rate = data.successes / data.total;
    if (rate < CANONICAL_SUCCESS_RATE) continue;

    const t = data.trial;
    canonicals.push({
      strategyId: `canonical-${data.taskType}-${Date.now()}`,
      taskType: t.taskType as any,
      pattern: buildPatternKey(t),
      params: {
        alpha: t.compressionWeights.alpha,
        beta: t.compressionWeights.beta,
        gamma: t.compressionWeights.gamma,
        mmrLambda: state.policy.retrieval.mmrLambda,
        topK: t.retrievalTopK,
      },
      successRate: rate,
      trialCount: data.total,
      discoveredBy: t.context,
      discoveredAt: now,
    });
  }

  return canonicals;
}

/**
 * buildPatternKey: creates a string key from a trial's key parameters.
 */
function buildPatternKey(trial: Trial): string {
  const { alpha, beta, gamma } = trial.compressionWeights;
  return [
    alpha.toFixed(2),
    beta.toFixed(2),
    gamma.toFixed(2),
    trial.modelTier,
    trial.retrievalTopK,
    trial.temperatureSchedule[0].toFixed(2),
  ].join("|");
}
