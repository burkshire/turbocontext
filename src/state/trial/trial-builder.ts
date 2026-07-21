// ============================================================================
// Turbocontext v5 — Trial Builder
// ============================================================================
// Pure functions that construct Trial objects from invocation data.
// RL fields (predictedQuality, surprise, counterfactuals, advantage,
// causalUtility, herGoals) are left null/empty — the RL engine populates
// them during recordTrial().
import * as crypto from "node:crypto";
import type { Trial, TrialInput, Hex16, UUID } from "../types.js";

/**
 * buildTrial: constructs a Trial from raw invocation data.
 *
 * RL theory: this is the (s,a,r,s') tuple constructor. The caller supplies
 * the state (task description, capability requirements), the action
 * (compression weights, temperature, model tier), the reward (qualityScore,
 * outcome), and the next-state partial (retrievedMemoryIds).
 *
 * The returned Trial has all RL fields set to neutral defaults — the
 * RL engine fills them in after prediction, TD update, and HER relabeling.
 */
export function buildTrial(input: TrialInput): Trial {
  const now = new Date().toISOString();

  return {
    id: buildTrialId(),
    timestamp: now,
    context: input.context,
    taskType: input.taskType,
    descriptionHash: hashDescription(input.description),
    descriptionLength: computeDescriptionLength(input.description),
    capabilityRequirements: input.capabilityRequirements,
    compressionRatio: input.compressionRatio,
    compressionWeights: input.compressionWeights,
    temperatureSchedule: input.temperatureSchedule,
    modelTier: input.modelTier,
    retrievalTopK: input.retrievalTopK,
    tokenBudgetUsed: input.tokenBudgetUsed,
    maxAttempts: input.maxAttempts,
    outcome: input.outcome,
    qualityScores: input.qualityScores,
    qualityScore: input.qualityScore,
    costUsd: input.costUsd,
    latencyMs: input.latencyMs,
    attemptCount: input.attemptCount,
    bestAttemptIndex: input.bestAttemptIndex,
    // RL fields — filled by recordTrial()
    predictedQuality: null,
    surprise: 0,
    counterfactuals: [],
    curriculumPhase: 0,
    retrievedMemoryIds: input.retrievedMemoryIds,
    referencedMemoryIds: input.referencedMemoryIds,
    advantage: null,
    causalUtility: 0,
    herGoals: [],
  };
}

/**
 * buildTrialId: generates a v4 UUID via crypto.randomUUID().
 *
 * RL theory: IDs are used as trace keys in the TD(λ) eligibility chain.
 * UUIDs ensure no collisions across sessions or restarts — critical
 * for cross-context buffer sync where IDs must be globally unique.
 */
export function buildTrialId(): UUID {
  return crypto.randomUUID();
}

/**
 * hashDescription: returns the first 16 hex characters of a SHA-256 hash
 * of the task description. Used for deduplication and fast lookup without
 * storing full descriptions in the trial record.
 */
export function hashDescription(desc: string): Hex16 {
  const hash = crypto.createHash("sha256").update(desc, "utf-8").digest("hex");
  return hash.slice(0, 16);
}

/**
 * computeDescriptionLength: returns the character length of the description,
 * bounded to MAX_TRIAL_DESCRIPTION_LENGTH.
 */
export function computeDescriptionLength(desc: string): number {
  return Math.min(desc.length, 50000);
}
