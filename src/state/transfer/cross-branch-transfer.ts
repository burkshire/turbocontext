// ============================================================================
// Turbocontext v6 — Cross-Branch Transfer
// ============================================================================
//
// Transfers learned compression parameters from data-rich task types to
// data-sparse ones via Jaccard similarity over capability requirements.
//
// Karpathy principle: "transfer_experiment" type — use knowledge from
// well-explored domains to bootstrap under-explored ones.
//
// Ported from autoresearch/agent_v5_integration.py CrossBranchTransfer class.
// ============================================================================

import type { TaskType, IndexedMemory, SharedStateV5, Trial } from "../types.js";
import { TaskType as TT } from "../types.js";

// ── Constants ──

/** Minimum source trials to be eligible for transfer. */
const MIN_SOURCE_TRIALS = 10;
/** Minimum Jaccard similarity to trigger transfer. */
const TRANSFER_THRESHOLD = 0.4;

// ── All valid V5 task types ──

const ALL_TASK_TYPES: TaskType[] = [
  TT.CODE_REVIEW, TT.CODE_GENERATION, TT.DEBUGGING,
  TT.REFACTORING, TT.DOCUMENTATION, TT.ARCHITECTURE,
];

// ── Public API ──

/**
 * taskTypeSimilarity: computes Jaccard overlap of capability requirements
 * from shared active/consolidated memories for two task types.
 *
 * Returns 0.0-1.0. Higher = more similar = better transfer candidates.
 */
export function taskTypeSimilarity(
  a: TaskType,
  b: TaskType,
  memories: IndexedMemory[],
): number {
  const capsA = new Set<string>();
  const capsB = new Set<string>();

  for (const mem of memories) {
    if (mem.status === "cold") continue;
    const reqs = mem.capabilityRequirements ?? [];
    if (mem.taskType === a) for (const r of reqs) capsA.add(r);
    if (mem.taskType === b) for (const r of reqs) capsB.add(r);
  }

  if (capsA.size === 0 || capsB.size === 0) return 0;
  const intersection = new Set([...capsA].filter(x => capsB.has(x)));
  const union = new Set([...capsA, ...capsB]);
  return intersection.size / Math.max(union.size, 1);
}

/**
 * transferPolicy: finds the best source task type and returns blended
 * compression parameters for the target task type.
 *
 * Only transfers when:
 *   1. Target has < MIN_SOURCE_TRIALS trials (needs help)
 *   2. Source has >= MIN_SOURCE_TRIALS trials (has data)
 *   3. Jaccard similarity >= TRANSFER_THRESHOLD (meaningful relationship)
 *
 * Returns { compression: {alpha,beta,gamma}, similarity, sourceTaskType }
 * or null if no suitable source exists.
 */
export function transferPolicy(
  targetTT: TaskType,
  state: SharedStateV5,
): {
  compression: { alpha: number; beta: number; gamma: number };
  similarity: number;
  sourceTaskType: TaskType;
} | null {
  // Count trials per task type
  const trialCounts = new Map<TaskType, number>();
  for (const t of state.trials) {
    trialCounts.set(t.taskType, (trialCounts.get(t.taskType) ?? 0) + 1);
  }

  const targetCount = trialCounts.get(targetTT) ?? 0;
  if (targetCount >= MIN_SOURCE_TRIALS) return null; // already has enough data

  let bestSource: TaskType | null = null;
  let bestSim = 0;
  let bestParams: { alpha: number; beta: number; gamma: number } | null = null;

  for (const sourceTT of ALL_TASK_TYPES) {
    if (sourceTT === targetTT) continue;
    const sourceCount = trialCounts.get(sourceTT) ?? 0;
    if (sourceCount < MIN_SOURCE_TRIALS) continue;

    const sim = taskTypeSimilarity(targetTT, sourceTT, state.memories);
    if (sim >= TRANSFER_THRESHOLD && sim > bestSim) {
      bestSim = sim;
      bestSource = sourceTT;
      bestParams = extractSourceParams(sourceTT, state);
    }
  }

  if (!bestSource || !bestParams) return null;

  return {
    compression: bestParams,
    similarity: bestSim,
    sourceTaskType: bestSource,
  };
}

/**
 * blendParams: blends base parameters toward transferred parameters
 * weighted by similarity. Higher similarity → more transfer influence.
 *
 * param = (1 - similarity) * base + similarity * transfer
 */
export function blendParams(
  base: { alpha: number; beta: number; gamma: number },
  transfer: { alpha: number; beta: number; gamma: number },
  similarity: number,
): { alpha: number; beta: number; gamma: number } {
  const t = Math.min(1, Math.max(0, similarity));
  return {
    alpha: (1 - t) * base.alpha + t * transfer.alpha,
    beta: (1 - t) * base.beta + t * transfer.beta,
    gamma: 1.0 - ((1 - t) * base.alpha + t * transfer.alpha) - ((1 - t) * base.beta + t * transfer.beta),
  };
}

// ── Internal helpers ──

/** Extracts mean compression params from successful source trials. */
function extractSourceParams(
  sourceTT: TaskType,
  state: SharedStateV5,
): { alpha: number; beta: number; gamma: number } {
  const sourceTrials = state.trials.filter(
    t => t.taskType === sourceTT && t.outcome === "success"
  );

  if (sourceTrials.length === 0) {
    // Fall back to policy defaults
    return {
      alpha: state.policy.compression.alpha,
      beta: state.policy.compression.beta,
      gamma: state.policy.compression.gamma,
    };
  }

  const n = sourceTrials.length;
  return {
    alpha: sourceTrials.reduce((s, t) => s + t.compressionWeights.alpha, 0) / n,
    beta: sourceTrials.reduce((s, t) => s + t.compressionWeights.beta, 0) / n,
    gamma: sourceTrials.reduce((s, t) => s + t.compressionWeights.gamma, 0) / n,
  };
}
