// ============================================================================
// Turbocontext v5 — Memory Consolidation
// ============================================================================
//
// Consolidation: merge low-utility memories into summaries and archive
// cold memories to cold storage. This keeps the active memory pool small
// (< 200) for fast retrieval while preserving information.
//
// RL theory: consolidation is analogous to hippocampal replay — important
// patterns are strengthened, while low-utility memories are compressed.
// The consolidationCount field tracks how many memories went into a summary.
import type {
  SharedStateV5, IndexedMemory, ConsolidationResult,
  ConsolidationEntry, TaskType,
} from "../types.js";
import { ConsolidationAction } from "../types.js";
import {
  MAX_ACTIVE_MEMORIES, COLD_STORAGE_DAYS,
  COLD_STORAGE_RETRIEVAL_COUNT, DEGRADATION_THRESHOLD,
} from "../constants.js";

/**
 * consolidateMemories: runs one consolidation pass.
 *
 * Algorithm:
 *   1. Check if active memories <= 200 — return if no pressure.
 *   2. Identify LOW-UTILITY memories: causalUtility < 0.3 AND not retrieved
 *      recently (> COLD_STORAGE_DAYS days or null + createdAt > 30 days).
 *   3. Group by taskType.
 *   4. For groups with >= 3 memories: create consolidated summary memory,
 *      mark sources as status="consolidated".
 *   5. For memories never retrieved in COLD_STORAGE_RETRIEVAL_COUNT+ trials:
 *      move to cold storage (status="cold").
 *
 * Returns { consolidatedCount, archivedCount, tokensFreed }.
 */
export function consolidateMemories(
  state: SharedStateV5,
): { newState: SharedStateV5; result: ConsolidationResult } {
  const activeMemories = state.memories.filter(m => m.status === "active");

  // Check if consolidation is needed
  if (activeMemories.length <= MAX_ACTIVE_MEMORIES) {
    return {
      newState: state,
      result: { consolidatedCount: 0, archivedCount: 0, tokensFreed: 0 },
    };
  }

  const now = new Date();
  const nowISO = now.toISOString();
  let consolidatedCount = 0;
  let archivedCount = 0;
  let tokensFreed = 0;

  // Step 2: Identify low-utility memories
  const lowUtility = activeMemories.filter(m => {
    const isLowUtility = m.causalUtility < 0.3;
    const lastRet = m.lastRetrievedAt ? new Date(m.lastRetrievedAt) : null;
    const daysSinceRet = lastRet
      ? (now.getTime() - lastRet.getTime()) / (1000 * 60 * 60 * 24)
      : (now.getTime() - new Date(m.createdAt).getTime()) / (1000 * 60 * 60 * 24);
    return isLowUtility && daysSinceRet > COLD_STORAGE_DAYS;
  });

  // Step 3: Group by task type
  const groups = new Map<TaskType, IndexedMemory[]>();
  for (const m of lowUtility) {
    const group = groups.get(m.taskType) || [];
    group.push(m);
    groups.set(m.taskType, group);
  }

  // Step 4: Consolidate groups with >= 3 members
  for (const [taskType, group] of groups) {
    if (group.length < 3) continue;

    const consolidated = createConsolidatedMemory(group, taskType);
    state.memories.push(consolidated);

    // Mark sources as consolidated
    for (const source of group) {
      const idx = state.memories.findIndex(m => m.id === source.id);
      if (idx !== -1) {
        state.memories[idx].status = "consolidated";
      }
    }

    const saved = estimateTokensSaved(group, consolidated);
    tokensFreed += saved;
    consolidatedCount += group.length;

    // Log
    const entry: ConsolidationEntry = {
      timestamp: nowISO,
      action: ConsolidationAction.CONSOLIDATE,
      sourceMemoryIds: group.map(m => m.id),
      targetMemoryId: consolidated.id,
      tokensSaved: saved,
      qualityEstimate: estimateQualityDegradation(group),
      reason: `${group.length} low-utility ${taskType} memories consolidated to free space`,
    };
    state.consolidationLog.push(entry);
  }

  // Step 5: Archive cold memories
  const coldCandidates = activeMemories.filter(m => {
    return m.retrievalCount < COLD_STORAGE_RETRIEVAL_COUNT &&
      m.causalUtility < 0.2 &&
      !lowUtility.includes(m);
  });

  const toArchive = coldCandidates.slice(0, 50); // cap per pass
  for (const m of toArchive) {
    m.status = "cold";
    m.coldSince = nowISO;
    state.coldStorage.push(m);
    // Remove from active
    state.memories = state.memories.filter(mem => mem.id !== m.id);

    archivedCount += 1;
    tokensFreed += m.hypothesis.length + m.insight.length;

    const entry: ConsolidationEntry = {
      timestamp: nowISO,
      action: ConsolidationAction.ARCHIVE_COLD_STORAGE,
      sourceMemoryIds: [m.id],
      targetMemoryId: null,
      tokensSaved: m.hypothesis.length + m.insight.length,
      qualityEstimate: m.qualityScore,
      reason: `Cold: ${m.retrievalCount} retrievals, utility=${m.causalUtility.toFixed(3)}`,
    };
    state.consolidationLog.push(entry);
  }

  return {
    newState: state,
    result: { consolidatedCount, archivedCount, tokensFreed },
  };
}

/**
 * createConsolidatedMemory: merges a group of similar memories into one summary.
 *
 * Aggregates:
 *   - Thompson params: sum of individual alphas/betas
 *   - Causal utility: weighted average
 *   - Hypothesis/insight: template-based summary
 *   - consolidationCount: number of source memories
 */
function createConsolidatedMemory(
  sources: IndexedMemory[],
  taskType: TaskType,
): IndexedMemory {
  const now = new Date().toISOString();
  const id = `consolidated-${taskType}-${Date.now()}`;

  const totalAlpha = sources.reduce((s, m) => s + m.thompsonAlpha, 0);
  const totalBeta = sources.reduce((s, m) => s + m.thompsonBeta, 0);
  const avgCausal = sources.reduce((s, m) => s + m.causalUtility, 0) / sources.length;
  const avgQuality = sources.reduce((s, m) => s + m.qualityScore, 0) / sources.length;

  const insight = generateConsolidationInsight(sources, taskType);

  return {
    id,
    sourceTrialIds: sources.flatMap(m => m.sourceTrialIds),
    createdAt: now,
    lastRetrievedAt: null,
    retrievalCount: 0,
    taskType,
    capabilityRequirements: [...new Set(sources.flatMap(m => m.capabilityRequirements))],
    hypothesis: `[Consolidated ${sources.length} tasks re ${taskType}]`,
    insight,
    counterfactuals: [],
    outcome: avgQuality >= 0.5 ? "success" : "failure",
    qualityScore: avgQuality,
    compressionRatio: sources.reduce((s, m) => s + m.compressionRatio, 0) / sources.length,
    modelTier: "medium",
    paramsUsed: sources[0].paramsUsed,
    thompsonAlpha: totalAlpha,
    thompsonBeta: totalBeta,
    causalUtility: avgCausal,
    retrievalUtility: 0.5,
    tdError: 0,
    surprise: 0,
    consolidationCount: sources.length,
    status: "active",
    coldSince: null,
    expiresAt: null,
  };
}

/**
 * estimateQualityDegradation: estimates information loss from merging.
 *
 * degradation = std(qualityScores) * 0.5
 * Higher variance → worse consolidation → less reliable summary.
 */
export function estimateQualityDegradation(sources: IndexedMemory[]): number {
  if (sources.length < 2) return 0;
  const scores = sources.map(m => m.qualityScore);
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  const variance = scores.reduce((s, v) => s + (v - mean) ** 2, 0) / scores.length;
  return Math.sqrt(variance) * 0.5;
}

/**
 * estimateTokensSaved: approximate tokens freed by consolidation.
 *
 * tokensSaved = Σ source.insight.length - consolidated.insight.length
 */
export function estimateTokensSaved(
  sources: IndexedMemory[],
  consolidated: IndexedMemory,
): number {
  // Approximate token count: ~4 chars per token (GPT-family tokenizers).
  // Raw character count overestimates savings for CJK text (~1-2 chars/token).
  const sourceTokens = sources.reduce((s, m) => s + m.hypothesis.length + m.insight.length, 0) / 4;
  const consolidatedTokens = (consolidated.hypothesis.length + consolidated.insight.length) / 4;
  return Math.max(0, Math.round(sourceTokens - consolidatedTokens));
}

/**
 * generateConsolidationInsight: creates a summary insight from a group.
 */
function generateConsolidationInsight(
  sources: IndexedMemory[],
  taskType: TaskType,
): string {
  const successes = sources.filter(m => m.outcome === "success").length;
  const failures = sources.filter(m => m.outcome === "failure").length;
  const crashes = sources.filter(m => m.outcome === "crash").length;
  const best = sources.reduce((a, b) => a.qualityScore > b.qualityScore ? a : b);
  const medianCompression = [...sources].sort((a, b) => a.compressionRatio - b.compressionRatio)[Math.floor(sources.length / 2)];

  return (
    `${taskType}: ${successes} successes, ${failures} failures, ` +
    `${crashes} crashes. Best params: compressed at ` +
    `${best.compressionRatio.toFixed(2)} with ${best.modelTier} ` +
    `(quality ${best.qualityScore.toFixed(2)}). ` +
    `Median compression: ${medianCompression?.compressionRatio.toFixed(2) || "N/A"}. ` +
    `Key finding: ${best.insight.slice(0, 150)}`
  );
}
