// ============================================================================
// Turbocontext v5 — 7-Dimension MMR Retrieval
// ============================================================================
// Full retrieval scoring and MMR diversity re-ranking on IndexedMemory[].
//
// Replaces the trivial filter in rl-engine.ts:
//   filter(m => m.taskType === input.taskType).slice(0, topK)
//
// The 7 scoring dimensions (matching PolicyRetrieval.dimWeights):
//   1. idfOverlap        (w=0.25) — IDF-weighted keyword overlap
//   2. capabilityJaccard (w=0.20) — Jaccard similarity of capability reqs
//   3. taskTypeMatch     (w=0.10) — Exact > family prefix > other
//   4. recencyDecay      (w=0.15) — exp(-decay * days since last retrieval)
//   5. outcomeBonus      (w=0.10) — Success=1.0, failure=0.3, crash=0.1
//   6. infoDensity       (w=0.10) — Normalized information density
//   7. thompsonUtility   (w=0.10) — Beta sample from (alpha, beta) posterior
//
// After scoring, MMR diversity re-ranking ensures the final topK memories
// are diverse (not all from the same task type or with overlapping content).
// ============================================================================

import type { IndexedMemory, IDFCache } from "../types.js";
import type { PolicyRetrieval } from "../types.js";
import { sampleBeta } from "./thompson.js";

// ── Public API ──

export interface ScoredMemory {
  memory: IndexedMemory;
  score: number;
  dimScores: Record<string, number>;
}

export interface RetrievalQuery {
  taskType: string;
  description: string;
  capabilityRequirements?: string[];
}

/**
 * retrieveMemories: scores + ranks active memories using 7-dim scoring
 * followed by MMR diversity re-ranking.
 *
 * Process:
 *   1. Score all active memories on 7 dimensions
 *   2. Select top poolSize candidates (topK * 2.5)
 *   3. MMR re-rank with lambda from policy (default ~0.70)
 *   4. Return topK memories
 *
 * @param memories — all IndexedMemory entries (active + cold)
 * @param input — task description and type for scoring
 * @param policy — retrieval sub-policy with weights, lambda, topK
 * @param idfCache — optional IDF weights for rare-term boosting
 * @param rndBonusFn — optional function to add RND-based bonus per memory
 */
export function retrieveMemories(
  memories: IndexedMemory[],
  input: RetrievalQuery,
  policy: PolicyRetrieval,
  idfCache?: IDFCache,
  rndBonusFn?: (mem: IndexedMemory) => number,
): IndexedMemory[] {
  // Step 0: filter to active memories only
  const active = memories.filter(m => m.status === "active");
  if (active.length === 0) return [];

  // Step 1: score all active memories
  const scored: ScoredMemory[] = active.map(mem => ({
    memory: mem,
    score: scoreMemory(mem, input, policy.dimWeights, idfCache, rndBonusFn),
    dimScores: {},
  }));

  // Step 2: sort by score descending, take pool
  const poolSize = Math.min(
    Math.max(policy.topK * 3, policy.topK),
    scored.length,
  );
  scored.sort((a, b) => b.score - a.score);
  const pool = scored.slice(0, poolSize);

  // Step 3: MMR diversity re-ranking
  const selected = mmrReRank(pool, policy.topK, policy.mmrLambda);

  // Step 4: return memories in MMR order
  return selected.map(s => s.memory);
}

// ── Scoring ──

/**
 * scoreMemory: computes weighted sum of 7 dimension scores for one memory.
 */
export function scoreMemory(
  mem: IndexedMemory,
  input: RetrievalQuery,
  weights: Record<string, number>,
  idfCache?: IDFCache,
  rndBonusFn?: (mem: IndexedMemory) => number,
): number {
  const dims = computeDimensionScores(mem, input, idfCache, rndBonusFn);
  let score = 0;
  for (const [dim, weight] of Object.entries(weights)) {
    score += weight * (dims[dim] ?? 0);
  }
  return score;
}

/**
 * computeDimensionScores: computes raw scores for all 7 dimensions.
 */
export function computeDimensionScores(
  mem: IndexedMemory,
  input: RetrievalQuery,
  idfCache?: IDFCache,
  rndBonusFn?: (mem: IndexedMemory) => number,
): Record<string, number> {
  const now = Date.now();
  const daysSinceRetrieval = mem.lastRetrievedAt
    ? (now - new Date(mem.lastRetrievedAt).getTime()) / 86400000
    : 365; // never retrieved → treated as very old

  return {
    // 1. IDF-weighted keyword overlap
    idfOverlap: computeIDFOverlap(
      input.description,
      `${mem.hypothesis ?? ""} ${mem.insight ?? ""}`,
      idfCache,
    ),

    // 2. Jaccard similarity of capability requirements
    capabilityJaccard: computeCapabilityJaccard(
      input.capabilityRequirements ?? [],
      mem.capabilityRequirements ?? [],
    ),

    // 3. Task type match
    taskTypeMatch: computeTaskTypeMatch(input.taskType, mem.taskType),

    // 4. Exponential recency decay
    recencyDecay: Math.exp(-0.05 * daysSinceRetrieval),

    // 5. Outcome bonus
    outcomeBonus: mem.outcome === "success" ? 1.0
                : mem.outcome === "failure" ? 0.3
                : 0.1,

    // 6. Information density (normalized)
    infoDensity: computeInfoDensity(mem, daysSinceRetrieval),

    // 7. Thompson utility from Beta posterior
    // NOTE: retrievalUtility is a number (cached sample), not an object.
    // thompsonAlpha/Beta live directly on IndexedMemory.
    thompsonUtility: sampleBeta(
      mem.thompsonAlpha ?? 1,
      mem.thompsonBeta ?? 1,
    ),
  };
}

// ── Dimension implementations ──

/**
 * IDF overlap: tokenize both texts, weight by IDF, compute cosine similarity.
 * Rare terms (high IDF weight) contribute more to the overlap score.
 */
export function computeIDFOverlap(
  query: string,
  doc: string,
  idfCache?: IDFCache,
): number {
  const qTokens = tokenize(query);
  const dTokens = new Set(tokenize(doc));

  if (qTokens.length === 0) return 0;

  let overlap = 0;
  let queryWeight = 0;
  for (const token of qTokens) {
    const idf = idfCache?.weights?.[token] ?? 1.0;
    queryWeight += idf;
    if (dTokens.has(token)) {
      overlap += idf;
    }
  }
  return queryWeight > 0 ? overlap / queryWeight : 0;
}

/**
 * Capability Jaccard: |A ∩ B| / |A ∪ B|.
 */
export function computeCapabilityJaccard(
  inputReqs: string[],
  memReqs: string[],
): number {
  if (inputReqs.length === 0 && memReqs.length === 0) return 1.0;
  if (inputReqs.length === 0 || memReqs.length === 0) return 0;

  const setA = new Set(inputReqs);
  const setB = new Set(memReqs);

  let intersection = 0;
  for (const item of setA) {
    if (setB.has(item)) intersection++;
  }

  const union = new Set([...setA, ...setB]).size;
  return union > 0 ? intersection / union : 0;
}

/**
 * Task type match:
 *   Exact match → 1.0
 *   Same family prefix (e.g., "code_review" ↔ "code_generation") → 0.5
 *   Different family → 0.0
 */
export function computeTaskTypeMatch(
  inputType: string,
  memType: string,
): number {
  if (inputType === memType) return 1.0;
  const inFamily = inputType.split("_")[0] ?? "";
  const memFamily = memType.split("_")[0] ?? "";
  if (inFamily && inFamily === memFamily) return 0.5;
  return 0;
}

/**
 * Information density: normalized measure of how "rich" a memory is.
 * Factors in the length of counterfactuals/insight, retrieval count, and recency.
 */
export function computeInfoDensity(
  mem: IndexedMemory,
  daysSinceRetrieval: number,
): number {
  const insightLen = (mem.insight?.length ?? 0) + (mem.hypothesis?.length ?? 0);
  const density = Math.min(insightLen / 500, 1.0); // cap at 500 chars
  const retrievalNorm = Math.min((mem.retrievalCount ?? 0) / 10, 1.0); // cap at 10 retrievals
  const recencyNorm = Math.exp(-0.02 * daysSinceRetrieval);

  return 0.4 * density + 0.3 * retrievalNorm + 0.3 * recencyNorm;
}

// ── MMR diversity re-ranking ──

/**
 * MMR (Maximal Marginal Relevance) greedy selection.
 *
 * For each selection step:
 *   mmrScore = lambda * relevanceScore - (1 - lambda) * maxSimilarityToSelected
 *
 * lambda=1.0 → pure relevance ranking
 * lambda=0.0 → pure diversity (maximize dissimilarity to already selected)
 * Default lambda=0.70 → bias toward relevance but enforce diversity
 */
export function mmrReRank(
  candidates: ScoredMemory[],
  topK: number,
  mmrLambda: number,
): ScoredMemory[] {
  if (candidates.length <= topK) return candidates;

  const selected: ScoredMemory[] = [];
  const remaining = [...candidates];

  // First selection: highest-scored item
  remaining.sort((a, b) => b.score - a.score);

  while (selected.length < topK && remaining.length > 0) {
    let bestIdx = 0;
    let bestMmr = -Infinity;

    for (let i = 0; i < remaining.length; i++) {
      const relevance = remaining[i].score;
      let maxSim = 0;
      for (const s of selected) {
        const sim = computeMemorySimilarity(remaining[i].memory, s.memory);
        if (sim > maxSim) maxSim = sim;
      }
      const mmr = mmrLambda * relevance - (1 - mmrLambda) * maxSim * 10 // amplify diversity penalty
        + entropyBonus(remaining[i].memory, selected.map(s => s.memory)) * 0.15; // outcome diversity
      if (mmr > bestMmr) {
        bestMmr = mmr;
        bestIdx = i;
      }
    }

    selected.push(remaining[bestIdx]);
    remaining.splice(bestIdx, 1);
  }

  return selected;
}

/**
 * computeMemorySimilarity: simple similarity measure between two memories.
 * Based on shared task type family, capability overlap, and outcome.
 *
 * Karpathy-inspired: also considers outcome diversity. Two memories with
 * the same outcome are more "similar" (redundant) than memories with
 * different outcomes. This feeds into MMR to prevent outcome monoculture.
 */
export function computeMemorySimilarity(a: IndexedMemory, b: IndexedMemory): number {
  let sim = 0;

  // Task type: 1.0 for exact, 0.5 for same family, 0 for different
  if (a.taskType === b.taskType) sim += 0.5;
  else {
    const aFamily = (a.taskType as string).split("_")[0];
    const bFamily = (b.taskType as string).split("_")[0];
    if (aFamily && aFamily === bFamily) sim += 0.25;
  }

  // Outcome match: success↔success or failure↔failure boosts similarity
  if (a.outcome === b.outcome) sim += 0.25;

  // Capability overlap (if both have capabilities)
  if (a.capabilityRequirements?.length && b.capabilityRequirements?.length) {
    sim += 0.25 * computeCapabilityJaccard(
      a.capabilityRequirements,
      b.capabilityRequirements,
    );
  }

  return Math.min(sim, 1.0);
}

/**
 * entropyBonus: Karpathy-inspired information-theoretic diversity bonus.
 *
 * Rewards selecting memories with outcome distributions different from the
 * already-selected set. If all selected memories are successes, a failure
 * memory gets a high entropy bonus (it adds outcome diversity).
 *
 * Formula: bonus = -log(p_outcome(item) | selected_outcomes) * scale
 * Rare outcomes in the selected set get larger bonuses.
 *
 * This is the key insight from Karpathy's autoresearch: retrieval diversity
 * shouldn't just be about task types — it should be about OUTCOME diversity.
 * Showing the planner both successes AND failures leads to better decisions.
 */
export function entropyBonus(
  item: IndexedMemory,
  selected: IndexedMemory[],
): number {
  if (selected.length === 0) return 0.0;

  // Outcome distribution of already-selected memories
  const outcomeCounts: Record<string, number> = {};
  for (const sel of selected) {
    const o = sel.outcome;
    outcomeCounts[o] = (outcomeCounts[o] || 0) + 1;
  }

  const itemOutcome = item.outcome;
  const n = selected.length;

  // Probability of NOT seeing this outcome in current selection
  const pCurrent = (outcomeCounts[itemOutcome] || 0) / n;

  // If this outcome is underrepresented, entropy bonus is high
  // Use -log(p + ε) form: rare outcomes get large bonus
  const epsilon = 0.1;
  const bonus = -Math.log(pCurrent + epsilon) * 0.5; // scale factor
  return Math.max(0.0, bonus);
}

// ── Tokenizer ──

// Simple word tokenizer: lowercase, split on non-alphanumeric, min length 2
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9_]+/)
    .filter(t => t.length >= 2);
}
