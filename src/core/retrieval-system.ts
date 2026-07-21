// ============================================================
// v3.1 — Retrieval System (extracted from learner.ts)
// ============================================================
// Plateau detection, strategic directives, contrastive pairs,
// MMR lambda, IDF cache management.
//
// All functions are stateless — they take state as parameters
// and return results. The Learner class owns the state and
// delegates computation to these functions.
// ============================================================

import type {
  ExecutionRecord, TurboContextConfig,
  TaskType, BranchState,
  PlateauSignal, PlateauReason, PlateauRuleResult,
  StrategicDirective, ContrastivePair, IDFCache,
  RLExecutionRecord, RetrievalWeights, DEFAULT_RETRIEVAL_WEIGHTS,
  TwoPhaseRetrievalConfig, DEFAULT_TWO_PHASE_CONFIG,
} from "../types.js";
import {
  curiosityBonus, computeAdvantageForMemory,
  thompsonSample, computeSubsystemBaselines,
} from "./rl-core.js";

// ------------------------------------------------------------------
// Plateau Detection (4 rules from autoresearch)
// ------------------------------------------------------------------

/**
 * 多规则平台期检测（autoresearch: quantitative branch health signals）。
 *
 * 4 条检测规则，每条有独立置信度:
 *   1. improvement_stall: 最近 3 次 vs 前 2 次无改进，且速度平坦
 *   2. crash_dominant:     崩溃率 > 成功率 * 2
 *   3. novelty_collapse:   最近假设几乎相同（novelty < 0.15）
 *   4. slow_decline:       后半段平均质量 < 前半段
 */
export function detectPlateau(
  branches: Map<TaskType, BranchState>,
  taskType: TaskType,
): PlateauSignal {
  const branch = branches.get(taskType);
  const rules: PlateauRuleResult[] = [];

  if (!branch || branch.totalExperiments < 3) {
    return {
      isPlateaued: false,
      reason: "none",
      confidence: 0,
      rules: [{ rule: "insufficient_data", triggered: false, confidence: 0, detail: "Need ≥3 experiments" }],
    };
  }

  const total = branch.totalExperiments;
  const vel = branch.trajectory.improvementVelocity;
  const successes = branch.successCount;

  // Rule 1: Improvement stall
  const qualityHist = branch.trajectory.qualityHistory;
  let rule1Triggered = false;
  let rule1Confidence = 0;
  let rule1Detail = "";

  if (qualityHist.length >= 5) {
    const recent3 = qualityHist.slice(-3);
    const prior2 = qualityHist.slice(-5, -3);
    const recent3min = Math.min(...recent3);
    const prior2min = Math.min(...prior2);
    if (prior2min > 0 && recent3min >= prior2min && Math.abs(vel) < 0.005) {
      rule1Triggered = true;
      rule1Confidence = 0.85;
      rule1Detail = `No improvement: recent min=${recent3min.toFixed(3)} >= prior min=${prior2min.toFixed(3)}, vel=${vel.toFixed(4)}`;
    }
  }
  rules.push({ rule: "improvement_stall", triggered: rule1Triggered, confidence: rule1Confidence, detail: rule1Detail || "OK" });

  // Rule 2: Crash dominance
  let rule2Triggered = false;
  let rule2Confidence = 0;
  let rule2Detail = "";
  const crashCount = branch.recentFailures.filter(f =>
    f.failureReasons.some(r => r === "low_completeness" || r === "retries_exhausted")
  ).length;
  if (total >= 4 && crashCount > successes * 2) {
    rule2Triggered = true;
    rule2Confidence = 0.90;
    rule2Detail = `Crash dominant: ${crashCount} crashes vs ${successes} successes`;
  }
  rules.push({ rule: "crash_dominant", triggered: rule2Triggered, confidence: rule2Confidence, detail: rule2Detail || "OK" });

  // Rule 3: Novelty collapse
  const novelty = branch.trajectory.noveltyScore;
  let rule3Triggered = false;
  let rule3Confidence = 0;
  let rule3Detail = "";
  if (total >= 5 && novelty < 0.15) {
    rule3Triggered = true;
    rule3Confidence = 0.75;
    rule3Detail = `Novelty collapsed: ${novelty.toFixed(2)} (all recent tasks nearly identical)`;
  }
  rules.push({ rule: "novelty_collapse", triggered: rule3Triggered, confidence: rule3Confidence, detail: rule3Detail || "OK" });

  // Rule 4: Slow decline
  let rule4Triggered = false;
  let rule4Confidence = 0;
  let rule4Detail = "";
  if (qualityHist.length >= 6) {
    const half = Math.floor(qualityHist.length / 2);
    const firstHalf = qualityHist.slice(0, half);
    const secondHalf = qualityHist.slice(half);
    const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
    const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
    if (secondAvg < firstAvg * 0.995 && firstAvg > 0) {
      rule4Triggered = true;
      rule4Confidence = 0.60;
      rule4Detail = `Slow decline: 2nd half avg=${secondAvg.toFixed(3)} < 1st half avg=${firstAvg.toFixed(3)}`;
    }
  }
  rules.push({ rule: "slow_decline", triggered: rule4Triggered, confidence: rule4Confidence, detail: rule4Detail || "OK" });

  // 汇总：选置信度最高的触发规则
  const triggered = rules.filter(r => r.triggered);
  if (triggered.length === 0) {
    return { isPlateaued: false, reason: "none", confidence: 0, rules };
  }

  const best = triggered.reduce((a, b) => a.confidence > b.confidence ? a : b);
  const reason: PlateauReason = best.rule as PlateauReason;

  return {
    isPlateaued: true,
    reason,
    confidence: best.confidence,
    rules,
  };
}

// ------------------------------------------------------------------
// Strategic Directives (6 directive types)
// ------------------------------------------------------------------

/**
 * 生成战略指令（autoresearch: high-level planner guidance）。
 *
 * 根据平台期检测和分支指标生成具体行动指导:
 *   MOMENTUM  → 深入利用当前方向
 *   PLATEAU   → 切换到未探索的任务类型
 *   CAUTION   → 优先更小、更安全的变更
 *   DIVERSIFY → 尝试不同的角度
 *   STEADY    → 适度探索
 *   EXPLORE   → 实验太少，尝试多样化初始变更
 */
export function generateStrategicDirective(
  branches: Map<TaskType, BranchState>,
  taskType: TaskType,
  getActiveBranches: () => TaskType[],
): StrategicDirective {
  const branch = branches.get(taskType);
  const total = branch?.totalExperiments ?? 0;
  const vel = branch?.trajectory.improvementVelocity ?? 0;
  const stability = branch?.trajectory.stabilityScore ?? 0.5;
  const novelty = branch?.trajectory.noveltyScore ?? 0.5;
  const successes = branch?.successCount ?? 0;
  const failures = branch?.failureCount ?? 0;
  const successRate = total > 0 ? successes / total : 0;

  const metrics = { velocity: vel, stability, novelty, totalExperiments: total, successRate };

  // 太少实验 → 探索
  if (total < 3) {
    return {
      directive: "EXPLORE",
      message: `Too few experiments on '${taskType}' (${total} total). Try diverse initial changes.`,
      metrics,
      suggestedAction: "Try 2-3 different task types to establish baselines",
    };
  }

  const plateau = detectPlateau(branches, taskType);

  // 平台期 → 切换
  if (plateau.isPlateaued && plateau.confidence > 0.7) {
    if (plateau.reason === "crash_dominant") {
      return {
        directive: "CAUTION",
        message: `High crash rate on '${taskType}' (${failures} failures vs ${successes} successes). Prefer smaller, safer changes.`,
        metrics,
        suggestedAction: "Reduce task complexity; use simpler, well-tested mutation types",
      };
    }
    if (plateau.reason === "novelty_collapse") {
      return {
        directive: "DIVERSIFY",
        message: `Novelty collapsed on '${taskType}' (${novelty.toFixed(2)}). Recent tasks are too similar — try a different angle.`,
        metrics,
        suggestedAction: "Switch to a different task type or use a radically different approach",
      };
    }
    const activeBranches = getActiveBranches();
    const otherBranches = activeBranches.filter(b => b !== taskType);
    const suggestion = otherBranches.length > 0
      ? `Switch to: ${otherBranches.slice(0, 3).join(", ")}`
      : "Try orthogonal approach on same branch";
    return {
      directive: "PLATEAU",
      message: `Plateau on '${taskType}' (${plateau.reason}, confidence=${plateau.confidence.toFixed(2)}). Strongly consider switching.`,
      metrics,
      suggestedAction: suggestion,
    };
  }

  // 改进中且稳定 → 深入利用
  if (total >= 3 && vel > 0.005 && stability >= 0.4) {
    return {
      directive: "MOMENTUM",
      message: `Improving on '${taskType}' (vel=+${(vel * 100).toFixed(2)}%/exp, stability=${stability.toFixed(2)}). Exploit deeper.`,
      metrics,
      suggestedAction: "Build on recent successes; try combining complementary improvements",
    };
  }

  // 低新颖性 → 多样化
  if (total >= 5 && novelty < 0.3) {
    return {
      directive: "DIVERSIFY",
      message: `Low novelty on '${taskType}' (${novelty.toFixed(2)}). Try a different angle.`,
      metrics,
      suggestedAction: "Use a different mutation type or target a different quality dimension",
    };
  }

  // 默认 → 稳定探索
  return {
    directive: "STEADY",
    message: `Steady state on '${taskType}' — moderate exploration.`,
    metrics,
    suggestedAction: "Continue with current approach, test one new variation",
  };
}

// ------------------------------------------------------------------
// Adaptive MMR Lambda
// ------------------------------------------------------------------

/**
 * 自适应 MMR λ（autoresearch: adaptive lambda based on branch state）。
 *
 * 平台期 → 低 λ (更注重多样性，尝试新东西)
 * 动量期 → 高 λ (更注重相关性，深入利用)
 * 默认   → 0.65 (平衡)
 */
export function computeAdaptiveMmrLambda(
  branches: Map<TaskType, BranchState>,
  taskType: TaskType,
): number {
  const branch = branches.get(taskType);
  if (!branch || branch.totalExperiments < 3) return 0.65;

  const plateau = detectPlateau(branches, taskType);
  if (plateau.isPlateaued && plateau.confidence > 0.7) {
    return 0.40; // 平台期：优先多样性
  }

  const vel = branch.trajectory.improvementVelocity;
  if (vel > 0.01) return 0.85;  // 强动量：深度利用
  if (vel > 0.005) return 0.75; // 中等动量
  if (vel > 0.002) return 0.70; // 弱动量

  return 0.65; // 默认平衡
}

// ------------------------------------------------------------------
// Contrastive Pair Discovery
// ------------------------------------------------------------------

/**
 * 提取执行记录的特征向量（供对比对发现使用）。
 */
export function extractRecordFeatures(record: ExecutionRecord): string[] {
  const features: string[] = [];
  features.push(`type:${record.taskType}`);
  features.push(`model:${record.modelUsed}`);

  if (record.coverage) {
    for (const [cap, cov] of Object.entries(record.coverage)) {
      if (cov > 0.5) features.push(`cap:${cap}`);
    }
  }

  if (record.sourceFiles) {
    for (const f of record.sourceFiles) {
      const parts = f.split("/");
      if (parts.length >= 1) features.push(`file:${parts[parts.length - 1].replace(/\.[^.]+$/, "")}`);
    }
  }

  if (record.dimensionScores) {
    for (const [dim, score] of Object.entries(record.dimensionScores)) {
      if (score < 0.5) features.push(`weak:${dim}`);
      if (score > 0.85) features.push(`strong:${dim}`);
    }
  }

  return features;
}

/**
 * 发现对比对：相似任务类型但相反结果的实验。
 *
 * 这些对比对给算法提供因果洞察：
 * "为什么相似的实验一个成功一个失败？"
 */
export function findContrastivePairs(
  globalHistory: ExecutionRecord[],
  taskType: TaskType,
  qualityThreshold: number,
  maxAttempts: number,
  nPairs: number = 2,
): ContrastivePair[] {
  const successRecords = globalHistory.filter(r =>
    r.qualityScore >= qualityThreshold
  );
  const failureRecords = globalHistory.filter(r =>
    r.qualityScore < qualityThreshold
  );

  if (successRecords.length === 0 || failureRecords.length === 0) return [];

  const pairs: Array<{ score: number; pair: ContrastivePair }> = [];

  for (const success of successRecords) {
    const sFeatures = extractRecordFeatures(success);

    for (const failure of failureRecords) {
      if (success.taskType === failure.taskType && success.taskId === failure.taskId) continue;

      const fFeatures = extractRecordFeatures(failure);

      const intersection = sFeatures.filter(f => fFeatures.includes(f));
      const union = new Set([...sFeatures, ...fFeatures]);
      if (union.size === 0) continue;

      const jaccard = intersection.length / union.size;
      if (jaccard < 0.2) continue;

      const qualityDiff = success.qualityScore - failure.qualityScore;
      const recencyBonus = Math.exp(-0.5 * Math.max(0,
        (globalHistory.length - globalHistory.indexOf(failure)) / 5
      ));
      const score = jaccard * Math.abs(qualityDiff) * (0.5 + recencyBonus * 0.5);

      const sharedFeatures = intersection.slice(0, 5);

      const insight = [
        `Similar features [${sharedFeatures.slice(0, 3).join(", ")}]:`,
        `'${success.taskId}' succeeded (q=${(success.qualityScore * 100).toFixed(0)}%)`,
        `but '${failure.taskId}' failed (q=${(failure.qualityScore * 100).toFixed(0)}%) —`,
        `key difference may be task type context or compression strategy.`,
      ].join(" ");

      pairs.push({
        score,
        pair: {
          success: {
            taskType: success.taskType,
            description: success.taskId,
            quality: success.qualityScore,
            sourceFiles: success.sourceFiles || [],
          },
          failure: {
            taskType: failure.taskType,
            description: failure.taskId,
            quality: failure.qualityScore,
            sourceFiles: failure.sourceFiles || [],
            failureMode: failure.attemptCount >= maxAttempts ? "retries_exhausted" : "low_quality",
          },
          sharedFeatures,
          similarity: Math.round(jaccard * 1000) / 1000,
          insight,
        },
      });
    }
  }

  pairs.sort((a, b) => b.score - a.score);

  const seen = new Set<string>();
  const unique: ContrastivePair[] = [];
  for (const { pair } of pairs) {
    const key = `${pair.success.description}::${pair.failure.description}`;
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(pair);
      if (unique.length >= nPairs) break;
    }
  }

  return unique;
}

// ------------------------------------------------------------------
// Future Directions Synthesis
// ------------------------------------------------------------------

/**
 * 合成未来方向（autoresearch: rule-based synthesis）。
 * 根据实验结果自动生成下一步建议，无需额外 LLM 调用。
 */
export function synthesizeFutureDirections(
  branches: Map<TaskType, BranchState>,
  taskType: TaskType,
  qualityScore: number,
  attemptCount: number,
  isSuccess: boolean,
  isCrash: boolean,
  maxAttempts: number,
): string {
  const branch = branches.get(taskType);

  if (isCrash) {
    return "Avoid approach that caused crash; consider lower complexity or safer mutation type";
  }

  if (isSuccess) {
    const bestQ = branch?.bestQuality ?? 0;
    const improvement = qualityScore - bestQ;
    if (improvement > 0.05) {
      return `Strong improvement (+${(improvement * 100).toFixed(1)}%) on ${taskType}; explore deeper or combine with complementary changes`;
    }
    if (improvement > 0) {
      return `Modest improvement on ${taskType}; try refining further or testing on related task types`;
    }
    return `New best for ${taskType}; solidify by testing with variations`;
  }

  if (attemptCount >= maxAttempts) {
    return `Retries exhausted on ${taskType}; consider simpler task or lower quality threshold`;
  }
  if (qualityScore < 0.5) {
    return `Very low quality on ${taskType} (${(qualityScore * 100).toFixed(0)}%); re-evaluate approach or switch task type`;
  }
  return `Marginal failure on ${taskType} (${(qualityScore * 100).toFixed(0)}%); try orthogonal change or adjust compression weights`;
}

// ------------------------------------------------------------------
// Global IDF Cache Management
// ------------------------------------------------------------------

/** 英文停用词集合 */
const STOP_WORDS_SET = new Set([
  "the", "a", "an", "is", "are", "was", "were", "be", "been",
  "being", "have", "has", "had", "do", "does", "did", "will",
  "would", "could", "should", "may", "might", "shall", "can",
  "to", "of", "in", "for", "on", "with", "at", "by", "from",
  "as", "into", "through", "during", "before", "after", "above",
  "below", "between", "out", "off", "over", "under", "again",
  "further", "then", "once", "here", "there", "when", "where",
  "why", "how", "all", "each", "every", "both", "few", "more",
  "most", "other", "some", "such", "no", "nor", "not", "only",
  "own", "same", "so", "than", "too", "very", "just", "it", "its",
  "this", "that", "these", "those", "which", "what", "who", "whom",
  "and", "but", "or", "if", "because", "about", "up", "we", "our",
]);

/**
 * 更新 IDF 缓存。
 *
 * 增量更新策略:
 *   - 空缓存 → 完全重建
 *   - 文档数变化 >20% → 完全重建
 *   - 超过 1 小时未更新 → 完全重建
 *   - 否则保持（IDF 对低频变化不敏感）
 */
export function updateIDFCache(
  idfCache: IDFCache,
  fragments: Array<{ content: string }>,
): void {
  const newDocCount = fragments.length;
  const existingCount = idfCache.documentCount;

  const needsRebuild = existingCount === 0
    || Math.abs(newDocCount - existingCount) / Math.max(1, existingCount) > 0.2
    || (Date.now() - idfCache.lastUpdated) > 3600000;

  if (!needsRebuild) return;

  const df: Record<string, number> = {};
  for (const frag of fragments) {
    const words = new Set(
      frag.content.toLowerCase()
        .split(/[\s.,;:!?()\[\]{}"'`\n\r\t<>=\-+*/]+/)
        .filter(w => w.length > 3 && !STOP_WORDS_SET.has(w))
    );
    for (const w of words) {
      df[w] = (df[w] || 0) + 1;
    }
  }

  const weights: Record<string, number> = {};
  for (const [w, count] of Object.entries(df)) {
    weights[w] = Math.log((newDocCount + 2) / (count + 1)) + 0.5;
  }

  idfCache.weights = weights;
  idfCache.documentCount = newDocCount;
  idfCache.lastUpdated = Date.now();
  if (!idfCache.stopWords || idfCache.stopWords.size === 0) {
    idfCache.stopWords = STOP_WORDS_SET;
  }
}

/**
 * 创建默认 IDF 缓存。
 */
export function createDefaultIDFCache(): IDFCache {
  return {
    weights: {},
    documentCount: 0,
    lastUpdated: 0,
    stopWords: STOP_WORDS_SET,
  };
}

// ============================================================
// v3.3 — RL-powered retrieval enhancements
// ============================================================

/**
 * Thompson-sampled source memory boost for compressor scoring.
 *
 * Each source file maintains Beta(α,β) over its contribution value.
 * Instead of a static δ·outcome bonus, we sample from the distribution,
 * naturally balancing exploration (uncertain files occasionally boosted)
 * and exploitation (known-good files consistently boosted).
 *
 * α = 1 + successCount * 2  (successes observed)
 * β  = 1 + failureCount * 0.5  (failures observed)
 */
export function thompsonSourceBoost(
  attempts: number,
  successes: number,
): number {
  // Default: Beta(1,1) = uniform [0,1], mapped to [-0.05, +0.10] range
  const alpha = 1.0 + successes * 2.0;
  const beta = 1.0 + Math.max(0, attempts - successes) * 0.5;

  // Gamma method for Beta sampling
  const a = Math.max(0.1, alpha);
  const b = Math.max(0.1, beta);

  // Simple approximation: use mean + scaled variance for efficiency
  // Beta(α,β) mean = α/(α+β), variance = αβ / ((α+β)²(α+β+1))
  const mean = a / (a + b);
  const variance = (a * b) / ((a + b) * (a + b) * (a + b + 1));
  const stdDev = Math.sqrt(Math.max(0, variance));

  // Use normal approximation to Beta (valid for α,β > 1)
  // Map to boost range: mean~0.5 → boost~0, mean~0.9 → boost~+0.08
  const sampledMean = mean + (Math.random() - 0.5) * stdDev * 2;
  const clampedMean = Math.max(0.1, Math.min(0.9, sampledMean));
  const boost = (clampedMean - 0.5) * 0.20; // [-0.08, +0.08]

  return Math.round(boost * 10000) / 10000;
}

/**
 * Entropy-regularized MMR diversity bonus for context fragment selection.
 *
 * If all selected fragments are from the same content type or source directory,
 * a fragment with a different type/directory gets a bonus.
 * This prevents "content monoculture" where all selected context is
 * from the same subsystem.
 */
export function entropyMMRBonus(
  fragmentSource: string,
  fragmentType: string,
  selectedSources: string[],
  selectedTypes: string[],
): number {
  if (selectedSources.length === 0) return 0.0;

  // Source directory diversity
  const sourceDirs = new Map<string, number>();
  for (const s of selectedSources) {
    const dir = s.split("/").slice(0, -1).join("/") || s;
    sourceDirs.set(dir, (sourceDirs.get(dir) || 0) + 1);
  }
  const fragDir = fragmentSource.split("/").slice(0, -1).join("/") || fragmentSource;
  const n = selectedSources.length;
  const pSource = (sourceDirs.get(fragDir) || 0) / n;

  // Content type diversity
  const typeCounts = new Map<string, number>();
  for (const t of selectedTypes) {
    typeCounts.set(t, (typeCounts.get(t) || 0) + 1);
  }
  const pType = (typeCounts.get(fragmentType) || 0) / n;

  // Combined entropy bonus: reward underrepresented sources and types
  const epsilon = 0.1;
  const sourceBonus = -Math.log(pSource + epsilon) * 0.3;
  const typeBonus = -Math.log(pType + epsilon) * 0.2;

  return Math.max(0.0, sourceBonus + typeBonus);
}

/**
 * Outcome-aware contrastive pair scoring.
 *
 * v3.3 enhancement: beyond subsystem Jaccard overlap, also score pairs
 * where the QUALITY DIMENSIONS differ most. A pair where one execution
 * scored 0.9 on correctness but the other scored 0.3 carries a stronger
 * causal signal than a pair where both scored similarly on all dimensions.
 */
export function contrastiveDimensionScore(
  successDims: { completeness: number; correctness: number; consistency: number; format: number },
  failureDims: { completeness: number; correctness: number; consistency: number; format: number },
): number {
  const dimKeys = ["completeness", "correctness", "consistency", "format"] as const;
  let totalDelta = 0;
  for (const key of dimKeys) {
    const delta = Math.abs((successDims[key] ?? 0.5) - (failureDims[key] ?? 0.5));
    totalDelta += delta;
  }
  // Average dimension delta [0, 1] — higher = more informative contrast
  return Math.round((totalDelta / dimKeys.length) * 10000) / 10000;
}

/**
 * v3.9: Surprise-weighted retrieval bonus.
 *
 * Adapted from agent.py _compute_surprise (lines 1959-1983).
 * Karpathy principle: "Surprise = |predicted - actual|. High surprise →
 * the model's understanding was wrong → high learning value."
 *
 * Sources whose past experiments were highly surprising get a retrieval
 * boost — these are the experiments that taught us something unexpected.
 *
 * @param surpriseScores — surprise scores for a source's past experiments
 * @returns bonus in [0, 3] where 0 = completely predictable, 3 = highly surprising
 */
export function computeSurpriseBonus(surpriseScores: number[]): number {
  if (surpriseScores.length === 0) return 0.0;

  // Average surprise across past experiments
  const avg = surpriseScores.reduce((s, v) => s + v, 0) / surpriseScores.length;

  // Exponentially decaying bonus: mild surprise gets small bonus,
  // extreme surprises get large bonus via squared term
  const bonus = avg * 3.0 + Math.pow(Math.max(0, avg - 0.3), 2) * 5.0;

  return Math.round(Math.min(3.0, Math.max(0.0, bonus)) * 10000) / 10000;
}

/**
 * v3.9: Look up surprise scores for a source from execution history.
 *
 * Scans RL execution records (or regular execution records) to find
 * past experiments involving this source file, and returns their
 * surprise/prediction error scores.
 */
export function collectSurpriseScores(
  source: string,
  history: Array<{ sourceFiles?: string[]; qualityScore: number; dimensionScores?: { completeness: number; correctness: number; consistency: number; format: number } }>,
): number[] {
  const scores: number[] = [];
  for (const rec of history) {
    if (rec.sourceFiles?.includes(source)) {
      // Surprise proxy from dimension score variance
      if (rec.dimensionScores) {
        const dims = [rec.dimensionScores.completeness, rec.dimensionScores.correctness,
                      rec.dimensionScores.consistency, rec.dimensionScores.format];
        const mean = dims.reduce((s, v) => s + v, 0) / 4;
        const variance = dims.reduce((s, v) => s + (v - mean) ** 2, 0) / 4;
        scores.push(Math.sqrt(variance)); // std dev as surprise proxy
      }
    }
  }
  return scores;
}

// ============================================================
// v4.0 — Two-Phase Causal Retrieval (agent.py retrieve_relevant_memories)
// ============================================================
// Karpathy principle: "Learn from data, not hand-crafted rules."
//
// Phase 1: Similarity pool — score all memories on 10 dimensions,
//          keep top oversample × topK candidates.
// Phase 2: Causal re-rank — add advantage-weighted causal utility
//          (measures "did showing this memory actually help?"),
//          then MMR diversity re-ranking with entropy bonus.
//
// The structural difference from single-phase scoring:
//   causal_utility can only be measured for memories that have been
//   retrieved before — it needs downstream outcome data. By separating
//   phases, we avoid penalizing never-retrieved memories (which have
//   default causal_utility=0.5) and let similarity create the pool,
//   then causal evidence picks the best from within it.
// ============================================================

/**
 * v4.0: Normalize surprise bonus for retrieval scoring against global mean.
 *
 * agent.py _compute_surprise (lines 1959-1983):
 * Surprise = |predicted - actual|. High surprise → model was wrong → high learning value.
 *
 * This normalizes individual memory surprise against the global rolling mean,
 * so "surprising for this context" is relative, not absolute.
 */
export function computeRetrievalSurpriseBonus(
  surpriseScore: number,
  globalMeanSurprise: number,
): number {
  // Normalize: if this memory's surprise is above global mean, amplify;
  // if below, dampen (it's not surprising relative to typical outcomes)
  const epsilon = 0.01;
  const normalizationFactor = surpriseScore / Math.max(globalMeanSurprise, epsilon);
  // Raw bonus capped at 3.0, scaled by normalization
  const rawBonus = surpriseScore * 3.0;
  const bonus = rawBonus * Math.min(2.0, normalizationFactor);
  return Math.round(Math.min(3.0, Math.max(0.0, bonus)) * 10000) / 10000;
}

/**
 * v4.0: Compute curiosity/EIG bonus for retrieval scoring.
 *
 * agent.py _curiosity_bonus (lines 2008-2060):
 * Rewards memories from task types where:
 *   1. The predictive model is uncertain (few examples → high variance)
 *   2. The task type has been under-explored relative to others
 *   3. Past experiments in this area had high surprise
 *
 * Normalizes from [0,5] to [0,3] with curriculum-phase weight.
 */
export function computeCuriosityBonusForRetrieval(
  record: RLExecutionRecord,
  allRecords: RLExecutionRecord[],
  curriculumCuriosityWeight: number,
): number {
  const raw = curiosityBonus(record, allRecords);  // [0, 5]
  // Normalize from [0,5] to [0,3] per agent.py pattern (line 2640)
  const normalized = (raw / 5.0) * 3.0 * curriculumCuriosityWeight;
  return Math.round(Math.min(3.0, Math.max(0.0, normalized)) * 10000) / 10000;
}

/**
 * v4.0: Two-phase causal retrieval with MMR diversity re-ranking.
 *
 * This is the core v4 architectural upgrade — matching agent.py's
 * retrieve_relevant_memories() (lines 2505-2737).
 *
 * Phase 1 — Similarity pool:
 *   Score all experiments on 10 dimensions (7 original + surprise + curiosity
 *   + counterfactual), select top oversample × topK candidates.
 *
 * Phase 2 — Causal re-rank:
 *   Add advantage-weighted causal utility (high weight, measures
 *   "did showing this memory actually help?") and re-score the candidate pool.
 *   Then MMR for diversity with entropy bonus.
 *
 * @param experiments - All experiment records with RL fields
 * @param taskType - Current task type for branch matching
 * @param topK - Number of results to return
 * @param config - Two-phase retrieval configuration
 * @param subsystemBaseline - V(subsystem) = average causal_utility per task type
 * @param curriculumParams - Phase-specific surprise and curiosity weights
 * @param retrievalWeights - Scoring dimension weights (may be evolved)
 * @param options - Optional IDF cache, similarity query, MMR lambda
 * @returns Selected experiments, their IDs, phase scores, and pool size
 */
export function twoPhaseCausalRetrieval(
  experiments: RLExecutionRecord[],
  taskType: TaskType,
  topK: number,
  config: TwoPhaseRetrievalConfig,
  subsystemBaseline: Map<string, number>,
  curriculumParams: { surpriseWeight: number; curiosityWeight: number },
  retrievalWeights: RetrievalWeights,
  options?: {
    similarityQuery?: string;
    idfCache?: IDFCache;
    mmrLambda?: number;
  },
): {
  selected: RLExecutionRecord[];
  selectedIds: string[];
  phase1Scores: number[];
  phase2Scores: number[];
  poolSize: number;
} {
  if (experiments.length === 0) {
    return { selected: [], selectedIds: [], phase1Scores: [], phase2Scores: [], poolSize: 0 };
  }

  const total = experiments.length;
  const poolSize = Math.min(
    Math.max(topK * 2 + 3, Math.floor(topK * config.oversampleMultiplier)),
    total,
  );
  const mmrLambda = options?.mmrLambda ?? 0.65;
  const idf = options?.idfCache;

  // Fast path: if few experiments, skip Phase 1 pool selection
  if (total <= topK * 2) {
    const scored = experiments.map((exp, idx) => ({
      score: 0, idx, exp,
    }));
    return {
      selected: experiments.slice(0, topK),
      selectedIds: experiments.slice(0, topK).map(e => e.taskId),
      phase1Scores: new Array(Math.min(topK, total)).fill(0),
      phase2Scores: new Array(Math.min(topK, total)).fill(0),
      poolSize: total,
    };
  }

  // Normalize query words with IDF (if similarity query provided)
  const queryWords: Record<string, number> = {};
  if (options?.similarityQuery && idf) {
    const stopWords = idf.stopWords || new Set<string>();
    const raw = options.similarityQuery.toLowerCase().split(/[\s.,;:!?()\[\]{}"'`\n\r\t<>=\-+*/]+/);
    for (const w of raw) {
      if (w.length > 3 && !stopWords.has(w)) {
        queryWords[w] = idf.weights[w] ?? 1.0;
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // Phase 1: Similarity scoring (dimensions 1–10, no causal)
  // ═══════════════════════════════════════════════════════════════
  //
  // v4.0: 10-dimensional scoring (agent.py lines 2563-2651):
  //   1. IDF-weighted hypothesis similarity (0-10)
  //   2. Subsystem/taskType overlap (0-5)
  //   3. Branch match (0-3)
  //   4. Exponential recency decay (0-3)
  //   5. Outcome bonus (0-2)
  //   6. Information density bonus (0-2)
  //   7. Thompson-sampled retrieval utility (0-5)
  //   8. Surprise bonus (0-3) [v4 NEW]
  //   9. Curiosity/EIG bonus (0-3) [v4 NEW]
  //  10. Counterfactual value bonus (+1.5) [v4 NEW]

  const phase1Scored: Array<{ score: number; idx: number; exp: RLExecutionRecord }> = [];

  for (let idx = 0; idx < experiments.length; idx++) {
    const exp = experiments[idx];
    let score = 0.0;

    // 1. IDF-weighted hypothesis similarity (0-10)
    if (Object.keys(queryWords).length > 0) {
      const hypText = [
        exp.taskId,
        exp.taskType,
      ].join(" ").toLowerCase();
      let weightedOverlap = 0.0;
      const totalWeight = Object.values(queryWords).reduce((a, b) => a + b, 0);
      if (totalWeight > 0) {
        for (const [w, iw] of Object.entries(queryWords)) {
          if (hypText.includes(w)) {
            weightedOverlap += iw;
          }
        }
        score += (weightedOverlap / totalWeight) * 10.0 * retrievalWeights.semanticWeight;
      }
    }

    // 2. Subsystem/taskType overlap (0-5)
    // Within turbocontext, "subsystem" = taskType family
    if (exp.taskType === taskType) {
      score += 5.0 * retrievalWeights.taskOverlapWeight;
    } else {
      // Partial credit for related task types (same family prefix)
      const expFamily = exp.taskType.split("_")[0];
      const queryFamily = taskType.split("_")[0];
      if (expFamily === queryFamily) {
        score += 2.5 * retrievalWeights.taskOverlapWeight;
      }
    }

    // 3. Branch match (0-3)
    if (exp.taskType === taskType) {
      score += 3.0 * retrievalWeights.branchMatchWeight;
    }

    // 4. Exponential recency decay (0-3)
    const recency = Math.exp(-3.0 * (total - 1 - idx) / Math.max(total - 1, 1));
    score += recency * 3.0 * retrievalWeights.recencyWeight;

    // 5. Outcome bonus (0-2)
    const qualityThreshold = 0.85;
    if (exp.qualityScore >= qualityThreshold) {
      score += 2.0 * retrievalWeights.outcomeBonusWeight;
    } else if (exp.attemptCount >= 3 && exp.qualityScore < 0.5) {
      // Crash-like: very low quality with many attempts
      score += 0.5 * retrievalWeights.outcomeBonusWeight;
    }

    // 6. Information density bonus (0-2)
    // Proxy: how much metadata does this record carry?
    const infoScore = Math.min(2.0,
      (exp.taskId.length / 100) +
      ((exp.sourceFiles?.length ?? 0) / 10) +
      (Object.keys(exp.coverage ?? {}).length / 5)
    );
    score += infoScore * retrievalWeights.infoDensityWeight;

    // 7. Thompson-sampled retrieval utility (0-5)
    // v4.0: Sample from Beta(α,β) — naturally balances explore/exploit
    const tsSample = thompsonSample(
      (exp as any).thompsonAlpha ?? 1.0,
      (exp as any).thompsonBeta ?? 1.0,
    );
    score += tsSample * 5.0;

    // 8. v4.0: Surprise bonus (0-3) — curriculum-phase adaptive
    const surprise = (exp as any).surpriseScore ?? 0.5;
    score += surprise * 3.0 * curriculumParams.surpriseWeight * retrievalWeights.surpriseWeight;

    // 9. v4.0: Curiosity/EIG bonus (0-3) — curriculum-phase adaptive
    const curiosity = computeCuriosityBonusForRetrieval(
      exp, experiments, curriculumParams.curiosityWeight,
    );
    score += curiosity * retrievalWeights.curiosityWeight;

    // 10. v4.0: Counterfactual value bonus (+1.5)
    // Memories with counterfactual insights provide causal reasoning leverage
    const cf = (exp as any).counterfactual ?? "";
    if (cf && cf.length > 20) {
      score += 1.5 * retrievalWeights.counterfactualWeight;
    }

    phase1Scored.push({ score, idx, exp });
  }

  // Sort by Phase 1 score, keep top poolSize as candidate pool
  phase1Scored.sort((a, b) => b.score - a.score);
  const candidatePool = phase1Scored.slice(0, poolSize);

  // ═══════════════════════════════════════════════════════════════
  // Phase 2: Advantage-weighted causal re-rank within candidate pool
  // ═══════════════════════════════════════════════════════════════
  //
  // v4.0: Use advantage (causal - V(subsystem)) instead of raw
  // causal_utility. This removes the bias where memories from
  // "easy" task types get artificially high causal scores.
  // agent.py lines 2664-2681.

  const phase2Scored: Array<{ score: number; idx: number; exp: RLExecutionRecord }> = [];

  for (const { score: p1Score, idx, exp } of candidatePool) {
    // Advantage-weighted causal utility (0-8)
    const causal = (exp as any).causalUtility ?? 0.5;
    const adv = computeAdvantageForMemory(causal, exp.taskType, subsystemBaseline);
    // Map advantage [-1, 1] to causal score [0, 8]
    // Center at 4.0 (neutral advantage), range [-4, +4] around center
    const causalScore = Math.max(0.0, Math.min(8.0, 4.0 + adv * config.advantageScale));

    // Combined: Phase 1 similarity + Phase 2 advantage-weighted causal
    const combinedScore = p1Score + causalScore;
    phase2Scored.push({ score: combinedScore, idx, exp });
  }

  // Sort by combined score
  phase2Scored.sort((a, b) => b.score - a.score);

  // ═══════════════════════════════════════════════════════════════
  // MMR diversity re-ranking with entropy bonus
  // ═══════════════════════════════════════════════════════════════
  //
  // v4.0: Entropy-regularized MMR — adds bonus for outcome diversity
  // in addition to task type diversity. Prevents "outcome monoculture"
  // where all selected memories are successes or all are failures.
  // agent.py lines 2693-2733.

  if (phase2Scored.length <= topK) {
    const selected = phase2Scored.map(s => s.exp);
    return {
      selected,
      selectedIds: selected.map(e => e.taskId),
      phase1Scores: candidatePool.map(c => c.score),
      phase2Scores: phase2Scored.map(s => s.score),
      poolSize,
    };
  }

  const selected: Array<{ score: number; idx: number; exp: RLExecutionRecord }> = [];
  const remaining = [...phase2Scored];

  // First selection: highest combined score
  selected.push(remaining.shift()!);

  const eta = 1.5; // entropy bonus weight (0 = standard MMR, higher = more diverse)

  for (let i = 0; i < Math.min(topK - 1, remaining.length); i++) {
    let bestItem: typeof remaining[0] | null = null;
    let bestMmr = -Infinity;
    let bestRemIdx = -1;

    for (let ri = 0; ri < remaining.length; ri++) {
      const item = remaining[ri];
      const scoreI = item.score;

      // Max task type similarity to any already-selected
      let maxSim = 0.0;
      for (const sel of selected) {
        const sim = item.exp.taskType === sel.exp.taskType ? 1.0
          : item.exp.taskType.split("_")[0] === sel.exp.taskType.split("_")[0] ? 0.5
          : 0.0;
        if (sim > maxSim) maxSim = sim;
      }

      // Entropy bonus: reward outcome diversity
      const selectedOutcomes = selected.map(s =>
        s.exp.qualityScore >= 0.85 ? "success"
        : s.exp.attemptCount >= 3 && s.exp.qualityScore < 0.5 ? "crash"
        : "failure"
      );
      const itemOutcome = item.exp.qualityScore >= 0.85 ? "success"
        : item.exp.attemptCount >= 3 && item.exp.qualityScore < 0.5 ? "crash"
        : "failure";

      // Entropy bonus from outcome counts
      const outcomeCounts = new Map<string, number>();
      for (const o of selectedOutcomes) {
        outcomeCounts.set(o, (outcomeCounts.get(o) || 0) + 1);
      }
      const n = selectedOutcomes.length;
      const pCurrent = (outcomeCounts.get(itemOutcome) || 0) / n;
      const epsilon = 0.1;
      const entBonus = Math.max(0.0, -Math.log(pCurrent + epsilon) * 0.5);

      // Standard MMR + entropy bonus
      const mmr = (mmrLambda * scoreI
        - (1.0 - mmrLambda) * maxSim * 10.0
        + eta * entBonus);

      if (mmr > bestMmr) {
        bestMmr = mmr;
        bestItem = item;
        bestRemIdx = ri;
      }
    }

    if (bestItem !== null && bestRemIdx >= 0) {
      selected.push(bestItem);
      remaining.splice(bestRemIdx, 1);
    }
  }

  const finalSelected = selected.map(s => s.exp);
  return {
    selected: finalSelected,
    selectedIds: finalSelected.map(e => e.taskId),
    phase1Scores: candidatePool.map(c => c.score),
    phase2Scores: selected.map(s => s.score),
    poolSize,
  };
}
