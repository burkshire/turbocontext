// ============================================================
// Phase 1: Context Compression & Scoring
// ============================================================
// 公式:
//   score(cᵢ) = α · sim(embed(cᵢ), embed(T)) + β · recency(cᵢ) + γ · specificity(cᵢ)
//   其中 α + β + γ = 1，sim 为余弦相似度
//   recency(cᵢ) = 1 / (1 + days_since_last_modified)
//   specificity(cᵢ) = 1 - (len(cᵢ) / max_len)
// ============================================================

import type {
  Task, ContextFragment, CompressedContext, CompressedFragment,
  CapabilityRequirement, TurboContextConfig, IDFCache, RetrievalWeights,
} from "../types.js";
import { DEFAULT_RETRIEVAL_WEIGHTS } from "../types.js";
import type { EmbeddingProvider } from "./embeddings.js";
import { cosineSimilarity, normalizeSimilarity } from "./embeddings.js";
import { entropyMMRBonus } from "./retrieval-system.js";

/** 默认配置 */
export const DEFAULT_COMPRESSOR_CONFIG = {
  alpha: 0.55,
  beta: 0.20,
  gamma: 0.25,
  maxTokenBudget: 8000,
  minCoverage: 0.80,
};

// ============================================================
// v3.1 — Global IDF Cache (autoresearch: IDF-weighted retrieval)
// ============================================================

/** 停用词集合（不计入 IDF） */
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
  "they", "their", "them", "his", "her", "hers",
]);

/**
 * 构建/更新全局 IDF 缓存。
 *
 * 在所有上下文片段中计算词的逆文档频率。
 * 出现在很多片段中的词（如 "function", "code"）权重低，
 * 仅出现在少数片段中的关键词权重高。
 *
 * 公式: idf(w) = log((N + 2) / (df(w) + 1)) + 0.5
 */
export function buildIDFCache(
  fragments: ContextFragment[],
  existing?: IDFCache,
): IDFCache {
  const N = fragments.length;
  if (N === 0) {
    return {
      weights: {},
      documentCount: 0,
      lastUpdated: Date.now(),
      stopWords: STOP_WORDS_SET,
    };
  }

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
    // 平滑 IDF: 高频词 → 低权重，低频词 → 高权重
    weights[w] = Math.log((N + 2) / (count + 1)) + 0.5;
  }

  return {
    weights,
    documentCount: N,
    lastUpdated: Date.now(),
    stopWords: STOP_WORDS_SET,
  };
}

/**
 * IDF 加权查询词向量。
 * 将查询文本分解为词，每个词用其 IDF 权重表示。
 */
export function buildQueryVector(
  query: string,
  idfCache: IDFCache,
): Record<string, number> {
  const words = query.toLowerCase()
    .split(/[\s.,;:!?()\[\]{}"'`\n\r\t<>=\-+*/]+/)
    .filter(w => w.length > 3 && !STOP_WORDS_SET.has(w));

  const vector: Record<string, number> = {};
  for (const w of words) {
    vector[w] = idfCache.weights[w] || 1.0; // 未见词默认权重 1.0
  }
  return vector;
}

/**
 * 计算 IDF 加权相似度。
 *
 * 给定查询向量和文档文本，返回归一化的重叠分数。
 * 这是 autoresearch 中最核心的检索评分维度（0-10 尺度）。
 */
export function computeIDFSimilarity(
  queryVector: Record<string, number>,
  docText: string,
): number {
  const totalWeight = Object.values(queryVector).reduce((s, v) => s + v, 0);
  if (totalWeight === 0) return 0;

  const docLower = docText.toLowerCase();
  let weightedOverlap = 0;
  for (const [word, weight] of Object.entries(queryVector)) {
    if (docLower.includes(word)) {
      weightedOverlap += weight;
    }
  }
  return weightedOverlap / totalWeight;
}

/**
 * 信息密度评分（autoresearch: information density bonus）。
 *
 * 奖励包含丰富结构信息的片段：
 * - 函数/类/接口定义
 * - 导入/导出语句
 * - 注释/文档字符串
 * - 错误处理
 * - 测试断言
 *
 * 返回 0-1 的分数。
 */
export function computeInfoDensity(fragment: ContextFragment): number {
  const text = fragment.content;
  const structuralMarkers = [
    /\b(function|class|interface|type|enum|struct|trait|impl)\s+\w+/gi,
    /\b(import|export|from|require|include|mod|use)\s+/gi,
    /\/\*\*|\*\/|\/\/\/|#\s*TODO|#\s*FIXME|#\s*NOTE/gi,
    /\b(try|catch|throw|finally|except|rescue)\b/gi,
    /\b(expect|assert|assertEqual|should|test|it|describe)\b/gi,
    /\b(async|await|yield|return)\b/gi,
    /```[\s\S]*?```/gi,
  ];

  let uniqueHits = 0;
  const seenPatterns = new Set<number>();
  for (let i = 0; i < structuralMarkers.length; i++) {
    const matches = text.match(structuralMarkers[i]);
    if (matches && matches.length > 0 && !seenPatterns.has(i)) {
      uniqueHits++;
      seenPatterns.add(i);
    }
  }

  return Math.min(1.0, uniqueHits / structuralMarkers.length + 0.2);
}

/**
 * 指数衰减新近度评分（autoresearch: exponential recency decay）。
 *
 * 公式: recency = exp(-3.0 * position_from_end / (total - 1))
 *
 * 与简单的 1/(1+days) 不同，指数衰减能更好地区分
 * "刚刚修改" vs "很久以前" 的片段。
 *
 * 当 position=0（最新）→ 1.0
 * 当 position=total-1（最旧）→ ~0.05
 *
 * @param positionFromEnd - 从最新端开始的位置（0=最新）
 * @param total - 总片段数
 */
export function computeExpRecency(positionFromEnd: number, total: number): number {
  if (total <= 1) return 1.0;
  const normalizedPos = positionFromEnd / Math.max(1, total - 1);
  return Math.exp(-3.0 * normalizedPos);
}

/**
 * Jaccard 任务类型相似度（autoresearch: subsystem Jaccard）。
 *
 * 计算两个任务类型的能力需求重叠程度。
 */
export function computeTaskJaccard(
  reqsA: CapabilityRequirement[],
  reqsB: CapabilityRequirement[],
): number {
  const keysA = new Set(reqsA.map(r => r.name));
  const keysB = new Set(reqsB.map(r => r.name));
  if (keysA.size === 0 && keysB.size === 0) return 0;
  const intersection = [...keysA].filter(k => keysB.has(k)).length;
  const union = new Set([...keysA, ...keysB]).size;
  return union > 0 ? intersection / union : 0;
}

/**
 * MMR (Maximal Marginal Relevance) 多样性重排。
 *
 * 公式: MMR(cᵢ) = λ · score(cᵢ) - (1-λ) · max_{cⱼ∈Selected} sim(cᵢ, cⱼ)
 *
 * 从候选集中贪心选择：每步选 MMR 最高的候选项。
 * λ=1: 纯按分数排序（最大化相关性）
 * λ=0: 纯按多样性排序（最大化不相似度）
 *
 * @param candidates - (id, score, features) 三元组
 * @param topK - 保留数量
 * @param lambda - 相关性/多样性权衡 (0-1)
 * @param featureSimFn - 特征相似度函数
 */
export function mmrReRank<T>(
  candidates: Array<{ item: T; score: number; features: string[] }>,
  topK: number,
  lambda: number,
  featureSimFn: (a: string[], b: string[]) => number = jaccardSimilarity,
  // v3.9: Optional entropy bonus callback for outcome/content diversity
  entropyBonusFn?: (candidate: T) => number,
): T[] {
  if (candidates.length <= topK) {
    return candidates.map(c => c.item);
  }

  const remaining = [...candidates];
  const selected: Array<{ item: T; score: number; features: string[] }> = [];

  // 第一步：选最高分
  const first = remaining.reduce((best, c) => c.score > best.score ? c : best, remaining[0]);
  selected.push(first);
  remaining.splice(remaining.indexOf(first), 1);

  // 贪心 MMR 选择
  for (let step = 1; step < topK && remaining.length > 0; step++) {
    let bestIdx = -1;
    let bestMMR = -Infinity;

    for (let i = 0; i < remaining.length; i++) {
      const candidate = remaining[i];
      // 与已选中的最大相似度
      let maxSim = 0;
      for (const sel of selected) {
        const sim = featureSimFn(candidate.features, sel.features);
        if (sim > maxSim) maxSim = sim;
      }
      // v3.9: Entropy bonus — reward outcome/content diversity
      const entropyBonus = entropyBonusFn ? entropyBonusFn(candidate.item) : 0;
      const mmr = lambda * candidate.score - (1 - lambda) * maxSim * 10 + entropyBonus * 1.5;
      if (mmr > bestMMR) {
        bestMMR = mmr;
        bestIdx = i;
      }
    }

    if (bestIdx >= 0) {
      selected.push(remaining[bestIdx]);
      remaining.splice(bestIdx, 1);
    }
  }

  return selected.map(s => s.item);
}

/** Jaccard 相似度（两个字符串数组的重叠度） */
function jaccardSimilarity(a: string[], b: string[]): number {
  const setA = new Set(a);
  const setB = new Set(b);
  if (setA.size === 0 && setB.size === 0) return 0;
  const intersection = [...setA].filter(x => setB.has(x)).length;
  const union = new Set([...setA, ...setB]).size;
  return union > 0 ? intersection / union : 0;
}

/**
 * 主入口：压缩上下文
 *
 * 输入: 任务 T, 上下文片段集合 C, token 预算
 * 输出: 优化后的压缩上下文 C'
 *
 * v3.1 改进（autoresearch-inspired）:
 *   - 全局 IDF 加权检索
 *   - MMR 多样性重排
 *   - 信息密度加成
 *   - 指数衰减新近度
 *   - 优先级分层 token 预算
 *
 * @param idfCache - 全局 IDF 缓存（由 Learner 维护）
 * @param adaptiveMmrLambda - 自适应 MMR λ（由 Learner 根据分支状态计算）
 * @param retrievalWeights - 检索评分维度权重
 * @param sourceBoostFn - 可选的源文件历史表现加成函数（来自 Learner.getSourceBoost）
 * @param embeddingProvider - 可选的 embedding provider，用于替代 IDF 语义相似度
 */
export async function compressContext(
  task: Task,
  fragments: ContextFragment[],
  config: Partial<TurboContextConfig> & {
    sourceBoostFn?: (source: string) => number;
    idfCache?: IDFCache;
    adaptiveMmrLambda?: number;
    retrievalWeights?: RetrievalWeights;
    embeddingProvider?: EmbeddingProvider;
    /** v3.7: causal utility multiplier for Phase 2 re-ranking */
    causalBoostFn?: (fragment: ContextFragment, task: Task) => number;
  } = {}
): Promise<CompressedContext> {
  const cfg = { ...DEFAULT_COMPRESSOR_CONFIG, ...config };
  const sourceBoostFn = "sourceBoostFn" in config ? config.sourceBoostFn : undefined;
  const causalBoostFn = "causalBoostFn" in config ? config.causalBoostFn : undefined;
  const idfCache = config.idfCache || buildIDFCache(fragments);
  const mmrLambda = config.adaptiveMmrLambda ?? 0.65;
  const rw = config.retrievalWeights || DEFAULT_RETRIEVAL_WEIGHTS;
  const originalTokens = estimateTokens(fragments);

  // v3.2: Pre-compute embedding scores if provider is available.
  // This replaces IDF-based semantic similarity with cosine similarity
  // in embedding space, while keeping all other scoring dimensions.
  let embeddingScores: Map<string, number> | undefined;
  if (config.embeddingProvider && fragments.length > 0) {
    try {
      const queryEmb = await config.embeddingProvider.embedQuery(task.description);
      const fragmentEmbs = await config.embeddingProvider.embed(
        fragments.map(f => f.content)
      );
      embeddingScores = new Map();
      for (let i = 0; i < fragments.length; i++) {
        const raw = cosineSimilarity(queryEmb, fragmentEmbs[i]);
        embeddingScores.set(fragments[i].id, normalizeSimilarity(raw));
      }
    } catch (err) {
      // Embedding failed — log and fall back to IDF silently
      console.warn(
        `[TurboContext] Embedding provider failed, falling back to IDF: ${(err as Error).message}`
      );
      embeddingScores = undefined;
    }
  }

  // Step 1: 分解任务为能力需求
  const requirements = decomposeTask(task);

  // Step 2: 构建 IDF 加权查询向量
  const queryVector = buildQueryVector(task.description, idfCache);
  const queryTokens = Object.keys(queryVector).length;

  // Step 3: 多维度评分（autoresearch: 6-dimension scoring）
  // 按最后修改时间排序以计算指数衰减新近度
  const sortedByTime = [...fragments].sort((a, b) => b.lastModified - a.lastModified);

  const scored = fragments.map((f, idx) => {
    const recencyPos = sortedByTime.indexOf(f);
    const embScore = embeddingScores?.get(f.id);
    const score = calculateScoreV2(f, task, cfg, {
      queryVector,
      idfCache,
      recencyPosition: recencyPos >= 0 ? recencyPos : idx,
      totalFragments: fragments.length,
      sourceBoostFn,
      retrievalWeights: rw,
      embeddingScore: embScore,
      causalBoostFn,
    });
    return { fragment: f, score, features: extractFragmentFeatures(f, task) };
  });

  // Step 4: 按分数降序排列
  scored.sort((a, b) => b.score - a.score);

  // Step 5: 优先级分层 token 预算分配
  const budgetAlloc = allocateTokenBudget(cfg.maxTokenBudget, fragments.length, queryTokens);
  const selected = greedySelectV2(scored, requirements, cfg, task, budgetAlloc, mmrLambda);

  // Step 6: 压缩选中的片段
  const compressed = selected.map(s => compressFragment(s.fragment, s.score));

  // 计算覆盖率和压缩比
  const compressedTokens = estimateTokenCount(compressed.map(c => c.preservedSections.join("\n")));
  const coverage = computeCoverage(compressed, requirements);

  return {
    originalTokens,
    compressedTokens,
    compressionRatio: 1 - (compressedTokens / originalTokens),
    fragments: compressed,
    coverage,
  };
}

/**
 * 优先级分层 token 预算分配（autoresearch: priority-tier allocation）。
 *
 * P0 (40%): 最高分片段（核心上下文）
 * P1 (30%): MMR 多样性补充
 * P2 (20%): 能力覆盖补充
 * P3 (10%): 新近度补充
 */
function allocateTokenBudget(
  totalBudget: number,
  fragmentCount: number,
  queryTokenCount: number,
): { p0: number; p1: number; p2: number; p3: number } {
  // 片段少时简化分配
  if (fragmentCount <= 5) {
    return { p0: totalBudget, p1: 0, p2: 0, p3: 0 };
  }
  // 大规模时使用分层预算
  return {
    p0: Math.floor(totalBudget * 0.40),
    p1: Math.floor(totalBudget * 0.30),
    p2: Math.floor(totalBudget * 0.20),
    p3: Math.floor(totalBudget * 0.10),
  };
}

/**
 * 提取片段特征向量（供 MMR 多样性计算）。
 */
function extractFragmentFeatures(fragment: ContextFragment, task: Task): string[] {
  const features: string[] = [];
  const text = fragment.content.toLowerCase();

  // 内容类型特征
  features.push(`type:${fragment.contentType}`);

  // 语言/框架特征
  if (text.includes("function") || text.includes("const ") || text.includes("let ")) features.push("lang:js");
  if (text.includes("class ") && text.includes("public ")) features.push("lang:java");
  if (text.includes("def ") || text.includes("import ")) features.push("lang:python");
  if (text.includes("fn ") || text.includes("let mut")) features.push("lang:rust");
  if (text.includes("func ") && text.includes("package ")) features.push("lang:go");

  // 结构特征
  if (text.includes("interface ") || text.includes("type ")) features.push("struct:types");
  if (text.includes("async ") || text.includes("await ")) features.push("pattern:async");
  if (text.includes("try ") || text.includes("catch ")) features.push("pattern:error_handling");
  if (text.includes("test(") || text.includes("describe(")) features.push("pattern:testing");

  // 源路径特征
  const pathParts = fragment.source.split("/");
  if (pathParts.length > 2) features.push(`path:${pathParts.slice(0, -1).join("/")}`);

  return features;
}

/**
 * Step 1: 将任务分解为原子能力需求
 *
 * 通过关键词匹配和任务类型分析，提取任务所需的底层能力
 */
function decomposeTask(task: Task): CapabilityRequirement[] {
  const text = task.description.toLowerCase();

  const requirementPatterns: Record<string, { keywords: string[]; defaultWeight: number; description: string }> = {
    code_understanding: {
      keywords: ["understand", "read", "analyze", "解释", "理解", "阅读", "review", "审查"],
      defaultWeight: 0.25,
      description: "理解现有代码结构和逻辑"
    },
    pattern_recognition: {
      keywords: ["pattern", "detect", "find", "识别", "发现", "detect", "anti-pattern", "smell"],
      defaultWeight: 0.15,
      description: "识别代码模式和反模式"
    },
    code_generation: {
      keywords: ["write", "create", "implement", "generate", "add", "写", "创建", "实现", "生成"],
      defaultWeight: 0.25,
      description: "生成新的代码"
    },
    code_modification: {
      keywords: ["change", "modify", "update", "refactor", "fix", "改", "修改", "重构", "修复"],
      defaultWeight: 0.15,
      description: "修改现有代码"
    },
    error_detection: {
      keywords: ["bug", "error", "issue", "problem", "wrong", "bug", "错误", "问题", "缺陷"],
      defaultWeight: 0.10,
      description: "检测错误和潜在问题"
    },
    design: {
      keywords: ["design", "architecture", "structure", "organize", "设计", "架构", "结构"],
      defaultWeight: 0.10,
      description: "系统设计和架构决策"
    },
  };

  // 计算匹配得分并生成需求
  const matched = Object.entries(requirementPatterns).map(([name, pattern]) => {
    const matchCount = pattern.keywords.filter(kw => text.includes(kw)).length;
    const score = matchCount / pattern.keywords.length;
    return {
      name,
      weight: pattern.defaultWeight * (0.5 + 0.5 * score),
      description: pattern.description,
    };
  });

  // 过滤掉完全不匹配的，归一化权重
  const active = matched.filter(r => r.weight > 0.02);
  const totalWeight = active.reduce((s, r) => s + r.weight, 0);

  return active.map(r => ({
    ...r,
    weight: totalWeight > 0 ? r.weight / totalWeight : r.weight,
  }));
}

/**
 * Step 3 (v3.1): 六维度加权检索评分（autoresearch: 6-dimension scoring）。
 *
 * 公式:
 *   score = semantic(IDF) · w₁ + taskOverlap · w₂ + branchMatch · w₃
 *         + recency(exp) · w₄ + outcomeBoost · w₅ + infoDensity · w₆
 *
 * 所有维度归一化 0-1，用 retrievalWeights 控制各维度贡献。
 */
interface ScoreContext {
  queryVector: Record<string, number>;
  idfCache: IDFCache;
  recencyPosition: number;
  totalFragments: number;
  sourceBoostFn?: (source: string) => number;
  retrievalWeights: RetrievalWeights;
  /** v3.2: pre-computed embedding similarity [0-1], overrides IDF when set */
  embeddingScore?: number;
  /** v3.7: causal utility multiplier [0.5, 1.5], applied after normalization */
  causalBoostFn?: (fragment: ContextFragment, task: Task) => number;
}

function calculateScoreV2(
  fragment: ContextFragment,
  task: Task,
  config: { alpha: number; beta: number; gamma: number },
  ctx: ScoreContext,
): number {
  const rw = ctx.retrievalWeights;

  // Map α,β,γ to dimension group scaling factors.
  // Default α=0.55, β=0.20, γ=0.25 → each scale = 1.0, so defaults are unchanged.
  // When learner adjusts α,β,γ, the corresponding dimension groups scale proportionally.
  const alphaScale = config.alpha / 0.55;   // relevance group (semantic, task, branch)
  const betaScale  = config.beta  / 0.20;   // recency
  const gammaScale = config.gamma / 0.25;   // density group (outcome, infoDensity)

  // Effective weights = base weights × group scale
  const effSemantic    = rw.semanticWeight    * alphaScale;
  const effTask        = rw.taskOverlapWeight * alphaScale;
  const effBranch      = rw.branchMatchWeight * alphaScale;
  const effRecency     = rw.recencyWeight     * betaScale;
  const effOutcome     = rw.outcomeBonusWeight * gammaScale;
  const effInfoDensity = rw.infoDensityWeight  * gammaScale;

  const maxPossible = effSemantic + effTask + effBranch
    + effRecency + effOutcome + effInfoDensity;

  // 1. 语义相似度 (0-1 → 乘以 effective semanticWeight)
  // v3.2: use embedding cosine similarity when available, fall back to IDF
  const semSim = ctx.embeddingScore !== undefined
    ? ctx.embeddingScore
    : computeIDFSimilarity(ctx.queryVector, fragment.content);
  const semanticScore = semSim * effSemantic;

  // 2. 任务类型重叠 (0-1 → 乘以 effective taskOverlapWeight)
  const taskOverlap = typeCompatibility(fragment.contentType, task.type);
  const taskScore = taskOverlap * effTask;

  // 3. 分支匹配 (0-1 → 乘以 effective branchMatchWeight)
  const branchMatch = (fragment.contentType === "source" &&
    ["code_generation", "code_review", "code_refactor", "debugging"].includes(task.type)) ? 1.0 :
    (fragment.contentType === "test" && task.type === "testing") ? 1.0 :
    (fragment.contentType === "docs" && ["documentation", "analysis"].includes(task.type)) ? 1.0 : 0.3;
  const branchScore = branchMatch * effBranch;

  // 4. 指数衰减新近度 (0-1 → 乘以 effective recencyWeight)
  const recencyRaw = computeExpRecency(ctx.recencyPosition, ctx.totalFragments);
  const recencyScore = recencyRaw * effRecency;

  // 5. 历史表现加成 (0-1 → 乘以 effective outcomeBonusWeight)
  let outcomeRaw = 0.5; // 中性默认
  if (ctx.sourceBoostFn) {
    const boost = ctx.sourceBoostFn(fragment.source);
    outcomeRaw = 0.5 + boost;
    outcomeRaw = Math.max(0, Math.min(1, outcomeRaw));
  }
  const outcomeScore = outcomeRaw * effOutcome;

  // 6. 信息密度加成 (0-1 → 乘以 effective infoDensityWeight)
  const densityRaw = computeInfoDensity(fragment);
  const densityScore = densityRaw * effInfoDensity;

  const totalScore = semanticScore + taskScore + branchScore
    + recencyScore + outcomeScore + densityScore;

  // 归一化到 [0, 1]
  const normalized = totalScore / maxPossible;

  // v3.7: Phase 2 — causal re-rank multiplier
  // Applied after similarity normalization so causal signal gates relevance,
  // not competes with it as a 7th dimension.
  const causalFactor = ctx.causalBoostFn
    ? ctx.causalBoostFn(fragment, task)
    : 1.0;
  // Clamp factor to [0.5, 1.5] then multiply. Final result clamped to [0, 1].
  const factor = Math.max(0.5, Math.min(1.5, causalFactor));
  const boosted = normalized * factor;

  return Math.round(Math.min(1.0, boosted) * 10000) / 10000;
}

/**
 * calculateScore 的兼容包装（保持旧 API 可用）。
 */
function calculateScore(
  fragment: ContextFragment,
  task: Task,
  config: { alpha: number; beta: number; gamma: number },
  allFragments?: ContextFragment[],
  sourceBoostFn?: (source: string) => number,
): number {
  const fragments = allFragments || [fragment];
  const idfCache = buildIDFCache(fragments);
  const queryVector = buildQueryVector(task.description, idfCache);
  const sorted = [...fragments].sort((a, b) => b.lastModified - a.lastModified);
  const recencyPos = sorted.indexOf(fragment);

  return calculateScoreV2(fragment, task, config, {
    queryVector,
    idfCache,
    recencyPosition: recencyPos >= 0 ? recencyPos : 0,
    totalFragments: fragments.length,
    sourceBoostFn,
    retrievalWeights: DEFAULT_RETRIEVAL_WEIGHTS,
  });
}

/**
 * 语义相似度估计（改进版：TF-IDF 加权）
 *
 * 核心改进：
 * 1. 对所有片段中共同出现的通用词降权（如 "function", "code"）
 * 2. 仅在少数片段中出现的关键词加权
 * 3. 代码结构相似度加入任务类型感知
 */
function computeSemanticSimilarity(fragment: ContextFragment, task: Task, allFragments?: ContextFragment[]): number {
  const taskWords = tokenize(task.description);
  const content = fragment.content.toLowerCase();

  // 1. TF-IDF 风格关键词匹配
  // 对每个任务词，计算其在所有片段中的逆文档频率
  let keywordScore: number;
  if (taskWords.length === 0) {
    keywordScore = 0;
  } else if (allFragments && allFragments.length > 1) {
    // IDF 加权：出现在越少片段中的词越重要
    const idfScores = taskWords.map(w => {
      const containingCount = allFragments.filter(f =>
        f.content.toLowerCase().includes(w)
      ).length;
      // idf = log(N / df)，加 1 平滑
      const idf = Math.log(allFragments.length / (1 + containingCount)) + 1;
      return { word: w, idf };
    });
    const totalIdf = idfScores.reduce((s, i) => s + i.idf, 0);
    const matchedIdf = idfScores
      .filter(i => content.includes(i.word))
      .reduce((s, i) => s + i.idf, 0);
    keywordScore = totalIdf > 0 ? matchedIdf / totalIdf : 0;
  } else {
    // 没有上下文时退化为简单匹配
    const matchedWords = taskWords.filter(w => content.includes(w));
    keywordScore = matchedWords.length / taskWords.length;
  }

  // 2. 文件类型与任务的适配度
  const typeScore = typeCompatibility(fragment.contentType, task.type);

  // 3. 代码结构相似度（加入任务类型感知）
  const defPatterns = task.type === "code_generation" || task.type === "code_refactor"
    ? ["function", "class", "interface", "const", "let", "def", "impl", "return", "export"]
    : task.type === "code_review" || task.type === "debugging"
    ? ["function", "class", "interface", "type", "if", "for", "try", "catch", "throw"]
    : ["function", "class", "interface", "const", "let", "def", "impl"];
  const defMatches = defPatterns.filter(p => content.includes(p)).length;
  const structuralScore = defMatches / defPatterns.length;

  return 0.50 * keywordScore + 0.30 * typeScore + 0.20 * structuralScore;
}

/**
 * 内容类型与任务类型的兼容性评分
 */
function typeCompatibility(contentType: string, taskType: string): number {
  const compatibility: Record<string, Record<string, number>> = {
    source: {
      code_generation: 0.9, code_review: 1.0, code_refactor: 1.0,
      debugging: 1.0, testing: 0.7, analysis: 0.8,
      documentation: 0.4, design: 0.6, general: 0.5,
    },
    test: {
      testing: 1.0, code_review: 0.7, code_refactor: 0.6,
      debugging: 0.8, analysis: 0.5, general: 0.4,
      code_generation: 0.3, documentation: 0.2, design: 0.2,
    },
    docs: {
      documentation: 1.0, analysis: 0.7, design: 0.8,
      code_review: 0.3, general: 0.5, code_generation: 0.2,
      code_refactor: 0.2, debugging: 0.2, testing: 0.2,
    },
    config: {
      code_generation: 0.3, code_review: 0.3, code_refactor: 0.4,
      debugging: 0.5, analysis: 0.4, general: 0.3,
      documentation: 0.2, design: 0.3, testing: 0.3,
    },
  };
  return compatibility[contentType]?.[taskType] ?? 0.3;
}

/**
 * 新鲜度评分
 *
 * recency(cᵢ) = 1 / (1 + days_since_last_modified)
 * 最近修改的片段更有价值（代表当前活跃的代码）
 */
function computeRecency(fragment: ContextFragment): number {
  const now = Date.now();
  const msSinceModified = now - fragment.lastModified;
  const daysSinceModified = msSinceModified / (1000 * 60 * 60 * 24);
  return 1 / (1 + daysSinceModified);
}

/**
 * 特异性评分
 *
 * specificity(cᵢ) = 1 - (len(cᵢ) / max_len)
 * 信息密度高（短而精）的片段得分更高
 */
function computeSpecificity(fragment: ContextFragment): number {
  const MAX_REASONABLE_LENGTH = 5000; // 超过此长度被视为噪音
  const normalizedLen = Math.min(fragment.length / MAX_REASONABLE_LENGTH, 1);
  return 1 - normalizedLen;
}

/**
 * Step 5 (v3.1): 优先级分层 + MMR 多样性选择。
 *
 * P0: 最高分片段（核心上下文）
 * P1: MMR 多样性补充（避免选择过于相似的片段）
 * P2: 能力覆盖补充（确保所有能力需求都被覆盖）
 * P3: 新近度补充（确保最近的片段不会被遗漏）
 */
function greedySelectV2(
  scored: Array<{ fragment: ContextFragment; score: number; features: string[] }>,
  requirements: CapabilityRequirement[],
  config: { maxTokenBudget: number; minCoverage: number },
  task: Task,
  budgetAlloc: { p0: number; p1: number; p2: number; p3: number },
  mmrLambda: number,
  /** v3.8: SGS causal graph for d-separation redundancy elimination */
): Array<{ fragment: ContextFragment; score: number }> {
  const usedIds = new Set<string>();
  const result: Array<{ fragment: ContextFragment; score: number }> = [];

  // P0: 取最高分片段填充核心预算
  let budget = budgetAlloc.p0;
  for (const s of scored) {
    if (budget <= 0) break;
    const tokens = estimateTokenCount([s.fragment.content]);
    if (tokens <= budget && !usedIds.has(s.fragment.id)) {
      result.push({ fragment: s.fragment, score: s.score });
      usedIds.add(s.fragment.id);
      budget -= tokens;
    }
  }

  // P1: MMR 多样性补充（在剩余候选中用 MMR 重排）
  budget = budgetAlloc.p1;
  if (budget > 0) {
    const remaining = scored.filter(s => !usedIds.has(s.fragment.id));
    // 估算 P1 能选多少个片段
    const avgTokens = remaining.length > 0
      ? remaining.reduce((s, c) => s + estimateTokenCount([c.fragment.content]), 0) / remaining.length
      : 500;
    const p1Count = Math.max(1, Math.floor(budget / Math.max(1, avgTokens)));

    const mmrCandidates = remaining.map(s => ({
      item: s,
      score: s.score,
      features: s.features,
    }));

    // Causal independence filtering removed (causal-graph.ts deleted — never trained on real data).
    const mmrSelected = mmrReRank(mmrCandidates, p1Count, mmrLambda,
      jaccardSimilarity,
      // v4.1: Use entropyMMRBonus from retrieval-system for content/source diversity
      (candidate) => {
        const frag = (candidate as any).fragment as import("../types.js").ContextFragment;
        if (!frag) return 0;
        const selectedSources = result.map(r => r.fragment.source);
        const selectedTypes = result.map(r => r.fragment.contentType);
        return entropyMMRBonus(frag.source, frag.contentType, selectedSources, selectedTypes);
      },
    );
    for (const item of mmrSelected) {
      const tokens = estimateTokenCount([item.fragment.content]);
      if (tokens <= budget && !usedIds.has(item.fragment.id)) {
        result.push({ fragment: item.fragment, score: item.score });
        usedIds.add(item.fragment.id);
        budget -= tokens;
      }
    }
  }

  // P2: 能力覆盖补充
  budget = budgetAlloc.p2;
  if (budget > 0 && requirements.length > 0) {
    const remaining = scored.filter(s => !usedIds.has(s.fragment.id));
    const sortedReqs = [...requirements].sort((a, b) => b.weight - a.weight);

    for (const req of sortedReqs) {
      if (budget <= 0) break;
      // 找到覆盖此能力但尚未被选中的最佳片段
      const best = remaining
        .filter(s => !usedIds.has(s.fragment.id) && coversCapability(s.fragment, req))
        .sort((a, b) => b.score - a.score)[0];

      if (best) {
        const tokens = estimateTokenCount([best.fragment.content]);
        if (tokens <= budget) {
          result.push({ fragment: best.fragment, score: best.score });
          usedIds.add(best.fragment.id);
          budget -= tokens;
        }
      }
    }
  }

  // P3: 新近度补充（按 lastModified 排序，最新的优先）
  budget = budgetAlloc.p3;
  if (budget > 0) {
    const remaining = scored
      .filter(s => !usedIds.has(s.fragment.id))
      .sort((a, b) => b.fragment.lastModified - a.fragment.lastModified);

    for (const s of remaining) {
      if (budget <= 0) break;
      const tokens = estimateTokenCount([s.fragment.content]);
      if (tokens <= budget) {
        result.push({ fragment: s.fragment, score: s.score });
        usedIds.add(s.fragment.id);
        budget -= tokens;
      }
    }
  }

  return result;
}

/**
 * 判断片段是否覆盖某项能力
 */
function coversCapability(fragment: ContextFragment, requirement: CapabilityRequirement): boolean {
  const text = fragment.content.toLowerCase();
  const capabilityKeywords: Record<string, string[]> = {
    code_understanding: ["function", "class", "interface", "type", "def", "impl", "struct", "enum"],
    pattern_recognition: ["if", "for", "while", "switch", "match", "callback", "promise", "async"],
    code_generation: ["function", "const", "let", "var", "return", "export", "fn", "pub"],
    code_modification: ["mut", "let mut", "var", "set", "update", "push", "append"],
    error_detection: ["error", "exception", "throw", "panic", "unwrap", "catch", "try", "fail"],
    design: ["trait", "interface", "abstract", "impl", "extends", "protocol", "typeclass"],
  };

  const keywords = capabilityKeywords[requirement.name] || [];
  return keywords.some(kw => text.includes(kw));
}

/**
 * Step 5: 压缩单个片段（改进版）
 *
 * 策略:
 * 1. 保留关键结构（函数签名、类定义、接口）
 * 2. 用摘要替代实现细节（超过 5 行）
 * 3. 保留文档注释和多行注释起始
 * 4. 处理 string literal 中的括号避免误跟踪
 */
export function compressFragment(
  fragment: ContextFragment,
  score: number
): CompressedFragment {
  const lines = fragment.content.split("\n");
  const preserved: string[] = [];
  let bodyLines: string[] = [];
  let inBody = false;
  let braceDepth = 0;

  for (let li = 0; li < lines.length; li++) {
    const rawLine = lines[li];
    const trimmed = rawLine.trim();

    // 静态分析：判断是否结构性代码行
    if (isStructuralLine(trimmed)) {
      finalizeBody();
      preserved.push(rawLine);
      // 检查是否开启新的代码块
      const openChars = countOpenBraces(trimmed);
      if (openChars > 0) {
        inBody = true;
        braceDepth += openChars;
      }
      continue;
    }

    // 行级别的括号深度追踪（跳过字符串和注释内容）
    if (inBody) {
      const effectiveBraceDelta = countNetBraces(rawLine);
      braceDepth += effectiveBraceDelta;

      if (braceDepth <= 0) {
        // body 结束
        inBody = false;
        braceDepth = 0;
        finalizeBody();
        // 保留闭合后的行（如 else, catch, finally）
        if (trimmed.startsWith("else") || trimmed.startsWith("catch") ||
            trimmed.startsWith("finally") || trimmed.startsWith("end")) {
          preserved.push(rawLine);
        }
        continue;
      }
      bodyLines.push(rawLine);
    } else if (trimmed.length > 0 && !trimmed.startsWith("//") && !trimmed.startsWith("#")) {
      preserved.push(rawLine);
    }
  }

  // 文件结束前未闭合的 body
  finalizeBody();

  function finalizeBody() {
    if (bodyLines.length > 5) {
      preserved.push(`  // ... [${bodyLines.length} lines omitted]`);
    } else if (bodyLines.length > 0) {
      preserved.push(...bodyLines);
    }
    bodyLines = [];
  }

  return {
    original: fragment,
    score,
    preservedSections: preserved,
  };
}

/** 统计行中的开括号数（跳过字符串字面量） */
function countOpenBraces(line: string): number {
  let count = 0;
  let inStr = false;
  let strChar = "";
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inStr) {
      if (ch === strChar && line[i - 1] !== '\\') inStr = false;
      continue;
    }
    if (ch === '"' || ch === "'" || ch === '`') { inStr = true; strChar = ch; continue; }
    if (ch === '{' || ch === '(' || ch === '[') count++;
  }
  return count;
}

/** 统计行中的净括号数（开括号 - 闭括号，跳过字符串） */
function countNetBraces(line: string): number {
  let net = 0;
  let inStr = false;
  let strChar = "";
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inStr) {
      if (ch === strChar && line[i - 1] !== '\\') inStr = false;
      continue;
    }
    if (ch === '"' || ch === "'" || ch === '`') { inStr = true; strChar = ch; continue; }
    if (ch === '{' || ch === '(' || ch === '[') net++;
    if (ch === '}' || ch === ')' || ch === ']') net--;
  }
  return net;
}

/**
 * 判断一行是否为结构性代码（多语言支持）
 *
 * 涵盖: TypeScript, JavaScript, Python, Rust, Go, Java, Ruby, Swift, Kotlin
 */
function isStructuralLine(line: string): boolean {
  const structuralPatterns = [
    // 函数/方法/类定义
    /^(export\s+)?(function|class|interface|type|enum|struct|trait|impl|def|fn|pub)/i,
    /^(async\s+)?(function|fn|def)/i,
    /^func\s+\w+/i,  // Go
    /^def\s+\w+\s*\(/,  // Python
    /^class\s+\w+/,  // Python, Java, etc.
    // 变量/常量声明
    /^(const|let|var|val)\s+\w+\s*[=:]/,
    /^(let|var)\s+\w+\s*[:=]/i,  // Swift, Kotlin
    // 导入/导出/使用
    /^(import|export|from|use|require|include|mod)\s/,
    /^package\s/,
    /^#include\s/,
    // 访问修饰符
    /^(public|private|protected|static|abstract|sealed|open|internal)/,
    // 语句开头的关键块
    /^(if|else\s+if|for|while|do|switch|match|try|catch|finally|with)\b/,
    // 注解/装饰器
    /^(@\w+|#\[)/,
    // 文档注释
    /^\/\/\/?\s/,
    /^\s*\/\*\*/,
    /^\/\/ =+/,
    // 宏/属性
    /^#\[/i,  // Rust 属性
    /^#\s+(include|define|ifdef|ifndef|endif|pragma)/i,  // C/C++ 预处理
  ];
  return structuralPatterns.some(p => p.test(line));
}

/**
 * 计算能力覆盖度
 */
function computeCoverage(
  compressed: CompressedFragment[],
  requirements: CapabilityRequirement[]
): Record<string, number> {
  const coverage: Record<string, number> = {};
  for (const req of requirements) {
    const covered = compressed.filter(c => coversCapability(c.original, req)).length;
    coverage[req.name] = requirements.length > 0
      ? covered / Math.max(1, compressed.length)
      : 0;
  }
  return coverage;
}

/** 估算 token 数量（粗略：1 token ≈ 4 字符） */
export function estimateTokenCount(texts: string[]): number {
  return texts.reduce((sum, t) => sum + Math.ceil(t.length / 4), 0);
}

function estimateTokens(fragments: ContextFragment[]): number {
  return estimateTokenCount(fragments.map(f => f.content));
}

/** 分词（英文 + 中文） */
function tokenize(text: string): string[] {
  // 提取英文单词
  const english = text.match(/[a-zA-Z_][a-zA-Z0-9_]*/g) || [];
  // 提取中文（按字切分）
  const chinese = text.match(/[一-鿿]/g) || [];
  return [...english.map(w => w.toLowerCase()), ...chinese];
}

/**
 * greedySelect 兼容包装（v3.1: 委托给 greedySelectV2）。
 * 保持旧 API 可用，内部使用默认 MMR λ = 0.65 和简化预算。
 */
function greedySelect(
  scored: { fragment: ContextFragment; score: number }[],
  requirements: CapabilityRequirement[],
  config: { maxTokenBudget: number; minCoverage: number },
  task: Task
): { fragment: ContextFragment; score: number }[] {
  const withFeatures = scored.map(s => ({
    fragment: s.fragment,
    score: s.score,
    features: [s.fragment.contentType, ...s.fragment.source.split("/")],
  }));
  const budgetAlloc = {
    p0: Math.floor(config.maxTokenBudget * 0.60),
    p1: Math.floor(config.maxTokenBudget * 0.20),
    p2: Math.floor(config.maxTokenBudget * 0.15),
    p3: Math.floor(config.maxTokenBudget * 0.05),
  };
  return greedySelectV2(withFeatures, requirements, config, task, budgetAlloc, 0.65);
}

// 导出工具函数以便测试
export const _internal = {
  decomposeTask,
  calculateScore,
  calculateScoreV2,
  computeSemanticSimilarity,
  computeRecency,
  computeExpRecency,
  computeSpecificity,
  computeInfoDensity,
  computeIDFSimilarity,
  greedySelect,
  greedySelectV2,
  compressFragment,
  buildIDFCache,
  buildQueryVector,
  mmrReRank,
  computeTaskJaccard,
};
