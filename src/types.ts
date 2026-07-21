// ============================================================
// TurboContext: Core Type Definitions
// ============================================================

/** 任务定义 */
export interface Task {
  id: string;
  /** 任务描述文本 */
  description: string;
  /** 任务类型分类 */
  type: TaskType;
  /** 复杂度评分 (0-1) */
  complexity?: number;
  /** 质量阈值 (0-1) */
  qualityThreshold?: number;
  /** 延迟预算 (秒) */
  latencyBudget?: number;
}

export type TaskType =
  | "code_generation"
  | "code_review"
  | "code_refactor"
  | "documentation"
  | "analysis"
  | "design"
  | "debugging"
  | "testing"
  | "general";

/** 上下文片段 */
export interface ContextFragment {
  id: string;
  /** 内容 */
  content: string;
  /** 来源路径 */
  source: string;
  /** 最后修改时间 (Unix 时间戳) */
  lastModified: number;
  /** 内容类型 */
  contentType: "source" | "config" | "docs" | "test" | "output" | "other";
  /** 预计算的长度 */
  length: number;
}

/** 压缩后的上下文 */
export interface CompressedContext {
  originalTokens: number;
  compressedTokens: number;
  compressionRatio: number;
  fragments: CompressedFragment[];
  coverage: Record<string, number>; // 能力 -> 覆盖度
}

export interface CompressedFragment {
  original: ContextFragment;
  score: number;
  summary?: string;
  preservedSections: string[];
}

/** 能力需求 */
export interface CapabilityRequirement {
  name: string;
  weight: number;
  description: string;
}

/** 提示组件 */
export interface PromptComponent {
  role: string;
  type: "system" | "user" | "assistant";
  content: string;
  priority: number;
}

/** 生成的输出 */
export interface GenerationOutput {
  content: string;
  qualityScore: number;
  dimensionScores: QualityDimensions;
  attempt: number;
  modelUsed: string;
  tokensUsed: number;
  latencyMs: number;
  /** Execution verification metrics (populated when execution verifier runs) */
  executionMetrics?: ExecutionMetrics;
}

/** Metrics from execution-based verification (compilation, test run). */
export interface ExecutionMetrics {
  compiled: boolean;
  compilerExitCode: number | null;
  compilerErrors: number;
  compilerWarnings: number;
  projectType: string;
  /** Whether the smoke test passed (execution-level verification). */
  smokeTestPassed?: boolean;
}

/** 质量维度评分 */
export interface QualityDimensions {
  completeness: number;   // 完整性 (0-1)
  correctness: number;    // 正确性 (0-1)
  consistency: number;    // 一致性 (0-1)
  format: number;         // 格式合规 (0-1)
}

/** 质量评估结果 */
export interface QualityAssessment {
  score: number;
  dimensions: QualityDimensions;
  issues: QualityIssue[];
  passed: boolean;
}

export interface QualityIssue {
  dimension: keyof QualityDimensions;
  severity: "critical" | "major" | "minor";
  description: string;
  suggestion: string;
}

/** 模型层级 */
export type ModelTier = "fast" | "medium" | "deep";

export interface ModelConfig {
  tier: ModelTier;
  model: string;
  costPer1KTokens: number;
  avgLatencyMs: number;
  capabilities: string[];
}

/** 执行记录 */
export interface ExecutionRecord {
  taskId: string;
  taskType: TaskType;
  timestamp: number;
  compressionRatio: number;
  qualityScore: number;
  totalCost: number;
  latencyMs: number;
  attemptCount: number;
  modelUsed: string;
  coverage: Record<string, number>;
  dimensionScores: QualityDimensions;
  /** 本次执行涉及的源文件列表（供 compressor outcome boost） */
  sourceFiles?: string[];
  /** Execution verification metrics (populated when execution verifier runs) */
  executionMetrics?: ExecutionMetrics;
}

/** 算法配置 */
export interface TurboContextConfig {
  // 压缩阶段
  alpha: number;       // 语义相似度权重
  beta: number;        // 新鲜度权重
  gamma: number;       // 特异性权重
  maxTokenBudget: number;
  minCoverage: number; // 最低能力覆盖率

  // 生成阶段
  qualityThreshold: number;
  maxAttempts: number;
  temperatureSchedule: number[];

  // 优化阶段
  complexityThresholdLow: number;
  complexityThresholdHigh: number;

  // 学习阶段
  learningRate: number;
  historyWindow: number; // 保留最近 N 条记录
}

/** 分支轨迹统计 */
export interface BranchTrajectory {
  momentum: number;              // 改进速率（正=向好）
  improvementVelocity: number;   // 质量变化速度
  stabilityScore: number;        // 稳定性 (0-1)
  noveltyScore: number;          // 任务多样性 (0-1)
  qualityHistory: number[];      // 最近质量评分历史
  recentTaskIds: string[];       // 最近任务 ID
}

/** v3.8: Per-task-type source file stats (Causal Markov Condition — condition on task type) */
export interface PerTypeSourceStats {
  attempts: number;
  successes: number;
  lastQuality: number;
  /** Per-type ablated causal utility (if ablated for this task type) */
  ablatedCausalUtility?: number;
  ablationCount?: number;
}

/** 源文件历史表现 */
export interface SourceMemory {
  attempts: number;
  successes: number;
  lastQuality: number;
  lastUsed: number;  // Unix 时间戳
  /** v3.6: Clean causal signal from per-file ablation experiments (global) */
  ablatedCausalUtility?: number;
  /** v3.6: Number of times this file has been ablated (global) */
  ablationCount?: number;
  /** v3.8: Per-task-type conditional stats (Causal Markov Condition) */
  perType?: Partial<Record<TaskType, PerTypeSourceStats>>;
}

/** v3.6: Result of a per-file ablation experiment */
export interface AblationResult {
  sourceFile: string;
  taskType: TaskType;
  qualityWith: number;
  qualityWithout: number;
  causalDelta: number;       // qualityWith - qualityWithout, range [-1, +1]
  confidence: number;        // 0-1, reliability of this delta
  withCompiled?: boolean;
  withoutCompiled?: boolean;
  durationMs: number;
}

/** 单条失败记录 */
export interface FailureRecord {
  taskId: string;
  timestamp: number;
  qualityScore: number;
  attemptCount: number;
  failureReasons: string[];
}

/** 分支状态（每种 TaskType 一个分支） */
export interface BranchState {
  type: TaskType;
  totalExperiments: number;
  bestQuality: number;
  bestDescription: string;
  successCount: number;        // quality ≥ threshold
  failureCount: number;        // quality < threshold 或重试耗尽
  trajectory: BranchTrajectory;
  recentFailures: FailureRecord[];
  summary: string;             // 自动生成的总结
  lastSummaryExperimentCount: number;
  // 分支级配置覆盖（null = 使用全局值）
  alphaOverride: number | null;
  betaOverride: number | null;
  gammaOverride: number | null;
  qualityThresholdOverride: number | null;
  temperatureScheduleOverride: number[] | null;
}

/** 算法状态（持久化用） */
export interface AlgorithmState {
  config: TurboContextConfig;
  executionHistory: ExecutionRecord[];
  branches: Record<string, BranchState>;       // per-task-type branch state
  sourceMemory: Record<string, SourceMemory>;  // per-source outcome tracking
  strategyEvolution: StrategyEvolutionData;    // evolution experiments
  capabilityCoverageMatrix: Record<string, Record<string, number>>;
  taskComplexityCache: Record<string, number>;
  promptComponentUsage: Record<string, { uses: number; avgQuality: number }>;
}

// ---------------------------------------------------------------------------
// Strategy Evolution (v2.2 — 受 autoresearch 启发)
// ---------------------------------------------------------------------------

/** 分解策略的变异操作 */
/** 实验类型（autoresearch: hypothesis_test, parameter_sweep 等） */
export type ExperimentType =
  | "hypothesis_test"     // 测试一个具体假设
  | "parameter_sweep"     // 扫描参数空间
  | "ablation_study"      // 消融实验（移除某个特性）
  | "transfer_experiment" // 跨任务类型迁移
  | "boundary_probe"      // 探索崩溃边界
  | "adversarial_test";   // 对抗验证（挑战已有结论）

export type StrategyMutation =
  | { type: "merge_rounds"; roundIndices: [number, number]; newGoal: string; complexityDelta?: number }
  | { type: "split_round"; roundIndex: number; newGoalA: string; newGoalB: string; complexityDelta?: number }
  | { type: "remove_round"; roundIndex: number; complexityDelta?: number }
  | { type: "reorder_rounds"; newOrder: number[]; complexityDelta?: number }
  | { type: "add_quality_criterion"; roundIndex: number; criterion: string; complexityDelta?: number }
  | { type: "remove_quality_criterion"; roundIndex: number; criterionIndex: number; complexityDelta?: number }
  // v6: Multi-target mutations (Karpathy-inspired experiment type diversity)
  | { type: "mutate_compression_weights"; alpha: number; beta: number; gamma: number; complexityDelta?: number }
  | { type: "mutate_model_tiers"; theta1: number; theta2: number; complexityDelta?: number }
  | { type: "mutate_temperature"; schedule: [number, number, number]; complexityDelta?: number }
  | { type: "mutate_quality_weights"; dimWeights: [number, number, number, number]; complexityDelta?: number }
  | { type: "mutate_retrieval"; mmrLambda: number; topK: number; complexityDelta?: number };

/** 一次进化实验 */
export interface EvolutionExperiment {
  id: string;
  taskType: TaskType;
  parentStrategyId: string;
  mutation: StrategyMutation;
  status: "trial" | "kept" | "discarded" | "pending" | "crashed";
  trialCount: number;
  trialQualitySum: number;
  trialTokensSum: number;
  baselineCount: number;
  baselineQualitySum: number;
  baselineTokensSum: number;
  crashedEarly: boolean;
  startedAt: number;
  concludedAt: number | null;
}

/** 单条试验日志（autoresearch results.tsv 等价物） */
export interface TrialLogEntry {
  experimentId: string;
  taskType: TaskType;
  usingMutation: boolean;
  qualityScore: number;
  tokensUsed: number;
  timestamp: number;
  status: "success" | "crash" | "timeout";
}

/** 进化状态 */
export interface StrategyEvolutionData {
  experiments: EvolutionExperiment[];
  currentExperimentId: string | null;  // 正在进行的 trial
  totalExperiments: number;
  keptCount: number;
  discardedCount: number;
  /** 各任务类型的已保留变异栈（autoresearch: branch tip = best config） */
  canonicalStrategies: Record<string, StrategyMutation[]>;
  /** 所有 trial 的完整日志（autoresearch: results.tsv） */
  trialLog: TrialLogEntry[];
}

// ============================================================
// v3.0 — Autoresearch-inspired patterns
// ============================================================

/**
 * 统一效率指标（autoresearch 的 val_bpb 等价物）。
 * 单一数字，越低/越高越好，使实验间可直接比较。
 *
 * efficiency = qualityScore / (totalCost + ε)
 * 含义：每单位成本获得的平均质量。越高越好。
 */
export interface UnifiedMetric {
  /** 效率值（越高越好） */
  efficiency: number;
  /** 质量评分 (0-1) */
  quality: number;
  /** 总成本 (USD) */
  cost: number;
  /** 耗时 (ms) */
  latencyMs: number;
  /** 尝试次数 */
  attempts: number;
  /** 质量/成本权衡系数（默认 1.0，>1 = 更重视质量） */
  alpha: number;
  /** 简洁性加成（0-1，1 = 最简洁） */
  simplicityMultiplier: number;
}

/**
 * 单次自主实验运行记录（autoresearch 的一次 train.py run 等价物）。
 * 固定预算下的原子实验，结果与 baseline 直接比较。
 */
export interface ExperimentRun {
  /** 实验 ID */
  id: string;
  /** 任务类型 */
  taskType: TaskType;
  /** 时间戳 */
  timestamp: number;
  /** 应用的变异（null = baseline） */
  mutation: StrategyMutation | null;
  /** baseline 的统一指标 */
  baselineMetric: UnifiedMetric;
  /** 本次实验的统一指标 */
  experimentMetric: UnifiedMetric;
  /** 决策 */
  decision: "keep" | "discard";
  /** 改进幅度（百分比，正=变好） */
  deltaPercent: number;
  /** 耗时 (ms) */
  wallClockMs: number;
  /** 状态 */
  status: "success" | "crash" | "timeout" | "discarded";
  /** 崩溃原因（如有） */
  crashReason?: string;
  /** 实验编号（从 1 开始） */
  runNumber: number;
  /** v6: 实验类型（hypothesis_test, parameter_sweep 等） */
  experimentType: ExperimentType;
  /** v6: 简洁性评分（0-1，1 = 最简洁） */
  simplicityScore: number;
}

/**
 * 自主实验循环配置（autoresearch 的 program.md 等价物）。
 * 人类编辑此文件来引导 agent 的研究方向，
 * agent 只修改策略参数和变异提案。
 */
export interface Mission {
  /** 研究目标（一句话） */
  goal: string;
  /** 每次实验的固定 token 预算 */
  tokenBudgetPerRun: number;
  /** 每次实验的固定时间预算 (秒) */
  timeBudgetPerRun: number;
  /** 最大实验次数 */
  maxExperiments: number;
  /** 评估用的任务池（agent 从中采样） */
  taskPool: Task[];
  /** 上下文片段池（agent 从中采样） */
  contextPool: ContextFragment[];
  /** 变异方向约束（留空=自由探索） */
  explorationConstraints?: {
    /** 允许的变异类型白名单 */
    allowedMutations?: string[];
    /** 不允许触碰的参数 */
    frozenParams?: string[];
  };
  /** 人类备注（agent 读取但不修改） */
  humanNotes: string;
}

/**
 * 实验日志行（results.tsv 格式）。
 * 一行 = 一次实验，便于人类 morning review。
 */
export interface ExperimentLogEntry {
  run: number;
  timestamp: string;
  taskType: string;
  mutationType: string;
  mutationDesc: string;
  baselineEfficiency: number;
  experimentEfficiency: number;
  deltaPercent: number;
  decision: string;
  quality: number;
  cost: number;
  attempts: number;
  wallClockSec: number;
  status: string;
}

// ============================================================
// v3.1 — Autoresearch-optimized types (plateau detection, strategic
//        directives, contrastive pairs, MMR retrieval, IDF cache)
// ============================================================

/**
 * 平台期检测结果（autoresearch: quantitative branch health signals）。
 * 4 条检测规则，每条有独立置信度。
 */
export interface PlateauSignal {
  /** 是否处于平台期 */
  isPlateaued: boolean;
  /** 平台期原因 */
  reason: PlateauReason;
  /** 置信度 0-1（>0.7 = 强信号） */
  confidence: number;
  /** 各规则的独立判定 */
  rules: PlateauRuleResult[];
}

export type PlateauReason =
  | "none"
  | "improvement_stall"    // 最近 3 次 vs 前 2 次无改进
  | "crash_dominant"        // 崩溃率 > 成功率 * 2
  | "novelty_collapse"      // 最近假设几乎相同
  | "slow_decline";         // 后半段平均质量 < 前半段

export interface PlateauRuleResult {
  rule: string;
  triggered: boolean;
  confidence: number;
  detail: string;
}

/**
 * 战略指令（autoresearch: high-level planner guidance）。
 * 根据分支健康状态生成，指导下一步行动。
 */
export interface StrategicDirective {
  /** 指令类型 */
  directive: "MOMENTUM" | "PLATEAU" | "CAUTION" | "DIVERSIFY" | "STEADY" | "EXPLORE";
  /** 人类可读的指令文本 */
  message: string;
  /** 触发指令的指标快照 */
  metrics: {
    velocity: number;
    stability: number;
    novelty: number;
    totalExperiments: number;
    successRate: number;
  };
  /** 建议行动 */
  suggestedAction: string;
}

/**
 * 对比对（autoresearch: highest-signal context）。
 * 找到相似任务但相反结果的实验对，给 planner 因果洞察。
 */
export interface ContrastivePair {
  /** 成功的实验 */
  success: {
    taskType: TaskType;
    description: string;
    quality: number;
    sourceFiles: string[];
  };
  /** 失败的实验 */
  failure: {
    taskType: TaskType;
    description: string;
    quality: number;
    sourceFiles: string[];
    failureMode: string;
  };
  /** 共享的特征（关键词/文件/能力） */
  sharedFeatures: string[];
  /** Jaccard 相似度 */
  similarity: number;
  /** 自动生成的洞察文本 */
  insight: string;
}

/**
 * MMR 检索结果（autoresearch: diversity-aware retrieval）。
 */
export interface MMRRetrievalResult {
  /** 选中的片段 */
  selected: string[];
  /** 每步的 MMR 分数 */
  scores: number[];
  /** 使用的 λ 值（0=完全多样性, 1=完全相关性） */
  lambda: number;
  /** 多样性得分（选中片段间的平均不相似度） */
  diversityScore: number;
}

/**
 * 全局 IDF 缓存（autoresearch: IDF-weighted retrieval）。
 * 维护在所有片段中词的逆文档频率。
 */
export interface IDFCache {
  /** 词 → IDF 权重 */
  weights: Record<string, number>;
  /** 构建时的文档总数 */
  documentCount: number;
  /** 最后更新时间戳 */
  lastUpdated: number;
  /** 停用词集合（不计入 IDF） */
  stopWords: Set<string>;
}

/**
 * 检索评分维度权重（autoresearch: 10-dimension scoring, v4 extended）。
 */
export interface RetrievalWeights {
  /** IDF 加权语义相似度 (0-10 归一化) */
  semanticWeight: number;
  /** 任务类型重叠 (0-5 归一化) */
  taskOverlapWeight: number;
  /** 分支匹配 (0-3) */
  branchMatchWeight: number;
  /** 指数衰减新近度 (0-3) */
  recencyWeight: number;
  /** 结果加成 (0-2) */
  outcomeBonusWeight: number;
  /** 信息密度加成 (0-2) */
  infoDensityWeight: number;
  /** v4.0: Surprise bonus weight (0-3), curriculum-phase adaptive */
  surpriseWeight: number;
  /** v4.0: Curiosity/EIG bonus weight (0-3), curriculum-phase adaptive */
  curiosityWeight: number;
  /** v4.0: Counterfactual insight bonus (0-2), flat +1.5 if present */
  counterfactualWeight: number;
}

export const DEFAULT_RETRIEVAL_WEIGHTS: RetrievalWeights = {
  semanticWeight: 10.0,
  taskOverlapWeight: 5.0,
  branchMatchWeight: 3.0,
  recencyWeight: 3.0,
  outcomeBonusWeight: 2.0,
  infoDensityWeight: 2.0,
  surpriseWeight: 1.0,
  curiosityWeight: 1.0,
  counterfactualWeight: 1.0,
};

// ============================================================
// v3.3 — RL core types (from Karpathy autoresearch)
// ============================================================

/**
 * Thompson Sampling parameters for retrieval exploration.
 * Each memory maintains Beta(α,β) over its retrieval value.
 * Sampling during retrieval naturally balances explore/exploit.
 */
export interface ThompsonParams {
  alphaTs: number;   // successes observed
  betaTs: number;    // failures observed
}

/**
 * Eligibility trace for TD(λ) credit assignment.
 * Memories that contributed to success via a chain of retrievals
 * get partial credit, decaying exponentially with distance.
 */
export interface EligibilityTrace {
  memoryId: string;
  trace: number;     // decayed by γλ each iteration, bumped on retrieval
}

/**
 * Online predictive model state.
 * Lightweight linear model with SGD that predicts experiment outcomes.
 * Used for: surprise computation, curiosity bonus, planner guidance.
 */
export interface PredictiveModel {
  featureWeights: Record<string, number>;
  intercept: number;
  nUpdates: number;
  learningRate: number;
  recentAccuracy: number;  // EMA of prediction accuracy
}

/**
 * Curriculum learning phase.
 * Phase 0: broad exploration (high diversity, large mutations)
 * Phase 1: focused exploitation (deepen promising branches)
 * Phase 2: principled optimization (fine-tune based on learned principles)
 * Phase 3: adversarial refinement (challenge assumptions, verify old results)
 */
export type CurriculumPhase = 0 | 1 | 2 | 3;

export interface CurriculumState {
  phase: CurriculumPhase;
  phaseBoundaries: [number, number, number];  // exp counts for transitions
  phaseNames: [string, string, string, string];
  phaseParams: Record<CurriculumPhase, CurriculumPhaseParams>;
}

export interface CurriculumPhaseParams {
  mmrLambda: number;
  explorationBonus: number;
  mutationMagnitude: number;
  curiosityWeight: number;
  surpriseWeight: number;
  adversarialInterval: number;
  consolidationInterval: number;
  description: string;
}

/** Default curriculum phase boundaries and params */
export const DEFAULT_CURRICULUM: CurriculumState = {
  phase: 0,
  phaseBoundaries: [10, 30, 60],
  phaseNames: ["broad_exploration", "focused_exploitation",
               "principled_optimization", "adversarial_refinement"],
  phaseParams: {
    0: { mmrLambda: 0.35, explorationBonus: 2.0, mutationMagnitude: 0.25,
         curiosityWeight: 1.5, surpriseWeight: 1.2,
         adversarialInterval: 20, consolidationInterval: 15,
         description: "Broad exploration: try diverse task types, large changes" },
    1: { mmrLambda: 0.55, explorationBonus: 1.0, mutationMagnitude: 0.15,
         curiosityWeight: 1.0, surpriseWeight: 1.0,
         adversarialInterval: 15, consolidationInterval: 12,
         description: "Focused exploitation: deepen promising branches" },
    2: { mmrLambda: 0.70, explorationBonus: 0.5, mutationMagnitude: 0.08,
         curiosityWeight: 0.5, surpriseWeight: 0.8,
         adversarialInterval: 10, consolidationInterval: 10,
         description: "Principled optimization: fine-tune based on learned principles" },
    3: { mmrLambda: 0.60, explorationBonus: 0.8, mutationMagnitude: 0.06,
         curiosityWeight: 0.7, surpriseWeight: 1.0,
         adversarialInterval: 8, consolidationInterval: 8,
         description: "Adversarial refinement: challenge assumptions, verify old results" },
  },
};

/**
 * Experience library entry — records (scenario, mutation, outcome, delta)
 * to learn which mutations work best in which contexts.
 */
export interface ExperienceEntry {
  scenario: Record<string, string | number | string[]>;
  mutation: { target: string; old: unknown; new: unknown };
  outcome: "keep" | "revert" | "crash";
  delta: number;
  timestamp: string;
}

/**
 * Retrieval strategy state — self-evolution of the retrieval algorithm itself.
 * The same proposeMutation → recordTrial → decideKeepDiscard loop
 * is applied to retrieval hyperparameters.
 */
export interface RetrievalStrategyState {
  /** Evolvable scoring dimension weights */
  dimWeights: Record<string, number>;
  mmrLambda: number;
  topK: number;
  /** Token budget tiers: [<N_exps, <M_exps, >=M_exps] */
  tokenBudgetTiers: [number, number, number];
  /** Evolution state */
  generation: number;
  fitness: number;               // EMA of recent planner success rate
  fitnessHistory: Array<{
    generation: number;
    fitness: number;
    delta: number;
    decision: string;
    mutation: Record<string, unknown>;
    trials: number;
  }>;
  trialsInGeneration: number;
  successesInGeneration: number;
  mutationMagnitude: number;     // evolves: smaller when fit, larger when stuck
  pendingMutation: Record<string, unknown> | null;
  ancestorFitness: number;
  /** UCB tracking per dimension (for bandit-based mutation selection) */
  ucbDimCounts: Record<string, number>;
  ucbDimRewards: Record<string, number>;
  ucbTotalMutations: number;
}

/** Default retrieval strategy state */
export function createDefaultRetrievalStrategy(): RetrievalStrategyState {
  return {
    dimWeights: {
      hypothesis_similarity: 1.0,
      subsystem_overlap: 1.0,
      branch_match: 1.0,
      recency: 1.0,
      outcome_bonus: 1.0,
      info_density: 1.0,
      surprise_bonus: 1.0,
      curiosity_bonus: 1.0,
      counterfactual_bonus: 1.0,
    },
    mmrLambda: 0.65,
    topK: 5,
    tokenBudgetTiers: [1200, 2000, 2800],
    generation: 0,
    fitness: 0.5,
    fitnessHistory: [],
    trialsInGeneration: 0,
    successesInGeneration: 0,
    mutationMagnitude: 0.15,
    pendingMutation: null,
    ancestorFitness: 0.5,
    ucbDimCounts: {},
    ucbDimRewards: {},
    ucbTotalMutations: 0,
  };
}

/**
 * Adversarial verification record — periodic re-evaluation of old "success"
 * memories against current knowledge.
 */
export interface VerificationRecord {
  experimentCount: number;
  currentBest: number;
  currentAvg: number;
  gapToBest: number;
  newConfidence: number;
  timestamp: string;
}

/**
 * Memory consolidation attribution — tracks what information was
 * lost during compression for auditability.
 */
export interface ConsolidationAttribution {
  timestamp: string;
  preCoverageCount: number;
  postCoverageCount: number;
  coverageLoss: string[];
  totalTokensSaved: number;
  groups: Array<{
    subsystem: string;
    sources: number;
    tokensSaved: number;
    successes: number;
    failures: number;
  }>;
}

// ============================================================
// v4.0 — Karpathy-inspired types (surprise tracking, two-phase retrieval,
//        consolidation undo, counterfactual reasoning)
// ============================================================

/**
 * v4.0: Rolling surprise tracking (agent.py _update_surprise_stats).
 *
 * Maintains last 50 surprise values and a running global mean
 * used to normalize surprise bonuses in retrieval scoring.
 * Karpathy: "Tight feedback loops — every decision should have
 * measurable consequence."
 */
export interface SurpriseStats {
  /** Rolling average of last 50 surprise values */
  globalMeanSurprise: number;
  /** Last 50 values, FIFO */
  surpriseHistory: number[];
}

/**
 * v4.0: Two-phase retrieval configuration (agent.py Phase 1 + Phase 2).
 *
 * Phase 1: Similarity pool — score all memories, keep top oversample × topK.
 * Phase 2: Causal re-rank — add advantage-weighted causal utility, re-score.
 */
export interface TwoPhaseRetrievalConfig {
  /** Phase 1 oversampling multiplier (pool = topK × oversample) */
  oversampleMultiplier: number;
  /** Phase 2 advantage scaling factor (maps advantage to causal score) */
  advantageScale: number;
  /** Maximum Phase 1 dimensions used before causal re-rank */
  maxPhase1Dimensions: number;
}

export const DEFAULT_TWO_PHASE_CONFIG: TwoPhaseRetrievalConfig = {
  oversampleMultiplier: 2.5,
  advantageScale: 6.0,
  maxPhase1Dimensions: 9,
};

/**
 * v4.0: Cold storage undo log entry for each archived/consolidated memory.
 *
 * Karpathy principle: "Compression with attribution — know what was lost."
 * Stores enough metadata to recover or audit the archiving decision.
 */
export interface ConsolidationUndoInfo {
  /** Original experiment/memory ID before consolidation */
  originalId: string;
  /** Which consolidated entry this was merged into (null = cold storage) */
  mergedInto: string | null;
  /** First 200 chars of original hypothesis/description */
  originalHypothesis: string;
  /** Original quality score at time of archiving */
  originalQualityScore: number;
  /** ISO timestamp of when consolidation occurred */
  consolidatedAt: string;
}

/**
 * Enhanced execution record with RL tracking fields.
 */
export interface RLExecutionRecord extends ExecutionRecord {
  /** Thompson Sampling parameters */
  thompsonAlpha: number;
  thompsonBeta: number;
  /** Retrieval utility: this experiment's own outcome value (0-1) */
  retrievalUtility: number;
  /** Causal utility: "does showing this memory help?" (learned from downstream) */
  causalUtility: number;
  /** Which memories were shown to produce this execution */
  retrievedMemoryKeys: string[];
  /** Which memories the planner actually used (semantic overlap > threshold) */
  plannerReferencedKeys: string[];
  /** Surprise: |predicted - actual| */
  surpriseScore: number;
  /** Predicted outcome before execution (from predictive model) */
  predictedOutcome: number | null;
  /** Prediction error magnitude */
  predictionError: number | null;
  /** Counterfactual insight */
  counterfactual: string;
  /** Curriculum phase that produced this */
  curriculumPhase: CurriculumPhase;
  /** Adversarial verification history */
  verificationHistory: VerificationRecord[];
  /** Whether this entry has been consolidated */
  consolidated: boolean;
  /** If consolidated, which memory it was merged into */
  consolidatedInto: string | null;
  /** v4.0: Undo log for consolidated/archived memories (recovery metadata) */
  _consolidation_undo_info?: ConsolidationUndoInfo;
}

// ============================================================
// v4.1 — Causal Discovery Types (PC, FCI, GES, do-calculus)
// ============================================================
// From Spirtes-Glymour-Scheines (2000): Causation, Prediction, and Search
// Pearl (2009): Causality — do-calculus, back-door, front-door
// Zhang (2008): Completeness of orientation rules for causal discovery

/**
 * v4.1: FCI edge types for Partial Ancestral Graphs (PAGs).
 *
 * Unlike PC which outputs CPDAGs (assuming no latent confounders),
 * FCI outputs PAGs that can represent:
 *   - Direct causation:  A ∘→ B  or  A → B
 *   - Latent confounding: A ↔ B  (unmeasured common cause)
 *   - Selection bias:     A — B  (undirected, from conditioning on common effect)
 *   - Uncertainty:        A ∘−∘ B (unknown relationship)
 */
export type FCIEdgeMark = "tail" | "arrow" | "circle";

export interface FCIEdge {
  source: string;
  target: string;
  /** Left-side mark on source: tail (−), arrow (→), or circle (∘) */
  sourceMark: FCIEdgeMark;
  /** Right-side mark on target: tail (−), arrow (→), or circle (∘) */
  targetMark: FCIEdgeMark;
  /** Edge type string: "∘→" "↔" "→" "∘−∘" "—" */
  edgeType: string;
  strength: number;
  evidence: "conditional_independence" | "possible_dsep" | "orientation_rule";
}

/**
 * v4.1: Partial Ancestral Graph (FCI output).
 *
 * Represents an equivalence class of Maximal Ancestral Graphs (MAGs),
 * each encoding conditional independence among observed variables
 * when latent variables and selection bias may be present.
 */
export interface PAG {
  nodes: Set<string>;
  edges: FCIEdge[];
  /** Possible-D-SEP sets: nodes potentially in d-separating sets beyond adjacency */
  possibleDSep: Map<string, Set<string>>;
  /** Separation sets from skeleton discovery */
  sepSets: Map<string, Map<string, string[]>>;
  lastUpdated: number;
}

/**
 * v4.1: GES (Greedy Equivalence Search) configuration.
 *
 * Chickering (2002): Optimal Structure Identification With Greedy Search.
 * GES searches over Markov equivalence classes (CPDAGs) using a
 * decomposable score function, complementary to constraint-based methods.
 */
export interface GESConfig {
  /** Score function: "bic" (linear Gaussian) or "bdeu" (discrete) */
  scoreFunction: "bic" | "bdeu";
  /** Maximum number of parents per node (limits search complexity) */
  maxParents: number;
  /** Equivalent sample size for BDeu prior (default 1) */
  equivalentSampleSize: number;
  /** Whether to run the turning phase (Hauser & Bühlmann 2012) */
  enableTurningPhase: boolean;
  /** Penalty discount for BIC (higher = sparser graphs) */
  penaltyDiscount: number;
}

export const DEFAULT_GES_CONFIG: GESConfig = {
  scoreFunction: "bic",
  maxParents: 5,
  equivalentSampleSize: 1,
  enableTurningPhase: true,
  penaltyDiscount: 1.0,
};

/**
 * v4.1: Bootstrap confidence for causal edges.
 *
 * Resample ablation data to estimate edge stability.
 * An edge that appears in 95% of bootstrap samples is reliable;
 * one that appears in 40% is speculative.
 */
export interface BootstrapEdgeConfidence {
  source: string;
  target: string;
  /** Proportion of bootstrap samples where this edge appeared */
  stability: number;
  /** Proportion of samples where edge was oriented source→target */
  directionProb: number;
  /** Whether edge is "robust" (stability ≥ 0.8) */
  robust: boolean;
  /** Number of bootstrap iterations */
  iterations: number;
}

/**
 * v4.1: Intervention calculus result.
 *
 * do-calculus formalization of ablation experiments.
 * do(X = removed) means surgically removing file X from the context.
 */
export interface InterventionEffect {
  /** Target variable (file being intervened on) */
  target: string;
  /** Outcome variable (quality score) */
  outcome: string;
  /** Estimated Average Causal Effect: E[Q | do(X=removed)] - E[Q | do(X=present)] */
  ace: number;
  /** Whether the effect is identifiable from observational data alone */
  identifiable: boolean;
  /** Identification method used */
  method: "back_door" | "front_door" | "experimental" | "none";
  /** Adjustment set (variables to condition on for unbiased estimation) */
  adjustmentSet: string[];
  /** Confidence interval [lower, upper] */
  ciLower: number;
  ciUpper: number;
}

/**
 * v4.1: Back-door criterion check result.
 *
 * A set Z satisfies the back-door criterion for (X, Y) if:
 *   1. No node in Z is a descendant of X
 *   2. Z blocks all back-door paths from X to Y
 *
 * If such a Z exists, the causal effect can be estimated by adjusting for Z.
 */
export interface BackDoorResult {
  /** Whether a sufficient adjustment set exists */
  satisfiable: boolean;
  /** Minimal adjustment set (if satisfiable) */
  adjustmentSet: string[];
  /** All back-door paths that need blocking */
  backDoorPaths: string[][];
}

/**
 * v4.1: Causal minimality assumption (weaker alternative to faithfulness).
 *
 * Faithfulness can be violated by "accidental cancellations" (path cancellations).
 * Minimality only requires that no edge can be removed without changing the
 * independence relations — a strictly weaker condition.
 *
 * When faithfulness is suspected to be violated, fall back to minimality.
 */
export interface CausalMinimalityCheck {
  /** Whether faithfulness is likely violated for this edge */
  faithfulnessViolated: boolean;
  /** Whether minimality still supports this edge (edge is "minimal") */
  minimalitySatisfied: boolean;
  /** The conditional independence that would make this edge removable */
  criticalCITest: { x: string; y: string; conditioning: string[] } | null;
  /** Confidence in the edge direction under minimality */
  minimalDirectionConfidence: number;
}
