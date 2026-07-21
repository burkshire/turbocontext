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

// ============================================================
// IDF Cache (used by compressor Phase 1)
// ============================================================

export interface IDFCache {
  weights: Record<string, number>;
  documentCount: number;
  lastUpdated: number;
  stopWords: Set<string>;
}

// ============================================================
// Plan B: Session Memory Types
// ============================================================

export interface SessionRecord {
  id: string;
  timestamp: string;
  taskDescription: string;
  taskType: TaskType;
  workingDirectory: string;
  gitRef?: string;
  filesRead: string[];
  filesModified: string[];
  strategy: string;
  outcome: "success" | "partial" | "failure";
  selfAssessment: number;
  notes: string;
  keywords: string[];
  roundCount: number;
  tokenEstimate?: number;
}

export interface RecallRequest {
  taskDescription: string;
  workingDirectory: string;
  taskType?: TaskType;
  maxResults?: number;
  minSimilarity?: number;
}

export interface RecallResult {
  similarSessions: SimilarSession[];
  recommendedFiles: FileRecommendation[];
  recommendedStrategies: StrategyRecommendation[];
  summary: string;
  corpusStats: CorpusStats;
}

export interface SimilarSession {
  session: SessionRecord;
  similarity: number;
  matchReason: string;
}

export interface FileRecommendation {
  path: string;
  relevanceScore: number;
  sessionCount: number;
  usageType: "read" | "modified" | "both";
}

export interface StrategyRecommendation {
  strategy: string;
  score: number;
  occurrenceCount: number;
}

export interface CorpusStats {
  totalSessions: number;
  oldestSession: string;
  newestSession: string;
  perTaskType: Record<string, number>;
  totalUniqueFiles: number;
}

