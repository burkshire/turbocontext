// ============================================================================
// Turbocontext v5 — Core State Types
// ============================================================================
// Every type defined here is validated on load by validation.ts.
// RL theory: these types encode the MDP state space for the policy learner.
// Each Trial is a (s,a,r,s') transition; IndexedMemory is a distilled
// experience with Thompson-sampled retrieval utility.

// ── Primitives ──
export type UUID = string; // v4, validated by pattern
export type Hex16 = string; // first 16 hex chars of SHA-256
export type QualityDim = 0 | 1 | 2 | 3; // completeness, correctness, consistency, format

// ── Enums ──
export const TaskType = {
  CODE_REVIEW: "code_review",
  CODE_GENERATION: "code_generation",
  DEBUGGING: "debugging",
  REFACTORING: "refactoring",
  DOCUMENTATION: "documentation",
  ARCHITECTURE: "architecture",
} as const;
export type TaskType = (typeof TaskType)[keyof typeof TaskType];

/**
 * Map a V4/core TaskType (9 values) to the canonical V5 TaskType (6 values).
 *
 * Core TaskType has organic-growth values that don't map 1:1 to V5.
 * This function provides the canonical coercion, used at all cross-boundary
 * call sites so every module agrees on the type space.
 *
 * Mapping:
 *   code_generation → code_generation
 *   code_review     → code_review
 *   debugging       → debugging
 *   code_refactor   → refactoring
 *   documentation   → documentation
 *   analysis        → architecture
 *   design          → architecture
 *   testing         → code_generation   (nearest match)
 *   general         → code_generation   (nearest match)
 */
export function toV5TaskType(v4Type: string): TaskType {
  switch (v4Type) {
    case "code_generation": return TaskType.CODE_GENERATION;
    case "code_review":     return TaskType.CODE_REVIEW;
    case "debugging":       return TaskType.DEBUGGING;
    case "code_refactor":   return TaskType.REFACTORING;
    case "documentation":   return TaskType.DOCUMENTATION;
    case "analysis":
    case "design":          return TaskType.ARCHITECTURE;
    case "testing":
    case "general":
    default:                return TaskType.CODE_GENERATION;
  }
}

/**
 * Map a V5 TaskType back to the closest V4/core TaskType.
 * All V5 values exist in V4, so this is a direct identity cast.
 */
export function toV4TaskType(v5Type: TaskType): string {
  return v5Type; // all 6 V5 values are valid V4 TaskType strings
}

export const ModelTier = { FAST: "fast", MEDIUM: "medium", BEST: "best" } as const;
export type ModelTier = (typeof ModelTier)[keyof typeof ModelTier];

export const Outcome = { SUCCESS: "success", FAILURE: "failure", CRASH: "crash" } as const;
export type Outcome = (typeof Outcome)[keyof typeof Outcome];

export const ContextOrigin = { SKILL: "skill", AUTONOMOUS: "autonomous" } as const;
export type ContextOrigin = (typeof ContextOrigin)[keyof typeof ContextOrigin];

export const MemoryStatus = { ACTIVE: "active", COLD: "cold", CONSOLIDATED: "consolidated" } as const;
export type MemoryStatus = (typeof MemoryStatus)[keyof typeof MemoryStatus];

export const CurriculumPhase = {
  BROAD_EXPLORATION: 0,
  FOCUSED_EXPLOITATION: 1,
  PRINCIPLED_OPTIMIZATION: 2,
  ADVERRIAL_REFINEMENT: 3,
} as const;
export type CurriculumPhase = (typeof CurriculumPhase)[keyof typeof CurriculumPhase];

export const ConsolidationAction = { CONSOLIDATE: "consolidate", ARCHIVE_COLD_STORAGE: "archive_cold_storage" } as const;
export type ConsolidationAction = (typeof ConsolidationAction)[keyof typeof ConsolidationAction];

export const EvolutionDecision = { KEEP: "keep", REVERT: "revert", NO_MUTATION: "no_mutation" } as const;
export type EvolutionDecision = (typeof EvolutionDecision)[keyof typeof EvolutionDecision];

export const MutationDirection = { INCREASE: "increase", DECREASE: "decrease" } as const;
export type MutationDirection = (typeof MutationDirection)[keyof typeof MutationDirection];

export const RecordMode = { LITE: "lite", FULL: "full" } as const;
export type RecordMode = (typeof RecordMode)[keyof typeof RecordMode];

// ── Trial ──
/** A single invocation — the fundamental (s,a,r,s') unit of the RL system. */
export interface Trial {
  id: UUID;
  timestamp: string; // ISO 8601
  context: ContextOrigin;
  taskType: TaskType;
  descriptionHash: Hex16;
  descriptionLength: number;
  capabilityRequirements: string[];
  // Policy action taken
  compressionRatio: number; // 0–1
  compressionWeights: CompressionWeights;
  temperatureSchedule: [number, number, number];
  modelTier: ModelTier;
  retrievalTopK: number;
  tokenBudgetUsed: number;
  maxAttempts: number;
  // Outcomes
  outcome: Outcome;
  qualityScores: [number, number, number, number]; // each 0–1
  qualityScore: number;
  costUsd: number;
  latencyMs: number;
  attemptCount: number;
  bestAttemptIndex: number;
  // RL fields (populated by rl-engine)
  predictedQuality: number | null;
  surprise: number;
  counterfactuals: string[];
  curriculumPhase: number;
  retrievedMemoryIds: UUID[];
  referencedMemoryIds: UUID[];
  advantage: number | null;
  causalUtility: number;
  herGoals: HERGoal[];
  /** Which retrieval strategy generation this trial belongs to (for fitness filtering). */
  generation?: number;
}

export interface CompressionWeights {
  alpha: number; // hypothesis specificity
  beta: number; // code context
  gamma: number; // relevant examples
}

export interface HERGoal {
  goal: string;
  outcome: "success"; // always relabeled as success (Hindsight Experience Replay)
  reward: number; // 0–1
  insight: string;
}

// ── Indexed Memory ──
/** Distilled experience. Thompson-sampled retrieval balances explore/exploit. */
export interface IndexedMemory {
  id: UUID;
  sourceTrialIds: UUID[];
  createdAt: string;
  lastRetrievedAt: string | null;
  retrievalCount: number;
  taskType: TaskType;
  capabilityRequirements: string[];
  hypothesis: string;
  insight: string;
  counterfactuals: string[];
  outcome: Outcome;
  qualityScore: number;
  compressionRatio: number;
  modelTier: ModelTier;
  paramsUsed: MemoryParamsUsed;
  thompsonAlpha: number; // Beta(α,β) successes
  thompsonBeta: number; // failures
  causalUtility: number; // EMA of advantage-weighted quality
  retrievalUtility: number; // Thompson sample cache
  tdError: number;
  surprise: number;
  consolidationCount: number;
  status: MemoryStatus;
  coldSince: string | null;
  expiresAt: string | null;
}

export interface MemoryParamsUsed {
  alpha: number;
  beta: number;
  gamma: number;
  theta1: number;
  theta2: number;
  temperature: [number, number, number];
  tokenBudget: number;
}

// ── Policy State ──
/** The current learned policy. Mutated by the evolution engine. */
export interface PolicyState {
  compression: PolicyCompression;
  quality: PolicyQuality;
  temperature: PolicyTemperature;
  modelTiers: PolicyModelTiers;
  retrieval: PolicyRetrieval;
  exploration: PolicyExploration;
  perType: Partial<Record<TaskType, Partial<PolicyOverrides>>>;
}

export interface PolicyCompression {
  alpha: number;
  beta: number;
  gamma: number;
  theta1: number; // minimum compression threshold
  theta2: number; // aggressive compression threshold
}

export interface PolicyQuality {
  threshold: number; // accept if composite >= this
  maxAttempts: number;
  dimWeights: [number, number, number, number]; // completeness, correctness, consistency, format
}

export interface PolicyTemperature {
  t0: number; // first attempt
  t1: number; // retry 1
  t2: number; // retry 2+
}

export interface PolicyModelTiers {
  lowComplexity: number; // tokens below this → "fast"
  highComplexity: number; // tokens above this → "best"
}

export interface PolicyRetrieval {
  mmrLambda: number;
  topK: number;
  dimWeights: Record<string, number>;
  tokenBudgetTiers: [number, number, number]; // [small, medium, large]
  recencyDecay: number;
  outcomeBonus: number;
  infoDensityBonus: number;
}

export interface PolicyExploration {
  mutationMagnitude: number;
  ucbC: number;
  thompsonPriorStrength: number;
  rndWeight: number;
}

export interface PolicyOverrides {
  compression?: Partial<PolicyCompression>;
  quality?: Partial<PolicyQuality>;
  temperature?: Partial<PolicyTemperature>;
  modelTiers?: Partial<PolicyModelTiers>;
  retrieval?: Partial<PolicyRetrieval>;
  exploration?: Partial<PolicyExploration>;
}

// ── Value Function ──
/** TD(λ) value function tracking expected quality per task type. */
export interface ValueFunctionState {
  baselines: Record<TaskType, TaskTypeBaseline>;
  globalBaseline: number;
  traces: Record<string, number>; // paramName → eligibility trace
  td: TDHyperparams;
  memoryPriorities: Record<UUID, number>;
  maxPriority: number;
}

export interface TaskTypeBaseline {
  mean: number;
  ema: number;
  count: number;
  recentScores: number[]; // ring buffer, max 10
  slope: number; // linear regression of recentScores
  // v6: Branch health metrics (Karpathy BranchTracker-style)
  improvementVelocity: number; // rate of quality improvement per trial
  stabilityScore: number;      // (successes - crashes) / total, range [0,1]
  noveltyScore: number;        // 1 - avg pairwise hypothesis overlap
  plateauConfidence: number;   // 0.0-1.0 signal strength for plateau
  successCount: number;        // number of successful trials
  crashCount: number;          // number of crashed trials
  lastHypotheses: string[];    // last 5 hypotheses for novelty computation
}

export interface TDHyperparams {
  gamma: number; // discount factor (0.90)
  lambda: number; // trace decay (0.70)
  alpha: number; // learning rate (0.10)
  totalUpdates: number;
}

// ── Predictive Model ──
/** Linear logistic model: quality = σ(intercept + Σ w_i · x_i). Trained via SGD. */
export interface PredictiveModelState {
  weights: Record<string, number>;
  intercept: number;
  learningRate: number;
  nUpdates: number;
  featureStats: Record<string, FeatureStat>;
  recentErrors: number[]; // ring buffer, max 20
  calibrationCurve: CalibrationBin[]; // 10 bins
}

export interface FeatureStat {
  mean: number;
  std: number;
  n: number;
  M2: number; // Welford's online variance accumulator
}

export interface CalibrationBin {
  lower: number;
  upper: number;
  predictedMean: number;
  actualMean: number;
  count: number;
}

// ── Curiosity ──
/** RND (Random Network Distillation) exploration bonuses + IDF-weighted retrieval. */
export interface CuriosityState {
  idfCache: IDFCache;
  taskTypeExploration: Record<TaskType, TaskTypeExploration>;
  capabilityCoverage: Record<string, number>;
  surpriseStats: SurpriseStats;
  rnd: RNDState;
}

export interface IDFCache {
  weights: Record<string, number>;
  documentCount: number;
  lastRebuilt: string; // ISO 8601
}

export interface TaskTypeExploration {
  count: number;
  lastExplored: string;
  avgSurprise: number;
  successRate: number;
}

export interface SurpriseStats {
  globalMean: number;
  globalStd: number;
  recentValues: number[]; // ring buffer, max 50
  anomalyThreshold: number; // mean + 2*std
}

export interface RNDState {
  targetProjection: number[][]; // [featureDim × embedDim], fixed random
  predictorWeights: number[][]; // [embedDim × featureDim], learned
  predictorBias: number[]; // [embedDim], learned
  errorMean: number;
  errorStd: number;
}

// ── Retrieval Strategy ──
/** Self-evolving retrieval parameters. Mutated via UCB-guided log-normal proposals. */
export interface RetrievalStrategyState {
  active: ActiveRetrievalStrategy;
  ancestor: ActiveRetrievalStrategy | null;
  ancestorFitness: number;
  pendingMutation: PendingMutation | null;
  trialsInGeneration: number;
  generation: number;
  experienceLibrary: StrategyExperience[];
}

export interface ActiveRetrievalStrategy {
  mmrLambda: number;
  topK: number;
  dimWeights: Record<string, number>;
  tokenBudgetTiers: [number, number, number];
}

export interface PendingMutation {
  targetParam: string;
  oldValue: number;
  newValue: number;
}

export interface StrategyExperience {
  scenario: string; // dominant taskType + phase + trend
  mutation: { param: string; direction: MutationDirection };
  fitnessDelta: number;
  decision: EvolutionDecision;
}

// ── Curriculum ──
/** Phase-adaptive hyperparameter scheduling. Phase boundaries in trial count. */
export interface CurriculumState {
  phaseBoundaries: [number, number, number]; // transitions at these counts
  phases: Record<number, CurriculumPhaseConfig>;
}

export interface CurriculumPhaseConfig {
  name: string;
  learningInterval: number;
  mutationMagnitude: number;
  explorationRate: number;
  surpriseWeight: number;
  consolidationInterval: number;
  // v8: Phase-specific exploration params (autoresearch pattern)
  mmrLambda: number;           // MMR diversity (low=explore, high=exploit)
  curiosityWeight: number;     // curiosity bonus multiplier
  adversarialInterval: number; // how often to run adversarial verification
}

// ── Logs ──
export interface ConsolidationEntry {
  timestamp: string;
  action: ConsolidationAction;
  sourceMemoryIds: UUID[];
  targetMemoryId: UUID | null;
  tokensSaved: number;
  qualityEstimate: number;
  reason: string;
}

export interface EvolutionEntry {
  timestamp: string;
  generation: number;
  mutation: { param: string; oldValue: number; newValue: number };
  fitnessBefore: number;
  fitnessAfter: number;
  delta: number;
  decision: EvolutionDecision;
  scenario: string;
}

// ── Cross-Context Bridge ──
/** Buffers trials from Context A (skill) → Context B (agent) for full RL processing. */
export interface CrossContextBuffer {
  pendingTrialsFromSkill: PendingTrialsQueue;
  refinedInsights: RefinedInsights;
  canonicalStrategies: CanonicalStrategy[];
}

export interface PendingTrialsQueue {
  trials: Trial[];
  oldestPending: string;
  count: number;
}

export interface RefinedInsights {
  updatedMemoryUtils: Record<UUID, number>;
  discoveredPatterns: string[];
  recommendedPolicyDiffs: Partial<PolicyState>;
  lastSyncTimestamp: string;
  agentIterationsProcessed: number;
}

export interface CanonicalStrategy {
  strategyId: UUID;
  taskType: TaskType;
  pattern: string;
  params: Partial<PolicyCompression & PolicyRetrieval>;
  successRate: number;
  trialCount: number;
  discoveredBy: ContextOrigin;
  discoveredAt: string;
}

// ── Root State ──
/** The complete v5 state. Serialized to disk as state-v5.json. */
/** v5.1: Ablation result stored in V5 state for causal graph construction */
export interface AblationEntry {
  sourceFile: string;
  taskType: TaskType;
  qualityWith: number;
  qualityWithout: number;
  causalDelta: number;
  confidence: number;
  withCompiled?: boolean;
  withoutCompiled?: boolean;
  durationMs: number;
  timestamp: string;
}

export interface SharedStateV5 {
  version: 5;
  createdAt: string;
  lastUpdated: string;
  totalInvocations: number;
  trials: Trial[];
  memories: IndexedMemory[];
  coldStorage: IndexedMemory[];
  /** v5.1: Ablation results for causal graph construction */
  ablationResults: AblationEntry[];
  policy: PolicyState;
  valueFunction: ValueFunctionState;
  predictiveModel: PredictiveModelState;
  curiosity: CuriosityState;
  retrievalStrategy: RetrievalStrategyState;
  curriculum: CurriculumState;
  consolidationLog: ConsolidationEntry[];
  crossContextBuffer: CrossContextBuffer;
  evolutionLog: EvolutionEntry[];
}

// ── Validation ──
export interface ValidationError {
  path: string; // JSON path, e.g. "trials[3].qualityScores"
  code: string; // machine-readable error code
  message: string; // human-readable description
}

// ── Return types for RL engine operations ──
export interface QueryResult {
  compressionWeights: CompressionWeights;
  temperatureSchedule: [number, number, number];
  modelTier: ModelTier;
  retrievalParams: { mmrLambda: number; topK: number };
  tokenBudget: number;
  maxAttempts: number;
  qualityThreshold: number;
  retrievedMemories: IndexedMemory[];
  contrastiveInsights: string[];
  curriculumPhase: number;
  explorationBonus: number;
}

export interface RecordResult {
  surprise: number;
  tdError: number;
  counterfactuals: string[];
  herGoals: HERGoal[];
  memoriesUpdated: number;
  pendingSyncCount: number;
}

export interface EvolutionResult {
  generation: number;
  mutation: { param: string; oldValue: number; newValue: number } | null;
  fitnessDelta: number | null;
  decision: EvolutionDecision;
}

export interface ConsolidationResult {
  consolidatedCount: number;
  archivedCount: number;
  tokensFreed: number;
}

export interface VerificationResult {
  verifiedCount: number;
  staleCount: number;
  overturnedCount: number;
}

export interface SyncResult {
  trialsProcessed: number;
  insightsGenerated: number;
  policyDiffsApplied: number;
}

export interface StatusReport {
  totalTrials: number;
  activeMemories: number;
  coldMemories: number;
  curriculumPhase: number;
  perTaskType: Record<TaskType, {
    trialCount: number;
    avgQuality: number;
    baselineQuality: number;
    isPlateaued: boolean;
    improvementSlope: number;
  }>;
  predictiveModelAccuracy: number;
  surpriseMean: number;
  lastEvolution: string;
  lastConsolidation: string;
  lastCrossSync: string;
}

/** Trial construction input — everything except RL-computed fields. */
export interface TrialInput {
  context: ContextOrigin;
  taskType: TaskType;
  description: string;
  capabilityRequirements: string[];
  compressionRatio: number;
  compressionWeights: CompressionWeights;
  temperatureSchedule: [number, number, number];
  modelTier: ModelTier;
  retrievalTopK: number;
  tokenBudgetUsed: number;
  maxAttempts: number;
  outcome: Outcome;
  qualityScores: [number, number, number, number];
  qualityScore: number;
  costUsd: number;
  latencyMs: number;
  attemptCount: number;
  bestAttemptIndex: number;
  retrievedMemoryIds: UUID[];
  referencedMemoryIds: UUID[];
}

/** Retrieval pipeline input/output */
export interface RetrievalInput {
  taskType: TaskType;
  description: string;
  capabilityRequirements: string[];
}

export interface RetrievalOutput {
  memories: IndexedMemory[];
  contrastiveInsights: string[];
  scoringDebug?: RetrievalDebug;
}

export interface RetrievalDebug {
  perDimensionScores: Record<UUID, Record<string, number>>;
  postCausalScores: Record<UUID, number>;
  mmrSelectionOrder: UUID[];
}
