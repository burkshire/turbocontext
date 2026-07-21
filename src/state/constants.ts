// ============================================================================
// Turbocontext v5 — Constants & Defaults
// ============================================================================
// All magic numbers, limits, and default policies live here.
// RL theory: defaults encode the initial exploration strategy — high
// mutation magnitude + broad MMR lambda for curriculum phase 0.
import * as os from "node:os";
import type { PolicyState, CurriculumState, ValueFunctionState, PredictiveModelState, CuriosityState, RetrievalStrategyState, SharedStateV5 } from "./types.js";
import { TaskType } from "./types.js";

// ── File paths ──
export const STATE_DIR = `${os.homedir()}/.turbocontext`;
export const STATE_PATH = `${STATE_DIR}/state-v5.json`;
export const STATE_BACKUP_PATH = `${STATE_DIR}/state-v5.backup.json`;
export const TRIALS_LOG_PATH = `${STATE_DIR}/logs/trials.jsonl`;
export const EVOLUTION_LOG_PATH = `${STATE_DIR}/logs/evolution.jsonl`;
export const CONSOLIDATION_LOG_PATH = `${STATE_DIR}/logs/consolidation.jsonl`;
export const COLD_STORAGE_PATH = `${STATE_DIR}/cold-storage/archived-memories.json`;

// ── Size limits ──
export const MAX_ACTIVE_TRIALS = 10_000;
export const MAX_ACTIVE_MEMORIES = 200;
export const MAX_COLD_MEMORIES = 1_000;
export const MAX_TRIAL_DESCRIPTION_LENGTH = 50_000;
export const MAX_RETRIEVED_MEMORIES = 20;
export const MAX_RECENT_SCORES = 10;
export const MAX_RECENT_SURPRISE_VALUES = 50;
export const MAX_RECENT_ERRORS = 20;
export const PER_BUFFER_CAPACITY = 500;
export const PER_BATCH_SIZE = 32;
export const MAX_STATE_FILE_SIZE_BYTES = 2_000_000; // 2 MB

// ── Numerical guardrails ──
export const EPSILON = 1e-8;
export const MIN_TRACE = 0.001;
export const CLAMP_ADVANTAGE: [number, number] = [-0.5, 0.5];
export const SIGMOID_CLAMP: [number, number] = [-10, 10];
export const FITNESS_KEEP_THRESHOLD = 0.03;
export const FITNESS_REVERT_THRESHOLD = -0.05;
export const DEGRADATION_THRESHOLD = 0.30;
export const CANONICAL_TRIAL_THRESHOLD = 5;
export const CANONICAL_SUCCESS_RATE = 0.80;
export const COLD_STORAGE_DAYS = 30;
export const COLD_STORAGE_RETRIEVAL_COUNT = 50;
export const IDF_REBUILD_INTERVAL = 50;
export const PER_ALPHA = 0.6;
export const PER_BETA = 0.4;
export const PER_BETA_INCREMENT = 0.001;
export const DEFAULT_FEATURE_DIM = 13;
export const RND_EMBED_DIM = 32;

// ── RL hyperparameter defaults ──
export const DEFAULT_TD = { gamma: 0.90, lambda: 0.70, alpha: 0.10 };
export const DEFAULT_PREDICTIVE_MODEL_LR = 0.01;
/** EMA blending factor for online metric updates (0.3 = 30% new, 70% old). */
export const EMA_BLENDING_FACTOR = 0.3;

/** 13 feature names for the linear predictive model. */
export const FEATURE_NAMES: string[] = [
  "task_code_review", "task_code_generation", "task_debugging",
  "task_refactoring", "task_documentation", "task_architecture",
  "log_description_length", "compression_ratio",
  "model_tier_fast", "model_tier_best",
  "is_retry", "hour_of_day_sin", "log_token_budget",
];

// ── Tunable parameter paths (dot-notation, for UCB selector) ──
export const TUNABLE_PARAMS: string[] = [
  "compression.alpha", "compression.beta", "compression.gamma",
  "compression.theta1", "compression.theta2",
  "quality.threshold", "quality.maxAttempts",
  "temperature.t0", "temperature.t1", "temperature.t2",
  "modelTiers.lowComplexity", "modelTiers.highComplexity",
  "retrieval.mmrLambda", "retrieval.topK",
  "retrieval.recencyDecay", "retrieval.outcomeBonus", "retrieval.infoDensityBonus",
  "exploration.mutationMagnitude", "exploration.ucbC",
  "exploration.thompsonPriorStrength", "exploration.rndWeight",
];

/** Parameter-specific bounds — mutations clamp to these. */
export const PARAM_BOUNDS: Record<string, [number, number]> = {
  "compression.alpha":                 [0.10, 0.95],
  "compression.beta":                  [0.10, 0.95],
  "compression.gamma":                 [0.10, 0.95],
  "compression.theta1":                [0.05, 0.80],
  "compression.theta2":                [0.10, 0.90],
  "quality.threshold":                 [0.50, 0.95],
  "quality.maxAttempts":               [1,     5],
  "temperature.t0":                    [0.0,   1.0],
  "temperature.t1":                    [0.0,   1.0],
  "temperature.t2":                    [0.0,   1.5],
  "modelTiers.lowComplexity":          [500,   5000],
  "modelTiers.highComplexity":         [2000,  20000],
  "retrieval.mmrLambda":               [0.10,  0.95],
  "retrieval.topK":                    [1,     20],
  "retrieval.recencyDecay":            [0.01,  0.50],
  "retrieval.outcomeBonus":            [0.0,   0.50],
  "retrieval.infoDensityBonus":        [0.0,   0.30],
  "exploration.mutationMagnitude":     [0.01,  0.50],
  "exploration.ucbC":                  [0.5,   5.0],
  "exploration.thompsonPriorStrength": [0.5,   10.0],
  "exploration.rndWeight":             [0.0,   0.50],
};

// ── Default policy ──
/** Initial policy: moderate compression, broad exploration, small retrieval. */
export const DEFAULT_POLICY: PolicyState = {
  compression: { alpha: 0.60, beta: 0.20, gamma: 0.20, theta1: 0.30, theta2: 0.55 },
  quality: { threshold: 0.75, maxAttempts: 3, dimWeights: [0.25, 0.35, 0.20, 0.20] },
  quality: { threshold: 0.75, maxAttempts: 3, dimWeights: [0.25, 0.35, 0.20, 0.20] },
  temperature: { t0: 0.70, t1: 0.35, t2: 0.10 },
  modelTiers: { lowComplexity: 1500, highComplexity: 8000 },
  retrieval: {
    mmrLambda: 0.70, topK: 5,
    dimWeights: {
      idfOverlap: 0.25, capabilityJaccard: 0.20, taskTypeMatch: 0.10,
      recencyDecay: 0.15, outcomeBonus: 0.10, infoDensity: 0.10, thompsonUtility: 0.10,
    },
    tokenBudgetTiers: [8000, 16000, 32000],
    recencyDecay: 0.05, outcomeBonus: 0.15, infoDensityBonus: 0.10,
  },
  exploration: { mutationMagnitude: 0.15, ucbC: 2.0, thompsonPriorStrength: 2.0, rndWeight: 0.10 },
  perType: {},
};

// ── Default curriculum ──
// v8: Phase-specific exploration parameters (autoresearch pattern)
// Each phase adjusts MMR lambda, curiosity weight, and adversarial verification
// cadence in addition to the existing mutation/exploration/surprise params.
export const DEFAULT_CURRICULUM: CurriculumState = {
  phaseBoundaries: [10, 30, 60],
  phases: {
    0: { name: "broad_exploration", learningInterval: 3, mutationMagnitude: 0.25, explorationRate: 0.8, surpriseWeight: 1.2, consolidationInterval: 20, mmrLambda: 0.35, curiosityWeight: 1.5, adversarialInterval: 20 },
    1: { name: "focused_exploitation", learningInterval: 5, mutationMagnitude: 0.15, explorationRate: 0.5, surpriseWeight: 1.0, consolidationInterval: 15, mmrLambda: 0.55, curiosityWeight: 1.0, adversarialInterval: 15 },
    2: { name: "principled_optimization", learningInterval: 8, mutationMagnitude: 0.08, explorationRate: 0.2, surpriseWeight: 0.8, consolidationInterval: 10, mmrLambda: 0.70, curiosityWeight: 0.5, adversarialInterval: 10 },
    3: { name: "adversarial_refinement", learningInterval: 10, mutationMagnitude: 0.06, explorationRate: 0.4, surpriseWeight: 1.0, consolidationInterval: 8, mmrLambda: 0.60, curiosityWeight: 0.7, adversarialInterval: 8 },
  },
};

// ── Cold-start factory ──
/**
 * createFreshState: initializes a zero-experience SharedStateV5.
 *
 * Every subsystem begins with neutral priors:
 *   - Predictive model: zero weights, unit-variance feature stats
 *   - Value function: zero-EMA baselines, empty traces
 *   - Curiosity: empty IDF cache, freshly initialized RND
 *   - Retrieval strategy: default policy values, no ancestor
 *   - Curriculum: phase 0 (broad exploration)
 */
export function createFreshState(): SharedStateV5 {
  const now = new Date().toISOString();
  const taskTypes = Object.values(TaskType);
  const emptyBaseline = () => ({
    mean: 0, ema: 0, count: 0, recentScores: [] as number[], slope: 0,
    // v6: Branch health metrics (Karpathy BranchTracker-style)
    improvementVelocity: 0, stabilityScore: 0.5, noveltyScore: 0.5,
    plateauConfidence: 0, successCount: 0, crashCount: 0,
    lastHypotheses: [] as string[],
  });
  const emptyFeatureStat = () => ({ mean: 0, std: 1, n: 0, M2: 0 });
  const emptyExploration = () => ({ count: 0, lastExplored: now, avgSurprise: 0, successRate: 0 });

  return {
    version: 5, createdAt: now, lastUpdated: now, totalInvocations: 0,
    trials: [], memories: [], coldStorage: [],
    policy: structuredClone(DEFAULT_POLICY),
    valueFunction: {
      baselines: Object.fromEntries(taskTypes.map(t => [t, emptyBaseline()])) as any,
      globalBaseline: 0, traces: {},
      td: { ...DEFAULT_TD, totalUpdates: 0 },
      memoryPriorities: {}, maxPriority: 1.0,
    },
    predictiveModel: {
      weights: Object.fromEntries(FEATURE_NAMES.map(f => [f, 0])),
      intercept: 0, learningRate: DEFAULT_PREDICTIVE_MODEL_LR, nUpdates: 0,
      featureStats: Object.fromEntries(FEATURE_NAMES.map(f => [f, emptyFeatureStat()])),
      recentErrors: [],
      calibrationCurve: Array.from({ length: 10 }, (_, i) => ({
        lower: i / 10, upper: (i + 1) / 10,
        predictedMean: 0, actualMean: 0, count: 0,
      })),
    },
    curiosity: {
      idfCache: { weights: {}, documentCount: 0, lastRebuilt: now },
      taskTypeExploration: Object.fromEntries(taskTypes.map(t => [t, emptyExploration()])) as any,
      capabilityCoverage: {},
      surpriseStats: { globalMean: 0, globalStd: 0, recentValues: [], anomalyThreshold: 0 },
      rnd: initRND(),
    },
    retrievalStrategy: {
      active: {
        mmrLambda: DEFAULT_POLICY.retrieval.mmrLambda,
        topK: DEFAULT_POLICY.retrieval.topK,
        dimWeights: { ...DEFAULT_POLICY.retrieval.dimWeights },
        tokenBudgetTiers: [...DEFAULT_POLICY.retrieval.tokenBudgetTiers],
      },
      ancestor: null, ancestorFitness: 0, pendingMutation: null,
      trialsInGeneration: 0, generation: 0, experienceLibrary: [],
    },
    curriculum: structuredClone(DEFAULT_CURRICULUM),
    consolidationLog: [],
    crossContextBuffer: {
      pendingTrialsFromSkill: { trials: [], oldestPending: now, count: 0 },
      refinedInsights: {
        updatedMemoryUtils: {}, discoveredPatterns: [],
        recommendedPolicyDiffs: {}, lastSyncTimestamp: now,
        agentIterationsProcessed: 0,
      },
      canonicalStrategies: [],
    },
    evolutionLog: [],
    // v5.1: Ablation results for causal graph construction
    ablationResults: [],
  };
}

/**
 * initRND: creates a fresh Random Network Distillation state.
 *
 * RND theory (Burda et al. 2019): exploration bonus ∝ MSE between
 * a fixed random target network and a learned predictor network.
 * High MSE → novel state → high exploration bonus.
 * Matrix dimensions: [DEFAULT_FEATURE_DIM × RND_EMBED_DIM].
 * Weights drawn from N(0, 1).
 */
export function initRND(featureDim = DEFAULT_FEATURE_DIM, embedDim = RND_EMBED_DIM): import("./types").RNDState {
  const targetProjection: number[][] = Array.from({ length: featureDim }, () =>
    Array.from({ length: embedDim }, () => gaussianRandom())
  );
  return {
    targetProjection,
    predictorWeights: Array.from({ length: embedDim }, () =>
      Array.from({ length: featureDim }, () => 0)
    ),
    predictorBias: Array(embedDim).fill(0),
    errorMean: 0,
    errorStd: 1,
  };
}

/** Box-Muller transform for N(0,1) random values. */
function gaussianRandom(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ── Quality dimension names (for human-readable HER insights) ──
export const DIM_NAMES: Record<number, string> = {
  0: "completeness", 1: "correctness", 2: "consistency", 3: "format",
};
