// ============================================================================
// Turbocontext v5 — State Module Barrel Export
// ============================================================================
// All public types and classes re-exported from a single entry point.
// Importers use: import { SharedStateManager, type SharedStateV5, ... } from "./state.js";
export { SharedStateManager } from "./state-manager.js";

// Types
export type {
  SharedStateV5, Trial, IndexedMemory, PolicyState,
  ValueFunctionState, PredictiveModelState, CuriosityState,
  RetrievalStrategyState, CurriculumState, CrossContextBuffer,
  CompressionWeights, HERGoal, MemoryParamsUsed,
  PolicyCompression, PolicyQuality, PolicyTemperature,
  PolicyModelTiers, PolicyRetrieval, PolicyExploration, PolicyOverrides,
  TaskTypeBaseline, TDHyperparams, FeatureStat, CalibrationBin,
  IDFCache, TaskTypeExploration, SurpriseStats, RNDState,
  ActiveRetrievalStrategy, PendingMutation, StrategyExperience,
  CurriculumPhaseConfig, ConsolidationEntry, EvolutionEntry,
  CanonicalStrategy, PendingTrialsQueue, RefinedInsights,
  ValidationError, QueryResult, RecordResult, EvolutionResult,
  ConsolidationResult, VerificationResult, SyncResult, StatusReport,
  TrialInput, RetrievalInput, RetrievalOutput, RetrievalDebug,
} from "./types.js";

export {
  TaskType, ModelTier, Outcome, ContextOrigin,
  MemoryStatus, CurriculumPhase, ConsolidationAction,
  EvolutionDecision, MutationDirection, RecordMode,
} from "./types.js";

// Constants
export {
  STATE_DIR, STATE_PATH, STATE_BACKUP_PATH,
  TRIALS_LOG_PATH, EVOLUTION_LOG_PATH, CONSOLIDATION_LOG_PATH,
  COLD_STORAGE_PATH,
  MAX_ACTIVE_TRIALS, MAX_ACTIVE_MEMORIES, MAX_COLD_MEMORIES,
  MAX_TRIAL_DESCRIPTION_LENGTH, MAX_RETRIEVED_MEMORIES,
  MAX_STATE_FILE_SIZE_BYTES, PER_BUFFER_CAPACITY, PER_BATCH_SIZE,
  EPSILON, MIN_TRACE, CLAMP_ADVANTAGE, SIGMOID_CLAMP,
  FITNESS_KEEP_THRESHOLD, FITNESS_REVERT_THRESHOLD,
  DEGRADATION_THRESHOLD, CANONICAL_TRIAL_THRESHOLD,
  CANONICAL_SUCCESS_RATE, COLD_STORAGE_DAYS,
  IDF_REBUILD_INTERVAL, PER_ALPHA, PER_BETA, PER_BETA_INCREMENT,
  DEFAULT_FEATURE_DIM, RND_EMBED_DIM,
  DEFAULT_TD, DEFAULT_PREDICTIVE_MODEL_LR,
  FEATURE_NAMES, TUNABLE_PARAMS, PARAM_BOUNDS,
  DEFAULT_POLICY, DEFAULT_CURRICULUM, DIM_NAMES,
  createFreshState, initRND,
} from "./constants.js";

// Validation
export { validateState, migrateV4ToV5 } from "./validation.js";

// I/O
export {
  loadState, saveState,
  appendTrialLog, appendEvolutionLog, appendConsolidationLog,
  loadColdStorage, saveColdStorage,
  StateWriteError, StateSizeError,
} from "./io.js";

// Periodic Scheduler
export { PeriodicScheduler, PeriodicOp } from "./periodic-scheduler.js";
import type { PeriodicOp as _PeriodicOp } from "./periodic-scheduler.js";
export type { _PeriodicOp as PeriodicOpType };

// ── V5 New Modules (June 2026) ──

// Policy Manager
export {
  resolveEffectivePolicy,
  applyMutation,
  getParamValue,
  clonePolicy,
  normalizeDimWeights,
} from "./policy/policy-manager.js";

// RL: Thompson Sampling
export { sampleBeta, sampleGamma, gaussianRandom } from "./rl/thompson.js";

// RL: RND (Random Network Distillation)
export {
  ensureRNDInit,
  computeRNDEmbedding,
  computeRNDBonus,
  trainRNDPredictor,
} from "./rl/rnd.js";

// RL: Retrieval (7-dim MMR)
export {
  retrieveMemories,
  scoreMemory,
  computeIDFOverlap,
  computeCapabilityJaccard,
  computeTaskTypeMatch,
  computeInfoDensity,
  mmrReRank,
  computeMemorySimilarity,
} from "./rl/retrieval.js";
export type { ScoredMemory, RetrievalQuery } from "./rl/retrieval.js";

// RL Engine (main orchestrator)
export { RLEngineV5 } from "./rl/rl-engine.js";
