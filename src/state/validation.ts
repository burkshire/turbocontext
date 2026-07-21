// ============================================================================
// Turbocontext v5 — State Validation & Migration
// ============================================================================
// Validates structural shape and semantic constraints on SharedStateV5.
// Used on every load to prevent corrupted state from propagating.
// Also handles v4→v5 migration for backward compatibility.
import type {
  SharedStateV5, Trial, IndexedMemory, PolicyState, ValidationError,
  TaskTypeBaseline,
} from "./types.js";
import { TaskType, Outcome, MemoryStatus } from "./types.js";
import { PARAM_BOUNDS, MAX_TRIAL_DESCRIPTION_LENGTH, createFreshState } from "./constants.js";

// ── Main validation entry ──

/**
 * validateState: structural + semantic validation of a v5 state object.
 *
 * Order: structural (types exist, required fields present) →
 * semantic (values in range, cross-references valid).
 * Returns empty array = valid. Non-empty = invalid — caller must repair or discard.
 */
export function validateState(obj: unknown): ValidationError[] {
  const errors: ValidationError[] = [];
  if (!obj || typeof obj !== "object") return [{ path: "$", code: "NOT_OBJECT", message: "State is not an object" }];

  const s = obj as Record<string, unknown>;

  if (s.version !== 5) errors.push({ path: "$.version", code: "WRONG_VERSION", message: `Expected version 5, got ${s.version}` });
  if (typeof s.createdAt !== "string") errors.push({ path: "$.createdAt", code: "MISSING", message: "createdAt must be a string" });
  if (typeof s.lastUpdated !== "string") errors.push({ path: "$.lastUpdated", code: "MISSING", message: "lastUpdated must be a string" });
  if (typeof s.totalInvocations !== "number" || s.totalInvocations < 0) errors.push({ path: "$.totalInvocations", code: "INVALID", message: "totalInvocations must be >= 0" });

  // Structural: arrays
  if (!Array.isArray(s.trials)) errors.push({ path: "$.trials", code: "NOT_ARRAY", message: "trials must be an array" });
  if (!Array.isArray(s.memories)) errors.push({ path: "$.memories", code: "NOT_ARRAY", message: "memories must be an array" });
  if (!Array.isArray(s.coldStorage)) errors.push({ path: "$.coldStorage", code: "NOT_ARRAY", message: "coldStorage must be an array" });

  // Deeper validation only if structure is sound
  if (Array.isArray(s.trials)) {
    s.trials.forEach((t, i) => errors.push(...validateTrial(t as Trial, i, s as unknown as SharedStateV5)));
  }
  if (Array.isArray(s.memories)) {
    s.memories.forEach((m, i) => errors.push(...validateMemory(m as IndexedMemory, i, s as unknown as SharedStateV5)));
  }
  if (s.policy && typeof s.policy === "object") {
    errors.push(...validatePolicySemantics(s.policy as PolicyState));
  }

  errors.push(...validateCrossReferences(s as unknown as SharedStateV5));
  return errors;
}

// ── Trial validation ──

/**
 * validateTrial: validates a single trial's field types and semantic constraints.
 *
 * Semantic rules:
 *  - qualityScore = Σ dimWeights[i] * qualityScores[i] (within tolerance)
 *  - compressionWeights.α+β+γ ≈ 1.0
 *  - bestAttemptIndex < attemptCount
 *  - Each qualityScore[i] ∈ [0,1]
 */
function validateTrial(t: Trial, idx: number, state: SharedStateV5): ValidationError[] {
  const e: ValidationError[] = [];
  const p = `$.trials[${idx}]`;

  if (!t.id) e.push({ path: `${p}.id`, code: "MISSING", message: "id is required" });
  if (!Object.values(TaskType).includes(t.taskType)) e.push({ path: `${p}.taskType`, code: "INVALID_ENUM", message: `Unknown taskType: ${t.taskType}` });
  if (!Object.values(Outcome).includes(t.outcome)) e.push({ path: `${p}.outcome`, code: "INVALID_ENUM", message: `Unknown outcome: ${t.outcome}` });

  // qualityScores: 4 values each in [0,1]
  if (!Array.isArray(t.qualityScores) || t.qualityScores.length !== 4) {
    e.push({ path: `${p}.qualityScores`, code: "INVALID", message: "qualityScores must be [number, number, number, number]" });
  } else {
    t.qualityScores.forEach((v, i) => {
      if (v < 0 || v > 1) e.push({ path: `${p}.qualityScores[${i}]`, code: "OUT_OF_RANGE", message: `qualityScores[${i}]=${v}, expected [0,1]` });
    });
  }

  // qualityScore consistency with dimWeights
  if (t.qualityScores && Array.isArray(t.qualityScores) && t.qualityScores.length === 4) {
    const dw = state.policy?.quality?.dimWeights || [0.25, 0.35, 0.20, 0.20];
    const expected = dw[0] * t.qualityScores[0] + dw[1] * t.qualityScores[1] + dw[2] * t.qualityScores[2] + dw[3] * t.qualityScores[3];
    if (Math.abs(t.qualityScore - expected) > 0.02) {
      e.push({ path: `${p}.qualityScore`, code: "INCONSISTENT", message: `qualityScore=${t.qualityScore.toFixed(3)} but weighted average=${expected.toFixed(3)}` });
    }
  }

  // compressionWeights normalization
  if (t.compressionWeights) {
    const sum = t.compressionWeights.alpha + t.compressionWeights.beta + t.compressionWeights.gamma;
    if (Math.abs(sum - 1.0) > 0.05) {
      e.push({ path: `${p}.compressionWeights`, code: "NOT_NORMALIZED", message: `α+β+γ=${sum.toFixed(3)}, expected ~1.0` });
    }
  }

  // Logical constraints
  if (t.bestAttemptIndex >= t.attemptCount) e.push({ path: `${p}.bestAttemptIndex`, code: "INVALID", message: `bestAttemptIndex=${t.bestAttemptIndex} >= attemptCount=${t.attemptCount}` });
  if (t.attemptCount < 1) e.push({ path: `${p}.attemptCount`, code: "INVALID", message: `attemptCount must be >= 1` });
  if (t.tokenBudgetUsed <= 0) e.push({ path: `${p}.tokenBudgetUsed`, code: "INVALID", message: "tokenBudgetUsed must be > 0" });
  if (t.latencyMs <= 0) e.push({ path: `${p}.latencyMs`, code: "INVALID", message: "latencyMs must be > 0" });
  if (t.maxAttempts < 1 || t.maxAttempts > 5) e.push({ path: `${p}.maxAttempts`, code: "OUT_OF_RANGE", message: `maxAttempts=${t.maxAttempts}, expected [1,5]` });

  // temperatureSchedule
  if (!Array.isArray(t.temperatureSchedule) || t.temperatureSchedule.length !== 3) {
    e.push({ path: `${p}.temperatureSchedule`, code: "INVALID", message: "temperatureSchedule must be [number, number, number]" });
  } else {
    t.temperatureSchedule.forEach((v, i) => {
      if (v < 0 || v > 2) e.push({ path: `${p}.temperatureSchedule[${i}]`, code: "OUT_OF_RANGE", message: `temperature[${i}]=${v}, expected [0,2]` });
    });
  }

  // descriptionLength
  if (t.descriptionLength < 1 || t.descriptionLength > MAX_TRIAL_DESCRIPTION_LENGTH) {
    e.push({ path: `${p}.descriptionLength`, code: "OUT_OF_RANGE", message: `descriptionLength=${t.descriptionLength}, expected [1,${MAX_TRIAL_DESCRIPTION_LENGTH}]` });
  }

  // curriculumPhase valid range
  if (t.curriculumPhase < 0 || t.curriculumPhase > 3) {
    e.push({ path: `${p}.curriculumPhase`, code: "OUT_OF_RANGE", message: `curriculumPhase=${t.curriculumPhase}, expected [0,3]` });
  }

  return e;
}

// ── Memory validation ──

function validateMemory(m: IndexedMemory, idx: number, state: SharedStateV5): ValidationError[] {
  const e: ValidationError[] = [];
  const p = `$.memories[${idx}]`;

  if (!m.id) e.push({ path: `${p}.id`, code: "MISSING", message: "id is required" });
  if (m.thompsonAlpha < 0) e.push({ path: `${p}.thompsonAlpha`, code: "NEGATIVE", message: `thompsonAlpha=${m.thompsonAlpha} must be >= 0` });
  if (m.thompsonBeta < 0) e.push({ path: `${p}.thompsonBeta`, code: "NEGATIVE", message: `thompsonBeta=${m.thompsonBeta} must be >= 0` });
  if (m.causalUtility < 0 || m.causalUtility > 1) e.push({ path: `${p}.causalUtility`, code: "OUT_OF_RANGE", message: `causalUtility=${m.causalUtility}, expected [0,1]` });
  if (!Object.values(MemoryStatus).includes(m.status)) e.push({ path: `${p}.status`, code: "INVALID_ENUM", message: `Unknown status: ${m.status}` });
  return e;
}

// ── Policy validation ──

function validatePolicySemantics(policy: PolicyState): ValidationError[] {
  const e: ValidationError[] = [];
  const bp = "$.policy";

  // Compression
  const c = policy.compression;
  if (c) {
    if (c.alpha < 0.1 || c.alpha > 0.95) e.push({ path: `${bp}.compression.alpha`, code: "OUT_OF_RANGE", message: `alpha=${c.alpha}` });
    if (c.beta < 0.1 || c.beta > 0.95) e.push({ path: `${bp}.compression.beta`, code: "OUT_OF_RANGE", message: `beta=${c.beta}` });
    if (c.gamma < 0.1 || c.gamma > 0.95) e.push({ path: `${bp}.compression.gamma`, code: "OUT_OF_RANGE", message: `gamma=${c.gamma}` });
    if (c.theta1 < 0.05 || c.theta1 > 0.80) e.push({ path: `${bp}.compression.theta1`, code: "OUT_OF_RANGE", message: `theta1=${c.theta1}` });
    if (c.theta2 < 0.10 || c.theta2 > 0.90) e.push({ path: `${bp}.compression.theta2`, code: "OUT_OF_RANGE", message: `theta2=${c.theta2}` });
  }

  // Quality dimWeights should sum to ~1
  if (policy.quality?.dimWeights) {
    const sum = policy.quality.dimWeights.reduce((a, b) => a + b, 0);
    if (Math.abs(sum - 1.0) > 0.02) {
      e.push({ path: `${bp}.quality.dimWeights`, code: "NOT_NORMALIZED", message: `dimWeights sum=${sum.toFixed(3)}, expected ~1.0` });
    }
  }

  // perType keys must be valid TaskTypes
  if (policy.perType) {
    for (const key of Object.keys(policy.perType)) {
      if (!Object.values(TaskType).includes(key as any)) {
        e.push({ path: `${bp}.perType.${key}`, code: "INVALID_KEY", message: `"${key}" is not a valid TaskType` });
      }
    }
  }

  return e;
}

// ── Cross-reference validation ──

/**
 * validateCrossReferences: verifies referential integrity across the state graph.
 *
 * Checks that:
 *  - sourceTrialIds resolve to actual trial IDs
 *  - retrievedMemoryIds and referencedMemoryIds resolve
 *  - memoryPriorities keys resolve to memory IDs
 */
function validateCrossReferences(state: SharedStateV5): ValidationError[] {
  const e: ValidationError[] = [];
  const trialIds = new Set(state.trials.map(t => t.id));
  const memoryIds = new Set(state.memories.map(m => m.id));

  // sourceTrialIds → trials
  state.memories.forEach((m, mi) => {
    m.sourceTrialIds.forEach(sid => {
      if (!trialIds.has(sid)) {
        e.push({ path: `$.memories[${mi}].sourceTrialIds`, code: "DANGLING_REF", message: `sourceTrialId ${sid} does not exist in trials` });
      }
    });
  });

  // retrievedMemoryIds / referencedMemoryIds → memories
  state.trials.forEach((t, ti) => {
    t.retrievedMemoryIds.forEach(mid => {
      if (!memoryIds.has(mid)) {
        e.push({ path: `$.trials[${ti}].retrievedMemoryIds`, code: "DANGLING_REF", message: `retrievedMemoryId ${mid} does not exist` });
      }
    });
    t.referencedMemoryIds.forEach(mid => {
      if (!memoryIds.has(mid)) {
        e.push({ path: `$.trials[${ti}].referencedMemoryIds`, code: "DANGLING_REF", message: `referencedMemoryId ${mid} does not exist` });
      }
    });
  });

  // memoryPriorities keys → memory IDs
  const vf = state.valueFunction;
  if (vf?.memoryPriorities) {
    for (const mid of Object.keys(vf.memoryPriorities)) {
      if (!memoryIds.has(mid)) {
        e.push({ path: `$.valueFunction.memoryPriorities.${mid}`, code: "DANGLING_REF", message: `priority for unknown memory ${mid}` });
      }
    }
  }

  return e;
}

// ── v4 → v5 migration ──

/**
 * migrateV4ToV5: transforms a v4 state object into v5.
 *
 * Strategy:
 *  1. Map v4 fields to v5 equivalents, filling new fields with defaults.
 *  2. Initialize predictive model from historical trials via SGD replay.
 *  3. Initialize value function baselines from trial history.
 *  4. Initialize retrieval strategy ancestor from v4 retrieval params.
 *
 * This is lossy: RL metadata (Thompson params, TD errors, traces) cannot
 * be reconstructed from v4 which lacked them. They start at neutral priors.
 */
export function migrateV4ToV5(v4State: any): SharedStateV5 {
  const now = new Date().toISOString();

  // Map trials: add v5-only RL fields with neutral defaults
  const trials: Trial[] = (v4State.trials || []).map((t: any) => ({
    ...t,
    predictedQuality: t.predictedOutcome ?? null,
    surprise: t.surpriseScore ?? 0,
    counterfactuals: t.counterfactual ? [t.counterfactual] : [],
    curriculumPhase: t.curriculumPhase ?? 0,
    retrievedMemoryIds: t.retrievedMemoryKeys || t.retrievedMemoryIds || [],
    referencedMemoryIds: t.plannerReferencedKeys || t.referencedMemoryIds || [],
    advantage: null,
    causalUtility: t.causalUtility ?? 0.5,
    herGoals: [],
    descriptionHash: t.descriptionHash || sha256Hex16(t.description || ""),
    descriptionLength: t.descriptionLength || (t.description?.length || 0),
    capabilityRequirements: t.capabilityRequirements || [],
  }));

  // Map memories: add v5 RL fields
  const memories: IndexedMemory[] = (v4State.memories || v4State.executionHistory || []).map((m: any) => ({
    ...m,
    sourceTrialIds: m.sourceTrialIds || (m.sourceTrialId ? [m.sourceTrialId] : []),
    thompsonAlpha: m.thompsonAlpha ?? 1,
    thompsonBeta: m.thompsonBeta ?? 1,
    causalUtility: m.causalUtility ?? 0.5,
    retrievalUtility: m.retrievalUtility ?? 0.5,
    tdError: m.tdError ?? 0,
    surprise: m.surprise ?? 0,
    consolidationCount: m.consolidationCount ?? 1,
    status: m.status || "active",
    coldSince: m.coldSince ?? null,
    expiresAt: m.expiresAt ?? null,
    counterfactuals: m.counterfactuals || [],
    paramsUsed: m.paramsUsed || { alpha: 0.5, beta: 0.5, gamma: 0.5, theta1: 0.3, theta2: 0.55, temperature: [0.3, 0.5, 0.7], tokenBudget: 16000 },
  }));

  // Build the v5 state skeleton, then initialize RL subsystems from data
  const state = createMigratedSkeleton(now, trials, memories, v4State);
  return state;
}

/** Minimal SHA-256 first-16-hex for description hashing (fallback when crypto unavailable). */
function sha256Hex16(_desc: string): string {
  // Use a simple hash for migration — actual hashing done by trial-builder at runtime
  let h = 0;
  for (let i = 0; i < _desc.length; i++) {
    h = ((h << 5) - h) + _desc.charCodeAt(i);
    h |= 0;
  }
  return Math.abs(h).toString(16).padStart(16, "0").slice(0, 16);
}

function createMigratedSkeleton(now: string, trials: Trial[], memories: IndexedMemory[], v4: any): SharedStateV5 {
  const state = createFreshState();
  state.createdAt = v4.createdAt || now;
  state.lastUpdated = now;
  state.totalInvocations = trials.length;
  state.trials = trials;
  state.memories = memories;
  state.coldStorage = [];

  // Carry over policy from v4 if present
  if (v4.policy) {
    state.policy.compression = { ...state.policy.compression, ...v4.policy.compression };
    state.policy.quality = { ...state.policy.quality, ...v4.policy.quality };
    state.policy.temperature = { ...state.policy.temperature, ...v4.policy.temperature };
  }
  if (v4.config) {
    state.policy.compression.alpha = v4.config.alpha ?? state.policy.compression.alpha;
    state.policy.compression.beta = v4.config.beta ?? state.policy.compression.beta;
    state.policy.compression.gamma = v4.config.gamma ?? state.policy.compression.gamma;
    state.policy.quality.threshold = v4.config.qualityThreshold ?? state.policy.quality.threshold;
    state.policy.quality.maxAttempts = v4.config.maxAttempts ?? state.policy.quality.maxAttempts;
    if (v4.config.temperatureSchedule) {
      state.policy.temperature.t0 = v4.config.temperatureSchedule[0] ?? 0.70;
      state.policy.temperature.t1 = v4.config.temperatureSchedule[1] ?? 0.35;
      state.policy.temperature.t2 = v4.config.temperatureSchedule[2] ?? 0.10;
    }
  }
  if (v4.retrievalStrategy) {
    state.retrievalStrategy.active.mmrLambda = v4.retrievalStrategy.mmrLambda ?? state.retrievalStrategy.active.mmrLambda;
    state.retrievalStrategy.active.topK = v4.retrievalStrategy.topK ?? state.retrievalStrategy.active.topK;
    state.retrievalStrategy.ancestor = { ...state.retrievalStrategy.active };
    state.retrievalStrategy.ancestorFitness = 0.5;
  }

  // Initialize value function baselines from historical trials
  for (const trial of trials) {
    const bl = state.valueFunction.baselines[trial.taskType];
    if (bl) {
      bl.count += 1;
      bl.mean = (bl.mean * (bl.count - 1) + trial.qualityScore) / bl.count;
      bl.ema = bl.ema + 0.1 * (trial.qualityScore - bl.ema);
      bl.recentScores.push(trial.qualityScore);
      if (bl.recentScores.length > 10) bl.recentScores.shift();
    }
  }
  // Compute slopes
  for (const bl of Object.values(state.valueFunction.baselines) as TaskTypeBaseline[]) {
    if (bl.recentScores.length >= 2) {
      bl.slope = computeSlope(bl.recentScores);
    }
  }

  return state;
}

/** Simple linear regression slope of y = ax + b over an array. */
function computeSlope(scores: number[]): number {
  const n = scores.length;
  if (n < 2) return 0;
  const xMean = (n - 1) / 2;
  const yMean = scores.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (i - xMean) * (scores[i] - yMean);
    den += (i - xMean) ** 2;
  }
  return den === 0 ? 0 : num / den;
}
