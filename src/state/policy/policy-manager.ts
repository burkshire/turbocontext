// ============================================================================
// Turbocontext v5 — Policy Manager
// ============================================================================
// Pure functions for policy resolution, mutation, validation, and cloning.
// Extracted from state-manager.ts to give policy operations a dedicated home.
//
// All functions are stateless (except clonePolicy which uses structuredClone).
// The SharedStateManager delegates policy merging to resolveEffectivePolicy().
import type {
  PolicyState, PolicyOverrides, PolicyCompression,
  PolicyQuality, PolicyTemperature, PolicyModelTiers,
  PolicyRetrieval, PolicyExploration,
} from "../types.js";

// ── Policy resolution ──

/**
 * resolveEffectivePolicy: merges per-type overrides into the base policy.
 * Leaf-level override values replace base values.
 * Arrays (dimWeights, temperature sub-fields, etc.) are
 * fully replaced by overrides if present.
 */
export function resolveEffectivePolicy(
  base: PolicyState,
  overrides?: Partial<PolicyOverrides>,
): PolicyState {
  if (!overrides) return base;

  const merged: PolicyState = {
    compression: overrides.compression
      ? { ...base.compression, ...overrides.compression }
      : { ...base.compression },
    quality: overrides.quality
      ? { ...base.quality, ...overrides.quality }
      : { ...base.quality },
    temperature: overrides.temperature
      ? { ...base.temperature, ...overrides.temperature }
      : { ...base.temperature },
    modelTiers: overrides.modelTiers
      ? { ...base.modelTiers, ...overrides.modelTiers }
      : { ...base.modelTiers },
    retrieval: overrides.retrieval
      ? { ...base.retrieval, ...overrides.retrieval }
      : { ...base.retrieval },
    exploration: overrides.exploration
      ? { ...base.exploration, ...overrides.exploration }
      : { ...base.exploration },
    perType: base.perType,
  };

  return merged;
}

// ── Dot-notation parameter access ──

/**
 * getParamValue: reads a nested policy value by dot-notation path.
 *
 * Supported paths (matching TUNABLE_PARAMS in constants.ts):
 *   "compression.alpha", "quality.threshold", "temperature.t0",
 *   "modelTiers.lowComplexity", "retrieval.mmrLambda", "retrieval.topK",
 *   "retrieval.dimWeights.idfOverlap", "exploration.mutationMagnitude", etc.
 */
export function getParamValue(policy: PolicyState, paramPath: string): number {
  const parts = paramPath.split(".");
  let current: any = policy;
  for (const part of parts) {
    if (current == null || typeof current !== "object") {
      throw new Error(`Cannot read path "${paramPath}": "${part}" is not an object`);
    }
    if (!(part in current)) {
      throw new Error(`Cannot read path "${paramPath}": "${part}" not found`);
    }
    current = current[part];
  }
  if (typeof current !== "number") {
    throw new Error(`Path "${paramPath}" does not resolve to a number (got ${typeof current})`);
  }
  return current;
}

/**
 * applyMutation: sets a nested policy value by dot-notation path, returning a
 * new PolicyState (immutable update via structured clone).
 *
 * Returns a fresh PolicyState with the mutation applied.
 * The original policy is not modified.
 */
export function applyMutation(
  policy: PolicyState,
  paramPath: string,
  newValue: number,
): PolicyState {
  const cloned = clonePolicy(policy);
  const parts = paramPath.split(".");
  let current: any = cloned;
  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i];
    if (!(part in current) || typeof current[part] !== "object") {
      throw new Error(`Cannot set path "${paramPath}": "${part}" is not an object`);
    }
    // Clone the nested object so we don't mutate the original
    current[part] = { ...current[part] };
    current = current[part];
  }
  const lastPart = parts[parts.length - 1];
  if (typeof current[lastPart] !== "number") {
    throw new Error(
      `Cannot set path "${paramPath}": "${lastPart}" is not a number (got ${typeof current[lastPart]})`,
    );
  }
  current[lastPart] = newValue;
  return cloned;
}

// ── Utilities ──

/**
 * clonePolicy: deep-clones a PolicyState via structuredClone.
 */
export function clonePolicy(policy: PolicyState): PolicyState {
  return structuredClone(policy);
}

/**
 * normalizeDimWeights: ensures retrieval dimWeights sum to 1.0.
 * Returns a new weights object (does not mutate the input).
 * If the sum is 0, assigns uniform weights.
 */
export function normalizeDimWeights(weights: Record<string, number>): Record<string, number> {
  const sum = Object.values(weights).reduce((a, b) => a + b, 0);
  if (sum === 0) {
    const keys = Object.keys(weights);
    const uniform = 1 / keys.length;
    return Object.fromEntries(keys.map(k => [k, uniform]));
  }
  const normalized: Record<string, number> = {};
  for (const [k, v] of Object.entries(weights)) {
    normalized[k] = v / sum;
  }
  return normalized;
}
