// ============================================================================
// Turbocontext v5 — RND (Random Network Distillation)
// ============================================================================
// Exploration bonus via Random Network Distillation (Burda et al. 2019).
// Ported from the Python reference (turbocontext_v5_rl.py:889-942).
//
// Theory: A fixed random target network projects features into an embedding
// space. A learned predictor network tries to match the target output.
// MSE between target and predictor → exploration bonus.
//
// Familiar states → low MSE → low bonus (exploit).
// Novel states → high MSE → high bonus (explore).
//
// Both target and predictor use fixed seeds for deterministic initialization
// (matching the Python reference), ensuring reproducible exploration bonuses.
// ============================================================================

import type { RNDState } from "../types.js";
import { DEFAULT_FEATURE_DIM, FEATURE_NAMES } from "../constants.js";
import { sigmoid } from "./predictive-model.js";

// Fixed seeds matching Python _rnd_rng_target / _rnd_rng_predictor
const RND_EMBED_DIM = 32;
let _targetSeeded = false;
let _predictorSeeded = false;

// Simple seeded pseudo-random (matches Python's random.Random gauss behavior)
// We use a minimal linear congruential generator for deterministic init
function seededGauss(seed: number, counter: number): number {
  // Box-Muller with deterministic "random" from LCG
  const lcg = (s: number) => ((s * 1664525 + 1013904223) >>> 0) / 0xFFFFFFFF;
  let s = seed + counter * 127;
  let u = lcg(s);
  while (u === 0) { s = lcg(s); u = lcg(s); }
  let v = lcg(s + 1);
  while (v === 0) { s = lcg(s + 1); v = lcg(s + 1); }
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// Global counters for deterministic RNG
let _targetCounter = 0;
let _predictorCounter = 0;

/**
 * ensureRNDInit: lazy-initializes the RND matrices with fixed-seed determinism.
 * Called automatically by computeRNDBonus and trainRNDPredictor.
 */
export function ensureRNDInit(rnd: RNDState, featureDim: number = DEFAULT_FEATURE_DIM): void {
  // Check if already initialized by checking if predictorBias has values
  // (initRND in constants.ts creates zero-filled arrays, so we check non-zero length)
  if (rnd.targetProjection.length === featureDim &&
      rnd.predictorWeights.length === featureDim &&
      rnd.predictorBias.length === RND_EMBED_DIM) {
    // Already initialized by initRND() — but we need to verify targetProjection
    // has actual random values, not just zeros
    if (rnd.targetProjection[0] && rnd.targetProjection[0].length === RND_EMBED_DIM &&
        rnd.targetProjection[0][0] !== 0) {
      return; // Already properly initialized
    }
  }

  // Re-initialize with seeded random values
  rnd.targetProjection = Array.from({ length: featureDim }, (_, i) =>
    Array.from({ length: RND_EMBED_DIM }, (_, j) =>
      seededGauss(42, i * RND_EMBED_DIM + j)
    )
  );
  rnd.predictorWeights = Array.from({ length: featureDim }, (_, i) =>
    Array.from({ length: RND_EMBED_DIM }, (_, j) =>
      seededGauss(123, i * RND_EMBED_DIM + j) * 0.1
    )
  );
  rnd.predictorBias = new Array(RND_EMBED_DIM).fill(0);
}

// ── Embedding computation ──

/**
 * computeRNDEmbedding: projects features through target and predictor matrices.
 * Returns { target, pred } as [embedDim] arrays.
 */
export function computeRNDEmbedding(
  rnd: RNDState,
  features: Record<string, number>,
): { target: number[]; pred: number[] } {
  ensureRNDInit(rnd);

  const featVec = FEATURE_NAMES.map(k => features[k] ?? 0);
  const target: number[] = new Array(RND_EMBED_DIM).fill(0);
  const pred: number[] = [...rnd.predictorBias];

  for (let i = 0; i < featVec.length; i++) {
    const fi = featVec[i];
    for (let j = 0; j < RND_EMBED_DIM; j++) {
      target[j] += fi * rnd.targetProjection[i][j];
      pred[j] += fi * rnd.predictorWeights[i][j];
    }
  }

  return { target, pred };
}

// ── Bonus computation ──

/**
 * computeRNDBonus: exploration bonus from MSE between target and predictor.
 *
 * Returns a value in [0, 5]:
 *   bonus = 5 * sigmoid((mse - errorMean) / max(errorStd, 0.01))
 *
 * Novel states → high MSE → high bonus (up to ~5).
 * Familiar states → low MSE → low bonus (near 0).
 */
export function computeRNDBonus(rnd: RNDState, features: Record<string, number>): number {
  const { target, pred } = computeRNDEmbedding(rnd, features);

  let mse = 0;
  for (let j = 0; j < RND_EMBED_DIM; j++) {
    const err = target[j] - pred[j];
    mse += err * err;
  }
  mse /= RND_EMBED_DIM;

  const norm = (mse - rnd.errorMean) / Math.max(rnd.errorStd, 0.01);
  return 5.0 * sigmoid(norm);
}

// ── Predictor training ──

/**
 * trainRNDPredictor: one SGD step to reduce MSE between predictor and target.
 *
 * This shrinks the bonus for familiar states over time, driving exploration
 * toward genuinely novel experiences. Uses the same feature vector as the
 * predictive model (13 FEATURE_NAMES).
 *
 * Learning rate: 0.001 (matching Python reference).
 * EMA for error stats: alpha = 0.01.
 */
export function trainRNDPredictor(
  rnd: RNDState,
  features: Record<string, number>,
): void {
  ensureRNDInit(rnd);

  const featVec = FEATURE_NAMES.map(k => features[k] ?? 0);

  // Compute current target and pred
  const target: number[] = new Array(RND_EMBED_DIM).fill(0);
  const pred: number[] = [...rnd.predictorBias];

  for (let i = 0; i < featVec.length; i++) {
    const fi = featVec[i];
    for (let j = 0; j < RND_EMBED_DIM; j++) {
      target[j] += fi * rnd.targetProjection[i][j];
      pred[j] += fi * rnd.predictorWeights[i][j];
    }
  }

  // Compute errors and MSE
  const errors: number[] = new Array(RND_EMBED_DIM);
  let mse = 0;
  for (let j = 0; j < RND_EMBED_DIM; j++) {
    errors[j] = target[j] - pred[j];
    mse += errors[j] * errors[j];
  }
  mse /= RND_EMBED_DIM;

  // Update running error stats (EMA)
  rnd.errorMean = rnd.errorMean + 0.01 * (mse - rnd.errorMean);
  rnd.errorStd = rnd.errorStd + 0.01 * (Math.abs(mse - rnd.errorMean) - rnd.errorStd);

  // SGD update: gradient of MSE w.r.t. predictorWeights and predictorBias
  const lr = 0.001;
  for (let i = 0; i < featVec.length; i++) {
    const fi = featVec[i];
    for (let j = 0; j < RND_EMBED_DIM; j++) {
      // d(MSE)/d(w_ij) = -2 * error_j * fi / embedDim
      const grad = -2.0 * errors[j] * fi / RND_EMBED_DIM;
      rnd.predictorWeights[i][j] -= lr * grad;
    }
  }
  for (let j = 0; j < RND_EMBED_DIM; j++) {
    const grad = -2.0 * errors[j] / RND_EMBED_DIM;
    rnd.predictorBias[j] -= lr * grad;
  }
}
