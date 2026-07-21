// ============================================================================
// Turbocontext v5 — Thompson Sampling (Gamma-based Beta distribution)
// ============================================================================
// Proper Marsaglia-Tsang Gamma sampling for Beta-distribution Thompson Sampling.
// Ported from core/rl-core.ts (which uses these exact algorithms).
//
// Why NOT import from core/?  The state/ layer must not depend on core/.
// This file is a self-contained port of the Gamma sampler + Beta wrapper.
//
// The previous rl-engine.ts used a Box-Muller Normal approximation:
//   Beta(a,b) ~ N(a/(a+b), ab/((a+b)^2(a+b+1)))
// This is inaccurate in the tails (where exploration matters most).
// The Gamma method gives correct Beta samples for all (a,b) combinations.
// ============================================================================

import { sigmoid } from "./predictive-model.js";

// ── Public API ──

/**
 * sampleBeta: Sample from Beta(alpha, beta) distribution.
 *
 * Uses the Gamma method: Beta(a,b) = Gamma(a,1) / (Gamma(a,1) + Gamma(b,1)).
 * Both alpha and beta are clamped to a minimum of 0.1 for numeric stability.
 *
 * @returns A value in [0, 1] representing the sampled retrieval utility.
 */
export function sampleBeta(alpha: number, beta: number): number {
  const a = Math.max(0.1, alpha);
  const b = Math.max(0.1, beta);
  const x = sampleGamma(a);
  const y = sampleGamma(b);
  return x / (x + y);
}

// ── Gamma sampler ──

/**
 * sampleGamma: Marsaglia-Tsang rejection sampler for Gamma(shape, 1).
 *
 * For shape < 1: uses the shrink trick:
 *   Gamma(α,1) = Gamma(1+α,1) * U^(1/α)
 *
 * For shape >= 1: Marsaglia-Tsang method with
 *   d = shape - 1/3, c = 1/sqrt(9d)
 *   v = (1 + c*N(0,1))^3
 *   accept with probability based on N(0,1)^4 and log comparison
 */
export function sampleGamma(shape: number): number {
  if (shape < 1) {
    // Shrink trick: Gamma(α) = Gamma(1+α) * U^(1/α)
    const g = sampleGamma(shape + 1);
    return g * Math.pow(Math.random(), 1 / shape);
  }

  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);

  for (;;) {
    let x: number;
    let v: number;
    do {
      x = gaussianRandom();
      v = 1 + c * x;
    } while (v <= 0);

    v = v * v * v;
    const u = Math.random();
    // Fast path: ~98% of iterations pass this check
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    // Slow path: log comparison
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

// ── Gaussian random ──

/**
 * gaussianRandom: Box-Muller transform for N(0, 1).
 * Used internally by the Gamma sampler.
 */
export function gaussianRandom(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
