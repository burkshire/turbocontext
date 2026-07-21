// ============================================================================
// RND (Random Network Distillation) Tests
// ============================================================================
import { describe, it, expect } from "vitest";
import {
  ensureRNDInit,
  computeRNDEmbedding,
  computeRNDBonus,
  trainRNDPredictor,
} from "../rl/rnd.js";
import { initRND, FEATURE_NAMES, DEFAULT_FEATURE_DIM } from "../constants.js";
import type { RNDState } from "../types.js";

function makeFeatures(overrides: Partial<Record<string, number>> = {}): Record<string, number> {
  const defaults: Record<string, number> = {
    task_code_review: 0,
    task_code_generation: 1,
    task_debugging: 0,
    task_refactoring: 0,
    task_documentation: 0,
    task_architecture: 0,
    log_description_length: Math.log(1 + 150),
    compression_ratio: 0.5,
    model_tier_fast: 0,
    model_tier_best: 0,
    is_retry: 0,
    log_token_budget: Math.log(1 + 8000),
    hour_of_day_sin: 0,
  };
  const result: Record<string, number> = { ...defaults };
  for (const [k, v] of Object.entries(overrides)) {
    if (v !== undefined) result[k] = v;
  }
  return result;
}

describe("ensureRNDInit", () => {
  it("initializes a fresh RND state with target and predictor matrices", () => {
    const rnd = initRND();
    ensureRNDInit(rnd);
    expect(rnd.targetProjection.length).toBe(DEFAULT_FEATURE_DIM);
    expect(rnd.targetProjection[0].length).toBeGreaterThan(0);
    expect(rnd.predictorWeights.length).toBe(DEFAULT_FEATURE_DIM);
    expect(rnd.predictorBias.length).toBeGreaterThan(0);
  });

  it("is idempotent — calling twice produces same dimensions", () => {
    const rnd = initRND();
    ensureRNDInit(rnd);
    const firstTarget = rnd.targetProjection[0][0];
    ensureRNDInit(rnd);
    // Already initialized, values preserved
    expect(rnd.targetProjection.length).toBe(DEFAULT_FEATURE_DIM);
    expect(rnd.targetProjection[0][0]).toBe(firstTarget);
  });
});

describe("computeRNDEmbedding", () => {
  it("produces target and pred arrays of correct size", () => {
    const rnd = initRND();
    const features = makeFeatures();
    const { target, pred } = computeRNDEmbedding(rnd, features);
    expect(target.length).toBeGreaterThan(0);
    expect(pred.length).toBe(target.length);
  });

  it("produces different target and pred for initial (untrained) state", () => {
    const rnd = initRND();
    const features = makeFeatures();
    const { target, pred } = computeRNDEmbedding(rnd, features);
    // Target and pred should differ (predictor not yet trained)
    let allEqual = true;
    for (let i = 0; i < target.length; i++) {
      if (Math.abs(target[i] - pred[i]) > 1e-6) { allEqual = false; break; }
    }
    // With random init weights * 0.1, predictor outputs are small but non-zero
    // Target outputs are larger (full gaussian)
    expect(allEqual).toBe(false);
  });

  it("is deterministic for identical inputs", () => {
    const rnd = initRND();
    const features = makeFeatures();
    const r1 = computeRNDEmbedding(rnd, features);
    const r2 = computeRNDEmbedding(rnd, features);
    expect(r1.target).toEqual(r2.target);
    expect(r1.pred).toEqual(r2.pred);
  });
});

describe("computeRNDBonus", () => {
  it("returns a bonus in [0, 5]", () => {
    const rnd = initRND();
    const bonus = computeRNDBonus(rnd, makeFeatures());
    expect(bonus).toBeGreaterThanOrEqual(0);
    expect(bonus).toBeLessThanOrEqual(5);
  });

  it("returns positive bonus for any feature input", () => {
    const rnd = initRND();
    const b1 = computeRNDBonus(rnd, makeFeatures({ task_code_review: 1, task_code_generation: 0 }));
    const b2 = computeRNDBonus(rnd, makeFeatures({ task_general: 1 }));
    expect(b1).toBeGreaterThan(0);
    expect(b2).toBeGreaterThan(0);
  });
});

describe("trainRNDPredictor", () => {
  it("reduces RND bonus for familiar features after repeated training", () => {
    const rnd = initRND();
    const features = makeFeatures();

    const initialBonus = computeRNDBonus(rnd, features);

    // Train on the same features many times
    for (let i = 0; i < 50; i++) {
      trainRNDPredictor(rnd, features);
    }

    const finalBonus = computeRNDBonus(rnd, features);
    // After training, predictor should converge → MSE decreases → bonus decreases
    expect(finalBonus).toBeLessThan(initialBonus);
  });

  it("gives higher bonus for novel features than familiar ones", () => {
    const rnd = initRND();
    const familiar = makeFeatures({ task_code_generation: 1 });
    const novel = makeFeatures({ task_code_review: 1, task_code_generation: 0, task_general: 1 });

    // Train extensively on "familiar" features
    for (let i = 0; i < 100; i++) {
      trainRNDPredictor(rnd, familiar);
    }

    const familiarBonus = computeRNDBonus(rnd, familiar);
    const novelBonus = computeRNDBonus(rnd, novel);

    // Novel features should produce higher bonus
    expect(novelBonus).toBeGreaterThan(familiarBonus);
  });

  it("updates errorMean and errorStd after training", () => {
    const rnd = initRND();
    const initialMean = rnd.errorMean;

    for (let i = 0; i < 10; i++) {
      trainRNDPredictor(rnd, makeFeatures());
    }

    // Error stats should have moved from initial 0
    expect(rnd.errorMean).not.toBe(initialMean);
  });
});
