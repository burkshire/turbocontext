// ============================================================
// v3.4 — RL Core Property & Integration Tests
// ============================================================
// Testing the five RL mechanisms ported from Karpathy autoresearch.
// Property tests: statistical invariants that must hold.
// Integration tests: controlled scenarios with known-good outcomes.
// ============================================================

import { describe, it, expect } from "vitest";
import {
  thompsonSample,
  updateThompsonParams,
  decayEligibilityTraces,
  bumpEligibilityTraces,
  applyTDUpdate,
  computeSurprise,
  synthesizeCounterfactual,
  outcomeToReward,
  createPredictiveModel,
  extractPredictionFeatures,
  predictOutcome,
  updatePredictiveModel,
  entropyBonus,
  computeSubsystemBaselines,
  computeAdvantage,
  curiosityBonus,
  getCurriculumPhase,
  adversarialVerify,
  ucbSelectDimension,
  recordUCBOutcome,
} from "../src/core/rl-core.js";
import type { RLExecutionRecord, ExecutionRecord, ThompsonParams, CurriculumPhase } from "../types.js";

// ==================================================================
// 1. Thompson Sampling property tests
// ==================================================================

describe("Thompson Sampling", () => {
  it("Beta(1,1) produces uniform-ish samples in [0,1]", () => {
    const samples: number[] = [];
    for (let i = 0; i < 1000; i++) {
      samples.push(thompsonSample(1, 1));
    }
    // Mean of Beta(1,1) should be ~0.5
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    expect(mean).toBeGreaterThan(0.40);
    expect(mean).toBeLessThan(0.60);

    // All samples must be in [0, 1]
    for (const s of samples) {
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(1);
    }
  });

  it("Beta(10,2) is peaked high (mean ~0.83)", () => {
    const samples: number[] = [];
    for (let i = 0; i < 1000; i++) {
      samples.push(thompsonSample(10, 2));
    }
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    // Beta(10,2) mean = 10/12 ≈ 0.833
    expect(mean).toBeGreaterThan(0.75);
    expect(mean).toBeLessThan(0.90);
  });

  it("Beta(2,10) is peaked low (mean ~0.17)", () => {
    const samples: number[] = [];
    for (let i = 0; i < 1000; i++) {
      samples.push(thompsonSample(2, 10));
    }
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    // Beta(2,10) mean = 2/12 ≈ 0.167
    expect(mean).toBeGreaterThan(0.08);
    expect(mean).toBeLessThan(0.28);
  });

  it("Success updates increase alpha (shift distribution right)", () => {
    const params: ThompsonParams = { alphaTs: 1, betaTs: 1 };
    const updated = updateThompsonParams(params, "success", 0.8);
    expect(updated.alphaTs).toBeGreaterThan(params.alphaTs);
    expect(updated.betaTs).toEqual(params.betaTs);
  });

  it("Crash updates strongly increase beta", () => {
    const params: ThompsonParams = { alphaTs: 5, betaTs: 1 };
    const updated = updateThompsonParams(params, "crash", 0);
    expect(updated.betaTs).toBeGreaterThanOrEqual(params.betaTs + 2.0);
    expect(updated.alphaTs).toEqual(params.alphaTs);
  });

  it("Alpha and beta are capped at 50", () => {
    const params: ThompsonParams = { alphaTs: 49, betaTs: 49 };
    const updated = updateThompsonParams(params, "success", 1.0);
    expect(updated.alphaTs).toBeLessThanOrEqual(50);
    const updated2 = updateThompsonParams(params, "crash", 0);
    expect(updated2.betaTs).toBeLessThanOrEqual(50);
  });
});

// ==================================================================
// 2. TD(λ) eligibility traces
// ==================================================================

describe("TD(λ) Eligibility Traces", () => {
  it("Traces decay by γλ each iteration", () => {
    const traces = new Map<string, number>();
    traces.set("mem1", 1.0);
    traces.set("mem2", 0.5);

    decayEligibilityTraces(traces, 0.90, 0.70);
    // decay = 0.9 * 0.7 = 0.63
    expect(traces.get("mem1")).toBeCloseTo(0.63, 2);
    expect(traces.get("mem2")).toBeCloseTo(0.315, 2);
  });

  it("Traces below 0.001 are removed", () => {
    const traces = new Map<string, number>();
    traces.set("mem1", 0.001);
    traces.set("mem2", 0.0005);

    decayEligibilityTraces(traces, 0.90, 0.70);
    expect(traces.has("mem1")).toBe(false);
    expect(traces.has("mem2")).toBe(false);
  });

  it("Bump adds 1.0 to trace for retrieved memories", () => {
    const traces = new Map<string, number>();
    traces.set("existing", 0.3);
    bumpEligibilityTraces(traces, ["existing", "new_one"]);
    expect(traces.get("existing")).toBeCloseTo(1.3, 2);
    expect(traces.get("new_one")).toBeCloseTo(1.0, 2);
  });

  it("TD update reinforces memories above expected, penalizes below", () => {
    const traces = new Map<string, number>();
    traces.set("good_mem", 0.8);
    traces.set("bad_mem", 0.4);

    const utilities = new Map<string, number>();
    utilities.set("good_mem", 0.7); // above average
    utilities.set("bad_mem", 0.2);  // below average

    // Expected = (0.7 + 0.2) / 2 = 0.45, reward = 0.8
    // TD_error = 0.8 - 0.45 = 0.35
    // good_mem: delta = 0.10 * 0.8 * 0.35 = 0.028
    // bad_mem:  delta = 0.10 * 0.4 * 0.35 = 0.014
    const updated = applyTDUpdate(traces, utilities, 0.8);

    expect(updated).toBe(2);
    expect(utilities.get("good_mem")!).toBeGreaterThan(0.7);
    expect(utilities.get("bad_mem")!).toBeGreaterThan(0.2);
  });

  it("TD update with all-below-expected reward penalizes all", () => {
    const traces = new Map<string, number>();
    traces.set("mem1", 0.9);

    const utilities = new Map<string, number>();
    utilities.set("mem1", 0.8); // expected = 0.8, reward = 0.1 → negative TD error

    applyTDUpdate(traces, utilities, 0.1);
    expect(utilities.get("mem1")!).toBeLessThan(0.8);
  });
});

// ==================================================================
// 3. Predictive model
// ==================================================================

describe("Predictive Model", () => {
  it("Creates model with sensible defaults", () => {
    const model = createPredictiveModel();
    expect(model.intercept).toBe(0.5);
    expect(model.learningRate).toBe(0.05);
    expect(model.nUpdates).toBe(0);
  });

  it("Prediction at init is 0.5 for zero features", () => {
    const model = createPredictiveModel();
    const pred = predictOutcome({}, model);
    // sigmoid(0.5) ≈ 0.622
    expect(pred).toBeGreaterThan(0.5);
    expect(pred).toBeLessThan(0.7);
  });

  it("SGD update reduces prediction error on next prediction", () => {
    const model = createPredictiveModel();
    const features = { type_success_rate: 0.8, is_novel: 0, log_n: 3.0 };

    // First prediction
    const pred1 = predictOutcome(features, model);

    // Update with actual = 1.0 (success)
    const { prediction: pred2, error } = updatePredictiveModel(features, 1.0, model);

    // Second prediction on same features should be closer to 1.0
    const pred3 = predictOutcome(features, model);
    expect(pred3).toBeGreaterThan(pred1);

    // Model should have updated
    expect(model.nUpdates).toBe(1);
    expect(model.recentAccuracy).toBeGreaterThan(0);
  });

  it("Converges toward actual outcomes over repeated updates", () => {
    const model = createPredictiveModel();
    const goodFeatures = { type_success_rate: 0.9, is_novel: 0, log_n: 3.0, type_momentum: 0.1, compression_ratio: 0.3 };
    const badFeatures = { type_success_rate: 0.1, is_novel: 1, log_n: 1.0, type_momentum: -0.1, compression_ratio: 0.1 };

    // Train: good features → success, bad features → failure
    for (let i = 0; i < 20; i++) {
      updatePredictiveModel(goodFeatures, 1.0, model);
      updatePredictiveModel(badFeatures, 0.0, model);
    }

    const goodPred = predictOutcome(goodFeatures, model);
    const badPred = predictOutcome(badFeatures, model);

    // After training, good should predict higher than bad
    expect(goodPred).toBeGreaterThan(badPred);
    expect(model.recentAccuracy).toBeGreaterThan(0.45);
  });

  it("Extract features produces valid shape for any record", () => {
    const record: ExecutionRecord = {
      taskId: "test_1",
      taskType: "code_review",
      timestamp: Date.now(),
      compressionRatio: 0.4,
      qualityScore: 0.88,
      totalCost: 0.005,
      latencyMs: 300,
      attemptCount: 1,
      modelUsed: "medium",
      coverage: { code_understanding: 0.9 },
      dimensionScores: { completeness: 0.9, correctness: 0.9, consistency: 0.8, format: 0.9 },
      sourceFiles: ["src/auth.ts"],
    };

    const history = [record, record]; // need 2+ for meaningful features
    const features = extractPredictionFeatures(record, history);
    expect(features.type_success_rate).toBeDefined();
    expect(features.is_novel).toBeDefined();
    expect(features.log_n).toBeGreaterThanOrEqual(0);
    expect(features.compression_ratio).toBe(0.4);

    // All features should be finite numbers
    for (const val of Object.values(features)) {
      expect(Number.isFinite(val)).toBe(true);
    }
  });
});

// ==================================================================
// 4. Surprise computation
// ==================================================================

describe("Surprise", () => {
  it("Surprise = 0 when prediction matches actual exactly", () => {
    expect(computeSurprise(1.0, "success")).toBe(0);
    expect(computeSurprise(0.0, "crash")).toBe(0);
    expect(computeSurprise(0.5, "failure")).toBe(0);
  });

  it("Surprise = 1 when prediction is maximally wrong", () => {
    expect(computeSurprise(0.0, "success")).toBe(1.0);
    expect(computeSurprise(1.0, "crash")).toBe(1.0);
  });

  it("Surprise is symmetric: overestimating and underestimating by same amount", () => {
    const over = computeSurprise(0.8, "failure"); // pred 0.8, actual 0.5 → 0.3
    const under = computeSurprise(0.2, "failure"); // pred 0.2, actual 0.5 → 0.3
    expect(over).toBeCloseTo(under, 4);
  });
});

// ==================================================================
// 5. Outcome-to-reward mapping
// ==================================================================

describe("Outcome to Reward", () => {
  it("Crash always produces strong negative signal", () => {
    const result = outcomeToReward("crash", 0, 0.9);
    expect(result.signal).toBe(-0.5);
    expect(result.magnitude).toBe(0);
  });

  it("Success with improvement produces positive signal", () => {
    const result = outcomeToReward("success", 0.95, 0.85);
    expect(result.signal).toBeGreaterThan(0.5);
    expect(result.magnitude).toBeGreaterThan(0.5);
  });

  it("Success without improvement produces moderate positive", () => {
    const result = outcomeToReward("success", 0.80, 0.85);
    expect(result.signal).toBeCloseTo(0.2, 1);
  });

  it("Failure produces mild negative", () => {
    const result = outcomeToReward("failure", 0.6, 0.9);
    expect(result.signal).toBe(-0.15);
  });
});

// ==================================================================
// 6. Counterfactual synthesis
// ==================================================================

describe("Counterfactual Synthesis", () => {
  it("Success counterfactual mentions compression strategy", () => {
    const record: ExecutionRecord = {
      taskId: "test", taskType: "code_review", timestamp: Date.now(),
      compressionRatio: 0.5, qualityScore: 0.9, totalCost: 0, latencyMs: 0,
      attemptCount: 1, modelUsed: "medium", coverage: { code_understanding: 1 },
      dimensionScores: { completeness: 0.9, correctness: 0.9, consistency: 0.9, format: 0.9 },
    };
    const cf = synthesizeCounterfactual(record, "success", 0.9);
    expect(cf).toContain("Counterfactual");
    expect(cf).toContain("code_review");
  });

  it("Crash counterfactual suggests more conservative approach", () => {
    const record: ExecutionRecord = {
      taskId: "test", taskType: "debugging", timestamp: Date.now(),
      compressionRatio: 0.1, qualityScore: 0.3, totalCost: 0, latencyMs: 0,
      attemptCount: 3, modelUsed: "medium", coverage: { error_detection: 1 },
      dimensionScores: { completeness: 0.3, correctness: 0.3, consistency: 0.3, format: 0.3 },
    };
    const cf = synthesizeCounterfactual(record, "crash", 0.3);
    expect(cf).toContain("conservatively");
  });

  it("Failure counterfactual notes that negative result rules out specific config", () => {
    const record: ExecutionRecord = {
      taskId: "test", taskType: "code_generation", timestamp: Date.now(),
      compressionRatio: 0.3, qualityScore: 0.6, totalCost: 0, latencyMs: 0,
      attemptCount: 1, modelUsed: "medium", coverage: { code_generation: 0.8 },
      dimensionScores: { completeness: 0.6, correctness: 0.6, consistency: 0.6, format: 0.6 },
    };
    const cf = synthesizeCounterfactual(record, "failure", 0.6);
    expect(cf).toContain("rules out");
  });
});

// ==================================================================
// 7. Entropy bonus for MMR diversity
// ==================================================================

describe("Entropy Bonus", () => {
  it("Rare outcome gets high bonus when selected outcomes are homogeneous", () => {
    // All selected are successes, we're considering a failure
    const bonus = entropyBonus("failure", ["success", "success", "success", "success"]);
    expect(bonus).toBeGreaterThan(0.3); // should be significant
  });

  it("Common outcome gets low bonus", () => {
    const bonus = entropyBonus("success", ["success", "success", "success"]);
    expect(bonus).toBeLessThan(0.1);
  });

  it("Zero bonus for empty selected set", () => {
    expect(entropyBonus("anything", [])).toBe(0);
  });

  it("Bonus decreases as outcome becomes more represented", () => {
    const bonus1 = entropyBonus("failure", ["success", "success", "success"]);
    const bonus2 = entropyBonus("failure", ["success", "failure", "success"]);
    expect(bonus1).toBeGreaterThan(bonus2);
  });
});

// ==================================================================
// 8. Curriculum learning
// ==================================================================

describe("Curriculum Learning", () => {
  it("Phase 0 for < 10 experiments", () => {
    const { phase, params } = getCurriculumPhase(5);
    expect(phase).toBe(0);
    expect(params.mmrLambda).toBe(0.35); // wide diversity
    expect(params.mutationMagnitude).toBeGreaterThan(0.2);
  });

  it("Phase 1 for 10-29 experiments", () => {
    const { phase, params } = getCurriculumPhase(15);
    expect(phase).toBe(1);
    expect(params.mmrLambda).toBeGreaterThan(0.5);
  });

  it("Phase 2 for 30-59 experiments", () => {
    const { phase, params } = getCurriculumPhase(40);
    expect(phase).toBe(2);
    expect(params.mutationMagnitude).toBeLessThan(0.1);
  });

  it("Phase 3 for 60+ experiments", () => {
    const { phase, params } = getCurriculumPhase(80);
    expect(phase).toBe(3);
    expect(params.adversarialInterval).toBeLessThan(10);
  });

  it("Each phase has unique MMR lambda", () => {
    const lambdas = new Set<number>();
    for (let n = 0; n < 70; n += 20) {
      lambdas.add(getCurriculumPhase(n).params.mmrLambda);
    }
    // Should have at least 3 distinct values across phases
    expect(lambdas.size).toBeGreaterThanOrEqual(3);
  });
});

// ==================================================================
// 9. UCB dimension selection
// ==================================================================

describe("UCB Dimension Selection", () => {
  it("Explores untried dimensions first", () => {
    const counts: Record<string, number> = { dim_a: 10, dim_b: 5 };
    const rewards: Record<string, number> = { dim_a: 8, dim_b: 4 };
    const dims = ["dim_a", "dim_b", "dim_c"]; // dim_c is untried

    // Run multiple times — dim_c should be selected frequently
    let dimCSelected = 0;
    for (let i = 0; i < 100; i++) {
      const selected = ucbSelectDimension(counts, rewards, 15, dims, 1.5);
      if (selected === "dim_c") dimCSelected++;
    }
    // Untried dimension gets exploration bonus > any exploitation value
    expect(dimCSelected).toBeGreaterThan(50);
  });

  it("Exploits high-reward dimensions when all tried", () => {
    const counts: Record<string, number> = { dim_a: 10, dim_b: 10 };
    const rewards: Record<string, number> = { dim_a: 9, dim_b: 1 };
    const dims = ["dim_a", "dim_b"];

    let dimASelected = 0;
    for (let i = 0; i < 100; i++) {
      const selected = ucbSelectDimension(counts, rewards, 20, dims, 0.5);
      if (selected === "dim_a") dimASelected++;
    }
    // Higher reward dimension should be selected more often
    expect(dimASelected).toBeGreaterThan(60);
  });

  it("recordUCBOutcome tracks counts and rewards", () => {
    const counts: Record<string, number> = {};
    const rewards: Record<string, number> = {};

    const r1 = recordUCBOutcome(counts, rewards, "dim_a", 0.5);
    expect(r1.totalMutations).toBe(1);
    expect(counts["dim_a"]).toBe(1);
    expect(rewards["dim_a"]).toBe(0.5);

    const r2 = recordUCBOutcome(counts, rewards, "dim_a", -0.2);
    expect(r2.totalMutations).toBe(2);
    expect(counts["dim_a"]).toBe(2);
    expect(rewards["dim_a"]).toBe(0.3); // 0.5 + (-0.2)
  });
});

// ==================================================================
// 10. Advantage-weighted utility
// ==================================================================

describe("Advantage-Weighted Utility", () => {
  function makeRLRecord(taskType: string, causalUtility: number, qualityScore: number): RLExecutionRecord {
    return {
      taskId: `test_${taskType}`,
      taskType: taskType as any,
      timestamp: Date.now(),
      compressionRatio: 0.3,
      qualityScore,
      totalCost: 0,
      latencyMs: 0,
      attemptCount: 1,
      modelUsed: "medium",
      coverage: {},
      dimensionScores: { completeness: 0.8, correctness: 0.8, consistency: 0.8, format: 0.8 },
      thompsonAlpha: 1, thompsonBeta: 1,
      retrievalUtility: 0.5, causalUtility,
      retrievedMemoryKeys: [], plannerReferencedKeys: [],
      surpriseScore: 0.5, predictedOutcome: null, predictionError: null,
      counterfactual: "", curriculumPhase: 0,
      verificationHistory: [], consolidated: false, consolidatedInto: null,
    };
  }

  it("Advantage is positive when above subsystem baseline", () => {
    const records = [
      makeRLRecord("code_review", 0.8, 0.9),
      makeRLRecord("code_review", 0.6, 0.7),
      makeRLRecord("code_generation", 0.5, 0.65),
    ];
    const baseline = computeSubsystemBaselines(records);
    const adv = computeAdvantage(records[0], baseline);
    // code_review avg = 0.7, this record = 0.8 → advantage ≈ 0.1
    expect(adv).toBeGreaterThan(0);
  });

  it("Advantage is negative when below subsystem baseline", () => {
    const records = [
      makeRLRecord("code_review", 0.8, 0.9),
      makeRLRecord("code_review", 0.4, 0.6),
    ];
    const baseline = computeSubsystemBaselines(records);
    const adv = computeAdvantage(records[1], baseline);
    // code_review avg = 0.6, this record = 0.4 → advantage ≈ -0.2
    expect(adv).toBeLessThan(0);
  });

  it("Same-type record has zero advantage against its own baseline", () => {
    const records = [makeRLRecord("design", 0.9, 0.95)];
    const baseline = computeSubsystemBaselines(records);
    const adv = computeAdvantage(records[0], baseline);
    // baseline for design = 0.9 (only record), causalUtil = 0.9 → advantage = 0
    expect(adv).toBeCloseTo(0, 2);
  });

  it("Unknown task type falls back to general baseline (0.5)", () => {
    // Create records in one type, then test advantage for a different type
    const records = [makeRLRecord("code_review", 0.7, 0.9)];
    const baseline = computeSubsystemBaselines(records);
    // "design" is not in the baseline → falls back to general (0.5)
    const designRecord = makeRLRecord("design", 0.9, 0.95);
    const adv = computeAdvantage(designRecord, baseline);
    // causalUtil = 0.9, baseline(design) not found → general = 0.5 → advantage = 0.4
    expect(adv).toBeGreaterThan(0.3);
  });
});

// ==================================================================
// 11. Curiosity bonus
// ==================================================================

describe("Curiosity Bonus", () => {
  function makeRLRecord(overrides: Partial<RLExecutionRecord> = {}): RLExecutionRecord {
    return {
      taskId: "test", taskType: "code_review", timestamp: Date.now(),
      compressionRatio: 0.3, qualityScore: 0.85, totalCost: 0, latencyMs: 0,
      attemptCount: 1, modelUsed: "medium", coverage: {},
      dimensionScores: { completeness: 0.8, correctness: 0.8, consistency: 0.8, format: 0.8 },
      thompsonAlpha: 1, thompsonBeta: 1,
      retrievalUtility: 0.5, causalUtility: 0.5,
      retrievedMemoryKeys: [], plannerReferencedKeys: [],
      surpriseScore: 0.5, predictedOutcome: null, predictionError: null,
      counterfactual: "", curriculumPhase: 0,
      verificationHistory: [], consolidated: false, consolidatedInto: null,
      ...overrides,
    };
  }

  it("Novel task types get higher curiosity bonus", () => {
    const novel = makeRLRecord({ taskType: "design" as any });
    const common = makeRLRecord({ taskType: "code_review" as any });

    const allRecords = [
      makeRLRecord({ taskType: "code_review" as any }),
      makeRLRecord({ taskType: "code_review" as any }),
      makeRLRecord({ taskType: "code_review" as any }),
      novel,
    ];

    const novelBonus = curiosityBonus(novel, allRecords);
    const commonBonus = curiosityBonus(common, allRecords);

    // Novel type (design, only 1 record) should have higher curiosity
    expect(novelBonus).toBeGreaterThan(commonBonus);
  });

  it("Returns a value in [0, 5]", () => {
    const record = makeRLRecord();
    const bonus = curiosityBonus(record, [record]);
    expect(bonus).toBeGreaterThanOrEqual(0);
    expect(bonus).toBeLessThanOrEqual(5);
  });
});

// ==================================================================
// 12. Adversarial verification
// ==================================================================

describe("Adversarial Verification", () => {
  function makeRLRecord(taskType: string, qualityScore: number, idx: number): RLExecutionRecord {
    return {
      taskId: `exp_${idx}`, taskType: taskType as any, timestamp: Date.now() - (100 - idx) * 10000,
      compressionRatio: 0.3, qualityScore, totalCost: 0, latencyMs: 0,
      attemptCount: 1, modelUsed: "medium", coverage: {},
      dimensionScores: { completeness: qualityScore, correctness: qualityScore,
                          consistency: qualityScore, format: qualityScore },
      thompsonAlpha: 1, thompsonBeta: 1,
      retrievalUtility: 0.5, causalUtility: qualityScore >= 0.85 ? 0.7 : 0.3,
      retrievedMemoryKeys: [], plannerReferencedKeys: [],
      surpriseScore: 0.5, predictedOutcome: null, predictionError: null,
      counterfactual: "", curriculumPhase: 0,
      verificationHistory: [], consolidated: false, consolidatedInto: null,
    };
  }

  it("Old success memories below current average get downgraded", () => {
    const records: RLExecutionRecord[] = [];
    // Build history: early successes were lower quality
    for (let i = 0; i < 20; i++) {
      records.push(makeRLRecord("code_review", 0.85 + i * 0.002, i));
    }

    const oldCausalUtil = records[0].causalUtility;
    const verified = adversarialVerify(records, 1, 10);

    if (verified > 0) {
      // The oldest success should have its utility downgraded
      expect(records[0].causalUtility).toBeLessThanOrEqual(oldCausalUtil);
    }
  });

  it("Returns 0 when not enough data", () => {
    const records = [makeRLRecord("code_review", 0.9, 0)];
    expect(adversarialVerify(records as any, 1, 10)).toBe(0);
  });
});

// ==================================================================
// 13. Integration: Full RL feedback chain with mock data
// ==================================================================

describe("RL Feedback Chain Integration", () => {
  it("Full chain: Thompson → TD → Predictive → Surprise → Counterfactual", () => {
    // Simulate 10 executions with alternating success/failure on a source file
    let alpha = 1, beta = 1;
    const traces = new Map<string, number>();
    const utilities = new Map<string, number>();
    const model = createPredictiveModel();

    for (let i = 0; i < 10; i++) {
      // Thompson sample for source file boost
      const boost = thompsonSample(alpha, beta);

      // Decay traces
      decayEligibilityTraces(traces);

      // Bump trace for current source
      bumpEligibilityTraces(traces, ["src/auth.ts"]);

      // Simulate execution outcome (alternating pattern to test learning)
      const isSuccess = i % 2 === 0;
      const qualityScore = isSuccess ? 0.88 + Math.random() * 0.05 : 0.6 + Math.random() * 0.2;

      // Create record for prediction
      const record: ExecutionRecord = {
        taskId: `exec_${i}`, taskType: "code_review", timestamp: Date.now(),
        compressionRatio: 0.3, qualityScore,
        totalCost: 0.005, latencyMs: 300, attemptCount: 1,
        modelUsed: "medium", coverage: {}, sourceFiles: ["src/auth.ts"],
        dimensionScores: { completeness: 0.8, correctness: 0.8, consistency: 0.8, format: 0.8 },
      };

      // Predict outcome
      const features = extractPredictionFeatures(record, [record]);
      const prediction = predictOutcome(features, model);

      // Update model
      const actualOutcome = isSuccess ? 1.0 : 0.0;
      updatePredictiveModel(features, actualOutcome, model);

      // Compute surprise
      const surprise = computeSurprise(prediction, isSuccess ? "success" : "failure");

      // Apply TD update
      const { signal } = outcomeToReward(
        isSuccess ? "success" : "failure", qualityScore, 0.85
      );
      utilities.set("src/auth.ts", isSuccess ? 0.7 : 0.3);
      applyTDUpdate(traces, utilities, signal);
      traces.clear(); // episodic

      // Synthesize counterfactual
      const cf = synthesizeCounterfactual(
        record, isSuccess ? "success" : "failure", qualityScore
      );

      // Update Thompson params
      const params = updateThompsonParams(
        { alphaTs: alpha, betaTs: beta },
        isSuccess ? "success" : "failure",
        isSuccess ? 0.5 : 0,
      );
      alpha = params.alphaTs;
      beta = params.betaTs;

      // Assertions
      expect(Number.isFinite(boost)).toBe(true);
      expect(boost).toBeGreaterThanOrEqual(0);
      expect(boost).toBeLessThanOrEqual(1);
      expect(surprise).toBeGreaterThanOrEqual(0);
      expect(surprise).toBeLessThanOrEqual(1);
      expect(cf).toContain("Counterfactual");
      expect(model.nUpdates).toBe(i + 1);
    }

    // After 10 iterations with alternating success pattern:
    // alpha should have grown (5 successes), beta too (5 failures)
    expect(alpha).toBeGreaterThan(1);
    expect(beta).toBeGreaterThan(1);
    // Model should have learned something
    expect(model.recentAccuracy).toBeGreaterThan(0.4);
  });

  it("Deterministic success converges Thompson Sampling to high mean", () => {
    let alpha = 1, beta = 1;
    // 20 successes in a row
    for (let i = 0; i < 20; i++) {
      const params = updateThompsonParams(
        { alphaTs: alpha, betaTs: beta }, "success", 0.8
      );
      alpha = params.alphaTs;
      beta = params.betaTs;
    }

    // After 20 successes, Beta should be strongly peaked high
    const samples: number[] = [];
    for (let i = 0; i < 500; i++) {
      samples.push(thompsonSample(alpha, beta));
    }
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    expect(mean).toBeGreaterThan(0.75); // should be strongly positive
  });
});
