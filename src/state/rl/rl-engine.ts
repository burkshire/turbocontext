// ============================================================================
// Turbocontext v5 — RL Engine (Main Orchestrator)
// ============================================================================
//
// THE MAIN ENTRY POINT for all RL operations. Both execution modes
// (Lite and Full) call this class. It orchestrates the full pipeline:
// predictive model, value function, PER buffer, Thompson sampling,
// counterfactuals, HER, curiosity/RND, retrieval evolution, and
// cross-context sync.
//
// Lite mode (< 50ms):  predict → surprise → Thompson update → SGD →
//                       counterfactuals → HER → append → periodic evolution
// Full mode (deeper):   + decay/bump traces → TD(λ) → PER mini-batch →
//                        + value function update → curiosity/RND update →
//                        + consolidation → adversarial verification → sync
import type {
  SharedStateV5, Trial, QueryResult, RecordResult,
  EvolutionResult, ConsolidationResult, VerificationResult, SyncResult,
  StatusReport, RecordMode, TaskType, RetrievalInput,
} from "../types.js";
import { ContextOrigin } from "../types.js";
import { SharedStateManager } from "../state-manager.js";
import { extractFeatures, predictQuality, sgdUpdate, sigmoidDerivative } from "./predictive-model.js";
import {
  updateBaseline, updateAllBaselines, decayTraces, bumpTraces,
  applyTDUpdate, compositeReward, computeAdvantage,
  updateMemoryPriorities,
} from "./value-function.js";
import { PrioritizedReplayBuffer } from "./per-buffer.js";
import { sampleBeta } from "./thompson.js";
import { computeRNDBonus, trainRNDPredictor } from "./rnd.js";
import { retrieveMemories, type RetrievalQuery } from "./retrieval.js";
import { synthesizeCounterfactuals } from "../trial/counterfactuals.js";
import { herRelabel } from "../trial/her.js";
import { evolveRetrievalStrategy, incrementGenerationTrials } from "../evolution/retrieval-evolution.js";
import { consolidateMemories } from "../memory/consolidation.js";
import { getCurrentPhase, isLearningStep, isConsolidationStep } from "../curriculum/curriculum.js";
import { enqueueTrialForSync, consumePendingTrialsFromSkill } from "../bridge/cross-context-buffer.js";
import { appendTrialLog, appendEvolutionLog, appendConsolidationLog } from "../io.js";
import { PER_BATCH_SIZE, IDF_REBUILD_INTERVAL, EPSILON } from "../constants.js";

export class RLEngineV5 {
  private stateManager: SharedStateManager;
  private perBuffer: PrioritizedReplayBuffer;

  constructor(stateManager: SharedStateManager) {
    this.stateManager = stateManager;
    this.perBuffer = new PrioritizedReplayBuffer();
  }

  // ── Lifecycle ──

  /** Factory: create engine from disk or fresh. */
  static create(statePath?: string): RLEngineV5 {
    let sm = SharedStateManager.load(statePath);
    if (!sm) sm = SharedStateManager.create(statePath);
    return new RLEngineV5(sm);
  }

  /** Factory: create engine with V4→V5 migration. Returns null if neither v5 nor v4 state exists. */
  static loadOrMigrate(statePath?: string): RLEngineV5 | null {
    const sm = SharedStateManager.loadOrMigrate(statePath);
    if (!sm) return null;
    return new RLEngineV5(sm);
  }

  /** Factory: create engine with NO disk I/O — for tests and ephemeral use */
  static createInMemory(): RLEngineV5 {
    return new RLEngineV5(SharedStateManager.create(":memory:"));
  }

  saveState(): void {
    // v5.1: No-op for in-memory engines — no disk path configured
    if (!this.stateManager.hasPath()) return;
    this.stateManager.save();
  }

  // ── PRE-EXECUTION: Query optimal parameters ──

  /**
   * queryOptimalParams: given a task, returns the optimal parameters,
   * retrieved memories, and contrastive insights.
   *
   * Called at the START of every /turbocontext invocation or agent iteration.
   * Uses the effective policy (base + per-type overrides) and the current
   * curriculum phase to select parameters.
   */
  queryOptimalParams(input: RetrievalInput): QueryResult {
    const state = this.stateManager.getSnapshot();
    const policy = this.stateManager.getEffectivePolicy(input.taskType);
    const phase = getCurrentPhase(state);
    const phaseCfg = state.curriculum.phases[phase];

    // Select model tier from description length
    const descLen = input.description.length;
    const modelTier = descLen <= policy.modelTiers.lowComplexity ? "fast" as const
      : descLen <= policy.modelTiers.highComplexity ? "medium" as const
      : "best" as const;

    const tokenBudget = descLen <= policy.modelTiers.lowComplexity
      ? policy.retrieval.tokenBudgetTiers[0]
      : descLen <= policy.modelTiers.highComplexity
        ? policy.retrieval.tokenBudgetTiers[1]
        : policy.retrieval.tokenBudgetTiers[2];

    // Phase 1: MMR retrieval using 7-dim scoring + diversity re-ranking
    const retrievalInput: RetrievalQuery = {
      taskType: input.taskType,
      description: input.description,
      capabilityRequirements: input.capabilityRequirements,
    };
    const retrievedMemories = retrieveMemories(
      state.memories,
      retrievalInput,
      policy.retrieval,
      state.curiosity.idfCache,
    );

    const contrastiveInsights = this.getContrastiveInsights(input.taskType);

    // RND exploration bonus: high for novel feature vectors, low for familiar ones
    const featuresForQuery = extractFeaturesFromRetrievalInput(input);
    const explorationBonus = computeRNDBonus(state.curiosity.rnd, featuresForQuery);

    return {
      compressionWeights: {
        alpha: policy.compression.alpha,
        beta: policy.compression.beta,
        gamma: policy.compression.gamma,
      },
      temperatureSchedule: [policy.temperature.t0, policy.temperature.t1, policy.temperature.t2],
      modelTier,
      retrievalParams: { mmrLambda: policy.retrieval.mmrLambda, topK: policy.retrieval.topK },
      tokenBudget,
      maxAttempts: policy.quality.maxAttempts,
      qualityThreshold: policy.quality.threshold,
      retrievedMemories: retrievedMemories,
      contrastiveInsights,
      curriculumPhase: phase,
      explorationBonus,
    };
  }

  // ── POST-EXECUTION: Record and learn ──

  /**
   * recordTrial: dispatches to Lite or Full mode recording.
   *
   * Lite mode: fast, < 50ms, no LLM calls.
   *   - Predict + surprise
   *   - Thompson update
   *   - Single SGD step
   *   - Counterfactuals + HER
   *   - Periodic evolution
   *   - Enqueue for cross-context sync
   *
   * Full mode: deep RL, unbounded time.
   *   - Decay/bump traces
   *   - Predict + surprise
   *   - Thompson update
   *   - TD(λ) update + PER mini-batch
   *   - Value function update
   *   - Curiosity/RND update
   *   - Counterfactuals + HER
   *   - Periodic: evolution, consolidation, adversarial
   *   - Cross-context sync
   */
  recordTrial(trial: Trial, mode: RecordMode): RecordResult {
    if (mode === "lite") return this.recordTrialLite(trial);
    return this.recordTrialFull(trial);
  }

  // ── Lite mode ──

  private recordTrialLite(trial: Trial): RecordResult {
    const state = this.stateManager.getSnapshot() as SharedStateV5;
    const pm = state.predictiveModel;

    // Step 1: Predict
    const features = extractFeatures(trial);
    const predicted = predictQuality(pm, features);
    trial.predictedQuality = predicted;

    // Step 2: Surprise
    trial.surprise = Math.abs(predicted - trial.qualityScore);
    this.updateSurpriseStats(state, trial.surprise);

    // Step 3: Thompson update for retrieved memories
    this.updateThompsonForRetrieved(state, trial);

    // Step 4: Single SGD on predictive model
    const updatedPM = sgdUpdate(pm, features, trial.qualityScore);
    state.predictiveModel = updatedPM;

    // Step 5: Counterfactuals
    trial.counterfactuals = synthesizeCounterfactuals(trial, state);

    // Step 6: HER relabeling
    if (trial.outcome !== "success") {
      trial.herGoals = herRelabel(trial);
      // Feed HER rewards into learning: positive signal from "failed" trials
      this.applyHERFeedback(state, trial);
    }

    // Step 7: Curriculum phase (always set)
    trial.curriculumPhase = getCurrentPhase(state);

    // Step 8: Compute advantage
    // v5.1: Guard — ensure baseline exists
    let liteBaseline = state.valueFunction.baselines[trial.taskType];
    if (!liteBaseline) {
      liteBaseline = { mean: 0, ema: 0.5, count: 0, recentScores: [], slope: 0 };
      state.valueFunction.baselines[trial.taskType] = liteBaseline;
    }
    trial.advantage = computeAdvantage(liteBaseline, trial.qualityScore);
    trial.causalUtility = Math.min(1, Math.max(0, (1.0 + (trial.advantage ?? 0)) * trial.qualityScore));

    // Stamp trial with current evolution generation for fitness filtering
    trial.generation = state.retrievalStrategy.generation;

    // Step 9: Append + persist
    state.trials.push(trial);
    state.totalInvocations += 1;
    state.lastUpdated = new Date().toISOString();
    appendTrialLog(trial);

    // Step 10: Periodic evolution
    let evolutionResult: EvolutionResult | null = null;
    if (isLearningStep(state)) {
      const { result } = evolveRetrievalStrategy(state);
      evolutionResult = result;
      if (result.decision !== "no_mutation") {
        appendEvolutionLog({
          timestamp: new Date().toISOString(),
          generation: result.generation,
          mutation: result.mutation!,
          fitnessBefore: 0,
          fitnessAfter: 0,
          delta: result.fitnessDelta ?? 0,
          decision: result.decision,
          scenario: "",
        });
      }
    }
    incrementGenerationTrials(state);

    // Step 11: Periodic consolidation (every 20 Lite trials, matching curriculum phase 0)
    if (state.totalInvocations % 20 === 0 && state.totalInvocations > 0) {
      const { result: consResult } = consolidateMemories(state);
      if (consResult.consolidatedCount > 0 || consResult.archivedCount > 0) {
        const lastEntry = state.consolidationLog[state.consolidationLog.length - 1];
        if (lastEntry) appendConsolidationLog(lastEntry);
      }
    }

    // Step 12: Cross-context enqueue
    if (trial.context === ContextOrigin.SKILL) {
      state.crossContextBuffer = enqueueTrialForSync(state.crossContextBuffer, trial);
    }

    this.stateManager.bumpLastUpdated();
    this.stateManager.save();

    return {
      surprise: trial.surprise,
      tdError: 0,
      counterfactuals: trial.counterfactuals,
      herGoals: trial.herGoals,
      memoriesUpdated: trial.retrievedMemoryIds.length,
      pendingSyncCount: state.crossContextBuffer.pendingTrialsFromSkill.count,
    };
  }

  // ── Full mode ──

  private recordTrialFull(trial: Trial): RecordResult {
    const state = this.stateManager.getSnapshot() as SharedStateV5;
    const pm = state.predictiveModel;
    let vf = state.valueFunction;

    // Step 1: Decay eligibility traces
    vf = decayTraces(vf);

    // Step 2: Bump traces
    vf = bumpTraces(vf, trial.referencedMemoryIds, trial.retrievedMemoryIds);

    // Step 3: Predict + Surprise
    const features = extractFeatures(trial);
    const predicted = predictQuality(pm, features);
    trial.predictedQuality = predicted;
    trial.surprise = Math.abs(predicted - trial.qualityScore);
    this.updateSurpriseStats(state, trial.surprise);

    // Step 4: Thompson update
    this.updateThompsonForRetrieved(state, trial);

    // Step 5: Compute TD reward + error
    const tdReward = compositeReward(trial.outcome, trial.qualityScore);
    // v5.1: Guard — ensure baseline exists (handles V4 task types not in V5 TaskType enum)
    let baseline = vf.baselines[trial.taskType];
    if (!baseline) {
      baseline = { mean: 0, ema: 0.5, count: 0, recentScores: [], slope: 0 };
      vf.baselines[trial.taskType] = baseline;
    }
    const baselineEma = baseline.ema;
    const tdError = tdReward - baselineEma;

    // Step 6: Apply TD(λ) update through trace chain
    const { vf: updatedVf, memoryPatches } = applyTDUpdate(vf, state.memories as any, tdError);
    vf = updatedVf;
    for (const [memId, patch] of memoryPatches) {
      this.stateManager.updateMemory(memId, patch);
    }

    // Step 7: Compute advantage-weighted causal utility
    // v5.1: Use guarded baseline (already ensured above)
    trial.advantage = computeAdvantage(baseline, trial.qualityScore);
    const multiplier = 1.0 + Math.max(-0.5, Math.min(0.5, trial.advantage));
    trial.causalUtility = Math.min(1, Math.max(0, trial.qualityScore * multiplier));

    // Step 8: Update predictive model via PER mini-batch SGD
    this.perBuffer.add(features, trial.qualityScore, Math.abs(tdError));
    if (this.perBuffer.getSize() >= PER_BATCH_SIZE) {
      const batch = this.perBuffer.sample(PER_BATCH_SIZE);
      // Mini-batch SGD: per-weight gradients with sigmoid derivative and feature normalization.
      // Follows the same pattern as sgdUpdate in predictive-model.ts.
      // Weights updated online (one sample at a time) to handle importance weighting correctly.
      const batchIndices: number[] = [];
      const newErrors: number[] = [];
      for (const sample of batch) {
        const pred = predictQuality(pm, sample.features);
        const error = pred - sample.actualQuality;
        const grad = error * sigmoidDerivative(pred) * sample.importanceWeight;
        const lr = pm.learningRate;

        for (const name of Object.keys(pm.weights)) {
          const val = sample.features[name] ?? 0;
          const stat = pm.featureStats[name];
          const normVal = stat && stat.n >= 2 ? (val - stat.mean) / Math.max(stat.std, EPSILON) : val;
          pm.weights[name] -= lr * grad * normVal;
        }
        pm.intercept -= lr * grad;
        pm.nUpdates += 1;
        batchIndices.push(sample.index);
        newErrors.push(Math.abs(error));
      }
      if (batch.length > 0) {
        this.perBuffer.updatePriorities(batchIndices, newErrors);
      }
    }

    // Step 9: Update value function baselines
    vf = updateAllBaselines(vf, trial.taskType, trial.qualityScore);
    vf.td.totalUpdates += 1;
    state.valueFunction = vf;

    // Step 10: Update curiosity (RND)
    this.updateCuriosity(state, features, trial);

    // Step 11: Counterfactuals + HER
    trial.counterfactuals = synthesizeCounterfactuals(trial, state);
    if (trial.outcome !== "success") {
      // v5.1: Guard — use guarded baseline from step 5
      const herMean = baseline.mean;
      trial.herGoals = herRelabel(trial, [herMean, herMean, herMean, herMean]);
      // Feed HER rewards into value function and Thompson updates
      this.applyHERFeedback(state, trial);
    }

    // Step 12: Curriculum phase
    trial.curriculumPhase = getCurrentPhase(state);

    // Step 13: Periodic evolution
    if (state.totalInvocations % 4 === 0) {
      const { result } = evolveRetrievalStrategy(state);
      if (result.decision !== "no_mutation" && result.mutation) {
        appendEvolutionLog({
          timestamp: new Date().toISOString(),
          generation: result.generation,
          mutation: result.mutation,
          fitnessBefore: 0, fitnessAfter: 0,
          delta: result.fitnessDelta ?? 0,
          decision: result.decision,
          scenario: "",
        });
      }
    }
    incrementGenerationTrials(state);

    // Step 14: Periodic consolidation
    if (state.totalInvocations % 10 === 0) {
      const { result: consResult } = consolidateMemories(state);
      if (consResult.consolidatedCount > 0 || consResult.archivedCount > 0) {
        const lastEntry = state.consolidationLog[state.consolidationLog.length - 1];
        if (lastEntry) appendConsolidationLog(lastEntry);
      }
    }

    // Step 15: Cross-context sync
    let pendingSyncCount = state.crossContextBuffer.pendingTrialsFromSkill.count;
    if (pendingSyncCount > 0) {
      const { syncResult } = consumePendingTrialsFromSkill(
        state,
        (s, t) => this.recordTrialFull(t), // recursive full processing
      );
      pendingSyncCount = 0;
    }

    // Stamp trial with current evolution generation for fitness filtering
    trial.generation = state.retrievalStrategy.generation;

    // Step 16: Append + save
    state.trials.push(trial);
    state.totalInvocations += 1;
    state.lastUpdated = new Date().toISOString();
    appendTrialLog(trial);

    this.stateManager.bumpLastUpdated();
    this.stateManager.save();

    return {
      surprise: trial.surprise,
      tdError,
      counterfactuals: trial.counterfactuals,
      herGoals: trial.herGoals,
      memoriesUpdated: trial.retrievedMemoryIds.length,
      pendingSyncCount,
    };
  }

  // ── Periodic operations ──

  runEvolutionStep(): EvolutionResult {
    const state = this.stateManager.getSnapshot() as SharedStateV5;
    const { result } = evolveRetrievalStrategy(state);
    this.stateManager.bumpLastUpdated();
    return result;
  }

  runConsolidation(): ConsolidationResult {
    const state = this.stateManager.getSnapshot() as SharedStateV5;
    const { result } = consolidateMemories(state);
    this.stateManager.bumpLastUpdated();
    this.stateManager.save();
    return result;
  }

  /**
   * v8: Granular adversarial verification (autoresearch pattern).
   *
   * Three-tier scoring based on gap from current best/average:
   *   best_gap > 2%  → significant downgrade (obsolete success)
   *   avg_gap > 1%   → mild downgrade (above average but slipping)
   *   else           → boost confidence (adversarial test PASSED)
   *
   * Karpathy principle: "Build abstractions from experience."
   * A "success" at iteration 5 may be merely average by iteration 50.
   * This prevents "success inflation" in the retrieval system.
   */
  runAdversarialVerification(): VerificationResult {
    const state = this.stateManager.getSnapshot() as SharedStateV5;
    const active = state.memories.filter(m => m.status === "active");
    if (active.length < 10) return { verifiedCount: 0, staleCount: 0, overturnedCount: 0 };

    // Find successes with quality metrics
    const successes = active.filter(m => m.outcome === "success" && m.qualityScore > 0);
    if (successes.length < 5) return { verifiedCount: 0, staleCount: 0, overturnedCount: 0 };

    const bestScore = Math.max(...successes.map(m => m.qualityScore));
    const avgScore = successes.reduce((s, m) => s + m.qualityScore, 0) / successes.length;

    // Find candidates: oldest 10% of successes, not verified recently
    const oldestN = Math.min(5, Math.floor(active.length * 0.1));
    const candidates = active
      .filter(m => m.outcome === "success")
      .sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime())
      .slice(0, oldestN);

    let verifiedCount = 0, staleCount = 0, overturnedCount = 0;

    for (const mem of candidates) {
      if (bestScore <= 0) continue;

      // Three-tier gap scoring (autoresearch pattern)
      const bestGap = Math.max(0, (bestScore - mem.qualityScore) / Math.max(bestScore, 0.001));
      const avgGap = (mem.qualityScore - avgScore) / Math.max(avgScore, 0.001);

      if (bestGap > 0.02) {
        // >2% worse than current best → this "success" is obsolete
        this.stateManager.updateMemory(mem.id, {
          thompsonAlpha: Math.max(0.5, mem.thompsonAlpha * 0.7),
          thompsonBeta: Math.min(20, mem.thompsonBeta * 1.3),
          retrievalUtility: Math.max(0.2, mem.retrievalUtility * 0.8),
        });
        overturnedCount++;
      } else if (avgGap > 0.01) {
        // Above average but not best → mild downgrade
        this.stateManager.updateMemory(mem.id, {
          thompsonAlpha: Math.max(0.5, mem.thompsonAlpha * 0.85),
        });
        staleCount++;
      } else {
        // Still competitive → boost (adversarial test PASSED)
        this.stateManager.updateMemory(mem.id, {
          thompsonAlpha: Math.min(20, mem.thompsonAlpha * 1.05),
          causalUtility: Math.min(0.95, mem.causalUtility * 1.05),
        });
      }
      verifiedCount++;
    }

    this.stateManager.bumpLastUpdated();
    return { verifiedCount, staleCount, overturnedCount };
  }

  runCrossContextSync(): SyncResult {
    const state = this.stateManager.getSnapshot() as SharedStateV5;
    const { syncResult } = consumePendingTrialsFromSkill(
      state,
      (s, t) => this.recordTrialFull(t),
    );
    this.stateManager.bumpLastUpdated();
    this.stateManager.save();
    return syncResult;
  }

  // ── Introspection ──

  getStatus(): StatusReport { return this.stateManager.getStatusReport(); }

  /** v6: Access the underlying state snapshot for cross-branch transfer and analysis */
  getSnapshot(): Readonly<import("../types.js").SharedStateV5> {
    return this.stateManager.getSnapshot();
  }

  getContrastiveInsights(taskType: TaskType): string[] {
    return this.stateManager.getContrastiveInsights(taskType);
  }

  /** v5.1: Record an ablation result for causal graph construction */
  recordAblation(entry: import("../types.js").AblationEntry): void {
    this.stateManager.appendAblationResult(entry);
    this.stateManager.save();
  }

  /** v5.1: Get all ablation results for causal discovery */
  getAblationResults(): Readonly<import("../types.js").AblationEntry[]> {
    return this.stateManager.getAblationResults();
  }

  /**
   * v5.1: Causal boost for a source file given a task type.
   *
   * Uses V5 ablation results to compute how much a source file
   * causally contributes to quality for a given task type.
   * Returns 0 if no ablation data exists for the source.
   *
   * This is the V5 equivalent of V4's causal-graph-based boost,
   * using direct ablation measurements instead of graph inference.
   */
  getCausalBoost(source: string, taskType: TaskType): number {
    const ablations = this.stateManager.getAblationResults();
    const relevant = ablations.filter(
      a => a.sourceFile === source && a.taskType === taskType
    );
    if (relevant.length === 0) return 0;

    // Average causal delta across all ablations for this source+taskType
    const avgDelta = relevant.reduce((sum, a) => sum + a.causalDelta, 0) / relevant.length;
    // Map delta [-1, 1] to boost [0, 0.25]
    return Math.max(0, Math.min(0.25, (avgDelta + 1) / 8));
  }

  /**
   * getSourceBoostRL: Thompson-sampled boost for a source file.
   *
   * V5 equivalent of V4's sourceMemory-based Thompson boost.
   * Searches memories whose hypothesis mentions the source and returns
   * the average retrievalUtility (which is already Thompson-sampled).
   * Returns 0 if no relevant memories exist.
   */
  getSourceBoostRL(source: string): number {
    const memories = this.stateManager.getActiveMemories();
    const relevant = memories.filter(m => m.hypothesis.includes(source));
    if (relevant.length === 0) return 0;
    const avgUtility = relevant.reduce((sum, m) => sum + m.retrievalUtility, 0) / relevant.length;
    return Math.min(avgUtility, 0.25); // cap at 0.25 to match V4 boost limit
  }

  /**
   * getRetrievalContext: plateau signal, strategic directive, and adaptive MMR λ.
   *
   * V5 equivalent of V4's plateau detection + directive generation.
   * Uses value function baselines for plateau detection and retrieval
   * strategy fitness for MMR λ adaptation.
   */
  getRetrievalContext(taskType: TaskType): {
    plateau: { isPlateaued: boolean; reason: string };
    directive: { directive: string };
    adaptiveMmrLambda: number;
    idfCache: { weights: Record<string, number>; documentCount: number; lastRebuilt: string };
  } {
    const state = this.stateManager.getSnapshot();
    const baseline = state.valueFunction.baselines[taskType];
    const isPlateaued = baseline
      ? baseline.recentScores.length >= 5 && Math.abs(baseline.slope) < 0.005
      : false;
    const plateauReason = isPlateaued
      ? `improvement_stall (slope=${baseline?.slope?.toFixed(4) ?? "N/A"})`
      : "none";

    // Adaptive MMR λ: lower when plateaued (exploit known-good), higher when improving (explore)
    const strategy = state.retrievalStrategy;
    const baseMmrLambda = strategy.active?.mmrLambda ?? 0.70;
    const adaptiveMmrLambda = isPlateaued
      ? Math.max(0.25, baseMmrLambda - 0.15)
      : Math.min(0.75, baseMmrLambda + 0.05);

    // Strategic directive: adapt based on curriculum phase
    const phase = this.stateManager.getCurriculumPhase();
    const directives: Record<number, string> = {
      0: "DIVERSIFY — explore broadly across task types and parameter regions",
      1: "EXPLOIT — focus on best-known configurations, tighten around winners",
      2: "OPTIMIZE — fine-tune parameters within narrow ranges",
      3: "HARDEN — adversarial verify, stress-test edge cases",
    };
    const directive = directives[phase] ?? directives[0];

    return {
      plateau: { isPlateaued, reason: plateauReason },
      directive: { directive },
      adaptiveMmrLambda,
      idfCache: state.curiosity.idfCache,
    };
  }

  // ── Private helpers ──

  /**
   * updateThompsonForRetrieved: updates Beta(α,β) parameters for retrieved memories.
   *
   *   success → thompsonAlpha += (referenced ? 1.0 : 0.5)
   *   failure → thompsonBeta  += (referenced ? 1.0 : 0.5)
   *   crash   → thompsonBeta  += (referenced ? 2.0 : 1.0)
   *
   * Also updates lastRetrievedAt and retrievalUtility (Thompson sample cache).
   */
  private updateThompsonForRetrieved(state: SharedStateV5, trial: Trial): void {
    const refSet = new Set(trial.referencedMemoryIds);
    const now = new Date().toISOString();

    for (const memId of trial.retrievedMemoryIds) {
      const idx = state.memories.findIndex(m => m.id === memId);
      if (idx === -1) continue;

      const mem = state.memories[idx];
      const wasRef = refSet.has(memId);

      const patch: any = {
        lastRetrievedAt: now,
        retrievalCount: mem.retrievalCount + 1,
      };

      // Cap Thompson parameters to prevent Beta distribution convergence to point mass.
      // Without caps, Beta(500, 50) has std dev ~0.009 — effectively deterministic.
      // V4 caps at 50; we use the same bound for consistency.
      const MAX_THOMPSON = 50;
      switch (trial.outcome) {
        case "success":
          patch.thompsonAlpha = Math.min(mem.thompsonAlpha + (wasRef ? 1.0 : 0.5), MAX_THOMPSON);
          break;
        case "failure":
          patch.thompsonBeta = Math.min(mem.thompsonBeta + (wasRef ? 1.0 : 0.5), MAX_THOMPSON);
          break;
        case "crash":
          patch.thompsonBeta = Math.min(mem.thompsonBeta + (wasRef ? 2.0 : 1.0), MAX_THOMPSON);
          break;
        default:
          patch.thompsonBeta = Math.min(mem.thompsonBeta + 0.5, MAX_THOMPSON); // conservative
      }

      // Thompson sample cache: sample from Beta(α, β)
      const a = patch.thompsonAlpha ?? mem.thompsonAlpha;
      const b = patch.thompsonBeta ?? mem.thompsonBeta;
      patch.retrievalUtility = this.thompsonSampleUtility(a, b);

      Object.assign(state.memories[idx], patch);
    }
  }

  /**
   * thompsonSampleUtility: samples from Beta(α, β) distribution.
   *
   * Delegates to sampleBeta in thompson.ts which uses the proper
   * Gamma-based method (Marsaglia-Tsang rejection sampling).
   * Clamped to [0, 1] with minimum shape of 0.1 for numeric stability.
   */
  private thompsonSampleUtility(alpha: number, beta: number): number {
    return sampleBeta(alpha, beta);
  }

  /**
   * applyHERFeedback: feeds Hindsight Experience Replay rewards into the RL loop.
   *
   * HER (Andrychowicz et al. 2017) relabels failures as partial successes,
   * providing dense reward signal from sparse outcomes. This method:
   *   1. Updates value function baselines with HER rewards
   *   2. Provides positive Thompson signal for retrieved memories when HER
   *      identifies partial improvements
   *
   * Without this feedback, HER goals are computed but discarded — 131 lines of
   * goal synthesis with zero downstream effect on the learning system.
   */
  private applyHERFeedback(state: SharedStateV5, trial: Trial): void {
    const goals = trial.herGoals;
    if (!goals || goals.length === 0) return;

    // Use the maximum HER reward as the synthesized success signal
    const maxReward = Math.max(...goals.map(g => g.reward));

    // Update value function baseline: treat HER reward as a quality signal
    const vf = state.valueFunction;
    let baseline = vf.baselines[trial.taskType];
    if (!baseline) {
      baseline = { mean: 0, ema: 0.5, count: 0, recentScores: [], slope: 0 };
      vf.baselines[trial.taskType] = baseline;
    }
    // Blend HER reward into baseline EMA (so system learns from "good failures")
    baseline.ema = baseline.ema + 0.1 * (maxReward - baseline.ema);
    baseline.count += 1;

    // Provide positive Thompson signal for retrieved memories when HER found value
    // A high HER reward means the configuration was partially effective
    if (maxReward >= 0.5) {
      const refSet = new Set(trial.referencedMemoryIds);
      for (const memId of trial.retrievedMemoryIds) {
        const idx = state.memories.findIndex(m => m.id === memId);
        if (idx === -1) continue;
        const mem = state.memories[idx];
        const wasRef = refSet.has(memId);
        // Boost alpha (success count) proportional to HER reward
        const boost = wasRef ? maxReward : maxReward * 0.5;
        state.memories[idx].thompsonAlpha = Math.min(
          mem.thompsonAlpha + boost,
          50, // MAX_THOMPSON cap (consistent with updateThompsonForRetrieved)
        );
      }
    }
  }

  /** Updates running surprise statistics (mean, std, ring buffer). */
  private updateSurpriseStats(state: SharedStateV5, surprise: number): void {
    const ss = state.curiosity.surpriseStats;
    const n = ss.recentValues.length + 1;

    // Welford-style update
    const delta = surprise - ss.globalMean;
    ss.globalMean += delta / Math.max(n, 1);
    ss.globalStd = Math.sqrt(
      Math.max(0, ((ss.globalStd ** 2) * (n - 2) + delta * (surprise - ss.globalMean)) / Math.max(n - 1, 1))
    );
    ss.recentValues.push(surprise);
    if (ss.recentValues.length > 50) ss.recentValues.shift();
    ss.anomalyThreshold = ss.globalMean + 2 * ss.globalStd;
  }

  /** Updates curiosity: RND predictor, IDF cache, task-type exploration. */
  private updateCuriosity(state: SharedStateV5, features: Record<string, number>, trial: Trial): void {
    const c = state.curiosity;

    // Update task-type exploration
    const tex = c.taskTypeExploration[trial.taskType];
    if (tex) {
      tex.count += 1;
      tex.lastExplored = trial.timestamp;
      tex.avgSurprise = tex.avgSurprise + 0.1 * (trial.surprise - tex.avgSurprise);
      tex.successRate = tex.successRate + 0.1 * ((trial.outcome === "success" ? 1 : 0) - tex.successRate);
    }

    // Update capability coverage
    for (const cap of trial.capabilityRequirements) {
      c.capabilityCoverage[cap] = (c.capabilityCoverage[cap] || 0) + 1;
    }

    // Periodic IDF rebuild
    if (state.totalInvocations % IDF_REBUILD_INTERVAL === 0) {
      const allDocs = state.memories.map(m => `${m.hypothesis} ${m.insight}`);
      c.idfCache = rebuildIDFCache(allDocs);
    }

    // RND predictor training (Full mode only — Lite mode skips this)
    // One SGD step shrinks the bonus for familiar states, driving exploration to novel ones.
    trainRNDPredictor(c.rnd, features);
  }
}

// ── Mini IDF cache rebuild ──

function rebuildIDFCache(docs: string[]): { weights: Record<string, number>; documentCount: number; lastRebuilt: string } {
  const weights: Record<string, number> = {};
  const n = docs.length;
  if (n === 0) return { weights, documentCount: 0, lastRebuilt: new Date().toISOString() };

  for (const doc of docs) {
    const tokens = new Set(tokenize(doc));
    for (const token of tokens) {
      weights[token] = (weights[token] || 0) + 1;
    }
  }

  // IDF = log(N / (1 + df))
  for (const token of Object.keys(weights)) {
    weights[token] = Math.log(n / (1 + weights[token]));
  }

  return { weights, documentCount: n, lastRebuilt: new Date().toISOString() };
}

function tokenize(text: string): string[] {
  return text.toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(t => t.length > 1);
}

/**
 * extractFeaturesFromRetrievalInput: builds a feature vector for RND bonus
 * computation from a RetrievalInput (pre-execution context).
 *
 * Uses defaults for trial-only fields (compressionRatio, modelTier, etc.)
 * since no trial exists yet at query time.
 */
function extractFeaturesFromRetrievalInput(
  input: import("../types").RetrievalInput,
): Record<string, number> {
  const descLen = Math.min(input.description.length, 2000);
  const logDescLen = Math.log(1 + descLen);

  // Task type one-hot — must match FEATURE_NAMES in constants.ts
  // FEATURE_NAMES uses: code_review, code_generation, debugging, refactoring, documentation, architecture
  const taskTypes = ["code_review", "code_generation", "debugging", "refactoring", "documentation", "architecture"];
  const taskOneHot = Object.fromEntries(taskTypes.map(t => [`task_${t}`, t === input.taskType ? 1 : 0]));

  return {
    ...taskOneHot,
    log_description_length: logDescLen,
    compression_ratio: 0.5, // default: medium compression
    model_tier_fast: 0, // default: not fast
    model_tier_best: 0, // default: not best
    is_retry: 0, // pre-execution: not a retry
    log_token_budget: Math.log(1 + 8000), // default budget
    hour_of_day_sin: Math.sin((2 * Math.PI * new Date().getHours()) / 24),
  };
}
