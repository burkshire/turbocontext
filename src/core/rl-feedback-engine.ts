// ============================================================
// v3.4 — RLFeedbackEngine (extracted from learner.ts)
// ============================================================
// Owns all RL state and feedback logic. The Learner delegates
// RL operations to this engine, keeping learner.ts focused on
// coordination rather than RL implementation details.
//
// State owned:
//   - predictiveModel (SGD-updated linear outcome predictor)
//   - eligibilityTraces (TD(λ) credit assignment traces)
//   - retrievalStrategy (self-evolving retrieval hyperparameters)
//   - experienceLib (meta-model: which mutations work when)
//   - curriculumTotal (experiment counter for phase tracking)
// ============================================================

import type {
  ExecutionRecord, TaskType,
  PredictiveModel, RetrievalStrategyState,
  ExperienceEntry, SourceMemory,
  SurpriseStats, TwoPhaseRetrievalConfig,
} from "../types.js";
import { DEFAULT_TWO_PHASE_CONFIG } from "../types.js";
import { createDefaultRetrievalStrategy } from "../types.js";
import {
  thompsonSourceBoost,
} from "./retrieval-system.js";
import {
  proposeRetrievalStrategyMutation,
  decideRetrievalStrategyMutation,
  recordRetrievalStrategyTrial,
} from "./evolution-engine.js";
import {
  thompsonSample,
  decayEligibilityTraces,
  bumpEligibilityTraces,
  applyTDUpdate,
  createPredictiveModel,
  extractPredictionFeatures,
  predictOutcome,
  updatePredictiveModel,
  computeSurprise,
  synthesizeCounterfactual,
  getCurriculumPhase,
  outcomeToReward,
} from "./rl-core.js";

export class RLFeedbackEngine {
  // ── RL state ──
  predictiveModel: PredictiveModel = createPredictiveModel();
  eligibilityTraces: Map<string, number> = new Map();
  retrievalStrategy: RetrievalStrategyState = createDefaultRetrievalStrategy();
  experienceLib: ExperienceEntry[] = [];
  curriculumTotal: number = 0;

  // v4.0: Rolling surprise tracking (agent.py _update_surprise_stats)
  surpriseStats: SurpriseStats = {
    globalMeanSurprise: 0.5,
    surpriseHistory: [],
  };

  // ── External hooks (set by Learner after construction) ──
  /** Look up source file history for utility computation */
  private sourceMemoryProvider: (() => Map<string, SourceMemory>) | null = null;
  /** Look up best quality for a task type branch */
  private branchBestQualityProvider: ((taskType: TaskType) => number) | null = null;
  /** Look up branch quality threshold for outcome classification */
  private branchThresholdProvider: ((taskType: TaskType) => number) | null = null;
  /** Max attempts from config */
  private maxAttemptsProvider: (() => number) | null = null;

  /** Wire up dependencies that live on the Learner */
  setProviders(deps: {
    sourceMemory: () => Map<string, SourceMemory>;
    branchBestQuality: (tt: TaskType) => number;
    branchThreshold: (tt: TaskType) => number;
    maxAttempts: () => number;
  }): void {
    this.sourceMemoryProvider = deps.sourceMemory;
    this.branchBestQualityProvider = deps.branchBestQuality;
    this.branchThresholdProvider = deps.branchThreshold;
    this.maxAttemptsProvider = deps.maxAttempts;
  }

  // ── Source boost (Thompson Sampling) ──

  getSourceBoostRL(source: string): number {
    if (!this.sourceMemoryProvider) return 0;
    const mem = this.sourceMemoryProvider().get(source);
    if (!mem || mem.attempts < 2) return 0;
    return thompsonSourceBoost(mem.attempts, mem.successes);
  }

  // ── Full RL feedback loop ──

  applyRLFeedback(execution: ExecutionRecord): {
    tdUpdated: number; surprise: number; prediction: number; counterfactual: string;
  } {
    const maxAttempts = this.maxAttemptsProvider?.() ?? 3;
    const threshold = this.branchThresholdProvider?.(execution.taskType) ?? 0.85;
    const isSuccess = execution.qualityScore >= threshold;
    const outcome: "success" | "failure" | "crash" =
      execution.attemptCount >= maxAttempts && !isSuccess
        ? "crash" : isSuccess ? "success" : "failure";

    // 1. Decay traces
    decayEligibilityTraces(this.eligibilityTraces);

    // 2. Bump traces
    const sourceKeys = (execution.sourceFiles || []).map(f =>
      `source:${f.replace(/\.[^.]+$/, "")}`
    );
    bumpEligibilityTraces(this.eligibilityTraces, [
      `task:${execution.taskId}`,
      `type:${execution.taskType}`,
      ...sourceKeys,
    ]);

    // 3. Outcome → reward
    const bestQ = this.branchBestQualityProvider?.(execution.taskType) ?? 0.85;
    const { signal } = outcomeToReward(outcome, execution.qualityScore, bestQ);

    // 4. TD(λ) credit assignment
    // Traces decay naturally at γλ=0.63 per step and persist across updates
    // for multi-step temporal credit. Only clear on episode boundary, not after
    // every non-zero update (which would reduce to TD(0)).
    const utilityMap = this.buildSourceUtilityMap(sourceKeys);
    const tdUpdated = applyTDUpdate(this.eligibilityTraces, utilityMap, signal);

    // 5. Predictive model + surprise
    const features = extractPredictionFeatures(execution, [execution]);
    const { prediction, error } = updatePredictiveModel(
      features,
      outcome === "success" ? 1.0 : outcome === "crash" ? 0.0 : 0.5,
      this.predictiveModel,
    );
    const surprise = computeSurprise(prediction, outcome);

    // v4.0: Update rolling surprise stats for retrieval normalization
    this.updateSurpriseStats(surprise);

    // 6. Counterfactual
    const counterfactual = synthesizeCounterfactual(execution, outcome, execution.qualityScore);

    // 7. Retrieval strategy trial
    this.retrievalStrategy = recordRetrievalStrategyTrial(
      this.retrievalStrategy, outcome, execution.qualityScore, bestQ,
    );
    if (this.retrievalStrategy.trialsInGeneration >= 5 &&
        this.retrievalStrategy.pendingMutation) {
      const result = decideRetrievalStrategyMutation(this.retrievalStrategy);
      if (result) {
        console.log(
          `[TurboContext RL] Mutation ${result.decision} ` +
          `(Δ=${(result.delta * 100).toFixed(2)}%, gen=${this.retrievalStrategy.generation})`
        );
      }
    }

    // 8. Curriculum
    this.curriculumTotal++;

    return { tdUpdated, surprise, prediction, counterfactual };
  }

  private buildSourceUtilityMap(sourceKeys: string[]): Map<string, number> {
    const map = new Map<string, number>();
    if (this.sourceMemoryProvider) {
      const sourceMem = this.sourceMemoryProvider();
      for (const key of sourceKeys) {
        const src = key.replace("source:", "");
        const mem = sourceMem.get(src);
        map.set(key, mem && mem.attempts > 0 ? mem.successes / mem.attempts : 0.5);
      }
    }
    for (const [key] of this.eligibilityTraces) {
      if (!map.has(key)) map.set(key, 0.5);
    }
    return map;
  }

  // ── Diagnostics ──

  getPredictiveModelStats(): { accuracy: number; nUpdates: number; featureCount: number } {
    return {
      accuracy: this.predictiveModel.recentAccuracy,
      nUpdates: this.predictiveModel.nUpdates,
      featureCount: Object.keys(this.predictiveModel.featureWeights).length,
    };
  }

  getCurriculumContext(): ReturnType<typeof getCurriculumPhase> {
    return getCurriculumPhase(this.curriculumTotal);
  }

  // ── v4.0: Surprise stats ──

  /**
   * v4.0: Update rolling surprise statistics (agent.py _update_surprise_stats).
   *
   * Maintains last 50 surprise values and a running global mean.
   * Used to normalize surprise bonuses in retrieval scoring —
   * "surprising" is relative to what we typically observe.
   */
  private updateSurpriseStats(surprise: number): void {
    const hist = this.surpriseStats.surpriseHistory;
    hist.push(surprise);
    if (hist.length > 50) {
      this.surpriseStats.surpriseHistory = hist.slice(-50);
    }
    const activeHistory = this.surpriseStats.surpriseHistory;
    this.surpriseStats.globalMeanSurprise = activeHistory.length > 0
      ? Math.round(
          activeHistory.reduce((a, b) => a + b, 0) / activeHistory.length * 10000
        ) / 10000
      : 0.5;
  }

  /**
   * v4.0: Get the global mean surprise for retrieval normalization.
   * Falls back to 0.5 when insufficient history (< 10 values).
   */
  getGlobalMeanSurprise(): number {
    if (this.surpriseStats.surpriseHistory.length < 10) return 0.5;
    return this.surpriseStats.globalMeanSurprise;
  }

  // ── v4.0: Two-phase retrieval config ──

  /**
   * v4.0: Get two-phase retrieval configuration adjusted for current curriculum phase.
   *
   * Later phases use slightly larger oversampling (more candidates to re-rank)
   * and higher advantage scaling (more weight on causal evidence).
   */
  getTwoPhaseRetrievalConfig(): TwoPhaseRetrievalConfig {
    const curriculum = this.getCurriculumContext();
    // Adjust oversampling: more in explore phase (need wider net),
    // less in principled phase (narrower, more confident)
    const oversampleMultiplier = curriculum.phase === 0 ? 3.0
      : curriculum.phase === 1 ? 2.5
      : curriculum.phase === 2 ? 2.0
      : 2.5; // phase 3: balanced
    return {
      oversampleMultiplier,
      advantageScale: DEFAULT_TWO_PHASE_CONFIG.advantageScale,
      maxPhase1Dimensions: DEFAULT_TWO_PHASE_CONFIG.maxPhase1Dimensions,
    };
  }

  getRetrievalStrategy(): RetrievalStrategyState {
    return this.retrievalStrategy;
  }

  // ── Mutation ──

  proposeRetrievalMutation(): Record<string, unknown> | null {
    const result = proposeRetrievalStrategyMutation(this.retrievalStrategy);
    if (result) {
      this.retrievalStrategy = result.newStrategy;
      return result.mutation;
    }
    return null;
  }

  // ── Serialization ──

  toJSON(): Record<string, unknown> {
    return {
      predictiveModel: this.predictiveModel,
      retrievalStrategy: this.retrievalStrategy,
      experienceLib: this.experienceLib.slice(-200),
      curriculumTotal: this.curriculumTotal,
      // v4.0: Persist surprise stats for retrieval normalization continuity
      surpriseStats: {
        globalMeanSurprise: this.surpriseStats.globalMeanSurprise,
        surpriseHistory: this.surpriseStats.surpriseHistory.slice(-50),
      },
    };
  }

  fromJSON(data: Record<string, unknown>): void {
    if (data.predictiveModel) {
      this.predictiveModel = data.predictiveModel as PredictiveModel;
    }
    if (data.retrievalStrategy) {
      this.retrievalStrategy = data.retrievalStrategy as RetrievalStrategyState;
    } else {
      this.retrievalStrategy = createDefaultRetrievalStrategy();
    }
    if (data.experienceLib) {
      this.experienceLib = data.experienceLib as ExperienceEntry[];
    }
    if (typeof data.curriculumTotal === "number") {
      this.curriculumTotal = data.curriculumTotal;
    }
    // v4.0: Restore surprise stats
    if (data.surpriseStats) {
      const ss = data.surpriseStats as Record<string, unknown>;
      this.surpriseStats = {
        globalMeanSurprise: (ss.globalMeanSurprise as number) ?? 0.5,
        surpriseHistory: (ss.surpriseHistory as number[]) ?? [],
      };
    }
  }
}
