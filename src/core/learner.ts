// ============================================================
// Phase 5: Branch-based Continuous Learning
// ============================================================
// 从每次执行中学习，自动优化算法参数。
//
// 核心改进（来自 autoresearch 分支架构）：
//   1. 每个 TaskType 是一个独立"分支"，拥有独立参数和轨迹
//   2. 分支级轨迹追踪：momentum, improvementVelocity, stability, novelty
//   3. 周期性分支总结（每 N 次执行触发）
//   4. 源文件级历史表现追踪（供 compressor 评分加成）
//   5. 跨分支关联检测
// ============================================================

import type {
  ExecutionRecord, TurboContextConfig, QualityDimensions,
  TaskType, BranchState, BranchTrajectory, SourceMemory, FailureRecord,
  StrategyMutation, EvolutionExperiment, StrategyEvolutionData, TrialLogEntry,
  ExperimentRun, ExperimentLogEntry, UnifiedMetric, Mission,
  PlateauSignal, PlateauReason, PlateauRuleResult,
  StrategicDirective, ContrastivePair, IDFCache,
  ConsolidationAttribution, VerificationRecord,
  ConsolidationUndoInfo, RLExecutionRecord,
} from "../types.js";
import { DEFAULT_RETRIEVAL_WEIGHTS } from "../types.js";
import { mkdirSync, existsSync, readFileSync, writeFileSync } from "fs";
import { join } from "path";
import { homedir } from "os";

// v3.1+ — extracted subsystems
import {
  detectPlateau as detectPlateauImpl,
  generateStrategicDirective as generateStrategicDirectiveImpl,
  computeAdaptiveMmrLambda,
  findContrastivePairs as findContrastivePairsImpl,
  extractRecordFeatures,
  synthesizeFutureDirections as synthesizeFutureDirectionsImpl,
  updateIDFCache as updateIDFCacheImpl,
  createDefaultIDFCache,
} from "./retrieval-system.js";
import {
  getActiveMutation as getActiveMutationImpl,
  proposeMutation as proposeMutationImpl,
  recordTrial as recordTrialImpl,
  recordTrialCrash as recordTrialCrashImpl,
  decideKeepDiscard as decideKeepDiscardImpl,
  getCanonicalMutations as getCanonicalMutationsImpl,
  resetCanonicalStrategy as resetCanonicalStrategyImpl,
  getEvolutionStats as getEvolutionStatsImpl,
  getTrialLog as getTrialLogImpl,
} from "./evolution-engine.js";
import { RLFeedbackEngine } from "./rl-feedback-engine.js";
import { consolidateMemories, computeSubsystemBaselines } from "./rl-core.js";
import { twoPhaseCausalRetrieval } from "./retrieval-system.js";
import { learnFromHistory as learnFromHistoryFn } from "./learner-optimizer.js";
import type { LearningAdjustment } from "./learner-optimizer.js";

/** 所有可用的任务类型分支 */
const ALL_BRANCHES: TaskType[] = [
  "code_generation", "code_review", "code_refactor",
  "debugging", "testing", "analysis", "design",
  "documentation", "general",
];

/** 分支族系：相似的任务类型共享相近的最优参数 */
const BRANCH_FAMILIES: Record<string, TaskType[]> = {
  "generation": ["code_generation", "code_refactor"],
  "analysis": ["analysis", "code_review", "debugging"],
  "structural": ["design", "documentation"],
  "verification": ["testing"],
  "fallback": ["general"],
};

/** 学习器 */
export class Learner {
  private globalHistory: ExecutionRecord[] = [];
  private branches: Map<TaskType, BranchState> = new Map();
  private sourceMemory: Map<string, SourceMemory> = new Map();
  /** v3.8: Ablation history for causal graph construction */
  // ablationHistory removed — causal-graph.ts deleted (never trained on real data)
  private config: TurboContextConfig;
  private readonly MAX_HISTORY = 200;
  private readonly SUMMARY_INTERVAL = 5;
  private readonly EVOLUTION_TRIAL_SIZE = 5; // 每个 trial 跑 N 次后比较
  private readonly statePath: string;

  // 自进化系统（v2.2 → v2.4: autoresearch 深化）
  private evolution: StrategyEvolutionData = {
    experiments: [],
    currentExperimentId: null,
    totalExperiments: 0,
    keptCount: 0,
    discardedCount: 0,
    canonicalStrategies: {},
    trialLog: [],
  };
  private activeMutation: Map<TaskType, { experimentId: string; trialBaselines: number[] }> = new Map();

  // v3.1 — 全局 IDF 缓存（autoresearch: IDF-weighted retrieval）
  private idfCache: IDFCache = createDefaultIDFCache();

  // v3.4 — RL engine (extracted from learner, owns all RL state+logic)
  private rlEngine: RLFeedbackEngine = new RLFeedbackEngine();

  // v3.9 — Cold storage: archived low-utility memories (agent.py cold_storage)
  private coldStorage: ExecutionRecord[] = [];
  private readonly MAX_COLD_STORAGE = 500;
  private readonly coldStoragePath: string;

  // v3.9 — Consolidation attribution history (agent.py consolidation_attributions)
  private consolidationHistory: ConsolidationAttribution[] = [];

  // v3.9 — Adversarial verification audit trail (agent.py verification_history)
  private verificationHistory: VerificationRecord[] = [];
  private lastAdversarialVerificationAt: number = 0;
  private _lastColdStorageAt: number = 0;

  constructor(initialConfig: TurboContextConfig, statePath?: string) {
    this.config = { ...initialConfig };
    this.statePath = statePath || join(homedir(), ".turbocontext", "state.json");
    this.coldStoragePath = join(homedir(), ".turbocontext", "cold_storage.json");
    this.initBranches();

    // Wire RL engine to Learner state providers
    this.rlEngine.setProviders({
      sourceMemory: () => new Map(this.sourceMemory),
      branchBestQuality: (tt: TaskType) => this.branches.get(tt)?.bestQuality ?? 0.85,
      branchThreshold: (tt: TaskType) => this.getBranchQualityThreshold(tt),
      maxAttempts: () => this.config.maxAttempts,
    });

    this.load();
  }

  /** 初始化所有分支 */
  private initBranches(): void {
    for (const type of ALL_BRANCHES) {
      this.branches.set(type, this.createBranch(type));
    }
  }

  private createBranch(type: TaskType, source?: BranchState): BranchState {
    return {
      type,
      totalExperiments: source?.totalExperiments ?? 0,
      bestQuality: source?.bestQuality ?? 0,
      bestDescription: source?.bestDescription ?? "",
      successCount: source?.successCount ?? 0,
      failureCount: source?.failureCount ?? 0,
      trajectory: source?.trajectory ?? {
        momentum: 0,
        improvementVelocity: 0,
        stabilityScore: 0.5,
        noveltyScore: 0.5,
        qualityHistory: [],
        recentTaskIds: [],
      },
      recentFailures: source?.recentFailures ?? [],
      summary: source?.summary ?? "",
      lastSummaryExperimentCount: source?.lastSummaryExperimentCount ?? 0,
      alphaOverride: null,
      betaOverride: null,
      gammaOverride: null,
      qualityThresholdOverride: null,
      temperatureScheduleOverride: null,
    };
  }

  // ------------------------------------------------------------------
  // Recording
  // ------------------------------------------------------------------

  /**
   * 记录一次执行到对应分支
   */
  record(execution: ExecutionRecord): void {
    this.globalHistory.push(execution);
    if (this.globalHistory.length > this.MAX_HISTORY) {
      this.globalHistory.shift();
    }

    // 更新对应分支
    this.updateBranch(execution);

    // 更新源文件历史表现
    if (execution.sourceFiles && execution.sourceFiles.length > 0) {
      this.updateSourceMemory(execution, execution.sourceFiles);
    }

    // v5: RL feedback is now handled by RLEngineV5.recordTrial() in the engine.
    // The V4 RLFeedbackEngine is retained as a read-only query layer for
    // getRetrievalContext(), getSourceBoostRL(), getCausalBoost(), etc.
    // These query methods will be incrementally migrated to V5 (strangler fig pattern).

    this.save();
  }

  /**
   * 更新分支状态
   */
  private updateBranch(execution: ExecutionRecord): void {
    const type = execution.taskType;
    let branch = this.branches.get(type);
    if (!branch) {
      branch = this.createBranch(type);
      this.branches.set(type, branch);
    }

    branch.totalExperiments++;
    const isSuccess = execution.qualityScore >= (branch.qualityThresholdOverride ?? this.config.qualityThreshold);
    if (isSuccess) {
      branch.successCount++;
      if (execution.qualityScore > branch.bestQuality) {
        branch.bestQuality = execution.qualityScore;
        branch.bestDescription = `bpb=${execution.coverage ? Object.values(execution.coverage)[0] : '?'}`;
      }
    } else {
      branch.failureCount++;
      branch.recentFailures.push({
        taskId: execution.taskId,
        timestamp: execution.timestamp,
        qualityScore: execution.qualityScore,
        attemptCount: execution.attemptCount,
        failureReasons: this.extractFailureReasons(execution),
      });
      branch.recentFailures = branch.recentFailures.slice(-10);
    }

    // 轨迹更新
    const traj = branch.trajectory;
    traj.qualityHistory.push(execution.qualityScore);
    if (traj.qualityHistory.length > 15) traj.qualityHistory = traj.qualityHistory.slice(-15);
    traj.recentTaskIds.push(execution.taskId);
    if (traj.recentTaskIds.length > 10) traj.recentTaskIds = traj.recentTaskIds.slice(-10);

    // momentum: 最近 5 个质量评分的平均变化率
    const recent = traj.qualityHistory.slice(-5);
    if (recent.length >= 3) {
      const vel = (recent[recent.length - 1] - recent[0]) / recent.length;
      traj.improvementVelocity = Math.round(vel * 10000) / 10000;
      traj.momentum = Math.round(vel * 10000) / 10000;
    }

    // stability: (successes - failures) / total, clamped [0, 1]
    const total = branch.successCount + branch.failureCount;
    if (total > 0) {
      const raw = (branch.successCount - branch.failureCount) / total;
      traj.stabilityScore = Math.round(Math.max(0, Math.min(1, raw)) * 100) / 100;
    }

    // novelty: 最近任务描述的去重率
    const descs = execution.coverage ? Object.keys(execution.coverage) : [];
    if (traj.recentTaskIds.length >= 2) {
      const unique = new Set(traj.recentTaskIds);
      traj.noveltyScore = Math.round((unique.size / traj.recentTaskIds.length) * 100) / 100;
    }

    // 触发分支总结
    if (branch.totalExperiments >= this.SUMMARY_INTERVAL &&
        branch.totalExperiments - branch.lastSummaryExperimentCount >= this.SUMMARY_INTERVAL) {
      this.generateBranchSummary(type);
      branch.lastSummaryExperimentCount = branch.totalExperiments;
    }
  }

  /**
   * 更新源文件历史表现
   */
  private updateSourceMemory(execution: ExecutionRecord, sources: string[]): void {
    const isSuccess = execution.qualityScore >= this.config.qualityThreshold;
    const taskType = execution.taskType;
    for (const src of sources) {
      const existing = this.sourceMemory.get(src) || { attempts: 0, successes: 0, lastQuality: 0, lastUsed: 0 };
      existing.attempts++;
      if (isSuccess) existing.successes++;
      existing.lastQuality = execution.qualityScore;
      existing.lastUsed = execution.timestamp;
      // v3.8: Per-task-type conditional stats (Causal Markov Condition)
      if (!existing.perType) existing.perType = {};
      const pt = existing.perType[taskType] || { attempts: 0, successes: 0, lastQuality: 0 };
      pt.attempts++;
      if (isSuccess) pt.successes++;
      pt.lastQuality = execution.qualityScore;
      existing.perType[taskType] = pt;
      this.sourceMemory.set(src, existing);
    }
  }

  /**
   * v3.6: Record a per-file ablation result.
   *
   * Stores the clean causal signal as an EMA on SourceMemory.
   * α=0.3 balances new evidence against existing belief — a single ablation
   * is noisy (simulated output), multiple ablations converge to truth.
   */
  recordAblation(result: import("../types.js").AblationResult): void {
    const existing = this.sourceMemory.get(result.sourceFile) || {
      attempts: 0, successes: 0, lastQuality: 0, lastUsed: 0,
    };
    // Global EMA
    const prior = existing.ablatedCausalUtility ?? 0;
    existing.ablatedCausalUtility = Math.round((prior + 0.3 * (result.causalDelta - prior)) * 10000) / 10000;
    existing.ablationCount = (existing.ablationCount ?? 0) + 1;
    // v3.8: Per-type conditional ablation (Causal Markov Condition)
    if (!existing.perType) existing.perType = {};
    const pt = existing.perType[result.taskType] || { attempts: 0, successes: 0, lastQuality: 0 };
    const ptPrior = pt.ablatedCausalUtility ?? 0;
    pt.ablatedCausalUtility = Math.round((ptPrior + 0.3 * (result.causalDelta - ptPrior)) * 10000) / 10000;
    pt.ablationCount = (pt.ablationCount ?? 0) + 1;
    existing.perType[result.taskType] = pt;
    this.sourceMemory.set(result.sourceFile, existing);
    // Causal graph construction removed (never trained on real ablation data).
    // Source memory tracking retained — it's used by getCausalBoost().
  }

  private extractFailureReasons(execution: ExecutionRecord): string[] {
    const reasons: string[] = [];
    if (execution.attemptCount >= this.config.maxAttempts) {
      reasons.push("retries_exhausted");
    }
    if (execution.dimensionScores) {
      for (const [dim, score] of Object.entries(execution.dimensionScores)) {
        if (score < 0.5) reasons.push(`low_${dim}`);
      }
    }
    return reasons.length > 0 ? reasons : ["unknown"];
  }

  // ------------------------------------------------------------------
  // Branch Summary
  // ------------------------------------------------------------------

  /**
   * 为指定分支生成总结
   */
  private generateBranchSummary(type: TaskType): void {
    const branch = this.branches.get(type);
    if (!branch || branch.totalExperiments === 0) return;

    const lines: string[] = [];
    const total = branch.totalExperiments;

    lines.push(`Branch: ${type}`);
    if (branch.bestQuality > 0) {
      lines.push(`Best quality: ${(branch.bestQuality * 100).toFixed(1)}%`);
    }

    const sr = (branch.successCount / total * 100).toFixed(0);
    const fr = (branch.failureCount / total * 100).toFixed(0);
    lines.push(`Record: ${branch.successCount}/${total} pass (${sr}%), ${fr}% fail`);

    const vel = branch.trajectory.improvementVelocity;
    if (vel > 0.01) {
      lines.push(`Trend: improving (+${(vel * 100).toFixed(2)}%/exp)`);
    } else if (vel < -0.01) {
      lines.push(`Trend: declining (${(vel * 100).toFixed(2)}%/exp)`);
    } else {
      lines.push(`Trend: stable`);
    }
    lines.push(`Stability: ${(branch.trajectory.stabilityScore * 100).toFixed(0)}%`);

    // 重复失败模式
    const modeCounts = new Map<string, number>();
    for (const f of branch.recentFailures) {
      for (const reason of f.failureReasons) {
        modeCounts.set(reason, (modeCounts.get(reason) || 0) + 1);
      }
    }
    const repeated = [...modeCounts.entries()].filter(([, c]) => c >= 2);
    if (repeated.length > 0) {
      lines.push(`Repeated failures: ${repeated.map(([r, c]) => `${r} (${c}x)`).join(", ")}`);
    }

    // 收益递减
    if (total >= 8 && Math.abs(vel) < 0.005) {
      lines.push("⚠ Plateau — consider switching task type or approach");
    }

    branch.summary = lines.join("\n");
  }

  // ------------------------------------------------------------------
  // Learning
  // ------------------------------------------------------------------

  /**
   * 执行学习步骤，返回更新后的配置
   * 现在根据各分支状况分别调整，再汇总为全局最优
   */
  learn(): { config: TurboContextConfig; adjustments: LearningAdjustment[] } {
    const adjustments: LearningAdjustment[] = [];
    const n = this.globalHistory.length;
    if (n < 5) return { config: this.config, adjustments };

    const recent = this.globalHistory.slice(-50);

    // v6: Delegate parameter optimization to extracted learner-optimizer module
    const branchThresholds = new Map<string, { mean: number; count: number }>();
    for (const [type, branch] of this.branches) {
      branchThresholds.set(type, { mean: branch.avgQuality, count: branch.totalExperiments });
    }
    const optResult = learnFromHistoryFn(this.config, recent, branchThresholds);
    adjustments.push(...optResult.adjustments);

    // --- RL enhancements (stateful, stay in Learner) ---
    // Adversarial verification
    const verified = this.runAdversarialVerification();
    if (verified > 0) {
      adjustments.push({
        component: "adversarial_verification",
        changes: { verified: String(verified) },
        reason: `Adversarially verified ${verified} old success memories against current baseline`,
      });
    }

    // --- v3.9: Memory consolidation with attribution tracking ---
    // 6. Consolidate old low-utility records into summary entries
    // v4.0: Track pre/post subsystem coverage for attribution audit
    const preCoverageTypes = new Set(this.globalHistory.map(r => r.taskType));
    const consolidationResult = consolidateMemories(
      this.globalHistory as any[],
      60,
      20,
      () => this.globalHistory.length + 1,
    );
    if (consolidationResult.consolidatedCount > 0) {
      // v4.0: Compute post-consolidation coverage and detect losses
      const postCoverageTypes = new Set(this.globalHistory.map(r => r.taskType));
      const coverageLost: string[] = [];
      for (const t of preCoverageTypes) {
        if (!postCoverageTypes.has(t)) coverageLost.push(t);
      }

      // Build attribution record with v4 enhanced metadata
      const attribution: ConsolidationAttribution = {
        timestamp: new Date().toISOString(),
        preCoverageCount: preCoverageTypes.size,
        postCoverageCount: postCoverageTypes.size,
        coverageLoss: coverageLost,
        totalTokensSaved: consolidationResult.tokensSaved,
        groups: consolidationResult.groups || [],
      };
      this.consolidationHistory.push(attribution);
      if (this.consolidationHistory.length > 20) {
        this.consolidationHistory = this.consolidationHistory.slice(-20);
      }
      adjustments.push({
        component: "memory_consolidation",
        changes: {
          consolidated: String(consolidationResult.consolidatedCount),
          tokensSaved: String(consolidationResult.tokensSaved),
          coverageLoss: String(coverageLost.length),
        },
        reason: `Consolidated ${consolidationResult.consolidatedCount} old records across ${consolidationResult.groups?.length || 0} groups, saved ~${consolidationResult.tokensSaved} tokens`,
      });
    }

    // --- v3.9: Cold storage — archive persistently low-utility memories ---
    // 7. Move cold (never-referenced, low-utility) records to disk archive
    const curriculumCtx = this.rlEngine.getCurriculumContext();
    const forgetInterval = curriculumCtx.phase >= 3 ? 8 : 15;
    if (this.rlEngine.curriculumTotal - (this._lastColdStorageAt ?? 0) >= forgetInterval) {
      const archived = this.archiveColdMemories();
      if (archived > 0) {
        adjustments.push({
          component: "cold_storage",
          changes: { archived: String(archived) },
          reason: `Archived ${archived} cold memories to disk (adaptive forget interval=${forgetInterval})`,
        });
      }
      this._lastColdStorageAt = this.rlEngine.curriculumTotal;
    }

    return { config: this.config, adjustments };
  }

  /**
   * 分支级阈值学习
   * 每个分支根据其成功/失败历史调整 qualityThresholdOverride
   */
  private learnBranchThresholds(recent: ExecutionRecord[]): LearningAdjustment[] {
    const adjs: LearningAdjustment[] = [];

    for (const [type, branch] of this.branches) {
      if (branch.totalExperiments < 3) continue;

      const branchRecords = recent.filter(r => r.taskType === type);
      if (branchRecords.length < 3) continue;

      const avgQuality = branchRecords.reduce((s, r) => s + r.qualityScore, 0) / branchRecords.length;
      const passCount = branchRecords.filter(r => r.qualityScore >= this.config.qualityThreshold).length;
      const passRate = passCount / branchRecords.length;

      // 分支表现稳定且良好 → 可适当提高阈值
      if (passRate > 0.85 && branch.trajectory.stabilityScore > 0.7) {
        const oldVal = branch.qualityThresholdOverride;
        branch.qualityThresholdOverride = Math.min(0.95,
          (branch.qualityThresholdOverride ?? this.config.qualityThreshold) + 0.02);
        adjs.push({
          component: `branch_threshold_${type}`,
          changes: {
            threshold: `${(oldVal ?? this.config.qualityThreshold * 100).toFixed(0)}% → ${(branch.qualityThresholdOverride * 100).toFixed(0)}%`,
            passRate: `${(passRate * 100).toFixed(0)}%`,
            stability: branch.trajectory.stabilityScore.toFixed(2),
          },
          reason: `${type} branch: high pass rate (${(passRate * 100).toFixed(0)}%), increasing threshold`,
        });
      }

      // 分支表现不稳定 → 降低阈值
      if (passRate < 0.5 || branch.trajectory.stabilityScore < 0.3) {
        const oldVal = branch.qualityThresholdOverride;
        branch.qualityThresholdOverride = Math.max(0.6,
          (branch.qualityThresholdOverride ?? this.config.qualityThreshold) - 0.03);
        adjs.push({
          component: `branch_threshold_${type}`,
          changes: {
            threshold: `${(oldVal ?? this.config.qualityThreshold * 100).toFixed(0)}% → ${(branch.qualityThresholdOverride * 100).toFixed(0)}%`,
            passRate: `${(passRate * 100).toFixed(0)}%`,
            stability: branch.trajectory.stabilityScore.toFixed(2),
          },
          reason: `${type} branch: low pass rate (${(passRate * 100).toFixed(0)}%), lowering threshold`,
        });
      }
    }

    return adjs;
  }

  private learnCompressionWeights(recent: ExecutionRecord[]): LearningAdjustment | null {
    const highQuality = recent.filter(r => r.qualityScore >= 0.85);
    const lowQuality = recent.filter(r => r.qualityScore < 0.70);
    if (highQuality.length < 3 || lowQuality.length < 3) return null;

    const avgHighCompression = highQuality.reduce((s, r) => s + r.compressionRatio, 0) / highQuality.length;
    const avgLowCompression = lowQuality.reduce((s, r) => s + r.compressionRatio, 0) / lowQuality.length;

    const oldAlpha = this.config.alpha;
    const oldBeta = this.config.beta;
    const oldGamma = this.config.gamma;

    if (avgHighCompression - avgLowCompression > 0.2) {
      this.config.gamma = Math.min(0.40, this.config.gamma + 0.02);
      this.config.alpha = Math.max(0.35, this.config.alpha - 0.01);
    }
    if (avgLowCompression < 0.3) {
      this.config.alpha = Math.min(0.70, this.config.alpha + 0.02);
      this.config.gamma = Math.max(0.15, this.config.gamma - 0.01);
    }

    const total = this.config.alpha + this.config.beta + this.config.gamma;
    this.config.alpha /= total;
    this.config.beta /= total;
    this.config.gamma /= total;

    return {
      component: "compression_weights",
      changes: {
        alpha: `${oldAlpha.toFixed(3)} → ${this.config.alpha.toFixed(3)}`,
        beta: `${oldBeta.toFixed(3)} → ${this.config.beta.toFixed(3)}`,
        gamma: `${oldGamma.toFixed(3)} → ${this.config.gamma.toFixed(3)}`,
      },
      reason: `高质量压缩比 ${(avgHighCompression * 100).toFixed(0)}%, 低质量 ${(avgLowCompression * 100).toFixed(0)}%`,
    };
  }

  private learnComplexityThresholds(recent: ExecutionRecord[]): LearningAdjustment | null {
    if (recent.length < 10) return null;
    const oldLow = this.config.complexityThresholdLow;
    const oldHigh = this.config.complexityThresholdHigh;
    const byModel: Record<string, ExecutionRecord[]> = {};

    for (const r of recent) {
      (byModel[r.modelUsed] ||= []).push(r);
    }

    const fastRecords = byModel["fast"] || [];
    if (fastRecords.length >= 3) {
      const passRate = fastRecords.filter(r => r.qualityScore >= 0.8).length / fastRecords.length;
      if (passRate > 0.9) this.config.complexityThresholdLow = Math.min(0.45, this.config.complexityThresholdLow + 0.03);
      else if (passRate < 0.7) this.config.complexityThresholdLow = Math.max(0.20, this.config.complexityThresholdLow - 0.03);
    }

    const deepRecords = byModel["deep"] || [];
    if (deepRecords.length >= 3) {
      const failRate = deepRecords.filter(r => r.qualityScore < 0.8).length / deepRecords.length;
      if (failRate > 0.3) this.config.complexityThresholdHigh = Math.min(0.85, this.config.complexityThresholdHigh + 0.03);
    }

    return {
      component: "complexity_thresholds",
      changes: {
        thresholdLow: `${oldLow.toFixed(2)} → ${this.config.complexityThresholdLow.toFixed(2)}`,
        thresholdHigh: `${oldHigh.toFixed(2)} → ${this.config.complexityThresholdHigh.toFixed(2)}`,
      },
      reason: `基于 ${recent.length} 条历史记录`,
    };
  }

  private learnTemperatureSchedule(recent: ExecutionRecord[]): LearningAdjustment | null {
    if (recent.length < 5) return null;
    const avgAttempts = recent.reduce((s, r) => s + r.attemptCount, 0) / recent.length;

    if (avgAttempts <= 1.1) {
      this.config.temperatureSchedule[0] = Math.max(0.3, this.config.temperatureSchedule[0] - 0.05);
    } else if (avgAttempts >= 2.5) {
      this.config.temperatureSchedule[0] = Math.min(0.9, this.config.temperatureSchedule[0] + 0.05);
    }

    return {
      component: "temperature_schedule",
      changes: {
        newSchedule: `[${this.config.temperatureSchedule.map(t => t.toFixed(2)).join(", ")}]`,
        avgAttempts: avgAttempts.toFixed(2),
      },
      reason: `平均尝试次数 ${avgAttempts.toFixed(2)}`,
    };
  }

  // ------------------------------------------------------------------
  // Query API（供 compressor / index 使用）
  // ------------------------------------------------------------------

  /**
   * 获取指定分支的格式化总结
   */
  getBranchSummary(taskType: TaskType): string {
    const branch = this.branches.get(taskType);
    if (!branch || branch.totalExperiments === 0) {
      return `  Branch '${taskType}': no experiments yet`;
    }
    const velSign = branch.trajectory.improvementVelocity >= 0 ? "+" : "";
    const lines = [
      `  Active branch: ${taskType}`,
      `  Experiments: ${branch.totalExperiments} | Best quality: ${(branch.bestQuality * 100).toFixed(1)}%`,
      `  Momentum: ${velSign}${(branch.trajectory.momentum * 100).toFixed(2)}%/exp | Stability: ${(branch.trajectory.stabilityScore * 100).toFixed(0)}% | Novelty: ${(branch.trajectory.noveltyScore * 100).toFixed(0)}%`,
    ];
    if (branch.summary) {
      lines.push(`  ${branch.summary}`);
    }
    return lines.join("\n");
  }

  /**
   * 获取源文件的历史表现加成（供 compressor 使用）
   * 返回 0-0.1 的加成系数
   */
  getSourceBoost(source: string): number {
    const mem = this.sourceMemory.get(source);
    if (!mem || mem.attempts < 2) return 0;
    const successRate = mem.successes / mem.attempts;
    // 成功率 > 70% 的文件获得正加成
    if (successRate > 0.7) return Math.min(0.1, successRate * 0.1);
    // 成功率 < 40% 的文件获得负加成
    if (successRate < 0.4) return -0.05;
    return 0;
  }

  /**
   * 获取关联分支（同一族系 + 共享任务模式的分支）
   */
  getRelatedBranches(taskType: TaskType): Array<{ type: TaskType; overlap: number }> {
    const related: Array<{ type: TaskType; overlap: number }> = [];
    const branch = this.branches.get(taskType);
    if (!branch) return related;

    // 族系关联
    for (const [, members] of Object.entries(BRANCH_FAMILIES)) {
      if (members.includes(taskType)) {
        for (const m of members) {
          if (m !== taskType) related.push({ type: m, overlap: 3 });
        }
      }
    }

    // 历史重叠：共享失败模式的关联
    const failureSet = new Set(branch.recentFailures.map(f => JSON.stringify(f.failureReasons)));
    for (const [type, other] of this.branches) {
      if (type === taskType) continue;
      const overlap = other.recentFailures.filter(f =>
        f.failureReasons.some(r => failureSet.has(JSON.stringify([r])))
      ).length;
      if (overlap > 0) {
        const existing = related.find(r => r.type === type);
        if (existing) existing.overlap += overlap;
        else related.push({ type, overlap });
      }
    }

    return related.sort((a, b) => b.overlap - a.overlap).slice(0, 3);
  }

  /**
   * 获取指定分支的动态阈值（优先使用分支覆盖）
   */
  getBranchQualityThreshold(taskType: TaskType): number {
    const branch = this.branches.get(taskType);
    return branch?.qualityThresholdOverride ?? this.config.qualityThreshold;
  }

  /**
   * 获取所有分支的快照
   */
  getBranches(): Map<TaskType, BranchState> {
    return new Map(this.branches);
  }

  /**
   * 获取有活跃实验的分支列表
   */
  getActiveBranches(): TaskType[] {
    return ALL_BRANCHES.filter(t => {
      const b = this.branches.get(t);
      return b && b.totalExperiments > 0;
    });
  }

  /** v3.6: Get source memory map (for ablation target selection). */
  getSourceMemory(): Map<string, SourceMemory> {
    return this.sourceMemory;
  }

  /**
   * v3.8 → v4.1: Build causal graph using ensemble discovery (PC-stable + GES).
   *
   * Causal graph module removed — never trained on real ablation data.
   * Returns null; compressor uses V5's getCausalBoost() (direct ablation measurement).
   */
  getCausalGraph(): null {
    return null;
  }

  /**
   * Ablation target selection removed — causal graph + do-calculus never ran on real data.
   * Use V5 RLEngineV5.getCausalBoost() for source file utility measurement.
   */
  getAblationTargetSGS(): { target: string | null; reason: string } {
    return { target: null, reason: "ablation engine removed — use V5 getCausalBoost" };
  }

  /** Ablation history removed — causal-graph.ts + ablation-engine.ts deleted. */
  getAblationHistory(): import("../types.js").AblationResult[] {
    return [];
  }

  // ------------------------------------------------------------------
  // v3.4 — RL Feedback (delegated to RLFeedbackEngine)
  // ------------------------------------------------------------------

  /**
   * RL-powered source boost (Thompson Sampling).
   *
   * v3.6: When an ablation result exists for this file, the ablated causal
   * utility acts as a prior center — shifting the score toward the true
   * causal value rather than the noisy correlational success rate.
   */
  getSourceBoostRL(source: string): number {
    const mem = this.sourceMemory.get(source);
    // If we have a clean causal signal from ablation, blend it in
    if (mem?.ablatedCausalUtility !== undefined && (mem.ablationCount ?? 0) > 0) {
      const tsBoost = this.rlEngine.getSourceBoostRL(source);
      // 60% ablation causal signal + 40% Thompson Sampling correlation
      return Math.round((mem.ablatedCausalUtility * 0.6 + tsBoost * 0.4) * 10000) / 10000;
    }
    return this.rlEngine.getSourceBoostRL(source);
  }

  /**
   * v3.7: Get causal utility multiplier for retrieval scoring (Phase 2 re-rank).
   *
   * Returns a multiplier in [0.5, 1.5]:
   *   - >1.0: file is causally more valuable than similarity suggests → boost
   *   - 1.0: no causal data → neutral
   *   - <1.0: file is less valuable than similarity suggests → penalize
   *
   * Blend: 70% clean ablation signal + 30% correlational success rate.
   * Default 1.0 when insufficient data.
   */
  /**
   * v3.8: Conditional causal utility (Causal Markov Condition).
   *
   * Key insight from SGS: causal effect of a file depends on the task type.
   * auth.ts is vital in code_review (it IS the target), irrelevant in documentation.
   * Per-type stats capture this conditional structure.
   */
  getCausalBoost(source: string, taskType: TaskType): number {
    const mem = this.sourceMemory.get(source);
    if (!mem || mem.attempts < 2) return 1.0;

    // v3.8: Prefer per-type stats when available (Causal Markov Condition)
    const pt = mem.perType?.[taskType];
    const hasPerType = pt && pt.attempts >= 2;

    // Attempts/successes: condition on task type if possible
    const effAttempts = hasPerType ? pt!.attempts : mem.attempts;
    const effSuccesses = hasPerType ? pt!.successes : mem.successes;

    // Ablation signal: per-type preferred, fall back to global
    const perTypeAblation = hasPerType && pt!.ablatedCausalUtility !== undefined && (pt!.ablationCount ?? 0) > 0;
    const globalAblation = mem.ablatedCausalUtility !== undefined && (mem.ablationCount ?? 0) > 0;
    const hasAblation = perTypeAblation || globalAblation;
    const ablationSignal = perTypeAblation
      ? pt!.ablatedCausalUtility!
      : globalAblation
        ? mem.ablatedCausalUtility!
        : undefined;

    // Correlational signal from success rate (conditioned on task type)
    const successRate = effSuccesses / Math.max(1, effAttempts);
    // v3.8: Weight per-type correlation higher (less noisy than global)
    const correlationWeight = hasPerType ? 0.4 : 0.3;
    const correlationSignal = (successRate - 0.5) * (correlationWeight * 2); // [-0.3, +0.3] or [-0.4, +0.4]

    // Blend: 70% ablation + 30% correlation (with per-type boost)
    const ablationWeight = hasPerType ? 0.75 : 0.7;
    const blendedSignal = hasAblation
      ? (ablationSignal ?? 0) * ablationWeight + correlationSignal * (1 - ablationWeight)
      : correlationSignal;

    const multiplier = 1.0 + blendedSignal * 0.5;
    return Math.round(Math.max(0.5, Math.min(1.5, multiplier)) * 10000) / 10000;
  }

  /** Retrieval strategy state */
  getRetrievalStrategy() {
    return this.rlEngine.getRetrievalStrategy();
  }

  /** Predictive model diagnostics */
  getPredictiveModelStats() {
    return this.rlEngine.getPredictiveModelStats();
  }

  /** Current curriculum phase and params */
  getCurriculumContext() {
    return this.rlEngine.getCurriculumContext();
  }

  /** Propose retrieval strategy mutation */
  proposeRetrievalMutation() {
    return this.rlEngine.proposeRetrievalMutation();
  }

  /** RL Feedback Engine (for serialization and external access) */
  getRLEngine(): RLFeedbackEngine {
    return this.rlEngine;
  }

  /** Run adversarial verification of old successful memories */
  runAdversarialVerification(): number {
    if (this.rlEngine.curriculumTotal < 15) return 0;
    const successes = this.globalHistory.filter(
      r => r.qualityScore >= this.config.qualityThreshold
    );
    if (successes.length < 5) return 0;

    const currentAvg = successes.reduce((s, r) => s + r.qualityScore, 0) / successes.length;
    const currentBest = Math.max(...successes.map(r => r.qualityScore));
    let verified = 0;

    for (let i = 0; i < Math.min(this.globalHistory.length - 10, 5); i++) {
      const rec = this.globalHistory[i];
      if (rec.qualityScore < this.config.qualityThreshold) continue;
      const age = this.globalHistory.length - 1 - i;
      if (age < 10) continue;

      const avgGap = (rec.qualityScore - currentAvg) / Math.max(Math.abs(currentAvg), 0.001);
      if (avgGap < -0.02 && rec.sourceFiles) {
        for (const src of rec.sourceFiles) {
          const mem = this.sourceMemory.get(src);
          if (mem && mem.successes > 0) {
            const penalty = Math.min(3, Math.ceil(Math.abs(avgGap) * 10));
            mem.successes = Math.max(0, mem.successes - penalty);
            mem.lastQuality = Math.round((mem.lastQuality * 0.7 + currentAvg * 0.3) * 10000) / 10000;
          }
        }
        // v4.0: Also downgrade RL utility fields on the record itself
        // agent.py v4 lines 2136-2144: downgrade confidence, retrieval_utility, alpha_ts
        const rlRec = rec as Partial<RLExecutionRecord>;
        if (rlRec.causalUtility !== undefined) {
          rlRec.causalUtility = Math.max(0.2, (rlRec.causalUtility ?? 0.5) * 0.8);
        }
        if (rlRec.retrievalUtility !== undefined) {
          rlRec.retrievalUtility = Math.max(0.1, (rlRec.retrievalUtility ?? 0.5) * 0.75);
        }
        if (rlRec.thompsonAlpha !== undefined) {
          rlRec.thompsonAlpha = Math.max(0.5, (rlRec.thompsonAlpha ?? 1.0) * 0.7);
        }
        verified++;

        // v3.9: Record verification for audit trail
        this.verificationHistory.push({
          experimentCount: this.globalHistory.length,
          currentBest: Math.round(currentBest * 10000) / 10000,
          currentAvg: Math.round(currentAvg * 10000) / 10000,
          gapToBest: Math.round((currentBest - rec.qualityScore) * 10000) / 10000,
          newConfidence: Math.round((rec.qualityScore / Math.max(currentBest, 0.001)) * 10000) / 10000,
          timestamp: new Date().toISOString(),
        });
      } else if (avgGap >= 0 && rec.sourceFiles) {
        // v4.0: Competitive old success — boost (adversarial test PASSED)
        // agent.py v4 lines 2149-2152
        const rlRec = rec as Partial<RLExecutionRecord>;
        if (rlRec.causalUtility !== undefined) {
          rlRec.causalUtility = Math.min(0.95, (rlRec.causalUtility ?? 0.5) * 1.05);
        }
        if (rlRec.thompsonAlpha !== undefined) {
          rlRec.thompsonAlpha = Math.min(10, (rlRec.thompsonAlpha ?? 1.0) * 1.1);
        }
      }
    }
    // Bound verification history
    if (this.verificationHistory.length > 100) {
      this.verificationHistory = this.verificationHistory.slice(-50);
    }
    return verified;
  }

  /**
   * v3.9: Archive persistently low-utility memories to cold storage (disk).
   *
   * Adapted from agent.py _maybe_archive_cold_memories.
   * A memory goes to cold storage when:
   *   - It is among the oldest 30% of records
   *   - Its qualityScore is below the branch average
   *   - Its source files have been superseded by better results
   *
   * Cold storage is excluded from normal retrieval but queryable on demand.
   */
  private archiveColdMemories(): number {
    if (this.globalHistory.length < 20) return 0;

    let archived = 0;
    const keep: ExecutionRecord[] = [];

    for (const rec of this.globalHistory) {
      const branch = this.branches.get(rec.taskType);
      if (!branch) { keep.push(rec); continue; }

      const branchAvg = branch.bestQuality > 0 ? branch.bestQuality * 0.8 : this.config.qualityThreshold * 0.7;
      const isOld = this.globalHistory.indexOf(rec) < this.globalHistory.length * 0.3;
      const isLowUtility = rec.qualityScore < branchAvg;
      const hasBetterVersion = rec.sourceFiles?.some(src => {
        const mem = this.sourceMemory.get(src);
        return mem && mem.lastQuality > rec.qualityScore + 0.1;
      }) ?? false;

      if (isOld && isLowUtility && (hasBetterVersion || rec.attemptCount >= this.config.maxAttempts)) {
        // v4.0: Store undo log before archiving (agent.py _consolidation_undo_info)
        // Karpathy: "Compression with attribution — know what was lost."
        const undoInfo: ConsolidationUndoInfo = {
          originalId: rec.taskId,
          mergedInto: null,  // cold storage, not merged
          originalHypothesis: rec.taskId.slice(0, 200),
          originalQualityScore: rec.qualityScore,
          consolidatedAt: new Date().toISOString(),
        };
        (rec as any)._consolidation_undo_info = undoInfo;
        this.coldStorage.push(rec);
        archived++;
      } else {
        keep.push(rec);
      }
    }

    if (archived > 0) {
      this.globalHistory.length = 0;
      this.globalHistory.push(...keep);
      // Bound cold storage
      if (this.coldStorage.length > this.MAX_COLD_STORAGE) {
        this.coldStorage = this.coldStorage.slice(-this.MAX_COLD_STORAGE);
      }
      // Persist to disk
      try {
        const dir = this.coldStoragePath.substring(0, this.coldStoragePath.lastIndexOf("/"));
        mkdirSync(dir, { recursive: true });
        writeFileSync(this.coldStoragePath, JSON.stringify(this.coldStorage, null, 2));
      } catch { /* non-fatal */ }
    }

    return archived;
  }

  /** Get consolidation history for diagnostics */
  getConsolidationHistory(): ConsolidationAttribution[] {
    return [...this.consolidationHistory];
  }

  /** Recall memories from cold storage for a specific task type */
  recallFromColdStorage(taskType: TaskType): ExecutionRecord[] {
    try {
      if (!existsSync(this.coldStoragePath)) return [];
      const stored = JSON.parse(readFileSync(this.coldStoragePath, "utf-8")) as ExecutionRecord[];
      return stored.filter(r => r.taskType === taskType).slice(-5);
    } catch {
      return [];
    }
  }

  // ------------------------------------------------------------------
  // Stats & Persistence
  // ------------------------------------------------------------------

  getStats(): Map<TaskType, { count: number; avgQuality: number; avgCost: number; avgLatency: number }> {
    const stats = new Map<TaskType, { count: number; avgQuality: number; avgCost: number; avgLatency: number }>();
    for (const [type, branch] of this.branches) {
      if (branch.totalExperiments > 0) {
        const records = this.globalHistory.filter(r => r.taskType === type);
        stats.set(type, {
          count: branch.totalExperiments,
          avgQuality: records.length > 0 ? records.reduce((s, r) => s + r.qualityScore, 0) / records.length : 0,
          avgCost: records.length > 0 ? records.reduce((s, r) => s + r.totalCost, 0) / records.length : 0,
          avgLatency: records.length > 0 ? records.reduce((s, r) => s + r.latencyMs, 0) / records.length : 0,
        });
      }
    }
    return stats;
  }

  getConfig(): TurboContextConfig {
    return { ...this.config };
  }

  private getTaskTypeQuality(type: TaskType): number {
    const records = this.globalHistory.filter(r => r.taskType === type);
    if (records.length === 0) return 0;
    return records.reduce((s, r) => s + r.qualityScore, 0) / records.length;
  }

  getQualityTrend(): {
    average: number;
    trend: "improving" | "stable" | "declining";
    byType: Record<string, number>;
    branches: Record<string, { momentum: number; stability: number; bestQuality: number }>;
  } {
    if (this.globalHistory.length < 3) {
      return {
        average: 0, trend: "stable", byType: {},
        branches: {},
      };
    }

    const recent = this.globalHistory.slice(-20);
    const avg = recent.reduce((s, r) => s + r.qualityScore, 0) / recent.length;

    const third = Math.floor(recent.length / 3);
    let trend: "improving" | "stable" | "declining" = "stable";
    if (third > 0) {
      const early = recent.slice(0, third).reduce((s, r) => s + r.qualityScore, 0) / third;
      const late = recent.slice(-third).reduce((s, r) => s + r.qualityScore, 0) / third;
      trend = late > early ? "improving" : late < early ? "declining" : "stable";
    }

    const byType: Record<string, number> = {};
    for (const type of ALL_BRANCHES) {
      const q = this.getTaskTypeQuality(type);
      if (q > 0) byType[type] = Math.round(q * 100) / 100;
    }

    const branches: Record<string, { momentum: number; stability: number; bestQuality: number }> = {};
    for (const [type, branch] of this.branches) {
      if (branch.totalExperiments > 0) {
        branches[type] = {
          momentum: branch.trajectory.momentum,
          stability: branch.trajectory.stabilityScore,
          bestQuality: Math.round(branch.bestQuality * 100) / 100,
        };
      }
    }

    return {
      average: Math.round(avg * 100) / 100,
      trend,
      byType,
      branches,
    };
  }

  // ------------------------------------------------------------------
  // v3.1 — Plateau Detection & Strategic Directives (autoresearch)
  // ------------------------------------------------------------------

  /**
   * 多规则平台期检测（autoresearch: quantitative branch health signals）。
   *
   * 4 条检测规则，每条有独立置信度:
   *   1. improvement_stall: 最近 3 次 vs 前 2 次无改进，且速度平坦
   *   2. crash_dominant:     崩溃率 > 成功率 * 2
   *   3. novelty_collapse:   最近假设几乎相同（novelty < 0.15）
   *   4. slow_decline:       后半段平均质量 < 前半段
   *
   * 返回 PlateauSignal，包含 isPlateaued、reason、confidence 和各规则详情。
   */
  detectPlateau(taskType: TaskType): PlateauSignal {
    return detectPlateauImpl(this.branches, taskType);
  }

  /**
   * 生成战略指令（autoresearch: high-level planner guidance）。
   *
   * 根据平台期检测和分支指标生成具体行动指导:
   *   MOMENTUM  → 深入利用当前方向
   *   PLATEAU   → 切换到未探索的任务类型
   *   CAUTION   → 优先更小、更安全的变更
   *   DIVERSIFY → 尝试不同的角度
   *   STEADY    → 适度探索
   *   EXPLORE   → 实验太少，尝试多样化初始变更
   */
  generateStrategicDirective(taskType: TaskType): StrategicDirective {
    return generateStrategicDirectiveImpl(
      this.branches,
      taskType,
      () => this.getActiveBranches(),
    );
  }

  /**
   * 自适应 MMR λ（autoresearch: adaptive lambda based on branch state）。
   *
   * 平台期 → 低 λ (更注重多样性，尝试新东西)
   * 动量期 → 高 λ (更注重相关性，深入利用)
   * 默认   → 0.65 (平衡)
   */
  getAdaptiveMmrLambda(taskType: TaskType): number {
    return computeAdaptiveMmrLambda(this.branches, taskType);
  }

  // ------------------------------------------------------------------
  // v3.1 — Contrastive Pair Discovery (autoresearch: highest-signal context)
  // ------------------------------------------------------------------

  /**
   * 发现对比对：相似任务类型但相反结果的实验。
   *
   * 这些对比对给算法提供因果洞察：
   * "为什么相似的实验一个成功一个失败？"
   *
   * 返回 top N 个对比对，按相似度 × 结果差异排序。
   */
  findContrastivePairs(taskType: TaskType, nPairs: number = 2): ContrastivePair[] {
    return findContrastivePairsImpl(
      this.globalHistory,
      taskType,
      this.config.qualityThreshold,
      this.config.maxAttempts,
      nPairs,
    );
  }

  /**
   * 提取执行记录的特征向量（供对比对发现使用）。
   */
  private extractRecordFeatures(record: ExecutionRecord): string[] {
    return extractRecordFeatures(record);
  }

  // ------------------------------------------------------------------
  // v3.1 — Future Directions Synthesis (autoresearch: rule-based, no extra LLM)
  // ------------------------------------------------------------------

  /**
   * 合成未来方向（autoresearch: rule-based synthesis）。
   *
   * 根据实验结果自动生成下一步建议，无需额外 LLM 调用。
   */
  synthesizeFutureDirections(
    taskType: TaskType,
    qualityScore: number,
    attemptCount: number,
    isSuccess: boolean,
    isCrash: boolean,
  ): string {
    return synthesizeFutureDirectionsImpl(
      this.branches,
      taskType,
      qualityScore,
      attemptCount,
      isSuccess,
      isCrash,
      this.config.maxAttempts,
    );
  }

  // ------------------------------------------------------------------
  // v3.1 — Global IDF Cache Management (autoresearch: IDF-weighted retrieval)
  // ------------------------------------------------------------------

  /**
   * 获取当前 IDF 缓存（供 compressor 使用）。
   */
  getIDFCache(): IDFCache {
    return this.idfCache;
  }

  /**
   * 用最新的上下文片段更新 IDF 缓存。
   *
   * 增量更新策略:
   *   - 如果缓存为空或文档数变化 >20%，完全重建
   *   - 否则增量合并新词
   */
  updateIDFCache(fragments: Array<{ content: string }>): void {
    updateIDFCacheImpl(this.idfCache, fragments);
  }

  /**
   * 获取综合检索上下文（供 index.ts 在一次调用中获取所有 v3.1 数据）。
   *
   * 返回：IDF 缓存、自适应 MMR λ、战略指令、平台期信号。
   */
  getRetrievalContext(taskType: TaskType): {
    idfCache: IDFCache;
    adaptiveMmrLambda: number;
    directive: StrategicDirective;
    plateau: PlateauSignal;
    contrastivePairs: ContrastivePair[];
  } {
    return {
      idfCache: this.idfCache,
      adaptiveMmrLambda: computeAdaptiveMmrLambda(this.branches, taskType),
      directive: generateStrategicDirectiveImpl(this.branches, taskType, () => this.getActiveBranches()),
      plateau: detectPlateauImpl(this.branches, taskType),
      contrastivePairs: findContrastivePairsImpl(
        this.globalHistory, taskType,
        this.config.qualityThreshold, this.config.maxAttempts, 2
      ),
    };
  }

  // ------------------------------------------------------------------
  // v4.0 — Two-Phase Causal Retrieval (agent.py retrieve_relevant_memories)
  // ------------------------------------------------------------------

  /**
   * v4.0: Get two-phase causal retrieval results.
   *
   * Adapts the stateless twoPhaseCausalRetrieval() with Learner's internal
   * state — globalHistory as experiments, curriculum params from RL engine,
   * surprise stats for normalization, subsystem baselines from history.
   *
   * This is the v4 replacement for simple similarity-based retrieval.
   * Phase 1: similarity pool → Phase 2: advantage-weighted causal re-rank
   * → MMR diversity with entropy bonus.
   *
   * @param taskType - Current task type for branch matching and baselines
   * @param topK - Number of results to return (default 5)
   * @param similarityQuery - Optional query text for IDF-weighted similarity
   */
  getTwoPhaseRetrievalResults(
    taskType: TaskType,
    topK: number = 5,
    similarityQuery?: string,
  ): ReturnType<typeof twoPhaseCausalRetrieval> {
    // Cast globalHistory to RLExecutionRecord[] — records accumulate RL fields
    // via RLFeedbackEngine.applyRLFeedback() over time
    const experiments = this.globalHistory as RLExecutionRecord[];

    // Compute subsystem baselines from actual history
    const baseline = computeSubsystemBaselines(experiments);

    // Get curriculum-phase-adaptive params
    const curriculum = this.rlEngine.getCurriculumContext();

    // Get evolved retrieval strategy weights
    const strategy = this.rlEngine.getRetrievalStrategy();

    // Merge strategy-evolved dim weights with curriculum params
    const retrievalWeights = { ...DEFAULT_RETRIEVAL_WEIGHTS };
    if (strategy.dimWeights) {
      for (const [dim, weight] of Object.entries(strategy.dimWeights)) {
        if (dim in retrievalWeights) {
          (retrievalWeights as any)[dim] = weight;
        }
      }
    }

    // Get two-phase config (curriculum-adjusted)
    const config = this.rlEngine.getTwoPhaseRetrievalConfig();

    // Get adaptive MMR lambda from plateau/branch state
    const mmrLambda = computeAdaptiveMmrLambda(this.branches, taskType);

    return twoPhaseCausalRetrieval(
      experiments,
      taskType,
      topK,
      config,
      baseline,
      {
        surpriseWeight: curriculum.params.surpriseWeight,
        curiosityWeight: curriculum.params.curiosityWeight,
      },
      retrievalWeights,
      {
        similarityQuery,
        idfCache: this.idfCache,
        mmrLambda,
      },
    );
  }

  // ------------------------------------------------------------------
  // Self-Evolution (v2.2 — autoresearch-inspired keep/discard loop)
  // ------------------------------------------------------------------

  /**
   * 获取当前活跃的变异策略（供 composer 使用）
   */
  getActiveMutation(taskType: TaskType): StrategyMutation | null {
    return getActiveMutationImpl(this.evolution, this.activeMutation, taskType);
  }

  /**
   * 提出一个策略变异提案（autoresearch-style: propose a change）
   *
   * 在当前最佳策略基础上生成一个合理的变体：
   * - 合并两个轮次 → 减少轮数
   * - 移除一个轮次 → 简化流程
   * - 添加质量标准 → 加强控制
   */
  proposeMutation(taskType: TaskType): StrategyMutation | null {
    const result = proposeMutationImpl(
      this.branches, this.evolution, this.activeMutation,
      taskType, this.EVOLUTION_TRIAL_SIZE,
    );
    if (result) this.save();
    return result;
  }

  /**
   * 记录一次 trial 结果（autoresearch-style: run the experiment）
   *
   * 根据 activeMutation 判断当前执行是 trial 还是 baseline
   * v2.4: 记录完整试验日志 + token 效率追踪
   */
  recordTrial(taskType: TaskType, qualityScore: number, usingMutation: boolean, tokensUsed = 0): void {
    recordTrialImpl(
      this.evolution, this.activeMutation,
      taskType, qualityScore, usingMutation, tokensUsed,
      this.EVOLUTION_TRIAL_SIZE,
    );
    this.save();
  }

  /**
   * 记录一次崩溃的 trial（autoresearch-style: crash resilience）
   *
   * 崩溃不计算质量，但消耗一次 trial 机会
   * 如果 mutation 导致崩溃 → 立即标记为 crashed 并 auto-discard
   */
  recordTrialCrash(taskType: TaskType): void {
    recordTrialCrashImpl(this.evolution, this.activeMutation, taskType);
    this.save();
  }

  /**
   * 获取进化统计
   */
  getEvolutionStats(): { total: number; kept: number; discarded: number; active: number } {
    return getEvolutionStatsImpl(this.evolution, this.activeMutation);
  }

  /**
   * v6: Get raw evolution data for experiment type selection and analysis.
   */
  getEvolutionData(): import("../types.js").StrategyEvolutionData {
    return this.evolution;
  }

  /**
   * 获取指定任务类型的 canonical 策略栈（已保留的变异列表）
   *
   * autoresearch: the branch tip IS the best config
   * 这里返回所有已保留的变异，composer 会按顺序应用
   */
  getCanonicalMutations(taskType: TaskType): StrategyMutation[] {
    return getCanonicalMutationsImpl(this.evolution, taskType);
  }

  /**
   * 重置指定类型的 canonical 策略（回退到原始基线）
   *
   * autoresearch: git reset to undo bad ideas
   */
  resetCanonicalStrategy(taskType: TaskType): void {
    resetCanonicalStrategyImpl(this.evolution, taskType);
    this.save();
  }

  /**
   * 获取完整试验日志（autoresearch: results.tsv）
   */
  getTrialLog(): TrialLogEntry[] {
    return getTrialLogImpl(this.evolution);
  }

  // ------------------------------------------------------------------
  // Autoresearch Experiment Loop (v3.0)
  // ------------------------------------------------------------------

  /**
   * 将实验运行记录写入 TSV 日志文件（autoresearch: results.tsv）。
   * 人类早上醒来查看此文件。
   */
  writeExperimentLog(runs: ExperimentRun[], filePath?: string): void {
    const outputPath = filePath || join(homedir(), ".turbocontext", "results.tsv");
    const dir = join(outputPath, "..");
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });

    const header = [
      "run", "timestamp", "task_type", "mutation_type", "mutation_desc",
      "baseline_efficiency", "experiment_efficiency", "delta_percent",
      "decision", "quality", "cost", "attempts", "wall_clock_sec", "status",
    ].join("\t");

    const rows = runs.map(r => {
      const entry = this.toLogEntry(r);
      return [
        entry.run, entry.timestamp, entry.taskType, entry.mutationType, entry.mutationDesc,
        entry.baselineEfficiency.toFixed(2), entry.experimentEfficiency.toFixed(2),
        `${entry.deltaPercent >= 0 ? "+" : ""}${entry.deltaPercent.toFixed(2)}%`,
        entry.decision, entry.quality.toFixed(4), entry.cost.toFixed(6),
        entry.attempts, entry.wallClockSec.toFixed(1), entry.status,
      ].join("\t");
    });

    // Append mode: if file exists, skip header
    const fileExists = existsSync(outputPath);
    writeFileSync(outputPath, (fileExists ? "" : header + "\n") + rows.join("\n") + "\n", "utf-8");
  }

  /**
   * 将实验运行记录转换为日志条目
   */
  private toLogEntry(run: ExperimentRun): ExperimentLogEntry {
    return {
      run: run.runNumber,
      timestamp: new Date(run.timestamp).toISOString().replace("T", " ").slice(0, 19),
      taskType: run.taskType,
      mutationType: run.mutation?.type || "baseline",
      mutationDesc: run.mutation
        ? JSON.stringify(run.mutation).replace(/\t/g, " ")
        : "no mutation — baseline run",
      baselineEfficiency: run.baselineMetric.efficiency,
      experimentEfficiency: run.experimentMetric.efficiency,
      deltaPercent: run.deltaPercent,
      decision: run.decision,
      quality: run.experimentMetric.quality,
      cost: run.experimentMetric.cost,
      attempts: run.experimentMetric.attempts,
      wallClockSec: Math.round(run.wallClockMs / 100) / 10,
      status: run.status,
    };
  }

  /**
   * 生成实验摘要（人类 morning review）
   */
  getExperimentSummary(runs: ExperimentRun[]): string {
    const kept = runs.filter(r => r.decision === "keep");
    const discarded = runs.filter(r => r.decision === "discard");
    const crashed = runs.filter(r => r.status === "crash");

    const best = runs
      .filter(r => r.decision === "keep")
      .sort((a, b) => b.deltaPercent - a.deltaPercent);

    const lines: string[] = [
      `=== TurboContext Experiment Summary ===`,
      `Total: ${runs.length} experiments`,
      `Kept: ${kept.length} | Discarded: ${discarded.length} | Crashed: ${crashed.length}`,
      ``,
    ];

    if (best.length > 0) {
      lines.push(`Top improvements:`);
      for (const r of best.slice(0, 3)) {
        lines.push(
          `  #${r.runNumber} ${r.taskType}: ${r.mutation?.type || "baseline"} ` +
          `→ ${r.deltaPercent >= 0 ? "+" : ""}${r.deltaPercent.toFixed(2)}% ` +
          `(efficiency: ${r.experimentMetric.efficiency.toFixed(2)})`
        );
      }
    }

    if (crashed.length > 0) {
      lines.push(``);
      lines.push(`Crashes:`);
      for (const r of crashed) {
        lines.push(`  #${r.runNumber}: ${r.crashReason || "unknown"}`);
      }
    }

    lines.push(``);
    lines.push(`Canonical strategies accumulated:`);
    for (const [type, mutations] of Object.entries(this.evolution.canonicalStrategies)) {
      if (mutations.length > 0) {
        lines.push(`  ${type}: ${mutations.length} mutations kept`);
      }
    }

    return lines.join("\n");
  }

  /**
   * 加载 Mission 配置（autoresearch: read program.md）。
   * 人类编辑 mission.md，agent 读取后按指令执行。
   */
  static loadMission(filePath?: string): Mission | null {
    const missionPath = filePath || join(process.cwd(), "mission.md");
    try {
      if (!existsSync(missionPath)) return null;
      const raw = readFileSync(missionPath, "utf-8");

      // 解析 frontmatter (YAML-like)
      const fmMatch = raw.match(/^---\n([\s\S]*?)\n---/);
      if (!fmMatch) return null;

      const fm: Record<string, string> = {};
      for (const line of fmMatch[1].split("\n")) {
        const colonIdx = line.indexOf(":");
        if (colonIdx > 0) {
          fm[line.slice(0, colonIdx).trim()] = line.slice(colonIdx + 1).trim();
        }
      }

      const notes = raw.replace(/^---\n[\s\S]*?\n---/, "").trim();

      return {
        goal: fm.goal || "Optimize algorithm parameters for better quality/cost ratio",
        tokenBudgetPerRun: parseInt(fm.token_budget_per_run || "8000"),
        timeBudgetPerRun: parseInt(fm.time_budget_per_run || "300"),
        maxExperiments: parseInt(fm.max_experiments || "20"),
        taskPool: [],   // 由 engine 注入
        contextPool: [], // 由 engine 注入
        explorationConstraints: {
          allowedMutations: fm.allowed_mutations?.split(",").map(s => s.trim()),
          frozenParams: fm.frozen_params?.split(",").map(s => s.trim()),
        },
        humanNotes: notes,
      };
    } catch {
      return null;
    }
  }

  // ------------------------------------------------------------------
  // Persistence
  // ------------------------------------------------------------------

  private save(): void {
    try {
      const dir = join(homedir(), ".turbocontext");
      if (!existsSync(dir)) mkdirSync(dir, { recursive: true });

      const branchObj: Record<string, BranchState> = {};
      for (const [key, val] of this.branches) {
        branchObj[key] = val;
      }
      const sourceObj: Record<string, SourceMemory> = {};
      for (const [key, val] of this.sourceMemory) {
        sourceObj[key] = val;
      }

      writeFileSync(this.statePath, JSON.stringify({
        config: this.config,
        history: this.globalHistory.slice(-this.MAX_HISTORY),
        branches: branchObj,
        sourceMemory: sourceObj,
        evolution: {
          ...this.evolution,
          trialLog: this.evolution.trialLog.slice(-1000),
        },
        idfCache: {
          weights: this.idfCache.weights,
          documentCount: this.idfCache.documentCount,
          lastUpdated: this.idfCache.lastUpdated,
        },
        // v3.4 RL state (delegated to RLFeedbackEngine)
        ...this.rlEngine.toJSON(),
        // v3.9: Cold storage metadata + verification + consolidation history
        coldStorageCount: this.coldStorage.length,
        verificationHistory: this.verificationHistory.slice(-50),
        consolidationHistory: this.consolidationHistory.slice(-20),
      }, null, 2), "utf-8");
    } catch (err) {
      console.error("[TurboContext] Failed to persist state:", (err as Error).message);
    }
  }

  private load(): void {
    try {
      if (!existsSync(this.statePath)) return;
      const raw = readFileSync(this.statePath, "utf-8");
      const state = JSON.parse(raw);
      if (state.config) this.config = { ...this.config, ...state.config };
      if (state.history) this.globalHistory = state.history.slice(-this.MAX_HISTORY);
      if (state.branches) {
        for (const [key, val] of Object.entries(state.branches)) {
          this.branches.set(key as TaskType, val as BranchState);
        }
      }
      if (state.sourceMemory) {
        for (const [key, val] of Object.entries(state.sourceMemory)) {
          this.sourceMemory.set(key, val as SourceMemory);
        }
      }
      if (state.evolution) {
        this.evolution = state.evolution as StrategyEvolutionData;
        // 兼容旧状态文件：为缺失字段提供默认值
        if (!this.evolution.canonicalStrategies) {
          this.evolution.canonicalStrategies = {};
        }
        if (!this.evolution.trialLog) {
          this.evolution.trialLog = [];
        }
        // 如果 saved 实验已 concluded，清理 activeMutation
        if (this.evolution.currentExperimentId) {
          const exp = this.evolution.experiments.find(e => e.id === this.evolution.currentExperimentId);
          if (!exp || exp.status !== "pending") {
            this.evolution.currentExperimentId = null;
          } else {
            // 恢复 activeMutation
            this.activeMutation.set(exp.taskType, {
              experimentId: exp.id,
              trialBaselines: [],
            });
          }
        }
      }
      if (state.idfCache) {
        this.idfCache = {
          weights: state.idfCache.weights || {},
          documentCount: state.idfCache.documentCount || 0,
          lastUpdated: state.idfCache.lastUpdated || 0,
          stopWords: new Set(),
        };
      }
      // v3.4: Restore RL state from engine
      this.rlEngine.fromJSON(state as Record<string, unknown>);
      // v3.9: Restore cold storage metadata
      if (state.verificationHistory) {
        this.verificationHistory = (state.verificationHistory as VerificationRecord[]).slice(-50);
      }
      if (state.consolidationHistory) {
        this.consolidationHistory = (state.consolidationHistory as ConsolidationAttribution[]).slice(-20);
      }
    } catch (err) {
      console.error("[TurboContext] Failed to load state:", (err as Error).message);
    }
  }

  // ------------------------------------------------------------------
  // v3.9: Cold Storage — Public API
  // ------------------------------------------------------------------

  /** Query the cold storage archive for diagnostic purposes */
  getColdStorageCount(): number {
    return this.coldStorage.length;
  }

  /** Get full cold storage contents for inspection */
  getColdStorage(): ExecutionRecord[] {
    try {
      if (!existsSync(this.coldStoragePath)) return [...this.coldStorage];
      const stored = JSON.parse(readFileSync(this.coldStoragePath, "utf-8")) as ExecutionRecord[];
      return [...this.coldStorage, ...stored];
    } catch {
      return [...this.coldStorage];
    }
  }
}

/** 学习调整记录 */
export interface LearningAdjustment {
  component: string;
  changes: Record<string, string>;
  reason: string;
}
