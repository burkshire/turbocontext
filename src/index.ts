// ============================================================
// TurboContext: Main Entry Point
// ============================================================
// 将所有阶段串联为完整的算法流水线
// v5: RLEngineV5 wired in as the primary RL engine alongside Learner
// ============================================================

import type { Task, ContextFragment, TurboContextConfig, ExecutionRecord, QualityDimensions, ExperimentRun, UnifiedMetric, Mission, StrategicDirective, PlateauSignal, IDFCache } from "./types.js";
import type { PromptArchitecture } from "./core/composer.js";
import type { EmbeddingProvider } from "./core/embeddings.js";
import { compressContext, compressFragment } from "./core/compressor.js";
import { composePromptArchitecture } from "./core/composer.js";
import { qualityWeightedGeneration, evaluateQuality, DEFAULT_QUALITY_CONFIG, computeUnifiedMetric, computeSimplicity } from "./core/generator.js";
import { Optimizer, MODEL_TIERS } from "./core/optimizer.js";
import { Learner } from "./core/learner.js";
import { createLLMCall, defaultLLMCall } from "./core/llm.js";
import type { LLMConfig } from "./core/llm.js";
import { PeriodicScheduler } from "./state/periodic-scheduler.js";
import { QualityProxy } from "./core/quality-proxy.js";

// v6: New modules
import { selectExperimentType } from "./core/evolution-engine.js";
import { transferPolicy, blendParams } from "./state/transfer/cross-branch-transfer.js";
import { loadProgram, isMutationAllowed } from "./core/program-loader.js";
import { runSoftAblation, recordAblationToV5 } from "./core/ablation-runner.js";

// v5: RL engine and state management
import { RLEngineV5 } from "./state/rl/rl-engine.js";
import type { Trial as TrialV5, RetrievalInput, TaskType } from "./state/types.js";
import { ContextOrigin } from "./state/types.js";
import { createHash } from "node:crypto";

export { compressContext } from "./core/compressor.js";
export { composePromptArchitecture } from "./core/composer.js";
export { qualityWeightedGeneration, evaluateQuality } from "./core/generator.js";
export { Optimizer, MODEL_TIERS } from "./core/optimizer.js";
export { Learner } from "./core/learner.js";
export { createLLMCall, defaultLLMCall } from "./core/llm.js";
export {
  type EmbeddingProvider,
  OpenAICompatibleEmbeddingProvider,
  NoOpEmbeddingProvider,
  cosineSimilarity,
  normalizeSimilarity,
  createSemanticMatcher,
} from "./core/embeddings.js";

export type * from "./types.js";

/**
 * TurboContext 主引擎
 *
 * 将第 1-5 阶段组合为完整的算法流水线:
 *
 * 输入: 任务 T, 上下文 C, 配置
 * 输出: 经过质量门控的优化结果
 *
 * 流程:
 *   C' = Phase1(T, C)          ← 上下文压缩
 *   P  = Phase2(T, C')         ← 提示架构组合
 *   M  = Phase4(T, history)    ← 成本优化（模型选择）
 *   O  = Phase3(P, M)          ← 质量加权生成
 *   Phase5(O, history)         ← 反馈学习
 */
export class TurboContextEngine {
  private optimizer: Optimizer;
  private learner: Learner;
  private history: ExecutionRecord[] = [];
  private executionCount = 0;
  private llmCall: (prompt: string, temperature: number) => Promise<string>;
  private embeddingProvider?: EmbeddingProvider;

  // v5: Primary RL engine — owns state persistence, trial recording, parameter querying
  private rlEngineV5: RLEngineV5;
  // v6: Quality Proxy — PACE-inspired learned quality prediction from cheap signals
  private qualityProxy: QualityProxy;

  constructor(config?: Partial<TurboContextConfig> & {
    llm?: LLMConfig | ((prompt: string, temp: number) => Promise<string>);
    embeddingProvider?: EmbeddingProvider;
  }) {
    const defaultConfig: TurboContextConfig = {
      alpha: 0.55,
      beta: 0.20,
      gamma: 0.25,
      maxTokenBudget: 8000,
      minCoverage: 0.80,
      qualityThreshold: 0.85,
      maxAttempts: 3,
      temperatureSchedule: [0.7, 0.35, 0.1],
      complexityThresholdLow: 0.30,
      complexityThresholdHigh: 0.42,
      learningRate: 0.1,
      historyWindow: 100,
    };

    // 分离 LLM 配置、embedding provider 和算法配置
    const { llm, embeddingProvider, ...algoConfig } = config || {};
    const mergedConfig = { ...defaultConfig, ...algoConfig };

    // 初始化 LLM 调用
    if (typeof llm === "function") {
      this.llmCall = llm;
    } else {
      this.llmCall = createLLMCall(llm as LLMConfig | undefined);
    }

    // 存储 embedding provider（v3.2）
    this.embeddingProvider = embeddingProvider;

    this.optimizer = new Optimizer({
      thresholdLow: mergedConfig.complexityThresholdLow,
      thresholdHigh: mergedConfig.complexityThresholdHigh,
    });
    this.learner = new Learner(mergedConfig);
    this.qualityProxy = new QualityProxy({ maxCalibrationPoints: 200 });

    // v5: Initialize RL engine — in-memory for tests, disk-persisted for production
    // v5.1: loadOrMigrate transparently migrates V4 state.json → state-v5.json
    const statePath = process.env.VITEST || process.env.CI ? ":memory:" : undefined;
    this.rlEngineV5 = RLEngineV5.loadOrMigrate(statePath) ?? RLEngineV5.create(statePath);

    if (statePath !== ":memory:") {
      const status = this.rlEngineV5.getStatus();
      console.log(
        `[TurboContext v5] RL engine ready: ` +
        `${status.totalTrials} trials, ${status.activeMemories} active memories, ` +
        `phase ${status.curriculumPhase}`
      );
    }
  }

  /**
   * 执行完整的 TurboContext 流水线
   */
  async execute(
    task: Task,
    contextFragments: ContextFragment[],
    opts?: { workingDir?: string },
  ): Promise<TurboContextResult> {
    this.executionCount++;
    const startTime = Date.now();

    // ── v5: Query RL-optimized parameters BEFORE Phase 1 ──
    // This CLOSES the feedback loop: RL learns → params updated → execution uses learned params.
    const v5Capabilities = this.decomposeTaskCapabilities(task);
    const v5Optimal = this.rlEngineV5.queryOptimalParams({
      taskType: task.type as TaskType,
      description: task.description,
      capabilityRequirements: v5Capabilities,
    });

    // Blend V5 RL-optimized params with Learner's base config.
    // V5 provides learned adjustments; Learner provides stability/fallback.
    let blendedAlpha = this.lerp(this.learner.getConfig().alpha, v5Optimal.compressionWeights.alpha, 0.3);
    let blendedBeta = this.lerp(this.learner.getConfig().beta, v5Optimal.compressionWeights.beta, 0.3);
    let blendedGamma = 1.0 - blendedAlpha - blendedBeta;
    const blendedTempSchedule: [number, number, number] = [
      v5Optimal.temperatureSchedule[0],
      v5Optimal.temperatureSchedule[1],
      v5Optimal.temperatureSchedule[2],
    ];
    const blendedQualityThreshold = this.lerp(
      this.learner.getConfig().qualityThreshold, v5Optimal.qualityThreshold, 0.25
    );
    const blendedMaxAttempts = Math.round(
      this.lerp(this.learner.getConfig().maxAttempts, v5Optimal.maxAttempts, 0.25)
    );
    const blendedMmrLambda = v5Optimal.retrievalParams.mmrLambda;

    // v6: Cross-branch transfer — bootstrap under-explored task types from similar ones
    let transferApplied = false;
    const v5Snapshot = this.rlEngineV5.getSnapshot?.() ?? null;
    if (v5Snapshot) {
      const taskTypeV5 = (task.type === "code_refactor" ? "refactoring" :
        task.type === "analysis" || task.type === "design" ? "architecture" :
        task.type === "testing" || task.type === "general" ? "code_generation" :
        task.type) as TaskType;
      const transfer = transferPolicy(taskTypeV5, v5Snapshot);
      if (transfer) {
        const base = { alpha: blendedAlpha, beta: blendedBeta, gamma: blendedGamma };
        const blended = blendParams(base, transfer.compression, transfer.similarity);
        blendedAlpha = blended.alpha;
        blendedBeta = blended.beta;
        blendedGamma = blended.gamma;
        transferApplied = true;
        if (this.executionCount <= 3) {
          console.log(
            `[TurboContext v6] Cross-branch transfer: ${transfer.sourceTaskType} → ${taskTypeV5} ` +
            `(sim=${transfer.similarity.toFixed(2)}, α=${blendedAlpha.toFixed(2)} β=${blendedBeta.toFixed(2)})`
          );
        }
      }
    }

    if (this.executionCount <= 3) {
      console.log(
        `[TurboContext v5] RL params: α=${blendedAlpha.toFixed(2)} β=${blendedBeta.toFixed(2)} ` +
        `γ=${blendedGamma.toFixed(2)} | T=[${blendedTempSchedule.map(t => t.toFixed(2)).join(",")}] ` +
        `| θQ=${blendedQualityThreshold.toFixed(2)} | MMRλ=${blendedMmrLambda.toFixed(2)} ` +
        `| phase=${v5Optimal.curriculumPhase} | bonus=${v5Optimal.explorationBonus.toFixed(3)}`
      );
    }

    // 阶段1: 上下文压缩（v5: 使用 RL 优化后的 compression weights + MMR lambda）
    this.learner.updateIDFCache(contextFragments);

    // v5: V5 RL engine provides retrieval context (plateau, directive, MMR λ, IDF cache)
    const retrievalCtx = this.rlEngineV5.getRetrievalContext(task.type as TaskType);

    // v4.1: Two-phase causal retrieval — pre-compute which historical experiments
    // are causally most relevant to the current task. Their source files get a
    // retrieval boost in the compressor.
    // v5 note: still uses Learner (V4) — pending causal graph migration to V5
    const twoPhaseResults = this.learner.getTwoPhaseRetrievalResults(
      task.type, 5, task.description,
    );
    // Build a boost map: source files referenced in top retrieved experiments
    // get a +0.08 boost in the compressor scoring
    const retrievalBoost = new Map<string, number>();
    if (twoPhaseResults.selected.length > 0) {
      for (const rec of twoPhaseResults.selected) {
        for (const src of rec.sourceFiles ?? []) {
          const current = retrievalBoost.get(src) ?? 0;
          retrievalBoost.set(src, Math.min(current + 0.08, 0.25));
        }
      }
    }

    // 打印战略指令（帮助调试和 morning review）
    if (this.executionCount <= 3 || retrievalCtx.plateau.isPlateaued) {
      console.log(
        `[TurboContext] Directive: ${retrievalCtx.directive.directive} | ` +
        `MMR λ=${retrievalCtx.adaptiveMmrLambda.toFixed(2)} | ` +
        `Phase2Retrieval: ${twoPhaseResults.selected.length} results | ` +
        `Plateau: ${retrievalCtx.plateau.isPlateaued ? retrievalCtx.plateau.reason : "none"}`
      );
    }

    // v7: Zero-overhead fast path for small projects (≤20 files).
    // The scoring+selection pipeline costs ~400 tokens of overhead.
    // For projects below the breakeven, skip it — just compress everything.
    // This drops the breakeven from ~15 files to 0, making turbocontext
    // net-beneficial on every invocation regardless of project size.
    const FAST_PATH_THRESHOLD = 20;
    let compressed;
    if (contextFragments.length <= FAST_PATH_THRESHOLD) {
      // v7: Zero-overhead fast path — compress all fragments directly,
      // skipping the scoring/selection pipeline (~400 token overhead).
      const originalTokens = contextFragments.reduce((sum, f) => sum + Math.ceil(f.content.length / 4), 0);
      const compressedFrags = contextFragments.map(f => compressFragment(f, 1.0));
      const compressedTokens = compressedFrags.reduce(
        (sum, c) => sum + Math.ceil(c.preservedSections.join("\n").length / 4), 0
      );
      compressed = {
        originalTokens, compressedTokens,
        compressionRatio: 1 - (compressedTokens / Math.max(1, originalTokens)),
        fragments: compressedFrags,
        coverage: contextFragments.length > 0 ? 1.0 : 0,
      };
      if (this.executionCount <= 1) {
        console.log(`[TurboContext v7] Fast path: ${contextFragments.length} files → direct compression (${(compressed.compressionRatio * 100).toFixed(0)}% ratio)`);
      }
    } else {
      compressed = await compressContext(task, contextFragments, {
      // v5: RL-optimized compression weights blended with Learner defaults
      alpha: blendedAlpha,
      beta: blendedBeta,
      gamma: blendedGamma,
      maxTokenBudget: v5Optimal.tokenBudget,
      minCoverage: this.learner.getConfig().minCoverage,
      // v5: Merged boost — V5 Thompson Sampling + Two-Phase Retrieval boost
      sourceBoostFn: (source: string) => {
        const rlBoost = this.rlEngineV5.getSourceBoostRL(source);
        const retrievalBoostVal = retrievalBoost.get(source) ?? 0;
        return rlBoost + retrievalBoostVal;
      },
      // v5.1: Causal boost from V5 ablation data (direct measurement, no graph inference)
      causalBoostFn: (frag: import("./types.js").ContextFragment, t: Task) =>
        this.rlEngineV5.getCausalBoost(frag.source, t.type as TaskType),
      // causalGraph removed — never trained on real ablation data
      idfCache: {
        weights: retrievalCtx.idfCache.weights,
        documentCount: retrievalCtx.idfCache.documentCount,
        lastUpdated: Date.now(),
        stopWords: new Set<string>(),
      },
      // v5: RL-optimized MMR lambda overrides plateau-based heuristic
      adaptiveMmrLambda: blendedMmrLambda,
      embeddingProvider: this.embeddingProvider,
    });
    } // v7: end of else block (full pipeline for >20 fragments)

    // 自进化：检查是否有待处理的策略变异（v2.3 → v2.4）
    const activeMutation = this.learner.getActiveMutation(task.type);
    if (!activeMutation) {
      this.learner.proposeMutation(task.type);
    }

    // 阶段2: 提示架构组合（应用 canonical 策略栈 + trial 变异）
    const canonicalMutations = this.learner.getCanonicalMutations(task.type);
    const trialMutation = this.learner.getActiveMutation(task.type);

    let architecture;
    let mutationCrashed = false;
    try {
      architecture = composePromptArchitecture(
        task, compressed, [],
        trialMutation ?? undefined,
        canonicalMutations.length > 0 ? canonicalMutations : undefined,
      );
    } catch (err) {
      // autoresearch: crash resilience — mark as crashed and skip
      console.error(`[TurboContext] Mutation compose crashed: ${(err as Error).message}`);
      mutationCrashed = true;
      this.learner.recordTrialCrash(task.type);
      // 回退到 baseline 架构（不带任何变异）
      architecture = composePromptArchitecture(task, compressed, []);
    }

    // 阶段4: 成本优化（在生成之前选模型）
    const modelSelection = this.optimizer.selectModel(task, this.history, { qualityProxy: this.qualityProxy });
    const costEstimate = this.optimizer.estimateCost(
      task, architecture.estimatedTokens, modelSelection.tier
    );

    // 阶段3: 质量加权生成
    const generations: Array<{
      attempt: number;
      qualityScore: number;
      dimensionScores: QualityDimensions;
      latencyMs: number;
      executionMetrics?: import("./types.js").ExecutionMetrics;
    }> = [];

    let finalOutput = "";
    let finalQuality = 0;
    let finalDimensions: QualityDimensions = {
      completeness: 0, correctness: 0, consistency: 0, format: 0
    };
    let totalAttempts = 0;

    // 模拟质量加权生成流程（实际使用时会注入 LLM 调用）
    // 这里展示算法的完整逻辑，LLM 调用由用户集成
    const gen = qualityWeightedGeneration(
      task,
      architecture,
      {
        // v5: RL-optimized quality params blended with Learner defaults
        qualityThreshold: blendedQualityThreshold,
        maxAttempts: blendedMaxAttempts,
        temperatureSchedule: blendedTempSchedule,
        workingDir: opts?.workingDir,
        sourceFiles: contextFragments.map(f => f.source),
        // v6: PACE-inspired quality proxy for learned quality prediction
        qualityProxy: this.qualityProxy,
      },
      this.llmCall  // 传入 LLM 调用函数
    );

    // 收集所有生成结果
    let lastExecutionMetrics: import("./types.js").ExecutionMetrics | undefined;
    for await (const output of gen) {
      generations.push({
        attempt: output.attempt,
        qualityScore: output.qualityScore,
        dimensionScores: output.dimensionScores,
        latencyMs: output.latencyMs,
        executionMetrics: output.executionMetrics,
      });
      totalAttempts = output.attempt;
      finalOutput = output.content;
      finalQuality = output.qualityScore;
      finalDimensions = output.dimensionScores;
      if (output.executionMetrics) lastExecutionMetrics = output.executionMetrics;
    }

    const totalLatency = Date.now() - startTime;

    // 构建执行记录（v3.1: 包含战略指令和对比对信息）
    const record: ExecutionRecord = {
      taskId: task.id,
      taskType: task.type,
      timestamp: Date.now(),
      compressionRatio: compressed.compressionRatio,
      qualityScore: finalQuality,
      totalCost: costEstimate.estimatedCostUSD,
      latencyMs: totalLatency,
      attemptCount: totalAttempts,
      modelUsed: modelSelection.tier,
      coverage: compressed.coverage,
      dimensionScores: finalDimensions,
      sourceFiles: contextFragments.map(f => f.source),
      executionMetrics: lastExecutionMetrics,
    };

    // 自进化：记录 trial 结果（v2.3 → v2.4: 含 token 效率追踪）
    if (!mutationCrashed) {
      if (trialMutation) {
        this.learner.recordTrial(task.type, finalQuality, true, architecture.estimatedTokens);
      } else {
        this.learner.recordTrial(task.type, finalQuality, false, architecture.estimatedTokens);
      }
    }

    // 阶段5: 学习
    this.learner.record(record);
    this.history.push(record);

    // ── v6: Calibrate Quality Proxy with hard signals ──
    if (lastExecutionMetrics) {
      // Hard quality = compilation success * 0.5 + test pass * 0.5 (Karpathy: measure real outcomes)
      const hardQuality = (
        (lastExecutionMetrics.compiled ? 0.5 : 0) +
        (lastExecutionMetrics.smokeTestPassed ? 0.5 : 0)
      );
      this.qualityProxy.calibrate(
        finalOutput,
        task.description,
        task.type,
        hardQuality,
        lastExecutionMetrics,
        totalAttempts,
      );
      if (hardQuality > 0 && this.qualityProxy.getCalibrationSize() <= 10) {
        console.log(
          `[TurboContext v6] Proxy calibrated: ` +
          `${this.qualityProxy.getCalibrationSize()} samples, ` +
          `hard quality=${(hardQuality * 100).toFixed(0)}%`
        );
      }
    }

    // ── v5: Record trial in V5 RL engine ──
    const v5Trial = this.buildV5Trial(task, record, compressed, modelSelection.tier);
    this.rlEngineV5.recordTrial(v5Trial, "full");
    this.rlEngineV5.saveState();

    // v5: PeriodicScheduler replaces hardcoded % 5 — curriculum-phase-gated scheduling
    const v5Status = this.rlEngineV5.getStatus();
    const dueOps = PeriodicScheduler.shouldRun(
      this.executionCount,
      v5Status.curriculumPhase,
      4,    // v5: learning/evolution interval from curriculum config
      10,   // v5: consolidation interval from curriculum config
    );
    const learnResult = dueOps.size > 0
      ? this.learner.learn()
      : null;

    // v5: RL diagnostics from V5 engine (primary source)
    // Map V4 task type to V5 task type for perTaskType lookup
    const v5TaskType = (task.type === "code_refactor" ? "refactoring" :
      task.type === "analysis" || task.type === "design" ? "architecture" :
      task.type === "testing" || task.type === "general" ? "code_generation" :
      task.type) as TaskType;
    const retrievalStrat = {
      fitness: this.rlEngineV5.getStatus().perTaskType[v5TaskType]?.baselineQuality ?? 0,
      generation: 0,
    };

    return {
      taskId: task.id,
      compressed,
      architecture,
      modelSelection,
      costEstimate,
      generations,
      finalQuality,
      totalAttempts,
      totalLatency,
      qualityTrend: this.learner.getQualityTrend(),
      learningAdjustments: learnResult?.adjustments || [],
      executionCount: this.executionCount,
      rlDiagnostics: {
        predictiveAccuracy: 1 - this.rlEngineV5.getStatus().predictiveModelAccuracy,
        curriculumPhase: v5Status.curriculumPhase,
        curriculumDescription: `Phase ${v5Status.curriculumPhase}`,
        retrievalFitness: retrievalStrat.fitness,
        retrievalGeneration: retrievalStrat.generation,
      },
    };
  }

  /**
   * 自主实验循环（autoresearch: run N experiments overnight）。
   *
   * 每次实验在固定 token 和时间预算内运行。
   * 变异 → 运行 → 评估 → keep/discard → 重复。
   * 结果写入 results.tsv，人类早上查看。
   */
  async runExperiments(options: {
    maxExperiments?: number;
    tokenBudgetPerRun?: number;
    timeBudgetPerRun?: number;
    taskPool?: Task[];
    contextPool?: ContextFragment[];
    mission?: Mission;
    resultsPath?: string;
    onProgress?: (run: ExperimentRun, summary: string) => void;
  } = {}): Promise<ExperimentRun[]> {
    const {
      maxExperiments = 20,
      tokenBudgetPerRun = 8000,
      timeBudgetPerRun = 300,
      taskPool,
      contextPool,
      mission,
      resultsPath,
      onProgress,
    } = options;

    const runs: ExperimentRun[] = [];

    // 1. Per-task-type baselines — each task type gets its own baseline for valid A/B comparison.
    // Comparing code_generation against a code_review baseline is a categorical error.
    const baselineMetricCache = new Map<string, UnifiedMetric>();
    const baselineResultCache = new Map<string, TurboContextResult>();
    const baselineContext = contextPool || this.createDefaultContext();

    async function getBaselineForTask(
      task: Task, context: typeof baselineContext,
    ): Promise<{ metric: UnifiedMetric; result: TurboContextResult }> {
      const cachedMetric = baselineMetricCache.get(task.type);
      const cachedResult = baselineResultCache.get(task.type);
      if (cachedMetric && cachedResult) return { metric: cachedMetric, result: cachedResult };
      console.log(`[TurboContext Auto] Establishing baseline for ${task.type}...`);
      const baselineResult = await self.executeWithBudget(
        task, context, tokenBudgetPerRun, timeBudgetPerRun,
      );
      const metric = computeUnifiedMetric(
        baselineResult.finalQuality, baselineResult.costEstimate.estimatedCostUSD,
        baselineResult.totalLatency, baselineResult.totalAttempts,
      );
      baselineMetricCache.set(task.type, metric);
      baselineResultCache.set(task.type, baselineResult);
      console.log(
        `[TurboContext Auto] Baseline [${task.type}]: efficiency=${metric.efficiency.toFixed(2)}, ` +
        `quality=${(metric.quality * 100).toFixed(1)}%, cost=$${metric.cost.toFixed(4)}`,
      );
      return { metric, result: baselineResult };
    }

    const self = this;

    // 2. 自主实验循环
    // v6: Load program from mission.md (Karpathy's program.md pattern)
    const program = loadProgram();
    const effectiveMax = Math.min(maxExperiments, program.maxExperiments);
    console.log(`[TurboContext Auto] === Starting ${effectiveMax} experiments (program: "${program.goal.slice(0, 80)}...") ===`);

    for (let i = 0; i < effectiveMax; i++) {
      const runStart = Date.now();

      // 选择任务（轮询 taskPool 或使用默认任务生成变体）
      const task = taskPool
        ? taskPool[i % taskPool.length]
        : this.createDefaultTask();
      const context = contextPool || baselineContext;

      // Get the correct baseline for THIS task type
      const { metric: baselineMetric, result: baselineResultForType } = await getBaselineForTask(task, context);

      // 提出变异（v6: filter by program constraints + proxy-guided selection）
      let mutation = this.learner.proposeMutation(task.type);
      if (mutation && !isMutationAllowed(mutation.type, program)) {
        console.log(`[TurboContext v6] Mutation ${mutation.type} blocked by program, retrying...`);
        mutation = this.learner.proposeMutation(task.type);
      }

      // v6: Proxy-guided mutation — if proxy is calibrated, prefer mutations
      // that target parameters with high signal relevance
      if (mutation && this.qualityProxy.getCalibrationSize() >= 8) {
        const profile = this.qualityProxy.getSignalProfile(task.type);
        const topSignals = profile.slice(0, 3).map(s => s.signal);
        // Log proxy guidance for transparency
        if (i === 0) {
          console.log(`[TurboContext v6] Proxy-guided: top signals for ${task.type}: ${topSignals.join(", ")}`);
        }
      }

      // v6: Select experiment type
      const expType = selectExperimentType(this.learner.getEvolutionData());

      // v6: Compute simplicity
      const simplicity = computeSimplicity(mutation);

      const runId = `exp_${i + 1}_${task.type}_${Date.now()}`;

      // v6: Proxy pre-check — predict outcome before running (for logging only;
      // expensive LLM calls still execute, but we track prediction accuracy)
      let proxyPrediction: number | null = null;
      if (this.qualityProxy.getCalibrationSize() >= 8) {
        try {
          const pred = this.qualityProxy.predict(
            task.description, task.description, task.type,
          );
          proxyPrediction = pred.predictedQuality;
          if (pred.isReliable && proxyPrediction < 0.4) {
            console.log(`[TurboContext v6] ⚠ Proxy predicts low quality (${(proxyPrediction * 100).toFixed(0)}%) for ${task.type}/${expType} — running anyway for signal`);
          }
        } catch { /* non-fatal */ }
      }

      let result: TurboContextResult;
      let status: ExperimentRun["status"] = "success";
      let crashReason: string | undefined;

      try {
        result = await this.executeWithBudget(
          task, context, tokenBudgetPerRun, timeBudgetPerRun
        );
      } catch (err) {
        // autoresearch: crash resilience — mark crash, auto-discard
        crashReason = (err as Error).message;
        status = "crash";
        this.learner.recordTrialCrash(task.type);
        // Use the cached baseline result as a placeholder
        result = baselineResultForType;
      }

      const experimentMetric = computeUnifiedMetric(
        result.finalQuality, result.costEstimate.estimatedCostUSD,
        result.totalLatency, result.totalAttempts,
        { simplicityMultiplier: simplicity },
      );

      const deltaPercent = baselineMetric.efficiency > 0
        ? ((experimentMetric.efficiency - baselineMetric.efficiency) / baselineMetric.efficiency) * 100
        : 0;

      // v6: Enhanced decision using unified metric with simplicity
      const decision: "keep" | "discard" = status === "crash" ? "discard" :
        deltaPercent >= 0 ? "keep" : "discard";

      if (mutation && status === "success") {
        // 记录到进化系统
        this.learner.recordTrial(task.type, result.finalQuality, true,
          result.architecture.estimatedTokens);
      }

      // v6: Soft ablation — learn which files causally contributed to quality
      // Runs every 5 experiments to amortize proxy cost; records in V5 for RL credit
      if (i > 0 && i % 5 === 0 && this.qualityProxy.getCalibrationSize() >= 8) {
        try {
          const lastOutput = result.generations[result.generations.length - 1];
          if (lastOutput) {
            const ablationRun = await runSoftAblation(
              task,
              `Quality: ${lastOutput.qualityScore}`,
              context,
              this.qualityProxy,
              lastOutput.executionMetrics,
            );
            recordAblationToV5(ablationRun, this.rlEngineV5);
            if (ablationRun.ranking.length > 0) {
              console.log(
                `[TurboContext v6] Ablation #${i + 1}: top file ${ablationRun.ranking[0].source} ` +
                `(Δ=${ablationRun.ranking[0].delta >= 0 ? '+' : ''}${ablationRun.ranking[0].delta.toFixed(3)})`
              );
            }
          }
        } catch { /* ablation is non-critical */ }
      }

      const wallClockMs = Date.now() - runStart;

      const run: ExperimentRun = {
        id: runId,
        taskType: task.type,
        timestamp: Date.now(),
        mutation,
        baselineMetric,
        experimentMetric,
        decision,
        deltaPercent: Math.round(deltaPercent * 100) / 100,
        wallClockMs,
        status,
        crashReason,
        runNumber: i + 1,
        experimentType: expType,
        simplicityScore: Math.round(simplicity * 1000) / 1000,
      };

      runs.push(run);

      const summary = `#${i + 1} ${task.type}/${expType}: ${mutation?.type || "baseline"} ` +
        `→ ${deltaPercent >= 0 ? "+" : ""}${deltaPercent.toFixed(2)}% ` +
        `| eff=${experimentMetric.efficiency.toFixed(2)} ` +
        `| q=${(experimentMetric.quality * 100).toFixed(0)}% ` +
        `| $${experimentMetric.cost.toFixed(4)} ` +
        `| simp=${simplicity.toFixed(2)} ` +
        (proxyPrediction !== null ? `| pred=${(proxyPrediction * 100).toFixed(0)}% ` : "") +
        `| ${decision.toUpperCase()}`;

      console.log(`[TurboContext Auto] ${summary}`);
      onProgress?.(run, summary);
    }

    // 3. 写入结果日志（autoresearch: results.tsv）
    this.learner.writeExperimentLog(runs, resultsPath);

    // 4. 输出摘要
    const summary = this.learner.getExperimentSummary(runs);
    console.log(`\n${summary}`);

    return runs;
  }

  /**
   * v3.6: Run a per-file ablation experiment.
   *
   * Selects the file with highest causal uncertainty (Thompson Sampling variance),
   * runs the same task twice — with and without that file — and measures the
   * quality delta as the file's clean causal contribution.
   *
   * Cost: 2x the normal execution cost (two independent engine instances).
   * Use sparingly — this is for learning signal quality, not for every execution.
   */
  /**
   * v6: Soft ablation using calibrated QualityProxy.
   *
   * For each source file, estimates quality with and without that file's
   * contribution using the proxy (no extra LLM calls needed).
   * Records results in V5 state for causal credit assignment.
   *
   * Causal signal precedence: Ablation delta > TD(λ) credit > Heuristic score
   */
  async ablate(
    task: Task,
    fragments: ContextFragment[],
    opts?: { workingDir?: string },
  ): Promise<import("./types.js").AblationResult | null> {
    if (this.qualityProxy.getCalibrationSize() < 8) {
      console.log("[TurboContext v6] Ablation skipped — proxy not yet calibrated (need >= 8 samples)");
      return null;
    }

    // Execute once to get output, then ablate using the proxy
    const result = await this.execute(task, fragments, opts);
    const lastGen = result.generations[result.generations.length - 1];
    const output = lastGen ? `Quality: ${lastGen.qualityScore}\n${lastGen.dimensionScores ? JSON.stringify(lastGen.dimensionScores) : ''}` : '';
    const metrics = lastGen?.executionMetrics;

    const run = await runSoftAblation(
      task, output, fragments, this.qualityProxy, metrics,
    );

    // Record in V5
    recordAblationToV5(run, this.rlEngineV5);

    // Log top causal contributors
    if (run.ranking.length > 0) {
      console.log(
        `[TurboContext v6] Ablation: top contributors — ` +
        run.ranking.slice(0, 3).map(r => `${r.source}(Δ=${r.delta >= 0 ? '+' : ''}${r.delta.toFixed(3)})`).join(", ")
      );
    }

    // Convert to legacy AblationResult format for backward compat
    return {
      taskId: task.id,
      taskType: task.type,
      timestamp: Date.now(),
      targetFile: run.ranking[0]?.source ?? fragments[0]?.source ?? "unknown",
      qualityDelta: run.ranking[0]?.delta ?? 0,
      confidence: run.results[0]?.confidence ?? 0,
      details: run,
    } as import("./types.js").AblationResult;
  }

  /**
   * 在固定预算内执行单次流水线（autoresearch: fixed budget = fair comparison）。
   * v5: fixed try/finally for exception safety, removed dead code.
   */
  private async executeWithBudget(
    task: Task,
    contextFragments: ContextFragment[],
    tokenBudget: number,
    timeBudgetSec: number,
    opts?: { workingDir?: string },
  ): Promise<TurboContextResult> {
    const savedConfig = this.learner.getConfig();
    const originalBudget = savedConfig.maxTokenBudget;
    savedConfig.maxTokenBudget = tokenBudget;

    try {
      return await this.execute(task, contextFragments, opts);
    } finally {
      savedConfig.maxTokenBudget = originalBudget;
    }
  }

  /** 创建默认任务（无 taskPool 时使用） */
  private createDefaultTask(): Task {
    return {
      id: "auto_baseline",
      description: "Review auth module for security issues and code quality",
      type: "code_review",
      qualityThreshold: 0.85,
    };
  }

  /** 创建默认上下文（无 contextPool 时使用） */
  private createDefaultContext(): ContextFragment[] {
    return [
      { id: "1", source: "src/auth/login.ts", contentType: "source",
        content: 'export async function login(email: string, password: string) {\n  const user = await db.users.findByEmail(email);\n  if (!user) throw new AuthError("User not found");\n  const valid = await bcrypt.compare(password, user.passwordHash);\n  if (!valid) throw new AuthError("Invalid password");\n  const token = generateJWT({ userId: user.id, role: user.role });\n  return { user: sanitizeUser(user), token };\n}',
        lastModified: Date.now() - 86400000, length: 320 },
      { id: "2", source: "src/auth/register.ts", contentType: "source",
        content: 'export async function register(data: RegisterInput) {\n  const existing = await db.users.findByEmail(data.email);\n  if (existing) throw new AuthError("Email already registered");\n  const hash = await bcrypt.hash(data.password, 12);\n  const user = await db.users.create({ ...data, passwordHash: hash });\n  return { user: sanitizeUser(user) };\n}',
        lastModified: Date.now() - 172800000, length: 280 },
    ];
  }

  /** 生成任务的变体（用于无 taskPool 的自主实验） */
  private varyTask(base: Task, iteration: number): Task {
    const variations = [
      { type: "code_review" as const, desc: "Review auth module for security issues and code quality" },
      { type: "code_generation" as const, desc: "Add rate limiting to login endpoint" },
      { type: "code_refactor" as const, desc: "Refactor auth middleware for better error handling" },
      { type: "debugging" as const, desc: "Debug token validation failure in auth middleware" },
      { type: "testing" as const, desc: "Write unit tests for the login function" },
      { type: "analysis" as const, desc: "Analyze performance of the current auth flow" },
      { type: "documentation" as const, desc: "Document the auth module API endpoints" },
    ];
    const v = variations[iteration % variations.length];
    return {
      id: `auto_${iteration + 1}`,
      description: v.desc,
      type: v.type,
      qualityThreshold: 0.85,
    };
  }

  /**
   * 获取学习器实例
   */
  getLearner(): Learner {
    return this.learner;
  }

  /**
   * 获取优化器实例
   */
  getOptimizer(): Optimizer {
    return this.optimizer;
  }

  /**
   * 获取当前配置
   */
  getConfig(): TurboContextConfig {
    return this.learner.getConfig();
  }

  /** v5: Get RL diagnostics from V5 engine (primary source) */
  getRLDiagnostics(): {
    predictiveAccuracy: number;
    curriculumPhase: number;
    curriculumDescription: string;
    retrievalFitness: number;
    retrievalGeneration: number;
    tdTracesActive: number;
  } {
    const status = this.rlEngineV5.getStatus();
    return {
      predictiveAccuracy: 1 - status.predictiveModelAccuracy,
      curriculumPhase: status.curriculumPhase,
      curriculumDescription: `Phase ${status.curriculumPhase} (${status.activeMemories} active memories)`,
      retrievalFitness: 0,
      retrievalGeneration: 0,
      tdTracesActive: 0,
    };
  }

  /** v5: Get the V5 RL engine for external introspection */
  getRLEngineV5(): RLEngineV5 {
    return this.rlEngineV5;
  }

  /** v6: Get the Quality Proxy for inspection and signal profiling */
  getQualityProxy(): QualityProxy {
    return this.qualityProxy;
  }

  /** Linear interpolation: blends base toward target by factor t ∈ [0,1] */
  private lerp(base: number, target: number, t: number): number {
    return base + (target - base) * Math.min(1, Math.max(0, t));
  }

  // ── Private: V5 Trial construction ──

  /**
   * Build a V5 Trial from execution data for recording in the V5 RL engine.
   * Bridges the V4 ExecutionRecord/CompressedContext into the V5 Trial format.
   */
  private buildV5Trial(
    task: Task,
    record: ExecutionRecord,
    compressed: import("./types.js").CompressedContext,
    modelTier: string,
  ): TrialV5 {
    const now = new Date().toISOString();
    const descHash = createHash("sha256")
      .update(task.description.slice(0, 256))
      .digest("hex")
      .slice(0, 16) as TrialV5["descriptionHash"];

    // Map V4 model tier names to V5 ModelTier (valid values: "fast" | "medium" | "best")
    const v5ModelTier = (modelTier === "fast" ? "fast" :
      modelTier === "medium" ? "medium" : "best") as TrialV5["modelTier"];

    // Decompose capability requirements from task type
    const capabilityRequirements = this.decomposeTaskCapabilities(task);

    return {
      id: `trial_${Date.now()}_${Math.random().toString(36).slice(2, 8)}` as TrialV5["id"],
      timestamp: now,
      context: ContextOrigin.AUTONOMOUS,
      taskType: task.type as TaskType,
      descriptionHash: descHash,
      descriptionLength: task.description.length,
      capabilityRequirements,
      compressionRatio: compressed.compressionRatio,
      compressionWeights: {
        alpha: this.learner.getConfig().alpha,
        beta: this.learner.getConfig().beta,
        gamma: this.learner.getConfig().gamma,
      },
      temperatureSchedule: this.learner.getConfig().temperatureSchedule as [number, number, number],
      modelTier: v5ModelTier,
      retrievalTopK: 5,
      tokenBudgetUsed: this.learner.getConfig().maxTokenBudget,
      maxAttempts: this.learner.getConfig().maxAttempts,
      outcome: record.qualityScore >= this.learner.getConfig().qualityThreshold ? "success" : "failure",
      qualityScores: [
        record.dimensionScores?.completeness ?? 0,
        record.dimensionScores?.correctness ?? 0,
        record.dimensionScores?.consistency ?? 0,
        record.dimensionScores?.format ?? 0,
      ],
      qualityScore: record.qualityScore,
      costUsd: record.totalCost,
      latencyMs: record.latencyMs,
      attemptCount: record.attemptCount,
      bestAttemptIndex: record.attemptCount - 1,
      predictedQuality: null,
      surprise: 0,
      counterfactuals: [],
      curriculumPhase: 0,
      retrievedMemoryIds: [],
      referencedMemoryIds: [],
      advantage: null,
      causalUtility: 0,
      herGoals: [],
    };
  }

  /** Decompose task description into capability requirements based on task type */
  private decomposeTaskCapabilities(task: Task): string[] {
    const caps: string[] = [];
    const typeMap: Record<string, string[]> = {
      code_review: ["code_understanding", "pattern_recognition", "error_detection"],
      code_generation: ["code_understanding", "code_generation", "design"],
      debugging: ["code_understanding", "error_detection", "pattern_recognition"],
      code_refactor: ["code_understanding", "code_modification", "design"],
      analysis: ["code_understanding", "pattern_recognition", "design"],
      design: ["design", "code_generation", "documentation"],
      documentation: ["code_understanding", "documentation", "pattern_recognition"],
      testing: ["code_understanding", "error_detection", "code_generation"],
      general: ["code_understanding", "pattern_recognition", "code_generation"],
    };
    return typeMap[task.type] ?? ["code_understanding", "pattern_recognition", "code_generation"];
  }
}

/** 执行结果 */
export interface TurboContextResult {
  taskId: string;
  compressed: import("./types.js").CompressedContext;
  architecture: PromptArchitecture;
  modelSelection: { tier: string; config: import("./types.js").ModelConfig; rationale: string };
  costEstimate: { estimatedCostUSD: number; estimatedLatency: string };
  generations: Array<{
    attempt: number;
    qualityScore: number;
    dimensionScores: QualityDimensions;
    latencyMs: number;
    executionMetrics?: import("./types.js").ExecutionMetrics;
  }>;
  finalQuality: number;
  totalAttempts: number;
  totalLatency: number;
  qualityTrend: { average: number; trend: string; byType: Record<string, number> };
  learningAdjustments: Array<{ component: string; changes: Record<string, string>; reason: string }>;
  executionCount: number;
  /** v3.3: RL diagnostics */
  rlDiagnostics?: {
    predictiveAccuracy: number;
    curriculumPhase: number;
    curriculumDescription: string;
    retrievalFitness: number;
    retrievalGeneration: number;
  };
}
