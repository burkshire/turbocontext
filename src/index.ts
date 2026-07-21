// ============================================================
// TurboContext: Main Entry Point (Plan B — simplified)
// ============================================================
// 5-phase pipeline: compress → compose → optimize → generate → record
// ============================================================

import type {
  Task, ContextFragment, TurboContextConfig, ExecutionRecord,
  QualityDimensions,
} from "./types.js";
import { compressContext } from "./core/compressor.js";
import { composePromptArchitecture } from "./core/composer.js";
import type { PromptArchitecture } from "./core/composer.js";
import { qualityWeightedGeneration, evaluateQuality, DEFAULT_QUALITY_CONFIG } from "./core/generator.js";
import { Optimizer, MODEL_TIERS } from "./core/optimizer.js";
import { createLLMCall, defaultLLMCall } from "./core/llm.js";
import type { LLMConfig } from "./core/llm.js";

export { compressContext } from "./core/compressor.js";
export { composePromptArchitecture } from "./core/composer.js";
export { qualityWeightedGeneration, evaluateQuality } from "./core/generator.js";
export { Optimizer, MODEL_TIERS } from "./core/optimizer.js";
export { createLLMCall, defaultLLMCall } from "./core/llm.js";
export type * from "./types.js";

// ── Configuration ──

const DEFAULTS: TurboContextConfig = {
  alpha: 0.55,
  beta: 0.20,
  gamma: 0.25,
  maxTokenBudget: 8000,
  minCoverage: 0.80,
  qualityThreshold: 0.85,
  maxAttempts: 3,
  temperatureSchedule: [0.7, 0.35, 0.1],
  complexityThresholdLow: 0.30,
  complexityThresholdHigh: 0.50,
  learningRate: 0.1,
  historyWindow: 100,
};

// ── Result ──

export interface TurboContextResult {
  finalQuality: number;
  finalDimensions: QualityDimensions;
  totalAttempts: number;
  totalLatency: number;
  costEstimate: { estimatedCostUSD: number; tokensUsed: number; tier: string };
  coverage: Record<string, number>;
  architecture: PromptArchitecture;
  generations: Array<{ attempt: number; content: string; qualityScore: number }>;
  record: ExecutionRecord;
}

// ── Engine ──

export class TurboContextEngine {
  private config: TurboContextConfig;
  private optimizer: Optimizer;
  private history: ExecutionRecord[] = [];
  private executionCount = 0;

  constructor(config: Partial<TurboContextConfig> = {}) {
    this.config = { ...DEFAULTS, ...config };
    this.optimizer = new Optimizer();
  }

  /** Execute the 5-phase pipeline for a single task. */
  async execute(
    task: Task,
    contextFragments: ContextFragment[],
    options?: { workingDir?: string; llmConfig?: LLMConfig },
  ): Promise<TurboContextResult> {
    this.executionCount++;
    const startTime = Date.now();

    // ── Phase 1: Context Compression ──
    const compressed = await compressContext(task, contextFragments, this.config);

    // ── Phase 2: Prompt Architecture ──
    const architecture = composePromptArchitecture(task, compressed);
    const estimatedTokens = architecture.estimatedTokens ?? contextFragments.reduce((s, f) => s + Math.ceil(f.length / 4), 0);

    // ── Phase 4: Cost Optimization (before generation) ──
    const modelSelection = this.optimizer.selectModel(task, this.history);
    const costEstimate = this.optimizer.estimateCost(task, estimatedTokens, modelSelection.tier);

    // Check semantic cache
    const cached = this.optimizer.lookupCache(task, contextFragments.map(f => f.content).join("\n").slice(0, 500));
    if (cached) {
      const totalLatency = Date.now() - startTime;
      const cachedQuality: QualityDimensions = { completeness: cached.quality, correctness: cached.quality, consistency: cached.quality, format: cached.quality };
      const cacheCost = { estimatedCostUSD: costEstimate.estimatedCostUSD, tokensUsed: estimatedTokens, tier: modelSelection.tier };
      return this.buildResult(task, cached.quality, cachedQuality, 1, totalLatency, cacheCost, architecture);
    }

    // ── Phase 3: Quality-Weighted Generation ──
    const llmCall = options?.llmConfig ? createLLMCall(options.llmConfig) : defaultLLMCall;
    const qualityConfig = {
      ...DEFAULT_QUALITY_CONFIG,
      threshold: task.qualityThreshold ?? this.config.qualityThreshold,
      maxAttempts: this.config.maxAttempts,
      temperatureSchedule: this.config.temperatureSchedule,
    };

    let finalOutput = "";
    let finalQuality = 0;
    let finalDimensions: QualityDimensions = { completeness: 0, correctness: 0, consistency: 0, format: 0 };
    let totalAttempts = 0;
    const allGenerations: Array<{ attempt: number; content: string; qualityScore: number }> = [];

    const genIterator = qualityWeightedGeneration(task, architecture, qualityConfig, llmCall);
    for await (const gen of genIterator) {
      allGenerations.push({ attempt: gen.attempt, content: gen.content, qualityScore: gen.qualityScore });
      finalOutput = gen.content;
      finalQuality = gen.qualityScore;
      finalDimensions = gen.dimensionScores;
      totalAttempts = gen.attempt;
    }

    const totalLatency = Date.now() - startTime;

    // ── Phase 5: Record ──
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
    };
    this.history.push(record);

    // Write to semantic cache for future lookups
    this.optimizer.writeCache(
      task, contextFragments.map(f => f.content).join("\n").slice(0, 500),
      finalOutput, finalQuality, modelSelection.tier,
    );

    return {
      finalQuality,
      finalDimensions,
      totalAttempts,
      totalLatency,
      costEstimate: { estimatedCostUSD: costEstimate.estimatedCostUSD, tokensUsed: estimatedTokens, tier: modelSelection.tier },
      coverage: compressed.coverage,
      architecture,
      generations: allGenerations,
      record,
    };
  }

  private buildResult(
    task: Task, qualityScore: number, dimensionScores: QualityDimensions,
    attempts: number, latency: number,
    costEstimate: { estimatedCostUSD: number; tokensUsed: number; tier: string },
    architecture: PromptArchitecture,
  ): TurboContextResult {
    return {
      finalQuality: qualityScore,
      finalDimensions: dimensionScores,
      totalAttempts: attempts,
      totalLatency: latency,
      costEstimate,
      coverage: {},
      architecture,
      generations: [],
      record: {
        taskId: task.id,
        taskType: task.type,
        timestamp: Date.now(),
        compressionRatio: 0,
        qualityScore,
        totalCost: costEstimate.estimatedCostUSD,
        latencyMs: latency,
        attemptCount: attempts,
        modelUsed: costEstimate.tier,
        coverage: {},
        dimensionScores,
      },
    };
  }

  getExecutionCount(): number { return this.executionCount; }
  getHistory(): ExecutionRecord[] { return this.history; }
  getConfig(): TurboContextConfig { return { ...this.config }; }
}
