// ============================================================
// Phase 4: Cost-Latency Optimization
// ============================================================
// 动态模型选择和成本优化，为每个子任务选择最优模型层级
//
// 公式:
//   complexity(T) = Σᵢ wᵢ · complexity_dimensionᵢ(T)
//   如果 complexity < θ₁: fast model
//   如果 θ₁ ≤ complexity < θ₂: medium model
//   如果 complexity ≥ θ₂: deep model
//
// 缓存策略: 对相似任务使用语义哈希命中
// ============================================================

import type { Task, ModelTier, ModelConfig, ExecutionRecord } from "../types.js";

export const MODEL_TIERS: Record<ModelTier, ModelConfig> = {
  fast: {
    tier: "fast",
    model: "claude-haiku",
    costPer1KTokens: 0.00025,
    avgLatencyMs: 2000,
    capabilities: ["simple_qa", "formatting", "basic_code", "classification"],
  },
  medium: {
    tier: "medium",
    model: "claude-sonnet",
    costPer1KTokens: 0.003,
    avgLatencyMs: 5000,
    capabilities: ["code_gen", "analysis", "reasoning", "review", "debugging"],
  },
  deep: {
    tier: "deep",
    model: "claude-opus",
    costPer1KTokens: 0.015,
    avgLatencyMs: 15000,
    capabilities: ["complex_reasoning", "architecture", "security_audit", "critical_review"],
  },
};

/** 复杂度评估配置 */
interface ComplexityConfig {
  thresholdLow: number;
  thresholdHigh: number;
}

const DEFAULT_COMPLEXITY_CONFIG: ComplexityConfig = {
  thresholdLow: 0.30,
  thresholdHigh: 0.42,
};

/** 缓存条目 */
interface CacheEntry {
  taskFingerprint: string;
  result: string;
  quality: number;
  timestamp: number;
  modelUsed: ModelTier;
}

/**
 * 优化器：为任务选择最优模型并管理缓存
 */
export class Optimizer {
  private cache: Map<string, CacheEntry> = new Map();
  private config: ComplexityConfig;

  constructor(config?: Partial<ComplexityConfig>) {
    this.config = { ...DEFAULT_COMPLEXITY_CONFIG, ...config };
  }

  /**
   * 模型选择算法
   *
   * 输入: 任务 T, 复杂度评估, 延迟预算
   * 输出: 最优模型层级
   *
   * 公式:
   *   complexity = Σᵢ wᵢ · dimᵢ(T)
   *   model = f(complexity):
   *     < θ₁ → fast
   *     [θ₁, θ₂) → medium
   *     ≥ θ₂ → deep
   *
   *   同时考虑延迟预算：
   *   如果 latency_budget < fast_latency → 降级标记
   *   如果 latency_budget > deep_latency AND complexity < θ₁ → 可以使用 fast（缓存友好）
   */
  selectModel(
    task: Task,
    executionHistory: ExecutionRecord[],
    opts?: { qualityProxy?: import("./quality-proxy.js").QualityProxy },
  ): { tier: ModelTier; config: ModelConfig; rationale: string } {
    const complexity = this.estimateComplexity(task, executionHistory);

    // 基础规则
    let tier: ModelTier;
    let rationale: string;

    if (complexity < this.config.thresholdLow) {
      tier = "fast";
      rationale = `复杂度 ${complexity.toFixed(2)} < 阈值 ${this.config.thresholdLow}，使用快速模型`;
    } else if (complexity < this.config.thresholdHigh) {
      tier = "medium";
      rationale = `复杂度 ${complexity.toFixed(2)} ∈ [${this.config.thresholdLow}, ${this.config.thresholdHigh})，使用中等模型`;
    } else {
      tier = "deep";
      rationale = `复杂度 ${complexity.toFixed(2)} ≥ ${this.config.thresholdHigh}，使用深度模型`;
    }

    // v6: Quality Proxy — if proxy predicts high quality even with fast model, downgrade to save cost
    if (opts?.qualityProxy && opts.qualityProxy.getCalibrationSize() >= 5 && tier !== "fast") {
      try {
        const proxyPred = opts.qualityProxy.predict(task.description, task.type);
        if (proxyPred >= 0.80) {
          const downgradedTier: ModelTier = tier === "deep" ? "medium" : "fast";
          tier = downgradedTier;
          rationale += ` | v6 proxy predicts ${(proxyPred * 100).toFixed(0)}% quality → downgraded to ${downgradedTier}`;
        }
      } catch { /* proxy prediction failed, ignore */ }
    }

    // 延迟预算约束
    if (task.latencyBudget) {
      const modelConfig = MODEL_TIERS[tier];
      if (modelConfig.avgLatencyMs > task.latencyBudget * 1000 && tier !== "fast") {
        // 延迟预算不足，降级
        const downgradedTier: ModelTier = tier === "deep" ? "medium" : "fast";
        tier = downgradedTier;
        rationale += ` | 延迟预算 ${task.latencyBudget}s 不足，降级至 ${downgradedTier}`;
      } else if (tier === "fast" && task.latencyBudget >= 10) {
        // 如果有充足的延迟预算且任务适合快速模型，保留 fast
        rationale += ` | 延迟预算充足，保持快速模型`;
      }
    }

    return { tier, config: MODEL_TIERS[tier], rationale };
  }

  /**
   * 复杂度评估
   *
   * dimension:
   *   - task_type_complexity: 任务类型固有复杂度
   *   - description_ambiguity: 描述的模糊程度
   *   - context_requirement: 所需的上下文量
   *   - historical_complexity: 历史记录中的平均分
   */
  estimateComplexity(task: Task, history: ExecutionRecord[]): number {
    // 1. 任务类型基础复杂度
    const typeComplexities: Record<string, number> = {
      debugging: 0.35,
      code_review: 0.40,
      code_generation: 0.45,
      code_refactor: 0.55,
      testing: 0.40,
      analysis: 0.50,
      design: 0.65,
      documentation: 0.25,
      general: 0.35,
    };
    const typeComplexity = typeComplexities[task.type] || 0.35;

    // 2. 描述模糊程度（长度越短越模糊，长描述可能包含更多细节但也可能更杂乱）
    const descLen = task.description.length;
    const ambiguityScore = descLen < 20 ? 0.8 :
      descLen < 50 ? 0.5 :
      descLen < 200 ? 0.3 : 0.2;

    // 3. 历史记录中的复杂度（如果有）
    const relevantHistory = history.filter(h => h.taskType === task.type);
    let historicalComplexity = 0.3; // 默认
    if (relevantHistory.length > 0) {
      historicalComplexity = relevantHistory.reduce(
        (sum, h) => sum + (h.qualityScore < 0.8 ? 0.6 : 0.3), 0
      ) / relevantHistory.length;
    }

    // 4. 综合评分（加权平均）
    // 基础复杂度因子从 0.4 调整为 0.2，使输出范围 [0.24, 0.55]
    // 与默认阈值 θ₁=0.30, θ₂=0.50 配合，三个模型层级均可到达
    const complexity =
      0.40 * typeComplexity +
      0.15 * ambiguityScore +
      0.20 * historicalComplexity +
      0.25 * 0.2; // 基础复杂度因子

    return Math.round(Math.min(1, Math.max(0, complexity)) * 100) / 100;
  }

  /**
   * 缓存查找
   *
   * taskFingerprint = hash(task.type + task.description + context hash)
   * 如果缓存命中且相似度 > 0.95 → 返回缓存结果
   */
  lookupCache(task: Task, compressedContent: string): CacheEntry | null {
    const fingerprint = this.computeFingerprint(task, compressedContent);

    const cached = this.cache.get(fingerprint);
    if (!cached) return null;

    // 检查时效性（缓存有效期 5 分钟）
    const age = Date.now() - cached.timestamp;
    if (age > 5 * 60 * 1000) {
      this.cache.delete(fingerprint);
      return null;
    }

    // LRU: 命中后重新插入，将该条目移到 Map 末尾
    this.cache.delete(fingerprint);
    this.cache.set(fingerprint, { ...cached, timestamp: Date.now() });

    return cached;
  }

  /**
   * 写入缓存（LRU 淘汰）
   */
  writeCache(task: Task, compressedContent: string, result: string, quality: number, modelUsed: ModelTier): void {
    const fingerprint = this.computeFingerprint(task, compressedContent);

    // 如果已存在，先删除（重新插入到末尾）
    if (this.cache.has(fingerprint)) {
      this.cache.delete(fingerprint);
    }

    // 缓存上限 100 条，淘汰最久未访问的条目（Map 首部）
    if (this.cache.size >= 100) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) {
        this.cache.delete(firstKey);
      }
    }

    this.cache.set(fingerprint, {
      taskFingerprint: fingerprint,
      result,
      quality,
      timestamp: Date.now(),
      modelUsed,
    });
  }

  /**
   * 计算任务指纹（使用 SHA-256 避免缓存碰撞）
   */
  private computeFingerprint(task: Task, compressedContent: string): string {
    const { createHash } = require("crypto");
    const input = `${task.type}:${task.description}:${compressedContent.slice(0, 200)}`;
    return `${task.type}_${createHash("sha256").update(input).digest("hex").slice(0, 16)}`;
  }

  /**
   * 优化成本报告
   */
  estimateCost(
    task: Task,
    estimatedTokens: number,
    modelTier: ModelTier
  ): { estimatedCostUSD: number; estimatedLatency: string } {
    const model = MODEL_TIERS[modelTier];
    const cost = (estimatedTokens / 1000) * model.costPer1KTokens;

    // 考虑重试的期望成本
    const expectedAttempts = task.type === "code_generation" || task.type === "debugging"
      ? 1.5 : 1.2;
    const totalCost = cost * expectedAttempts;

    const latencySec = (model.avgLatencyMs / 1000) * expectedAttempts;

    return {
      estimatedCostUSD: Math.round(totalCost * 10000) / 10000,
      estimatedLatency: `${latencySec.toFixed(1)}s`,
    };
  }
}
