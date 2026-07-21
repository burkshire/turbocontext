// ============================================================
// Phase 3: Quality-Weighted Generation
// ============================================================
// 核心算法：质量门控的自适应生成
//
// 伪代码:
//   for attempt = 1 to maxAttempts:
//     temp = temperatureSchedule[attempt]
//     output = LLM.invoke(prompt, temp)
//     q_score = evaluateQuality(output)
//     if q_score >= threshold → 返回 output
//     否则生成 feedback，注入到下一轮 prompt
// ============================================================

import type { Task, GenerationOutput, QualityDimensions, QualityAssessment, QualityIssue, UnifiedMetric } from "../types.js";
import type { PromptArchitecture } from "./composer.js";
import { selectVerifier, blendedQuality, type Verifier, type VerificationResult } from "./verifier.js";
import { QualityProxy } from "./quality-proxy.js";

// Re-export for consumers
export { QualityProxy } from "./quality-proxy.js";

/** 默认质量阈值配置 */
export const DEFAULT_QUALITY_CONFIG = {
  qualityThreshold: 0.85,
  maxAttempts: 3,
  temperatureSchedule: [0.7, 0.35, 0.1],
};

/**
 * 质量加权生成主流程
 *
 * 输入: 提示架构 P, 质量要求 Q_threshold
 * 输出: 经过质量门控的生成结果
 *
 * 公式:
 *   attempt(k): oₖ ~ LLM(Pₖ, tₖ)
 *   Q(oₖ) = Σᵢ wᵢ · qᵢ(oₖ) , 其中 qᵢ ∈ {completeness, correctness, consistency, format}
 *   if Q(oₖ) ≥ θ: return oₖ
 *   else: fₖ = critique(oₖ), Pₖ₊₁ = Pₖ ⊕ fₖ
 */
export async function* qualityWeightedGeneration(
  task: Task,
  architecture: PromptArchitecture,
  config: Partial<typeof DEFAULT_QUALITY_CONFIG> & {
    workingDir?: string;
    sourceFiles?: string[];
    qualityProxy?: import("./quality-proxy.js").QualityProxy;
  } = {},
  llmCall: (prompt: string, temp: number) => Promise<string> = defaultLLMCall,
): AsyncGenerator<GenerationOutput, GenerationOutput, unknown> {
  const cfg = { ...DEFAULT_QUALITY_CONFIG, ...config };
  let currentPrompt = formatArchitecture(architecture);

  for (let attempt = 1; attempt <= cfg.maxAttempts; attempt++) {
    const tempIndex = Math.min(attempt - 1, cfg.temperatureSchedule.length - 1);
    const temperature = cfg.temperatureSchedule[tempIndex] ?? 0.1;
    const startTime = Date.now();

    // v6: Quality Proxy pre-check — predict BEFORE LLM call to skip if hopeless
    const preCheckProxy = (cfg as Record<string, unknown>).qualityProxy as QualityProxy | undefined;
    if (preCheckProxy && preCheckProxy.getCalibrationSize() >= 5 && attempt >= 2) {
      try {
        const preCheckPred = preCheckProxy.predict(
          "", // no content yet — structural prediction only
          task.description,
          task.type,
          undefined,
          attempt,
        );
        if (preCheckPred.predictedQuality < cfg.qualityThreshold * 0.7) {
          console.log(
            `[TurboContext v6] Proxy pre-check: predicted ${(preCheckPred.predictedQuality * 100).toFixed(0)}% ` +
            `< threshold ${(cfg.qualityThreshold * 70).toFixed(0)}% — injecting feedback, skipping LLM call`
          );
          // Inject feedback as if we got a low-quality result
          const fakeAssessment = evaluateQuality("", task, 0);
          fakeAssessment.score = preCheckPred.predictedQuality;
          const feedback = generateFeedback(fakeAssessment, currentPrompt);
          currentPrompt = injectFeedback(currentPrompt, feedback);
          // log as a skipped attempt (no token cost)
          yield {
            attempt, content: "[v6 proxy skip]", qualityScore: preCheckPred.predictedQuality,
            dimensionScores: fakeAssessment.dimensions, latencyMs: 1,
            attemptAttempted: attempt, attemptsAttempted: cfg.maxAttempts,
            feedbackInjected: true,
          } as GenerationOutput;
          continue;
        }
      } catch { /* pre-check failure is non-fatal */ }
    }

    // 生成
    const content = await llmCall(currentPrompt, temperature);
    const latencyMs = Date.now() - startTime;

    // 质量评估（使用分支特定阈值如果已配置）
    const assessment = evaluateQuality(content, task, cfg.qualityThreshold);

    // v6: Quality Proxy (PACE-inspired) — learned quality prediction from cheap signals
    const qualityProxy = (cfg as Record<string, unknown>).qualityProxy as QualityProxy | undefined;
    let proxyPrediction: number | null = null;
    if (qualityProxy && qualityProxy.getCalibrationSize() >= 8) {
      try {
        const prediction = qualityProxy.predict(
          content,
          task.description,
          task.type,
          undefined, // executionMetrics not available during generation
          attempt + 1,
        );
        proxyPrediction = prediction.predictedQuality;
      } catch (_err) {
        // Proxy failure is non-fatal
      }
    }

    // v3.4/v3.5: Run verifier for hard signal (Karpathy approach)
    let verifierResult: VerificationResult | null = null;
    try {
      const verifier = await selectVerifier(task);
      const wd = (cfg as Record<string, unknown>).workingDir as string | undefined;
      verifierResult = await verifier.verify(content, task, {
        workingDir: wd,
        sourceFiles: (cfg as Record<string, unknown>).sourceFiles as string[] | undefined,
      });
    } catch (_err) {
      // Verifier crash is non-fatal — fall back to regex assessment
    }

    // 估算 token
    const tokensUsed = Math.ceil(content.length / 3.5);

    // v3.5: Boost verifier weight when execution verification runs (compilation metrics present)
    const hasExecutionMetrics = verifierResult?.metrics?.compiled != null;
    // v3.9: Compilation is the GROUND TRUTH signal (agent.py val_bpb equivalent).
    // When code doesn't compile, quality is hard-capped regardless of regex score.
    const compiled = verifierResult?.metrics?.compiled === 1;
    const verifierWeight = hasExecutionMetrics ? 0.9 : 0.7;

    let effectiveScore: number;
    if (hasExecutionMetrics && !compiled) {
      // v3.9: Compilation FAILED → hard cap. Code that doesn't compile cannot be "high quality."
      const capped = Math.min(assessment.score, 0.45);
      effectiveScore = blendedQuality(capped, verifierResult!, verifierWeight).score;
      if (attempt === 1) {
        console.log(`[TurboContext] ⚠ Compilation FAILED (${verifierResult!.metrics!.errors} errors) → score capped at ${(capped * 100).toFixed(0)}%`);
      }
    } else if (hasExecutionMetrics && compiled) {
      effectiveScore = blendedQuality(assessment.score, verifierResult!, verifierWeight).score;
      if (attempt === 1) {
        console.log(`[TurboContext] ✓ Compilation passed → blended score ${(effectiveScore * 100).toFixed(0)}%`);
      }
    } else {
      effectiveScore = verifierResult
        ? blendedQuality(assessment.score, verifierResult, verifierWeight).score
        : assessment.score;
    }

    // v6: Blend Quality Proxy prediction into effective score
    // Proxy weight grows with calibration data size (max 0.4 blend with regex+verifier)
    if (proxyPrediction !== null) {
      const proxyWeight = Math.min(0.4, qualityProxy!.getCalibrationSize() / 50);
      effectiveScore = effectiveScore * (1 - proxyWeight) + proxyPrediction * proxyWeight;
    }
    const effectivePassed = effectiveScore >= cfg.qualityThreshold;

    const output: GenerationOutput = {
      content,
      qualityScore: effectiveScore,
      dimensionScores: assessment.dimensions,
      attempt,
      modelUsed: "dynamic",
      tokensUsed,
      latencyMs,
      // v3.5: Carry execution metrics through to the record
      executionMetrics: hasExecutionMetrics ? {
        compiled: verifierResult!.metrics!.compiled === 1,
        compilerExitCode: (verifierResult!.metrics!.exitCode as number) ?? null,
        compilerErrors: (verifierResult!.metrics!.errors as number) ?? 0,
        compilerWarnings: (verifierResult!.metrics!.warnings as number) ?? 0,
        projectType: verifierResult!.metrics!.projectType === 1 ? "typescript" : "unknown",
      } : undefined,
    };

    yield output;

    // 质量门控
    if (effectivePassed) {
      return output;
    }

    // 生成反馈并注入到下一轮
    // v3.9: When compilation fails, inject actual compiler errors into feedback
    // so the LLM can fix specific type errors rather than guessing.
    const feedback = verifierResult && !verifierResult.passed
      ? (hasExecutionMetrics && !compiled
          ? generateCompilerErrorFeedback(verifierResult, output)
          : generateFeedbackFromVerifier(verifierResult, output))
      : generateFeedback(assessment, output);
    currentPrompt = injectFeedback(currentPrompt, feedback, attempt);
  }

  // 所有尝试结束，返回最后一次的结果
  return await (async () => {
    const content = await llmCall(currentPrompt, 0.1);
    const assessment = evaluateQuality(content, task, cfg.qualityThreshold);
    return {
      content,
      qualityScore: assessment.score,
      dimensionScores: assessment.dimensions,
      attempt: cfg.maxAttempts,
      modelUsed: "dynamic",
      tokensUsed: Math.ceil(content.length / 3.5),
      latencyMs: 0,
    };
  })();
}

// ============================================================
// 质量评估引擎
// ============================================================

/**
 * 质量评估
 *
 * Q(o) = w₁ · q_completeness + w₂ · q_correctness + w₃ · q_consistency + w₄ · q_format
 *
 * 权重根据任务类型自动调整
 */
export function evaluateQuality(
  output: string,
  task: Task,
  threshold?: number,
): QualityAssessment {
  const effectiveThreshold = threshold ?? DEFAULT_QUALITY_CONFIG.qualityThreshold;
  const weights = getQualityWeights(task.type);

  const dimensions = {
    completeness: assessCompleteness(output, task),
    correctness: assessCorrectness(output, task),
    consistency: assessConsistency(output, task),
    format: assessFormat(output, task),
  };

  const score =
    weights.completeness * dimensions.completeness +
    weights.correctness * dimensions.correctness +
    weights.consistency * dimensions.consistency +
    weights.format * dimensions.format;

  const issues = detectIssues(output, dimensions);

  return {
    score: Math.round(score * 10000) / 10000,
    dimensions,
    issues,
    passed: score >= effectiveThreshold,
  };
}

/**
 * 获取任务类型对应的质量维度权重
 */
function getQualityWeights(taskType: string): QualityDimensions {
  const weightMap: Record<string, QualityDimensions> = {
    code_generation: { completeness: 0.30, correctness: 0.35, consistency: 0.20, format: 0.15 },
    code_review:     { completeness: 0.25, correctness: 0.25, consistency: 0.30, format: 0.20 },
    code_refactor:   { completeness: 0.20, correctness: 0.40, consistency: 0.30, format: 0.10 },
    debugging:       { completeness: 0.20, correctness: 0.50, consistency: 0.20, format: 0.10 },
    testing:         { completeness: 0.35, correctness: 0.30, consistency: 0.20, format: 0.15 },
    analysis:        { completeness: 0.30, correctness: 0.25, consistency: 0.25, format: 0.20 },
    documentation:   { completeness: 0.30, correctness: 0.15, consistency: 0.25, format: 0.30 },
    design:          { completeness: 0.25, correctness: 0.20, consistency: 0.30, format: 0.25 },
    general:         { completeness: 0.30, correctness: 0.25, consistency: 0.25, format: 0.20 },
  };
  return weightMap[taskType] || weightMap.general;
}

/**
 * 完整性评估
 *
 * 检查输出是否覆盖了任务的所有关键方面。
 * 使用 token 级关键词覆盖 + 代码结构完整性双层评估。
 */
function assessCompleteness(output: string, task: Task): number {
  const text = output.toLowerCase();
  const taskText = task.description.toLowerCase();

  // 1. 提取原子关键词
  const keywords = extractKeywords(taskText);

  if (keywords.length === 0) return 0.8;

  // 2. Token 级覆盖：拆分关键词为独立词，按匹配比例打分
  let totalWeight = 0;
  let matchedWeight = 0;
  for (const kw of keywords) {
    const parts = kw.split(/[\s\-_]+/);
    const matchCount = parts.filter(p => text.includes(p)).length;
    totalWeight += parts.length;
    matchedWeight += matchCount;
  }
  let score = totalWeight > 0 ? matchedWeight / totalWeight : 0.8;

  // 3. 代码/结构化任务：叠加结构完整性评估
  if (["code_generation", "code_refactor", "debugging", "testing"].includes(task.type)) {
    const structuralScore = assessStructuralCompleteness(output, task);
    score = 0.5 * score + 0.5 * structuralScore;
  }

  return Math.min(1, Math.round(score * 100) / 100);
}

/** 英文停用词 */
const STOP_WORDS = new Set([
  "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
  "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
  "been", "being", "have", "has", "had", "do", "does", "did", "will",
  "would", "could", "should", "may", "might", "can", "shall", "this",
  "that", "these", "those", "it", "its", "we", "you", "he", "she",
  "they", "i", "me", "my", "your", "our", "not", "no", "nor",
  "so", "if", "than", "then", "just", "also", "very", "too", "all",
  "each", "every", "both", "few", "some", "any", "most", "other",
  "into", "over", "about", "up", "out", "after", "before", "between",
  "add", "need", "use", "make", "get", "set", "put", "let",
]);

/** 中文停用词 */
const CN_STOP_WORDS = new Set([
  "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
  "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
  "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
  "所", "为", "所以", "因为", "但是", "然而", "可以", "这个", "那个",
]);

/**
 * 从任务描述中提取原子关键词。
 * 将复合短语拆分为独立单词，过滤停用词，保留 bigram 短语做精准匹配。
 */
function extractKeywords(text: string): string[] {
  // 1. 从动词短语中提取关键名词短语
  const nounPhrases = extractNounPhrases(text);

  // 2. 将所有短语拆分为原子单词
  const allWords = new Set<string>();
  const bigrams: string[] = [];

  for (const phrase of nounPhrases) {
    const words = phrase.split(/[\s\-_]+/).filter(w => w.length > 0);
    for (const w of words) {
      const clean = w.replace(/[.,;!?()[\]{}"'`]/g, "").trim();
      if (clean.length >= 3 && !STOP_WORDS.has(clean) && !CN_STOP_WORDS.has(clean)) {
        allWords.add(clean);
      }
    }
    // 生成 bigram（两词组合），比单关键词更精准
    for (let i = 0; i < words.length - 1; i++) {
      const w1 = words[i].replace(/[.,;!?()[\]{}"'`]/g, "").trim();
      const w2 = words[i + 1].replace(/[.,;!?()[\]{}"'`]/g, "").trim();
      if (w1.length >= 2 && w2.length >= 2
          && !STOP_WORDS.has(w1) && !STOP_WORDS.has(w2)) {
        bigrams.push(`${w1} ${w2}`.toLowerCase());
      }
    }
  }

  // 3. 也提取描述中的独立内容词（不依赖动词模式匹配）
  const descWords = text
    .replace(/[.,;!?()[\]{}"'`\n]/g, " ")
    .split(/\s+/)
    .filter(w => {
      const clean = w.toLowerCase().trim();
      return clean.length >= 3
        && !STOP_WORDS.has(clean)
        && !CN_STOP_WORDS.has(clean);
    })
    .map(w => w.toLowerCase());

  for (const w of descWords) {
    if (!STOP_WORDS.has(w) && !CN_STOP_WORDS.has(w)) {
      allWords.add(w);
    }
  }

  // 4. 组合：bigrams（高权重但少）+ 单词（广覆盖）
  const result: string[] = [...bigrams];
  for (const w of allWords) {
    if (!result.includes(w)) {
      result.push(w);
    }
  }

  return result.slice(0, 20);
}

/**
 * 从任务文本中提取名词短语。
 * 使用动词作为锚点，提取后面的宾语部分；同时提取连接词后的名词短语。
 */
function extractNounPhrases(text: string): string[] {
  const phrases: string[] = [];

  // 动词锚点模式：动词 + (最多4个词的名词短语)
  const verbAnchors = [
    /\b(?:implement|create|write|add|modify|update|delete|refactor|optimize|test|deploy|build|setup|configure|fix|handle|support|enable|migrate|convert|replace|remove)\s+(\S+(?:\s+\S+){0,4})/gi,
    /(?:实现|创建|编写|写|修改|增加|删除|更新|重构|优化|测试|部署|构建|配置|修复|处理|支持|启用|迁移|转换|替换|移除)\s*(\S+(?:\s+\S+){0,3})/g,
  ];

  for (const pattern of verbAnchors) {
    let match;
    while ((match = pattern.exec(text)) !== null) {
      const phrase = match[1].toLowerCase().replace(/[.,;!?]$/g, "").trim();
      if (phrase.length > 1 && phrase.length < 80) {
        phrases.push(phrase);
      }
    }
  }

  // 连接词模式：with/using/for/via/by + 名词短语
  const connectorPatterns = [
    /\b(?:with|using|for|via|by|and|including|like)\s+(\S+(?:\s+\S+){0,3})/gi,
    /(?:用|通过|以及|包括|比如)\s*(\S+(?:\s+\S+){0,2})/g,
  ];

  for (const pattern of connectorPatterns) {
    let match;
    while ((match = pattern.exec(text)) !== null) {
      const phrase = match[1].toLowerCase().replace(/[.,;!?]$/g, "").trim();
      if (phrase.length > 1 && phrase.length < 80) {
        phrases.push(phrase);
      }
    }
  }

  return phrases;
}

/**
 * 代码/结构化输出的结构完整性评估。
 * 根据任务类型检查输出中是否有预期的结构元素。
 */
function assessStructuralCompleteness(output: string, task: Task): number {
  const checks: Array<{ name: string; pattern: RegExp; weight: number }> = [];

  if (task.type === "code_generation" || task.type === "code_refactor") {
    // 函数/方法定义
    checks.push(
      { name: "function_def", pattern: /\b(function|async\s+function|=>|\w+\s*\([^)]*\)\s*[:{=]|:\s*(Promise|void|string|number|boolean|any)\b)/i, weight: 0.15 },
      { name: "type_def", pattern: /\b(interface|type\s+\w+\s*=|enum|class)\b/i, weight: 0.10 },
      { name: "error_handling", pattern: /\b(try\b|catch\b|throw\s+new|\.catch\s*\(|Error\b)/i, weight: 0.10 },
      { name: "imports", pattern: /\b(import\s+|from\s+['"]|require\s*\()/i, weight: 0.05 },
      { name: "exports", pattern: /\b(export\s+(default\s+)?(function|class|const|interface|type|async)|module\.exports)/i, weight: 0.05 },
      { name: "code_block", pattern: /```[\s\S]*?```/, weight: 0.10 },
    );
  } else if (task.type === "testing") {
    checks.push(
      { name: "test_fn", pattern: /\b(test|it|describe)\s*\(/i, weight: 0.20 },
      { name: "assertions", pattern: /\b(expect|assert|assertEqual|should|\.to\.|\.toEqual)\b/i, weight: 0.15 },
      { name: "imports", pattern: /\b(import\s+|require\s*\()/i, weight: 0.10 },
      { name: "code_block", pattern: /```[\s\S]*?```/, weight: 0.10 },
    );
  } else if (task.type === "debugging") {
    checks.push(
      { name: "root_cause", pattern: /\b(root\s*cause|issue|problem|caused\s+by|because)\b/i, weight: 0.15 },
      { name: "solution", pattern: /\b(fix|solution|resolve|patch|change|update)\b/i, weight: 0.15 },
      { name: "code_block", pattern: /```[\s\S]*?```/, weight: 0.10 },
    );
  } else {
    // 非代码任务：检查是否有结构化内容
    checks.push(
      { name: "headings", pattern: /^#{1,4}\s+\S/m, weight: 0.10 },
      { name: "lists", pattern: /^[\s]*[-*\d]+[.)]\s+\S/m, weight: 0.10 },
      { name: "paragraphs", pattern: /\S.{40,}/, weight: 0.10 },
    );
  }

  if (checks.length === 0) return 0.8;

  let passed = 0;
  let totalWeight = 0;
  for (const check of checks) {
    totalWeight += check.weight;
    if (check.pattern.test(output)) {
      passed += check.weight;
    }
  }

  return totalWeight > 0 ? passed / totalWeight : 0.8;
}

/**
 * 正确性评估
 *
 * 检测输出中的潜在错误信号
 */
function assessCorrectness(output: string, task: Task): number {
  const text = output.toLowerCase();

  // 正确性信号
  let score = 1.0;

  // 错误信号列表
  const errorSignals = [
    { pattern: /(i'?m\s+(not|un|in)|i\s+can'?t|i\s+cannot|i\s+do not\s+know)/i, penalty: 0.15 },
    { pattern: /(sorry|apologize|i\s+don'?t\s+know|as\s+an\s+ai)/i, penalty: 0.10 },
    { pattern: /(placeholder|todo|fixme|to\s+do|not\s+implemented)/i, penalty: 0.20 },
    { pattern: /(assuming|guess|might\s+be|perhaps|maybe|probably)/i, penalty: 0.05 },
    { pattern: /(incomplete|partial|rough\s+draft|unfinished)/i, penalty: 0.10 },
  ];

  for (const { pattern, penalty } of errorSignals) {
    if (pattern.test(text)) {
      score -= penalty;
    }
  }

  // 代码相关的额外检查
  if (task.type === "code_generation" || task.type === "code_refactor") {
    // 检查是否有语法错误模式
    const syntaxIssues = [
      /undefined\s+is\s+not\s+a\s+function/i,
      /cannot\s+read\s+property/i,
      /is\s+not\s+defined/i,
      /typeerror|referenceerror|syntaxerror/i,
    ];
    for (const pattern of syntaxIssues) {
      if (pattern.test(text)) {
        score -= 0.25;
      }
    }
  }

  return Math.max(0, Math.round(score * 100) / 100);
}

/**
 * 一致性评估
 *
 * 检查输出内部是否自洽，以及与上下文是否一致
 */
function assessConsistency(output: string, task: Task): number {
  const text = output.toLowerCase();

  let score = 1.0;

  // 1. 检查术语一致性（同一概念是否使用同一术语）
  const termVariations = [
    [/user[\s-]?id/g, /user[\s-]?identifier/g],
    [/api[\s-]?key/g, /api[\s-]?token/g],
    [/db/g, /database/g, /data[\s-]?store/g],
  ];

  for (const variations of termVariations) {
    const found = variations.filter(v => text.match(v));
    if (found.length > 1) {
      score -= 0.05 * (found.length - 1); // 轻罚
    }
  }

  // 2. 检查矛盾信号
  const contradictionPatterns = [
    [/do\s+not\s+use/i, /use\s+this/i, /recommend\s+using/i],
    [/never\s+use/i, /always\s+use/i],
    [/not\s+recommended/i, /best\s+practice/i],
  ];

  for (const patterns of contradictionPatterns) {
    const found = patterns.filter(p => p.test(text));
    if (found.length > 1) {
      score -= 0.1;
    }
  }

  // 3. 数字一致性
  const numbers = text.match(/\b\d+\b/g);
  if (numbers && numbers.length > 1) {
    // 检查是否有明显的数字矛盾（如 "3 steps" 但只列出 2 个）
    // 这个检查较复杂，暂时简化为检查是否有 count 声明与实际不符
    const countPatterns = text.match(/(\d+)\s*(steps|points|parts|sections|reasons|ways|methods)/gi);
    if (countPatterns) {
      for (const match of countPatterns) {
        const num = parseInt(match.match(/\d+/)?.[0] || "0");
        const unit = match.replace(/\d+\s*/, "").trim();
        if (num > 0) {
          // 尝试找到对应的列举项
          const listItems = text.match(new RegExp(`(?:^|\\n)\\s*[\\d\\.\\-\\*]`, "gm"));
          const itemCount = listItems?.length || 0;
          // 如果声明的数量远大于实际列举数量，减分
          if (num > itemCount * 1.5 && itemCount > 0) {
            score -= 0.05;
          }
        }
      }
    }
  }

  return Math.max(0, Math.round(score * 100) / 100);
}

/**
 * 格式合规性评估
 */
function assessFormat(output: string, task: Task): number {
  let score = 1.0;

  // 1. 如果期望代码但没有代码块
  if (["code_generation", "code_refactor", "debugging"].includes(task.type)) {
    if (!output.includes("```") && !output.includes("`")) {
      score -= 0.3;
    }
  }

  // 2. 代码块是否完整（有成对的反引号）
  const codeBlockStarts = (output.match(/```/g) || []).length;
  if (codeBlockStarts % 2 !== 0) {
    score -= 0.4;
  }

  // 3. 是否有明显格式问题
  if (output.length > 0 && output.length < 10) {
    score -= 0.5; // 输出太短，可能不完整
  }

  // 4. 换行结构
  const lines = output.split("\n");
  const longLines = lines.filter(l => l.length > 200 && !l.startsWith("```"));
  if (longLines.length > 3) {
    score -= 0.1;
  }

  return Math.max(0, Math.round(score * 100) / 100);
}

/**
 * 检测具体问题
 */
function detectIssues(output: string, dimensions: QualityDimensions): QualityIssue[] {
  const issues: QualityIssue[] = [];
  const text = output.toLowerCase();

  if (dimensions.completeness < 0.6) {
    issues.push({
      dimension: "completeness",
      severity: "critical",
      description: "输出覆盖率不足，缺少关键内容",
      suggestion: "请确保覆盖任务描述中的所有关键需求和约束",
    });
  } else if (dimensions.completeness < 0.8) {
    issues.push({
      dimension: "completeness",
      severity: "major",
      description: "部分内容缺失",
      suggestion: "检查是否遗漏了某些子任务或边界情况",
    });
  }

  if (dimensions.correctness < 0.7) {
    issues.push({
      dimension: "correctness",
      severity: "critical",
      description: "输出中存在潜在错误",
      suggestion: "使用更多确定性语言，避免猜测和不完整实现",
    });
  }

  if (dimensions.consistency < 0.7) {
    issues.push({
      dimension: "consistency",
      severity: "major",
      description: "内部不一致",
      suggestion: "统一术语，确保前后论述一致",
    });
  }

  if (dimensions.format < 0.6) {
    issues.push({
      dimension: "format",
      severity: "major",
      description: "格式不符合要求",
      suggestion: "严格遵循指定的输出格式，使用正确的代码块标记",
    });
  }

  return issues;
}

// ============================================================
// 反馈生成与注入
// ============================================================

/**
 * 生成反馈文本注入到下一轮 prompt
 */
function generateFeedback(assessment: QualityAssessment, output: GenerationOutput): string {
  const criticalIssues = assessment.issues.filter(i => i.severity === "critical");
  const majorIssues = assessment.issues.filter(i => i.severity === "major");

  const parts: string[] = [];

  parts.push(`## 质量反馈（第 ${output.attempt} 轮）`);
  parts.push(`总体质量评分: ${(assessment.score * 100).toFixed(1)}%`);
  parts.push(`目标阈值: ${(DEFAULT_QUALITY_CONFIG.qualityThreshold * 100)}%`);
  parts.push("");

  // 维度得分
  parts.push("### 各维度得分");
  const dims = assessment.dimensions;
  for (const [key, val] of Object.entries(dims)) {
    const bar = "█".repeat(Math.round(val * 10)) + "░".repeat(Math.round((1 - val) * 10));
    parts.push(`- ${key}: ${bar} ${(val * 100).toFixed(0)}%`);
  }
  parts.push("");

  // 问题描述
  if (criticalIssues.length > 0 || majorIssues.length > 0) {
    parts.push("### 需要改进的问题");
    for (const issue of criticalIssues) {
      parts.push(`- [严重] ${issue.description}`);
      parts.push(`  → ${issue.suggestion}`);
    }
    for (const issue of majorIssues) {
      parts.push(`- [重要] ${issue.description}`);
      parts.push(`  → ${issue.suggestion}`);
    }
    parts.push("");
  }

  // 具体改进指令
  parts.push("### 改进要求");
  if (assessment.dimensions.completeness < 0.8) {
    parts.push("- 请更全面地覆盖任务要求，确保没有遗漏");
  }
  if (assessment.dimensions.correctness < 0.8) {
    parts.push("- 请使用更确定的语言，避免推测性表述");
  }
  if (assessment.dimensions.consistency < 0.8) {
    parts.push("- 请保持术语和论点的一致性");
  }
  if (assessment.dimensions.format < 0.8) {
    parts.push("- 请严格遵守输出格式要求");
  }

  return parts.join("\n");
}

/**
 * v3.4: Generate feedback from verifier result (Karpathy approach).
 *
 * Unlike the regex-based feedback which says vague things like
 * "improve completeness", verifier feedback is specific and actionable:
 * "3 empty function bodies found" or "No file:line references in review".
 */
function generateFeedbackFromVerifier(
  verifierResult: VerificationResult,
  output: GenerationOutput,
): string {
  const parts: string[] = [];
  parts.push(`## 验证反馈（第 ${output.attempt} 轮）`);
  parts.push(`验证结果: ${verifierResult.passed ? "通过" : "未通过"}`);
  parts.push(`硬信号分数: ${(verifierResult.hardSignal * 100).toFixed(0)}%`);
  parts.push("");
  parts.push("### 具体问题");
  // Split details into individual actionable items
  const detailItems = verifierResult.details.split("; ").filter(d => d.length > 0);
  for (const detail of detailItems) {
    parts.push(`- ${detail}`);
  }
  parts.push("");
  parts.push("### 改进要求");
  parts.push("请修复以上具体问题。不要泛泛改进——针对每一条验证发现的问题逐一修正。");

  return parts.join("\n");
}

/**
 * v3.9: Generate feedback from compilation failure, including actual compiler errors.
 *
 * Adapted from agent.py pattern: when execution fails, feed the actual error
 * output back to the LLM so it can fix specific issues rather than guessing.
 * This is the key feedback loop — the LLM sees what tsc complained about and fixes it.
 */
function generateCompilerErrorFeedback(
  verifierResult: VerificationResult,
  output: GenerationOutput,
): string {
  const parts: string[] = [];
  const errors = verifierResult.metrics?.errors ?? 0;
  const warnings = verifierResult.metrics?.warnings ?? 0;

  parts.push(`## Compilation Failure (Attempt ${output.attempt})`);
  parts.push(`Your code has ${errors} compilation error(s) and ${warnings} warning(s).`);
  parts.push("");
  parts.push("### Compiler Errors (fix these first):");
  // Parse verifier details for error messages
  const detailItems = verifierResult.details.split("; ").filter(
    d => d.includes("error TS") || d.includes(": error") || d.includes("Compilation failed") || d.includes("SyntaxError")
  );
  if (detailItems.length > 0) {
    for (const detail of detailItems.slice(0, 10)) {
      parts.push(`- ${detail}`);
    }
  } else {
    // Fallback: show raw details
    parts.push(`- ${verifierResult.details.substring(0, 500)}`);
  }
  parts.push("");
  parts.push("### Required Changes");
  parts.push("Fix the above compilation errors. Do NOT add new features or change the overall design.");
  parts.push("Only fix what the compiler complained about. Keep changes minimal.");

  return parts.join("\n");
}

/**
 * 将反馈注入到 prompt 中
 */
function injectFeedback(prompt: string, feedback: string, attempt: number): string {
  // Remove previous feedback if any (language-agnostic: strip between markers or after last round separator)
  const cleanPrompt = prompt.replace(/\n---\n## Quality Feedback[\s\S]*$/, "").replace(/\n## 质量反馈[\s\S]*$/, "");

  // 在末尾追加新的反馈
  return `${cleanPrompt}\n\n${feedback}\n\n请根据以上反馈改进你的输出。质量必须达到阈值。`;
}

/**
 * 将架构格式化为单个 prompt 字符串
 */
function formatArchitecture(architecture: PromptArchitecture): string {
  return architecture.rounds.map((round, i) => {
    return `=== Round ${round.sequence} ===\n\n` +
      `[System]\n${round.systemPrompt}\n\n` +
      `[User]\n${round.userPrompt}\n`;
  }).join("\n---\n");
}

/** 默认 LLM 调用（占位，实际使用时替换） */
async function defaultLLMCall(prompt: string, temperature: number): Promise<string> {
  return `[TurboContext] Simulated output for prompt (${prompt.length} chars, temp=${temperature})`;
}

// ============================================================
// v3.0 — Unified metric（autoresearch val_bpb 等价物）
// ============================================================

const METRIC_EPSILON = 0.0001; // 避免除以零

/**
 * 计算统一效率指标。
 *
 * efficiency = qualityScore / (totalCost + ε)
 *
 * 这是 autoresearch 的 val_bpb 等价物——单一数字，
 * 使所有实验在同一尺度上直接可比。
 * 越高越好：高质量、低成本 = 高效率。
 */
export function computeUnifiedMetric(
  qualityScore: number,
  totalCost: number,
  latencyMs: number,
  attempts: number,
  opts?: { alpha?: number; simplicityMultiplier?: number },
): UnifiedMetric {
  const alpha = opts?.alpha ?? 1.0;
  const simplicityMult = opts?.simplicityMultiplier ?? 1.0;
  // efficiency = quality / (cost + latency_penalty), scaled by alpha and simplicity
  const latencyPenalty = latencyMs / 1000 * 0.0001; // ~$0.0001 per second of latency
  const rawEfficiency = (qualityScore * alpha * simplicityMult) / (totalCost + latencyPenalty + METRIC_EPSILON);
  return {
    efficiency: Math.round(rawEfficiency * 100) / 100,
    quality: Math.round(qualityScore * 10000) / 10000,
    cost: Math.round(totalCost * 10000) / 10000,
    latencyMs,
    attempts,
    alpha,
    simplicityMultiplier: Math.round(simplicityMult * 1000) / 1000,
  };
}

/**
 * computeSimplicity: estimates the simplicity of a mutation.
 * Fewer changed parameters + smaller magnitude → higher simplicity.
 * Returns [0, 1] where 1 = simplest (e.g., deleting code, reducing rounds).
 */
export function computeSimplicity(mutation: import("../types.js").StrategyMutation | null | undefined): number {
  if (!mutation) return 1.0; // baseline = simplest
  switch (mutation.type) {
    case "remove_round":
    case "remove_quality_criterion":
      return 0.95; // removing = simplest
    case "merge_rounds":
      return 0.80;
    case "reorder_rounds":
      return 0.85;
    case "mutate_retrieval":
    case "mutate_temperature":
      return 0.70;
    case "mutate_compression_weights":
      return 0.60;
    case "mutate_quality_weights":
      return 0.55;
    case "mutate_model_tiers":
      return 0.50;
    case "split_round":
    case "add_quality_criterion":
      return 0.40; // adding = more complex
    default:
      return 0.65;
  }
}
