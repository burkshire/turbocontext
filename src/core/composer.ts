// ============================================================
// Phase 2: Prompt Architecture Composition
// ============================================================
// 将任务分解为多轮对话序列，每个子任务选择最优提示模板组合
//
// 结构:
//   任务 T → 子任务序列 S = {s₁, s₂, ..., sₘ}
//   每个 sᵢ → role + task + context + format + quality
// ============================================================

import type { Task, PromptComponent, CompressedContext, CapabilityRequirement, StrategyMutation } from "../types.js";

/** 完整的提示架构 */
export interface PromptArchitecture {
  taskId: string;
  rounds: PromptRound[];
  estimatedTokens: number;
  componentUsage: Record<string, number>;
}

export interface PromptRound {
  sequence: number;
  goal: string;
  systemPrompt: string;
  userPrompt: string;
  expectedOutput: string;
  qualityCriteria: string[];
  dependsOn: number[]; // 依赖的前序轮次序号
}

/**
 * 主入口：构建提示架构
 *
 * @param canonicalMutations - 已保留的变异栈（autoresearch: branch tip = best config）
 * @param trialMutation - 正在试验的变异
 */
export function composePromptArchitecture(
  task: Task,
  context: CompressedContext,
  requirements: CapabilityRequirement[],
  trialMutation?: StrategyMutation,
  canonicalMutations?: StrategyMutation[],
): PromptArchitecture {
  // Step 1: 将任务分解为有序子任务（先应用 canonical 策略，再应用 trial 变异）
  const subTasks = decomposeTask(task, context, trialMutation, canonicalMutations);

  // Step 2: 为每个子任务构建提示轮次
  const rounds: PromptRound[] = subTasks.map((st, i) => {
    const systemPrompt = buildSystemPrompt(st, task);
    const userPrompt = buildUserPrompt(st, context, requirements);

    return {
      sequence: i + 1,
      goal: st.goal,
      systemPrompt,
      userPrompt,
      expectedOutput: st.outputFormat,
      qualityCriteria: st.qualityCriteria,
      dependsOn: st.dependsOn,
    };
  });

  const totalTokens = rounds.reduce(
    (sum, r) => sum + Math.ceil((r.systemPrompt.length + r.userPrompt.length) / 4),
    0
  );

  // 统计组件使用情况
  const componentUsage: Record<string, number> = {};
  for (const r of rounds) {
    const components = extractComponents(r);
    for (const c of components) {
      componentUsage[c] = (componentUsage[c] || 0) + 1;
    }
  }

  return { taskId: task.id, rounds, estimatedTokens: totalTokens, componentUsage };
}

/** 子任务定义 */
interface SubTask {
  goal: string;
  contextFocus: string[];
  outputFormat: string;
  qualityCriteria: string[];
  dependsOn: number[];
  requiresPreviousOutput: boolean;
}

/**
 * 智能任务分解
 *
 * 根据任务类型和压缩后的上下文，自动分解为最优的子任务序列。
 * 先应用 canonical 策略栈，再应用 trial 变异。
 */
function decomposeTask(
  task: Task,
  context: CompressedContext,
  trialMutation?: StrategyMutation,
  canonicalMutations?: StrategyMutation[],
): SubTask[] {
  const type = task.type;
  const hasCode = context.compressionRatio < 0.9; // 有实质代码内容

  const decompositionStrategies: Record<string, () => SubTask[]> = {
    code_review: () => [
      {
        goal: "理解变更上下文和目的",
        contextFocus: ["code_understanding"],
        outputFormat: "变更概述",
        qualityCriteria: ["准确理解变更范围", "识别变更的核心目的"],
        dependsOn: [],
        requiresPreviousOutput: false,
      },
      {
        goal: "逐模块检查代码质量",
        contextFocus: ["code_understanding", "pattern_recognition", "error_detection"],
        outputFormat: "问题列表（严重程度 + 描述 + 建议）",
        qualityCriteria: ["覆盖所有变更模块", "每个问题都有具体位置", "建议可执行"],
        dependsOn: [0],
        requiresPreviousOutput: true,
      },
      {
        goal: "生成审查总结",
        contextFocus: ["code_understanding"],
        outputFormat: "总结报告（通过/需修改/拒绝）",
        qualityCriteria: ["汇总准确", "优先级明确", "总体评价公平"],
        dependsOn: [0, 1],
        requiresPreviousOutput: true,
      },
    ],

    code_generation: () => [
      {
        goal: "分析需求和约束",
        contextFocus: ["code_understanding", "design"],
        outputFormat: "需求分析和实现计划",
        qualityCriteria: ["覆盖所有需求点", "识别技术约束", "方案合理"],
        dependsOn: [],
        requiresPreviousOutput: false,
      },
      {
        goal: "生成核心实现",
        contextFocus: ["code_generation"],
        outputFormat: "完整的代码实现",
        qualityCriteria: ["符合需求", "遵循项目风格", "类型安全", "包含必要注释"],
        dependsOn: [0],
        requiresPreviousOutput: true,
      },
      {
        goal: "检查生成代码的质量和安全性",
        contextFocus: ["error_detection", "pattern_recognition"],
        outputFormat: "质量报告 + 修正",
        qualityCriteria: ["无安全漏洞", "正确处理错误", "性能合理"],
        dependsOn: [1],
        requiresPreviousOutput: true,
      },
    ],

    debugging: () => [
      {
        goal: "理解 Bug 上下文和复现路径",
        contextFocus: ["code_understanding"],
        outputFormat: "Bug 分析和复现步骤",
        qualityCriteria: ["定位准确", "复现步骤清晰"],
        dependsOn: [],
        requiresPreviousOutput: false,
      },
      {
        goal: "生成修复方案",
        contextFocus: ["code_modification", "error_detection"],
        outputFormat: "修复代码 + 解释",
        qualityCriteria: ["修复正确", "无副作用", "覆盖边界情况"],
        dependsOn: [0],
        requiresPreviousOutput: true,
      },
      {
        goal: "验证修复",
        contextFocus: ["testing", "error_detection"],
        outputFormat: "测试用例 + 验证结果",
        qualityCriteria: ["覆盖修复路径", "包含回归测试"],
        dependsOn: [1],
        requiresPreviousOutput: true,
      },
    ],

    code_refactor: () => [
      {
        goal: "分析现有结构和重构目标",
        contextFocus: ["code_understanding", "design", "pattern_recognition"],
        outputFormat: "重构计划（目标结构 + 迁移步骤）",
        qualityCriteria: ["彻底理解现有结构", "目标架构合理", "迁移步骤细粒度"],
        dependsOn: [],
        requiresPreviousOutput: false,
      },
      {
        goal: "执行逐步重构",
        contextFocus: ["code_modification", "code_generation"],
        outputFormat: "重构后的代码",
        qualityCriteria: ["保持相同语义", "改善结构", "无破坏性变更"],
        dependsOn: [0],
        requiresPreviousOutput: true,
      },
      {
        goal: "验证重构结果",
        contextFocus: ["testing", "error_detection"],
        outputFormat: "验证报告",
        qualityCriteria: ["行为一致", "接口兼容"],
        dependsOn: [1],
        requiresPreviousOutput: true,
      },
    ],

    analysis: () => [
      {
        goal: "收集和整理相关信息",
        contextFocus: ["code_understanding"],
        outputFormat: "信息摘要和相关统计数据",
        qualityCriteria: ["全面", "准确", "无偏见"],
        dependsOn: [],
        requiresPreviousOutput: false,
      },
      {
        goal: "深度分析",
        contextFocus: ["pattern_recognition", "error_detection"],
        outputFormat: "分析结果 + 洞察",
        qualityCriteria: ["逻辑严谨", "有数据支撑", "洞察有深度"],
        dependsOn: [0],
        requiresPreviousOutput: true,
      },
      {
        goal: "生成结论和建议",
        contextFocus: ["design"],
        outputFormat: "结论报告 + 行动建议",
        qualityCriteria: ["结论有依据", "建议可执行", "优先级明确"],
        dependsOn: [0, 1],
        requiresPreviousOutput: true,
      },
    ],

    general: () => [
      {
        goal: "全面理解任务要求",
        contextFocus: ["code_understanding"],
        outputFormat: "任务理解确认",
        qualityCriteria: ["准确理解需求"],
        dependsOn: [],
        requiresPreviousOutput: false,
      },
      {
        goal: "执行核心任务",
        contextFocus: ["code_generation", "code_modification", "analysis"],
        outputFormat: "任务产出",
        qualityCriteria: ["符合需求", "质量达标"],
        dependsOn: [0],
        requiresPreviousOutput: true,
      },
      {
        goal: "质量复核",
        contextFocus: ["error_detection", "pattern_recognition"],
        outputFormat: "复核报告",
        qualityCriteria: ["覆盖要点", "无遗漏"],
        dependsOn: [1],
        requiresPreviousOutput: true,
      },
    ],
  };

  // 选择策略，如果不存在则回退到 general
  const strategy = decompositionStrategies[type] || decompositionStrategies.general;
  let base = strategy();

  // 应用 canonical 策略栈（已保留的变异，按顺序累积；autoresearch: branch tip = best config）
  if (canonicalMutations && canonicalMutations.length > 0) {
    for (const canonicalMutation of canonicalMutations) {
      base = applyMutation(base, canonicalMutation);
    }
  }

  // 应用正在试验的变异（v2.3 — autoresearch-inspired self-evolution）
  if (trialMutation) {
    base = applyMutation(base, trialMutation);
  }

  return base;
}

/**
 * 将变异操作应用到基础分解上（自进化 v2.3）
 */
function applyMutation(base: SubTask[], mutation: StrategyMutation): SubTask[] {
  switch (mutation.type) {
    case "merge_rounds": {
      const [i, j] = mutation.roundIndices;
      if (i >= base.length || j >= base.length || i === j) return base;
      const [a, b] = [Math.min(i, j), Math.max(i, j)];
      const merged: SubTask = {
        goal: mutation.newGoal || `${base[a].goal} 且 ${base[b].goal}`,
        contextFocus: [...new Set([...base[a].contextFocus, ...base[b].contextFocus])],
        outputFormat: `${base[a].outputFormat}; ${base[b].outputFormat}`,
        qualityCriteria: [...base[a].qualityCriteria, ...base[b].qualityCriteria],
        dependsOn: base[a].dependsOn.filter(d => d !== a && d !== b),
        requiresPreviousOutput: base[a].requiresPreviousOutput || base[b].requiresPreviousOutput,
      };
      const result = [...base];
      result.splice(a, b - a + 1, merged);
      // oldIdx<a→oldIdx; a≤oldIdx≤b→a(merged); oldIdx>b→oldIdx-(b-a)
      const mapIdx = (oldIdx: number) =>
        oldIdx < a ? oldIdx : oldIdx <= b ? a : oldIdx - (b - a);
      return fixDependsOn(result, mapIdx);
    }
    case "remove_round": {
      const idx = mutation.roundIndex;
      if (idx < 0 || idx >= base.length) return base;
      const result = [...base];
      result.splice(idx, 1);
      const mapIdx = (oldIdx: number) =>
        oldIdx < idx ? oldIdx : oldIdx > idx ? oldIdx - 1 : undefined;
      return fixDependsOn(result, mapIdx);
    }
    case "reorder_rounds": {
      if (mutation.newOrder.length !== base.length) return base;
      const result = mutation.newOrder.map(i => base[i]);
      const mapIdx = (oldIdx: number) => {
        const newIdx = mutation.newOrder.indexOf(oldIdx);
        return newIdx >= 0 ? newIdx : undefined;
      };
      return fixDependsOn(result, mapIdx);
    }
    case "split_round": {
      const idx = mutation.roundIndex;
      if (idx < 0 || idx >= base.length) return base;
      const original = base[idx];
      const half = Math.ceil(original.qualityCriteria.length / 2);
      const a: SubTask = {
        goal: mutation.newGoalA || original.goal,
        contextFocus: [...original.contextFocus],
        outputFormat: original.outputFormat,
        qualityCriteria: original.qualityCriteria.slice(0, half),
        dependsOn: original.dependsOn,
        requiresPreviousOutput: original.requiresPreviousOutput,
      };
      const b: SubTask = {
        goal: mutation.newGoalB || `${original.goal}（验证）`,
        contextFocus: [...original.contextFocus],
        outputFormat: original.outputFormat,
        qualityCriteria: original.qualityCriteria.slice(half),
        dependsOn: [idx], // 依赖于拆分后的前半部分
        requiresPreviousOutput: true,
      };
      const result = [...base];
      result.splice(idx, 1, a, b);
      // oldIdx<idx→oldIdx; oldIdx==idx→idx(a); oldIdx>idx→oldIdx+1
      const mapIdx = (oldIdx: number) =>
        oldIdx < idx ? oldIdx : oldIdx === idx ? idx : oldIdx + 1;
      return fixDependsOn(result, mapIdx);
    }
    case "add_quality_criterion": {
      const idx = mutation.roundIndex;
      if (idx < 0 || idx >= base.length) return base;
      const result = [...base];
      result[idx] = {
        ...result[idx],
        qualityCriteria: [...result[idx].qualityCriteria, mutation.criterion],
      };
      return result;
    }
    case "remove_quality_criterion": {
      const idx = mutation.roundIndex;
      if (idx < 0 || idx >= base.length) return base;
      const ci = mutation.criterionIndex;
      if (ci < 0 || ci >= base[idx].qualityCriteria.length) return base;
      const result = [...base];
      result[idx] = {
        ...result[idx],
        qualityCriteria: result[idx].qualityCriteria.filter((_, i) => i !== ci),
      };
      return result;
    }
    default:
      return base;
  }
}

/**
 * 修正变异后的 dependsOn 索引。
 *
 * 仅过滤掉无效旧索引不够——删除/合并轮次后索引会移位。
 * 接受一个映射函数将旧依赖索引转换为新索引。
 */
function fixDependsOn(
  subTasks: SubTask[],
  mapIndex?: (oldIdx: number) => number | undefined,
): SubTask[] {
  return subTasks.map((st, newIdx) => {
    const remapped = st.dependsOn
      .map(oldIdx => (mapIndex ? mapIndex(oldIdx) : oldIdx))
      .filter((d): d is number =>
        d !== undefined && d >= 0 && d < subTasks.length && d < newIdx,
      );
    // 去重（当两个旧索引映射到同一新索引时可能出现）
    const unique = [...new Set(remapped)];
    return { ...st, dependsOn: unique };
  });
}

/**
 * 构建系统提示
 */
function buildSystemPrompt(subTask: SubTask, task: Task): string {
  const roleDefinitions: Record<string, string> = {
    code_review: "你是一位资深的代码审查专家。你严格、细致、公正。",
    code_generation: "你是一位高效的软件工程师。你写出的代码质量高、符合最佳实践。",
    debugging: "你是一位调试专家。你擅长通过分析找到 bug 的根本原因。",
    analysis: "你是一位系统分析师。你深入、全面、客观。",
    design: "你是一位软件架构师。你设计的系统可扩展、可维护。",
    testing: "你是一位测试工程师。你编写的测试覆盖全面且有价值。",
    code_refactor: "你是一位重构专家。你改善代码结构而不改变其行为。",
    documentation: "你是一位技术写作专家。你写的文档清晰、准确、有用。",
    general: "你是一位全能的 AI 助手，能够高质量地完成各种任务。",
  };

  const role = roleDefinitions[task.type] || roleDefinitions.general;

  return `${role}

## 当前目标
${subTask.goal}

## 输出要求
${subTask.outputFormat}

## 质量标准
${subTask.qualityCriteria.map((q, i) => `${i + 1}. ${q}`).join("\n")}

请严格按照上述格式输出。`;
}

/**
 * 构建用户提示
 */
function buildUserPrompt(
  subTask: SubTask,
  context: CompressedContext,
  requirements: CapabilityRequirement[]
): string {
  let prompt = "";

  // 添加相关上下文的摘要
  if (context.fragments.length > 0) {
    prompt += "## 相关上下文\n\n";
    for (const f of context.fragments) {
      prompt += `### ${f.original.source}\n`;
      prompt += f.preservedSections.join("\n") + "\n\n";
    }
  }

  prompt += `## 任务\n请根据以上上下文，${subTask.goal}。\n`;
  prompt += `输出格式：${subTask.outputFormat}\n`;

  if (subTask.requiresPreviousOutput) {
    prompt += `\n注意：请结合前一步的输出进行本次工作。`;
  }

  return prompt;
}

/** 从提示轮次中提取使用的组件名 */
function extractComponents(round: PromptRound): string[] {
  const components: string[] = [];
  if (round.systemPrompt.includes("资深")) components.push("role-expert");
  if (round.systemPrompt.includes("质量标准")) components.push("quality-criteria");
  if (round.userPrompt.includes("相关上下文")) components.push("context-block");
  if (round.userPrompt.includes("输出格式")) components.push("format-spec");
  return components;
}
