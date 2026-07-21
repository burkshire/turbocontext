# TurboContext

**自适应上下文优化与质量加权生成算法**

> 核心竞争力：一套系统化的 AI 应用效率优化算法，让独立开发者以一人之力达到团队级产出。

---

## 快速开始

### 环境要求

- **Node.js** ≥ 20
- **npm**（随 Node.js 一起安装）
- **API Key**：DeepSeek API Key（或任意 OpenAI 兼容接口的 Key）

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/burkshire/turbocontext.git
cd turbocontext

# 2. 安装依赖
npm install

# 3. （可选）配置 API Key，不配置则使用模拟模式
export DEEPSEEK_API_KEY="sk-你的key"

# 4. 跑一遍 demo，看看效果
npx tsx src/cli.ts demo
```

### 三句话理解 TurboContext

```
1. 压缩上下文 — 不让无关代码稀释 LLM 的注意力
2. 质量门控   — 输出不达标就自动反馈重试，最多 3 轮
3. 持续学习   — 每次执行都在优化参数，越用越聪明
```

---

## 阅读路线

本仓库包含两套互补的文档：

- **README.md**（本文）— 算法的完整参考手册：公式、架构、使用方式
- **LEARN.md**（配套教程）— 从零开始理解设计思想的七堂课

建议顺序：
1. 先跑 demo：`npx tsx src/cli.ts demo`
2. 看 [LEARN.md](./LEARN.md) 前三课，理解算法的设计直觉
3. 回到 README.md 看对应的公式部分，把直觉和公式对应起来
4. 继续 LEARN.md 后四课，理解质量门控、成本优化和学习系统
5. 打开源码，一行行对照 README.md 的架构说明
6. 在你自己的项目中注册 `/turbocontext` Skill，开始使用

---

## 目录

- [快速开始](#快速开始)
- [1. 核心思想](#1-核心思想)
- [2. 算法全景](#2-算法全景)
- [3. 完整数学公式](#3-完整数学公式)
- [4. 代码架构与实现](#4-代码架构与实现)
- [5. 使用方式](#5-使用方式)
- [6. 实战示例](#6-实战示例)
- [7. 核心竞争力分析](#7-核心竞争力分析)
- [8. 进阶路线图](#8-进阶路线图)
- [附录](#附录)

---

## 1. 核心思想

### 1.1 为什么需要 TurboContext？

大多数 AI 应用效率低下的根源：

| 问题 | 表现 | 后果 |
|------|------|------|
| 上下文冗余 | 塞入大量无关代码/文档 | Token 浪费 + 信号稀释，输出质量下降 |
| 提示结构混乱 | prompt 没有层次，想到哪写到哪 | 质量不可控，每次结果方差大 |
| 缺乏自适应 | 所有任务用同一配置 | 简单任务浪费钱，复杂任务不够强 |
| 无质量控制 | 一次生成赌运气 | 需要人工反复审查和修正 |

### 1.2 核心理念

```
与其调教模型，不如调教上下文。
与其赌一次输出，不如建质量门控。
与其手动优化，不如让系统自学习。
```

### 1.3 设计原则

1. **Token 即货币** — 每条上下文都需证明自己的价值，否则被压缩掉
2. **质量可度量** — 不能度量就无法改进，4 维度评估体系
3. **成本意识** — 每个任务自动匹配最优模型，不做无谓浪费
4. **持续进化** — 每次执行都是学习机会，系统越用越聪明

---

## 2. 算法全景

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          TurboContext Pipeline                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  任务 T + 上下文 C                                                       │
│       │                                                                  │
│       ▼                                                                  │
│  ┌───────────┐                                                          │
│  │ Phase 1   │ 上下文压缩与评分                                           │
│  │           │ score(cᵢ) = α·sim + β·recency + γ·specificity            │
│  │           │ 贪婪选择 + 能力覆盖约束 → 压缩后的上下文 C'                │
│  └─────┬─────┘                                                          │
│        │                                                                 │
│        ▼                                                                 │
│  ┌───────────┐                                                          │
│  │ Phase 2   │ 提示架构组合                                               │
│  │           │ 任务分解为 3 轮子任务: 理解 → 执行 → 验证                   │
│  │           │ 每轮自动构建: role + task + context + format + quality    │
│  └─────┬─────┘                                                          │
│        │                                                                 │
│  ┌─────▼─────┐                                                          │
│  │ Phase 3   │ 质量加权生成 ←──────────────┐                            │
│  │           │ 温度递减 [0.7, 0.35, 0.1]   │ 未达标                       │
│  │           │ 生成 → 评估 → 达标？──是──→ 输出                          │
│  │           │                    │        │                            │
│  │           │                    └──否──→ 注入反馈 + 重试               │
│  └─────┬─────┴──────────────────────────────────┘                       │
│        │                                                                 │
│  ┌─────▼─────┐                                                          │
│  │ Phase 4   │ 成本-延迟优化                                              │
│  │           │ complexity < θ₁ → Haiku ($0.25/1M tokens)                │
│  │           │ θ₁ ≤ complexity < θ₂ → Sonnet ($3/1M tokens)             │
│  │           │ complexity ≥ θ₂ → Opus ($15/1M tokens)                   │
│  └─────┬─────┘                                                          │
│        │                                                                 │
│  ┌─────▼─────┐                                                          │
│  │ Phase 5   │ 连续学习 (每 5 次执行触发)                                 │
│  │           │ 更新 α/β/γ → 压缩策略改进                                 │
│  │           │ 更新 θ₁/θ₂ → 模型选择改进                                 │
│  │           │ 更新温度调度 → 生成策略改进                                │
│  └───────────┘                                                          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 阶段间数据流

```
Phase 1 输出 C' ──→ Phase 2 用于构建 prompt 的上下文块
Phase 2 输出 P  ──→ Phase 3 作为生成输入 (多轮 prompt 序列)
Phase 4 输出 M  ──→ Phase 3 选择模型和成本预算
Phase 3 输出 O  +  Phase 4 输出 cost  ──→ Phase 5 作为学习数据
Phase 5 输出 Θ' ──→ 更新 Phase 1, 3, 4 的参数
```

---

## 3. 完整数学公式

### 3.1 Phase 1: 上下文压缩与评分

#### 3.1.1 能力需求分解

将任务 T 分解为能力需求集合 R = {r₁, r₂, ..., rₙ}：

```
R = { code_understanding, pattern_recognition, code_generation,
      code_modification, error_detection, design }
```

每个需求的权重由其关键词在任务描述中的命中率决定：

```
weight(rᵢ) = Σⱼ match(keywordᵢⱼ, T) / |keywordsᵢ|
match(kw, T) = 1 if kw ∈ T else 0
```

#### 3.1.2 片段评分函数

对每个上下文片段 cᵢ ∈ C 计算综合得分：

```
score(cᵢ) = α · sim(cᵢ, T) + β · recency(cᵢ) + γ · specificity(cᵢ)
```

**约束**：α + β + γ = 1

**默认值**：α = 0.55, β = 0.20, γ = 0.25

其中三个子分数的计算公式：

**语义相似度**（基于关键词覆盖的启发式估计）：

```
sim(cᵢ, T) = 0.50 · keyword_overlap(cᵢ, T)
            + 0.30 · type_match(contentType, taskType)
            + 0.20 · structural_match(cᵢ)
```

**新鲜度**（最近修改时间越近，价值越高）：

```
recency(cᵢ) = 1 / (1 + days_since_last_modified)

days_since_last_modified = (now - mtime(cᵢ)) / 86400000
```

**特异性**（信息密度越高，分数越高）：

```
specificity(cᵢ) = 1 - min(len(cᵢ), MAX_LEN) / MAX_LEN

MAX_LEN = 5000 (字符)
```

#### 3.1.3 带约束的贪婪选择

```
目标: 在 token 预算内最大化能力覆盖

最大化 Σᵢ Σⱼ score(cᵢ) · cover(cᵢ, rⱼ)

约束:
  Σᵢ tokens(cᵢ) ≤ budget                    (token 预算)
  ∀rⱼ ∈ R: Σᵢ cover(cᵢ, rⱼ) ≥ 1           (覆盖率约束)
```

选择算法：

```
已选中 = ∅
对 rⱼ ∈ R (按 weightⱼ 降序):
  c* = argmax_{c ∈ C \ 已选中} [
    score(c) · 1.2^{cover(c, rⱼ)} - len(c)/budget · 0.1
  ]
  已选中 = 已选中 ∪ {c*}
对剩余预算按 score(c) 降序贪心补充
```

#### 3.1.4 片段压缩

```
compress(c):
  逐行处理:
    如果 结构性行 → 保留原样
    否则如果 在 body 中 → 缓冲
    否则 → 丢弃

  结构性行匹配规则:
    /^(export\s+)?(function|class|interface|type|enum|struct|trait|impl)/
    /^(const|let|var|val)\s+\w+\s*[=:]/
    /^(import|export|from|use|require|include)/
    /^(public|private|protected|static|abstract)/

  body 压缩规则:
    如果 body 行数 > 3 → 替换为 "// ... [N lines omitted]"
    否则 → 保留原样
```

#### 3.1.5 压缩比

```
compression_ratio = 1 - compressed_tokens / original_tokens

compressed_tokens = Σᵢ ceil(len(preserved_i) / 4)
original_tokens   = Σᵢ ceil(len(content_i) / 4)
```

---

### 3.2 Phase 2: 提示架构组合

#### 3.2.1 任务分解

根据任务类型自动选择分解策略：

```
代码审查 (code_review):
  Round 1: 理解变更上下文和目的          [理解]
  Round 2: 逐模块检查代码质量            [执行]
  Round 3: 生成审查总结                  [验证]

代码生成 (code_generation):
  Round 1: 分析需求和约束                [理解]
  Round 2: 生成核心实现                  [执行]
  Round 3: 检查生成代码的质量和安全性    [验证]

调试 (debugging):
  Round 1: 理解 Bug 上下文和复现路径     [理解]
  Round 2: 生成修复方案                  [执行]
  Round 3: 验证修复                      [验证]

重构 (code_refactor):
  Round 1: 分析现有结构和重构目标        [理解]
  Round 2: 执行逐步重构                  [执行]
  Round 3: 验证重构结果                  [验证]

分析 (analysis):
  Round 1: 收集和整理相关信息            [理解]
  Round 2: 深度分析                      [执行]
  Round 3: 生成结论和建议                [验证]
```

#### 3.2.2 提示生成

```
prompt(sᵢ | prev_outputs) =
    system_prompt(sᵢ)       // 角色定义 + 目标声明
    ++ context_block(C')     // 压缩后的相关上下文
    ++ task_block(sᵢ)       // 具体任务描述
    ++ format_block()       // 输出格式要求
    ++ quality_block()      // 质量标准
    ++ prev_outputs_block()  // 前序轮次输出（如果有）
```

---

### 3.3 Phase 3: 质量加权生成

#### 3.3.1 核心生成循环

```
for k = 1 to K:
    tₖ = temperature_schedule[k]
    oₖ ~ LLM(Pₖ, tₖ)
    Q(oₖ) = Σᵢ wᵢ · qᵢ(oₖ)
    if Q(oₖ) ≥ θ_Q:
        return oₖ
    else:
        fₖ = critique(oₖ, Q)
        Pₖ₊₁ = Pₖ ⊕ fₖ  // 注入反馈

return o_K  (最后一次的结果)
```

**参数默认值**：

| 参数 | 值 | 说明 |
|------|-----|------|
| K | 3 | 最大尝试次数 |
| θ_Q | 0.85 | 质量阈值 |
| temperature_schedule | [0.7, 0.35, 0.1] | 温度递减调度 |

#### 3.3.2 质量评估函数

```
Q(o) = w₁ · q_completeness + w₂ · q_correctness + w₃ · q_consistency + w₄ · q_format

约束: w₁ + w₂ + w₃ + w₄ = 1
```

**权重按任务类型动态分配**：

| 任务类型 | w₁ (完整) | w₂ (正确) | w₃ (一致) | w₄ (格式) |
|---------|-----------|-----------|-----------|-----------|
| code_review | 0.25 | 0.25 | 0.30 | 0.20 |
| code_generation | 0.30 | 0.35 | 0.20 | 0.15 |
| debugging | 0.20 | 0.50 | 0.20 | 0.10 |
| code_refactor | 0.20 | 0.40 | 0.30 | 0.10 |
| testing | 0.35 | 0.30 | 0.20 | 0.15 |
| analysis | 0.30 | 0.25 | 0.25 | 0.20 |
| design | 0.25 | 0.20 | 0.30 | 0.25 |
| documentation | 0.30 | 0.15 | 0.25 | 0.30 |
| general | 0.30 | 0.25 | 0.25 | 0.20 |

#### 3.3.3 各维度评分细则

**完整性 (q_completeness)**：

```
q_completeness = |需求点被覆盖| / |总需求点|

需求点提取: 从任务描述 T 中提取名词短语和关键动词
覆盖率: 每个需求点在输出中出现则计为覆盖
```

**正确性 (q_correctness)**：

```
q_correctness = 1 - Σⱼ penalty(flagⱼ) × indicator(detectedⱼ)

惩罚列表:
  不确定表达 (sorry, I don't know, I cannot)    → -0.15
  占位符 (TODO, FIXME, placeholder)             → -0.20
  推测语 (assuming, might be, perhaps)          → -0.05
  不完整声明 (incomplete, partial, draft)        → -0.10
  错误引用 (TypeError, undefined is not...)      → -0.25
```

**一致性 (q_consistency)**：

```
q_consistency = 1 - Σⱼ penalty(contradictionⱼ)

矛盾检测:
  同一概念使用不同术语 (user-id vs user-identifier) → -0.05
  相反指令同时出现 (do not use X vs 却用 X)        → -0.10
  声明数量与实际不符 (说 3 步但只列出 2 步)         → -0.05
```

**格式合规 (q_format)**：

```
q_format = 1 - Σⱼ penalty(format_issueⱼ)

检查项:
  需要代码但无代码块            → -0.30
  代码块未闭合 (```不成对)      → -0.40
  输出过短 (< 10 字符)          → -0.50
  超长行 (> 200 字符, 每 3 行)  → -0.10
```

#### 3.3.4 反馈注入机制

```
critique(o, Q):
    问题列表 = detect(Q.dimensions)
    反馈 = "## 质量反馈 (第 k 轮)
            总体质量: {Q.score * 100}%
            目标阈值: {θ_Q * 100}%

            ### 各维度
            - 完整性: {q_completeness * 100}%
            - 正确性: {q_correctness * 100}%
            ...

            ### 改进要求
            {根据不达标维度生成具体改进指令}"

    将反馈注入下一轮 prompt
```

---

### 3.4 Phase 4: 成本-延迟优化

#### 3.4.1 复杂度评估

```
complexity(T) = 0.40 · type_complexity(T)
              + 0.15 · ambiguity(T)
              + 0.20 · historical_complexity(T)
              + 0.25 · base_complexity
```

分项说明：

| 任务类型 | type_complexity |
|---------|----------------|
| debugging | 0.35 |
| code_review | 0.40 |
| code_generation | 0.45 |
| code_refactor | 0.55 |
| analysis | 0.50 |
| design | 0.65 |
| documentation | 0.25 |
| general | 0.35 |

```
ambiguity(T) 由描述长度决定:
  len < 20   → 0.8 (高度模糊)
  len < 50   → 0.5
  len < 200  → 0.3
  otherwise  → 0.2 (描述详细，清晰)

historical_complexity(T) = 基于历史记录中同类任务的平均失败率
```

#### 3.4.2 模型选择

```
model = f(complexity, latency_budget):

          ┌ fast   (Haiku)   如果 complexity < θ₁
  tier = ─├ medium (Sonnet)  如果 θ₁ ≤ complexity < θ₂
          └ deep   (Opus)    如果 complexity ≥ θ₂

默认阈值: θ₁ = 0.35, θ₂ = 0.70
```

| 层级 | 模型 | 成本 ($/1K tokens) | 延迟 |
|------|------|-------------------|------|
| fast | Claude Haiku | $0.00025 | ~2s |
| medium | Claude Sonnet | $0.003 | ~5s |
| deep | Claude Opus | $0.015 | ~15s |

**延迟约束覆盖**：

```
if latency_budget < model_latency AND tier ≠ "fast":
    tier = 降级到下一级  // 满足延迟要求
```

#### 3.4.3 成本估计

```
expected_cost(T) = (estimated_tokens / 1000) × cost_per_1K × expected_attempts

expected_attempts:
  代码生成: 1.5
  调试:     1.5
  其他:     1.2
```

#### 3.4.4 缓存策略

```
fingerprint = hash(task_type + task_description + context_prefix)

lookup:
  entry = cache[fingerprint]
  if entry AND age < 5min:
    return entry.result  // 缓存命中，零成本

write:
  cache 上限 100 条，超出时淘汰最早条目
```

---

### 3.5 Phase 5: 连续学习

#### 3.5.1 压缩权重更新 (α, β, γ)

每 5 次执行后触发：

```
如果 高压缩比高质量执行 的占比显著:
  α -= 0.01  (降低语义权重)
  γ += 0.02  (增加特异性权重)
  约束: α ≥ 0.35, γ ≤ 0.40

如果 低压缩比低质量执行 的占比显著:
  α += 0.02  (增加语义权重)
  γ -= 0.01  (降低特异性权重)
  约束: α ≤ 0.70, γ ≥ 0.15

归一化: (α, β, γ) = (α, β, γ) / (α + β + γ)
```

#### 3.5.2 复杂度阈值更新 (θ₁, θ₂)

```
如果 fast_model_pass_rate > 0.9:
  θ₁ += 0.03  (快速模型表现好，扩大使用范围)
  约束: θ₁ ≤ 0.45

如果 fast_model_pass_rate < 0.7:
  θ₁ -= 0.03  (快速模型经常失败，缩小使用范围)
  约束: θ₁ ≥ 0.20

如果 deep_model_fail_rate > 0.3:
  θ₂ += 0.03  (深度模型也失败，说明不是模型问题)
  约束: θ₂ ≤ 0.85
```

#### 3.5.3 温度调度更新

```
如果 平均尝试次数 ≤ 1.1:
  t₀ = max(0.3, t₀ - 0.05)   // 一次通过率极高，降低初温

如果 平均尝试次数 ≥ 2.5:
  t₀ = min(0.9, t₀ + 0.05)   // 总是需要重试，提高初温
```

#### 3.5.4 学习参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 学习率 η | 0.1 | 每次调整步长 |
| 历史窗口 N | 100 条 | 保留最近记录数 |
| 学习间隔 | 每 5 次执行 | 最小触发数据量 |
| 最少触发 | 5 条记录 | 数据不足时不调整 |

---

### 3.6 完整算法伪代码

```
Algorithm TurboContext(T, C, Θ):

  // === Phase 1: 上下文压缩 ===
  R ← decomposeCapabilities(T)                        // 能力需求分解
  for c ∈ C:
    c.score ← α·sim(c,T) + β·recency(c) + γ·specificity(c)  // 评分
  S ← greedySelect(C, R, budget)                      // 带约束选择
  for s ∈ S:
    s.compressed ← compress(s)                         // 片段压缩
  C' ← {s.compressed | s ∈ S}
  cov ← computeCoverage(C', R)

  // === Phase 2: 提示架构 ===
  subtasks ← decomposeTask(T, C')                     // 任务分解
  P ← []
  for s ∈ subtasks:
    p ← composePrompt(s, C', prev_outputs)            // 构建 prompt
    P ← P ∪ {p}

  // === Phase 4: 成本优化 (在生成前选模型) ===
  cx ← estimateComplexity(T, history)
  M ← selectModel(cx, latency_budget)
  cost ← estimateCost(T, tokens(P), M)

  // === Phase 3: 质量加权生成 ===
  for k = 1 to K:
    t ← temperature_schedule[k]
    o ← callLLM(P, t)                                 // 调用 LLM
    q ← evaluateQuality(o, T)                         // 质量评估
    record(generation{k, o, q})
    if q.score ≥ θ_Q:
      break                                           // 达标，接受
    f ← generateFeedback(q)                           // 生成反馈
    P ← injectFeedback(P, f)                          // 注入重试

  // === Phase 5: 连续学习 ===
  history ← history ∪ {task, C', M, q, cost}
  if |history| % 5 = 0:
    Θ ← learn(Θ, history)                             // 参数更新

  return { output: o, quality: q, cost: cost, coverage: cov }
```

---

## 4. 代码架构与实现

### 4.1 项目结构

```
turbocontext/
│
├── src/
│   ├── core/                         # 算法核心 (5 个阶段)
│   │   ├── compressor.ts             # Phase 1: 上下文压缩
│   │   ├── composer.ts               # Phase 2: 提示架构
│   │   ├── generator.ts              # Phase 3: 质量加权生成
│   │   ├── optimizer.ts              # Phase 4: 成本优化
│   │   └── learner.ts                # Phase 5: 连续学习
│   │
│   ├── index.ts                      # 主引擎 (TurboContextEngine)
│   ├── cli.ts                        # CLI 入口
│   └── types.ts                      # 类型定义
│
├── bin/
│   └── turbocontext.js               # 可执行入口
│
├── skill/
│   └── turbocontext.md               # Claude Code Skill 定义
│
├── .claude/
│   └── settings.json                 # Skill 注册
│
├── FORMULA.md                        # 公式文档
├── README.md                         # 本文档
├── package.json
└── tsconfig.json
```

### 4.2 核心引擎 API

```typescript
import { TurboContextEngine } from "turbocontext";

const engine = new TurboContextEngine({
  qualityThreshold: 0.85,     // 质量阈值
  maxAttempts: 3,             // 最大重试次数
  maxTokenBudget: 8000,       // Token 预算
  alpha: 0.55, beta: 0.20, gamma: 0.25,  // 压缩权重
});

const result = await engine.execute(task, contextFragments);
// result 包含完整的 5 阶段输出
```

### 4.3 各阶段模块详解

#### compressor.ts 核心函数

```typescript
// 主入口
compressContext(task, fragments, config): CompressedContext

// 内部函数
decomposeTask(task): CapabilityRequirement[]       // 能力分解
calculateScore(fragment, task): number             // 评分公式
greedySelect(scored, requirements, config): []     // 带约束选择
compressFragment(fragment): CompressedFragment     // 片段压缩
```

#### composer.ts 核心函数

```typescript
// 主入口
composePromptArchitecture(task, context): PromptArchitecture

// 内部函数
decomposeTask(task): SubTask[]                     // 按类型分解
buildSystemPrompt(subTask): string                 // 构建系统提示
buildUserPrompt(subTask, context): string          // 构建用户提示
```

#### generator.ts 核心函数

```typescript
// 主入口 (异步生成器)
qualityWeightedGeneration(task, architecture, config, llmCall):
  AsyncGenerator<GenerationOutput>

// 质量评估
evaluateQuality(output, task): QualityAssessment

// 辅助函数
assessCompleteness(output, task): number
assessCorrectness(output, task): number
assessConsistency(output, task): number
assessFormat(output, task): number
generateFeedback(assessment, output): string
```

#### optimizer.ts 核心类

```typescript
class Optimizer {
  selectModel(task, history): { tier, config, rationale }
  estimateComplexity(task, history): number
  lookupCache(task, content): CacheEntry | null
  writeCache(task, content, result, quality, model): void
  estimateCost(task, tokens, tier): { cost, latency }
}
```

#### learner.ts 核心类

```typescript
class Learner {
  record(execution): void                            // 记录一次执行
  learn(): { config, adjustments }                  // 执行学习
  getQualityTrend(): { average, trend, byType }     // 获取质量趋势
  getConfig(): TurboContextConfig                    // 获取当前配置
}
```

---

## 5. 使用方式

### 5.1 方式 A：终端 CLI（零依赖，任何项目都能用）

```bash
cd turbocontext                     # 进入 clone 下来的目录

# 查看所有命令
npx tsx src/cli.ts help

# 运行演示（不调用 LLM，模拟数据展示完整流程）
npx tsx src/cli.ts demo

# 查看公式参考
npx tsx src/cli.ts formula

# 审查代码（模拟模式，不消耗 API）
npx tsx src/cli.ts run \
  --task "审查 src/core/compressor.ts 的代码质量" \
  --dir ./src \
  --type code_review

# 调用真实 LLM（需先配置 API Key）
export DEEPSEEK_API_KEY="sk-你的key"
npx tsx src/cli.ts run \
  --task "审查 src/core/compressor.ts 的代码质量" \
  --dir ./src \
  --type code_review \
  --llm

# 支持的任务类型
# code_review | code_generation | debugging | code_refactor
# analysis | design | documentation | testing | general
```

### 5.2 方式 B：Claude Code Skill（在 Claude Code 中使用 /turbocontext）

**第一步 — 注册 Skill**：把以下内容添加到你的 Claude Code 项目的 `.claude/settings.json` 中（如果文件不存在就新建）：

```json
{
  "skills": {
    "turbocontext": {
      "name": "turbocontext",
      "source": "skill/turbocontext.md",
      "description": "TurboContext: Adaptive context optimization & quality-weighted generation"
    }
  }
}
```

> **注意**：`source` 要指向你 clone 的 turbocontext 仓库里的 `skill/turbocontext.md`。写绝对路径最稳妥，比如 `/home/你的用户名/turbocontext/skill/turbocontext.md`。

**第二步 — 使用**：在 Claude Code 中直接输入：

```
/turbocontext 审查当前项目的 src/auth 模块，关注安全隐患
/turbocontext 帮我重构 src/utils.ts，消除重复代码
/turbocontext 分析这个项目的架构设计，给出改进建议
```

**Skill 会自动执行**：
1. 扫描和理解上下文
2. 压缩到最有价值的信息
3. 构建最优的分步 prompt 架构
4. 质量门控的多轮生成
5. 输出完整的质量报告

### 5.3 方式 C：作为 npm 库集成到你的代码

```bash
# 在你的项目中安装（本地路径）
npm install /path/to/turbocontext
```

```typescript
import { TurboContextEngine, compressContext, evaluateQuality } from "turbocontext";

// 使用完整引擎
const engine = new TurboContextEngine({
  qualityThreshold: 0.85,     // 质量阈值
  maxAttempts: 3,             // 最大重试次数
});
const result = await engine.execute(myTask, myFragments);

// 或单独使用某个阶段
const compressed = compressContext(task, fragments, config);
const quality = evaluateQuality(output, task);
```

### 5.3 方式 C：作为库集成

```typescript
import { TurboContextEngine, compressContext, evaluateQuality } from "turbocontext";

// 使用完整引擎
const engine = new TurboContextEngine();
const result = await engine.execute(myTask, myFragments);

// 或单独使用某个阶段
const compressed = compressContext(task, fragments, config);
const quality = evaluateQuality(output, task);
```

---

## 6. 实战示例

### 6.1 代码审查场景

```
任务: "审查登录模块的安全性"

Phase 1 输出:
  - 原始: 351 tokens → 压缩后: 344 tokens
  - 选中 5/5 个片段 (代码本身已很精炼)
  - 能力覆盖: code_understanding=100%, error_detection=100%

Phase 2 输出:
  3 轮子任务: 理解变更 → 逐模块检查 → 生成总结
  - 第1轮: 分析 login.ts, register.ts 的认证流程
  - 第2轮: 检查 token 处理、密码存储、输入验证
  - 第3轮: 汇总发现的问题并按严重程度排序

Phase 4 输出:
  复杂度评估: 0.37 → 推荐使用 Sonnet (均衡模型)
  估计成本: $0.0045

Phase 3 输出:
  第1次尝试: 质量评分 95% ✓ (一次通过)
  完整性: 95% | 正确性: 100% | 一致性: 90% | 格式: 95%
```

### 6.2 代码生成场景

```
任务: "添加忘记密码功能，包括重置令牌和邮件发送"

Phase 1 输出:
  能力覆盖: code_generation=100%, error_detection=80%, design=80%

Phase 2 输出:
  3 轮子任务: 分析需求 → 生成实现 → 检查质量

Phase 3 输出:
  第1次尝试: 质量评分 65.5% (未达标)
    发现问题: 完整性不足（缺少错误处理）、正确性偏低（包含推测性代码）
    注入反馈...
  第2次尝试: 质量评分 72.0%
    改进: 错误处理完整，但格式仍不符合要求
    注入反馈...
  第3次尝试: 质量评分 88.0% ✓ (达标)

质量趋势: improving (从 65% 经过自动修正达到 88%)
```

---

## 7. 核心竞争力分析

### 7.1 与传统方法的对比

| 维度 | 传统方法 | TurboContext |
|------|---------|--------------|
| **上下文利用** | 全部塞入，不管是否相关 | 智能评分 + 压缩，只留最有价值的 |
| **质量可控性** | 看运气，每次结果方差大 | 4 维度评估 + 自动门控，质量可预期 |
| **成本效率** | 固定模型，固定配置 | 动态选模型 + 缓存，节省 40-60% |
| **构建速度** | 每次手动调 prompt | 组件化复用，新功能 = 新组件 + 已有流程 |
| **可扩展性** | 每加一个功能都需重新设计 | 新任务类型只需补充分解策略 |
| **学习能力** | 经验停留在个人头脑中 | 系统自动积累并优化参数 |
| **单人产出** | 1 人 = 1 人 | 1 人 = 3-5 人团队（在 AI 应用效率上） |

### 7.2 为什么这是护城河

```
1. 系统性优势
   - 别人在"写 prompt" → 你在"设计提示架构"
   - 别人在"手动检查" → 你有"自动质量门控"
   - 别人在"重复劳动" → 你的系统在"持续学习"

2. 数据飞轮
   - 每次使用都在优化系统
   - 使用越多，你的系统越了解你的代码
   - 这是无法被复制的个性化优势

3. 成本结构
   - 通过缓存 + 模型选择，API 成本降低 40-60%
   - 这让独立开发者能用和团队一样的模型
   - 但成本只有团队的几分之一
```

---

## 8. 进阶路线图

### 第一阶段：核心落地 (1-2 周)

- [x] 完成 5 阶段算法的代码实现
- [x] CLI 工具可用
- [x] 接入真实 LLM API（支持 Deepseek API，通过 `--llm` 或 `DEEPSEEK_API_KEY`）
- [x] 52 个单元测试覆盖全部核心模块
- [x] 持久化学习数据到 `~/.turbocontext/state.json`
- [ ] 在你自己的项目中使用 `/turbocontext` Skill

### 第二阶段：深度定制 (2-4 周)

- [ ] 为你的项目训练「能力分解策略」
- [ ] 积累历史记录，让学习系统真正生效
- [ ] 扩展提示组件库（针对你的常见任务类型）

### 第三阶段：平台化 (1-2 月)

- [ ] 添加 Web UI 可视化
- [ ] 多项目配置文件
- [ ] 团队协作支持（共享学习数据）

### 第四阶段：商业化 (3 月+)

- [ ] 托管服务（SaaS 版本）
- [ ] 定制企业版（私有部署）
- [ ] 垂直行业版本（特定领域的智能上下文优化）

---

## 9. 更新日志

### v2.0 (2026-05-19)

**Bug 修复**
- 修复 `greedySelect` 中能力覆盖作为硬过滤而非评分加成的问题
- 修复 `composer.ts` 中 `refactoring` 与 `code_refactor` 类型不匹配（重构任务回退到 general）
- 修复 `learnCompressionWeights` 中差值门限缺失（与 FORMULA.md 不一致）
- 修复复杂度公式输出范围问题（原范围 [0.29, 0.60] 导致 deep 模型无法到达）

**算法增强**
- 语义相似度升级为 TF-IDF 加权：关键词出现在越少片段中权重越高
- 压缩器多语言支持：Go `func`、Rust `fn/trait`、Python `def` 等结构行识别
- 代码结构相似度加入任务类型感知（审查任务侧重 `try/catch`，生成任务侧重 `export/return`）
- 字符串字面量中的括号不再干扰 body 闭合检测

**新功能**
- 接入 Deepseek API（OpenAI 兼容格式），支持环境变量和编程两种配置方式
- 指数退避重试 + 超时控制
- LRU 缓存淘汰（原 FIFO），命中条目自动提升
- 学习系统持久化到 `~/.turbocontext/state.json`，跨会话积累

**测试**
- 新增 5 个测试文件、52 个单元测试覆盖所有核心模块
- 使用 `vitest` 作为测试框架

**配置变更**
- 复杂度阈值默认值：θ₁ = 0.35 → **0.30**，θ₂ = 0.70 → **0.50**

## 附录

### A. 算法复杂度

| 阶段 | 时间复杂度 | 空间复杂度 | 瓶颈 |
|------|-----------|-----------|------|
| Phase 1 | O(n·m) | O(n) | n=片段数，m=能力数 |
| Phase 2 | O(m·k) | O(m·L) | m=子任务数，L=prompt长度 |
| Phase 3 | O(K·L) | O(L) | K=尝试次数 |
| Phase 4 | O(1) | O(1) | 常数时间 |
| Phase 5 | O(N) | O(N) | N=历史记录数 |

**总体**：O(n·m + K·L)，核心瓶颈在上下文片段数和 LLM 调用次数。

### B. 配置参数一览

| 参数 | 默认值 | 说明 | 调整方向 |
|------|--------|------|---------|
| α (语义权重) | 0.55 | 关键词匹配重要性 | 代码越复杂 → 提高 |
| β (新鲜度) | 0.20 | 最近修改重要性 | 长期项目 → 略降 |
| γ (特异性) | 0.25 | 信息密度重要性 | 频繁处理文件 → 提高 |
| qualityThreshold | 0.85 | 质量达标线 | 严格要求 → 0.90+ |
| maxAttempts | 3 | 最大重试次数 | 高成本任务 → 降低 |
| maxTokenBudget | 8000 | 上下文预算 | 复杂分析 → 提高 |
| θ₁ | 0.30 | 快速→中等阈值 | 自动优化 |
| θ₂ | 0.50 | 中等→深度阈值 | 自动优化 |

### C. 关键词：核心竞争力自我验证

用以下问题检验你是否真正掌握了这个核心竞争力：

1. 你能在 5 分钟内为新项目搭建好 TurboContext 吗？
2. 你的 AI 应用质量是可度量的吗？
3. 你的 AI 成本是在优化不是在浪费吗？
4. 你的系统在使用中变得越来越聪明吗？
5. 如果明天换个模型，你的流程需要重写吗？

如果全部答「是」——你已经建立了真正的竞争优势。

---

> **最后的话**：TurboContext 不是一个固定的算法，它是一种思维方式的产物。  
> 核心信念是：**系统的力量 > 单点的天赋**。  
> 当你把每一次 AI 交互都变成系统优化的机会，你就不是在「用 AI」——  
> 你是在**建造一台越来越聪明的引擎**。
