# TurboContext：完整算法公式体系（v2.0 优化版）

> 本文档包含完整的数学公式和算法伪代码。
> 更新至 v2.0，反映所有优化改进（TF-IDF 加权、LRU 缓存、复杂度范围修复、Deepseek API 集成）。

---

## 核心公式：一次表达整个算法

整个 TurboContext 算法可以浓缩为一个统一公式：

```
O* = argmax_{k∈[1,K], m∈{fast,medium,deep}}

     [ Σᵢ wᵢ · qᵢ( LLM( P( compress(T,C | α,β,γ), T ), tₖ ) ) ]

     · exp( -λ · cost_per_token(m) · tokens(C'_m) · retry_factor(m,T) )

     约束: tokens(C'_m) ≤ budget   ∧   Q(oₖ) ≥ θ_Q
```

而此公式背后的学习回路是一个递归优化：

```
Θ_{n+1} = Θ_n  +  η · ∇J(H_n)

Θ = { α, β, γ, θ_Q, θ₁, θ₂, tₖ, wᵢ, λ }
J(H) = 历史质量-成本损失函数
```

10 个符号涵盖全部 5 个阶段：

| 符号 | 含义 | 对应阶段 |
|------|------|---------|
| `compress(T,C)` | 按权重 α,β,γ 压缩上下文 | Phase 1 |
| `P(., T)` | 按任务类型分解为多轮提示 | Phase 2 |
| `LLM(p, tₖ)` | 第 k 次尝试，温度 tₖ | Phase 2→3 |
| `Q = Σ wᵢ·qᵢ` | 4 维质量加权评分 | Phase 3 |
| `m = f(complexity)` | 成本-复杂度驱动的模型选择 | Phase 4 |
| `λ(H)` | 成本敏感度，从历史学习 | Phase 5 |
| `Θ ← learn(H)` | 全部参数的自适应更新 | Phase 5 |

### 人话版本

> **在 token 预算内，用最合适的模型，把最有用的上下文喂给 LLM，多次尝试直到质量达标，然后从结果中学习下次怎么选得更好。**

这是**算法内核**。以下各章是它展开后的工程实现。

### 与 Facemash Elo 的对比

| 对比项 | Facemash Elo | TurboContext |
|--------|-------------|-------------|
| 公式结构 | 单一静态函数 | 多阶段复合 + 学习回路 |
| 输入 | 两名选手的当前 Elo | 任务 + 上下文 + 历史 |
| 输出 | 胜率预测 | 经质量门控的生成结果 |
| 学习方式 | 全人类共同排名 | 每个用户的自适应配置 |
| 决策维度 | 1 维 (评分差) | 4 维 (质量) + 成本 + 延迟 |

共同点：都是一个自封闭的反馈系统——行为被评估，评估结果调整后续行为。

---

## Phase 1: 上下文压缩与评分 (Context Compression)

### 1.1 能力需求分解

将任务 T 分解为能力需求集合 R：

```
R = {r₁, r₂, ..., rₙ}
r_i = (name_i, weight_i, description_i)

weight_i = Σ_j match(keyword_ij, T) / |keywords_i|
match(kw, T) = 1  if kw ∈ T, 0  otherwise
```

预定义的六项能力：

| 能力 | 关键词示例 | 默认权重 |
|------|-----------|---------|
| code_understanding | understand, read, analyze, 审查 | 0.25 |
| pattern_recognition | pattern, detect, find, 识别 | 0.15 |
| code_generation | write, create, implement, 生成 | 0.25 |
| code_modification | change, modify, refactor, 重构 | 0.15 |
| error_detection | bug, error, issue, 缺陷 | 0.10 |
| design | design, architecture, 架构 | 0.10 |

### 1.2 片段评分函数

```
score(cᵢ) = α · sim(cᵢ, T | C) + β · recency(cᵢ) + γ · specificity(cᵢ)
约束: α + β + γ = 1
默认: α = 0.55, β = 0.20, γ = 0.25
```

#### 语义相似度 — TF-IDF 加权（优化版）

原始公式为简单关键词匹配，v2.0 升级为 TF-IDF 加权：

```
sim(cᵢ, T | C) = 0.50 · keyword_idf(cᵢ, T, C) + 0.30 · type_match(cᵢ, T) + 0.20 · struct_match(cᵢ, T)

keyword_idf(cᵢ, T, C) = Σ_j [idf(w_j) · 1{w_j ∈ cᵢ}] / Σ_j idf(w_j)
其中 w_j = tokenize(T) 的第 j 个词

idf(w) = log(|C| / (1 + count({c ∈ C | w ∈ c}))) + 1
```

**与原始版的区别**：
- 原始版：`matchedWords.length / taskWords.length`，所有词等权
- 优化版：IDF 加权，出现在越少片段中的词权重越高。`function`、`class` 等通用词自动降权

#### 代码结构相似度 — 任务感知（优化版）

```
struct_match(cᵢ, T) = defMatches / |defPatterns(T)|

defPatterns(T):
  审查/调试: [function, class, interface, type, if, for, try, catch, throw]
  生成/重构: [function, class, interface, const, let, def, impl, return, export]
  其他:      [function, class, interface, const, let, def, impl]
```

**与原始版的区别**：原始版对所有任务类型使用相同的关键词集，优化版根据任务类型差异化选择。

#### 新鲜度

```
recency(cᵢ) = 1 / (1 + days_since_last_modified)
days_since_last_modified = (now - mtime(cᵢ)) / 86400000
```

#### 特异性

```
specificity(cᵢ) = 1 - min(len(cᵢ), MAX_LEN) / MAX_LEN
MAX_LEN = 5000
```

### 1.3 带约束的贪婪选择（优化版）

```
目标: 在 token 预算内最大化能力覆盖
最大化 Σ_i Σ_j score(cᵢ) · cover(cᵢ, rⱼ)
约束: Σ_i tokens(cᵢ) ≤ budget
      ∀rⱼ ∈ R: Σ_i cover(cᵢ, rⱼ) ≥ 1

算法:
  已选中 = ∅
  对 rⱼ ∈ R (按 weightⱼ 降序):
    // 优化: 候选不按能力预过滤，改用评分加成
    c* = argmax_{c ∈ C \ 已选中} [
      score(c) · 1.2^{cover(c, rⱼ)} - |c|/budget × 0.1
    ]
    已选中 = 已选中 ∪ {c*}

  对剩余预算按 adjustedScore(c) 降序补充
```

**与原始版的区别**：
- 原始版用 `coversCapability` 做硬过滤 → 不覆盖当前能力的片段直接被排除
- 优化版去掉硬过滤，`1.2^{cover}` 作为评分加成（覆盖→+20%，不覆盖→不变）
- 高评分但不覆盖当前能力的片段仍然可以凭总分入选

### 1.4 片段压缩

```
compress(c):
  逐行处理:
    如果 isStructuralLine(line) → 保留原样，进入 body 模式
    否则如果 inBody:
      统计 netBraceDepth（跳过字符串字面量中的括号）
      如果 braceDepth ≤ 0 → body 结束，输出摘要
      否则 → 缓冲到 bodyLines
    否则如果 不是空行/注释 → 保留

  isStructuralLine 匹配规则（多语言支持）:
    /^(export\s+)?(function|class|interface|type|enum|struct|trait|impl|def|fn|pub)/
    /^func\s+\w+/                                             // Go
    /^def\s+\w+\s*\(/                                         // Python
    /^class\s+\w+/                                             // Python, Java, etc.
    /^(const|let|var|val)\s+\w+\s*[=:]/
    /^(import|export|from|use|require|include|mod)\s/
    /^package\s/  /^#include\s/
    /^(public|private|protected|static|abstract|sealed|open|internal)/
    /^(if|else\s+if|for|while|do|switch|match|try|catch|finally|with)\b/
    /^(@\w+|#\[)/                                             // 注解/属性

  body 压缩:
    如果 |body| > 5 行 → "// ... [N lines omitted]"
    否则 → 保留原样

  括号追踪（优化版）:
    跳过字符串字面量 ("、'、`) 内的括号
    正确识别 else/catch/finally 等闭合跟随结构
```

**与原始版的区别**：
- 原始版：BraceDepth 追踪不排除字符串中的括号；body 摘要阈值为 3 行；结构性行不识别 `else if`、`catch`、`finally`
- 优化版：跳过字符串字面量中的括号避免误闭合；摘要阈值改为 5 行减少过度压缩；新增 Go、Python、Rust 语言模式；识别 `else if`/`catch`/`finally`/`end` 等闭合跟随结构

### 1.5 压缩比

```
compression_ratio = 1 - compressed_tokens / original_tokens
compressed_tokens = Σ_i ceil(len(preserved_i) / 4)
original_tokens   = Σ_i ceil(len(content_i) / 4)
```

### 1.6 Token 估算

```
estimate_token_count(text) = Σ ceil(len(text) / 4)
```

---

## Phase 2: 提示架构组合 (Prompt Composition)

### 2.1 任务分解

```
T → S = {s₁, s₂, s₃}  (固定 3 轮)

任务类型 → 分解策略:
  code_review:
    s₁ = 理解变更上下文和目的
    s₂ = 逐模块检查代码质量
    s₃ = 生成审查总结

  code_generation:
    s₁ = 分析需求和约束
    s₂ = 生成核心实现
    s₃ = 检查生成代码的质量和安全性

  debugging:
    s₁ = 理解 Bug 上下文和复现路径
    s₂ = 生成修复方案
    s₃ = 验证修复

  code_refactor:         ← v2.0 修复：原为 refactoring，与类型定义不匹配
    s₁ = 分析现有结构和重构目标
    s₂ = 执行逐步重构
    s₃ = 验证重构结果

  analysis:
    s₁ = 收集和整理相关信息
    s₂ = 深度分析
    s₃ = 生成结论和建议

  general:
    s₁ = 全面理解任务要求
    s₂ = 执行核心任务
    s₃ = 质量复核
```

### 2.2 提示生成

```
P = {prompt(s₁), prompt(s₂ | o₁), ..., prompt(s₃ | o₁, o₂)}

prompt(sᵢ | prev_outputs) =
    system_prompt(sᵢ) ++
    context_block(C') ++
    task_block(sᵢ) ++
    format_block() ++
    quality_block() ++
    (if i > 1: prev_outputs_block(prev_outputs))

system_prompt(sᵢ):
    角色定义: f(task_type)
    当前目标: sᵢ.goal
    输出格式: sᵢ.outputFormat
    质量标准: sᵢ.qualityCriteria
```

---

## Phase 3: 质量加权生成 (Quality-Weighted Generation)

### 3.1 核心生成循环

```
for k = 1 to K:
    tₖ    = temperature_schedule[k]
    oₖ    = LLM(Pₖ, tₖ)              // 调用 LLM（支持真实 API 或模拟）
    Q(oₖ) = Σᵢ wᵢ · qᵢ(oₖ)          // 四维度加权评估
    if Q(oₖ) ≥ θ_Q:
        return oₖ
    else:
        fₖ = critique(oₖ, Q)          // 自动生成反馈
        Pₖ₊₁ = Pₖ ⊕ fₖ               // 注入反馈 + 重试

return o_K   (最后一次结果)
```

**参数**：
```
K         = 3 (最大尝试次数)
θ_Q       = 0.85 (质量阈值)
t_schedule = [0.7, 0.35, 0.1] (温度递减调度)
```

### 3.2 LLM 调用接口（v2.0 新增）

```
createLLMCall(config):
    apiKey  = config.apiKey || env.DEEPSEEK_API_KEY
    baseUrl = config.baseUrl || env.DEEPSEEK_BASE_URL || "https://api.deepseek.com"
    model   = config.model || "deepseek-chat"
    maxRetries = config.maxRetries ?? 3
    timeout    = config.timeoutMs ?? 60000

    返回 async function llmCall(prompt, temperature):
        messages = parsePromptToChatML(prompt)    // 解析 Phase 2 架构为 ChatML
        body = { model, messages, temperature, max_tokens: 4096 }

        for attempt = 1 to maxRetries:
            try:
                response = POST(baseUrl + "/v1/chat/completions", body, timeout)
                return response.choices[0].message.content
            catch:
                wait exponential_backoff(attempt)
                retry

        return "[TurboContext LLM Error: ...]"   // 所有重试失败后不崩溃
```

defaultLLMCall（无 API key 时的回退）生成与任务类型匹配的模拟输出，使质量评估可实际运行。

### 3.3 质量评估函数

```
Q(o) = w₁ · q_cmpl + w₂ · q_corr + w₃ · q_cnst + w₄ · q_fmt
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
| documentation | 0.30 | 0.15 | 0.25 | 0.30 |
| design | 0.25 | 0.20 | 0.30 | 0.25 |
| general | 0.30 | 0.25 | 0.25 | 0.20 |

### 3.4 维度评分细则

**完整性**：
```
q_cmpl = |需求点被覆盖| / |总需求点|

需求点提取: 从 T 中提取名词短语和关键动词（正则匹配模式）:
  /(实现|创建|添加|修改|删除)\s*(\S+)/g
  /(需要|要求|必须)\s*(\S+(?:\s+\S+){0,2})/g
  /\b(implement|create|add|fix)\s+(\S+(?:\s+\S+){0,3})/gi
```

**正确性**：
```
q_corr = 1 - Σⱼ penalty(flagⱼ) × indicator(detectedⱼ)

惩罚列表:
  不确定表达 (sorry, I don't know, I cannot)     → 0.10
  AI 回避 (as an AI, I'm not sure)              → 0.10
  占位符 (TODO, FIXME, placeholder)             → 0.20
  推测语 (assuming, might be, perhaps)          → 0.05
  不完整声明 (incomplete, partial, draft)        → 0.10
  错误引用 (TypeError, undefined is not function) → 0.25
```

**一致性**：
```
q_cnst = 1 - Σⱼ penalty(contradictionⱼ)

矛盾检测:
  同一概念使用不同术语 (user-id vs user-identifier)   → -0.05
  相反指令同时出现 (do not use X vs 却用 X)          → -0.10
  声明数量与实际不符 (声称 3 步但只列出 2 步)         → -0.05
```

**格式合规**：
```
q_fmt = 1 - Σⱼ penalty(format_issueⱼ)

检查项:
  需要代码但无代码块            → -0.30
  代码块未闭合 (```不成对)      → -0.40
  输出过短 (< 10 字符)          → -0.50
  超长行 (> 200 字符, 每 3 行)  → -0.10
```

### 3.5 反馈生成

```
critique(o, Q):
    问题 = detect_issues(o, Q.dimensions)
    反馈 = "## 质量反馈 (第 k 轮)
            总体质量: {Q.score × 100}%
            目标阈值: {θ_Q × 100}%

            ### 各维度得分
            - 完整性: {q_cmpl × 100}%
            - 正确性: {q_corr × 100}%
            ...

            ### 改进要求
            {对不达标维度的具体改进指令}"
    返回 反馈
```

---

## Phase 4: 成本-延迟优化 (Cost-Latency Optimization)

### 4.1 复杂度评估（优化版）

```
complexity(T | H) = w_type · type_base(T) + w_ambig · ambiguity(T)
                  + w_hist · historical(T) + w_base · baseline

权重: w_type = 0.40, w_ambig = 0.15, w_hist = 0.20, w_base = 0.25

type_base(T):          // 任务类型基础复杂度
  documentation → 0.25,  debugging → 0.35,  general → 0.35
  code_review → 0.40,    testing → 0.40,     code_generation → 0.45
  analysis → 0.50,       code_refactor → 0.55
  design → 0.65

ambiguity(T) = f(|T|):
  |T| < 20   → 0.8   // 高度模糊
  |T| < 50   → 0.5
  |T| < 200  → 0.3
  otherwise  → 0.2   // 描述详细

historical(T | H) = Σ_i [quality_i < 0.8 → 0.6 else 0.3] / N_history
  // 默认 0.3（无历史时）
baseline = 0.2       // ← v2.0 从 0.4 降低，扩大输出范围
```

**与原始版的区别**：
- 原始版 `baseline = 0.4`，输出范围 [0.29, 0.60]
- 优化版 `baseline = 0.2`，输出范围 [0.24, 0.55]
- 原范围中 deep 模型阈值 (θ₂=0.70) **永远无法到达**

### 4.2 模型选择

```
model = f(complexity, latency_budget):

  if complexity < θ₁:   tier = "fast"   (Haiku,  $0.25/1M tokens, ~2s latency)
  elif complexity < θ₂: tier = "medium" (Sonnet, $3/1M tokens,   ~5s latency)
  else:                 tier = "deep"   (Opus,   $15/1M tokens,  ~15s latency)

  延迟约束覆盖:
    if latency_budget < model_latency AND tier ≠ "fast":
      降级到下一级
    if latency_budget >> model_latency: 保持当前级别

默认阈值: θ₁ = 0.30, θ₂ = 0.50    ← v2.0 从 (0.35, 0.70) 调整
```

### 4.3 成本模型

```
expected_cost(T) = (estimated_tokens / 1000) · cost_per_1k · expected_attempts

expected_attempts:
  代码生成/调试: 1.5
  其他: 1.2
```

### 4.4 缓存策略（优化版：LRU 淘汰）

```
fingerprint = hash(task_type + task_description + context_prefix_200chars)

lookup:
  entry = cache[fingerprint]
  if entry AND (now - entry.timestamp) < 5min:
    // LRU 提升: 删除并重新插入到 Map 末尾
    cache.delete(fingerprint)
    cache.set(fingerprint, { ...entry, timestamp: now })
    return entry.result
  else:
    return null

write:
  if cache.size ≥ 100:
    // LRU 淘汰: Map 首部是最久未访问的条目
    evict cache.keys().next()
  cache[fingerprint] = { result, quality, timestamp, model }
```

**与原始版的区别**：
- 原始版：FIFO 淘汰（`entries().next()` 删除最早插入的条目），命中不更新位置
- 优化版：LRU 淘汰，命中后通过 `delete + set` 将条目移到 Map 末尾

---

## Phase 5: 连续学习 (Continuous Learning)

### 5.1 持久化（v2.0 新增）

```
状态存储:
  路径: ~/.turbocontext/state.json
  内容: { config, history[], taskTypeStats{} }

保存触发器:
  每次 record() 调用后自动保存

加载触发器:
  Learner 构造时自动从磁盘加载
  文件不存在 → 使用默认配置

保留上限:
  history: 最近 200 条
  超出的条目自动丢弃
```

### 5.2 压缩权重更新（优化版）

```
每 5 次执行后触发:

如果 avgCompression_high - avgCompression_low > 0.2   ← 修复：原始实现缺少 > 0.2 门限
  且 high_quality_count ≥ 3:
  α -= 0.01   (降低语义权重，压缩效果好时保留更多信息密度)
  γ += 0.02   (增加特异性权重)
  约束: α ≥ 0.35, γ ≤ 0.40

如果 avgCompression_low < 0.3  且 low_quality_count ≥ 3:
  α += 0.02   (增加语义权重，信息丢失过多时更依赖语义匹配)
  γ -= 0.01   (降低特异性权重)
  约束: α ≤ 0.70, γ ≥ 0.15

归一化: (α, β, γ) = (α, β, γ) / (α + β + γ)

其中:
  avgCompression_high = mean({compression_i | quality_i ≥ 0.85})
  avgCompression_low  = mean({compression_i | quality_i < 0.70})
```

**与原始版的区别**：
- 原始条件：`avgHighCompression > 0.5 && avgHighCompression > avgLowCompression`
- 优化条件：`avgHighCompression - avgLowCompression > 0.2`（与 FORMULA.md 一致）
- 原始条件少了差值门限，导致只要有高压缩就触发，无论低压缩表现如何

### 5.3 复杂度阈值更新

```
如果 fast_model_pass_rate > 0.9 AND fast_count ≥ 3:
  θ₁ += 0.03   (扩大快速模型使用范围)
  约束: θ₁ ≤ 0.45

如果 fast_model_pass_rate < 0.7 AND fast_count ≥ 3:
  θ₁ -= 0.03   (缩小快速模型使用范围)
  约束: θ₁ ≥ 0.20

如果 deep_model_fail_rate > 0.3 AND deep_count ≥ 3:
  θ₂ += 0.03   (减少不必要的深度调用)
  约束: θ₂ ≤ 0.85
```

### 5.4 温度调度更新

```
如果 avg_attempts ≤ 1.1:
  t₀ = max(0.3, t₀ - 0.05)   // 一次成功率高，降低初温

如果 avg_attempts ≥ 2.5:
  t₀ = min(0.9, t₀ + 0.05)   // 总是需重试，提高初温
```

### 5.5 学习率与窗口

```
学习率:      η = 0.1
历史窗口:    N = 100 条最新记录
学习间隔:    每 5 次执行触发一次学习
最少触发:    至少 5 条记录才执行学习
持久化路径:  ~/.turbocontext/state.json
```

---

## 完整算法伪代码

```
Algorithm TurboContext(T, C, Θ):

  // === Phase 1: 上下文压缩 ===
  R    ← decomposeCapabilities(T)
  for c ∈ C:
    c.score ← α·sim(c,T,C) + β·recency(c) + γ·specificity(c)   // TF-IDF 加权
  S    ← greedySelect(C, R, budget)                            // 无硬过滤
  for s ∈ S:
    s.compressed ← compress(s)                                 // 字符串感知压缩
  C'   ← {s.compressed | s ∈ S}
  cov  ← computeCoverage(C', R)

  // === Phase 2: 提示架构 ===
  subtasks ← decomposeTaskByType(T)                            // code_refactor 已修复
  P ← []
  for s ∈ subtasks:
    p ← composePrompt(s, C', prev_outputs)
    P ← P ∪ {p}

  // === Phase 4: 成本优化 ===
  cx   ← estimateComplexity(T, history)                        // baseline=0.2
  M    ← selectModel(cx, latency_budget)                       // θ₁=0.30, θ₂=0.50
  cost ← estimateCost(T, tokens(P), M)
  cacheResult ← lookupCache(T, C')                             // LRU 淘汰
  if cacheResult: return cacheResult

  // === Phase 3: 质量加权生成 ===
  for k = 1 to K:
    t    ← temperature_schedule[k]
    o    ← callLLM(P, t)                                       // Deepseek API / 模拟
    q    ← evaluateQuality(o, T)
    record(generation{k, o, q})
    if q.score ≥ θ_Q:
      writeCache(T, C', o, q, M)                               // 缓存结果
      break
    f    ← generateFeedback(q)
    P    ← injectFeedback(P, f)

  // === Phase 5: 连续学习 ===
  history ← history ∪ {task, C', M, q, cost}
  save()                                                       // 持久化到磁盘
  if |history| % 5 = 0:
    Θ ← learn(Θ, history)

  return { output: o, quality: q, cost, coverage: cov, learning_adjustments }
```

---

## 算法复杂度

| 阶段 | 时间复杂度 | 空间复杂度 | 瓶颈 |
|------|-----------|-----------|------|
| Phase 1 | O(n·m + n·log n) | O(n) | n=片段数, m=能力数 |
| Phase 2 | O(m·k) | O(m·L) | m=子任务数, L=prompt长度 |
| Phase 3 | O(K·L·R) | O(L) | K=尝试次数, R=重试次数 |
| Phase 4 | O(1) | O(100) | 常数（缓存上限 100 条）|
| Phase 5 | O(N) | O(N) | N=历史记录数（上限 200）|

总体：**O(n·m + K·L)**，核心瓶颈在上下文片段数和 LLM 调用次数。

---

## 配置参数一览（v2.0）

| 参数 | 默认值 | 说明 | 变更 |
|------|--------|------|------|
| α (语义权重) | 0.55 | 关键词匹配重要性 | — |
| β (新鲜度) | 0.20 | 最近修改重要性 | — |
| γ (特异性) | 0.25 | 信息密度重要性 | — |
| qualityThreshold | 0.85 | 质量达标线 | — |
| maxAttempts | 3 | 最大重试次数 | — |
| maxTokenBudget | 8000 | 上下文预算 | — |
| θ₁ | **0.30** | 快速→中等阈值 | 0.35→0.30 |
| θ₂ | **0.50** | 中等→深度阈值 | 0.70→0.50 |
| baseline | **0.2** | 复杂度基准因子 | 0.4→0.2 |
| 缓存策略 | **LRU** | 淘汰策略 | FIFO→LRU |
| 持久化 | **~/.turbocontext/state.json** | 学习数据 | 新增 |
| LLM 后端 | **Deepseek API** | API 接入 | 新增 |
| 语义评分 | **TF-IDF 加权** | 关键词权重 | 等权→IDF |
