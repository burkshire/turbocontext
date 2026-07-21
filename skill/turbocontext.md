---
name: turbocontext
description: |
  Adaptive context optimization with RL learning. Compresses context, builds multi-round
  prompts, applies quality-gated generation, selects optimal model tier, and continuously
  learns from outcomes. Usage: /turbocontext <task> [--dir <path>] [--type <task_type>]
---

# TurboContext v6 — Algorithm Skill

When invoked with `/turbocontext`, apply the 5-phase pipeline to the user's request.

## Phase 1: Context Compression

1. **Scan**: Read files in the working directory (or `--dir` path). Collect structural metadata.
2. **Score**: For each file, compute relevance to the task:
   - `score = α·keyword_similarity + β·recency + γ·information_density`
   - Defaults: α=0.55, β=0.20, γ=0.25 (RL-evolved per task type)
3. **Select**: Within the token budget (default 8000), greedily pick highest-scoring files.
   Apply MMR diversity re-ranking: `argmax λ·relevance - (1-λ)·max_similarity_to_selected`
4. **Compress**: Strip function bodies (keep signatures), collapse imports, remove comments.
   Output: compressed context C' that fits within token budget.

## Phase 2: Prompt Architecture

Decompose task into 3 rounds (understand → execute → verify):

| Task Type | Round 1 | Round 2 | Round 3 |
|-----------|---------|---------|---------|
| code_review | Understand changes | Check code quality | Generate review |
| code_generation | Analyze requirements | Generate implementation | Check quality/safety |
| debugging | Understand bug context | Generate fix | Verify fix |
| code_refactor | Analyze structure | Execute refactor | Verify result |
| analysis/design | Gather information | Deep analysis | Generate conclusions |

For each round, build: `system_prompt + context_block(C') + task_block + format_block + quality_criteria + (previous_outputs if not first round)`.

## Phase 3: Quality-Gated Generation

```
for attempt in 1..maxAttempts:
    temperature = schedule[attempt]  # default: [0.7, 0.35, 0.1]
    output = LLM(prompt, temperature)
    quality = evaluate(output)      # 4 dims: completeness, correctness, consistency, format
    if quality >= threshold:         # default: 0.85
        return output
    feedback = critique(output, quality)
    prompt = inject_feedback(prompt, feedback)
```

**Quality evaluation** uses heuristics (structural checks, code block presence, error patterns) AND the V6 Quality Proxy when calibrated (predicts real quality from cheap signal extraction).

**Hard signal** (when applicable): compile generated code, run smoke tests. Code that doesn't compile is capped at 0.45 quality regardless of heuristic score.

## Phase 4: Cost Optimization

1. Estimate task complexity from type + description length + history.
2. Select model tier: `complexity < θ₁ → fast, θ₁ ≤ complexity < θ₂ → medium, complexity ≥ θ₂ → deep`
   - Defaults: θ₁=0.30, θ₂=0.42 (RL-evolved)
3. Apply latency constraints if specified.
4. Check semantic cache (LRU, 100 entries, 5-min TTL) before calling LLM.

## Phase 5: RL Learning

After every invocation, record a Trial and update the RL engine:

1. **Predict quality** (bootstrap ensemble), compute surprise (|predicted - actual|)
2. **Thompson update**: retrieved memories get Beta(α,β) updated based on outcome
3. **TD(λ) credit**: eligibility traces propagate reward backward through retrieval chain
4. **SGD update**: predictive model learns from surprise-weighted gradient
5. **Counterfactuals**: synthesize 2-5 "what-if" insights (no LLM call needed)
6. **Periodic** (every N invocations where N=curriculum learning interval):
   - Evolution step: mutate retrieval/policy params, keep if fitness improves
   - Memory consolidation: merge low-utility entries, archive cold ones
   - Adversarial verification: re-score old successes against current best

## State Persistence

State is persisted to `~/.turbocontext/state-v5.json` (atomic writes with backup rotation).
JSONL audit logs at `~/.turbocontext/logs/{trials,evolution,consolidation}.jsonl`.

## Output Format

```
## TurboContext Result

**Task:** <description>
**Compression:** <N> files selected, <ratio>% compression ratio
**Architecture:** 3 rounds (<task_type>)
**Model:** <tier> ($<cost>)
**Quality:** <score>% | Attempts: <N>
**Coverage:** <covered capabilities>
**V6 Diagnostics:** curriculum phase <N>, proxy accuracy <N>%

<generated output>
```

## Configuration

Default parameters (evolved by RL engine per task type):
- Compression: α=0.55, β=0.20, γ=0.25
- Quality: threshold=0.85, maxAttempts=3
- Temperature: [0.7, 0.35, 0.1]
- Model tiers: θ₁=0.30, θ₂=0.42
- Retrieval: MMR λ=0.70, topK=5
- Budget: 8000 tokens

## Quick Reference

- Dev run: `npx tsx src/cli.ts run -t "..." -d ./src --type code_review`
- Demo: `npx tsx src/cli.ts demo`
- Tests: `npx vitest run` (378 tests)
- Formula reference: `FORMULA_V5.md`
- Learn guide: `LEARN.md`
