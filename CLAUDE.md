# TurboContext — AI Agent Onboarding

> **TL;DR:** TurboContext is an adaptive context optimization algorithm with a reinforcement learning engine. It compresses LLM context, builds multi-round prompt architectures, applies quality-gated generation, selects optimal models, and continuously learns from outcomes. V6 adds Karpathy-inspired experiment diversity, unified efficiency metric, simplicity criterion, cross-branch transfer, and branch health tracking.

## Project Identity

- **Language:** TypeScript (ES2022 modules, `"type": "module"`) + one Python 3 self-contained reference implementation
- **Framework:** Zero external runtime dependencies beyond CLI utilities (chalk, commander, zod, cli-progress); no Express, no React
- **Test framework:** Vitest 4.x (`npm test` or `npx vitest run`)
- **Runtime:** Node.js ≥20 (uses `node:fs`, `node:child_process`, `node:os`, `node:path`)
- **Package manager:** npm (has `package-lock.json`)
- **Entry point:** `bin/turbocontext.js` (compiled) or `npx tsx src/cli.ts` (dev)
- **State file:** `~/.turbocontext/state-v5.json` (persistent RL state, migrated from V4 state.json)
- **Not a git repo** — no version control. Be careful with destructive changes.

## Test Map

| Test File | Covers | Tests |
|-----------|--------|-------|
| `compressor.test.ts` | Phase 1: scoring, selection, compression | ~25 |
| `composer.test.ts` | Phase 2: prompt architecture decomposition | ~15 |
| `generator.test.ts` | Phase 3: quality evaluation, feedback | ~12 |
| `optimizer.test.ts` | Phase 4: model selection, caching, cost | ~15 |
| `learner.test.ts` | Phase 5: branch learning, plateau detect | ~25 |
| `ablation-engine.test.ts` | Ablation: target selection, execution | ~20 |
| `causal-graph.test.ts` | Causal discovery: skeleton, v-structure, FCI, GES | ~22 |
| `rl-core.test.ts` | RL primitives: Thompson, TD(λ), curriculum | ~30 |
| `execution-verifier.test.ts` | Execution verification pipeline | ~20 |
| `project-compiler.test.ts` | Compilation + smoke test pipeline | ~18 |
| `experiment-e2e.test.ts` | End-to-end experiments | ~25 |
| `quality-proxy.test.ts` | V6 quality proxy calibration | ~20 |
| `v5-feedback-loop.test.ts` | V5 feedback loop integration | ~15 |
| `eval-v31.ts` | Evaluation benchmarks | ~15 |
| `state/__tests__/policy.test.ts` | V5 policy resolution | ~8 |
| `state/__tests__/retrieval.test.ts` | V5 7-dim MMR retrieval | ~10 |
| `state/__tests__/thompson.test.ts` | V5 Thompson sampling | ~8 |
| `state/__tests__/rnd.test.ts` | V5 RND curiosity | ~8 |
| `autonomous-experiment-loop.test.ts` | **NEW v6** — E2E autonomous experiment loop | ~6 |
| Additional test files | Various component tests | ~50 |

**Total: 370 tests across 22 files, all passing as of 2026-07-21.**

## Current State (2026-07-21 — v6 Evolution)

- **All 370 tests passing**, 22 test files (up from 364/21)
- **V5→V6 evolution** complete:
  - **Unified Efficiency Metric**: Single "north star" metric with simplicity multiplier (Karpathy's `val_bpb` equivalent)
  - **6 Experiment Types**: hypothesis_test, parameter_sweep, ablation_study, transfer_experiment, boundary_probe, adversarial_test
  - **Multi-Target Mutations**: Compression weights, model tiers, temperature, quality weights, retrieval params
  - **Simplicity Criterion**: 0.02× bonus for simpler changes (stronger than v2.3's 0.01×)
  - **Branch Health Tracking**: improvementVelocity, stabilityScore, noveltyScore, plateauConfidence per task type
  - **Cross-Branch Transfer**: Bootstrap under-explored task types from similar ones (Jaccard ≥ 0.4, ≥10 source trials)
  - **Program Loader**: mission.md → ResearchProgram with mutation filtering (Karpathy's program.md pattern)
  - **Quality Proxy Integration**: Pre-call skip for hopeless attempts, model tier downgrade optimization
  - **State Persistence Fix**: V4→V5 migration, `state-v5.json` now correctly persisted
  - **New modules**: `src/core/program-loader.ts`, `src/state/transfer/cross-branch-transfer.ts`
- **Skill definition** at `skill/turbocontext.md` (102KB)
- **Python reference** (`turbocontext_v5_rl.py`, 2000+ lines) — needs v6 sync
- **V5 state layer** (`src/state/`) with full RL pipeline: TD(λ), PER, Thompson, RND, HER, curriculum, evolution, memory consolidation, cross-context buffer

## Key Design Patterns

### V6 RL Loop (the heart of the learning system)
```
Execution → Trial construction → Record in SharedStateManager
    → RL pipeline: TD(λ) credit assignment, PER, Thompson sampling
    → Predictive model update (quality prediction)
    → Surprise computation (prediction error)
    → Curriculum phase check (every N invocations)
    → Evolution cycle: propose multi-target mutation → trial → keep/discard
    → Cross-branch transfer (when task type has <10 trials)
    → Memory consolidation (when pool > 200 entries)
    → Adversarial verification (verify old successes still hold)
    → Branch health update (velocity, stability, novelty, plateau)
```

### Autoresearch-Inspired Experiment Loop
```
Program loading (mission.md) → Experiment type selection (UCB-weighted)
    → Mutation proposal (6 types, program-filtered)
    → Simplicity computation → Execute with fixed budget
    → Unified metric evaluation (quality / cost × simplicity)
    → Keep/discard with simplicity bonus
```

## Module Catalog

### New v6 Modules

- **`src/core/program-loader.ts`** — Reads mission.md YAML frontmatter, returns ResearchProgram with constraints, budgets, allowed mutations
- **`src/state/transfer/cross-branch-transfer.ts`** — Cross-task-type parameter transfer via Jaccard similarity over capability requirements
- **`tests/autonomous-experiment-loop.test.ts`** — E2E test running 5 experiments in memory mode with simulated LLM

### Enhanced v6 Modules

- **`src/types.ts`** — Added ExperimentType (6 types), extended UnifiedMetric with alpha/simplicityMultiplier, ExperimentRun with experimentType/simplicityScore
- **`src/index.ts`** — Integrated cross-branch transfer, experiment type selection, simplicity, program loading, V4→V5 migration, quality proxy in model selection
- **`src/core/evolution-engine.ts`** — Extended proposeMutation with 5 new mutation targets, selectExperimentType function, stronger simplicity criterion (0.02×)
- **`src/core/generator.ts`** — Enhanced computeUnifiedMetric with alpha/simplicity, v6 proxy pre-check for LLM call skipping, computeSimplicity function
- **`src/core/optimizer.ts`** — Optional qualityProxy in selectModel for cost-saving tier downgrades
- **`src/state/trial/counterfactuals.ts`** — Expanded from 3 to 8+ templates (big_improvement, marginal, repeated_failure, plateau, crash_boundary)
- **`src/state/types.ts`** — Extended TaskTypeBaseline with 7 branch health metrics
- **`src/state/constants.ts`** — Updated emptyBaseline with new health metric defaults
- **`src/state/io.ts`** — Added migrateV4ToV5FromDisk, better error logging on save failures
- **`src/state/state-manager.ts`** — Added loadOrMigrate factory, save success logging
- **`src/state/rl/rl-engine.ts`** — Added loadOrMigrate factory, getSnapshot for cross-branch transfer
- **`src/core/learner.ts`** — Added getEvolutionData for experiment type selection
- **`CLAUDE.md`** — Updated to v6 state (this file)

## Architecture Map

```
turbocontext/
├── src/
│   ├── index.ts                  # TurboContextEngine: main public API, orchestrates all 5 phases
│   ├── cli.ts                    # CLI (commander-based): run, demo, formula, learn, state, eval
│   ├── types.ts                  # ALL shared types (32KB) — Task, ContextFragment, ExecutionRecord, etc.
│   │
│   ├── core/                     # Phase implementations + V5 RL subsystems
│   │   ├── compressor.ts         # Phase 1: context scoring + constraint-aware selection + MMR re-rank
│   │   ├── composer.ts           # Phase 2: task → multi-round prompt architecture decomposition
│   │   ├── generator.ts          # Phase 3: quality-gated generation with temperature annealing
│   │   ├── optimizer.ts          # Phase 4: model-tier selection (fast/medium/deep) + cache
│   │   ├── learner.ts            # Phase 5: branch-based continuous learning (66KB — largest module)
│   │   ├── retrieval-system.ts   # Stateless retrieval: plateau detect, MMR λ, IDF, causal retrieval
│   │   ├── rl-core.ts            # RL primitives: Thompson, TD(λ), predictive model, curriculum, UCB
│   │   ├── evolution-engine.ts   # propose→trial→keep/discard loop for strategy mutations
│   │   ├── rl-feedback-engine.ts # RLFeedbackEngine: eligibility traces, surprise, retrieval strategy
│   │   ├── verifier.ts           # Hard-signal output verification (structural, review, code checks)
│   │   ├── execution-verifier.ts # Composite verifier: compile + smoke-test + structural
│   │   ├── project-compiler.ts   # Compile & smoke-test LLM-generated code in isolated temp dir
│   │   ├── llm.ts                # DeepSeek API client (OpenAI-compatible) with retries
│   │   ├── embeddings.ts         # Pluggable embedding provider (OpenAI-compatible, LRU-cached)
│   │   ├── quality-proxy.ts      # V6: PACE-inspired learned quality predictor
│   │   ├── calibration-generator.ts # V6: synthetic calibration data generator
│   │   ├── signal-extractor.ts   # V6: code signal extraction for calibration
│   │
│   └── state/                    # V5 State management layer (added June 2026)
│       ├── index.ts              # Barrel export: re-exports everything from this directory
│       ├── types.ts              # MDP state space: Trial, PolicyState, ValueFunctionState, etc.
│       ├── constants.ts          # All magic numbers, RL hyperparameters, bounds, createFreshState()
│       ├── io.ts                 # Atomic JSON I/O with backup rotation + JSONL audit logs
│       ├── state-manager.ts      # SharedStateManager: sole mutable owner, dirty-flag persistence
│       ├── periodic-scheduler.ts # PeriodicScheduler: curriculum-phase-gated operation scheduling
│       ├── validation.ts         # State validation on load + v4→v5 migration
│       ├── bridge/               # CrossContextBuffer: skill↔agent async trial queuing
│       ├── curriculum/           # 4-phase curriculum learning scheduler
│       ├── evolution/            # Retrieval strategy self-evolution (meta-learning)
│       ├── memory/               # Memory consolidation + cold storage
│       ├── rl/                   # RL pipeline: TD(λ) credit, PER, Thompson, counterfactual, HER, RND
│       └── trial/                # Trial construction + enrichment
│
├── src/turbocontext_v5_rl.py     # Self-contained Python reference (2000 lines, stdlib only)
├── skill/turbocontext.md         # Claude Code skill definition (100KB)
├── tests/                        # 14 test files + src/state/__tests__/ (5 co-located), 306 tests total
├── dist/                         # Compiled JS output (tsc)
├── bin/turbocontext.js           # CLI entry point after compilation
├── FORMULA_V5.md                 # Full mathematical formula reference (v5, 60KB)
├── LEARN.md                      # 38-lesson tutorial (298KB)
├── README.md                     # Algorithm reference manual
├── mission.md                    # Research mission: optimize for quality/cost efficiency
└── .claude/settings.json         # Registers /turbocontext skill
```

## Module Dependency Flow

```
                        ┌─────────────────────────────────────────────┐
                        │            src/types.ts (32KB)              │
                        │   ALL shared type definitions live here     │
                        └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬───────┘
                           │  │  │  │  │  │  │  │  │  │  │  │
    ┌──────────────────────┘  │  │  │  │  │  │  │  │  │  │  │
    ▼                         ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼
┌────────────┐    ┌──────────────────────────────────────────────────┐
│ embeddings │◄───┤ compressor ──► causal-graph ◄── intervention     │
│ (standalone│    │ (Phase 1)    (causal discovery)   (do-calculus)  │
│  no deps)  │    └──────┬───────┘                                  │
└────────────┘           │                                           │
                         ▼                                           │
                    ┌──────────┐                                     │
                    │ composer │  Phase 2: prompt architecture        │
                    └────┬─────┘                                     │
                         │                                           │
              ┌──────────┼──────────┐                               │
              ▼          ▼          ▼                               │
        ┌──────────┐ ┌──────────┐ ┌──────────┐                      │
        │ generator│ │optimizer │ │ verifier │                      │
        │ (Phase 3)│ │(Phase 4) │ │(hard sig)│                      │
        └────┬─────┘ └────┬─────┘ └────┬─────┘                      │
             │            │            │                             │
             └────────────┼────────────┘                             │
                          ▼                                          │
              ┌───────────────────────┐                              │
              │      learner.ts       │  Phase 5: orchestrates ──┐   │
              │      (68KB)           │                          │   │
              └──┬──┬──┬──┬──┬──┬──┬─┘                          │   │
                 │  │  │  │  │  │  │                              │   │
    ┌────────────┘  │  │  │  │  │  └──────────────┐              │   │
    ▼               ▼  ▼  ▼  ▼  ▼                 ▼              │   │
┌──────────┐  ┌──────────────────────┐  ┌────────────────────┐   │   │
│retrieval │  │  rl-feedback-engine  │  │  evolution-engine  │   │   │
│-system   │  │  (predictive model,  │  │  (propose→trial→   │   │   │
│(plateau, │  │   traces, strategy)  │  │   keep/discard)    │   │   │
│ MMR, IDF)│  └──────────┬───────────┘  └─────────┬──────────┘   │   │
└────┬─────┘             │                        │              │   │
     │                   ▼                        ▼              │   │
     │           ┌──────────────┐        ┌──────────────────┐    │   │
     │           │   rl-core    │        │ablation-engine   │    │   │
     │           │ (Thompson,   │        │(per-file causal  │    │   │
     │           │  TD(λ), UCB) │        │ contribution)    │    │   │
     │           └──────┬───────┘        └──────────────────┘    │   │
     │                  │                                         │   │
     └──────────────────┼─────────────────────────────────────────┘   │
                        │                                             │
                        ▼                                             ▼
              ┌──────────────────┐                        ┌──────────────┐
              │counterfactual.ts │                        │state/ layer  │
              │(template synth)  │                        │(V5, optional)│
              └──────────────────┘                        └──────────────┘
```

**Key dependency rules:**
- `types.ts` has NO internal imports — it's the root of all type dependencies
- `compressor.ts` depends on `embeddings.ts`, `causal-graph.ts`, and `retrieval-system.ts`
- `learner.ts` is the god module — it imports from 9 other modules
- `state/` modules are V5 and can import from each other but NOT from `core/`
- `core/` does NOT import from `state/` — the bridge pattern is one-directional
- `llm.ts`, `embeddings.ts`, and `project-compiler.ts` have NO internal dependencies (leaf nodes)

## Entry Points and Commands

### Development (running source directly)
```bash
npx tsx src/cli.ts help              # Show all commands
npx tsx src/cli.ts demo              # Run demonstration
npx tsx src/cli.ts formula           # Show formula reference
npx tsx src/cli.ts run -t "..." -d ./src --type code_review
npx tsx src/cli.ts experiment        # Run autonomous experiment loop
npx tsx src/cli.ts ablate -t "..." -d ./src  # Run per-file ablation
```

### Build and Production
```bash
npm run build       # tsc → dist/
npm start           # node bin/turbocontext.js
```

### Testing
```bash
npx vitest run                    # Run all 260 tests once
npx vitest                        # Watch mode
npx vitest run tests/compressor   # Run specific test file
```
- 11 test files in `tests/` directory, 4 co-located tests in `src/state/__tests__/`
- No `vitest.config.ts` exists — vitest uses defaults
- Tests import from `../src/core/...` or `../src/state/...`
- All 260 tests pass as of 2026-07-05

## Module Catalog

### Phase 1: Context Compression — `src/core/compressor.ts` (40KB, V3)
- **Purpose:** Scores fragments by semantic similarity (IDF-weighted keyword overlap or embedding cosine), recency, and information density. Applies MMR diversity re-ranking, causal-graph redundancy elimination, and budget-constrained greedy selection.
- **Main export:** `compressContext(task, fragments, config) → Promise<CompressedContext>`
- **Key helpers:** `buildIDFCache`, `mmrReRank`, `estimateTokenCount`, `computeInfoDensity`
- **Depends on:** `types.ts`, `embeddings.ts`, `causal-graph.ts`, `retrieval-system.ts`
- **Config defaults:** α=0.55, β=0.20, γ=0.25, maxTokenBudget=8000, minCoverage=0.80

### Phase 2: Prompt Architecture — `src/core/composer.ts` (16KB, V2)
- **Purpose:** Decomposes tasks into ordered sub-task sequences (understand→execute→verify), building system+user prompts with role, context, format, and quality criteria blocks.
- **Main export:** `composePromptArchitecture(task, context, requirements, trialMutation?, canonicalMutations?) → PromptArchitecture`
- **Depends on:** `types.ts` only
- **Task types supported:** code_review, code_generation, debugging, code_refactor, analysis, design, documentation, testing, general

### Phase 3: Quality-Weighted Generation — `src/core/generator.ts` (29KB, V3)
- **Purpose:** Iterative LLM generation with temperature annealing [0.7, 0.35, 0.1], 4-dimension quality evaluation (completeness, correctness, consistency, format), and feedback injection for retries. V3 integrates hard-signal verifier alongside heuristic quality assessment.
- **Main exports:** `qualityWeightedGeneration` (async generator), `evaluateQuality`, `computeUnifiedMetric`
- **Default config:** qualityThreshold=0.85, maxAttempts=3
- **Depends on:** `types.ts`, `composer.ts` (type-only), `verifier.ts`

### Phase 4: Cost Optimization — `src/core/optimizer.ts` (8.6KB, V4)
- **Purpose:** Complexity estimation → model tier selection (Haiku/Sonnet/Opus or equivalent). LRU semantic cache (100 entries, 5-min TTL). Cost estimation.
- **Main export:** `Optimizer` class with `selectModel()`, `estimateComplexity()`, `lookupCache()`, `writeCache()`, `estimateCost()`
- **Default thresholds:** θ₁=0.30 (fast→medium), θ₂=0.42 (medium→deep)
- **Depends on:** `types.ts` only

### Phase 5: Continuous Learning — `src/core/learner.ts` (66KB, V4)
- **Purpose:** THE god module. Records every execution, manages per-task-type branches, detects plateaus, triggers evolution cycles, delegates to RLFeedbackEngine, manages causal graph construction, and handles memory consolidation.
- **Main export:** `Learner` class, `LearningAdjustment` interface
- **Depends on:** types.ts, fs, path, os, retrieval-system, evolution-engine, rl-feedback-engine, causal-graph, rl-core, intervention-calculus
- **⚠️ WARNING:** 68KB, 10 dependencies. Do NOT refactor without running the full test suite. Changes here touch everything.

### Retrieval System — `src/core/retrieval-system.ts` (40KB, V3)
- **Purpose:** Stateless retrieval computation functions extracted from learner.ts. Plateau detection, strategic directives, contrastive pair finding, adaptive MMR λ, IDF cache management, Thompson source boosts, entropy MMR bonuses, and two-phase causal retrieval.
- **Key exports:** `detectPlateau`, `generateStrategicDirective`, `computeAdaptiveMmrLambda`, `findContrastivePairs`, `twoPhaseCausalRetrieval`, `updateIDFCache`, `thompsonSourceBoost`, `entropyMMRBonus`, `computeCuriosityBonusForRetrieval`
- **Depends on:** types.ts, rl-core.ts

### Causal Graph — `src/core/causal-graph.ts` (68KB, V2)
- **Purpose:** Causal structure learning from ablation data. Implements SGS skeleton discovery, v-structure orientation, Meek rules, d-separation, Markov blanket, FCI (latent confounders), GES (score-based), PC-stable, bootstrap confidence, causal minimality, ensemble discovery.
- **Key exports:** `CausalGraph`, `FaithfulnessAlert`, `buildCausalSkeleton`, `detectFaithfulnessViolations`, `isDSeparated`, `findMarkovBlanket`, `buildPAG_FCI`, `runGES`, `bootstrapEdgeConfidence`, `ensembleCausalDiscovery`
- **Depends on:** types.ts

### RL Core — `src/core/rl-core.ts` (38KB, V3)
- **Purpose:** Pure RL primitives — Thompson Sampling, TD(λ) with eligibility traces, advantage-weighted utility, online predictive model (linear logistic), curriculum learning (4 phases), UCB-guided mutation selection, adversarial verification, memory consolidation, counterfactual synthesis.
- **Key exports:** `thompsonSample`, `decayEligibilityTraces`, `applyTDUpdate`, `predictOutcome`, `updatePredictiveModel`, `computeSurprise`, `getCurriculumPhase`, `adversarialVerify`, `consolidateMemories`, `ucbSelectDimension`, `curiosityBonus`
- **Depends on:** types.ts

### Evolution Engine — `src/core/evolution-engine.ts` (22KB, V3)
- **Purpose:** Autoresearch-style evolution loop: proposeMutation → recordTrial → decideKeepDiscard. Also evolves retrieval strategy hyperparameters via UCB-guided log-normal mutations.
- **Key exports:** `proposeMutation`, `recordTrial`, `decideKeepDiscard`, `getCanonicalMutations`, `getEvolutionStats`, `predictBestMutation`
- **Depends on:** types.ts, rl-core.ts

### RL Feedback Engine — `src/core/rl-feedback-engine.ts` (11KB, V3)
- **Purpose:** Owns all RL state — predictive model, eligibility traces, retrieval strategy, experience library, curriculum. A delegate class that Learner instantiates.
- **Main export:** `RLFeedbackEngine` class
- **Depends on:** types.ts, retrieval-system.ts, evolution-engine.ts, rl-core.ts

### Counterfactual Synthesis — `src/core/counterfactual.ts` (18KB, V3)
- **Purpose:** Template-based counterfactual insight generation WITHOUT an extra LLM call. Task-type-specific templates produce "what-if" insights for success/failure/crash outcomes.
- **Main exports:** `synthesizeCounterfactualInsight`, `synthesizeCounterfactualFromRecord`
- **Depends on:** types.ts (type-only)

### Ablation Engine — `src/core/ablation-engine.ts` (6.2KB, V3)
- **Purpose:** Per-file ablation: execute same task with and without a target file, measure quality delta as the file's direct causal contribution. Cleaner signal than TD(λ) credit assignment.
- **Main exports:** `selectAblationTarget`, `runAblation`, `computeAblationConfidence`
- **Depends on:** types.ts

### Intervention Calculus — `src/core/intervention-calculus.ts` (17KB, V3)
- **Purpose:** Pearl do-calculus formalization: back-door adjustment, front-door mediation, identifiability checking, expected information gain for optimal ablation target selection.
- **Key exports:** `findBackDoorAdjustment`, `findFrontDoorMediator`, `checkIdentifiability`, `estimateCausalEffect`, `selectOptimalAblationTarget`
- **Depends on:** causal-graph.ts, types.ts

### Verifier — `src/core/verifier.ts` (20KB, V3)
- **Purpose:** Hard-signal output verification (Karpathy philosophy: measure real outcomes). Tests code structure (bracket matching, parsing), review output specificity, and general output structure.
- **Main exports:** `CodeVerifier`, `ReviewVerifier`, `StructuralVerifier`, `selectVerifier`, `verifierToRLReward`, `blendedQuality`
- **Depends on:** types.ts, execution-verifier.ts

### Execution Verifier — `src/core/execution-verifier.ts` (8.7KB, V3)
- **Purpose:** Composite verifier wrapping CodeVerifier with TypeScript compilation and smoke-test layer. Signal precedence: smoke test > compilation > structural.
- **Main export:** `ExecutionCodeVerifier`
- **Depends on:** types.ts, verifier.ts, project-compiler.ts

### Project Compiler — `src/core/project-compiler.ts` (20KB, V1)
- **Purpose:** Extract code blocks from LLM output, write to temp dir, run tsc compilation and smoke tests. Project type auto-detection (TypeScript/JavaScript/Python/Go/Rust).
- **Key exports:** `detectProjectType`, `extractAndWriteCodeBlocks`, `compileTypeScript`, `smokeTestTypeScript`, `createTempDir`, `cleanupTempDir`
- **Depends on:** node:child_process, node:fs, node:os, node:path

### LLM Client — `src/core/llm.ts` (11KB, V2)
- **Purpose:** DeepSeek API client (OpenAI-compatible format). Configurable with retries, timeout, and simulated fallback.
- **Main exports:** `createLLMCall`, `defaultLLMCall`
- **Depends on:** nothing (leaf node)

### Embeddings — `src/core/embeddings.ts` (12KB, V3)
- **Purpose:** Pluggable embedding abstraction. OpenAI-compatible provider with LRU cache, or NoOp fallback. Also exports `cosineSimilarity` and `normalizeSimilarity`.
- **Main exports:** `EmbeddingProvider` (interface), `OpenAICompatibleEmbeddingProvider`, `NoOpEmbeddingProvider`, `createSemanticMatcher`, `cosineSimilarity`
- **Depends on:** nothing (leaf node)

### State Management Layer — `src/state/` (V5, June 2026)
- **state-manager.ts** — `SharedStateManager`: sole mutable owner, dirty-flag persistence, snapshot accessors, maintenance gate checks
- **periodic-scheduler.ts** — `PeriodicScheduler`: curriculum-phase-gated scheduling (replaces hardcoded % 5). All intervals (evolution, consolidation, verification, IDF rebuild) adapt to the current curriculum phase.
- **types.ts** — Complete MDP state space: Trial (s,a,r,s'), PolicyState (7 sub-policies), ValueFunctionState (per-task-type baselines), PredictiveModelState (linear logistic), CuriosityState (RND), RetrievalStrategyState (self-evolving), CurriculumState, CrossContextBuffer
- **constants.ts** — All magic numbers, RL hyperparams, file paths, `createFreshState()`, tunable param definitions with dot-notation paths and bounds
- **io.ts** — Atomic state I/O: temp→backup→rename+fsync write, multi-fallback read, JSONL audit logs
- **validation.ts** — Structural + semantic validation on load, lossy v4→v5 migration
- **bridge/** — CrossContextBuffer: async queuing between skill context (Lite) and agent context (Full)
- **curriculum/** — 4-phase curriculum: Broad Exploration → Focused Exploitation → Principled Optimization → Adversarial Refinement
- **evolution/** — Retrieval strategy self-evolution via UCB-guided log-normal mutations
- **memory/** — Consolidation (merge low-utility→summary, archive cold) + cold storage serialization
- **rl/** — Full RL pipeline: TD(λ) credit, prioritized experience replay (PER), Thompson sampling, counterfactual synthesis, HER (hindsight experience replay), RND (random network distillation)
- **trial/** — Trial construction from raw invocation data, RL record enrichment

### Python V5 Reference — `src/turbocontext_v5_rl.py` (91KB, 2000 lines)
- Self-contained (stdlib only), mirrors the TypeScript state/ and core/rl-* modules
- Main class: `RLEngineV5` — sole public API, orchestrates recording, retrieval, evolution, and consolidation
- Maps 1:1 conceptually to `SharedStateManager` + `RLFeedbackEngine` + `Learner` in TypeScript
- Use as reference for algorithm semantics, not as runtime dependency

### Skill Definition — `skill/turbocontext.md` (102KB)
- Registers `/turbocontext` slash command in Claude Code via `.claude/settings.json`
- Two modes: Lite (Context A, skill invocation) and Full (Context B, autonomous agent)
- Full 5-phase pipeline executed per invocation

## Test Map

| Test File | Covers | Lines |
|-----------|--------|-------|
| `compressor.test.ts` | Phase 1: scoring, selection, compression | ~9.9K |
| `composer.test.ts` | Phase 2: prompt architecture decomposition | ~6K |
| `generator.test.ts` | Phase 3: quality evaluation, feedback | ~4.4K |
| `optimizer.test.ts` | Phase 4: model selection, caching, cost | ~6K |
| `learner.test.ts` | Phase 5: branch learning, plateau detect | ~12K |
| `ablation-engine.test.ts` | Ablation: target selection, execution | ~9K |
| `causal-graph.test.ts` | Causal discovery: skeleton, v-structure, FCI, GES | ~8.4K |
| `rl-core.test.ts` | RL primitives: Thompson, TD(λ), curriculum | ~28K |
| `execution-verifier.test.ts` | Execution verification pipeline | ~16K |
| `project-compiler.test.ts` | Compilation + smoke test pipeline | ~11K |
| `experiment-e2e.test.ts` | End-to-end experiments | ~17K |
| `state/__tests__/policy.test.ts` | V5 policy resolution | ~3.5K |
| `state/__tests__/retrieval.test.ts` | V5 7-dim MMR retrieval | ~4K |
| `state/__tests__/thompson.test.ts` | V5 Thompson sampling | ~3K |
| `state/__tests__/rnd.test.ts` | V5 RND curiosity | ~3K |

**Total: 260 tests across 15 files, all passing as of 2026-07-05.**

## Coding Conventions

1. **Imports:** Always use `.js` extension for internal imports (TS source, but Node resolution): `import { Task } from "../types.js"` NOT `"../types"` or `"../types.ts"`
2. **Module system:** ESM only (`"type": "module"` in package.json). `import`/`export`, never `require`.
3. **Type imports:** Use `import type { ... }` for type-only imports. This prevents runtime import cycles.
4. **Type definitions:** ALL shared types go in `src/types.ts`. Module-specific types can stay in their module. DO NOT create new `types.ts` files in subdirectories without strong reason.
5. **Testing:** Use vitest. Test files in `tests/` directory. Import from `../src/core/module.js`. Run `npx vitest run` before committing changes.
6. **Function style:** `function` declarations for top-level, arrow functions for callbacks. No classes unless mutable state is required (counterexample: `Optimizer` uses a class; `retrieval-system.ts` uses only functions).
7. **Naming:** camelCase for variables/functions, PascalCase for classes/interfaces, UPPER_SNAKE_CASE for constants.
8. **No default exports** — use named exports exclusively. Exception: none currently in this codebase.
9. **Error handling:** Modules should throw descriptive errors. Callers (especially `learner.ts`) catch and record them as `crash` outcomes for RL learning.
10. **State management:** `SharedStateManager` is the ONLY mutable owner of `SharedStateV5`. All other subsystems receive snapshots or return patches.

## What Is UNACCEPTABLE

1. **Do NOT** delete or weaken existing tests. 260 tests pass; your changes must not reduce this count.
2. **Do NOT** change import extensions from `.js` to anything else — this breaks Node ESM resolution.
3. **Do NOT** create circular dependencies between `core/` and `state/`. Core does not import from state.
4. **Do NOT** add runtime dependencies to `package.json` without explicit discussion. The project is intentionally lightweight (chalk, commander, zod, cli-progress only).
5. **Do NOT** rename entry points (`src/index.ts`, `src/cli.ts`, `bin/turbocontext.js`) or the skill file.
6. **Do NOT** modify `src/types.ts` without checking all consumers — 32KB of types used by every module. Use `grep -r "YourNewType" src/` after adding.
7. **Do NOT** touch the Python file (`turbocontext_v5_rl.py`) without also updating the TypeScript equivalent, and vice versa. They are kept in semantic sync.
8. **Do NOT** leave uncommitted changes across multiple modules. Each session should produce a complete, testable change.

## Key Design Patterns

### V5 RL Loop (the heart of the learning system)
```
Execution → Trial construction → Record in SharedStateManager
    → RL pipeline: TD(λ) credit assignment, PER, Thompson sampling
    → Predictive model update (quality prediction)
    → Surprise computation (prediction error)
    → Curriculum phase check (every N invocations)
    → Evolution cycle: propose mutation → trial → keep/discard
    → Memory consolidation (when pool > 200 entries)
    → Adversarial verification (verify old successes still hold)
```

### Quality Gate (Phase 3)
```
Generate (temp=t[k]) → Evaluate (4 dims) → Q ≥ 0.85? → Output
                                            ↓ No
                                     Inject feedback → Retry (k+1)
```

### Causal Signal Precedence (V5)
```
Ablation delta (direct causal) > TD(λ) credit (temporal) > Heuristic score
```

## Current State (2026-07-05)

- **All 260 tests passing**, 15 test files
- **V5 state layer** (`src/state/`) is complete with full RL pipeline, including:
  - `RLEngineV5`: primary RL engine — wired into `TurboContextEngine` as of July 2026
  - `SharedStateManager`: dirty-flag persistence to `~/.turbocontext/state-v5.json`, JSONL audit logs
  - `PeriodicScheduler`: curriculum-phase-gated operation scheduling (synced with Python)
- **V5 engine wiring**: `TurboContextEngine` constructs `RLEngineV5`, calls `recordTrial("full")` after each execution, saves to `state-v5.json`. The engine imports from both `core/` and `state/` — it is the designated bridge point between V4 and V5.
- **Python reference** (`turbocontext_v5_rl.py`) is complete (2000 lines), V5.1 with:
  - `PeriodicScheduler`: phase-gated intervals (evolution, consolidation, verification, IDF rebuild)
  - Dirty-flag persistence: `save_state()` no-op when `!dirty`, `save_force()` for guaranteed writes
  - JSONL audit logs: immutable append-only `trials.jsonl`, `evolution.jsonl`, `consolidation.jsonl`
  - Parameter defaults synced with TS `constants.ts`
- **Skill definition** is at 102KB — V5 with full RL integration
- **Versioning:** Core modules span V1-V5. V1: project-compiler. V2: composer, causal-graph, llm. V3: compressor, generator, verifier, embeddings, rl-core, evolution-engine, counterfactual, ablation-engine, intervention-calculus, retrieval-system, rl-feedback-engine, execution-verifier. V4: optimizer, learner. V5: all state/ modules.
- **Roadmap items** from README: Phase 1 (core landing) complete; Phase 2 (deep customization) in progress; Phase 3 (platformization with Web UI) planned; Phase 4 (commercialization) future.
- **Research mission** (`mission.md`): optimize algorithm parameters for maximum quality/cost efficiency across all task types, max 20 experiments (3/20 completed).

## Before You Exit — Self-Checklist

- [ ] `npx vitest run` passes with 260 tests?
- [ ] No `.js`→`.ts` import extension changes?
- [ ] No new circular dependencies between core/ and state/?
- [ ] No new runtime dependencies added to package.json?
- [ ] If `types.ts` was modified, did you grep for all consumers?
- [ ] If Python file was modified, is TypeScript equivalent also updated?
- [ ] Changes are focused on one module/feature, not scattered across 5+ files?
- [ ] V5 state persisted after execution? (`~/.turbocontext/state-v5.json` updated)
