# TurboContext — AI Agent Onboarding

> **TL;DR:** TurboContext is a cross-session context memory system for Claude Code. It remembers which files and strategies worked in similar past sessions so Claude can start each task with relevant context. Also includes a standalone 5-phase context optimization CLI.

## Project Identity

- **Language:** TypeScript (ES2022 modules, `"type": "module"`)
- **Runtime:** Node.js ≥20
- **Test framework:** Vitest 4.x (`npm test` or `npx vitest run`)
- **Package manager:** npm
- **Entry points:**
  - `bin/turbocontext.js` — standalone CLI (compiled)
  - `npx tsx src/cli.ts` — dev mode
  - `npx tsx src/mcp-server.ts` — MCP server for Claude Code integration
- **State file:** `~/.turbocontext/sessions.json` + `~/.turbocontext/sessions.idx`
- **No RL infrastructure** — Plan B (July 2026) removed 21,000 lines of RL code (state/, rl-core.ts, evolution-engine.ts, retrieval-system.ts, etc.)

## Architecture

```
┌─────────────────────────────────────┐
│     /turbocontext skill (60 lines)   │  Claude Code integration
│     RECALL → EXECUTE → RECORD       │
└──────────────┬──────────────────────┘
               │ stdio (JSON-RPC 2.0)
┌──────────────▼──────────────────────┐
│  MCP Server (src/mcp-server.ts)      │  3 tools: recall/record/status
│  SessionMemory (session-memory.ts)   │  IDF-weighted keyword similarity
└─────────────────────────────────────┘

Standalone CLI (secondary interface):
  compressor → composer → optimizer → generator → record
```

## Project Structure

```
turbocontext/
├── src/
│   ├── index.ts              # TurboContextEngine: 5-phase pipeline (~200 lines)
│   ├── cli.ts                # CLI: run, demo, formula
│   ├── types.ts              # Shared types (~260 lines)
│   ├── mcp-server.ts         # MCP server (stdio transport)
│   └── core/
│       ├── compressor.ts     # Phase 1: context scoring + selection
│       ├── composer.ts       # Phase 2: prompt architecture
│       ├── generator.ts      # Phase 3: quality-gated generation
│       ├── optimizer.ts      # Phase 4: model tier selection + cache
│       ├── session-memory.ts # Cross-session recall/record
│       ├── verifier.ts       # Hard signal output verification
│       ├── execution-verifier.ts
│       ├── project-compiler.ts
│       └── llm.ts            # LLM API client
├── skill/turbocontext.md     # Claude Code skill definition
├── tests/                    # 7 test files, 92 tests
├── FORMULA_V5.archived.md    # Historical: RL formula reference
├── LEARN.md                  # Historical: 41-lesson tutorial
└── mission.md                # Current mission
```

## Commands

```bash
npx tsx src/cli.ts help              # Show commands
npx tsx src/cli.ts demo              # Run demonstration
npx tsx src/cli.ts formula           # Show formula reference
npx tsx src/cli.ts run -t "..." -d ./src --type code_review
npx tsx src/mcp-server.ts            # Start MCP server (for Claude Code)
```

## Testing

```bash
npx vitest run                       # Run all 92 tests
npx vitest                           # Watch mode
npx vitest run tests/session-memory  # Run specific file
```

| Test File | Covers |
|-----------|--------|
| `compressor.test.ts` | Phase 1: scoring, selection, compression |
| `composer.test.ts` | Phase 2: prompt architecture decomposition |
| `generator.test.ts` | Phase 3: quality evaluation, feedback |
| `optimizer.test.ts` | Phase 4: model selection, caching, cost |
| `execution-verifier.test.ts` | Execution verification pipeline |
| `project-compiler.test.ts` | Compilation + smoke test pipeline |
| `session-memory.test.ts` | Recall, record, similarity, pruning |

## Key Design Patterns

### Skill → MCP → SessionMemory

When `/turbocontext` is invoked in Claude Code:
1. `turbocontext_recall` → find similar past sessions → get file/strategy recommendations
2. Claude executes the task using its native reasoning (3-round: understand→execute→verify)
3. `turbocontext_record` → save session for future recall

### Similarity Formula

```
similarity = IDF_keyword_overlap * 0.65
           + task_type_bonus (exact=0.25, related=0.10)
           + recency_factor * 0.15 (14-day halflife)
           + outcome_bonus (success=0.05, partial=0.02)
```

### Quality Gate (Phase 3, standalone CLI)

```
Generate (temp=t[k]) → Evaluate (4 dims) → Q ≥ 0.85? → Output
                                           ↓ No
                                    Inject feedback → Retry (k+1)
```

## Coding Conventions

1. **Imports:** Always use `.js` extension: `import { Task } from "../types.js"`
2. **Module system:** ESM only. `import`/`export`, never `require`.
3. **Named exports only** — no default exports.
4. **Naming:** camelCase for variables/functions, PascalCase for classes/interfaces.
5. **Testing:** vitest. Test files in `tests/`. `npx vitest run` before committing.

## What Is UNACCEPTABLE

1. **Do NOT** add RL infrastructure back. The 21,000-line deletion was intentional.
2. **Do NOT** add new runtime dependencies without discussion.
3. **Do NOT** change import extensions from `.js`.
4. **Do NOT** rename entry points (`src/index.ts`, `src/cli.ts`, `src/mcp-server.ts`).
5. **Do NOT** reintroduce the `StrategyMutation` or quality proxy system.

## Plan B History

In July 2026, the project was cut from ~26,000 lines (52 modules, 378 tests) to
~5,000 lines (11 modules, 92 tests). The deleted systems included:

- RL engine (TD(λ), Thompson sampling, PER, HER, RND)
- Evolution engine (propose→trial→keep/discard loop)
- Retrieval system (MMR re-ranking, plateau detection, IDF cache management)
- Quality proxy (PACE-inspired learned quality prediction)
- Causal graph discovery (SGS, FCI, GES, do-calculus)
- Ablation engine (per-file causal contribution)
- Curriculum learning (4-phase scheduler)
- Cross-branch transfer
- Memory consolidation
- Python reference implementation (turbocontext_v5_rl.py)

The reason: none of these systems ever produced real learning. 3,500 lines were
already deleted earlier (causal graph, do-calculus, ablation engine) for the
same reason. Algorithmic correctness ≠ system works.

The core insight — cross-session context memory — was preserved and implemented
in ~500 lines with IDF-weighted keyword similarity.
