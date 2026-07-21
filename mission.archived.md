---
goal: Optimize TurboContext algorithm parameters for maximum quality/cost efficiency across all task types. Higher efficiency = better.
token_budget_per_run: 8000
time_budget_per_run: 300
max_experiments: 20
allowed_mutations: merge_rounds, split_round, remove_round, reorder_rounds, add_quality_criterion, remove_quality_criterion
frozen_params: learningRate, historyWindow
---

# TurboContext Research Mission

## What we are optimizing

We want to find the best algorithm configuration that maximizes **efficiency**
(quality_score / dollar_cost) across all task types.

## Rules of the game

1. Each experiment runs with a **fixed token budget** so results are directly comparable.
2. The **single metric** is efficiency (higher = better). Quality alone doesn't matter if cost explodes.
3. Mutations that **simplify** (merge rounds, remove criteria) get a bonus — prefer simplicity.
4. Mutations that **crash** are auto-discarded. Don't try them again.

## Exploration strategy

- Try simplifying the prompt architecture first (fewer rounds = lower cost).
- Then try adding quality criteria to catch gaps in the output.
- If a branch plateaus (no improvement after 5+ experiments), switch task types.
- Keep mutations that give **any** positive delta in efficiency, even small.

## What to look at in the morning

Open `~/.turbocontext/results.tsv` and check:
- How many experiments were kept vs discarded?
- What was the best improvement (% delta)?
- Which task types benefited most from evolution?
- Are there any canonical strategies worth permanently adopting?
