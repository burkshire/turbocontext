// ============================================================================
// Turbocontext v6 — Program Loader
// ============================================================================
//
// Reads mission.md (the "program.md" equivalent from Karpathy's autoresearch)
// and returns a ResearchProgram with constraints, budgets, and allowed
// mutations. The human edits mission.md; the engine enforces its constraints.
// ============================================================================

import * as fs from "node:fs";
import * as path from "node:path";

export interface ResearchProgram {
  /** Human-readable goal */
  goal: string;
  /** Token budget per experiment run */
  tokenBudgetPerRun: number;
  /** Time budget per experiment run (seconds) */
  timeBudgetPerRun: number;
  /** Maximum number of experiments to run */
  maxExperiments: number;
  /** Allowed mutation types (empty = all allowed) */
  allowedMutations: string[];
  /** Frozen parameters — mutations targeting these are blocked */
  frozenParams: string[];
}

/** Default program when no mission.md exists. */
const DEFAULT_PROGRAM: ResearchProgram = {
  goal: "Optimize TurboContext algorithm parameters for maximum quality/cost efficiency",
  tokenBudgetPerRun: 8000,
  timeBudgetPerRun: 300,
  maxExperiments: 20,
  allowedMutations: [
    "merge_rounds", "split_round", "remove_round", "reorder_rounds",
    "add_quality_criterion", "remove_quality_criterion",
    "mutate_compression_weights", "mutate_model_tiers",
    "mutate_temperature", "mutate_quality_weights", "mutate_retrieval",
  ],
  frozenParams: ["learningRate", "historyWindow"],
};

/**
 * loadProgram: parses mission.md YAML frontmatter into a ResearchProgram.
 *
 * Looks for mission.md in the working directory or a provided path.
 * Falls back to DEFAULT_PROGRAM if the file doesn't exist or can't be parsed.
 */
export function loadProgram(missionPath?: string): ResearchProgram {
  const p = missionPath || path.join(process.cwd(), "mission.md");

  let raw: string;
  try {
    raw = fs.readFileSync(p, "utf-8");
  } catch {
    console.log(`[turbocontext] No mission.md found at ${p}, using defaults`);
    return { ...DEFAULT_PROGRAM };
  }

  // Parse YAML frontmatter (between the first two --- markers)
  const frontmatterMatch = raw.match(/^---\n([\s\S]*?)\n---/);
  if (!frontmatterMatch) {
    console.log("[turbocontext] No frontmatter in mission.md, using defaults");
    return { ...DEFAULT_PROGRAM };
  }

  const frontmatter = frontmatterMatch[1];

  // Simple YAML-like key: value parser (no heavy dependency needed)
  const parsed: Record<string, string> = {};
  for (const line of frontmatter.split("\n")) {
    const colonIdx = line.indexOf(":");
    if (colonIdx === -1) continue;
    const key = line.slice(0, colonIdx).trim();
    const value = line.slice(colonIdx + 1).trim();
    parsed[key] = value;
  }

  const program: ResearchProgram = {
    goal: parsed["goal"] || DEFAULT_PROGRAM.goal,
    tokenBudgetPerRun: parseInt(parsed["token_budget_per_run"] ?? "", 10) || DEFAULT_PROGRAM.tokenBudgetPerRun,
    timeBudgetPerRun: parseInt(parsed["time_budget_per_run"] ?? "", 10) || DEFAULT_PROGRAM.timeBudgetPerRun,
    maxExperiments: parseInt(parsed["max_experiments"] ?? "", 10) || DEFAULT_PROGRAM.maxExperiments,
    allowedMutations: parsed["allowed_mutations"]
      ? parsed["allowed_mutations"].split(",").map(s => s.trim()).filter(Boolean)
      : DEFAULT_PROGRAM.allowedMutations,
    frozenParams: parsed["frozen_params"]
      ? parsed["frozen_params"].split(",").map(s => s.trim()).filter(Boolean)
      : DEFAULT_PROGRAM.frozenParams,
  };

  console.log(
    `[turbocontext v6] Loaded program: max ${program.maxExperiments} experiments, ` +
    `${program.allowedMutations.length} allowed mutations, ` +
    `${program.frozenParams.length} frozen params`
  );

  return program;
}

/**
 * isMutationAllowed: checks whether a mutation type is permitted by the program.
 */
export function isMutationAllowed(
  mutationType: string,
  program: ResearchProgram,
): boolean {
  if (program.allowedMutations.length === 0) return true; // empty = all allowed
  return program.allowedMutations.includes(mutationType);
}

/**
 * isParamFrozen: checks whether a parameter is frozen (cannot be mutated).
 */
export function isParamFrozen(
  paramName: string,
  program: ResearchProgram,
): boolean {
  return program.frozenParams.includes(paramName);
}
