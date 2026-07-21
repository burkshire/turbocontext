// ============================================================
// v3.4 — Hard-Signal Verifier (Karpathy philosophy)
// ============================================================
// "Don't use an LLM to judge LLM output. Measure real outcomes."
//
// The Verifier replaces the regex-based quality assessment in the
// RL reward path. Instead of asking "does this output look correct?"
// it asks "did applying this output produce the expected result?"
//
// Three verifier types, matching autoresearch's three outcomes:
//   PASS  → reward = +1.0 (equivalent to "val_bpb improved")
//   FAIL  → reward = -0.5 (equivalent to "val_bpb unchanged/worse")
//   CRASH → reward = -1.0 (equivalent to "training crashed with NaN")
//
// Architecture:
//   Verifier.verify(output, task, context) → VerificationResult
//   { passed: boolean, crashed: boolean, hardSignal: number, details: string }
// ============================================================

import type { Task } from "../types.js";
// NOTE: ExecutionCodeVerifier is imported lazily below to break the circular
// dependency between verifier.ts ↔ execution-verifier.ts.
// ESM circular imports cause ReferenceError depending on load order;
// dynamic import() inside selectVerifier() defers resolution to call time.

// ------------------------------------------------------------------
// Verification Result — Karpathy's "单数字替代四维度加权"
// ------------------------------------------------------------------

export interface VerificationResult {
  /** Did the output achieve its objective? */
  passed: boolean;
  /** Did the verification itself fail (equivalent to autoresearch crash)? */
  crashed: boolean;
  /**
   * Hard reward signal in [-1, +1].
   *   +1.0 = verified success (code compiles AND passes tests)
   *   +0.5 = structural pass (looks correct, can't fully verify)
   *   -0.3 = structural failure (required elements missing)
   *   -0.5 = verified failure (code doesn't compile / tests fail)
   *   -1.0 = crash (verification harness itself failed)
   */
  hardSignal: number;
  /** Human-readable verification details (for diagnostics, not scoring) */
  details: string;
  /** Additional metrics from verification (e.g., test pass ratio) */
  metrics?: Record<string, number>;
}

// ------------------------------------------------------------------
// Verifier Interface — polymorphism over task types
// ------------------------------------------------------------------

export interface Verifier {
  /** Produces a hard verification signal for an LLM output + task. */
  verify(
    output: string,
    task: Task,
    context?: { workingDir?: string; sourceFiles?: string[] },
  ): Promise<VerificationResult>;
}

// ------------------------------------------------------------------
// 1. Code Verifier — compile + syntax check (no test runner needed)
// ------------------------------------------------------------------

/**
 * Verifies code outputs by checking for structural validity.
 *
 * For full Karpathy-style verification (compile + test), this would need
 * a sandbox environment. The current implementation does structural checks
 * that correlate with compilability: language detection, syntax validity
 * markers, import consistency, and code block completeness.
 *
 * This is NOT regex-based quality assessment. It checks objective,
 * verifiable properties of the code — properties that would cause a
 * compiler or interpreter to fail.
 */
export class CodeVerifier implements Verifier {
  async verify(
    output: string,
    task: Task,
    _context?: { workingDir?: string; sourceFiles?: string[] },
  ): Promise<VerificationResult> {
    const details: string[] = [];
    let penalty = 0;

    // 1. Extract code blocks
    const codeBlocks = extractCodeBlocks(output);
    if (codeBlocks.length === 0 && expectsCode(task)) {
      return {
        passed: false,
        crashed: false,
        hardSignal: -0.5,
        details: "No code blocks found in output that requires code",
        metrics: { codeBlocks: 0 },
      };
    }
    details.push(`${codeBlocks.length} code block(s) found`);

    // 2. For each code block, check objective structural properties
    let totalChecks = 0;
    let passedChecks = 0;

    for (const block of codeBlocks) {
      const lang = block.lang || detectLanguage(block.code);

      // 2a. Brackets/braces/parens balanced?
      const balance = checkBracketBalance(block.code);
      totalChecks++;
      if (balance.balanced) {
        passedChecks++;
      } else {
        penalty += 0.15;
        details.push(`Unbalanced ${balance.detail} in ${lang} block`);
      }

      // 2b. Detect fatal patterns (would not compile/run)
      const fatalPatterns = detectFatalPatterns(block.code, lang);
      totalChecks++;
      if (fatalPatterns.length === 0) {
        passedChecks++;
      } else {
        penalty += 0.2 * fatalPatterns.length;
        details.push(`Fatal patterns in ${lang}: ${fatalPatterns.join(", ")}`);
      }

      // 2c. Imports have matching usage? (heuristic, but objective)
      const importCheck = checkImportUsage(block.code, lang);
      totalChecks++;
      if (importCheck.unusedImports === 0) {
        passedChecks++;
      } else {
        penalty += 0.05 * Math.min(importCheck.unusedImports, 3);
        details.push(
          `${importCheck.unusedImports} potentially unused import(s) in ${lang}`
        );
      }

      // 2d. Function/class definitions have bodies?
      const bodyCheck = checkDefinitionBodies(block.code, lang);
      totalChecks++;
      if (bodyCheck.emptyBodies === 0) {
        passedChecks++;
      } else {
        penalty += 0.25 * bodyCheck.emptyBodies; // empty body = would crash
        details.push(
          `${bodyCheck.emptyBodies} empty definition(s) in ${lang}`
        );
      }
    }

    // 3. Compute hard signal
    const structuralScore = totalChecks > 0
      ? passedChecks / totalChecks
      : 0.5;
    const signal = Math.max(-1.0, Math.min(1.0, structuralScore - penalty));

    const passed = signal >= 0.3; // need at least 70% checks passing

    return {
      passed,
      crashed: false,
      hardSignal: Math.round(signal * 10000) / 10000,
      details: details.join("; ") || "All structural checks passed",
      metrics: { codeBlocks: codeBlocks.length, checksPassed: passedChecks, totalChecks },
    };
  }
}

// ------------------------------------------------------------------
// 2. Review Verifier — did the review find real, actionable issues?
// ------------------------------------------------------------------

/**
 * Verifies code review outputs by checking that findings are specific,
 * actionable, and reference concrete code locations.
 *
 * A "good" review has:
 *   - Specific file:line references
 *   - Concrete severity classifications
 *   - Actionable fix suggestions (not vague advice)
 *   - No false positives (confident claims without evidence)
 *
 * This is NOT regex quality. It's checking the review's internal
 * consistency and specificity — properties that predict whether a
 * human would find the review useful.
 */
export class ReviewVerifier implements Verifier {
  async verify(
    output: string,
    task: Task,
    _context?: { workingDir?: string; sourceFiles?: string[] },
  ): Promise<VerificationResult> {
    const details: string[] = [];
    let score = 0.5; // neutral baseline

    // 1. Does the review identify specific locations? (file:line patterns)
    const locationRefs = output.match(
      /(?:^|\s)([\w./-]+\.\w{1,6})(?::|, line | at line | L)(\d+)/gm
    );
    if (locationRefs && locationRefs.length >= 1) {
      score += 0.15;
      details.push(`${locationRefs.length} specific location reference(s)`);
    } else {
      score -= 0.1;
      details.push("No specific file:line references — review is vague");
    }

    // 2. Does it classify severity? (Critical/High/Medium/Low)
    const severityMarkers = output.match(
      /\b(critical|high|medium|low|severe|minor|major|blocker|info|warning)\b.*?\b(risk|issue|bug|vuln|problem|concern)\b/gim
    );
    if (severityMarkers && severityMarkers.length >= 1) {
      score += 0.1;
      details.push("Issues classified by severity");
    }

    // 3. Does it suggest fixes? (imperative verbs + code patterns)
    const fixSuggestions = output.match(
      /(?:consider|suggest|recommend|should|change|replace|modify|update|add|remove|fix|patch|refactor)\s+(?:\w+\s+){0,5}(?:to|with|by|using|from)/gim
    );
    if (fixSuggestions && fixSuggestions.length >= 2) {
      score += 0.15;
      details.push(`${fixSuggestions.length} actionable fix suggestion(s)`);
    } else if (!fixSuggestions || fixSuggestions.length === 0) {
      score -= 0.1;
      details.push("No actionable fix suggestions");
    }

    // 4. Does it avoid vague hand-waving?
    const vaguePatterns = output.match(
      /(?:code\s+quality\s+could\s+be\s+improved|needs?\s+more\s+work|consider\s+refactoring|generally\s+good|looks?\s+fine|seems?\s+ok)/gim
    );
    if (vaguePatterns && vaguePatterns.length > 2) {
      score -= 0.15;
      details.push("Contains vague/generic assessments without specifics");
    }

    // 5. Internal consistency: does it claim to find N issues but list M?
    const claimMatch = output.match(/(?:found|identified|detected)\s+(\d+)\s+(?:issue|bug|problem|vuln)/gi);
    if (claimMatch) {
      const claimed = parseInt(claimMatch[0].match(/\d+/)![0], 10);
      const actualItems = (output.match(/^[\s]*[-*\d]+[.)]\s+\*\*/gm) || []).length;
      if (Math.abs(claimed - actualItems) > 1 && actualItems > 0) {
        score -= 0.1;
        details.push(`Claims ${claimed} issues but lists ~${actualItems}`);
      }
    }

    const signal = Math.max(-1.0, Math.min(1.0, score));
    return {
      passed: signal >= 0.4,
      crashed: false,
      hardSignal: Math.round(signal * 10000) / 10000,
      details: details.join("; ") || "Review structure is adequate",
      metrics: {
        locationRefs: locationRefs?.length ?? 0,
        fixSuggestions: fixSuggestions?.length ?? 0,
      },
    };
  }
}

// ------------------------------------------------------------------
// 3. Fallback Verifier — structural checks for non-executable tasks
// ------------------------------------------------------------------

/**
 * For tasks where execution verification is impossible (analysis,
 * documentation, design), use minimal structural checks.
 *
 * This is NOT the regex quality assessment. It checks only objective,
 * binary properties: "does the output contain the expected sections?",
 * "is it above minimum length?", "does it address the task at all?"
 */
export class StructuralVerifier implements Verifier {
  async verify(
    output: string,
    task: Task,
    _context?: { workingDir?: string; sourceFiles?: string[] },
  ): Promise<VerificationResult> {
    const details: string[] = [];
    let signal = 0.0;

    // 1. Minimum output length (objective)
    if (output.length < 50) {
      return {
        passed: false,
        crashed: false,
        hardSignal: -0.5,
        details: `Output too short (${output.length} chars) — likely incomplete`,
      };
    }
    signal += 0.2;

    // 2. Contains structured sections? (headings, lists, paragraphs)
    const hasHeadings = /^#{1,4}\s+\S/m.test(output);
    const hasLists = /^[\s]*[-*\d]+[.)]\s+\S/m.test(output);
    const hasParagraphs = /\S.{60,}/m.test(output);

    if (hasHeadings) { signal += 0.15; details.push("Has structured headings"); }
    if (hasLists) { signal += 0.1; details.push("Has lists"); }
    if (hasParagraphs) { signal += 0.1; details.push("Has substantive paragraphs"); }

    if (!hasHeadings && !hasLists && !hasParagraphs) {
      signal -= 0.3;
      details.push("Output lacks any structure");
    }

    // 3. Task keyword presence (is it on-topic at all?)
    const taskWords = task.description
      .toLowerCase()
      .split(/\s+/)
      .filter(w => w.length > 3);
    const outputLower = output.toLowerCase();
    const matchedWords = taskWords.filter(w => outputLower.includes(w));
    const keywordRatio = matchedWords.length / Math.max(taskWords.length, 1);

    if (keywordRatio > 0.5) {
      signal += 0.1;
    } else if (keywordRatio < 0.2) {
      signal -= 0.2;
      details.push(`Low keyword overlap (${(keywordRatio * 100).toFixed(0)}%) — possibly off-topic`);
    }

    const hardSignal = Math.max(-1.0, Math.min(1.0, signal));
    return {
      passed: hardSignal >= 0.3,
      crashed: false,
      hardSignal: Math.round(hardSignal * 10000) / 10000,
      details: details.join("; ") || "Structural checks passed",
      metrics: { keywordRatio: Math.round(keywordRatio * 100) / 100 },
    };
  }
}

// ------------------------------------------------------------------
// Verifier Factory
// ------------------------------------------------------------------

/**
 * Select the appropriate verifier based on task type.
 *
 * Code tasks → CodeVerifier (structural integrity checks)
 * Review/debug tasks → ReviewVerifier (specificity + actionability)
 * Analysis/design/documentation → StructuralVerifier (basic structure)
 */
export async function selectVerifier(task: Task): Promise<Verifier> {
  switch (task.type) {
    case "code_generation":
    case "code_refactor":
    case "testing": {
      // Lazy import to break circular dependency with execution-verifier.ts
      const { ExecutionCodeVerifier } = await import("./execution-verifier.js");
      return new ExecutionCodeVerifier(); // v3.5: structural + compilation
    }
    case "code_review":
    case "debugging":
      return new ReviewVerifier();
    case "analysis":
    case "design":
    case "documentation":
    default:
      return new StructuralVerifier();
  }
}

// ------------------------------------------------------------------
// Helper: Code block extraction
// ------------------------------------------------------------------

interface CodeBlock {
  lang: string;
  code: string;
}

function extractCodeBlocks(output: string): CodeBlock[] {
  const blocks: CodeBlock[] = [];
  const regex = /```(\w*)\n([\s\S]*?)```/g;
  let match;
  while ((match = regex.exec(output)) !== null) {
    blocks.push({ lang: match[1] || "", code: match[2].trim() });
  }
  return blocks;
}

function expectsCode(task: Task): boolean {
  return ["code_generation", "code_refactor", "debugging", "testing"].includes(task.type);
}

// ------------------------------------------------------------------
// Helper: Language detection
// ------------------------------------------------------------------

function detectLanguage(code: string): string {
  if (/\b(function|const|let|var|=>|import.*from|export)\b/.test(code)) return "js/ts";
  if (/\b(def|class|import|from|print|lambda)\b/.test(code)) return "python";
  if (/\b(func|package|import|fmt\.|go\s)/.test(code)) return "go";
  if (/\b(fn|pub|impl|struct|trait|let\s+mut)\b/.test(code)) return "rust";
  if (/\b(public|private|class|void|static|String)\b/.test(code)) return "java";
  return "unknown";
}

// ------------------------------------------------------------------
// Helper: Bracket balance
// ------------------------------------------------------------------

function checkBracketBalance(code: string): { balanced: boolean; detail: string } {
  const pairs: [string, string, string][] = [
    ["{", "}", "braces"],
    ["(", ")", "parens"],
    ["[", "]", "brackets"],
  ];

  for (const [open, close, name] of pairs) {
    let depth = 0;
    let inString: string | null = null;
    for (let i = 0; i < code.length; i++) {
      const ch = code[i];
      // Skip string contents
      if (inString) {
        if (ch === inString && code[i - 1] !== "\\") inString = null;
        continue;
      }
      if (ch === '"' || ch === "'" || ch === "`") {
        inString = ch;
        continue;
      }
      if (ch === open) depth++;
      if (ch === close) depth--;
    }
    if (depth !== 0) {
      return { balanced: false, detail: `${name} (${depth > 0 ? "+" : ""}${depth})` };
    }
  }
  return { balanced: true, detail: "all balanced" };
}

// ------------------------------------------------------------------
// Helper: Fatal patterns (would prevent compilation/execution)
// ------------------------------------------------------------------

function detectFatalPatterns(code: string, lang: string): string[] {
  const fatal: string[] = [];

  // Language-agnostic fatal patterns
  if (/undefined\s+is\s+not\s+a\s+function/i.test(code)) fatal.push("TypeError reference");
  if (/cannot\s+read\s+property/i.test(code)) fatal.push("null reference error");
  if (/is\s+not\s+defined/i.test(code)) fatal.push("undefined reference");
  if (/syntax\s*error/i.test(code)) fatal.push("syntax error message in output");

  // Language-specific
  if (lang === "js/ts" || lang === "typescript" || lang === "javascript") {
    if (/import\s+.*\s+from\s+['"]\.['"]/.test(code)) fatal.push("empty import path");
    if (/const\s+\w+\s*:\s*\w+\s*=\s*;$/.test(code)) fatal.push("uninitialized const");
  }
  if (lang === "python") {
    if (/:\s*$/.test(code.split("\n").find(l => /^\s+/.test(l)) || "")) fatal.push("empty block after colon");
  }

  return fatal;
}

// ------------------------------------------------------------------
// Helper: Import usage check
// ------------------------------------------------------------------

function checkImportUsage(code: string, _lang: string): { unusedImports: number } {
  // Extract imported symbols
  const imported = new Set<string>();
  const importRegex = /import\s+(?:{([^}]+)}|(\w+))\s+from/g;
  let match;
  while ((match = importRegex.exec(code)) !== null) {
    const symbols = (match[1] || match[2] || "").split(",").map(s => s.trim().split(/\s+as\s+/)[0]);
    for (const sym of symbols) {
      if (sym && sym.length > 1) imported.add(sym);
    }
  }

  // Check if each import is used (after the import statements)
  const importEndLine = code.split("\n").findIndex(l => !l.trim().startsWith("import"));
  const body = code.split("\n").slice(Math.max(0, importEndLine)).join("\n");

  let unused = 0;
  for (const sym of imported) {
    if (!body.includes(sym)) unused++;
  }
  return { unusedImports: unused };
}

// ------------------------------------------------------------------
// Helper: Empty definition bodies
// ------------------------------------------------------------------

function checkDefinitionBodies(code: string, _lang: string): { emptyBodies: number } {
  let emptyBodies = 0;

  // Match function/method/class definitions followed by empty body
  const patterns = [
    /(?:function|def|class|interface|fn|pub\s+fn)\s+\w+[^{]*\{\s*\}/g,
    /(?:=>|\)\s*:\s*\w+)\s*\{\s*\}/g,
    /\bpass\s*$/gm, // Python empty body
  ];

  for (const pattern of patterns) {
    const matches = code.match(pattern);
    if (matches) emptyBodies += matches.length;
  }

  return { emptyBodies };
}

// ------------------------------------------------------------------
// Karpathy-style unified signal: one number to rule them all
// ------------------------------------------------------------------

/**
 * Convert a verifier result into the scalar reward used by RL.
 *
 * This is the TurboContext equivalent of autoresearch's:
 *   reward = val_bpb < best_val_bpb ? +1.0 : -0.5
 *
 * The hardSignal is already in [-1, +1]. We pass it through
 * directly as the RL reward, replacing the old regex-based
 * quality score.
 */
export function verifierToRLReward(result: VerificationResult): number {
  return result.hardSignal;
}

/**
 * Combine the old regex quality score with the new hard verifier signal.
 *
 * Transitional: during the migration period, use a weighted blend.
 * Weight shifts toward hard signal as verifier confidence increases.
 */
export function blendedQuality(
  regexQuality: number,       // old evaluateQuality() score (0-1)
  verifierResult: VerificationResult,
  verifierWeight: number = 0.7, // how much to trust verifier vs regex
): { score: number; isHardSignal: boolean } {
  const hardScore = (verifierResult.hardSignal + 1) / 2; // map [-1,1] → [0,1]
  const blended = verifierWeight * hardScore + (1 - verifierWeight) * regexQuality;
  return {
    score: Math.round(blended * 10000) / 10000,
    isHardSignal: verifierResult.passed || verifierResult.crashed,
  };
}
