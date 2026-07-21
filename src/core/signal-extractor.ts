// ============================================================================
// TurboContext V6 — Signal Extractor (PACE-inspired)
// ============================================================================
//
// Extracts cheap, fast (<10ms) signals from LLM output and execution context.
// These signals feed into the QualityProxy regression to predict real quality.
//
// PACE mapping:
//   Source instances → Cheap signals extracted from output
//   Target benchmark  → Real code quality (execution verification)
//
// All signals are computable without additional API calls — zero cost.
// ============================================================================

import type { ExecutionMetrics } from "../types.js";

/** A vector of cheap signals extracted from a single execution output */
export interface SignalVector {
  /** Execution-based hard signals (only available for code tasks) */
  compilationSuccess: number;       // 0 or 1 — did tsc compile?
  testPassRate: number;             // 0–1 — fraction of smoke tests passed
  /** Structural signals (always available) */
  codeBlockCount: number;           // number of ``` code blocks
  hasErrorPatterns: number;         // 0 or 1 — inverse: 1 = clean, 0 = errors detected
  keywordCoverage: number;          // 0–1 — fraction of task keywords found in output
  structuralScore: number;          // 0–1 — bracket matching, block closure
  /** Size signals */
  logResponseLength: number;        // log(1 + chars) — normalized response size
  attemptEfficiency: number;        // 1/attemptCount — higher = got it right first try
}

/** Metadata for each signal (for interpretability, following PACE Fig 3) */
export const SIGNAL_META: Record<keyof SignalVector, { label: string; category: string }> = {
  compilationSuccess:  { label: "Compilation",       category: "Execution" },
  testPassRate:         { label: "Test Pass Rate",    category: "Execution" },
  codeBlockCount:       { label: "Code Blocks",       category: "Structure" },
  hasErrorPatterns:     { label: "No Error Patterns", category: "Correctness" },
  keywordCoverage:      { label: "Keyword Coverage",  category: "Completeness" },
  structuralScore:      { label: "Structural Score",  category: "Structure" },
  logResponseLength:    { label: "Response Length",   category: "Size" },
  attemptEfficiency:    { label: "Attempt Efficiency", category: "Efficiency" },
};

/** Default signal vector (used when no execution data is available) */
export const DEFAULT_SIGNALS: SignalVector = {
  compilationSuccess: 0,
  testPassRate: 0,
  codeBlockCount: 0,
  hasErrorPatterns: 1,
  keywordCoverage: 0.5,
  structuralScore: 0.5,
  logResponseLength: 0,
  attemptEfficiency: 1,
};

// ============================================================================
// Signal Extraction
// ============================================================================

/**
 * Extract a SignalVector from LLM output text and optional execution metrics.
 *
 * All signals are O(n) in output length — no API calls, no disk I/O.
 * Target: <10ms for typical outputs (10K chars).
 */
export function extractSignals(
  output: string,
  taskDescription: string,
  executionMetrics?: ExecutionMetrics,
  attemptCount = 1,
): SignalVector {
  // ── Execution-based hard signals ──
  const compilationSuccess = executionMetrics?.compiled === true ? 1 : 0;
  const testPassRate = executionMetrics?.smokeTestPassed === true
    ? 1
    : executionMetrics?.smokeTestPassed === false
      ? 0
      : 0; // unknown = 0

  // ── Structural signals ──
  const codeBlockCount = countCodeBlocks(output);
  const hasErrorPatterns = detectErrorPatterns(output) ? 0 : 1; // invert: 1 = clean
  const keywordCoverage = computeKeywordCoverage(output, taskDescription);
  const structuralScore = computeStructuralScore(output);

  // ── Size signals ──
  const logResponseLength = Math.log(1 + output.length) / Math.log(1 + 10000); // normalize
  const attemptEfficiency = Math.min(1, 1 / Math.max(1, attemptCount));

  return {
    compilationSuccess,
    testPassRate,
    codeBlockCount,
    hasErrorPatterns,
    keywordCoverage,
    structuralScore,
    logResponseLength,
    attemptEfficiency,
  };
}

/**
 * Extract signals with normalized code block count (capped and log-scaled).
 * Useful for regression where raw counts would dominate.
 */
export function extractNormalizedSignals(
  output: string,
  taskDescription: string,
  executionMetrics?: ExecutionMetrics,
  attemptCount = 1,
): number[] {
  const raw = extractSignals(output, taskDescription, executionMetrics, attemptCount);
  return signalVectorToArray(raw);
}

/** Convert SignalVector to a plain number array (for regression input) */
export function signalVectorToArray(s: SignalVector): number[] {
  return [
    s.compilationSuccess,
    s.testPassRate,
    Math.min(1, s.codeBlockCount / 10),  // normalize: cap at 10 blocks
    s.hasErrorPatterns,
    s.keywordCoverage,
    s.structuralScore,
    s.logResponseLength,
    s.attemptEfficiency,
  ];
}

/** Array length matches SignalVector field count */
export const SIGNAL_DIMENSION = 8;

// ============================================================================
// Private signal computation helpers
// ============================================================================

function countCodeBlocks(text: string): number {
  const matches = text.match(/```/g);
  return matches ? Math.floor(matches.length / 2) : 0;
}

function detectErrorPatterns(text: string): boolean {
  const errorSignals = [
    /(i'?m\s+(not|un|in)|i\s+can'?t|i\s+cannot|i\s+do not\s+know)/i,
    /(sorry|apologize|i\s+don'?t\s+know|as\s+an\s+ai)/i,
    /(placeholder|todo|fixme|to\s+do|not\s+implemented)/i,
    /(incomplete|partial|rough\s+draft|unfinished)/i,
    /undefined\s+is\s+not\s+a\s+function/i,
    /cannot\s+read\s+property/i,
    /typeerror|referenceerror|syntaxerror/i,
  ];
  return errorSignals.some(p => p.test(text));
}

function computeKeywordCoverage(output: string, taskDescription: string): number {
  const keywords = extractKeywords(taskDescription);
  if (keywords.length === 0) return 0.5;
  const lowerOutput = output.toLowerCase();
  let matched = 0;
  for (const kw of keywords) {
    const parts = kw.split(/[\s\-_]+/);
    const matchCount = parts.filter(p => lowerOutput.includes(p)).length;
    if (matchCount >= parts.length * 0.5) matched++;
  }
  return matched / keywords.length;
}

function extractKeywords(text: string): string[] {
  return text.toLowerCase()
    .replace(/[^a-z0-9\s\-_]/g, " ")
    .split(/\s+/)
    .filter(w => w.length > 3)
    .filter(w => !["this", "that", "with", "from", "have", "been", "were", "they", "will", "when"].includes(w));
}

function computeStructuralScore(output: string): number {
  let score = 1.0;

  // Check code block closure
  const codeBlockMarkers = (output.match(/```/g) || []).length;
  if (codeBlockMarkers % 2 !== 0) score -= 0.3;

  // Check bracket balance
  const brackets = [
    ["{", "}"], ["(", ")"], ["[", "]"],
  ];
  for (const [open, close] of brackets) {
    const openCount = (output.match(new RegExp("\\" + open, "g")) || []).length;
    const closeCount = (output.match(new RegExp("\\" + close, "g")) || []).length;
    if (Math.abs(openCount - closeCount) > 3) score -= 0.1;
  }

  // Check minimum length
  if (output.length < 50) score -= 0.3;

  return Math.max(0, Math.min(1, score));
}
