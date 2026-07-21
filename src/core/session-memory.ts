// ============================================================
// Session Memory — Cross-session context recall for Claude Code
// ============================================================
// Plan B: replaces the 21,000-line RL superstructure with
// a simple IDF-weighted session memory system.
//
// Storage: ~/.turbocontext/sessions.json (JSON array)
//          ~/.turbocontext/sessions.idx (IDF weights cache)
// ============================================================

import type {
  SessionRecord, RecallRequest, RecallResult,
  SimilarSession, FileRecommendation, StrategyRecommendation,
  CorpusStats, TaskType,
} from "../types.js";
import { readFileSync, writeFileSync, existsSync, mkdirSync, renameSync, unlinkSync } from "node:fs";
import { join, dirname } from "node:path";
import { homedir } from "node:os";

// ── Constants ──

const DEFAULT_SESSIONS_DIR = join(homedir(), ".turbocontext");
const DEFAULT_SESSIONS_PATH = join(DEFAULT_SESSIONS_DIR, "sessions.json");
const DEFAULT_INDEX_PATH = join(DEFAULT_SESSIONS_DIR, "sessions.idx");
const MAX_SESSIONS = 1000;
const MAX_AGE_DAYS = 90;
const RECENCY_HALFLIFE_DAYS = 14;
const RECENCY_DECAY = Math.log(2) / RECENCY_HALFLIFE_DAYS; // ≈ 0.05
const TOP_KEYWORDS = 15;
const MIN_KEYWORD_LENGTH = 3;
const MAX_RECOMMENDED_FILES = 10;
const MAX_RECOMMENDED_STRATEGIES = 5;

// English stopwords (reused from compressor.ts pattern)
const STOP_WORDS = new Set([
  "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
  "have", "has", "had", "do", "does", "did", "will", "would", "could",
  "should", "may", "might", "shall", "can", "need", "dare", "ought",
  "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
  "as", "into", "through", "during", "before", "after", "above", "below",
  "between", "under", "again", "further", "then", "once", "here", "there",
  "when", "where", "why", "how", "all", "both", "each", "few", "more",
  "most", "other", "some", "such", "only", "own", "same", "so", "than",
  "too", "very", "just", "not", "and", "but", "or", "if", "while",
  "because", "until", "about", "this", "that", "these", "those",
  "which", "what", "who", "whom", "whose", "it", "its", "they", "them",
  "their", "he", "she", "him", "her", "his", "we", "us", "our", "you", "your",
  "also", "get", "set", "use", "using", "make", "making", "see",
]);

// Related task types for bonus scoring
const RELATED_TASK_TYPES: Record<string, string[]> = {
  code_review: ["code_refactor", "debugging"],
  code_generation: ["code_refactor", "testing"],
  code_refactor: ["code_review", "code_generation"],
  debugging: ["code_review", "testing"],
  testing: ["code_generation", "debugging"],
  analysis: ["documentation", "design"],
  design: ["analysis", "code_generation"],
  documentation: ["analysis", "code_review"],
  general: [],
};

// ── IDF Index ──

interface IDFIndex {
  weights: Record<string, number>;
  documentCount: number;
  lastUpdated: string;
}

// ── SessionMemory class ──

export class SessionMemory {
  private sessions: SessionRecord[] = [];
  private idx: IDFIndex = { weights: {}, documentCount: 0, lastUpdated: "" };
  private sessionsPath: string;
  private indexPath: string;
  private dirty = false;

  constructor(sessionsPath?: string, indexPath?: string) {
    this.sessionsPath = sessionsPath || DEFAULT_SESSIONS_PATH;
    this.indexPath = indexPath || DEFAULT_INDEX_PATH;
  }

  /** Load existing corpus from disk. */
  static load(sessionsPath?: string, indexPath?: string): SessionMemory {
    const sm = new SessionMemory(sessionsPath, indexPath);
    sm.loadFromDisk();
    return sm;
  }

  // ── Public API ──

  /** Find similar past sessions and return actionable recommendations. */
  recall(request: RecallRequest): RecallResult {
    const {
      taskDescription,
      workingDirectory,
      taskType,
      maxResults = 5,
      minSimilarity = 0.10,
    } = request;

    const queryKeywords = extractKeywords(taskDescription);

    // Score all sessions against the query
    const scored: SimilarSession[] = [];
    for (const session of this.sessions) {
      // Scope: only sessions from the same working directory
      if (!session.workingDirectory.startsWith(workingDirectory) &&
          !workingDirectory.startsWith(session.workingDirectory)) {
        continue;
      }

      const ageDays = (Date.now() - new Date(session.timestamp).getTime()) / 86400000;
      const similarity = computeSimilarity(
        queryKeywords, session.keywords, this.idx.weights,
        taskType, session.taskType as TaskType,
        ageDays, session.outcome,
      );

      if (similarity >= minSimilarity) {
        scored.push({
          session,
          similarity: Math.round(similarity * 1000) / 1000,
          matchReason: buildMatchReason(similarity, session, taskType),
        });
      }
    }

    // Sort by similarity descending
    scored.sort((a, b) => b.similarity - a.similarity);
    const topSessions = scored.slice(0, maxResults);

    // Aggregate recommendations
    const { files, strategies } = aggregateRecommendations(topSessions, workingDirectory);

    // Build summary
    const summary = buildSummary(topSessions, files, strategies);

    return {
      similarSessions: topSessions,
      recommendedFiles: files.slice(0, MAX_RECOMMENDED_FILES),
      recommendedStrategies: strategies.slice(0, MAX_RECOMMENDED_STRATEGIES),
      summary,
      corpusStats: this.stats(),
    };
  }

  /** Record a completed session. */
  record(record: Omit<SessionRecord, "id" | "timestamp" | "keywords">): SessionRecord {
    const id = generateId();
    const session: SessionRecord = {
      ...record,
      id,
      timestamp: new Date().toISOString(),
      keywords: extractKeywords(record.taskDescription),
    };

    this.sessions.push(session);
    this.dirty = true;

    // Auto-prune if over limit
    if (this.sessions.length > MAX_SESSIONS) {
      this.prune({ maxSessions: MAX_SESSIONS });
    }

    this.save();
    return session;
  }

  /** Prune old or excess sessions. Returns number removed. */
  prune(options?: { maxAgeDays?: number; maxSessions?: number }): number {
    const maxAge = options?.maxAgeDays ?? MAX_AGE_DAYS;
    const maxSessions = options?.maxSessions ?? MAX_SESSIONS;
    const before = this.sessions.length;

    // Remove by age
    const cutoff = Date.now() - maxAge * 86400000;
    this.sessions = this.sessions.filter(s =>
      new Date(s.timestamp).getTime() > cutoff
    );

    // Remove oldest if still over limit
    if (this.sessions.length > maxSessions) {
      this.sessions.sort((a, b) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      );
      this.sessions = this.sessions.slice(0, maxSessions);
    }

    const removed = before - this.sessions.length;
    if (removed > 0) {
      this.dirty = true;
      this.save();
    }
    return removed;
  }

  /** Get corpus statistics. */
  stats(): CorpusStats {
    const perTaskType: Record<string, number> = {};
    const allFiles = new Set<string>();

    for (const s of this.sessions) {
      perTaskType[s.taskType] = (perTaskType[s.taskType] || 0) + 1;
      for (const f of s.filesRead) allFiles.add(f);
      for (const f of s.filesModified) allFiles.add(f);
    }

    const timestamps = this.sessions.map(s => s.timestamp).sort();
    return {
      totalSessions: this.sessions.length,
      oldestSession: timestamps[0] || "",
      newestSession: timestamps[timestamps.length - 1] || "",
      perTaskType,
      totalUniqueFiles: allFiles.size,
    };
  }

  /** Force save to disk. */
  save(): void {
    if (!this.dirty && existsSync(this.sessionsPath)) return;

    const dir = dirname(this.sessionsPath);
    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });

    // Rebuild IDF index before saving
    this.rebuildIndex();

    // Atomic write: temp → rename
    const tmpSessions = this.sessionsPath + ".tmp";
    const tmpIndex = this.indexPath + ".tmp";

    try {
      writeFileSync(tmpSessions, JSON.stringify(this.sessions, null, 2), "utf-8");
      writeFileSync(tmpIndex, JSON.stringify(this.idx), "utf-8");
      renameSync(tmpSessions, this.sessionsPath);
      renameSync(tmpIndex, this.indexPath);
      this.dirty = false;
    } catch (err) {
      // Clean up temp files on failure
      try { unlinkSync(tmpSessions); } catch { /* best effort */ }
      try { unlinkSync(tmpIndex); } catch { /* best effort */ }
      console.error(`[SessionMemory] Failed to save: ${(err as Error).message}`);
    }
  }

  getSessionCount(): number { return this.sessions.length; }

  // ── Private ──

  private loadFromDisk(): void {
    try {
      if (existsSync(this.sessionsPath)) {
        const raw = readFileSync(this.sessionsPath, "utf-8");
        this.sessions = JSON.parse(raw);
        if (!Array.isArray(this.sessions)) this.sessions = [];
      }
      if (existsSync(this.indexPath)) {
        const raw = readFileSync(this.indexPath, "utf-8");
        this.idx = JSON.parse(raw);
      }
    } catch (err) {
      console.error(`[SessionMemory] Failed to load: ${(err as Error).message}`);
      this.sessions = [];
      this.idx = { weights: {}, documentCount: 0, lastUpdated: "" };
    }
  }

  private rebuildIndex(): void {
    const df: Record<string, number> = {}; // document frequency
    for (const session of this.sessions) {
      const seen = new Set(session.keywords);
      for (const kw of seen) {
        df[kw] = (df[kw] || 0) + 1;
      }
    }

    const N = this.sessions.length;
    const weights: Record<string, number> = {};
    for (const [kw, count] of Object.entries(df)) {
      weights[kw] = Math.log((N + 1) / (count + 1)) + 1; // smoothed IDF
    }

    this.idx = {
      weights,
      documentCount: N,
      lastUpdated: new Date().toISOString(),
    };
  }
}

// ── Keyword Extraction ──

function extractKeywords(text: string): string[] {
  // 1. Split camelCase and snake_case FIRST (before lowercasing)
  // "authMiddleware" → ["auth", "Middleware"]
  const splitTokens = text
    .replace(/[_-]/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .split(/\s+/);

  // 2. Lowercase, remove punctuation, filter
  const cleaned = splitTokens
    .map(t => t.toLowerCase().replace(/[^\w]/g, ""))
    .filter(t => t.length >= MIN_KEYWORD_LENGTH);

  // 3. Remove stopwords, deduplicate, keep meaningful tokens
  const freq: Record<string, number> = {};
  for (const t of cleaned) {
    const lower = t.toLowerCase();
    if (STOP_WORDS.has(lower)) continue;
    if (lower.length < MIN_KEYWORD_LENGTH) continue;
    freq[lower] = (freq[lower] || 0) + 1;
  }

  // 4. Sort by frequency, take top N
  return Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, TOP_KEYWORDS)
    .map(([k]) => k);
}

// ── Similarity Computation ──

function computeSimilarity(
  queryKeywords: string[],
  sessionKeywords: string[],
  idfWeights: Record<string, number>,
  queryType: TaskType | undefined,
  sessionType: TaskType,
  ageDays: number,
  outcome: "success" | "partial" | "failure",
): number {
  // 1. IDF-weighted Jaccard similarity
  const querySet = new Set(queryKeywords);
  const sessionSet = new Set(sessionKeywords);
  let weightedOverlap = 0;
  let weightedUnion = 0;

  for (const kw of new Set([...queryKeywords, ...sessionKeywords])) {
    const w = idfWeights[kw] ?? 1.0;
    const inQuery = querySet.has(kw);
    const inSession = sessionSet.has(kw);
    if (inQuery && inSession) weightedOverlap += w;
    if (inQuery || inSession) weightedUnion += w;
  }
  const keywordScore = weightedUnion > 0 ? weightedOverlap / weightedUnion : 0;

  // 2. Task type bonus
  let taskTypeBonus = 0;
  if (queryType && queryType === sessionType) {
    taskTypeBonus = 0.25;
  } else if (queryType && areRelatedTaskTypes(queryType as string, sessionType as string)) {
    taskTypeBonus = 0.10;
  }

  // 3. Recency decay (exponential, 14-day halflife)
  const recencyFactor = Math.exp(-RECENCY_DECAY * ageDays);

  // 4. Outcome bonus
  const outcomeBonus = outcome === "success" ? 0.05 :
    outcome === "partial" ? 0.02 : 0;

  // 5. Combined score
  return Math.min(1.0, keywordScore * 0.65 + taskTypeBonus + recencyFactor * 0.15 + outcomeBonus);
}

function areRelatedTaskTypes(a: string, b: string): boolean {
  const related = RELATED_TASK_TYPES[a];
  return related ? related.includes(b) : false;
}

function buildMatchReason(
  similarity: number,
  session: SessionRecord,
  queryType: TaskType | undefined,
): string {
  const parts: string[] = [];
  if (queryType && queryType === session.taskType) parts.push("same task type");
  if (similarity > 0.5) parts.push("strong keyword match");
  else if (similarity > 0.2) parts.push("partial keyword match");
  if (session.outcome === "success") parts.push("successful outcome");
  return parts.join(", ") || "keyword overlap";
}

// ── Recommendation Aggregation ──

function aggregateRecommendations(
  similarSessions: SimilarSession[],
  workingDirectory: string,
): { files: FileRecommendation[]; strategies: StrategyRecommendation[] } {
  // File scores: weighted by session similarity × self-assessment
  const fileMap = new Map<string, { score: number; count: number; types: Set<"read" | "modified"> }>();
  const strategyMap = new Map<string, { score: number; count: number }>();

  for (const { session, similarity } of similarSessions) {
    const w = similarity * (0.5 + 0.5 * (session.selfAssessment || 0.5));

    for (const f of session.filesRead) {
      const entry = fileMap.get(f) || { score: 0, count: 0, types: new Set<"read" | "modified">() };
      entry.score += w;
      entry.count++;
      entry.types.add("read");
      fileMap.set(f, entry);
    }
    for (const f of session.filesModified) {
      const entry = fileMap.get(f) || { score: 0, count: 0, types: new Set<"read" | "modified">() };
      entry.score += w * 1.2; // modified files: slight signal boost
      entry.count++;
      entry.types.add("modified");
      fileMap.set(f, entry);
    }

    if (session.strategy) {
      const entry = strategyMap.get(session.strategy) || { score: 0, count: 0 };
      entry.score += w;
      entry.count++;
      strategyMap.set(session.strategy, entry);
    }
  }

  // Scope: only recommend files under the working directory
  const files: FileRecommendation[] = [];
  for (const [path, entry] of fileMap) {
    if (!path.startsWith(workingDirectory)) continue;
    const shortPath = path.replace(workingDirectory + "/", "");
    files.push({
      path: shortPath,
      relevanceScore: Math.round(entry.score * 1000) / 1000,
      sessionCount: entry.count,
      usageType: entry.types.has("modified") && entry.types.has("read") ? "both" :
        entry.types.has("modified") ? "modified" : "read",
    });
  }
  files.sort((a, b) => b.relevanceScore - a.relevanceScore);

  // Strategies: deduplicate and rank
  const strategies: StrategyRecommendation[] = [];
  for (const [strategy, entry] of strategyMap) {
    strategies.push({
      strategy,
      score: Math.round(entry.score * 1000) / 1000,
      occurrenceCount: entry.count,
    });
  }
  strategies.sort((a, b) => b.score - a.score);

  return { files, strategies };
}

function buildSummary(
  sessions: SimilarSession[],
  files: FileRecommendation[],
  strategies: StrategyRecommendation[],
): string {
  if (sessions.length === 0) return "No similar past sessions found. This is a cold start — your session will be recorded for future recall.";

  const parts: string[] = [];
  parts.push(`Found ${sessions.length} similar past session(s).`);

  if (files.length > 0) {
    const topFiles = files.slice(0, 3).map(f => `\`${f.path}\` (${f.sessionCount}×)`).join(", ");
    parts.push(`Top files: ${topFiles}.`);
  }

  if (strategies.length > 0) {
    parts.push(`Top strategy: "${strategies[0].strategy}"`);
  }

  return parts.join(" ");
}

// ── Helpers ──

function generateId(): string {
  const chars = "abcdefghijklmnopqrstuvwxyz0123456789";
  let id = "";
  for (let i = 0; i < 12; i++) {
    id += chars[Math.floor(Math.random() * chars.length)];
  }
  return id;
}
