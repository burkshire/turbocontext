// ============================================================================
// Turbocontext v5 — Atomic State I/O with Backup Rotation
// ============================================================================
//
// WRITE PATH (crash-safe atomic protocol):
//   1. Serialize to JSON, verify size < MAX_STATE_FILE_SIZE_BYTES
//   2. Write to temp file (.tmp), fsync
//   3. Unlink old backup, rename current → backup
//   4. Rename temp → current, fsync directory
//   On crash: at least one of [current, backup, temp] is intact.
//
// READ PATH:
//   1. Try current → validate
//   2. If parse/validation fails, try backup
//   3. If both fail, return null → caller creates fresh state
//
// APPEND-ONLY LOGS: JSONL append with fsync. Immutable audit trail.
import * as fs from "node:fs";
import * as path from "node:path";
import type { SharedStateV5, Trial, EvolutionEntry, ConsolidationEntry, IndexedMemory } from "./types.js";
import { validateState } from "./validation.js";
import {
  STATE_PATH, STATE_BACKUP_PATH, STATE_DIR,
  TRIALS_LOG_PATH, EVOLUTION_LOG_PATH, CONSOLIDATION_LOG_PATH,
  COLD_STORAGE_PATH, MAX_STATE_FILE_SIZE_BYTES,
} from "./constants.js";

// ── Public API ──

/**
 * loadState: attempts to load and validate state from disk.
 *
 * Tries STATE_PATH first. On any failure (ENOENT, corrupt JSON, validation
 * errors), falls back to STATE_BACKUP_PATH. If both fail, returns null —
 * caller should create a fresh state via createFreshState().
 *
 * Design: loads the FULL state (including coldStorage for completeness).
 * Cold memories are kept in-memory — callers should be aware of memory limits.
 */
export function loadState(statePath?: string): SharedStateV5 | null {
  const p = statePath || STATE_PATH;
  ensureStateDir();

  // Try primary
  const primary = tryLoadFile(p);
  if (primary) return primary;

  // Try backup
  const backup = tryLoadFile(STATE_BACKUP_PATH);
  if (backup) {
    // Restore backup to primary location
    try {
      atomicWrite(p, JSON.stringify(backup, null, 2));
    } catch { /* best effort */ }
    return backup;
  }

  // Attempt recovery from temp file
  const tmpPath = p + ".tmp";
  const tmp = tryLoadFile(tmpPath);
  if (tmp) {
    try { atomicWrite(p, JSON.stringify(tmp, null, 2)); } catch { /* best effort */ }
    return tmp;
  }

  return null;
}

/** Save state to disk using the atomic write protocol. */
export function saveState(state: SharedStateV5, statePath?: string): void {
  const p = statePath || STATE_PATH;
  ensureStateDir();

  let json: string;
  try {
    json = JSON.stringify(state, null, 2);
  } catch (err) {
    console.error(`[turbocontext] Failed to serialize state: ${err instanceof Error ? err.message : String(err)}`);
    return;
  }

  // Size guard: force consolidation if state exceeds limit
  if (json.length > MAX_STATE_FILE_SIZE_BYTES) {
    console.error(`[turbocontext] State size ${json.length} exceeds ${MAX_STATE_FILE_SIZE_BYTES} bytes — forcing consolidation`);
    throw new StateSizeError(json.length, MAX_STATE_FILE_SIZE_BYTES);
  }

  try {
    atomicWrite(p, json);
  } catch (err) {
    console.error(`[turbocontext] Failed to save state to ${p}: ${err instanceof Error ? err.message : String(err)}`);
    throw err;
  }
}

/**
 * migrateV4ToV5FromDisk: reads the legacy v4 state.json, maps its fields to
 * v5 state, and writes the result to state-v5.json. Returns the migrated
 * state or null if no v4 state exists or migration fails.
 *
 * V4 → V5 field mapping:
 *   history[]              → trials[]    (ExecutionRecord → Trial)
 *   sourceMemory{}          → memories[]  (object values → IndexedMemory[])
 *   config{}                → policy.*
 *   retrievalStrategy{}     → retrievalStrategy.*
 *   predictiveModel{}       → predictiveModel.*
 */
export function migrateV4ToV5FromDisk(): import("./types.js").SharedStateV5 | null {
  const v4Path = `${STATE_DIR}/state.json`;
  const v5Path = STATE_PATH;

  // Skip if v5 already exists
  if (fs.existsSync(v5Path)) return null;
  // Skip if no v4 state
  if (!fs.existsSync(v4Path)) return null;

  console.log("[turbocontext] Migrating V4 state.json → state-v5.json...");

  let v4Raw: any;
  try {
    const raw = fs.readFileSync(v4Path, "utf-8");
    v4Raw = JSON.parse(raw);
  } catch (err) {
    console.error(`[turbocontext] Failed to read/parse V4 state: ${err instanceof Error ? err.message : String(err)}`);
    return null;
  }

  // Remap V4 keys to what migrateV4ToV5 expects
  const remapped = {
    ...v4Raw,
    trials: v4Raw.history || v4Raw.trials || [],
    memories: v4Raw.sourceMemory
      ? Object.entries(v4Raw.sourceMemory as Record<string, any>).map(([k, v]: [string, any]) => ({
          id: k,
          sourceTrialIds: v.trialIds || [],
          taskType: v.taskType || "code_generation",
          hypothesis: v.hypothesis || "",
          insight: v.insight || "",
          outcome: v.outcome || (v.qualityScore && v.qualityScore > 0.7 ? "success" : "failure"),
          qualityScore: v.qualityScore || 0,
          compressionRatio: v.compressionRatio || 0.5,
          modelTier: v.modelTier || "medium",
          ...v,
        }))
      : (v4Raw.memories || []),
  };

  // Now call the schema-level migrator
  const { migrateV4ToV5 } = require("./validation.js") as typeof import("./validation.js");
  const v5State = migrateV4ToV5(remapped);

  // Write the migrated state
  try {
    saveState(v5State, v5Path);
    console.log(
      `[turbocontext] Migration complete: ${v5State.trials.length} trials, ` +
      `${v5State.memories.length} memories → ${v5Path}`
    );
  } catch (err) {
    console.error(`[turbocontext] Failed to write migrated state: ${err instanceof Error ? err.message : String(err)}`);
  }

  return v5State;
}

/** Append a trial to the JSONL trial log. Immutable audit trail. */
export function appendTrialLog(trial: Trial): void {
  ensureLogDir();
  appendJSONL(TRIALS_LOG_PATH, trial);
}

/** Append an evolution entry to the JSONL log. */
export function appendEvolutionLog(entry: EvolutionEntry): void {
  ensureLogDir();
  appendJSONL(EVOLUTION_LOG_PATH, entry);
}

/** Append a consolidation entry to the JSONL log. */
export function appendConsolidationLog(entry: ConsolidationEntry): void {
  ensureLogDir();
  appendJSONL(CONSOLIDATION_LOG_PATH, entry);
}

/** Load cold-storage memories from the on-disk archive. */
export function loadColdStorage(): IndexedMemory[] {
  try {
    if (!fs.existsSync(COLD_STORAGE_PATH)) return [];
    const raw = fs.readFileSync(COLD_STORAGE_PATH, "utf-8");
    return JSON.parse(raw) as IndexedMemory[];
  } catch {
    return [];
  }
}

/** Save cold-storage memories to the on-disk archive. */
export function saveColdStorage(memories: IndexedMemory[]): void {
  ensureStateDir();
  const dir = path.dirname(COLD_STORAGE_PATH);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(COLD_STORAGE_PATH, JSON.stringify(memories, null, 2), { mode: 0o600 });
  fsyncPath(COLD_STORAGE_PATH);
}

// ── Internal helpers ──

/**
 * atomicWrite: writes data to `targetPath` atomically.
 *
 * Protocol:
 *   1. Write to ${target}.tmp
 *   2. fsync the temp file
 *   3. If backup exists, unlink it
 *   4. Rename target → backup (if target exists)
 *   5. Rename temp → target
 *   6. fsync parent directory
 *
 * On crash at any step, at least one of [target, backup, temp] survives.
 * Next load() tries all three in order.
 */
function atomicWrite(targetPath: string, data: string): void {
  const tmpPath = targetPath + ".tmp";
  const dir = path.dirname(targetPath);

  // Step 1-2: Write temp + fsync
  fs.writeFileSync(tmpPath, data, { mode: 0o600, encoding: "utf-8" });
  fsyncPath(tmpPath);

  // Step 3: Remove old backup
  try { fs.unlinkSync(STATE_BACKUP_PATH); } catch { /* ok if not exists */ }

  // Step 4: Rename current → backup
  try { fs.renameSync(targetPath, STATE_BACKUP_PATH); } catch { /* ok if not exists */ }

  // Step 5: Rename temp → current
  try {
    fs.renameSync(tmpPath, targetPath);
  } catch (err) {
    // If rename fails, temp file still has valid data — not ideal but safe
    throw new StateWriteError(targetPath, err instanceof Error ? err.message : String(err));
  }

  // Step 6: fsync directory to persist metadata
  fsyncDir(dir);
}

/** Reads a file, parses JSON, validates as SharedStateV5. Returns null on any failure. */
function tryLoadFile(filePath: string): SharedStateV5 | null {
  try {
    if (!fs.existsSync(filePath)) return null;
    const raw = fs.readFileSync(filePath, "utf-8");
    const obj = JSON.parse(raw);
    const errors = validateState(obj);
    if (errors.length > 0) {
      console.error(`[turbocontext] State validation failed for ${filePath}: ${errors.length} error(s)`);
      if (errors.length <= 5) errors.forEach(e => console.error(`  ${e.path}: ${e.message}`));
      return null;
    }
    return obj as SharedStateV5;
  } catch (err) {
    console.error(`[turbocontext] Failed to load state from ${filePath}:`, err);
    return null;
  }
}

/** Append a single JSON object as a JSONL line, then fsync. */
function appendJSONL(logPath: string, entry: unknown): void {
  const line = JSON.stringify(entry) + "\n";
  fs.appendFileSync(logPath, line, { mode: 0o600, encoding: "utf-8" });
  fsyncPath(logPath);
}

function ensureStateDir(): void {
  if (!fs.existsSync(STATE_DIR)) {
    fs.mkdirSync(STATE_DIR, { recursive: true, mode: 0o700 });
  }
}

function ensureLogDir(): void {
  const logDir = path.dirname(TRIALS_LOG_PATH);
  if (!fs.existsSync(logDir)) {
    fs.mkdirSync(logDir, { recursive: true, mode: 0o700 });
  }
}

function fsyncPath(filePath: string): void {
  try {
    const fd = fs.openSync(filePath, "r+");
    fs.fsyncSync(fd);
    fs.closeSync(fd);
  } catch { /* best effort: fsync may fail on some FS */ }
}

function fsyncDir(dirPath: string): void {
  try {
    const fd = fs.openSync(dirPath, "r");
    fs.fsyncSync(fd);
    fs.closeSync(fd);
  } catch { /* best effort */ }
}

// ── Error types ──

export class StateWriteError extends Error {
  constructor(
    public readonly path: string,
    message: string,
  ) {
    super(`State write failed [${path}]: ${message}`);
    this.name = "StateWriteError";
  }
}

export class StateSizeError extends Error {
  constructor(
    public readonly actualSize: number,
    public readonly maxSize: number,
  ) {
    super(`State size ${actualSize} exceeds maximum ${maxSize}`);
    this.name = "StateSizeError";
  }
}
