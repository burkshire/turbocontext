// ============================================================================
// Turbocontext v5 — SharedStateManager
// ============================================================================
//
// THE ONLY class that holds a mutable reference to SharedStateV5.
// All other subsystems receive read-only snapshots or operate via
// pure functions that return patches. This prevents accidental
// state corruption from concurrent writes.
//
// Mutations set a `dirty` flag. save() is a no-op when !dirty.
// Call saveForce() for guaranteed persistence.
import type {
  SharedStateV5, Trial, IndexedMemory, PolicyState, PolicyOverrides,
  ValueFunctionState, PredictiveModelState, CuriosityState,
  RetrievalStrategyState, CurriculumState, CrossContextBuffer,
  EvolutionEntry, ConsolidationEntry, CanonicalStrategy,
  CurriculumPhaseConfig, TaskType, StatusReport,
} from "./types.js";
import { CurriculumPhase, MemoryStatus } from "./types.js";
import { loadState, saveState, migrateV4ToV5FromDisk } from "./io.js";
import { createFreshState, STATE_PATH, IDF_REBUILD_INTERVAL } from "./constants.js";
import { resolveEffectivePolicy } from "./policy/policy-manager.js";

export class SharedStateManager {
  private state: SharedStateV5;
  private dirty: boolean;
  private statePath: string;

  /**
   * Constructs a StateManager over an existing state snapshot.
   * Prefer static factories: create() for fresh, load() for persisted.
   */
  constructor(state?: SharedStateV5, statePath?: string) {
    this.state = state || createFreshState();
    this.dirty = false;
    // v5.1: ":memory:" sentinel → in-memory only, no disk I/O
    this.statePath = statePath || STATE_PATH;
  }

  // ── Static factories ──

  /** create: returns a fresh, zero-experience StateManager. */
  static create(statePath?: string): SharedStateManager {
    return new SharedStateManager(createFreshState(), statePath);
  }

  /**
   * load: attempts to load persisted state. Returns null if no state
   * file exists, it is corrupt, or statePath is ":memory:".
   * Caller should call create() instead.
   */
  static load(statePath?: string): SharedStateManager | null {
    // v5.1: ":memory:" sentinel — skip disk I/O entirely
    if (statePath === ":memory:") return null;
    const state = loadState(statePath);
    if (!state) return null;
    return new SharedStateManager(state, statePath);
  }

  /**
   * loadOrMigrate: attempts to load v5 state. If not found, migrates from
   * v4 state.json. If migration also fails, returns null for fresh creation.
   */
  static loadOrMigrate(statePath?: string): SharedStateManager | null {
    // v5.1: ":memory:" sentinel — skip disk I/O entirely
    if (statePath === ":memory:") return null;

    // Try direct v5 load first
    const direct = SharedStateManager.load(statePath);
    if (direct) return direct;

    // Try migration from v4
    const migrated = migrateV4ToV5FromDisk();
    if (migrated) {
      return new SharedStateManager(migrated, statePath);
    }

    return null;
  }

  /** v5.1: Whether this manager has a real disk path (not in-memory). */
  hasPath(): boolean {
    return this.statePath !== ":memory:";
  }

  // ── Accessors (return read-only snapshots) ──

  getSnapshot(): Readonly<SharedStateV5> { return this.state; }
  getTrials(): Readonly<Trial[]> { return this.state.trials; }
  getMemories(): Readonly<IndexedMemory[]> { return this.state.memories; }
  getActiveMemories(): Readonly<IndexedMemory[]> { return this.state.memories.filter(m => m.status === "active"); }
  getColdMemories(): Readonly<IndexedMemory[]> { return this.state.coldStorage; }
  getPolicy(): Readonly<PolicyState> { return this.state.policy; }

  /**
   * getEffectivePolicy: resolves the merged policy for a task type.
   * Base policy values are overridden at the leaf level by perType entries.
   */
  getEffectivePolicy(taskType: TaskType): Readonly<PolicyState> {
    const base = this.state.policy;
    const overrides = base.perType[taskType];
    if (!overrides) return base;
    return resolveEffectivePolicy(base, overrides);
  }

  getValueFunction(): Readonly<ValueFunctionState> { return this.state.valueFunction; }
  getPredictiveModel(): Readonly<PredictiveModelState> { return this.state.predictiveModel; }
  getCuriosity(): Readonly<CuriosityState> { return this.state.curiosity; }
  getRetrievalStrategy(): Readonly<RetrievalStrategyState> { return this.state.retrievalStrategy; }
  getCurriculum(): Readonly<CurriculumState> { return this.state.curriculum; }

  /** getCurriculumPhase: determines the current curriculum phase from totalInvocations. */
  getCurriculumPhase(): number {
    const n = this.state.totalInvocations;
    const [b1, b2, b3] = this.state.curriculum.phaseBoundaries;
    if (n < b1) return CurriculumPhase.BROAD_EXPLORATION;
    if (n < b2) return CurriculumPhase.FOCUSED_EXPLOITATION;
    if (n < b3) return CurriculumPhase.PRINCIPLED_OPTIMIZATION;
    return CurriculumPhase.ADVERRIAL_REFINEMENT;
  }

  getCurriculumConfig(): CurriculumPhaseConfig {
    return this.state.curriculum.phases[this.getCurriculumPhase()];
  }

  getCrossContextBuffer(): Readonly<CrossContextBuffer> { return this.state.crossContextBuffer; }

  // ── Mutators (all set dirty=true) ──

  appendTrial(trial: Trial): void {
    this.state.trials.push(trial);
    this.dirty = true;
  }

  appendMemory(memory: IndexedMemory): void {
    this.state.memories.push(memory);
    this.dirty = true;
  }

  /** updateMemory: applies a partial patch to an existing memory by ID. No-op if not found. */
  updateMemory(memoryId: string, patch: Partial<IndexedMemory>): void {
    const idx = this.findMemoryIndex(memoryId);
    if (idx === -1) return;
    Object.assign(this.state.memories[idx], patch);
    this.dirty = true;
  }

  /** moveToColdStorage: marks memories as "cold", sets coldSince, moves to coldStorage array. */
  moveToColdStorage(memoryIds: string[]): void {
    const ids = new Set(memoryIds);
    const toMove = this.state.memories.filter(m => ids.has(m.id));
    const now = new Date().toISOString();
    for (const m of toMove) {
      m.status = "cold";
      m.coldSince = now;
    }
    this.state.coldStorage.push(...toMove);
    this.state.memories = this.state.memories.filter(m => !ids.has(m.id));
    this.dirty = true;
  }

  markConsolidated(memoryId: string): void {
    const idx = this.findMemoryIndex(memoryId);
    if (idx === -1) return;
    this.state.memories[idx].status = "consolidated";
    this.dirty = true;
  }

  updatePolicy(patch: Partial<PolicyState>): void {
    Object.assign(this.state.policy, patch);
    this.dirty = true;
  }

  updatePolicyPerType(taskType: TaskType, patch: Partial<PolicyOverrides>): void {
    if (!this.state.policy.perType[taskType]) {
      this.state.policy.perType[taskType] = {};
    }
    Object.assign(this.state.policy.perType[taskType]!, patch);
    this.dirty = true;
  }

  updateValueFunction(patch: Partial<ValueFunctionState>): void {
    Object.assign(this.state.valueFunction, patch);
    this.dirty = true;
  }

  updatePredictiveModel(patch: Partial<PredictiveModelState>): void {
    Object.assign(this.state.predictiveModel, patch);
    this.dirty = true;
  }

  updateCuriosity(patch: Partial<CuriosityState>): void {
    Object.assign(this.state.curiosity, patch);
    this.dirty = true;
  }

  updateRetrievalStrategy(patch: Partial<RetrievalStrategyState>): void {
    Object.assign(this.state.retrievalStrategy, patch);
    this.dirty = true;
  }

  appendEvolutionEntry(entry: EvolutionEntry): void {
    this.state.evolutionLog.push(entry);
    this.dirty = true;
  }

  /** v5.1: Store an ablation result for causal graph construction */
  appendAblationResult(entry: import("./types.js").AblationEntry): void {
    this.state.ablationResults.push(entry);
    this.dirty = true;
  }

  /** v5.1: Get all ablation results (read-only) */
  getAblationResults(): Readonly<import("./types.js").AblationEntry[]> {
    return this.state.ablationResults;
  }

  appendConsolidationEntry(entry: ConsolidationEntry): void {
    this.state.consolidationLog.push(entry);
    this.dirty = true;
  }

  updateCrossContextBuffer(patch: Partial<CrossContextBuffer>): void {
    Object.assign(this.state.crossContextBuffer, patch);
    this.dirty = true;
  }

  /** clearPendingTrialsFromSkill: consumes the pending queue, returns consumed trials. */
  clearPendingTrialsFromSkill(): Trial[] {
    const trials = [...this.state.crossContextBuffer.pendingTrialsFromSkill.trials];
    this.state.crossContextBuffer.pendingTrialsFromSkill.trials = [];
    this.state.crossContextBuffer.pendingTrialsFromSkill.count = 0;
    this.dirty = true;
    return trials;
  }

  incrementInvocation(): void {
    this.state.totalInvocations += 1;
    this.bumpLastUpdated();
  }

  bumpLastUpdated(): void {
    this.state.lastUpdated = new Date().toISOString();
    this.dirty = true;
  }

  // ── Persistence ──

  /** save: persists state to disk if dirty. No-op when in-memory or !dirty. */
  save(): void {
    if (!this.dirty || this.statePath === ":memory:") return;
    const prevUpdated = this.state.lastUpdated;
    saveState(this.state, this.statePath);
    this.dirty = false;
    if (this.state.trials.length > 0 && this.state.trials.length % 10 === 0) {
      console.log(
        `[turbocontext v5] State saved: ${this.state.trials.length} trials, ` +
        `${this.state.memories.length} memories, phase ${this.getCurriculumPhase()}`
      );
    }
  }

  /** saveForce: persists state to disk regardless of dirty flag. No-op when in-memory. */
  saveForce(): void {
    if (this.statePath === ":memory:") return;
    saveState(this.state, this.statePath);
    this.dirty = false;
  }

  isDirty(): boolean { return this.dirty; }

  // ── Derived queries (computed from snapshot, no mutation) ──

  getTrialCount(): number { return this.state.trials.length; }
  getActiveMemoryCount(): number { return this.getActiveMemories().length; }
  getColdMemoryCount(): number { return this.state.coldStorage.length; }

  /** getStatusReport: generates a human-readable status summary. */
  getStatusReport(): StatusReport {
    const tasks = Object.keys(this.state.valueFunction.baselines) as TaskType[];
    const perTaskType: StatusReport["perTaskType"] = {} as any;
    for (const tt of tasks) {
      const bl = this.state.valueFunction.baselines[tt];
      const taskTrials = this.state.trials.filter(t => t.taskType === tt);
      perTaskType[tt] = {
        trialCount: taskTrials.length,
        avgQuality: bl.count > 0 ? bl.mean : 0,
        baselineQuality: bl.ema,
        isPlateaued: bl.recentScores.length >= 5 ? Math.abs(bl.slope) < 0.005 : false,
        improvementSlope: bl.slope,
      };
    }
    const recentErrors = this.state.predictiveModel.recentErrors;
    const avgError = recentErrors.length > 0 ? recentErrors.reduce((a, b) => a + b, 0) / recentErrors.length : 0;
    return {
      totalTrials: this.state.totalInvocations,
      activeMemories: this.getActiveMemoryCount(),
      coldMemories: this.getColdMemoryCount(),
      curriculumPhase: this.getCurriculumPhase(),
      perTaskType,
      predictiveModelAccuracy: 1 - Math.min(avgError, 1),
      surpriseMean: this.state.curiosity.surpriseStats.globalMean,
      lastEvolution: this.state.evolutionLog.length > 0 ? this.state.evolutionLog[this.state.evolutionLog.length - 1].timestamp : this.state.createdAt,
      lastConsolidation: this.state.consolidationLog.length > 0 ? this.state.consolidationLog[this.state.consolidationLog.length - 1].timestamp : this.state.createdAt,
      lastCrossSync: this.state.crossContextBuffer.refinedInsights.lastSyncTimestamp,
    };
  }

  /**
   * getContrastiveInsights: finds memories with opposite outcomes for the same
   * task type. Returns natural-language insight strings describing the contrasts.
   */
  getContrastiveInsights(taskType: TaskType): string[] {
    const relevant = this.getActiveMemories().filter(m => m.taskType === taskType);
    const successes = relevant.filter(m => m.outcome === "success");
    const failures = relevant.filter(m => m.outcome === "failure");
    if (successes.length === 0 || failures.length === 0) return [];

    const insights: string[] = [];
    const best = successes.reduce((a, b) => a.qualityScore > b.qualityScore ? a : b);
    const worst = failures.reduce((a, b) => a.qualityScore < b.qualityScore ? a : b);

    insights.push(
      `Best ${taskType}: quality=${best.qualityScore.toFixed(2)}, params α=${best.paramsUsed.alpha.toFixed(2)} β=${best.paramsUsed.beta.toFixed(2)}, model=${best.modelTier}`
    );
    insights.push(
      `Worst ${taskType}: quality=${worst.qualityScore.toFixed(2)}, params α=${worst.paramsUsed.alpha.toFixed(2)} β=${worst.paramsUsed.beta.toFixed(2)}, model=${worst.modelTier}`
    );
    if (best.compressionRatio !== worst.compressionRatio) {
      insights.push(
        `Compression gap: best=${best.compressionRatio.toFixed(2)} vs worst=${worst.compressionRatio.toFixed(2)}`
      );
    }

    return insights;
  }

  getCanonicalStrategies(taskType?: TaskType): CanonicalStrategy[] {
    const strats = this.state.crossContextBuffer.canonicalStrategies;
    if (taskType) return strats.filter(s => s.taskType === taskType);
    return strats;
  }

  // ── Periodic maintenance checks ──

  /** isTimeForEvolution: true when totalInvocations % learningInterval === 0. */
  isTimeForEvolution(): boolean {
    const interval = this.getCurriculumConfig().learningInterval;
    return this.state.totalInvocations > 0 && this.state.totalInvocations % interval === 0;
  }

  /** isTimeForConsolidation: true when totalInvocations % consolidationInterval === 0. */
  isTimeForConsolidation(): boolean {
    const interval = this.getCurriculumConfig().consolidationInterval;
    return this.state.totalInvocations > 0 && this.state.totalInvocations % interval === 0;
  }

  /** isTimeForIDFRebuild: true every IDF_REBUILD_INTERVAL trials. */
  isTimeForIDFRebuild(): boolean {
    return this.state.totalInvocations > 0 && this.state.totalInvocations % IDF_REBUILD_INTERVAL === 0;
  }

  /**
   * shouldRunAdversarialVerification: true when enough memories exist to
   * justify re-evaluation. Threshold: at least 20 active memories.
   */
  shouldRunAdversarialVerification(): boolean {
    return this.getActiveMemoryCount() >= 20;
  }

  // ── Private helpers ──

  private findMemoryIndex(id: string): number {
    return this.state.memories.findIndex(m => m.id === id);
  }
}

// ── Internal utility (deepMergePolicy moved to ./policy/policy-manager.js) ──
