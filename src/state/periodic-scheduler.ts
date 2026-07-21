// ============================================================================
// Turbocontext v5 — PeriodicScheduler
// ============================================================================
//
// Curriculum-phase-gated periodic operation scheduler.
//
// After each trial, the scheduler decides which periodic operations should run.
// Unlike the earlier hardcoded "% 4 / % 10" approach, every interval is gated
// by the current curriculum phase so that:
//   - Phase 0 (broad_exploration): frequent evolution, rare consolidation
//   - Phase 3 (adversarial_refinement): rare evolution, frequent verification
//
// This is the TypeScript counterpart to Python's PeriodicScheduler class,
// with one improvement: verification and IDF rebuild intervals are also
// phase-configurable rather than hardcoded.
//
// Usage:
//   const sched = new PeriodicScheduler(stateManager);
//   const ops = sched.afterTrial();  // → Set<"evolution"|"consolidation"|"verification"|"idf_rebuild">
//   if (ops.has("evolution")) { ... }
import type { SharedStateManager } from "./state-manager.js";

// ── Operation labels ──
export const PeriodicOp = {
  EVOLUTION: "evolution",
  CONSOLIDATION: "consolidation",
  VERIFICATION: "verification",
  IDF_REBUILD: "idf_rebuild",
} as const;
export type PeriodicOp = (typeof PeriodicOp)[keyof typeof PeriodicOp];

/**
 * PeriodicScheduler decides which periodic operations to run after each trial,
 * gated by curriculum phase intervals.
 *
 * Design:
 *   - Evolution interval  = curriculum.learningInterval   (phase-configurable)
 *   - Consolidation interval = curriculum.consolidationInterval (phase-configurable)
 *   - Verification interval = phase-gated: every 8/6/4/3 trials depending on phase
 *   - IDF rebuild interval  = phase-gated: every 50/40/30/25 trials depending on phase
 *
 * All intervals require trialCount > 0 (no op on the first trial).
 */
export class PeriodicScheduler {
  private stateManager: SharedStateManager;

  /** Per-phase verification intervals (decreasing — verify more often as we refine). */
  private static readonly VERIFICATION_INTERVALS: Record<number, number> = {
    0: 8,   // broad_exploration: verify occasionally
    1: 6,   // focused_exploitation
    2: 4,   // principled_optimization: verify more often
    3: 3,   // adversarial_refinement: verify frequently
  };

  /** Per-phase IDF rebuild intervals (decreasing — rebuild more often as memories grow). */
  private static readonly IDF_REBUILD_INTERVALS: Record<number, number> = {
    0: 50,  // broad_exploration: few memories, rebuild rarely
    1: 40,  // focused_exploitation
    2: 30,  // principled_optimization
    3: 25,  // adversarial_refinement: many memories, rebuild often
  };

  constructor(stateManager: SharedStateManager) {
    this.stateManager = stateManager;
  }

  /**
   * afterTrial: determines which periodic operations should run.
   *
   * Call this after each trial is recorded. Returns a Set of operation labels.
   * The caller is responsible for executing the operations in a sensible order.
   *
   * Phase-gating rationale (RL theory):
   *   - Early phases (broad exploration): evolve frequently to explore parameter space,
   *     consolidate rarely (few memories to merge).
   *   - Late phases (adversarial refinement): evolve rarely (policy is stable),
   *     verify and consolidate frequently (many memories, distribution shift risk).
   */
  afterTrial(): Set<PeriodicOp> {
    const ops = new Set<PeriodicOp>();
    const trialCount = this.stateManager.getTrialCount();
    if (trialCount === 0) return ops;

    const phase = this.stateManager.getCurriculumPhase();
    const config = this.stateManager.getCurriculumConfig();

    // Evolution: gated by curriculum learning interval
    if (trialCount % config.learningInterval === 0) {
      ops.add(PeriodicOp.EVOLUTION);
    }

    // Consolidation: gated by curriculum consolidation interval
    if (trialCount % config.consolidationInterval === 0) {
      ops.add(PeriodicOp.CONSOLIDATION);
    }

    // Adversarial verification: gated by phase-specific interval
    const verifyInterval = PeriodicScheduler.VERIFICATION_INTERVALS[phase] ?? 4;
    if (trialCount % verifyInterval === 0) {
      ops.add(PeriodicOp.VERIFICATION);
    }

    // IDF rebuild: gated by phase-specific interval
    const idfInterval = PeriodicScheduler.IDF_REBUILD_INTERVALS[phase] ?? 50;
    if (trialCount % idfInterval === 0) {
      ops.add(PeriodicOp.IDF_REBUILD);
    }

    return ops;
  }

  /**
   * shouldRun: static helper for callers that don't have a SharedStateManager.
   *
   * Given raw trialCount, curriculum phase, and phase config, returns which
   * periodic operations are due. This is the lightweight code path used by
   * the V4 Learner and TurboContextEngine until they fully migrate to V5 state.
   */
  static shouldRun(
    trialCount: number,
    phase: number,
    learningInterval: number,
    consolidationInterval: number,
  ): Set<PeriodicOp> {
    const ops = new Set<PeriodicOp>();
    if (trialCount === 0) return ops;

    if (trialCount % learningInterval === 0) {
      ops.add(PeriodicOp.EVOLUTION);
    }
    if (trialCount % consolidationInterval === 0) {
      ops.add(PeriodicOp.CONSOLIDATION);
    }
    const verifyInterval = PeriodicScheduler.VERIFICATION_INTERVALS[phase] ?? 4;
    if (trialCount % verifyInterval === 0) {
      ops.add(PeriodicOp.VERIFICATION);
    }
    const idfInterval = PeriodicScheduler.IDF_REBUILD_INTERVALS[phase] ?? 50;
    if (trialCount % idfInterval === 0) {
      ops.add(PeriodicOp.IDF_REBUILD);
    }
    return ops;
  }

  /**
   * getNextOps: returns a human-readable summary of when the next periodic ops
   * will fire, useful for debugging and status reports.
   */
  getNextOps(): Record<PeriodicOp, number> {
    const trialCount = this.stateManager.getTrialCount();
    const phase = this.stateManager.getCurriculumPhase();
    const config = this.stateManager.getCurriculumConfig();

    const nextMultiple = (n: number, interval: number): number =>
      interval - (n % interval);

    return {
      [PeriodicOp.EVOLUTION]: nextMultiple(trialCount, config.learningInterval),
      [PeriodicOp.CONSOLIDATION]: nextMultiple(trialCount, config.consolidationInterval),
      [PeriodicOp.VERIFICATION]: nextMultiple(
        trialCount,
        PeriodicScheduler.VERIFICATION_INTERVALS[phase] ?? 4,
      ),
      [PeriodicOp.IDF_REBUILD]: nextMultiple(
        trialCount,
        PeriodicScheduler.IDF_REBUILD_INTERVALS[phase] ?? 50,
      ),
    };
  }
}
