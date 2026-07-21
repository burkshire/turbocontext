// ============================================================================
// Turbocontext v5 — Prioritized Experience Replay Buffer
// ============================================================================
//
// Data structure: Sum-Tree for O(log N) sampling + O(log N) priority updates.
// Leaf nodes = experiences. Internal nodes = sum of children's priorities.
// Root = total priority sum.
//
// RL theory (Schaul et al. 2016): PER improves sample efficiency by
// prioritizing experiences with high TD error — the model learns more
// from surprising outcomes. Importance-sampling weights correct the
// bias introduced by non-uniform sampling.
import { PER_ALPHA, PER_BETA, PER_BETA_INCREMENT, PER_BUFFER_CAPACITY, PER_BATCH_SIZE, EPSILON } from "../constants.js";

interface PERBufferEntry {
  features: Record<string, number>;
  actualQuality: number;
  priority: number;
}

export interface PERBatchSample {
  features: Record<string, number>;
  actualQuality: number;
  importanceWeight: number;
  index: number;
}

/**
 * PrioritizedReplayBuffer: sum-tree-backed experience store.
 *
 * The sum-tree represents priorities as a complete binary tree stored
 * in a flat Float64Array (2 * capacity - 1). Leaves are at indices
 * [capacity-1, 2*capacity-2]. Each internal node at index i has
 * children at 2*i+1 and 2*i+2.
 *
 * Sampling: pick a random value s in [0, totalSum), traverse tree to
 * find the leaf where cumulative sum crosses s.
 */
export class PrioritizedReplayBuffer {
  private capacity: number;
  private alpha: number;
  private beta: number;
  private betaIncrement: number;

  private tree: Float64Array;
  private data: (PERBufferEntry | null)[];
  private size: number;
  private ptr: number; // circular write pointer

  constructor(
    capacity = PER_BUFFER_CAPACITY,
    alpha = PER_ALPHA,
    beta = PER_BETA,
    betaIncrement = PER_BETA_INCREMENT,
  ) {
    this.capacity = capacity;
    this.alpha = alpha;
    this.beta = beta;
    this.betaIncrement = betaIncrement;
    this.tree = new Float64Array(2 * capacity - 1);
    this.data = new Array(capacity).fill(null);
    this.size = 0;
    this.ptr = 0;
  }

  /**
   * add: inserts an experience with priority = |tdError|^α.
   * If buffer is full, overwrites oldest entry (circular).
   * O(log N) for tree update.
   */
  add(features: Record<string, number>, actualQuality: number, tdError: number): void {
    const priority = Math.pow(Math.abs(tdError) + EPSILON, this.alpha);
    const entry: PERBufferEntry = { features, actualQuality, priority };

    const idx = this.ptr;
    this.data[idx] = entry;

    // Update sum-tree
    const treeIdx = this.getSumTreeIndex(idx);
    const oldPriority = this.getTreeValue(treeIdx);
    this.setTreeValue(treeIdx, priority);
    this.propagate(treeIdx, priority - oldPriority);

    this.ptr = (this.ptr + 1) % this.capacity;
    if (this.size < this.capacity) this.size += 1;
  }

  /**
   * sample: draws a batch of experiences weighted by priority.
   *
   * Each sample's probability ∝ priority_i / totalPriority.
   * Importance weight: w_i = (N * P(i))^(-beta) / max_j(w_j)
   * Beta is annealed toward 1.0 over time.
   *
   * O(batchSize * log N).
   * @returns Array of PERBatchSample with features, actualQuality, and importanceWeight.
   */
  sample(batchSize = PER_BATCH_SIZE): PERBatchSample[] {
    const actualSize = Math.min(batchSize, this.size);
    if (actualSize === 0) return [];

    const totalPriority = this.getTreeSum();
    if (totalPriority <= 0) {
      // Uniform fallback: all priorities zero
      return this.uniformSample(actualSize);
    }

    const samples: PERBatchSample[] = [];
    const segment = totalPriority / actualSize;

    for (let i = 0; i < actualSize; i++) {
      const s = Math.random() * segment + i * segment;
      const treeIdx = this.sampleTree(s);
      const dataIdx = treeIdx - (this.capacity - 1);
      const entry = this.data[dataIdx];
      if (!entry) continue;

      // Importance-sampling weight
      const prob = entry.priority / totalPriority;
      const weight = Math.pow(this.size * prob, -this.beta);

      samples.push({
        features: entry.features,
        actualQuality: entry.actualQuality,
        importanceWeight: weight,
        index: dataIdx,
      });
    }

    // Normalize weights by max
    const maxWeight = samples.length > 0 ? Math.max(...samples.map(s => s.importanceWeight)) : 1;
    for (const s of samples) {
      s.importanceWeight = maxWeight > 0 ? s.importanceWeight / maxWeight : 1;
    }

    // Anneal beta
    this.beta = Math.min(this.beta + this.betaIncrement, 1.0);

    return samples;
  }

  /**
   * updatePriorities: refreshes priorities for the sampled batch after learning.
   * O(batchSize * log N).
   */
  updatePriorities(indices: number[], tdErrors: number[]): void {
    for (let i = 0; i < indices.length; i++) {
      const dataIdx = indices[i];
      const entry = this.data[dataIdx];
      if (!entry) continue;

      const newPriority = Math.pow(Math.abs(tdErrors[i]) + EPSILON, this.alpha);
      entry.priority = newPriority;

      const treeIdx = this.getSumTreeIndex(dataIdx);
      const oldVal = this.getTreeValue(treeIdx);
      this.setTreeValue(treeIdx, newPriority);
      this.propagate(treeIdx, newPriority - oldVal);
    }
  }

  // ── Queries ──

  getSize(): number { return this.size; }
  isFull(): boolean { return this.size >= this.capacity; }
  isEmpty(): boolean { return this.size === 0; }

  // ── Sum-tree operations ──

  private getSumTreeIndex(dataIndex: number): number {
    return dataIndex + this.capacity - 1;
  }

  private getTreeSum(): number {
    return this.tree[0];
  }

  private getTreeValue(idx: number): number {
    return this.tree[idx] || 0;
  }

  private setTreeValue(idx: number, value: number): void {
    this.tree[idx] = value;
  }

  /**
   * propagate: walks up the tree from `idx`, adding `change` to each ancestor.
   * O(log N).
   */
  private propagate(idx: number, change: number): void {
    let i = idx;
    while (i > 0) {
      i = Math.floor((i - 1) / 2); // parent index
      this.tree[i] += change;
    }
  }

  /**
   * sampleTree: finds the leaf where cumulative sum crosses `s`.
   * O(log N) — descends from root, branching on left/right child.
   */
  private sampleTree(s: number): number {
    let idx = 0;
    while (idx < this.capacity - 1) {
      const left = 2 * idx + 1;
      const right = 2 * idx + 2;
      const leftVal = this.getTreeValue(left);
      if (s <= leftVal) {
        idx = left;
      } else {
        s -= leftVal;
        idx = right;
      }
    }
    return idx;
  }

  /** Uniform sampling fallback when all priorities are zero. */
  private uniformSample(count: number): PERBatchSample[] {
    const samples: PERBatchSample[] = [];
    const indices = new Set<number>();
    while (indices.size < count && indices.size < this.size) {
      indices.add(Math.floor(Math.random() * this.size));
    }
    for (const dataIdx of indices) {
      const entry = this.data[dataIdx];
      if (!entry) continue;
      samples.push({
        features: entry.features,
        actualQuality: entry.actualQuality,
        importanceWeight: 1.0,
        index: dataIdx,
      });
    }
    return samples;
  }
}
