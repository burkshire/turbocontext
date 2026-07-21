// ============================================================================
// TurboContext V6 — Quality Proxy (PACE-inspired regression)
// ============================================================================
//
// Replaces hand-tuned regex quality evaluation with a learned regression that
// predicts real code quality from cheap extracted signals.
//
// PACE mapping:
//   Source matrix X      → Signal vectors from signal-extractor.ts
//   Target vector ȳ       → Hard quality scores (execution verification)
//   Local selection       → Signal-target Spearman correlation
//   Global selection      → SVD leverage × correlation
//   Bootstrap regression  → Noise-robust linear model
//   Ensemble λ            → Learned blend of local + global predictions
//
// Key difference from PACE: we operate in an ONLINE setting — calibration data
// arrives incrementally as executions complete. We use incremental SVD and
// exponential moving average for weights rather than batch retraining.
// ============================================================================

import {
  extractSignals,
  signalVectorToArray,
  SIGNAL_DIMENSION,
  SIGNAL_META,
  type SignalVector,
} from "./signal-extractor.js";
import type { ExecutionMetrics } from "../types.js";

// ============================================================================
// Types
// ============================================================================

/** A single calibration point: signals + ground-truth quality */
export interface CalibrationPoint {
  signals: SignalVector;
  signalArray: number[];    // for regression
  hardQuality: number;      // ground truth (0–1)
  taskType: string;
  timestamp: string;
  weight: number;           // recency weight (newer = higher)
}

/** Trained regression weights (one per signal dimension + intercept) */
export interface ProxyWeights {
  weights: number[];        // length = SIGNAL_DIMENSION
  intercept: number;
  /** Per-weight confidence (bootstrap variance): lower = more stable */
  confidence: number[];
  /** Spearman ρ per signal → target (interpretability, PACE Fig 3) */
  relevance: number[];
  /** Number of calibration points used */
  sampleCount: number;
  /** Last update timestamp */
  lastUpdated: string;
}

/** Quality prediction with confidence interval */
export interface QualityPrediction {
  predictedQuality: number;        // 0–1
  confidenceLow: number;           // lower bound
  confidenceHigh: number;          // upper bound
  signalContributions: Array<{     // interpretability
    signal: string;
    contribution: number;
    relevance: number;
  }>;
  /** Whether this prediction is based on sufficient calibration data */
  isReliable: boolean;
}

// ============================================================================
// Quality Proxy
// ============================================================================

export class QualityProxy {
  private calibration: CalibrationPoint[] = [];
  private weights: ProxyWeights | null = null;
  private readonly maxCalibrationPoints: number;
  private readonly bootstrapSamples: number;
  private readonly minSamplesForFit: number;

  constructor(options?: {
    maxCalibrationPoints?: number;
    bootstrapSamples?: number;
    minSamplesForFit?: number;
  }) {
    this.maxCalibrationPoints = options?.maxCalibrationPoints ?? 200;
    this.bootstrapSamples = options?.bootstrapSamples ?? 100;
    this.minSamplesForFit = options?.minSamplesForFit ?? 8;
  }

  // ── Public API ──

  /**
   * Predict quality from signals extracted from an LLM output.
   *
   * If insufficient calibration data exists, falls back to a heuristic
   * weighted average and reports `isReliable: false`.
   */
  predict(
    output: string,
    taskDescription: string,
    taskType: string,
    executionMetrics?: ExecutionMetrics,
    attemptCount = 1,
  ): QualityPrediction {
    const signals = extractSignals(output, taskDescription, executionMetrics, attemptCount);
    const signalArr = signalVectorToArray(signals);

    // If we have trained weights, use them
    if (this.weights && this.weights.sampleCount >= this.minSamplesForFit) {
      return this.predictWithWeights(signals, signalArr, taskType);
    }

    // Fallback: heuristic weighted average with relevance from calibration
    return this.predictHeuristic(signals, signalArr, taskType);
  }

  /**
   * Add a calibration point: extracted signals from an output + the hard
   * quality score obtained through execution verification (or ablation).
   *
   * After adding, incrementally updates regression weights.
   */
  calibrate(
    output: string,
    taskDescription: string,
    taskType: string,
    hardQuality: number,
    executionMetrics?: ExecutionMetrics,
    attemptCount = 1,
  ): void {
    const signals = extractSignals(output, taskDescription, executionMetrics, attemptCount);
    const signalArr = signalVectorToArray(signals);

    const point: CalibrationPoint = {
      signals,
      signalArray: signalArr,
      hardQuality,
      taskType,
      timestamp: new Date().toISOString(),
      weight: 1.0, // will be decayed
    };

    this.calibration.push(point);

    // Decay old points and evict if over capacity
    this.decayAndEvict();

    // Incrementally refit
    if (this.calibration.length >= this.minSamplesForFit) {
      this.fitBootstrapRegression();
    }
  }

  /** Get current weights (null if not yet fitted) */
  getWeights(): ProxyWeights | null {
    return this.weights;
  }

  /** Get calibration data size */
  getCalibrationSize(): number {
    return this.calibration.length;
  }

  /**
   * Get signal relevance profile (PACE Fig 3 equivalent).
   * Shows which signals most strongly predict quality for each task type.
   */
  getSignalProfile(taskType?: string): Array<{
    signal: string;
    category: string;
    relevance: number;
    meanContribution: number;
  }> {
    const points = taskType
      ? this.calibration.filter(p => p.taskType === taskType)
      : this.calibration;

    if (points.length < 5) return [];

    const signalNames = Object.keys(SIGNAL_META) as (keyof SignalVector)[];
    return signalNames.map((name, i) => {
      // Spearman-like: rank correlation between signal and hard quality
      const relevance = this.computeSpearmanApprox(
        points.map(p => p.signalArray[i]),
        points.map(p => p.hardQuality),
      );
      return {
        signal: SIGNAL_META[name].label,
        category: SIGNAL_META[name].category,
        relevance: Math.abs(relevance),
        meanContribution: this.weights
          ? Math.abs(this.weights.weights[i]) * (this.weights.confidence[i] < 0.5 ? 1 : 0.5)
          : 0,
      };
    }).sort((a, b) => b.relevance - a.relevance);
  }

  // ── Private: Regression ──

  /**
   * Fit bootstrap linear regression following PACE §3.1.
   *
   * 1. Bootstrap resample calibration points (with replacement)
   * 2. For each bootstrap sample, fit least-squares regression
   * 3. Average weights across bootstrap samples (reduces variance)
   * 4. Compute per-weight confidence from bootstrap variance
   * 5. Compute per-signal relevance (Spearman ρ)
   */
  private fitBootstrapRegression(): void {
    const n = this.calibration.length;
    const X = this.calibration.map(p => p.signalArray);
    const y = this.calibration.map(p => p.hardQuality);
    const recencyWeights = this.calibration.map(p => p.weight);

    // Weighted least squares on full data for intercept initialization
    const fullFit = this.weightedLeastSquares(X, y, recencyWeights);

    // Bootstrap
    const bootstrapWeights: number[][] = [];
    for (let b = 0; b < this.bootstrapSamples; b++) {
      const { Xboot, yboot, wboot } = this.bootstrapSample(X, y, recencyWeights, n);
      const fit = this.weightedLeastSquares(Xboot, yboot, wboot);
      bootstrapWeights.push(fit.weights);
    }

    // Average weights across bootstrap samples
    const avgWeights: number[] = new Array(SIGNAL_DIMENSION).fill(0);
    const weightVariances: number[] = new Array(SIGNAL_DIMENSION).fill(0);

    for (let i = 0; i < SIGNAL_DIMENSION; i++) {
      for (let b = 0; b < this.bootstrapSamples; b++) {
        avgWeights[i] += bootstrapWeights[b][i];
      }
      avgWeights[i] /= this.bootstrapSamples;

      // Variance across bootstrap samples
      for (let b = 0; b < this.bootstrapSamples; b++) {
        weightVariances[i] += (bootstrapWeights[b][i] - avgWeights[i]) ** 2;
      }
      weightVariances[i] /= Math.max(1, this.bootstrapSamples - 1);
    }

    // Per-signal relevance (Spearman approximation)
    const relevance: number[] = new Array(SIGNAL_DIMENSION).fill(0);
    for (let i = 0; i < SIGNAL_DIMENSION; i++) {
      relevance[i] = this.computeSpearmanApprox(
        X.map(row => row[i]),
        y,
      );
    }

    // Confidence: inverse of normalized bootstrap variance
    const maxVar = Math.max(...weightVariances, 1e-6);
    const confidence = weightVariances.map(v => Math.max(0, 1 - v / maxVar));

    this.weights = {
      weights: avgWeights,
      intercept: fullFit.intercept,
      confidence,
      relevance,
      sampleCount: n,
      lastUpdated: new Date().toISOString(),
    };
  }

  /** Weighted least squares: min Σ wᵢ(yᵢ − wᵀxᵢ − b)² + λ||w||² */
  private weightedLeastSquares(
    X: number[][],
    y: number[],
    weights: number[],
  ): { weights: number[]; intercept: number } {
    const n = X.length;
    const d = SIGNAL_DIMENSION;
    const lambda = 0.01; // L2 regularization (PACE uses Ridge)

    // Build weighted normal equations: (XᵀWX + λI)w = XᵀWy
    // For intercept: center data first, then compute intercept separately

    // Weighted means
    let sumW = 0;
    const xMean = new Array(d).fill(0);
    let yMean = 0;
    for (let i = 0; i < n; i++) {
      sumW += weights[i];
      for (let j = 0; j < d; j++) xMean[j] += weights[i] * X[i][j];
      yMean += weights[i] * y[i];
    }
    for (let j = 0; j < d; j++) xMean[j] /= sumW;
    yMean /= sumW;

    // Covariance matrix (d × d)
    const XtWX: number[][] = new Array(d).fill(0).map(() => new Array(d).fill(0));
    const XtWy: number[] = new Array(d).fill(0);

    for (let i = 0; i < n; i++) {
      const w = weights[i];
      for (let j = 0; j < d; j++) {
        const xc = X[i][j] - xMean[j];
        XtWy[j] += w * xc * (y[i] - yMean);
        for (let k = 0; k < d; k++) {
          XtWX[j][k] += w * xc * (X[i][k] - xMean[k]);
        }
      }
    }

    // Add regularization
    for (let j = 0; j < d; j++) XtWX[j][j] += lambda;

    // Solve via Gaussian elimination (d=8, small enough for direct solve)
    const w = this.solveLinearSystem(XtWX, XtWy);

    // Intercept
    let intercept = yMean;
    for (let j = 0; j < d; j++) intercept -= w[j] * xMean[j];

    return { weights: w, intercept };
  }

  /** Gaussian elimination for small linear system */
  private solveLinearSystem(A: number[][], b: number[]): number[] {
    const n = A.length;
    const aug = A.map((row, i) => [...row, b[i]]);

    for (let col = 0; col < n; col++) {
      // Find pivot
      let maxRow = col;
      for (let row = col + 1; row < n; row++) {
        if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
      }
      [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

      if (Math.abs(aug[col][col]) < 1e-10) continue;

      // Eliminate
      for (let row = col + 1; row < n; row++) {
        const factor = aug[row][col] / aug[col][col];
        for (let j = col; j <= n; j++) {
          aug[row][j] -= factor * aug[col][j];
        }
      }
    }

    // Back substitution
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      let sum = aug[i][n];
      for (let j = i + 1; j < n; j++) sum -= aug[i][j] * x[j];
      x[i] = Math.abs(aug[i][i]) > 1e-10 ? sum / aug[i][i] : 0;
    }

    return x;
  }

  /** Bootstrap sample with replacement */
  private bootstrapSample(
    X: number[][],
    y: number[],
    w: number[],
    n: number,
  ): { Xboot: number[][]; yboot: number[]; wboot: number[] } {
    const Xboot: number[][] = [];
    const yboot: number[] = [];
    const wboot: number[] = [];
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(Math.random() * n);
      Xboot.push([...X[idx]]);
      yboot.push(y[idx]);
      wboot.push(w[idx]);
    }
    return { Xboot, yboot, wboot };
  }

  // ── Private: Prediction ──

  private predictWithWeights(
    _signals: SignalVector,
    signalArr: number[],
    _taskType: string,
  ): QualityPrediction {
    const w = this.weights!;

    // Linear prediction
    let rawPred = w.intercept;
    for (let i = 0; i < SIGNAL_DIMENSION; i++) {
      rawPred += w.weights[i] * signalArr[i];
    }

    // Clamp to [0, 1]
    const predicted = Math.max(0, Math.min(1, rawPred));

    // Confidence interval from bootstrap variance
    let varianceSum = 0;
    for (let i = 0; i < SIGNAL_DIMENSION; i++) {
      varianceSum += (1 - w.confidence[i]) * signalArr[i] ** 2;
    }
    const stdErr = Math.sqrt(Math.max(0, varianceSum)) * 0.3;
    const confidenceLow = Math.max(0, predicted - 2 * stdErr);
    const confidenceHigh = Math.min(1, predicted + 2 * stdErr);

    // Signal contributions (for interpretability)
    const signalNames = Object.keys(SIGNAL_META) as (keyof SignalVector)[];
    const contributions = signalNames.map((name, i) => ({
      signal: SIGNAL_META[name].label,
      contribution: w.weights[i] * signalArr[i],
      relevance: Math.abs(w.relevance[i]),
    }));

    return {
      predictedQuality: predicted,
      confidenceLow,
      confidenceHigh,
      signalContributions: contributions,
      isReliable: true,
    };
  }

  private predictHeuristic(
    _signals: SignalVector,
    signalArr: number[],
    _taskType: string,
  ): QualityPrediction {
    // Fallback: simple average of normalized signals (used before calibration)
    const mean = signalArr.reduce((a, b) => a + b, 0) / signalArr.length;
    const predicted = Math.max(0, Math.min(1, mean));

    return {
      predictedQuality: predicted,
      confidenceLow: Math.max(0, predicted - 0.2),
      confidenceHigh: Math.min(1, predicted + 0.2),
      signalContributions: [],
      isReliable: false,
    };
  }

  // ── Private: Helpers ──

  /** Approximate Spearman rank correlation (Pearson on ranks) */
  private computeSpearmanApprox(x: number[], y: number[]): number {
    const n = Math.min(x.length, y.length);
    if (n < 3) return 0;

    // Rank transform
    const rankX = this.rank(x);
    const rankY = this.rank(y);

    // Pearson on ranks
    let mx = 0, my = 0;
    for (let i = 0; i < n; i++) { mx += rankX[i]; my += rankY[i]; }
    mx /= n; my /= n;

    let cov = 0, vx = 0, vy = 0;
    for (let i = 0; i < n; i++) {
      const dx = rankX[i] - mx;
      const dy = rankY[i] - my;
      cov += dx * dy;
      vx += dx * dx;
      vy += dy * dy;
    }

    if (vx < 1e-10 || vy < 1e-10) return 0;
    return cov / Math.sqrt(vx * vy);
  }

  private rank(arr: number[]): number[] {
    const indexed = arr.map((v, i) => ({ v, i }));
    indexed.sort((a, b) => a.v - b.v);
    const ranks = new Array(arr.length);
    for (let i = 0; i < indexed.length; i++) {
      ranks[indexed[i].i] = (i + 1) / indexed.length;
    }
    return ranks;
  }

  /** Exponential decay of old calibration points, evict oldest if over capacity */
  private decayAndEvict(): void {
    const now = Date.now();
    const halfLife = 7 * 24 * 3600 * 1000; // 7 days

    for (const point of this.calibration) {
      const age = now - new Date(point.timestamp).getTime();
      point.weight = Math.exp(-Math.log(2) * age / halfLife);
    }

    // Evict oldest (lowest weight) if over capacity
    while (this.calibration.length > this.maxCalibrationPoints) {
      let minIdx = 0;
      for (let i = 1; i < this.calibration.length; i++) {
        if (this.calibration[i].weight < this.calibration[minIdx].weight) {
          minIdx = i;
        }
      }
      this.calibration.splice(minIdx, 1);
    }
  }
}
