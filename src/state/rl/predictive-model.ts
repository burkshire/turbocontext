// ============================================================================
// Turbocontext v5 — Predictive Model (Linear Logistic Regression)
// ============================================================================
//
// Quality predictor: quality = σ(intercept + Σ w_i · x_i)
// Features: 13 dims (6 task-type one-hot + 5 continuous + 1 binary + 1 time).
// Trained via online SGD (Lite) or mini-batch SGD with PER importance
// weights (Full). Uses Welford's algorithm for online mean/variance tracking.
import type { Trial, PredictiveModelState, FeatureStat } from "../types.js";
import { FEATURE_NAMES, EPSILON, SIGMOID_CLAMP } from "../constants.js";

// ── Feature extraction ──

/**
 * extractFeatures: extracts 13 normalized features from a trial.
 *
 * Features:
 *   1-6: taskType one-hot encoding
 *   7:   log(descriptionLength)
 *   8:   compressionRatio
 *   9:   modelTier_fast (binary)
 *   10:  modelTier_best (binary)
 *   11:  is_retry (binary: attemptCount > 1)
 *   12:  hour_of_day (sin-normalized)
 *   13:  log(tokenBudgetUsed)
 *
 * RL theory: these features constitute the state vector in the MDP.
 * The one-hot task-type encoding allows the linear model to learn
 * per-task-type quality baselines.
 */
export function extractFeatures(trial: Trial): Record<string, number> {
  const hour = new Date(trial.timestamp).getHours();
  return {
    task_code_review: trial.taskType === "code_review" ? 1 : 0,
    task_code_generation: trial.taskType === "code_generation" ? 1 : 0,
    task_debugging: trial.taskType === "debugging" ? 1 : 0,
    task_refactoring: trial.taskType === "refactoring" ? 1 : 0,
    task_documentation: trial.taskType === "documentation" ? 1 : 0,
    task_architecture: trial.taskType === "architecture" ? 1 : 0,
    log_description_length: Math.log(Math.max(trial.descriptionLength, 1)),
    compression_ratio: trial.compressionRatio,
    model_tier_fast: trial.modelTier === "fast" ? 1 : 0,
    model_tier_best: trial.modelTier === "best" ? 1 : 0,
    is_retry: trial.attemptCount > 1 ? 1 : 0,
    hour_of_day_sin: Math.sin((2 * Math.PI * hour) / 24),
    log_token_budget: Math.log(Math.max(trial.tokenBudgetUsed, 1)),
  };
}

// ── Prediction ──

/**
 * predictQuality: computes predicted quality = sigmoid(intercept + Σ w_i · norm(x_i)).
 *
 * Each feature is normalized: (x - μ) / (σ + ε)
 * Uses running stats in featureStats. If σ ≈ 0 (never seen this feature),
 * treats it as zero-mean unit-variance (uniformative).
 *
 * Sigmoid input is clamped to [-10, 10] to prevent overflow.
 */
export function predictQuality(
  model: PredictiveModelState,
  features: Record<string, number>,
): number {
  let z = model.intercept;
  for (const name of FEATURE_NAMES) {
    const val = features[name] ?? 0;
    const stat = model.featureStats[name];
    const nval = stat ? normalize(val, stat) : val;
    z += (model.weights[name] ?? 0) * nval;
  }
  return sigmoid(clamp(z, SIGMOID_CLAMP[0], SIGMOID_CLAMP[1]));
}

// ── SGD update ──

/**
 * sgdUpdate: performs a single SGD step on the predictive model.
 *
 * Algorithm (Lite mode):
 *   pred = sigmoid(z)
 *   error = pred - actual
 *   For each weight: w_i -= lr * error * norm_x_i * sigmoid'(pred)
 *   intercept -= lr * error * sigmoid'(pred)
 *
 * Also updates featureStats online via Welford's algorithm and
 * tracks recent prediction errors for accuracy monitoring.
 *
 * Returns a NEW model state (immutable update pattern).
 */
export function sgdUpdate(
  model: PredictiveModelState,
  features: Record<string, number>,
  actualQuality: number,
): PredictiveModelState {
  const pred = predictQuality(model, features);
  const error = pred - actualQuality;
  const grad = error * sigmoidDerivative(pred);
  const lr = model.learningRate;

  const newWeights = { ...model.weights };
  const newStats = { ...model.featureStats };

  for (const name of FEATURE_NAMES) {
    const val = features[name] ?? 0;

    // Update feature statistics (Welford)
    newStats[name] = updateFeatureStat(newStats[name] || { mean: 0, std: 1, n: 0, M2: 0 }, val);

    // Update weight
    const normVal = normalize(val, newStats[name]);
    newWeights[name] = (newWeights[name] ?? 0) - lr * grad * normVal;
  }

  const newIntercept = model.intercept - lr * grad;

  // Track recent errors (ring buffer, max 20)
  const recentErrors = [...model.recentErrors, Math.abs(error)];
  if (recentErrors.length > 20) recentErrors.shift();

  return {
    ...model,
    weights: newWeights,
    intercept: newIntercept,
    nUpdates: model.nUpdates + 1,
    featureStats: newStats,
    recentErrors,
    calibrationCurve: updateCalibrationCurve(model.calibrationCurve, pred, actualQuality),
  };
}

// ── Welford's online algorithm ──

/**
 * updateFeatureStat: Welford's online algorithm for mean and variance.
 *
 * RL theory (Welford 1962): maintaining running statistics without storing
 * all observations is essential for online learning. This algorithm is
 * numerically stable and O(1) per update.
 *
 *   n_new = n + 1
 *   delta = x - mean
 *   mean_new = mean + delta / n_new
 *   delta2 = x - mean_new
 *   M2_new = M2 + delta * delta2
 *   std_new = sqrt(M2_new / (n_new - 1))  [for n >= 2]
 */
export function updateFeatureStat(stat: FeatureStat, value: number): FeatureStat {
  const n = stat.n + 1;
  const delta = value - stat.mean;
  const mean = stat.mean + delta / n;
  const delta2 = value - mean;
  const M2 = stat.M2 + delta * delta2;
  const std = n >= 2 ? Math.sqrt(Math.max(0, M2 / (n - 1))) : stat.std;
  return { mean, std, n, M2 };
}

// ── Sigmoid utilities ──

/** sigmoid: 1 / (1 + exp(-x)). Clamped input prevents overflow. */
export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-clamp(x, SIGMOID_CLAMP[0], SIGMOID_CLAMP[1])));
}

/** sigmoidDerivative: σ(x) * (1 - σ(x)). */
export function sigmoidDerivative(x: number): number {
  const s = sigmoid(x);
  return s * (1 - s);
}

// ── Normalization ──

export function normalize(value: number, stat: FeatureStat): number {
  if (stat.n < 2) return value; // not enough data, use raw
  return (value - stat.mean) / (Math.max(stat.std, EPSILON));
}

// ── Calibration ──

/**
 * updateCalibrationCurve: updates the calibration bin for a prediction.
 *
 * 10 equal-width bins over [0, 1]. For each bin, tracks the running
 * means of predicted and actual quality. This allows calibration
 * diagnostics — detecting systematic over/under-prediction.
 */
function updateCalibrationCurve(
  curve: PredictiveModelState["calibrationCurve"],
  predicted: number,
  actual: number,
): PredictiveModelState["calibrationCurve"] {
  const clamped = clamp(predicted, 0, 1);
  const binIdx = Math.min(Math.floor(clamped * 10), 9);
  const bin = { ...curve[binIdx] };
  bin.count += 1;
  bin.predictedMean += (clamped - bin.predictedMean) / bin.count;
  bin.actualMean += (actual - bin.actualMean) / bin.count;
  const newCurve = [...curve];
  newCurve[binIdx] = bin;
  return newCurve;
}

// ── Utility ──

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}
