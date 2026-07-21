#!/usr/bin/env python3
"""
turbocontext_v5_rl.py — Production RL Engine for Turbocontext v5.
Self-contained, stdlib-only. No external dependencies.

RL architecture:
  Thompson-sampled memory retrieval → MMR diversification → contrastive mining
  TD(λ) value function over trials with eligibility traces on memories
  Online SGD predictive model (lite: single-step; full: PER mini-batch)
  Curiosity-driven exploration via RND (Random Network Distillation) + IDF cache
  Self-evolution of retrieval hyperparams via UCB-guided mutations
  Periodic consolidation, adversarial verification, cross-context sync
  Hindsight Experience Replay (HER) for failure- and crash-boundary learning
  Curriculum learning across four phases: broad_exploration → adversarial_refinement

Thread-safety: state is loaded/saved atomically via tmp-file + rename.
Backward-compatible: from_dict handles missing keys via dataclass defaults.
"""
import json
import math
import os
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Literal, Optional, Union, get_origin, get_args

# ── Type Aliases ────────────────────────────────────────────────────────────
TaskType = Literal["code_review", "code_generation", "debugging",
                   "refactoring", "documentation", "architecture"]
ModelTier = Literal["fast", "medium", "best"]
Outcome = Literal["success", "failure", "crash"]
Context = Literal["skill", "autonomous"]
RecordMode = Literal["lite", "full"]
MemoryStatus = Literal["active", "cold", "consolidated"]
TASK_TYPES: list[TaskType] = ["code_review", "code_generation", "debugging",
                               "refactoring", "documentation", "architecture"]

# ── Utility Functions ───────────────────────────────────────────────────────
def _sigmoid(x: float) -> float:
    """Sigmoid with overflow guard. Maps R→(0,1)."""
    return 1.0 / (1.0 + math.exp(-max(min(x, 20.0), -20.0)))

def _inv_sigmoid(y: float) -> float:
    """Inverse sigmoid with clip-to-domain guard."""
    yc = max(min(y, 1.0 - 1e-12), 1e-12)
    return math.log(yc / (1.0 - yc))

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _ema(old: float, new: float, alpha: float = 0.1) -> float:
    """Exponential moving average update."""
    return old + alpha * (new - old)

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0.0 else default

def _jaccard(a: list[str], b: list[str]) -> float:
    """Jaccard similarity between two token lists."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)

def _sha256_trunc(text: str, n_hex: int = 16) -> str:
    """Truncated SHA-256 hex digest."""
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:n_hex]

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _days_between(iso1: str, iso2: str) -> float:
    """Days elapsed between two ISO-8601 timestamps."""
    try:
        d1 = datetime.fromisoformat(iso1.replace("Z", "+00:00"))
        d2 = datetime.fromisoformat(iso2.replace("Z", "+00:00"))
        return (d2 - d1).total_seconds() / 86400.0
    except (ValueError, TypeError):
        return 0.0

def _log_normal_sample(mu: float, sigma: float) -> float:
    """Sample from log-normal(μ,σ). Used for parameter mutations."""
    return math.exp(random.gauss(mu, sigma))

def _beta_sample(alpha: float, beta: float) -> float:
    """Thompson sample from Beta(α,β) distribution."""
    a, b = max(alpha, 0.01), max(beta, 0.01)
    return random.betavariate(a, b)

def _composite_reward(outcome: Outcome, quality_score: float) -> float:
    """Map trial outcome→scalar reward for TD learning."""
    if outcome == "success":
        return quality_score
    elif outcome == "failure":
        return quality_score * 0.3
    else:
        return 0.0

def _str_to_tasktype(s: str) -> TaskType:
    """Safe string→TaskType cast with fallback."""
    if s in TASK_TYPES:
        return s  # type: ignore[return-value]
    return "code_generation"

# ── Data Classes ────────────────────────────────────────────────────────────
@dataclass
class CompressionWeights:
    alpha: float = 0.60
    beta: float = 0.50
    gamma: float = 0.50   # synced with TS constants.ts (was 0.45)
    theta1: float = 0.30
    theta2: float = 0.55  # synced with TS constants.ts (was 0.60)

@dataclass
class HERGoal:
    goal: str = ""
    outcome: Outcome = "success"
    reward: float = 0.0
    insight: str = ""

@dataclass
class Trial:
    id: str = ""
    timestamp: str = ""
    context: Context = "skill"
    task_type: TaskType = "code_generation"
    description_hash: str = ""
    description_length: int = 0
    capability_requirements: list[str] = field(default_factory=list)
    compression_ratio: float = 0.5
    compression_weights: CompressionWeights = field(default_factory=CompressionWeights)
    temperature_schedule: list[float] = field(default_factory=list)
    model_tier: ModelTier = "medium"
    retrieval_top_k: int = 5
    token_budget_used: int = 0
    max_attempts: int = 3
    outcome: Outcome = "success"
    quality_scores: list[float] = field(default_factory=list)
    quality_score: float = 0.0
    cost_usd: float = 0.0
    latency_ms: int = 0
    attempt_count: int = 1
    best_attempt_index: int = 0
    predicted_quality: Optional[float] = None
    surprise: float = 0.0
    counterfactuals: list[str] = field(default_factory=list)
    curriculum_phase: int = 0
    retrieved_memory_ids: list[str] = field(default_factory=list)
    referenced_memory_ids: list[str] = field(default_factory=list)
    advantage: Optional[float] = None
    causal_utility: float = 0.0
    her_goals: list[HERGoal] = field(default_factory=list)

@dataclass
class IndexedMemory:
    id: str = ""
    source_trial_ids: list[str] = field(default_factory=list)
    created_at: str = ""
    last_retrieved_at: Optional[str] = None
    retrieval_count: int = 0
    task_type: TaskType = "code_generation"
    capability_requirements: list[str] = field(default_factory=list)
    hypothesis: str = ""
    insight: str = ""
    counterfactuals: list[str] = field(default_factory=list)
    outcome: Outcome = "success"
    quality_score: float = 0.0
    compression_ratio: float = 0.5
    model_tier: ModelTier = "medium"
    params_used: dict[str, Any] = field(default_factory=dict)
    thompson_alpha: float = 1.0
    thompson_beta: float = 1.0
    causal_utility: float = 0.5
    retrieval_utility: float = 0.5
    td_error: float = 0.0
    surprise: float = 0.0
    consolidation_count: int = 1
    status: MemoryStatus = "active"
    cold_since: Optional[str] = None
    expires_at: Optional[str] = None

@dataclass
class ActiveRetrievalConfig:
    mmr_lambda: float = 0.70  # synced with TS constants.ts
    top_k: int = 5
    dim_weights: dict[str, float] = field(default_factory=lambda: {
        "hypothesis_overlap": 0.25, "capability_overlap": 0.20,
        "task_type_match": 0.10, "recency": 0.15, "outcome_bonus": 0.10,
        "info_density": 0.10, "thompson_utility": 0.10,
    })  # synced with TS constants.ts DEFAULT_POLICY.retrieval.dimWeights
    token_budget_tiers: tuple[int, int, int] = (8000, 16000, 32000)

@dataclass
class PolicyState:
    compression: CompressionWeights = field(default_factory=CompressionWeights)
    quality_threshold: float = 0.75
    max_attempts: int = 3
    dim_weights: tuple[float, float, float, float] = (0.25, 0.35, 0.20, 0.20)
    temperature_t0: float = 0.70  # synced: annealing schedule descends (was 0.3)
    temperature_t1: float = 0.35  # synced (was 0.5)
    temperature_t2: float = 0.10  # synced (was 0.7)
    model_low_complexity: float = 1500.0  # synced with TS constants.ts (was 1000.0)
    model_high_complexity: float = 8000.0  # synced with TS constants.ts (was 5000.0)
    retrieval: ActiveRetrievalConfig = field(default_factory=ActiveRetrievalConfig)
    mutation_magnitude: float = 0.15
    ucb_c: float = 2.0
    rnd_weight: float = 0.10  # synced with TS constants.ts (was 1.0)
    recency_decay: float = 0.05
    outcome_bonus: float = 0.15
    info_density_bonus: float = 0.10
    per_type: dict[str, dict[str, float]] = field(default_factory=dict)

@dataclass
class Baseline:
    mean: float = 0.5
    ema: float = 0.5
    count: int = 0
    recent_scores: list[float] = field(default_factory=list)
    slope: float = 0.0

@dataclass
class ValueFunctionState:
    baselines: dict[str, Baseline] = field(default_factory=dict)
    global_baseline: float = 0.5
    traces: dict[str, float] = field(default_factory=dict)
    gamma: float = 0.90
    lambda_: float = 0.70
    alpha: float = 0.10
    total_updates: int = 0
    memory_priorities: dict[str, float] = field(default_factory=dict)
    max_priority: float = 1.0

@dataclass
class PredictiveModelState:
    weights: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    learning_rate: float = 0.01
    n_updates: int = 0
    feature_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    recent_errors: list[float] = field(default_factory=list)

@dataclass
class SurpriseStats:
    global_mean: float = 0.0
    global_std: float = 0.01
    recent_values: list[float] = field(default_factory=list)
    anomaly_threshold: float = 0.02

@dataclass
class RNDState:
    target_projection: list[list[float]] = field(default_factory=list)
    predictor_weights: list[list[float]] = field(default_factory=list)
    predictor_bias: list[float] = field(default_factory=list)
    error_mean: float = 0.0
    error_std: float = 1.0
    initialized: bool = False

@dataclass
class CuriosityState:
    idf_weights: dict[str, float] = field(default_factory=dict)
    idf_doc_count: int = 0
    idf_last_rebuilt: str = ""
    task_type_exploration: dict[str, dict[str, float]] = field(default_factory=dict)
    capability_coverage: dict[str, int] = field(default_factory=dict)
    surprise: SurpriseStats = field(default_factory=SurpriseStats)
    rnd: RNDState = field(default_factory=RNDState)

@dataclass
class PendingMutation:
    target_param: str = ""
    old_value: float = 0.0
    new_value: float = 0.0

@dataclass
class EvolutionEntry:
    timestamp: str = ""
    generation: int = 0
    mutation: dict[str, Any] = field(default_factory=dict)
    fitness_before: float = 0.0
    fitness_after: float = 0.0
    delta: float = 0.0
    decision: Literal["keep", "revert"] = "keep"
    scenario: str = ""

@dataclass
class RetrievalStrategyState:
    active: ActiveRetrievalConfig = field(default_factory=ActiveRetrievalConfig)
    ancestor: Optional[ActiveRetrievalConfig] = None
    ancestor_fitness: float = 0.0
    pending_mutation: Optional[PendingMutation] = None
    trials_in_generation: int = 0
    generation: int = 0
    experience_library: list[dict[str, Any]] = field(default_factory=list)

@dataclass
class PhaseConfig:
    name: str = ""
    learning_interval: int = 5
    mutation_magnitude: float = 0.15
    exploration_rate: float = 0.5
    surprise_weight: float = 1.0
    consolidation_interval: int = 15

@dataclass
class CurriculumState:
    phase_boundaries: tuple[int, int, int] = (10, 30, 60)
    phases: dict[int, PhaseConfig] = field(default_factory=lambda: {
        0: PhaseConfig("broad_exploration", 3, 0.25, 0.8, 1.2, 20),
        1: PhaseConfig("focused_exploitation", 5, 0.15, 0.5, 1.0, 15),
        2: PhaseConfig("principled_optimization", 8, 0.08, 0.2, 0.8, 10),
        3: PhaseConfig("adversarial_refinement", 10, 0.06, 0.4, 1.0, 8),
    })

@dataclass
class ConsolidationEntry:
    timestamp: str = ""
    action: str = ""
    source_memory_ids: list[str] = field(default_factory=list)
    target_memory_id: Optional[str] = None
    tokens_saved: int = 0
    quality_estimate: float = 0.0
    reason: str = ""

@dataclass
class CanonicalStrategy:
    strategy_id: str = ""
    task_type: TaskType = "code_generation"
    pattern: str = ""
    params: dict[str, float] = field(default_factory=dict)
    success_rate: float = 0.0
    trial_count: int = 0
    discovered_by: Context = "autonomous"
    discovered_at: str = ""

@dataclass
class CrossContextBuffer:
    pending_trials: list[Trial] = field(default_factory=list)
    pending_count: int = 0
    oldest_pending: str = ""
    refined_insights: dict[str, Any] = field(default_factory=dict)
    canonical_strategies: list[CanonicalStrategy] = field(default_factory=list)
    last_sync: str = ""

@dataclass
class ContrastivePair:
    success_memory: IndexedMemory = field(default_factory=IndexedMemory)
    failure_memory: IndexedMemory = field(default_factory=IndexedMemory)
    similarity: float = 0.0
    shared_capabilities: list[str] = field(default_factory=list)

# ── Return types ────────────────────────────────────────────────────────────
@dataclass
class QueryResult:
    compression_weights: CompressionWeights = field(default_factory=CompressionWeights)
    temperature_schedule: list[float] = field(default_factory=list)
    model_tier: ModelTier = "medium"
    retrieval_params: dict[str, Any] = field(default_factory=dict)
    token_budget: int = 8000
    max_attempts: int = 3
    quality_threshold: float = 0.75
    retrieved_memories: list[IndexedMemory] = field(default_factory=list)
    contrastive_insights: list[str] = field(default_factory=list)
    curriculum_phase: int = 0
    exploration_bonus: float = 0.0

@dataclass
class RecordResult:
    surprise: float = 0.0
    td_error: float = 0.0
    counterfactuals: list[str] = field(default_factory=list)
    her_goals: list[HERGoal] = field(default_factory=list)
    memories_updated: int = 0
    pending_sync_count: int = 0

@dataclass
class StatusReport:
    total_trials: int = 0
    active_memories: int = 0
    cold_memories: int = 0
    curriculum_phase: int = 0
    per_task_type: dict[str, dict[str, Any]] = field(default_factory=dict)
    predictive_model_accuracy: float = 0.0
    surprise_mean: float = 0.0
    last_evolution: str = ""
    last_consolidation: str = ""
    last_cross_sync: str = ""

# ── Feature Extraction ─────────────────────────────────────────────────────
FEATURE_NAMES: list[str] = [
    "task_code_review", "task_code_generation", "task_debugging",
    "task_refactoring", "task_documentation", "task_architecture",
    "description_length_log", "compression_ratio",
    "alpha", "beta", "gamma", "temperature_mean",
    "model_tier_fast", "model_tier_best", "is_retry",
    "hour_of_day", "top_k", "token_budget_log",
]

def extract_features(trial: Trial) -> dict[str, float]:
    """Convert a Trial into a fixed-dimension feature vector for predictive model.
    Uses one-hot task type encoding + log-scaled numerics + binary flags."""
    feats: dict[str, float] = {}
    for tt in TASK_TYPES:
        feats[f"task_{tt}"] = 1.0 if trial.task_type == tt else 0.0
    feats["description_length_log"] = math.log(max(trial.description_length, 1))
    feats["compression_ratio"] = trial.compression_ratio
    feats["alpha"] = trial.compression_weights.alpha
    feats["beta"] = trial.compression_weights.beta
    feats["gamma"] = trial.compression_weights.gamma
    temps = trial.temperature_schedule
    feats["temperature_mean"] = sum(temps) / max(len(temps), 1) if temps else 0.5
    feats["model_tier_fast"] = 1.0 if trial.model_tier == "fast" else 0.0
    feats["model_tier_best"] = 1.0 if trial.model_tier == "best" else 0.0
    feats["is_retry"] = 1.0 if trial.attempt_count > 1 else 0.0
    try:
        d = datetime.fromisoformat(trial.timestamp.replace("Z", "+00:00"))
        feats["hour_of_day"] = float(d.hour) / 24.0
    except (ValueError, TypeError):
        feats["hour_of_day"] = 0.5
    feats["top_k"] = float(trial.retrieval_top_k)
    feats["token_budget_log"] = math.log(max(trial.token_budget_used, 1))
    return feats

def _normalize_features(feats: dict[str, float], stats: dict[str, dict[str, float]]) -> dict[str, float]:
    """Z-score normalize each feature using stored mean/std. Missing stats → pass through."""
    normed: dict[str, float] = {}
    for k, v in feats.items():
        if k in stats:
            s = stats[k]
            normed[k] = (v - s["mean"]) / max(s["std"], 1e-8)
        else:
            normed[k] = v
    return normed

def _update_feature_stats(stats: dict[str, dict[str, float]], feats: dict[str, float]) -> None:
    """Online (Welford-style) update of per-feature mean and std."""
    for k, v in feats.items():
        if k not in stats:
            stats[k] = {"mean": v, "std": 0.0, "n": 1.0}
        else:
            s = stats[k]
            n = s["n"]
            new_n = n + 1.0
            delta = v - s["mean"]
            s["mean"] += delta / new_n
            if new_n > 1.0:
                s["std"] = math.sqrt(((n - 1.0) * s["std"] ** 2 + delta * (v - s["mean"])) / (new_n - 1.0))
            s["n"] = new_n

# ── Predictive Model ───────────────────────────────────────────────────────
def predict_quality(model: PredictiveModelState, trial: Trial) -> float:
    """Linear model + sigmoid: predict quality∈(0,1) from trial features.
    RL theory: this is the V-function approximator for the expected quality."""
    feats = extract_features(trial)
    normed = _normalize_features(feats, model.feature_stats)
    score = model.intercept + sum(model.weights.get(k, 0.0) * v for k, v in normed.items())
    return _sigmoid(score)

def update_predictive_model(model: PredictiveModelState, trial: Trial, mode: RecordMode,
                            per_buffer: Optional["PrioritizedReplayBuffer"] = None) -> float:
    """Online SGD update of quality predictor. Returns surprise=|pred-actual|.
    Lite mode: single SGD step. Full mode: PER mini-batch with importance weighting."""
    feats = extract_features(trial)
    _update_feature_stats(model.feature_stats, feats)
    normed = _normalize_features(feats, model.feature_stats)

    pred = _sigmoid(model.intercept + sum(model.weights.get(k, 0.0) * v for k, v in normed.items()))
    actual = trial.quality_score
    surprise = abs(pred - actual)
    if not math.isfinite(surprise):
        surprise = 0.5

    lr = model.learning_rate / (1.0 + 0.0001 * model.n_updates)  # decay
    error = pred - actual
    sig_deriv = max(pred * (1.0 - pred), 1e-12)

    if mode == "lite":
        for k, v in normed.items():
            grad = 2.0 * error * v * sig_deriv
            model.weights[k] = model.weights.get(k, 0.0) - lr * grad
        model.intercept -= lr * 2.0 * error * sig_deriv
        model.n_updates += 1
    else:
        if per_buffer is not None:
            priority = max(surprise, 0.01)
            per_buffer.add(normed, actual, priority)
            if per_buffer.size() >= 32:
                batch = per_buffer.sample(32)
                indices: list[int] = []
                td_errors: list[float] = []
                for sample in batch:
                    bf = sample["features"]
                    bp = _sigmoid(model.intercept + sum(model.weights.get(k, 0.0) * v for k, v in bf.items()))
                    ba = sample["actual_quality"]
                    be = bp - ba
                    bd = max(bp * (1.0 - bp), 1e-12)
                    iw = sample["importance_weight"]
                    for k, v in bf.items():
                        model.weights[k] = model.weights.get(k, 0.0) - lr * 2.0 * be * v * bd * iw
                    model.intercept -= lr * 2.0 * be * bd * iw
                    model.n_updates += 1
                    indices.append(sample["index"])
                    td_errors.append(abs(be))
                per_buffer.update_priorities(indices, td_errors)

    model.recent_errors.append(surprise)
    if len(model.recent_errors) > 20:
        model.recent_errors = model.recent_errors[-20:]
    return surprise

# ── Prioritized Replay Buffer (Full mode) ───────────────────────────────────
class PrioritizedReplayBuffer:
    """Fixed-capacity circular buffer with importance-weighted PER sampling.
    RL theory: breaks temporal correlations in online SGD; prioritizes high-surprise transitions."""

    def __init__(self, capacity: int = 500, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer: list[dict[str, Any]] = []
        self.ptr: int = 0
        self._max_priority: float = 1.0

    def add(self, features: dict[str, float], actual_quality: float, priority: float) -> None:
        entry = {"features": features, "actual_quality": actual_quality,
                 "priority": max(priority, 1e-6), "index": len(self.buffer)}
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            # When overwriting, if the evicted entry held _max_priority we must
            # recompute so future normalizations are not inflated.
            old_priority = self.buffer[self.ptr]["priority"]
            entry["index"] = self.ptr
            self.buffer[self.ptr] = entry
            self.ptr = (self.ptr + 1) % self.capacity
            if old_priority >= self._max_priority:
                self._max_priority = max((e["priority"] for e in self.buffer), default=1e-6)
        if priority > self._max_priority:
            self._max_priority = priority

    def sample(self, batch_size: int = 32) -> list[dict[str, Any]]:
        if not self.buffer:
            return []
        priorities = [e["priority"] ** self.alpha for e in self.buffer]
        total = sum(priorities)
        if total <= 0.0:
            result = random.choices(self.buffer, k=min(batch_size, len(self.buffer)))
        else:
            probs = [p / total for p in priorities]
            result = random.choices(self.buffer, weights=probs, k=min(batch_size, len(self.buffer)))
        n = len(self.buffer)
        for r in result:
            p = r["priority"] ** self.alpha
            total_p = sum(e["priority"] ** self.alpha for e in self.buffer)
            prob = p / max(total_p, 1e-12)
            w = (1.0 / (n * prob)) ** self.beta
            r["importance_weight"] = w
        max_w = max((r.get("importance_weight", 1.0) for r in result), default=1.0)
        for r in result:
            r["importance_weight"] = r.get("importance_weight", 1.0) / max(max_w, 1e-12)
        return result

    def update_priorities(self, indices: list[int], td_errors: list[float]) -> None:
        old_max = self._max_priority
        for idx, err in zip(indices, td_errors):
            for entry in self.buffer:
                if entry["index"] == idx:
                    old_prio = entry["priority"]
                    entry["priority"] = abs(err) + 1e-6
                    if abs(err) + 1e-6 > self._max_priority:
                        self._max_priority = abs(err) + 1e-6
                    elif old_prio >= old_max:
                        # We reduced the entry that held the max — recompute.
                        self._max_priority = max((e["priority"] for e in self.buffer), default=1e-6)
                    break

    def size(self) -> int:
        return len(self.buffer)

# ── Thompson Sampling ───────────────────────────────────────────────────────
def thompson_sample_utility(mem: IndexedMemory) -> float:
    """Thompson-sampled retrieval utility from Beta(α,β) posterior.
    RL theory: Bayesian exploration-exploitation — naturally balances trying
    uncertain memories vs exploiting known-good ones."""
    return _beta_sample(mem.thompson_alpha, mem.thompson_beta)

# ── Memory Retrieval (MMR) ──────────────────────────────────────────────────
def _memory_similarity(a: IndexedMemory, b: IndexedMemory) -> float:
    """Cosine-style similarity between two memories across key dimensions."""
    sim = 0.0
    sim += 0.3 * (1.0 if a.task_type == b.task_type else 0.0)
    sim += 0.4 * _jaccard(a.capability_requirements, b.capability_requirements)
    sim += 0.1 * (1.0 if a.outcome == b.outcome else 0.0)
    sim += 0.1 * (1.0 if a.model_tier == b.model_tier else 0.0)
    sim += 0.1 * (1.0 - abs(a.compression_ratio - b.compression_ratio))
    return sim

def retrieve_memories(
    state: Any,  # SharedStateV5-like
    task_type: TaskType,
    description: str,
    capability_requirements: list[str],
    retrieval_cfg: ActiveRetrievalConfig,
    idf_weights: dict[str, float],
    recency_decay: float,
    outcome_bonus: float,
    info_density_bonus: float,
) -> tuple[list[IndexedMemory], list[ContrastivePair]]:
    """Full 3-phase retrieval: 7-dim scoring → causal re-rank → MMR diversification.
    RL theory: MMR (Maximal Marginal Relevance) ensures diversity in retrieved set,
    preventing information cascade collapse."""
    now = _iso_now()
    task_words = set(description.lower().split())
    task_caps = set(capability_requirements)

    # Phase 1: Score every active memory
    scored: list[tuple[IndexedMemory, float, dict[str, float]]] = []
    for mem in state.memories:
        if mem.status != "active":
            continue
        dims: dict[str, float] = {}
        mem_words = set((mem.hypothesis + " " + mem.insight).lower().split())
        inter = task_words & mem_words
        if idf_weights and task_words:
            idf_score = sum(idf_weights.get(w, 1.0) for w in inter) / max(len(task_words | mem_words), 1)
        else:
            idf_score = len(inter) / max(len(task_words | mem_words), 1)
        dims["hypothesis_overlap"] = idf_score
        dims["capability_overlap"] = _jaccard(capability_requirements, mem.capability_requirements)
        dims["task_type_match"] = 1.0 if mem.task_type == task_type else 0.3
        days = _days_between(mem.last_retrieved_at or mem.created_at, now)
        dims["recency"] = math.exp(-recency_decay * days)
        dims["outcome_bonus"] = outcome_bonus if mem.outcome == "success" else 0.0
        dims["info_density"] = mem.surprise * mem.causal_utility * info_density_bonus
        dims["thompson_utility"] = thompson_sample_utility(mem)
        total = sum(dims[k] * retrieval_cfg.dim_weights.get(k, 0.0) for k in dims)
        scored.append((mem, total, dims))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Phase 2: Causal re-rank
    for i, (mem, score, dims) in enumerate(scored):
        scored[i] = (mem, score + mem.causal_utility * 0.3, dims)
    scored.sort(key=lambda x: x[1], reverse=True)

    # Phase 3: MMR selection
    selected: list[IndexedMemory] = []
    remaining = list(scored)
    k = retrieval_cfg.top_k
    while len(selected) < k and remaining:
        if not selected:
            selected.append(remaining.pop(0)[0])
        else:
            best_idx = -1
            best_mmr = -float("inf")
            for i, (mem, rel, _) in enumerate(remaining):
                max_sim = max((_memory_similarity(mem, s) for s in selected), default=0.0)
                mmr = retrieval_cfg.mmr_lambda * rel - (1.0 - retrieval_cfg.mmr_lambda) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            if best_idx >= 0:
                selected.append(remaining.pop(best_idx)[0])
            else:
                break

    # Contrastive mining
    successes = [m for m, _, _ in scored if m.outcome == "success"]
    failures = [m for m, _, _ in scored if m.outcome in ("failure", "crash")]
    pairs: list[ContrastivePair] = []
    for s_mem in successes[:20]:
        for f_mem in failures[:20]:
            sim = _jaccard(s_mem.capability_requirements, f_mem.capability_requirements)
            if sim > 0.3:
                rec = max(
                    math.exp(-0.05 * _days_between(s_mem.last_retrieved_at or s_mem.created_at, now)),
                    math.exp(-0.05 * _days_between(f_mem.last_retrieved_at or f_mem.created_at, now)),
                )
                pairs.append(ContrastivePair(
                    success_memory=s_mem, failure_memory=f_mem,
                    similarity=sim * rec,
                    shared_capabilities=list(set(s_mem.capability_requirements) & set(f_mem.capability_requirements)),
                ))
    pairs.sort(key=lambda p: p.similarity, reverse=True)
    return selected, pairs[:3]

# ── Value Function with TD(λ) ───────────────────────────────────────────────
def _find_memory(state: Any, mem_id: str) -> Optional[IndexedMemory]:
    """Linear scan for memory by ID. O(M) with M≤200."""
    for m in state.memories:
        if m.id == mem_id:
            return m
    return None

def decay_eligibility_traces(state: Any) -> None:
    """Multiply all traces by γ·λ, prune below threshold.
    RL theory: eligibility traces implement TD(λ) credit assignment across
    trial sequences, decaying contribution geometrically."""
    decay = state.value_function.gamma * state.value_function.lambda_
    threshold = 0.001
    dead: list[str] = []
    for mid in state.value_function.traces:
        state.value_function.traces[mid] *= decay
        if state.value_function.traces[mid] < threshold:
            dead.append(mid)
    for mid in dead:
        del state.value_function.traces[mid]

def bump_eligibility_traces(state: Any, memory_ids: list[str], multiplier: float = 1.0) -> None:
    """Increment trace for each memory by multiplier.
    Referenced memories get full bump (1.0); retrieved-only get partial (0.5)."""
    for mid in memory_ids:
        state.value_function.traces[mid] = state.value_function.traces.get(mid, 0.0) + multiplier

def compute_advantage(state: Any, task_type: TaskType, quality_score: float) -> float:
    """Advantage = actual quality - baseline EMA.
    RL theory: advantage normalization reduces variance in TD updates."""
    if task_type in state.value_function.baselines:
        baseline = state.value_function.baselines[task_type].ema
    else:
        baseline = state.value_function.global_baseline
    return quality_score - baseline

def update_value_function(state: Any, trial: Trial) -> float:
    """Full TD(λ) update: updates baseline EMA, eligibility traces, and memory
    causal utilities via trace-weighted TD error.
    RL theory: TD(λ) bridges temporal credit across trials; memory utilities
    reflect long-term contribution to quality outcomes."""
    vf = state.value_function
    t = trial.task_type
    if t not in vf.baselines:
        vf.baselines[t] = Baseline()
    bl = vf.baselines[t]
    reward = _composite_reward(trial.outcome, trial.quality_score)
    td_error = reward - bl.ema
    bl.ema += vf.alpha * td_error
    bl.count += 1
    if bl.count > 1:
        bl.mean = (bl.mean * (bl.count - 1) + trial.quality_score) / bl.count
    else:
        bl.mean = trial.quality_score
    bl.recent_scores.append(trial.quality_score)
    if len(bl.recent_scores) > 10:
        bl.recent_scores = bl.recent_scores[-10:]
    if len(bl.recent_scores) >= 6:
        half = len(bl.recent_scores) // 2
        bl.slope = (sum(bl.recent_scores[half:]) / (len(bl.recent_scores) - half)
                    - sum(bl.recent_scores[:half]) / half)
    vf.global_baseline = _ema(vf.global_baseline, trial.quality_score, 0.05)
    advantage = compute_advantage(state, t, trial.quality_score)
    adv_mult = 1.0 + _clamp(advantage, -0.5, 0.5)
    for mid, trace_val in list(vf.traces.items()):
        mem = _find_memory(state, mid)
        if mem is None:
            continue
        update = vf.alpha * trace_val * td_error * adv_mult
        mem.causal_utility = _clamp(mem.causal_utility + update, 0.0, 1.0)
        mem.td_error = abs(td_error)
        vf.memory_priorities[mid] = abs(td_error) + 0.01
        vf.max_priority = max(vf.max_priority, abs(td_error) + 0.01)
    vf.total_updates += 1
    return td_error

# ── UCB Parameter Selection ─────────────────────────────────────────────────
TUNABLE_PARAMS = [
    "compression.alpha", "compression.beta", "compression.gamma",
    "compression.theta1", "compression.theta2",
    "quality_threshold", "max_attempts",
    "temperature_t0", "temperature_t1", "temperature_t2",
    "model_low_complexity", "model_high_complexity",
    "retrieval.mmr_lambda", "retrieval.top_k", "recency_decay",
    "outcome_bonus", "info_density_bonus",
    "mutation_magnitude", "ucb_c", "rnd_weight",
]
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "compression.alpha": (0.1, 1.0), "compression.beta": (0.1, 1.0),
    "compression.gamma": (0.1, 1.0), "compression.theta1": (0.1, 0.8),
    "compression.theta2": (0.2, 0.9), "quality_threshold": (0.5, 0.95),
    "max_attempts": (1.0, 5.0), "temperature_t0": (0.0, 1.0),
    "temperature_t1": (0.0, 1.0), "temperature_t2": (0.0, 1.0),
    "model_low_complexity": (100.0, 5000.0), "model_high_complexity": (500.0, 20000.0),
    "retrieval.mmr_lambda": (0.3, 0.95), "retrieval.top_k": (1.0, 15.0),
    "recency_decay": (0.01, 0.5), "outcome_bonus": (0.0, 0.5),
    "info_density_bonus": (0.0, 0.3), "mutation_magnitude": (0.02, 0.5),
    "ucb_c": (0.5, 5.0), "rnd_weight": (0.0, 5.0),
}

def select_parameter_to_tune(evolution_log: list[EvolutionEntry], ucb_c: float) -> str:
    """UCB over evolution history to pick which parameter to mutate next.
    RL theory: UCB balances exploration (trying untuned params) vs exploitation
    (retuning params with positive fitness history)."""
    total_m = max(len(evolution_log), 1)
    best_param = TUNABLE_PARAMS[0]
    best_ucb = -float("inf")
    for param in TUNABLE_PARAMS:
        relevant = [e for e in evolution_log if e.mutation.get("param") == param]
        n = len(relevant)
        if n == 0:
            ucb = float("inf")
        else:
            avg_reward = sum(e.delta for e in relevant if e.decision == "keep") / max(n, 1)
            ucb = avg_reward + ucb_c * math.sqrt(math.log(total_m) / max(n, 0.1))
        if ucb > best_ucb:
            best_ucb = ucb
            best_param = param
    return best_param

# ── HER ─────────────────────────────────────────────────────────────────────
def her_relabel(trial: Trial) -> list[HERGoal]:
    """Generate hindsight relabelings for non-success trials.
    RL theory: HER converts failures into learning signals by reinterpreting
    the outcome as success toward an alternate goal (crash boundary, ruled-out
    config, side improvement on a quality dimension)."""
    if trial.outcome == "success":
        return []
    goals: list[HERGoal] = []
    if trial.outcome == "crash":
        for cap in trial.capability_requirements[:2]:
            goals.append(HERGoal(
                goal=f"find_crash_boundary_for_{cap}",
                outcome="success", reward=0.7,
                insight=f"Established crash boundary for {cap} under compression={trial.compression_ratio}",
            ))
    goals.append(HERGoal(
        goal="eliminate_ineffective_configuration", outcome="success", reward=0.5,
        insight=f"Ruled out config (alpha={trial.compression_weights.alpha}, "
                f"beta={trial.compression_weights.beta}, gamma={trial.compression_weights.gamma}, "
                f"model={trial.model_tier}) for {trial.task_type}",
    ))
    dim_names = ["completeness", "correctness", "consistency", "format"]
    for i, (score, name) in enumerate(zip(trial.quality_scores, dim_names)):
        if score > 0.6:
            goals.append(HERGoal(
                goal=f"optimize_{name}", outcome="success",
                reward=min(score - 0.5, 0.4),
                insight=f"Improved {name} to {score:.2f} despite overall quality below threshold",
            ))
    seen: set[str] = set()
    deduped: list[HERGoal] = []
    for g in goals:
        if g.insight not in seen:
            seen.add(g.insight)
            deduped.append(g)
    return deduped[:5]

# ── Counterfactuals ─────────────────────────────────────────────────────────
def synthesize_counterfactuals(trial: Trial) -> list[str]:
    """Heuristic counterfactual generation from trial outcome.
    RL theory: counterfactuals approximate the causal effect of parameter
    changes without requiring an LLM call, providing interpretable signals."""
    cfs: list[str] = []
    cw = trial.compression_weights
    if trial.outcome == "success":
        alt = "best" if trial.model_tier != "best" else "fast"
        cfs.append(
            f"Without compression (alpha={cw.alpha}, gamma={cw.gamma}) and "
            f"temperature {trial.temperature_schedule[0] if trial.temperature_schedule else 0.5}, "
            f"quality would likely be lower. Additional tuning of theta2 might compound gains."
        )
        cfs.append(
            f"Using model_tier={alt} instead of {trial.model_tier} might "
            f"{'further improve' if alt == 'best' else 'still achieve similar results at lower cost'}"
        )
    elif trial.outcome == "failure":
        alt = "best" if trial.model_tier != "best" else "medium"
        cfs.append(
            f"Parameters (alpha={cw.alpha}, beta={cw.beta}) did not help {trial.task_type}. "
            f"Higher gamma (more examples) or '{alt}' model tier might succeed. "
            f"Negative result rules out THIS configuration, not the approach."
        )
        cfs.append(
            f"Consider shifting compression: lower alpha ({cw.alpha * 0.8:.2f}), "
            f"higher gamma ({cw.gamma * 1.3:.2f})"
        )
    else:
        cfs.append(
            f"Conservative params: lower compression [{trial.compression_ratio * 0.7:.2f}], "
            f"higher temperature [{trial.temperature_schedule[0] * 1.5:.2f}], "
            f"or faster model might be more stable. Consider gradual rollout."
        )
        cfs.append(
            f"Crash at compression={trial.compression_ratio}, "
            f"token_budget={trial.token_budget_used}. Try doubling budget or "
            f"reducing compression to {trial.compression_ratio * 0.6:.2f}"
        )
    return cfs

# ── RND (Random Network Distillation) ─────────────────────────────────────
_RND_DIM = 8
_rnd_rng_target = random.Random(42)
_rnd_rng_predictor = random.Random(123)

def _ensure_rnd_init(rnd: RNDState, feat_dim: int) -> None:
    """Lazy-init RND matrices with fixed seeds for determinism.
    RL theory: RND provides an intrinsic exploration bonus proportional to
    prediction error of a fixed random target network — novel states get higher bonus."""
    if not rnd.initialized:
        rnd.target_projection = [[_rnd_rng_target.gauss(0.0, 1.0) for _ in range(_RND_DIM)]
                                 for _ in range(feat_dim)]
        rnd.predictor_weights = [[_rnd_rng_predictor.gauss(0.0, 0.1) for _ in range(_RND_DIM)]
                                 for _ in range(feat_dim)]
        rnd.predictor_bias = [0.0] * _RND_DIM
        rnd.initialized = True

def compute_rnd_bonus(rnd: RNDState, features: dict[str, float]) -> float:
    """MSE between fixed target and learned predictor embeddings, normalized.
    Returns exploration bonus scaled to [0, 5]."""
    feat_vec = [features.get(k, 0.0) for k in FEATURE_NAMES]
    _ensure_rnd_init(rnd, len(feat_vec))
    target = [0.0] * _RND_DIM
    pred = list(rnd.predictor_bias)
    for i in range(len(feat_vec)):
        for j in range(_RND_DIM):
            target[j] += feat_vec[i] * rnd.target_projection[i][j]
            pred[j] += feat_vec[i] * rnd.predictor_weights[i][j]
    mse = sum((t - p) ** 2 for t, p in zip(target, pred)) / _RND_DIM
    norm = (mse - rnd.error_mean) / max(rnd.error_std, 0.01)
    return 5.0 * _sigmoid(norm)

def train_rnd_predictor(rnd: RNDState, features: dict[str, float]) -> None:
    """One SGD step to reduce MSE between predictor and fixed target.
    This shrinks the bonus for familiar states, driving exploration to novel ones."""
    feat_vec = [features.get(k, 0.0) for k in FEATURE_NAMES]
    _ensure_rnd_init(rnd, len(feat_vec))
    target = [0.0] * _RND_DIM
    pred = list(rnd.predictor_bias)
    for i in range(len(feat_vec)):
        for j in range(_RND_DIM):
            target[j] += feat_vec[i] * rnd.target_projection[i][j]
            pred[j] += feat_vec[i] * rnd.predictor_weights[i][j]
    errors = [t - p for t, p in zip(target, pred)]
    mse = sum(e ** 2 for e in errors) / _RND_DIM
    rnd.error_mean = _ema(rnd.error_mean, mse, 0.01)
    rnd.error_std = _ema(rnd.error_std, abs(mse - rnd.error_mean), 0.01)
    lr = 0.001
    for i in range(len(feat_vec)):
        for j in range(_RND_DIM):
            grad = -2.0 * errors[j] * feat_vec[i] / _RND_DIM
            rnd.predictor_weights[i][j] -= lr * grad
    for j in range(_RND_DIM):
        rnd.predictor_bias[j] -= lr * (-2.0 * errors[j] / _RND_DIM)

# ── IDF Cache ───────────────────────────────────────────────────────────────
def rebuild_idf_cache(state: Any) -> None:
    """Rebuild IDF weights from all active memory hypotheses+insights.
    RL theory: IDF weighting gives rare but relevant terms higher retrieval
    weight, improving hypothesis-overlap scoring diversity."""
    doc_count = 0
    term_doc: dict[str, int] = defaultdict(int)
    for mem in state.memories:
        if mem.status != "active":
            continue
        doc_count += 1
        text = (mem.hypothesis + " " + mem.insight).lower()
        for term in set(text.split()):
            term_doc[term] += 1
    if doc_count == 0:
        return
    state.curiosity.idf_weights = {
        t: math.log((doc_count + 1.0) / (c + 1.0)) + 1.0
        for t, c in term_doc.items()
    }
    state.curiosity.idf_doc_count = doc_count
    state.curiosity.idf_last_rebuilt = _iso_now()

# ── Curriculum ──────────────────────────────────────────────────────────────
def get_curriculum_phase(trial_count: int) -> int:
    """Map trial count to curriculum phase [0-3].
    Phases: 0=broad_exploration(<10), 1=focused_exploitation(<30),
    2=principled_optimization(<60), 3=adversarial_refinement(≥60)."""
    for i, bound in enumerate((10, 30, 60)):
        if trial_count < bound:
            return i
    return 3

def get_curriculum_params(state: Any) -> PhaseConfig:
    """Return PhaseConfig for current curriculum stage."""
    phase = get_curriculum_phase(len(state.trials))
    return state.curriculum.phases[phase]

# ── Retrieval Strategy Evolution ────────────────────────────────────────────
def _get_param(obj: Any, path: str) -> Optional[float]:
    """Walk dot-path to retrieve a float parameter value."""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    val = getattr(obj, parts[-1], None)
    return float(val) if val is not None else None

def _set_param(obj: Any, path: str, value: float) -> None:
    """Walk dot-path to set a float parameter value."""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    # Handle retrieval sub-object
    if parts[-1] in ("mmr_lambda", "top_k"):
        setattr(getattr(obj, "retrieval"), parts[-1], value if parts[-1] == "mmr_lambda" else int(round(value)))
    else:
        setattr(obj, parts[-1], value)

def _compute_fitness(state: Any) -> float:
    """EMA of recent quality scores weighted by success rate."""
    recent = state.trials[-20:]
    if not recent:
        return 0.5
    q_sum = sum(t.quality_score for t in recent)
    s_rate = sum(1 for t in recent if t.outcome == "success") / len(recent)
    return (q_sum / len(recent)) * 0.7 + s_rate * 0.3

def _dominant_task_type(state: Any) -> str:
    recent = state.trials[-20:]
    if not recent:
        return "unknown"
    counts: dict[str, int] = defaultdict(int)
    for t in recent:
        counts[t.task_type] += 1
    return max(counts, key=counts.get)

def _trend(state: Any) -> str:
    recent = state.trials[-20:]
    if len(recent) < 5:
        return "insufficient_data"
    half = len(recent) // 2
    first = sum(t.quality_score for t in recent[:half]) / half
    second = sum(t.quality_score for t in recent[half:]) / (len(recent) - half)
    delta = second - first
    return "improving" if delta > 0.05 else ("declining" if delta < -0.05 else "stable")

def _clone_config(cfg: ActiveRetrievalConfig) -> ActiveRetrievalConfig:
    return ActiveRetrievalConfig(
        mmr_lambda=cfg.mmr_lambda, top_k=cfg.top_k,
        dim_weights=dict(cfg.dim_weights),
        token_budget_tiers=cfg.token_budget_tiers,
    )

def evolve_retrieval_strategy(state: Any) -> dict[str, Any]:
    """Self-evolution of retrieval hyperparams via UCB-guided mutations.
    RL theory: meta-learning over hyperparameter space — propose, test over
    N trials, keep if fitness improves or revert if it degrades."""
    rs = state.retrieval_strategy
    if rs.pending_mutation is None:
        param = select_parameter_to_tune(state.evolution_log, state.policy.ucb_c)
        old_val = _get_param(state.policy, param)
        if old_val is None:
            return {"generation": rs.generation, "mutation": None, "fitness_delta": None, "decision": "no_mutation"}
        phase = get_curriculum_params(state)
        new_val = old_val * _log_normal_sample(0.0, phase.mutation_magnitude)
        lo, hi = PARAM_BOUNDS.get(param, (0.0, 100.0))
        new_val = _clamp(new_val, lo, hi)
        if abs(new_val - old_val) / max(abs(old_val), 1e-8) < 0.01:
            return {"generation": rs.generation, "mutation": None, "fitness_delta": None, "decision": "no_mutation"}
        # Round integer params so the mutation log matches the actual stored value.
        effective_new_val = int(round(new_val)) if param == "retrieval.top_k" else new_val
        rs.ancestor = _clone_config(rs.active)
        rs.ancestor_fitness = _compute_fitness(state)
        rs.pending_mutation = PendingMutation(target_param=param, old_value=old_val, new_value=effective_new_val)
        _set_param(state.policy, param, effective_new_val)
        # Also update retrieval active copy
        if param.startswith("retrieval."):
            sub = param.split(".", 1)[1]
            setattr(rs.active, sub, effective_new_val if sub == "mmr_lambda" else int(round(effective_new_val)))
        elif param == "recency_decay" or param == "outcome_bonus" or param == "info_density_bonus":
            rs.active.dim_weights[{"recency_decay": "recency", "outcome_bonus": "outcome_bonus",
                                   "info_density_bonus": "info_density"}.get(param, param)] = 0.1  # approximate
        rs.trials_in_generation = 0
        return {"generation": rs.generation,
                "mutation": {"param": param, "old_value": old_val, "new_value": new_val},
                "fitness_delta": None, "decision": "no_mutation"}

    if rs.trials_in_generation >= 4:
        cur_fitness = _compute_fitness(state)
        delta = cur_fitness - rs.ancestor_fitness
        scenario = f"{_dominant_task_type(state)}_{get_curriculum_phase(len(state.trials))}_{_trend(state)}"
        decision: Literal["keep", "revert"] = "revert" if delta < -0.05 else "keep"
        if decision == "revert" and rs.ancestor is not None:
            rs.active = rs.ancestor
            _set_param(state.policy, rs.pending_mutation.target_param, rs.pending_mutation.old_value)
        rs.experience_library.append({
            "scenario": scenario,
            "mutation": {"param": rs.pending_mutation.target_param,
                         "direction": "increase" if rs.pending_mutation.new_value > rs.pending_mutation.old_value else "decrease"},
            "fitness_delta": delta, "decision": decision,
        })
        if len(rs.experience_library) > 100:
            rs.experience_library = rs.experience_library[-100:]
        state.evolution_log.append(EvolutionEntry(
            timestamp=_iso_now(), generation=rs.generation,
            mutation={"param": rs.pending_mutation.target_param,
                      "old_value": rs.pending_mutation.old_value,
                      "new_value": rs.pending_mutation.new_value},
            fitness_before=rs.ancestor_fitness, fitness_after=cur_fitness,
            delta=delta, decision=decision, scenario=scenario,
        ))
        result = {"generation": rs.generation,
                  "mutation": {"param": rs.pending_mutation.target_param,
                               "old_value": rs.pending_mutation.old_value,
                               "new_value": rs.pending_mutation.new_value},
                  "fitness_delta": delta, "decision": decision}
        rs.generation += 1
        rs.pending_mutation = None
        rs.ancestor = None
        rs.trials_in_generation = 0
        return result
    return {"generation": rs.generation, "mutation": None, "fitness_delta": None, "decision": "no_mutation"}

# ── Memory Consolidation ────────────────────────────────────────────────────
def _estimate_tokens(mem: IndexedMemory) -> int:
    text = mem.hypothesis + " " + mem.insight + " " + " ".join(mem.counterfactuals)
    return len(text.split())

def consolidate_memories(state: Any) -> dict[str, int]:
    """Merge low-utility active memories into consolidated summaries; archive cold ones.
    RL theory: hierarchical memory consolidation prevents retrieval cost explosion
    while preserving key insights at coarser granularity."""
    active = [m for m in state.memories if m.status == "active"]
    if len(active) <= 200:
        return {"consolidated": 0, "archived": 0, "tokens_freed": 0}
    consolidated = 0
    archived = 0
    tokens_freed = 0
    now = _iso_now()
    low_utility = [m for m in active
                   if m.causal_utility < 0.3
                   and (m.last_retrieved_at is None or _days_between(m.last_retrieved_at, now) > 30)]
    by_type: dict[str, list[IndexedMemory]] = defaultdict(list)
    for m in low_utility:
        by_type[m.task_type].append(m)
    for tt, group in by_type.items():
        if len(group) < 3:
            continue
        successes = sum(1 for m in group if m.outcome == "success")
        avg_quality = sum(m.quality_score for m in group) / len(group)
        best = max(group, key=lambda m: m.quality_score)
        cons = IndexedMemory(
            id=f"cons_{uuid.uuid4().hex[:8]}",
            source_trial_ids=[tid for m in group for tid in m.source_trial_ids],
            created_at=now, task_type=_str_to_tasktype(tt),
            capability_requirements=list(set(cap for m in group for cap in m.capability_requirements)),
            hypothesis=f"[Consolidated {len(group)} memories re {tt}]",
            insight=f"{tt}: {successes} successes, {len(group)-successes} failures. "
                    f"Best: alpha={best.params_used.get('alpha','?')}, "
                    f"beta={best.params_used.get('beta','?')}. Avg quality: {avg_quality:.2f}",
            outcome="success" if successes > len(group) - successes else "failure",
            quality_score=avg_quality,
            compression_ratio=sum(m.compression_ratio for m in group) / len(group),
            model_tier=best.model_tier, params_used=dict(best.params_used),
            thompson_alpha=sum(m.thompson_alpha for m in group) / len(group),
            thompson_beta=sum(m.thompson_beta for m in group) / len(group),
            causal_utility=sum(m.causal_utility for m in group) / len(group),
            consolidation_count=len(group), status="active",
        )
        for m in group:
            m.status = "consolidated"
            tokens_freed += _estimate_tokens(m)
        state.memories.append(cons)
        consolidated += 1
        state.consolidation_log.append(ConsolidationEntry(
            timestamp=now, action="consolidate",
            source_memory_ids=[m.id for m in group],
            target_memory_id=cons.id, tokens_saved=tokens_freed // len(group),
            quality_estimate=avg_quality,
            reason=f"Low-utility ({tt}, causal_utility<0.3)",
        ))
    for m in active:
        if m.status != "active":
            continue
        days = _days_between(m.last_retrieved_at or m.created_at, now)
        if days > 50:
            m.status = "cold"
            m.cold_since = now
            archived += 1
    if archived > 0:
        state.consolidation_log.append(ConsolidationEntry(
            timestamp=now, action="archive_cold_storage",
            source_memory_ids=[m.id for m in active if m.status == "cold"],
            tokens_saved=tokens_freed, quality_estimate=0.0,
            reason="Not retrieved in 50+ days",
        ))
    return {"consolidated": consolidated, "archived": archived, "tokens_freed": tokens_freed}

# ── Adversarial Verification ────────────────────────────────────────────────
def adversarial_verify(state: Any) -> dict[str, int]:
    """Re-score oldest 10% of active memories against current value function.
    Demote those with degraded utility.
    RL theory: distribution shift means old memories may no longer be relevant;
    periodic adversarial verification prevents policy from exploiting stale knowledge."""
    active = [m for m in state.memories if m.status == "active"]
    if len(active) < 10:
        return {"verified": 0, "stale": 0, "overturned": 0}
    active.sort(key=lambda m: m.created_at)
    n_verify = max(1, len(active) // 10)
    verified = 0
    stale = 0
    overturned = 0
    for mem in active[:n_verify]:
        verified += 1
        bl = state.value_function.baselines.get(mem.task_type)
        if bl is None:
            continue
        degradation = bl.ema - mem.causal_utility
        if degradation > 0.3:
            mem.thompson_alpha *= 0.5
            mem.thompson_beta *= 2.0
            if mem.thompson_alpha < 0.1:
                mem.status = "cold"
                mem.cold_since = _iso_now()
                overturned += 1
            else:
                stale += 1
    return {"verified": verified, "stale": stale, "overturned": overturned}

# ── Cross-Context Sync ──────────────────────────────────────────────────────
def sync_cross_context(state: Any, per_buffer: Optional[PrioritizedReplayBuffer] = None) -> dict[str, int]:
    """Consume pending trials from skill context, run full RL pipeline on them.
    RL theory: cross-context sync shares learning signals between the skill
    (lite, frequent) and autonomous (full, deep) execution contexts."""
    buf = state.cross_context_buffer
    pending = buf.pending_trials
    if not pending:
        return {"trials_processed": 0, "insights_generated": 0, "policy_diffs": 0}
    processed = 0
    for trial in list(pending):
        feats = extract_features(trial)
        update_predictive_model(state.predictive_model, trial, "full", per_buffer)
        state.curiosity.surprise.recent_values.append(trial.surprise)
        if len(state.curiosity.surprise.recent_values) > 50:
            state.curiosity.surprise.recent_values = state.curiosity.surprise.recent_values[-50:]
        state.curiosity.surprise.global_mean = (
            sum(state.curiosity.surprise.recent_values) / len(state.curiosity.surprise.recent_values))
        train_rnd_predictor(state.curiosity.rnd, feats)
        for mid in trial.referenced_memory_ids:
            mem = _find_memory(state, mid)
            if mem:
                if trial.outcome == "success":
                    mem.thompson_alpha += 1.0
                elif trial.outcome == "failure":
                    mem.thompson_beta += 1.0
                else:
                    mem.thompson_beta += 2.0
                mem.last_retrieved_at = _iso_now()
                mem.retrieval_count += 1
        for mid in set(trial.retrieved_memory_ids) - set(trial.referenced_memory_ids):
            mem = _find_memory(state, mid)
            if mem:
                if trial.outcome == "success":
                    mem.thompson_alpha += 0.5
                else:
                    mem.thompson_beta += 0.5
        if not any(t.id == trial.id for t in state.trials):
            state.trials.append(trial)
        processed += 1
    # Generate insights: detect canonical strategies from recent trials
    insights: list[str] = []
    recent = state.trials[-50:]
    for tt in TASK_TYPES:
        tt_trials = [t for t in recent if t.task_type == tt]
        if len(tt_trials) < 5:
            continue
        succ = [t for t in tt_trials if t.outcome == "success"]
        if len(succ) < 3:
            continue
        failures = [t for t in tt_trials if t.outcome != "success"]
        avg_alpha = sum(t.compression_weights.alpha for t in succ) / len(succ)
        if failures:
            fail_alpha = sum(t.compression_weights.alpha for t in failures) / len(failures)
            if avg_alpha > fail_alpha * 1.1:
                insights.append(f"For {tt}, higher alpha ({avg_alpha:.2f} vs {fail_alpha:.2f}) correlates with success")
        insights.append(f"{tt}: avg quality={sum(t.quality_score for t in tt_trials)/len(tt_trials):.2f}, "
                        f"success_rate={len(succ)/len(tt_trials):.0%}")
    # Canonical strategies: promote patterns with ≥5 trials and >80% success
    policy_diffs = 0
    by_type2: dict[str, list[Trial]] = defaultdict(list)
    for t in state.trials[-100:]:
        by_type2[t.task_type].append(t)
    for tt, trials in by_type2.items():
        if len(trials) < 5:
            continue
        succ2 = [t for t in trials if t.outcome == "success"]
        if len(succ2) / len(trials) < 0.8:
            continue
        avg_params = {
            "alpha": sum(t.compression_weights.alpha for t in succ2) / len(succ2),
            "beta": sum(t.compression_weights.beta for t in succ2) / len(succ2),
            "gamma": sum(t.compression_weights.gamma for t in succ2) / len(succ2),
            "top_k": int(sum(t.retrieval_top_k for t in succ2) / len(succ2)),
        }
        pattern = (f"For {tt}: alpha={avg_params['alpha']:.2f}, beta={avg_params['beta']:.2f}, "
                   f"gamma={avg_params['gamma']:.2f}, top_k={avg_params['top_k']}")
        existing = [s for s in buf.canonical_strategies if s.task_type == tt and s.pattern == pattern]
        if not existing:
            buf.canonical_strategies.append(CanonicalStrategy(
                strategy_id=f"cs_{uuid.uuid4().hex[:8]}", task_type=_str_to_tasktype(tt),
                pattern=pattern, params=avg_params,
                success_rate=len(succ2) / len(trials), trial_count=len(trials),
                discovered_by="autonomous", discovered_at=_iso_now(),
            ))
            policy_diffs += 1
    buf.refined_insights = {
        "discovered_patterns": insights,
        "updated_memory_utils": {m.id: m.causal_utility for m in state.memories if m.status == "active"},
    }
    buf.last_sync = _iso_now()
    buf.pending_trials = []
    buf.pending_count = 0
    buf.oldest_pending = ""
    return {"trials_processed": processed, "insights_generated": len(insights), "policy_diffs": policy_diffs}

# ── Periodic Scheduler ──────────────────────────────────────────────────────
# Per-phase verification intervals (decreasing — verify more often as we refine).
_VERIFICATION_INTERVALS: dict[int, int] = {0: 8, 1: 6, 2: 4, 3: 3}
# Per-phase IDF rebuild intervals (decreasing — rebuild more often as memories grow).
_IDF_REBUILD_INTERVALS: dict[int, int] = {0: 50, 1: 40, 2: 30, 3: 25}

class PeriodicScheduler:
    """Decides which periodic operations to run after each trial,
    gated by curriculum phase intervals.

    All intervals are phase-configurable (unlike the earlier hardcoded
    % 4 / % 50 approach). Synced with TS PeriodicScheduler."""

    def __init__(self, state: Any):
        self.state = state

    def after_trial(self, trial_count: int) -> set[str]:
        """Return set of operation labels that should run after this trial."""
        if trial_count == 0:
            return set()
        phase_params = get_curriculum_params(self.state)
        phase = get_curriculum_phase(trial_count)
        ops: set[str] = set()

        # Evolution: gated by curriculum learning interval
        if trial_count % phase_params.learning_interval == 0:
            ops.add("evolution")

        # Consolidation: gated by curriculum consolidation interval
        if trial_count % phase_params.consolidation_interval == 0:
            ops.add("consolidation")

        # Adversarial verification: gated by phase-specific interval
        verify_interval = _VERIFICATION_INTERVALS.get(phase, 4)
        if trial_count % verify_interval == 0:
            ops.add("verification")

        # IDF rebuild: gated by phase-specific interval
        idf_interval = _IDF_REBUILD_INTERVALS.get(phase, 50)
        if trial_count % idf_interval == 0:
            ops.add("idf_rebuild")

        return ops

# ── RLEngineV5 (Top-Level API) ──────────────────────────────────────────────
class RLEngineV5:
    """Unified RL Engine for Turbocontext v5.
    Pure-functions-over-state design. State is loaded/saved atomically.
    Holds state in memory plus optional PER buffer (full mode).

    V5.1 additions (synced with TS SharedStateManager):
      - Dirty flag: save() is a no-op when !dirty. Mutators set dirty=True.
      - JSONL audit logs: immutable append-only trial/evolution/consolidation logs.
      - save_force(): guaranteed persistence regardless of dirty flag."""

    def __init__(self, state_path: str = "~/.turbocontext/state-v5.json"):
        self.state_path = os.path.expanduser(state_path)
        self.backup_path = self.state_path.replace(".json", ".backup.json")
        self.log_dir = os.path.join(os.path.dirname(self.state_path), "logs")
        self.state: Any = None
        self.per_buffer: Optional[PrioritizedReplayBuffer] = None
        # V5.1: Dirty-flag persistence (synced with TS SharedStateManager)
        self._dirty: bool = False
        # V5.1: JSONL audit log paths
        self._trials_log_path = os.path.join(self.log_dir, "trials.jsonl")
        self._evolution_log_path = os.path.join(self.log_dir, "evolution.jsonl")
        self._consolidation_log_path = os.path.join(self.log_dir, "consolidation.jsonl")
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    # ── Serialization ──
    @staticmethod
    def _dataclass_to_dict(obj: Any) -> Any:
        """Recursively convert dataclass/object→dict, handling nested structures."""
        if obj is None:
            return None
        if isinstance(obj, (int, float, str, bool)):
            return obj
        if isinstance(obj, (list, tuple)):
            return [RLEngineV5._dataclass_to_dict(x) for x in obj]
        if isinstance(obj, dict):
            return {k: RLEngineV5._dataclass_to_dict(v) for k, v in obj.items()}
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for f_name in obj.__dataclass_fields__:
                val = getattr(obj, f_name)
                result[f_name] = RLEngineV5._dataclass_to_dict(val)
            return result
        if hasattr(obj, '__dict__'):
            result = {}
            for k, v in vars(obj).items():
                if not k.startswith('_'):
                    result[k] = RLEngineV5._dataclass_to_dict(v)
            return result
        return str(obj)


    def load_state(self) -> Any:
        """Load state from JSON. Create default if missing. Migrate v4→v5 if needed."""
        try:
            with open(self.state_path, 'r') as f:
                raw = json.load(f)
            self.state = self._build_state(raw)
        except FileNotFoundError:
            self.state = self._default_state()
            self.save_state()
        except (json.JSONDecodeError, KeyError) as e:
            try:
                with open(self.backup_path, 'r') as f:
                    raw = json.load(f)
                self.state = self._build_state(raw)
            except Exception:
                raise RuntimeError(
                    f"Failed to load state from {self.state_path} or backup. "
                    f"Corruption detected. Original error: {e}"
                )
        if self.per_buffer is None:
            self.per_buffer = PrioritizedReplayBuffer(500)
            for trial in self.state.trials[-500:]:
                feats = extract_features(trial)
                normed = _normalize_features(feats, self.state.predictive_model.feature_stats)
                self.per_buffer.add(normed, trial.quality_score, abs(trial.surprise) + 0.01)
        return self.state

    def _default_state(self) -> Any:
        """Build a fresh default state object (instance attributes, not class-level)."""
        now = _iso_now()
        class State: pass
        st = State()
        st.version = 5
        st.created_at = now
        st.last_updated = now
        st.total_invocations = 0
        st.trials: list[Trial] = []
        st.memories: list[IndexedMemory] = []
        st.policy = PolicyState()
        st.value_function = ValueFunctionState()
        st.predictive_model = PredictiveModelState()
        st.curiosity = CuriosityState()
        st.retrieval_strategy = RetrievalStrategyState()
        st.curriculum = CurriculumState()
        st.consolidation_log: list[ConsolidationEntry] = []
        st.cross_context_buffer = CrossContextBuffer()
        st.evolution_log: list[EvolutionEntry] = []
        return st

    def _build_state(self, raw: dict[str, Any]) -> Any:
        """Build state object from raw dict, recursively deserializing all nested structures."""
        class BuiltState:
            pass
        st = BuiltState()
        st.version = raw.get("version", 5)
        st.created_at = raw.get("created_at", _iso_now())
        st.last_updated = raw.get("last_updated", _iso_now())
        st.total_invocations = raw.get("total_invocations", 0)
        st.trials = self._deserialize_list(raw.get("trials", []), Trial)
        st.memories = self._deserialize_list(raw.get("memories", []), IndexedMemory)
        st.policy = self._deserialize_obj(raw.get("policy", {}), PolicyState)
        st.value_function = self._deserialize_obj(raw.get("value_function", {}), ValueFunctionState)
        st.predictive_model = self._deserialize_obj(raw.get("predictive_model", {}), PredictiveModelState)
        st.curiosity = self._deserialize_obj(raw.get("curiosity", {}), CuriosityState)
        st.retrieval_strategy = self._deserialize_obj(raw.get("retrieval_strategy", {}), RetrievalStrategyState)
        st.curriculum = self._deserialize_obj(raw.get("curriculum", {}), CurriculumState)
        st.consolidation_log = self._deserialize_list(raw.get("consolidation_log", []), ConsolidationEntry)
        st.cross_context_buffer = self._deserialize_obj(raw.get("cross_context_buffer", {}), CrossContextBuffer)
        st.evolution_log = self._deserialize_list(raw.get("evolution_log", []), EvolutionEntry)
        return st

    def _deserialize_list(self, data: list[Any], cls: type) -> list[Any]:
        if not isinstance(data, list):
            return []
        return [self._deserialize_obj(item, cls) if isinstance(item, dict) else item for item in data]

    def _deserialize_obj(self, data: dict[str, Any], cls: type) -> Any:
        """Deserialize a dict into a dataclass, recursively handling nested dataclass fields."""
        if not isinstance(data, dict):
            return cls()
        kwargs: dict[str, Any] = {}
        field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
        for f_name in cls.__dataclass_fields__:
            if f_name in data:
                val = data[f_name]
                f_type = field_types[f_name]
                kwargs[f_name] = self._cast_field(val, f_type)
        return cls(**kwargs)

    def _cast_field(self, val: Any, f_type: Any) -> Any:
        """Cast a JSON value to the expected field type, recursing into nested dataclasses."""
        if val is None:
            return None
        origin = get_origin(f_type)
        args = get_args(f_type)
        # Optional[X] / Union[X, None]
        if origin is Union:
            non_none = [a for a in args if a is not type(None)]
            if val is None and type(None) in args:
                return None
            if non_none:
                return self._cast_field(val, non_none[0])
            return val
        # list[X]
        if origin is list:
            inner = args[0] if args else str
            if isinstance(val, list):
                if hasattr(inner, '__dataclass_fields__'):
                    return [self._deserialize_obj(item, inner) if isinstance(item, dict) else item for item in val]
                return [self._cast_field(item, inner) for item in val]
            return []
        # dict[K,V] — recursively deserialize values if they are dataclasses
        if origin is dict:
            if not isinstance(val, dict):
                return {}
            # Cast keys to expected key type (e.g., str→int for dict[int, ...])
            key_type = args[0] if len(args) >= 1 else str
            def _cast_key(k):
                try:
                    return key_type(k)
                except (ValueError, TypeError):
                    return k
            if len(args) >= 2 and hasattr(args[1], '__dataclass_fields__'):
                return {_cast_key(k): self._deserialize_obj(v, args[1]) if isinstance(v, dict) else v
                        for k, v in val.items()}
            return {_cast_key(k): v for k, v in val.items()}
        # tuple[X,Y,Z]
        if origin is tuple:
            return tuple(val) if isinstance(val, (list, tuple)) else val
        # Literal[...]
        if origin is Literal:
            return val
        # Nested dataclass
        if hasattr(f_type, '__dataclass_fields__'):
            if isinstance(val, dict):
                return self._deserialize_obj(val, f_type)
            return val
        # Primitive casts
        try:
            return f_type(val)
        except (ValueError, TypeError):
            return val

    # ── Persistence (V5.1: dirty-flag gated) ──

    def is_dirty(self) -> bool:
        """True if in-memory state has been modified since last save."""
        return self._dirty

    def _mark_dirty(self) -> None:
        """Set the dirty flag. Called by every mutator."""
        self._dirty = True

    def save_state(self) -> bool:
        """Atomic save: write to .tmp, then rename. Rotates backup.
        V5.1: no-op if !dirty (returns False). Use save_force() to bypass."""
        if self.state is None:
            return False
        if not self._dirty:
            return False
        self.state.last_updated = _iso_now()
        data = self._dataclass_to_dict(self.state)
        tmp = self.state_path + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        if os.path.exists(self.state_path):
            try:
                os.rename(self.state_path, self.backup_path)
            except OSError:
                pass
        os.rename(tmp, self.state_path)
        self._dirty = False
        return True

    def save_force(self) -> bool:
        """Guaranteed persistence regardless of dirty flag."""
        self._dirty = True
        return self.save_state()

    # ── JSONL Audit Logs (V5.1, synced with TS src/state/io.ts) ──

    def _append_jsonl(self, log_path: str, entry: dict[str, Any]) -> None:
        """Append a single JSON object as a JSONL line, then fsync.
        Non-fatal on failure — logging is best-effort."""
        try:
            line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
            with open(log_path, 'a') as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        except (OSError, TypeError):
            pass  # best-effort audit logging

    def _append_trial_log(self, trial: Trial) -> None:
        """Log a trial to the immutable JSONL audit trail."""
        self._append_jsonl(self._trials_log_path,
                          {"type": "trial", "data": self._dataclass_to_dict(trial)})

    def _append_evolution_log(self, entry: EvolutionEntry) -> None:
        """Log an evolution event to the immutable JSONL audit trail."""
        self._append_jsonl(self._evolution_log_path,
                          {"type": "evolution", "data": self._dataclass_to_dict(entry)})

    def _append_consolidation_log(self, entry: ConsolidationEntry) -> None:
        """Log a consolidation event to the immutable JSONL audit trail."""
        self._append_jsonl(self._consolidation_log_path,
                          {"type": "consolidation", "data": self._dataclass_to_dict(entry)})

    # ── Query ──
    def query_optimal_params(
        self, task_type: TaskType, description: str,
        capability_requirements: list[str],
    ) -> QueryResult:
        """Determine optimal execution parameters for an upcoming trial.
        Used by both Context A (skill) and Context B (agent.py)."""
        assert self.state is not None, "Call load_state() first"
        p = self.state.policy
        complexity = len(description)
        if complexity <= p.model_low_complexity:
            model_tier: ModelTier = "fast"
        elif complexity >= p.model_high_complexity:
            model_tier = "best"
        else:
            model_tier = "medium"
        if complexity <= 1000:
            token_budget = p.retrieval.token_budget_tiers[0]
        elif complexity <= 3000:
            token_budget = p.retrieval.token_budget_tiers[1]
        else:
            token_budget = p.retrieval.token_budget_tiers[2]
        memories, contrastive = retrieve_memories(
            self.state, task_type, description, capability_requirements,
            p.retrieval, self.state.curiosity.idf_weights,
            p.recency_decay, p.outcome_bonus, p.info_density_bonus,
        )
        for mem in memories:
            mem.last_retrieved_at = _iso_now()
            mem.retrieval_count += 1
        insights: list[str] = []
        for cp in contrastive:
            insights.append(
                f"SIMILAR ({', '.join(cp.shared_capabilities)}) but OPPOSITE: "
                f"SUCCESS ({cp.success_memory.insight[:120]}) vs "
                f"FAILURE ({cp.failure_memory.insight[:120]})"
            )
        rnd_bonus = compute_rnd_bonus(self.state.curiosity.rnd,
                                       extract_features_dummy(task_type, description, capability_requirements))
        return QueryResult(
            compression_weights=CompressionWeights(
                alpha=p.compression.alpha, beta=p.compression.beta, gamma=p.compression.gamma),
            temperature_schedule=[p.temperature_t0, p.temperature_t1, p.temperature_t2],
            model_tier=model_tier,
            retrieval_params={"mmr_lambda": p.retrieval.mmr_lambda, "top_k": p.retrieval.top_k},
            token_budget=token_budget, max_attempts=p.max_attempts,
            quality_threshold=p.quality_threshold,
            retrieved_memories=memories, contrastive_insights=insights,
            curriculum_phase=get_curriculum_phase(len(self.state.trials)),
            exploration_bonus=rnd_bonus,
        )

    # ── Record ──
    def record_trial(self, trial: Trial, mode: RecordMode = "lite") -> RecordResult:
        """Record a trial and run RL learning pipeline.
        Lite: fast SGD + Thompson + counterfactuals + HER.
        Full: adds TD(λ), PER, RND training."""
        assert self.state is not None
        st = self.state
        trial.predicted_quality = predict_quality(st.predictive_model, trial)
        surprise = update_predictive_model(st.predictive_model, trial, mode, self.per_buffer)
        trial.surprise = surprise
        trial.her_goals = her_relabel(trial)
        self._update_thompson_beliefs(st, trial, mode)
        # HER: process relabeled goals for Thompson updates
        for hg in trial.her_goals:
            if hg.outcome == "success":
                for mid in trial.referenced_memory_ids:
                    mem = _find_memory(st, mid)
                    if mem:
                        mem.thompson_alpha += hg.reward
        td_error = 0.0
        if mode == "full":
            decay_eligibility_traces(st)
            bump_eligibility_traces(st, trial.referenced_memory_ids, 1.0)
            bump_eligibility_traces(st, list(set(trial.retrieved_memory_ids) - set(trial.referenced_memory_ids)), 0.5)
            td_error = update_value_function(st, trial)
            # HER: apply value function updates with HER rewards via trace-weighted TD
            vf = st.value_function
            bl = vf.baselines.get(trial.task_type)
            if bl is not None:
                for hg in trial.her_goals:
                    her_td = hg.reward - bl.ema
                    bl.ema += vf.alpha * her_td
                    for mid, trace_val in list(vf.traces.items()):
                        mem = _find_memory(st, mid)
                        if mem is None:
                            continue
                        update = vf.alpha * trace_val * her_td
                        mem.causal_utility = _clamp(mem.causal_utility + update, 0.0, 1.0)
            trial.advantage = compute_advantage(st, trial.task_type, trial.quality_score)
        trial.counterfactuals = synthesize_counterfactuals(trial)
        trial.curriculum_phase = get_curriculum_phase(len(st.trials))
        st.trials.append(trial)
        st.total_invocations += 1
        self._mark_dirty()
        feats = extract_features(trial)
        self._update_curiosity(st, trial, surprise)
        if mode == "full":
            train_rnd_predictor(st.curiosity.rnd, feats)
        sched = PeriodicScheduler(st)
        ops = sched.after_trial(len(st.trials))
        if "evolution" in ops:
            self.run_evolution_step()
        if "consolidation" in ops:
            self.run_consolidation()
        if "verification" in ops and mode == "full":
            self.run_adversarial_verification()
        if "idf_rebuild" in ops:
            rebuild_idf_cache(st)
        if mode == "lite":
            buf = st.cross_context_buffer
            buf.pending_trials.append(trial)
            buf.pending_count = len(buf.pending_trials)
            if not buf.oldest_pending or (trial.timestamp and trial.timestamp < buf.oldest_pending):
                buf.oldest_pending = trial.timestamp
        # V5.1: JSONL audit log (best-effort, non-fatal)
        self._append_trial_log(trial)
        self.save_state()
        return RecordResult(
            surprise=surprise, td_error=td_error,
            counterfactuals=trial.counterfactuals, her_goals=trial.her_goals,
            memories_updated=len(trial.referenced_memory_ids),
            pending_sync_count=st.cross_context_buffer.pending_count if mode == "lite" else 0,
        )

    def _update_thompson_beliefs(self, st: Any, trial: Trial, mode: RecordMode) -> None:
        """Update Beta posterior parameters for retrieved/referenced memories."""
        fm = 1.0 if mode == "full" else 0.5
        hm = 0.5 if mode == "full" else 1.0
        for mid in trial.referenced_memory_ids:
            mem = _find_memory(st, mid)
            if mem:
                if trial.outcome == "success":
                    mem.thompson_alpha += fm
                elif trial.outcome == "failure":
                    mem.thompson_beta += fm
                else:
                    mem.thompson_beta += 2.0 * fm
        for mid in set(trial.retrieved_memory_ids) - set(trial.referenced_memory_ids):
            mem = _find_memory(st, mid)
            if mem:
                if trial.outcome == "success":
                    mem.thompson_alpha += hm
                else:
                    mem.thompson_beta += hm
        # Decay: prevent unbounded growth of Beta parameters so Thompson
        # sampling retains a meaningful exploration-exploitation balance.
        _thompson_cap = 100.0
        _thompson_decay = 0.9995
        all_ids = set(trial.referenced_memory_ids) | (set(trial.retrieved_memory_ids) - set(trial.referenced_memory_ids))
        for mid in all_ids:
            mem = _find_memory(st, mid)
            if mem:
                if mem.thompson_alpha > _thompson_cap or mem.thompson_beta > _thompson_cap:
                    mem.thompson_alpha *= 0.5
                    mem.thompson_beta *= 0.5
                else:
                    mem.thompson_alpha *= _thompson_decay
                    mem.thompson_beta *= _thompson_decay

    def _update_curiosity(self, st: Any, trial: Trial, surprise: float) -> None:
        """Update curiosity stats: surprise distribution, task-type exploration, capability coverage."""
        c = st.curiosity
        ss = c.surprise
        ss.recent_values.append(surprise)
        if len(ss.recent_values) > 50:
            ss.recent_values = ss.recent_values[-50:]
        ss.global_mean = sum(ss.recent_values) / max(len(ss.recent_values), 1)
        variance = sum((x - ss.global_mean) ** 2 for x in ss.recent_values) / max(len(ss.recent_values) - 1, 1)
        ss.global_std = math.sqrt(max(variance, 1e-8))
        ss.anomaly_threshold = ss.global_mean + 2.0 * ss.global_std
        if trial.task_type not in c.task_type_exploration:
            c.task_type_exploration[trial.task_type] = {"count": 0.0, "avg_surprise": 0.0, "success_rate": 0.5}
        tte = c.task_type_exploration[trial.task_type]
        tte["count"] = tte.get("count", 0.0) + 1.0
        tte["avg_surprise"] = _ema(tte.get("avg_surprise", 0.0), surprise, 0.1)
        tte["success_rate"] = _ema(tte.get("success_rate", 0.5),
                                   1.0 if trial.outcome == "success" else 0.0, 0.1)
        for cap in trial.capability_requirements:
            c.capability_coverage[cap] = c.capability_coverage.get(cap, 0) + 1

    # ── Periodic Operations ──
    def run_evolution_step(self) -> dict[str, Any]:
        result = evolve_retrieval_strategy(self.state)
        self._mark_dirty()
        # Log if a mutation was actually proposed
        if result.get("mutation") is not None:
            entry = EvolutionEntry(
                timestamp=_iso_now(),
                generation=result.get("generation", 0),
                mutation=result.get("mutation", {}),
                fitness_before=0.0,
                fitness_after=0.0,
                delta=result.get("fitness_delta") or 0.0,
                decision=result.get("decision", "keep"),
                scenario="",
            )
            self._append_evolution_log(entry)
        return result

    def run_consolidation(self) -> dict[str, int]:
        result = consolidate_memories(self.state)
        self._mark_dirty()
        if result.get("consolidated", 0) > 0 or result.get("archived", 0) > 0:
            entry = ConsolidationEntry(
                timestamp=_iso_now(),
                action="consolidate" if result.get("consolidated", 0) > 0 else "archive_cold_storage",
                source_memory_ids=[],
                target_memory_id=None,
                tokens_saved=result.get("tokens_freed", 0),
                quality_estimate=0.0,
                reason=f"Consolidated {result.get('consolidated', 0)}, archived {result.get('archived', 0)}",
            )
            self._append_consolidation_log(entry)
        return result

    def run_adversarial_verification(self) -> dict[str, int]:
        result = adversarial_verify(self.state)
        self._mark_dirty()
        return result

    def run_cross_context_sync(self) -> dict[str, int]:
        result = sync_cross_context(self.state, self.per_buffer)
        self._mark_dirty()
        return result

    # ── Introspection ──
    def get_status(self) -> StatusReport:
        st = self.state
        active_m = sum(1 for m in st.memories if m.status == "active")
        cold_m = sum(1 for m in st.memories if m.status == "cold")
        phase = get_curriculum_phase(len(st.trials))
        per_tt: dict[str, dict[str, Any]] = {}
        for tt in TASK_TYPES:
            tt_trials = [t for t in st.trials if t.task_type == tt]
            if tt_trials:
                avg_q = sum(t.quality_score for t in tt_trials) / len(tt_trials)
                bl = st.value_function.baselines.get(tt, Baseline())
                per_tt[tt] = {"trial_count": len(tt_trials), "avg_quality": round(avg_q, 3),
                              "baseline_quality": round(bl.ema, 3),
                              "is_plateaued": abs(bl.slope) < 0.01 and len(tt_trials) >= 10,
                              "improvement_slope": round(bl.slope, 4)}
            else:
                per_tt[tt] = {"trial_count": 0, "avg_quality": 0.0, "baseline_quality": 0.5,
                              "is_plateaued": False, "improvement_slope": 0.0}
        errors = st.predictive_model.recent_errors
        acc = 1.0 - (sum(errors) / len(errors)) if errors else 0.0
        sm = st.curiosity.surprise.global_mean
        le = st.evolution_log[-1].timestamp if st.evolution_log else ""
        lc = st.consolidation_log[-1].timestamp if st.consolidation_log else ""
        ls = st.cross_context_buffer.last_sync
        return StatusReport(
            total_trials=len(st.trials), active_memories=active_m, cold_memories=cold_m,
            curriculum_phase=phase, per_task_type=per_tt,
            predictive_model_accuracy=round(acc, 4), surprise_mean=round(sm, 4),
            last_evolution=le, last_consolidation=lc, last_cross_sync=ls,
        )

    def get_contrastive_insights(self, task_type: TaskType) -> list[str]:
        """High-signal contrastive insights for a specific task type."""
        st = self.state
        recent = [t for t in st.trials[-50:] if t.task_type == task_type]
        succ = [t for t in recent if t.outcome == "success"]
        fail = [t for t in recent if t.outcome != "success"]
        insights: list[str] = []
        if succ and fail:
            s_alpha = sum(t.compression_weights.alpha for t in succ) / len(succ)
            f_alpha = sum(t.compression_weights.alpha for t in fail) / len(fail)
            if s_alpha != f_alpha:
                d = "higher" if s_alpha > f_alpha else "lower"
                insights.append(f"Successful {task_type} used {d} alpha ({s_alpha:.2f} vs {f_alpha:.2f})")
            s_gamma = sum(t.compression_weights.gamma for t in succ) / len(succ)
            f_gamma = sum(t.compression_weights.gamma for t in fail) / len(fail)
            if s_gamma != f_gamma:
                d = "higher" if s_gamma > f_gamma else "lower"
                insights.append(f"Successful {task_type} used {d} gamma ({s_gamma:.2f} vs {f_gamma:.2f})")
        if len(recent) >= 5:
            avg_q = sum(t.quality_score for t in recent) / len(recent)
            td = _trend(st)
            insights.append(f"{task_type}: {len(recent)} recent trials, avg quality {avg_q:.2f}, trend {td}")
        return insights


def extract_features_dummy(task_type: TaskType, description: str,
                           capability_requirements: list[str]) -> dict[str, float]:
    """Extract a minimal feature vector for RND bonus computation without a full Trial."""
    feats: dict[str, float] = {}
    for tt in TASK_TYPES:
        feats[f"task_{tt}"] = 1.0 if task_type == tt else 0.0
    feats["description_length_log"] = math.log(max(len(description), 1))
    feats["compression_ratio"] = 0.5
    feats["alpha"] = 0.6
    feats["beta"] = 0.5
    feats["gamma"] = 0.45
    feats["temperature_mean"] = 0.5
    feats["model_tier_fast"] = 0.0
    feats["model_tier_best"] = 0.0
    feats["is_retry"] = 0.0
    feats["hour_of_day"] = 0.5
    feats["top_k"] = 5.0
    feats["token_budget_log"] = math.log(8000)
    return feats


# ============================================================================
# V6 Features (synced with TypeScript v6, 2026-07-21)
# ============================================================================

# ── Cross-Branch Transfer ─────────────────────────────────────────────────

def task_type_similarity(a: TaskType, b: TaskType,
                         memories: list[dict[str, Any]]) -> float:
    """Jaccard similarity of capability requirements from shared memories."""
    caps_a = set()
    caps_b = set()
    for mem in memories:
        if mem.get("status") == "cold":
            continue
        reqs = mem.get("capabilityRequirements", [])
        if mem.get("taskType") == a:
            caps_a.update(reqs)
        if mem.get("taskType") == b:
            caps_b.update(reqs)
    if not caps_a or not caps_b:
        return 0.0
    return len(caps_a & caps_b) / max(len(caps_a | caps_b), 1)


def transfer_policy(target_tt: TaskType, state: Any,
                    min_source_trials: int = 10,
                    transfer_threshold: float = 0.4
                    ) -> dict[str, Any] | None:
    """Find best source task type and return blended compression params."""
    trial_counts: dict[TaskType, int] = defaultdict(int)
    for t in state.trials:
        trial_counts[t.taskType] += 1

    if trial_counts.get(target_tt, 0) >= min_source_trials:
        return None

    best_source = None
    best_sim = 0.0
    best_params = None

    for source_tt in TASK_TYPES:
        if source_tt == target_tt:
            continue
        if trial_counts.get(source_tt, 0) < min_source_trials:
            continue
        sim = task_type_similarity(target_tt, source_tt, state.memories)
        if sim >= transfer_threshold and sim > best_sim:
            best_sim = sim
            best_source = source_tt
            succ = [t for t in state.trials
                    if t.taskType == source_tt and t.outcome == "success"]
            if succ:
                best_params = {
                    "alpha": sum(t.compressionWeights.alpha for t in succ) / len(succ),
                    "beta": sum(t.compressionWeights.beta for t in succ) / len(succ),
                    "gamma": sum(t.compressionWeights.gamma for t in succ) / len(succ),
                }

    if not best_source or not best_params:
        return None
    return {"compression": best_params, "similarity": best_sim,
            "sourceTaskType": best_source}


def blend_params(base: dict[str, float], transfer: dict[str, float],
                 similarity: float) -> dict[str, float]:
    """Blend parameters: param = (1-sim)*base + sim*transfer."""
    t = max(0.0, min(1.0, similarity))
    return {
        "alpha": (1 - t) * base["alpha"] + t * transfer["alpha"],
        "beta": (1 - t) * base["beta"] + t * transfer["beta"],
        "gamma": 1.0 - ((1 - t) * base["alpha"] + t * transfer["alpha"])
                       - ((1 - t) * base["beta"] + t * transfer["beta"]),
    }


# ── Experiment Type Selection ─────────────────────────────────────────────

EXPERIMENT_TYPES = ["hypothesis_test", "parameter_sweep", "ablation_study",
                    "transfer_experiment", "boundary_probe", "adversarial_test"]


def select_experiment_type(total_experiments: int) -> str:
    """Pick experiment type with phase-weighted probabilities (Karpathy-style)."""
    if total_experiments < 10:
        weights = [0.40, 0.25, 0.10, 0.10, 0.10, 0.05]
    elif total_experiments < 30:
        weights = [0.30, 0.20, 0.15, 0.15, 0.10, 0.10]
    else:
        weights = [0.20, 0.15, 0.20, 0.15, 0.10, 0.20]

    r = random.random()
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r < cumulative:
            return EXPERIMENT_TYPES[i]
    return EXPERIMENT_TYPES[0]


# ── Simplicity Criterion ──────────────────────────────────────────────────

def compute_simplicity(mutation_type: str | None) -> float:
    """Estimate simplicity of a mutation. 1.0 = simplest (deleting code)."""
    if mutation_type is None:
        return 1.0
    scores = {
        "remove_round": 0.95, "remove_quality_criterion": 0.95,
        "merge_rounds": 0.80, "reorder_rounds": 0.85,
        "mutate_retrieval": 0.70, "mutate_temperature": 0.70,
        "mutate_compression_weights": 0.60, "mutate_quality_weights": 0.55,
        "mutate_model_tiers": 0.50,
        "split_round": 0.40, "add_quality_criterion": 0.40,
    }
    return scores.get(mutation_type, 0.65)


# ── Unified Efficiency Metric ─────────────────────────────────────────────

def compute_unified_metric(quality: float, cost: float, latency_ms: int,
                           attempts: int, alpha: float = 1.0,
                           simplicity_mult: float = 1.0) -> dict[str, float]:
    """Single north-star metric: efficiency = quality * alpha * simplicity / (cost + latency_penalty)."""
    latency_penalty = latency_ms / 1000.0 * 0.0001
    raw_eff = (quality * alpha * simplicity_mult) / max(cost + latency_penalty, 1e-10)
    return {
        "efficiency": round(raw_eff, 2), "quality": round(quality, 4),
        "cost": round(cost, 4), "latencyMs": latency_ms, "attempts": attempts,
        "alpha": alpha, "simplicityMultiplier": round(simplicity_mult, 3),
    }


# ── CLI (dev/test entry point) ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Turbocontext v5 RL Engine")
    ap.add_argument("--state-path", default="~/.turbocontext/state-v5.json")
    ap.add_argument("--status", action="store_true", help="Print engine status")
    ap.add_argument("--query", action="store_true", help="Test query_optimal_params")
    ap.add_argument("--task-type", default="code_generation", choices=TASK_TYPES)
    ap.add_argument("--description", default="Write a function to sort a list")
    ap.add_argument("--capabilities", default="code_generation,algorithm_design")
    ap.add_argument("--record", action="store_true", help="Test record_trial")
    ap.add_argument("--mode", default="lite", choices=["lite", "full"])
    ap.add_argument("--sync", action="store_true", help="Run cross-context sync")
    ap.add_argument("--consolidate", action="store_true", help="Run consolidation")
    ap.add_argument("--evolve", action="store_true", help="Run evolution step")
    ap.add_argument("--verify", action="store_true", help="Run adversarial verification")
    args = ap.parse_args()

    engine = RLEngineV5(args.state_path)
    engine.load_state()

    if args.status:
        report = engine.get_status()
        print(json.dumps(engine._dataclass_to_dict(report), indent=2))

    if args.query:
        caps = [c.strip() for c in args.capabilities.split(",")]
        result = engine.query_optimal_params(
            task_type=args.task_type,
            description=args.description,
            capability_requirements=caps,
        )
        out = engine._dataclass_to_dict(result)
        # Truncate long memory lists for readability
        if "retrieved_memories" in out:
            out["retrieved_memories"] = [m.get("id", "") for m in out["retrieved_memories"]]
        print(json.dumps(out, indent=2))

    if args.record:
        caps = [c.strip() for c in args.capabilities.split(",")]
        trial = Trial(
            id=f"t_{uuid.uuid4().hex[:6]}",
            timestamp=_iso_now(),
            context="skill",
            task_type=args.task_type,
            description_hash=_sha256_trunc(args.description),
            description_length=len(args.description),
            capability_requirements=caps,
            compression_ratio=0.5,
            compression_weights=CompressionWeights(),
            temperature_schedule=[0.3, 0.5, 0.7],
            model_tier="medium",
            retrieval_top_k=5,
            token_budget_used=8000,
            outcome="success",
            quality_scores=[0.8, 0.9, 0.7, 0.85],
            quality_score=0.8125,
            retrieved_memory_ids=[],
            referenced_memory_ids=[],
        )
        res = engine.record_trial(trial, mode=args.mode)
        print(json.dumps(engine._dataclass_to_dict(res), indent=2))

    if args.sync:
        res = engine.run_cross_context_sync()
        print(json.dumps(res, indent=2))

    if args.consolidate:
        res = engine.run_consolidation()
        print(json.dumps(res, indent=2))

    if args.evolve:
        res = engine.run_evolution_step()
        print(json.dumps(res, indent=2))

    if args.verify:
        res = engine.run_adversarial_verification()
        print(json.dumps(res, indent=2))
