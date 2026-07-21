// ============================================================
// Embedding Abstraction Layer (v3.2)
// ============================================================
// Pluggable semantic embedding support for the retrieval pipeline.
//
// When an EmbeddingProvider is configured, compressContext uses
// cosine similarity in embedding space to replace IDF-weighted
// keyword overlap as the primary semantic signal.
//
// Without a provider, the pipeline falls back to IDF (unchanged).
// ============================================================

// ------------------------------------------------------------------
// EmbeddingProvider Interface
// ------------------------------------------------------------------

/**
 * Embedding provider for semantic text similarity.
 *
 * Implementations can wrap:
 *   - OpenAI text-embedding-3 (small/large)
 *   - DeepSeek embeddings API
 *   - Local models via transformers.js or ollama
 *   - Any OpenAI-compatible embedding endpoint
 */
export interface EmbeddingProvider {
  /** Human-readable provider name (for logging/metrics). */
  readonly name: string;
  /** Embedding vector dimension (e.g. 1536 for text-embedding-3-small). */
  readonly dimension: number;
  /**
   * Embed a batch of texts.
   * Must return one vector per input text, in the same order.
   */
  embed(texts: string[]): Promise<number[][]>;
  /**
   * Embed a single query string.
   * Default: delegates to embed([query])[0], but providers may
   * use a different model or preprocessing for queries.
   */
  embedQuery(query: string): Promise<number[]>;
}

// ------------------------------------------------------------------
// Utility: Cosine Similarity
// ------------------------------------------------------------------

/**
 * Compute cosine similarity between two vectors.
 * Returns a value in [0, 1] for non-negative embeddings (like most
 * transformer models), or [-1, 1] in the general case.
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(
      `Vector dimension mismatch: ${a.length} vs ${b.length}`
    );
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom === 0) return 0;
  return dot / denom;
}

/**
 * Normalize similarity from [-1, 1] to [0, 1].
 * Most embedding models produce non-negative vectors so this is
 * rarely needed, but included for safety with unusual models.
 */
export function normalizeSimilarity(raw: number): number {
  return (raw + 1) / 2;
}

// ------------------------------------------------------------------
// LRU Cache
// ------------------------------------------------------------------

/**
 * Simple LRU cache for embedding vectors.
 * Avoids re-embedding identical texts within a session.
 */
class LRUCache<K, V> {
  private map = new Map<K, V>();
  private readonly maxSize: number;

  constructor(maxSize: number = 500) {
    this.maxSize = maxSize;
  }

  get(key: K): V | undefined {
    const value = this.map.get(key);
    if (value !== undefined) {
      // LRU promotion: move to end
      this.map.delete(key);
      this.map.set(key, value);
    }
    return value;
  }

  set(key: K, value: V): void {
    if (this.map.has(key)) {
      this.map.delete(key);
    } else if (this.map.size >= this.maxSize) {
      // Evict least recently used (first key)
      const first = this.map.keys().next().value;
      if (first !== undefined) this.map.delete(first);
    }
    this.map.set(key, value);
  }

  get size(): number {
    return this.map.size;
  }

  clear(): void {
    this.map.clear();
  }
}

// ------------------------------------------------------------------
// OpenAI-Compatible Embedding Provider
// ------------------------------------------------------------------

export interface OpenAIEmbeddingConfig {
  /** API key. Falls back to OPENAI_API_KEY or DEEPSEEK_API_KEY env var. */
  apiKey?: string;
  /** API base URL. Default: https://api.openai.com/v1 */
  baseUrl?: string;
  /** Model name. Default: text-embedding-3-small */
  model?: string;
  /** Max texts per batch request. Default: 20 */
  maxBatchSize?: number;
  /** Max cached embeddings. Default: 500 */
  cacheSize?: number;
  /** Request timeout in ms. Default: 30000 */
  timeoutMs?: number;
  /** Max retries on transient errors. Default: 3 */
  maxRetries?: number;
}

/**
 * OpenAI-compatible embedding provider.
 *
 * Works with:
 *   - OpenAI text-embedding-3-small / text-embedding-3-large
 *   - DeepSeek embeddings (same API format)
 *   - Any self-hosted OpenAI-compatible embedding server
 *   - Ollama with OpenAI-compatible API layer
 */
export class OpenAICompatibleEmbeddingProvider implements EmbeddingProvider {
  readonly name: string;
  readonly dimension: number;

  private readonly apiKey: string;
  private readonly baseUrl: string;
  private readonly model: string;
  private readonly maxBatchSize: number;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;
  private readonly cache: LRUCache<string, number[]>;

  // Known model dimensions (for validation)
  private static readonly KNOWN_DIMENSIONS: Record<string, number> = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
  };

  constructor(config: OpenAIEmbeddingConfig = {}) {
    this.apiKey =
      config.apiKey ||
      process.env.OPENAI_API_KEY ||
      process.env.DEEPSEEK_API_KEY ||
      "";
    this.baseUrl = config.baseUrl || "https://api.openai.com/v1";
    this.model = config.model || "text-embedding-3-small";
    this.maxBatchSize = config.maxBatchSize || 20;
    this.timeoutMs = config.timeoutMs || 30000;
    this.maxRetries = config.maxRetries ?? 3;
    this.cache = new LRUCache(config.cacheSize || 500);
    this.name = `openai:${this.model}`;
    this.dimension =
      OpenAICompatibleEmbeddingProvider.KNOWN_DIMENSIONS[this.model] || 1536;
  }

  async embed(texts: string[]): Promise<number[][]> {
    if (!this.apiKey) {
      throw new Error(
        "OpenAICompatibleEmbeddingProvider: no API key configured. " +
        "Set OPENAI_API_KEY environment variable or pass apiKey in config."
      );
    }

    // Check cache first
    const results: Array<number[] | null> = texts.map(t => {
      const cached = this.cache.get(t);
      return cached ?? null;
    });

    // Find uncached indices
    const uncachedIndices: number[] = [];
    const uncachedTexts: string[] = [];
    for (let i = 0; i < texts.length; i++) {
      if (results[i] === null) {
        uncachedIndices.push(i);
        uncachedTexts.push(texts[i]);
      }
    }

    // Batch-embed uncached texts
    for (let i = 0; i < uncachedTexts.length; i += this.maxBatchSize) {
      const batch = uncachedTexts.slice(i, i + this.maxBatchSize);
      const embeddings = await this.embedBatch(batch);
      for (let j = 0; j < batch.length; j++) {
        const idx = uncachedIndices[i + j];
        results[idx] = embeddings[j];
        this.cache.set(batch[j], embeddings[j]);
      }
    }

    return results as number[][];
  }

  async embedQuery(query: string): Promise<number[]> {
    const results = await this.embed([query]);
    return results[0];
  }

  private async embedBatch(texts: string[]): Promise<number[][]> {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), this.timeoutMs);

        const response = await fetch(
          `${this.baseUrl}/embeddings`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "Authorization": `Bearer ${this.apiKey}`,
            },
            body: JSON.stringify({
              model: this.model,
              input: texts,
            }),
            signal: controller.signal,
          }
        );

        clearTimeout(timeout);

        if (!response.ok) {
          const body = await response.text().catch(() => "");
          const err = new Error(
            `Embedding API error ${response.status}: ${body.slice(0, 200)}`
          );
          // Don't retry on 4xx (except 429 rate limit)
          if (response.status !== 429 && response.status < 500) {
            throw err;
          }
          lastError = err;
          continue;
        }

        const data = await response.json() as {
          data: Array<{ embedding: number[] }>;
        };

        // Sort by index (API may return out of order)
        return data.data
          .sort((a: any, b: any) => (a.index || 0) - (b.index || 0))
          .map((item: { embedding: number[] }) => item.embedding);

      } catch (err) {
        if ((err as Error).name === "AbortError") {
          lastError = new Error(`Embedding API timeout after ${this.timeoutMs}ms`);
        } else {
          lastError = err as Error;
        }
      }

      // Exponential backoff
      if (attempt < this.maxRetries) {
        const delay = Math.min(1000 * Math.pow(2, attempt), 8000);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    throw lastError || new Error("Embedding API: unknown error after retries");
  }

  /** Clear the embedding cache. */
  clearCache(): void {
    this.cache.clear();
  }

  /** Get cache statistics for monitoring. */
  getCacheStats(): { size: number; maxSize: number } {
    return { size: this.cache.size, maxSize: 500 };
  }
}

// ------------------------------------------------------------------
// Semantic Matcher Factory
// ------------------------------------------------------------------

/**
 * Create an async semantic matcher from an embedding provider.
 *
 * Returns a match() function that scores candidate texts against
 * a query, returning cosine similarity in [0, 1].
 *
 * Usage:
 *   const matcher = createSemanticMatcher(provider);
 *   const scores = await matcher.match("security review", ["auth.ts code...", "utils.ts code..."]);
 *   // scores = [0.87, 0.32]
 */
export function createSemanticMatcher(provider: EmbeddingProvider): {
  match(query: string, candidates: string[]): Promise<number[]>;
  dispose(): void;
} {
  return {
    async match(query: string, candidates: string[]): Promise<number[]> {
      if (candidates.length === 0) return [];

      const queryEmbedding = await provider.embedQuery(query);
      const candidateEmbeddings = await provider.embed(candidates);

      return candidateEmbeddings.map(emb => {
        const raw = cosineSimilarity(queryEmbedding, emb);
        return normalizeSimilarity(raw);
      });
    },
    dispose(): void {
      // No-op for API-based providers. Local model providers
      // could override to free resources.
    },
  };
}

// ------------------------------------------------------------------
// No-Op Provider (throws on use, for unconfigured setups)
// ------------------------------------------------------------------

/**
 * No-op embedding provider.
 *
 * Throws a descriptive error when embed() is called, guiding the
 * user to configure a real provider.  Used as the default when
 * no embedding backend is configured.
 */
export class NoOpEmbeddingProvider implements EmbeddingProvider {
  readonly name = "noop";
  readonly dimension = 0;

  async embed(_texts: string[]): Promise<number[][]> {
    throw new Error(
      "No embedding provider configured. " +
      "Set up an OpenAICompatibleEmbeddingProvider or implement " +
      "your own EmbeddingProvider. See docs for details."
    );
  }

  async embedQuery(_query: string): Promise<number[]> {
    throw new Error(
      "No embedding provider configured. " +
      "Set up an OpenAICompatibleEmbeddingProvider or implement " +
      "your own EmbeddingProvider."
    );
  }
}
