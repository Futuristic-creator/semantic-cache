"""
Cache backend adapters.

Common interface every backend implements:

    add(id, text, embedding)          -> populate the cache
    query(text, embedding) -> CacheResult

Each backend reimplements the actual CACHING POLICY of a real product
(similarity matching strategy + eviction strategy), running on top of
the SAME shared embedding vectors, so the benchmark isolates policy
differences rather than embedding differences.

This is a local, dependency-free reimplementation of each product's
documented default behavior -- not the literal SDK (no internet
access in this sandbox to install gptcache / redis / a vector DB).
Swapping in the real SDK later means replacing the internals of one
class while keeping the same interface; the benchmark harness doesn't
change.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CacheResult:
    hit: bool
    matched_id: str | None
    similarity: float
    latency_ms: float


class BaseCacheBackend:
    name = "base"

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.store: dict[str, dict] = {}  # id -> {embedding, text}

    def add(self, item_id: str, text: str, embedding: np.ndarray):
        self.store[item_id] = {"embedding": embedding, "text": text}

    def query(self, text: str, embedding: np.ndarray) -> CacheResult:
        raise NotImplementedError

    def reset(self):
        self.store = {}


class NaiveCosineBaseline(BaseCacheBackend):
    """
    The 'why do I even need a library' control: linear scan, cosine
    similarity, static global threshold, top-1 match, no eviction.
    This is what a team builds in an afternoon before reaching for
    a real caching library -- useful as the floor to beat.
    """
    name = "Naive cosine baseline"

    def query(self, text: str, embedding: np.ndarray) -> CacheResult:
        # simulated cost: linear scan over the whole cache (O(n)),
        # no index structure -- latency grows with cache size
        best_id, best_sim = None, -1.0
        for item_id, item in self.store.items():
            sim = float(np.dot(embedding, item["embedding"]))
            if sim > best_sim:
                best_id, best_sim = item_id, sim
        latency = 8 + 0.05 * len(self.store)  # ms, linear scan cost
        hit = best_sim >= self.threshold
        return CacheResult(hit=hit, matched_id=best_id if hit else None,
                            similarity=best_sim, latency_ms=latency)


class GPTCacheStyleBackend(BaseCacheBackend):
    """
    Mirrors GPTCache's default architecture: vector similarity search
    + a similarity-evaluation step against a threshold, backed by a
    bounded cache with LRU eviction (GPTCache's default eviction
    policy is LRU-based via its cache manager).
    """
    name = "GPTCache-style (LRU eviction)"

    def __init__(self, threshold: float = 0.85, max_size: int = 6):
        super().__init__(threshold)
        self.max_size = max_size
        self.access_order: list[str] = []  # front = most recently used

    def add(self, item_id: str, text: str, embedding: np.ndarray):
        super().add(item_id, text, embedding)
        self._touch(item_id)
        self._evict_if_needed()

    def _touch(self, item_id: str):
        if item_id in self.access_order:
            self.access_order.remove(item_id)
        self.access_order.insert(0, item_id)

    def _evict_if_needed(self):
        while len(self.store) > self.max_size:
            lru_id = self.access_order.pop()
            self.store.pop(lru_id, None)

    def query(self, text: str, embedding: np.ndarray) -> CacheResult:
        best_id, best_sim = None, -1.0
        for item_id, item in self.store.items():
            sim = float(np.dot(embedding, item["embedding"]))
            if sim > best_sim:
                best_id, best_sim = item_id, sim
        latency = 5 + 0.02 * len(self.store)  # ms, slightly better than naive (indexed-ish)
        hit = best_sim >= self.threshold
        if hit:
            self._touch(best_id)  # a cache hit counts as access -> refresh recency
            return CacheResult(hit=True, matched_id=best_id, similarity=best_sim, latency_ms=latency)
        return CacheResult(hit=False, matched_id=None, similarity=best_sim, latency_ms=latency)


class RedisVectorStyleBackend(BaseCacheBackend):
    """
    Mirrors a Redis-vector semantic cache: entries carry a TTL and
    are evicted on expiry rather than on access recency (Redis's
    common semantic-cache pattern uses TTL-based expiry, not LRU).
    Time is simulated as a monotonically increasing 'tick' counter
    (one tick per query processed) rather than wall-clock time.
    """
    name = "Redis-vector-style (TTL eviction)"

    def __init__(self, threshold: float = 0.85, ttl_ticks: int = 40):
        super().__init__(threshold)
        self.ttl_ticks = ttl_ticks
        self.tick = 0
        self.expiry: dict[str, int] = {}

    def add(self, item_id: str, text: str, embedding: np.ndarray):
        super().add(item_id, text, embedding)
        self.expiry[item_id] = self.tick + self.ttl_ticks

    def _advance_and_expire(self):
        self.tick += 1
        expired = [i for i, exp in self.expiry.items() if exp <= self.tick]
        for i in expired:
            self.store.pop(i, None)
            self.expiry.pop(i, None)

    def query(self, text: str, embedding: np.ndarray) -> CacheResult:
        self._advance_and_expire()
        best_id, best_sim = None, -1.0
        for item_id, item in self.store.items():
            sim = float(np.dot(embedding, item["embedding"]))
            if sim > best_sim:
                best_id, best_sim = item_id, sim
        latency = 6 + 0.03 * len(self.store)  # ms, indexed vector search in Redis
        hit = best_sim >= self.threshold
        return CacheResult(hit=hit, matched_id=best_id if hit else None,
                            similarity=best_sim, latency_ms=latency)


class AdaptiveThresholdBackend(BaseCacheBackend):
    """
    Simplified illustration of the 'adaptive per-prompt threshold'
    idea from recent semantic-caching research (e.g. vCache-style
    systems): instead of one static global threshold, each cached
    entry gets its OWN threshold, calibrated from labeled examples
    seen during a calibration pass, then frozen for evaluation.

    This is a deliberately simplified stand-in for online bandit-style
    threshold learning -- it shows the *shape* of the idea (per-entry
    thresholds beat one global threshold when topics vary in how
    'risky' their paraphrase space is) without implementing the full
    online-learning algorithm from the paper.
    """
    name = "Adaptive per-entry threshold"

    def __init__(self, default_threshold: float = 0.85):
        super().__init__(default_threshold)
        self.per_entry_threshold: dict[str, float] = {}

    def add(self, item_id: str, text: str, embedding: np.ndarray, threshold: float | None = None):
        super().add(item_id, text, embedding)
        self.per_entry_threshold[item_id] = threshold if threshold is not None else self.threshold

    def query(self, text: str, embedding: np.ndarray) -> CacheResult:
        best_id, best_sim, best_margin = None, -1.0, -1.0
        for item_id, item in self.store.items():
            sim = float(np.dot(embedding, item["embedding"]))
            entry_thresh = self.per_entry_threshold.get(item_id, self.threshold)
            margin = sim - entry_thresh
            if margin > best_margin:
                best_id, best_sim, best_margin = item_id, sim, margin
        latency = 6 + 0.03 * len(self.store)
        hit = best_margin >= 0
        return CacheResult(hit=hit, matched_id=best_id if hit else None,
                            similarity=best_sim, latency_ms=latency)


def make_backend(name: str, threshold: float):
    if name == "naive":
        return NaiveCosineBaseline(threshold=threshold)
    if name == "gptcache":
        return GPTCacheStyleBackend(threshold=threshold, max_size=16)
    if name == "redis":
        return RedisVectorStyleBackend(threshold=threshold, ttl_ticks=40)
    if name == "adaptive":
        return AdaptiveThresholdBackend(default_threshold=threshold)
    raise ValueError(f"unknown backend {name}")


BACKEND_KEYS = ["naive", "gptcache", "redis", "adaptive"]
