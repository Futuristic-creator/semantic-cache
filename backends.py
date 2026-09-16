from dataclasses import dataclass, field
import numpy as np


@dataclass
class CacheResult:
    hit: bool
    matched_id: str | None
    similarity: float
    latency_ms: float
    tier: str = "decided"


class BaseCacheBackend:
    name = "base"

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.store: dict[str, dict] = {}

    def add(self, item_id: str, text: str, embedding: np.ndarray):
        self.store[item_id] = {"embedding": embedding, "text": text}

    def query(self, text: str, embedding: np.ndarray) -> CacheResult:
        raise NotImplementedError

    def reset(self):
        self.store = {}


class NaiveCosineBaseline(BaseCacheBackend):
    name = "Naive cosine baseline"

    def query(self, text: str, embedding: np.ndarray) -> CacheResult:
        best_id, best_sim = None, -1.0
        for item_id, item in self.store.items():
            sim = float(np.dot(embedding, item["embedding"]))
            if sim > best_sim:
                best_id, best_sim = item_id, sim
        latency = 8 + 0.05 * len(self.store)
        hit = best_sim >= self.threshold
        return CacheResult(hit=hit, matched_id=best_id if hit else None,
                            similarity=best_sim, latency_ms=latency)


class GPTCacheStyleBackend(BaseCacheBackend):
    name = "GPTCache-style (LRU eviction)"

    def __init__(self, threshold: float = 0.85, max_size: int = 6):
        super().__init__(threshold)
        self.max_size = max_size
        self.access_order: list[str] = []

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
        latency = 5 + 0.02 * len(self.store)
        hit = best_sim >= self.threshold
        if hit:
            self._touch(best_id)
            return CacheResult(hit=True, matched_id=best_id, similarity=best_sim, latency_ms=latency)
        return CacheResult(hit=False, matched_id=None, similarity=best_sim, latency_ms=latency)


class RedisVectorStyleBackend(BaseCacheBackend):
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
        latency = 6 + 0.03 * len(self.store)
        hit = best_sim >= self.threshold
        return CacheResult(hit=hit, matched_id=best_id if hit else None,
                            similarity=best_sim, latency_ms=latency)


class AdaptiveThresholdBackend(BaseCacheBackend):
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


class ConfidenceTieredBackend(BaseCacheBackend):
    name = "EchoCache (Echo Check verification)"

    def __init__(self, threshold: float = 0.85, low_threshold: float | None = None):
        super().__init__(threshold)
        self.low_threshold = low_threshold if low_threshold is not None else threshold * 0.6

    def query(self, text: str, embedding: np.ndarray) -> CacheResult:
        best_id, best_sim = None, -1.0
        for item_id, item in self.store.items():
            sim = float(np.dot(embedding, item["embedding"]))
            if sim > best_sim:
                best_id, best_sim = item_id, sim
        latency = 6 + 0.03 * len(self.store)

        if best_sim >= self.threshold:
            return CacheResult(hit=True, matched_id=best_id, similarity=best_sim, latency_ms=latency, tier="auto_hit")
        if best_sim >= self.low_threshold:
            return CacheResult(hit=False, matched_id=best_id, similarity=best_sim, latency_ms=latency, tier="candidate")
        return CacheResult(hit=False, matched_id=None, similarity=best_sim, latency_ms=latency, tier="miss")


def make_backend(name: str, threshold: float):
    if name == "naive":
        return NaiveCosineBaseline(threshold=threshold)
    if name == "gptcache":
        return GPTCacheStyleBackend(threshold=threshold, max_size=16)
    if name == "redis":
        return RedisVectorStyleBackend(threshold=threshold, ttl_ticks=40)
    if name == "adaptive":
        return AdaptiveThresholdBackend(default_threshold=threshold)
    if name == "confidence_tiered":
        return ConfidenceTieredBackend(threshold=threshold)
    raise ValueError(f"unknown backend {name}")


BACKEND_KEYS = ["naive", "gptcache", "redis", "adaptive", "confidence_tiered"]
