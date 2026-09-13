"""
Unit tests for the semantic caching benchmark.

Uses stdlib unittest rather than pytest (kept dependency-free on
purpose -- this project intentionally has the smallest possible
dependency footprint: numpy + scikit-learn only).

Run with:
    python3 -m unittest test_backends.py -v
"""
import unittest
import numpy as np

from embedder import SharedEmbedder
from backends import (
    NaiveCosineBaseline,
    GPTCacheStyleBackend,
    RedisVectorStyleBackend,
    AdaptiveThresholdBackend,
    make_backend,
    BACKEND_KEYS,
)


class TestSharedEmbedder(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "how do I reset my password",
            "how do I cancel my subscription",
            "what is the weather today",
        ]
        self.embedder = SharedEmbedder(self.corpus)

    def test_embeddings_are_normalized(self):
        vec = self.embedder.embed("how do I reset my password")
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_identical_text_has_similarity_one(self):
        text = "how do I reset my password"
        a = self.embedder.embed(text)
        b = self.embedder.embed(text)
        sim = SharedEmbedder.cosine_similarity(a, b)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_unrelated_text_has_low_similarity(self):
        a = self.embedder.embed("how do I reset my password")
        b = self.embedder.embed("what is the weather today")
        sim = SharedEmbedder.cosine_similarity(a, b)
        self.assertLess(sim, 0.3)

    def test_zero_vector_handled_without_error(self):
        # a string with no vocabulary overlap at all should not crash
        # on normalization (division by zero guard in embed())
        vec = self.embedder.embed("zzz qqq xxx")
        self.assertFalse(np.isnan(vec).any())


class TestNaiveCosineBaseline(unittest.TestCase):
    def setUp(self):
        corpus = ["how do I reset my password", "how do I cancel my subscription"]
        self.embedder = SharedEmbedder(corpus)
        self.backend = NaiveCosineBaseline(threshold=0.5)
        self.backend.add("c1", "how do I reset my password",
                          self.embedder.embed("how do I reset my password"))

    def test_exact_match_is_a_hit(self):
        emb = self.embedder.embed("how do I reset my password")
        result = self.backend.query("how do I reset my password", emb)
        self.assertTrue(result.hit)
        self.assertEqual(result.matched_id, "c1")

    def test_unrelated_query_is_a_miss(self):
        emb = self.embedder.embed("how do I cancel my subscription")
        result = self.backend.query("how do I cancel my subscription", emb)
        self.assertFalse(result.hit)

    def test_empty_cache_never_hits(self):
        empty_backend = NaiveCosineBaseline(threshold=0.1)
        emb = self.embedder.embed("anything")
        result = empty_backend.query("anything", emb)
        self.assertFalse(result.hit)
        self.assertIsNone(result.matched_id)

    def test_no_eviction_cache_grows_unbounded(self):
        for i in range(20):
            self.backend.add(f"extra{i}", f"filler text {i}",
                              self.embedder.embed(f"filler text {i}"))
        self.assertEqual(len(self.backend.store), 21)  # original + 20


class TestGPTCacheStyleBackend(unittest.TestCase):
    def test_lru_eviction_removes_least_recently_used(self):
        corpus = [f"topic {i}" for i in range(10)]
        embedder = SharedEmbedder(corpus)
        backend = GPTCacheStyleBackend(threshold=0.99, max_size=3)
        for i in range(3):
            backend.add(f"id{i}", f"topic {i}", embedder.embed(f"topic {i}"))
        self.assertEqual(len(backend.store), 3)
        # adding a 4th entry should evict the least-recently-used one (id0,
        # since it was added first and never touched again)
        backend.add("id3", "topic 3", embedder.embed("topic 3"))
        self.assertEqual(len(backend.store), 3)
        self.assertNotIn("id0", backend.store)
        self.assertIn("id3", backend.store)

    def test_cache_hit_refreshes_recency(self):
        corpus = [f"topic {i}" for i in range(10)]
        embedder = SharedEmbedder(corpus)
        backend = GPTCacheStyleBackend(threshold=0.5, max_size=2)
        backend.add("id0", "topic 0", embedder.embed("topic 0"))
        backend.add("id1", "topic 1", embedder.embed("topic 1"))
        # touch id0 via a query so it becomes most-recently-used
        backend.query("topic 0", embedder.embed("topic 0"))
        backend.add("id2", "topic 2", embedder.embed("topic 2"))
        # id1 should be evicted now, not id0, since id0 was just touched
        self.assertNotIn("id1", backend.store)
        self.assertIn("id0", backend.store)


class TestRedisVectorStyleBackend(unittest.TestCase):
    def test_entries_expire_after_ttl_ticks(self):
        corpus = ["topic a", "topic b"]
        embedder = SharedEmbedder(corpus)
        backend = RedisVectorStyleBackend(threshold=0.5, ttl_ticks=3)
        backend.add("id0", "topic a", embedder.embed("topic a"))
        # each query() call advances the internal tick counter
        for _ in range(5):
            backend.query("topic b", embedder.embed("topic b"))
        self.assertNotIn("id0", backend.store)  # should have expired by now


class TestAdaptiveThresholdBackend(unittest.TestCase):
    def test_per_entry_threshold_overrides_default(self):
        corpus = ["topic a", "topic b"]
        embedder = SharedEmbedder(corpus)
        backend = AdaptiveThresholdBackend(default_threshold=0.9)
        backend.add("strict", "topic a", embedder.embed("topic a"), threshold=0.99)
        backend.add("lenient", "topic b", embedder.embed("topic b"), threshold=0.01)
        self.assertEqual(backend.per_entry_threshold["strict"], 0.99)
        self.assertEqual(backend.per_entry_threshold["lenient"], 0.01)


class TestBackendFactory(unittest.TestCase):
    def test_all_backend_keys_are_constructible(self):
        for key in BACKEND_KEYS:
            backend = make_backend(key, threshold=0.5)
            self.assertIsNotNone(backend)

    def test_unknown_backend_key_raises(self):
        with self.assertRaises(ValueError):
            make_backend("not_a_real_backend", threshold=0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
