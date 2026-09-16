import unittest
from metrics import MetricsRegistry, record_cache_query, REGISTRY


class TestMetricsRegistry(unittest.TestCase):
    def setUp(self):
        self.reg = MetricsRegistry()

    def test_counter_increments(self):
        self.reg.inc_counter("hits_total", {"backend": "naive"})
        self.reg.inc_counter("hits_total", {"backend": "naive"})
        text = self.reg.render_prometheus_text()
        self.assertIn('hits_total{backend="naive"} 2', text)

    def test_counter_separates_by_labels(self):
        self.reg.inc_counter("hits_total", {"backend": "naive"})
        self.reg.inc_counter("hits_total", {"backend": "redis"})
        text = self.reg.render_prometheus_text()
        self.assertIn('hits_total{backend="naive"} 1', text)
        self.assertIn('hits_total{backend="redis"} 1', text)

    def test_histogram_reports_count_and_avg(self):
        for v in [10, 20, 30]:
            self.reg.observe_histogram("latency_ms", v, {"backend": "naive"})
        text = self.reg.render_prometheus_text()
        self.assertIn('latency_ms_count{backend="naive"} 3', text)
        self.assertIn('latency_ms_avg{backend="naive"} 20.000', text)

    def test_output_is_valid_prometheus_text_shape(self):
        self.reg.inc_counter("x")
        text = self.reg.render_prometheus_text()
        self.assertTrue(text.startswith("# TYPE"))

    def test_record_cache_query_helper_uses_shared_registry(self):
        before = REGISTRY.render_prometheus_text()
        record_cache_query("adaptive", hit=True, latency_ms=5.0)
        after = REGISTRY.render_prometheus_text()
        self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
