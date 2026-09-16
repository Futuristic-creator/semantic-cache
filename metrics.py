from collections import defaultdict
from threading import Lock


class MetricsRegistry:
    def __init__(self):
        self._lock = Lock()
        self._counters: dict[tuple, int] = defaultdict(int)
        self._histograms: dict[tuple, list] = defaultdict(list)

    def inc_counter(self, name: str, labels: dict | None = None, value: int = 1):
        key = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            self._counters[key] += value

    def observe_histogram(self, name: str, value_ms: float, labels: dict | None = None):
        key = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            self._histograms[key].append(value_ms)

    def render_prometheus_text(self) -> str:
        lines = []
        with self._lock:
            counter_names = {k[0] for k in self._counters}
            for name in sorted(counter_names):
                lines.append(f"# TYPE {name} counter")
                for (n, labels), value in sorted(self._counters.items()):
                    if n != name:
                        continue
                    label_str = _format_labels(labels)
                    lines.append(f"{name}{label_str} {value}")

            hist_names = {k[0] for k in self._histograms}
            for name in sorted(hist_names):
                lines.append(f"# TYPE {name} summary")
                for (n, labels), values in sorted(self._histograms.items()):
                    if n != name or not values:
                        continue
                    label_str = _format_labels(labels)
                    count = len(values)
                    total = sum(values)
                    avg = total / count
                    p95 = sorted(values)[int(0.95 * (count - 1))]
                    lines.append(f"{name}_count{label_str} {count}")
                    lines.append(f"{name}_sum{label_str} {total:.3f}")
                    lines.append(f"{name}_avg{label_str} {avg:.3f}")
                    lines.append(f"{name}_p95{label_str} {p95:.3f}")
        return "\n".join(lines) + "\n"


def _format_labels(labels: tuple) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in labels)
    return "{" + inner + "}"


REGISTRY = MetricsRegistry()


def record_cache_query(backend: str, hit: bool, latency_ms: float):
    REGISTRY.inc_counter("semcache_queries_total", {"backend": backend, "result": "hit" if hit else "miss"})
    REGISTRY.observe_histogram("semcache_query_latency_ms", latency_ms, {"backend": backend})
