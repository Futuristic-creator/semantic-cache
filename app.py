from flask import Flask, jsonify, request, send_from_directory

from dataset import load_dataset
from embedder import SharedEmbedder
from backends import make_backend
from metrics import REGISTRY, record_cache_query

app = Flask(__name__, static_folder=".")

CANONICAL, TEST_QUERIES = load_dataset()
CORPUS_FOR_DEMO = [c["text"] for c in CANONICAL] + [t["text"] for t in TEST_QUERIES]
EMBEDDER = SharedEmbedder(CORPUS_FOR_DEMO)


def _fresh_backends(threshold: float):
    backends = {key: make_backend(key, threshold) for key in ["naive", "gptcache", "redis", "adaptive", "confidence_tiered"]}
    for c in CANONICAL:
        emb = EMBEDDER.embed(c["text"])
        for b in backends.values():
            b.add(c["id"], c["text"], emb)
    return backends


DEFAULT_THRESHOLD = 0.25
LIVE_BACKENDS = _fresh_backends(DEFAULT_THRESHOLD)


@app.route("/")
def dashboard():
    return send_from_directory(".", "dashboard.html")


@app.route("/api/canonical")
def canonical_list():
    return jsonify(CANONICAL)


@app.route("/api/query", methods=["POST"])
def live_query():
    body = request.get_json(force=True)
    text = (body or {}).get("text", "").strip()
    threshold = float((body or {}).get("threshold", DEFAULT_THRESHOLD))
    if not text:
        return jsonify({"error": "text is required"}), 400

    backends = _fresh_backends(threshold)
    emb = EMBEDDER.embed(text)

    results = {}
    for key, backend in backends.items():
        r = backend.query(text, emb)
        record_cache_query(key, r.hit, r.latency_ms)
        matched_text = None
        if r.matched_id:
            match = next((c for c in CANONICAL if c["id"] == r.matched_id), None)
            matched_text = match["text"] if match else None
        results[key] = {
            "name": backend.name,
            "hit": r.hit,
            "matched_topic": matched_text,
            "similarity": round(r.similarity, 4),
            "latency_ms": round(r.latency_ms, 2),
            "tier": r.tier,
        }
    return jsonify({"query": text, "threshold": threshold, "results": results})


@app.route("/metrics")
def metrics():
    return REGISTRY.render_prometheus_text(), 200, {"Content-Type": "text/plain; version=0.0.4"}


if __name__ == "__main__":
    print("Semantic cache demo running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
