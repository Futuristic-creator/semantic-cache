from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from dataset import load_dataset
from embedder import SharedEmbedder
from backends import make_backend
from metrics import REGISTRY, record_cache_query

app = FastAPI(
    title="Semantic Cache Benchmark API",
    description="Live comparison of semantic caching policies (naive, GPTCache-style, "
                 "Redis-style, adaptive) against a shared query.",
    version="1.0.0",
)

CANONICAL, TEST_QUERIES = load_dataset()
CORPUS_FOR_DEMO = [c["text"] for c in CANONICAL] + [t["text"] for t in TEST_QUERIES]
EMBEDDER = SharedEmbedder(CORPUS_FOR_DEMO)
DEFAULT_THRESHOLD = 0.25
BACKEND_KEYS = ["naive", "gptcache", "redis", "adaptive", "confidence_tiered"]


def _fresh_backends(threshold: float):
    backends = {key: make_backend(key, threshold) for key in BACKEND_KEYS}
    for c in CANONICAL:
        emb = EMBEDDER.embed(c["text"])
        for b in backends.values():
            b.add(c["id"], c["text"], emb)
    return backends


class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The query to test against every backend")
    threshold: float = Field(DEFAULT_THRESHOLD, ge=0.0, le=1.0, description="Similarity threshold to apply")


class BackendResult(BaseModel):
    name: str
    hit: bool
    matched_topic: Optional[str]
    similarity: float
    latency_ms: float
    tier: str


class QueryResponse(BaseModel):
    query: str
    threshold: float
    results: dict[str, BackendResult]


class CanonicalTopic(BaseModel):
    id: str
    topic: str
    text: str


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse("dashboard.html")


@app.get("/api/canonical", response_model=list[CanonicalTopic])
def canonical_list():
    return CANONICAL


@app.post("/api/query", response_model=QueryResponse)
def live_query(req: QueryRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    backends = _fresh_backends(req.threshold)
    emb = EMBEDDER.embed(text)

    results: dict[str, BackendResult] = {}
    for key, backend in backends.items():
        r = backend.query(text, emb)
        record_cache_query(key, r.hit, r.latency_ms)
        matched_text = None
        if r.matched_id:
            match = next((c for c in CANONICAL if c["id"] == r.matched_id), None)
            matched_text = match["text"] if match else None
        results[key] = BackendResult(
            name=backend.name,
            hit=r.hit,
            matched_topic=matched_text,
            similarity=round(r.similarity, 4),
            latency_ms=round(r.latency_ms, 2),
            tier=r.tier,
        )

    return QueryResponse(query=text, threshold=req.threshold, results=results)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return REGISTRY.render_prometheus_text()


@app.get("/health")
def health():
    return {"status": "ok"}
