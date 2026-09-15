# Semantic Caching Benchmark — Architecture Decision Record

## Context

Every team running LLMs in production eventually reaches for semantic
caching to cut inference cost and latency: instead of calling the LLM
for every request, check whether a semantically similar request has
already been answered, and serve the cached answer if so.

The pitch is simple. The failure mode is not: a cache that's too
aggressive serves the **wrong cached answer** to a query that only
*looks* similar to something already cached ("cancel my subscription"
vs "cancel my subscription renewal reminder"). This benchmark exists
to make that tradeoff visible and measurable, and to compare how
different caching *policies* — not just different libraries — handle
it.

## What this is (and isn't)

- **Embeddings**: a shared TF-IDF vectorizer (scikit-learn), fit once
  on the full corpus and used identically by every backend. This
  isolates the thing actually being compared — caching *policy* — from
  embedding quality. In a networked environment, swap `embedder.py`
  for a real model (OpenAI/Voyage/sentence-transformers); nothing else
  changes.
- **Backends**: faithful reimplementations of each product's
  documented default *policy*, not the literal SDKs:
  - **Naive cosine baseline** — linear scan, static threshold, no
    eviction. The "what a team builds in an afternoon" control.
  - **GPTCache-style** — similarity search + threshold, bounded cache
    with **LRU eviction** (GPTCache's default cache-manager behavior).
  - **Redis-vector-style** — vector search + threshold, **TTL-based
    eviction** (the common pattern for Redis semantic caches).
  - **Adaptive per-entry threshold** — a simplified stand-in for
    recent "adaptive threshold" research (e.g. vCache-style systems):
    instead of one global threshold, each cached entry gets its own
    threshold, calibrated from a small labeled calibration slice.
- **Dataset**: 10 hand-curated "customer support" topics (password
  reset, cancel subscription, refunds, etc.), each with an exact
  duplicate, 3 paraphrases, and 2 **traps** — queries with high lexical
  overlap but different intent — plus 10 unrelated control queries.
  70 test queries total. This is small by design: built for a clear,
  inspectable demo and ADR
  
- **Cost/latency model**: illustrative, configurable constants
  (`LLM_CALL_COST_USD = $0.0015`, `LLM_CALL_LATENCY_MS = 800`), not
  measured production invoices. Swap in your own numbers in
  `benchmark.py`.
- **Realism fix worth calling out**: every cache MISS in the benchmark
  triggers a simulated LLM call *and writes the new answer back into
  the cache* — exactly like a real semantic cache. Without this, a
  bounded/TTL cache would never experience real eviction pressure and
  would look artificially better than it would in production.

## Key findings

1. **At a typical operating threshold (0.25), every non-adaptive
   backend served a wrong cached answer for effectively all 20 trap
   queries** (naive: 20/20 false hits, GPTCache-style: 19/20,
   Redis-style: 19/20). The adaptive backend correctly rejected 6/20 —
   a real improvement, but far from solved. **The trap-query failure
   mode is not a fringe edge case in this benchmark — at a normal
   threshold, it's the default outcome.**
2. **Eviction policy affects correctness, not just capacity.** Once
   the benchmark writes new answers back on every miss (see above),
   the GPTCache-style (LRU) backend's precision *collapses* at
   moderate thresholds (down to ~0.27 at t=0.30) — worse than the
   naive baseline. Evicting by recency rather than by quality means a
   growing pool of low-value cached entries (including cached traps)
   increases false-hit risk over time. This is a genuine argument for
   pairing eviction with some notion of entry quality, not just
   recency or TTL.
3. **The naive unbounded baseline has the best raw F1 at low
   thresholds — and that's a trap in itself.** It's not a viable
   production choice regardless: an unbounded cache grows forever.
   Benchmarks like this one are a snapshot; they don't capture
   operational constraints (memory growth, cost of the index itself)
   that make "unbounded" a non-starter at scale. Don't let a
   leaderboard number override that.
4. **The adaptive backend trades peak performance for stability.** It
   doesn't top every metric, but its hit-rate/precision curve is flat
   across the entire threshold sweep — because it isn't using the
   swept threshold at all for calibrated entries. That's the practical
   value proposition: less manual threshold-tuning babysitting, at the
   cost of needing a calibration/feedback loop to set per-entry
   thresholds in the first place (here, simplified to a one-off
   labeled calibration pass; a real system would do this online from
   production feedback signals).

## Decision matrix

| Situation | Recommendation | Why |
|---|---|---|
| Prototyping, low query volume, cache is small and short-lived | Naive baseline | Simplicity wins; eviction/precision issues haven't kicked in yet at small scale |
| High query volume, cost-sensitive, tolerant of occasional wrong answers (e.g. internal tooling) | GPTCache-style (LRU) | Bounded memory footprint; accept the precision tradeoff explicitly, monitor false-hit rate |
| Multi-tenant or compliance-sensitive (e.g. fintech, healthcare) | Avoid pure similarity-threshold caching, or use it only behind an answer-verification step | False hits here are a correctness/compliance incident, not just an inconvenience |
| Long-running production system with real traffic feedback available | Adaptive per-entry threshold | Justifies the calibration investment; reduces manual tuning burden over time |
| Anything customer-facing where a wrong cached answer is costly | Add a verification/judge layer on top of ANY of these | This benchmark shows similarity threshold alone — at any of the policies tested — is not sufficient by itself for high-stakes correctness |


### A note on `app.py` vs `app_fastapi.py`

This repo intentionally ships two API layers, and it's worth being
direct about why: current AI-infra job postings consistently name
**FastAPI** specifically (async, Pydantic-validated requests, an
auto-generated `/docs` page) rather than Flask, so `app_fastapi.py`
is written to that spec. But it was built in a sandbox with no
internet access to install `fastapi`/`uvicorn`, so **it has not been
runtime-tested** — only syntax-checked. `app.py` (Flask) is the
version that was actually run and verified end-to-end, including the
live `/api/query` trap-query demo shown in this README.

Before using `app_fastapi.py` in an interview or demo:
```bash
pip install -r requirements-fastapi.txt
uvicorn app_fastapi:app --reload
open http://localhost:8000/docs
```
Both versions share the exact same `metrics.py` observability layer
and the same backend/embedder logic underneath — only the web
framework differs.

## Observability

`/metrics` (on both API versions) exposes Prometheus-compatible text
output: query counts by backend and result (hit/miss), and latency
histograms (count/sum/avg/p95) by backend. Point a real Prometheus at
this in production and alert on hit-rate and latency drift — this is
the piece most teams skip (see the "production reliability" discussion
in this project's broader context) and it's fully unit-tested in
`test_metrics.py` independent of which web framework is running it.

## Containerization & CI

`Dockerfile` builds `app_fastapi.py` into a container with a
`/health` liveness endpoint (the kind of thing a Kubernetes probe
would hit). `.github/workflows/ci.yml` runs the full test suite,
regenerates the benchmark, and rebuilds the dashboard on every push —
neither is build/run-tested in this offline sandbox (no network to
pull the base image or install packages), so verify locally with
`docker build .` before relying on either.

## How to run

```bash
pip install -r requirements.txt

python3 benchmark.py            # writes results.json, prints a quick summary
python3 build_dashboard.py      # regenerates dashboard.html from results.json
python3 -m unittest test_backends.py -v   # run the test suite (14 tests)

python3 app.py                  # live demo: http://localhost:5000
```

`dashboard.html` can also be opened directly as a static file (no
server) for the charts/tables — only the "Try it yourself" live-query
panel needs `app.py` running, since that panel calls a real API rather
than reading precomputed results.

## Testing

19 unit tests (14 backend/embedder + 5 metrics) cover: embedding
normalization and similarity behavior, each backend's hit/miss logic,
LRU eviction order (GPTCache-style), TTL expiry (Redis-style),
per-entry threshold overrides (adaptive), the backend factory, and
the Prometheus metrics registry. Run with:
```bash
python3 -m unittest discover -p "test_*.py" -v
```

## How to extend toward a real (networked) environment

- Replace `embedder.py`'s TF-IDF vectorizer with a real embedding
  model call.
- Replace each backend's internal `store`/`query` with a real SDK
  call (GPTCache client, Redis `FT.SEARCH`, a hosted vector DB) behind
  the same `add()` / `query()` interface — the benchmark harness in
  `benchmark.py` does not need to change.
- Swap the hand-curated dataset for a QQP/STS-B sample plus your own
  production trap-query set once you have real traffic to mine.
- Replace the illustrative cost/latency constants with your actual
  measured API costs and p50/p99 latencies.
