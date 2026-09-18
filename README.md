# Semantic Caching Benchmark

Compares semantic caching **policies** on cost, latency, and precision — specifically the failure mode most demos skip: serving the wrong cached answer.

## Scope: 

- **Retrieval algorithm**: not the point, and not what's being tested. Production retrieval should use Elasticsearch/OpenSearch (HNSW) or pgvector — commodity, solved, actively optimized by people who do that full-time. This project uses TF-IDF + linear scan only because the build sandbox had no internet to install a real vector DB or embedding model.
- **Scope**: the policy layer on top of retrieval — threshold calibration, eviction strategy, and verification tradeoffs. That layer is retrieval-engine-agnostic; swap the embedder, nothing else changes.

## Backends

| Backend | Policy |
|---|---|
| Naive | Linear scan, static threshold, no eviction |
| GPTCache-style | LRU eviction, bounded cache |
| Redis-style | TTL eviction |
| Adaptive | Per-entry threshold, calibrated from labeled data |
| EchoCache | 3-band: hit / Echo Check (verify) / miss — not binary |

## Key findings

1. **At a typical threshold, every non-tiered backend served a wrong answer on ~19-20 of 20 trap queries** (naive 20/20, GPTCache-style 19/20, Redis-style 19/20). Traps: high lexical overlap, different intent — e.g. "cancel my subscription" vs "cancel my subscription renewal reminder."
2. **Eviction policy affects correctness, not just capacity.** Once cache writes-back on every miss (realistic), GPTCache-style's LRU eviction lets low-quality entries persist by recency, and precision drops below the naive baseline at moderate thresholds.
3. **EchoCache improves aggregate precision (~0.50 vs ~0.40-0.49) and cost savings, but does not fix high-lexical-overlap traps.** A trap with genuinely high textual similarity lands in the "hit" band, same as naive — Echo Check only rescues *borderline*-similarity cases. At threshold 0.25 it still misses 20/20 traps. No threshold-based method fully eliminates false-hit risk; only ~10-16 of 70 queries per threshold actually trigger an Echo Check, so cost stays low while precision improves where it can.
4. **Unbounded (naive) cache has the best raw F1 at low thresholds — and that's not a viable production choice regardless**, since it grows forever. Don't chase the benchmark number over the operational constraint.

<img width="1047" height="501" alt="Screenshot 2026-09-18 at 12 10 56" src="https://github.com/user-attachments/assets/ea98285d-81bd-44bc-9ec8-fa9139f86ee5" />


```

## Run it

```bash
pip install -r requirements.txt
python3 benchmark.py
python3 build_dashboard.py
python3 -m unittest discover -p "test_*.py" -v
python3 app.py
```

FastAPI version: `pip install -r requirements-fastapi.txt && uvicorn app_fastapi:app --reload` — written to spec, verify before relying on it in an interview.

## Extending toward production

Swap `embedder.py` for a real embedding model. Swap each backend's `store`/`query` internals for a real Elasticsearch/pgvector call behind the same interface — `benchmark.py` doesn't change. Replace the illustrative cost constants with real measured numbers.


Dashboards rendered by project , helps in decision making

<img width="1357" height="702" alt="Screenshot 2026-09-18 at 12 06 04" src="https://github.com/user-attachments/assets/c7e4be33-3436-4271-912e-d2818de6f239" />


<img width="1221" height="576" alt="Screenshot 2026-09-18 at 12 06 25" src="https://github.com/user-attachments/assets/0b105c23-563a-4b2e-861f-0439291990b2" />

<img width="1159" height="557" alt="Screenshot 2026-09-18 at 12 06 41" src="https://github.com/user-attachments/assets/37a67905-2a2b-42da-8a72-5345a9c6ecda" />

<img width="1290" height="733" alt="Screenshot 2026-09-18 at 12 07 14" src="https://github.com/user-attachments/assets/308dd683-c177-4cbb-a19c-13c5fe767d7b" />

<img width="1216" height="660" alt="Screenshot 2026-09-18 at 12 07 36" src="https://github.com/user-attachments/assets/b1de8486-91e6-469c-84c1-18b84e7940d4" />

<img width="1145" height="588" alt="Screenshot 2026-09-18 at 12 08 24" src="https://github.com/user-attachments/assets/58731942-e9df-4ed1-91e8-97c92d44ced8" />






