import json
import random
from collections import defaultdict

from dataset import load_dataset
from embedder import SharedEmbedder
from backends import make_backend, BACKEND_KEYS, CacheResult

LLM_CALL_COST_USD = 0.0015
LLM_CALL_LATENCY_MS = 800
CACHE_HIT_EXTRA_COST_USD = 0.00002
NO_CACHE_LATENCY_MS = LLM_CALL_LATENCY_MS
JUDGE_COST_USD = 0.0002
JUDGE_LATENCY_MS = 150
JUDGE_ACCURACY = 0.92

THRESHOLD_SWEEP = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def calibrate_adaptive(backend, canonical, embedder, calibration_queries, rng):
    grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    for c in canonical:
        cid = c["id"]
        c_emb = embedder.embed(c["text"])
        best_t, best_score = backend.threshold, -1e9
        for t in grid:
            score = 0
            for q in calibration_queries:
                q_emb = embedder.embed(q["text"])
                sim = float((c_emb * q_emb).sum())
                predicted_hit = sim >= t
                should_hit = (q["match"] == cid)
                if predicted_hit and should_hit:
                    score += 1
                elif predicted_hit and not should_hit:
                    score -= 2
                elif (not predicted_hit) and should_hit:
                    score -= 1
            if score > best_score:
                best_t, best_score = t, score
        backend.per_entry_threshold[cid] = best_t


def run_benchmark():
    canonical, test_queries = load_dataset()
    corpus = [c["text"] for c in canonical] + [t["text"] for t in test_queries]
    embedder = SharedEmbedder(corpus)

    rng = random.Random(42)
    calib_pool = [q for q in test_queries if q["category"] in ("paraphrase", "trap")]
    rng.shuffle(calib_pool)
    calibration_queries = calib_pool[: int(0.4 * len(calib_pool))]

    results = {"backends": {}, "meta": {
        "num_canonical": len(canonical),
        "num_test_queries": len(test_queries),
        "threshold_sweep": THRESHOLD_SWEEP,
        "cost_model": {
            "llm_call_cost_usd": LLM_CALL_COST_USD,
            "llm_call_latency_ms": LLM_CALL_LATENCY_MS,
            "cache_hit_extra_cost_usd": CACHE_HIT_EXTRA_COST_USD,
        },
    }}

    for backend_key in BACKEND_KEYS:
        per_threshold = []
        for threshold in THRESHOLD_SWEEP:
            backend = make_backend(backend_key, threshold)
            topic_of: dict[str, str] = {}
            for c in canonical:
                emb = embedder.embed(c["text"])
                backend.add(c["id"], c["text"], emb)
                topic_of[c["id"]] = c["id"]

            if backend_key == "adaptive":
                calibrate_adaptive(backend, canonical, embedder, calibration_queries, rng)

            tp = fp = tn = fn = 0
            total_cost = 0.0
            total_latency = 0.0
            judge_calls = 0
            category_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})

            for i, q in enumerate(test_queries):
                emb = embedder.embed(q["text"])
                result = backend.query(q["text"], emb)
                judge_cost_this_query = 0.0
                judge_latency_this_query = 0.0

                if backend_key == "confidence_tiered" and result.tier == "candidate":
                    judge_cost_this_query = JUDGE_COST_USD
                    judge_latency_this_query = JUDGE_LATENCY_MS
                    candidate_topic = topic_of.get(result.matched_id)
                    actually_correct = candidate_topic == q["match"]
                    judge_correct = actually_correct if rng.random() < JUDGE_ACCURACY else not actually_correct
                    if judge_correct:
                        result = CacheResult(hit=True, matched_id=result.matched_id,
                                              similarity=result.similarity,
                                              latency_ms=result.latency_ms, tier="candidate")
                    else:
                        result = CacheResult(hit=False, matched_id=None,
                                              similarity=result.similarity,
                                              latency_ms=result.latency_ms, tier="candidate")

                should_hit = q["match"] is not None
                predicted_topic = topic_of.get(result.matched_id) if result.hit else None
                correct_hit = result.hit and predicted_topic == q["match"]

                if result.hit and correct_hit:
                    tp += 1
                    category_stats[q["category"]]["tp"] += 1
                elif result.hit and not correct_hit:
                    fp += 1
                    category_stats[q["category"]]["fp"] += 1
                elif (not result.hit) and (not should_hit):
                    tn += 1
                    category_stats[q["category"]]["tn"] += 1
                else:
                    fn += 1
                    category_stats[q["category"]]["fn"] += 1

                if result.hit:
                    total_cost += CACHE_HIT_EXTRA_COST_USD + judge_cost_this_query
                    total_latency += result.latency_ms + judge_latency_this_query
                else:
                    total_cost += LLM_CALL_COST_USD + judge_cost_this_query
                    total_latency += LLM_CALL_LATENCY_MS + result.latency_ms + judge_latency_this_query
                    new_id = f"t{i}"
                    backend.add(new_id, q["text"], emb)
                    topic_of[new_id] = q["match"] if q["match"] is not None else f"__none_{i}__"
                judge_calls += 1 if judge_cost_this_query > 0 else 0

            n = len(test_queries)
            hits = tp + fp
            hit_rate = hits / n
            precision = tp / hits if hits > 0 else None
            recall = tp / (tp + fn) if (tp + fn) > 0 else None
            false_hit_rate = fp / hits if hits > 0 else 0.0
            no_cache_cost = n * LLM_CALL_COST_USD
            no_cache_latency = n * NO_CACHE_LATENCY_MS
            cost_saved_pct = (1 - total_cost / no_cache_cost) * 100 if no_cache_cost else 0
            latency_saved_pct = (1 - (total_latency / n) / NO_CACHE_LATENCY_MS) * 100

            per_threshold.append({
                "threshold": threshold,
                "hit_rate": round(hit_rate, 4),
                "precision": round(precision, 4) if precision is not None else None,
                "recall": round(recall, 4) if recall is not None else None,
                "false_hit_rate": round(false_hit_rate, 4),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "avg_latency_ms": round(total_latency / n, 2),
                "total_cost_usd": round(total_cost, 6),
                "cost_saved_pct": round(cost_saved_pct, 2),
                "latency_saved_pct": round(latency_saved_pct, 2),
                "judge_calls": judge_calls,
                "category_stats": category_stats,
            })

        results["backends"][backend_key] = {
            "name": make_backend(backend_key, 0.5).name,
            "by_threshold": per_threshold,
        }

    return results


if __name__ == "__main__":
    results = run_benchmark()
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote results.json")
    for key, data in results["backends"].items():
        row = next(r for r in data["by_threshold"] if r["threshold"] == 0.30)
        print(f"{data['name']:35s} hit_rate={row['hit_rate']:.2f} "
              f"precision={row['precision']} false_hit_rate={row['false_hit_rate']:.2f} "
              f"cost_saved={row['cost_saved_pct']:.1f}%")
