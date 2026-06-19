import os
import time
import psutil
import numpy as np
from typing import List, Dict, Any, Tuple
from vector_engine.app.hnsw_index import HNSWIndex
from vector_engine.app.vector_store import VectorStore
from vector_engine.app.search import brute_force_search

def generate_data(n_vectors: int, n_queries: int, dimension: int) -> Tuple[List[str], np.ndarray, np.ndarray]:
    np.random.seed(42)
    base_vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    norms = np.linalg.norm(base_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    base_vectors = base_vectors / norms
    
    query_vectors = np.random.randn(n_queries, dimension).astype(np.float32)
    q_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1.0
    query_vectors = query_vectors / q_norms
    
    ids = [f"vec_{i}" for i in range(n_vectors)]
    return ids, base_vectors, query_vectors

def calculate_recall(exact_results: List[List[str]], approx_results: List[List[str]], k: int) -> float:
    total_recall = 0.0
    for exact, approx in zip(exact_results, approx_results):
        exact_set = set(exact[:k])
        approx_set = set(approx[:k])
        if exact_set:
            intersection = exact_set.intersection(approx_set)
            total_recall += len(intersection) / k
    return total_recall / len(exact_results)

def run_benchmark() -> Dict[str, Any]:
    n_vectors = 10000
    n_queries = 100
    dimension = 64
    
    ids, base_vectors, query_vectors = generate_data(n_vectors, n_queries, dimension)
    
    # 1. Ground Truth (Exact Brute Force)
    store = VectorStore(dimension=dimension)
    store.add_vectors(ids, base_vectors)
    
    exact_results = []
    for q in query_vectors:
        res = brute_force_search(store, q, top_k=10)
        exact_results.append([r.id for r in res])
        
    # 2. Benchmark HNSW
    process = psutil.Process()
    mem_before = process.memory_info().rss
    
    index = HNSWIndex(dimension=dimension, max_elements=n_vectors)
    
    # Measure build CPU and memory
    process.cpu_percent(interval=None)
    t0 = time.perf_counter()
    index.add_vectors(ids, base_vectors)
    build_time = (time.perf_counter() - t0) * 1000.0
    build_cpu = process.cpu_percent(interval=None)
    
    mem_after = process.memory_info().rss
    mem_consumed_mb = (mem_after - mem_before) / (1024 * 1024)
    
    # Measure query performance
    latencies = []
    approx_results = []
    
    process.cpu_percent(interval=None)
    t_start = time.perf_counter()
    for q in query_vectors:
        t_q0 = time.perf_counter()
        res = index.search(q, top_k=10)
        latencies.append((time.perf_counter() - t_q0) * 1000.0) # ms
        approx_results.append([r.id for r in res])
    total_query_time = time.perf_counter() - t_start
    query_cpu = process.cpu_percent(interval=None)
    
    # Metrics
    qps = n_queries / total_query_time
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    recall_1 = calculate_recall(exact_results, approx_results, 1)
    recall_5 = calculate_recall(exact_results, approx_results, 5)
    recall_10 = calculate_recall(exact_results, approx_results, 10)
    
    return {
        "index_type": "HNSW",
        "recall_at_1": recall_1,
        "recall_at_5": recall_5,
        "recall_at_10": recall_10,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_p99_ms": p99,
        "qps": qps,
        "build_time_ms": build_time,
        "build_cpu_percent": build_cpu,
        "query_cpu_percent": query_cpu,
        "memory_mb": mem_consumed_mb
    }

if __name__ == "__main__":
    res = run_benchmark()
    import json
    print(json.dumps(res, indent=2))
