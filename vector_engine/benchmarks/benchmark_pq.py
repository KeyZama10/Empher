import os
import time
import psutil
import numpy as np
from typing import List, Dict, Any, Tuple
from vector_engine.app.pq import ProductQuantizer
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
    n_subspaces = 8
    n_centroids = 256
    
    ids, base_vectors, query_vectors = generate_data(n_vectors, n_queries, dimension)
    
    # 1. Ground Truth (Exact Brute Force on raw vectors)
    store_raw = VectorStore(dimension=dimension)
    store_raw.add_vectors(ids, base_vectors)
    
    exact_results = []
    for q in query_vectors:
        res = brute_force_search(store_raw, q, top_k=10)
        exact_results.append([r.id for r in res])
        
    # 2. Benchmark PQ
    process = psutil.Process()
    mem_before = process.memory_info().rss
    
    pq = ProductQuantizer(n_subspaces=n_subspaces, n_centroids=n_centroids)
    
    # Measure training and encoding
    process.cpu_percent(interval=None)
    t0 = time.perf_counter()
    pq.train(base_vectors)
    codes = pq.encode(base_vectors)
    build_time = (time.perf_counter() - t0) * 1000.0
    build_cpu = process.cpu_percent(interval=None)
    
    mem_after = process.memory_info().rss
    mem_consumed_mb = (mem_after - mem_before) / (1024 * 1024)
    
    # Reconstruct vectors (decode) to evaluate search quality
    reconstructed_vectors = pq.decode(codes)
    
    # Create a vector store with reconstructed vectors to evaluate recall loss
    store_pq = VectorStore(dimension=dimension)
    store_pq.add_vectors(ids, reconstructed_vectors)
    
    # Measure query performance using the reconstructed vectors
    latencies = []
    approx_results = []
    
    process.cpu_percent(interval=None)
    t_start = time.perf_counter()
    for q in query_vectors:
        t_q0 = time.perf_counter()
        res = brute_force_search(store_pq, q, top_k=10)
        latencies.append((time.perf_counter() - t_q0) * 1000.0) # ms
        approx_results.append([r.id for r in res])
    total_query_time = time.perf_counter() - t_start
    query_cpu = process.cpu_percent(interval=None)
    
    # Compression Ratio Calculation
    # Raw size: N * D * 4 bytes (float32)
    # PQ size: N * M * 1 byte (uint8) + codebook size (M * K * d * 4 bytes)
    raw_size_bytes = n_vectors * dimension * 4
    pq_size_bytes = (n_vectors * n_subspaces * 1) + (n_subspaces * n_centroids * (dimension // n_subspaces) * 4)
    compression_ratio = raw_size_bytes / pq_size_bytes
    
    # Metrics
    qps = n_queries / total_query_time
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    recall_1 = calculate_recall(exact_results, approx_results, 1)
    recall_5 = calculate_recall(exact_results, approx_results, 5)
    recall_10 = calculate_recall(exact_results, approx_results, 10)
    
    return {
        "index_type": "ProductQuantization",
        "recall_at_1": recall_1,
        "recall_at_5": recall_5,
        "recall_at_10": recall_10,
        "recall_loss_at_10": 1.0 - recall_10,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_p99_ms": p99,
        "qps": qps,
        "build_time_ms": build_time,
        "build_cpu_percent": build_cpu,
        "query_cpu_percent": query_cpu,
        "memory_mb": mem_consumed_mb,
        "compression_ratio": compression_ratio
    }

if __name__ == "__main__":
    res = run_benchmark()
    import json
    print(json.dumps(res, indent=2))
