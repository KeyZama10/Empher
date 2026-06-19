import gc
import time
import os
import tracemalloc
import numpy as np
from typing import List, Dict, Any, Tuple

from vector_engine.app.vector_store import VectorStore
from vector_engine.app.ivf_index import IVFIndex
from vector_engine.app.hnsw_index import HNSWIndex
from vector_engine.app.search import brute_force_search

def generate_synthetic_data(
    n_vectors: int, 
    n_queries: int, 
    dimension: int, 
    seed: int = 42
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Generate reproducible L2-normalized synthetic vector datasets."""
    np.random.seed(seed)
    base_vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    norms = np.linalg.norm(base_vectors, axis=1, keepdims=True)
    base_vectors = base_vectors / np.where(norms == 0.0, 1.0, norms)
    
    vector_ids = [f"vec_{i}" for i in range(n_vectors)]
    
    query_vectors = np.random.randn(n_queries, dimension).astype(np.float32)
    q_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    query_vectors = query_vectors / np.where(q_norms == 0.0, 1.0, q_norms)
    
    return base_vectors, vector_ids, query_vectors

def measure_brute_force(
    base_vectors: np.ndarray, 
    vector_ids: List[str], 
    query_vectors: np.ndarray, 
    top_k: int
) -> Dict[str, Any]:
    """Measure indexing and query performance of Brute Force index."""
    gc.collect()
    tracemalloc.start()
    
    t_start = time.perf_counter()
    store = VectorStore(dimension=base_vectors.shape[1])
    store.add_vectors(vector_ids, base_vectors)
    indexing_time = (time.perf_counter() - t_start) * 1000.0  # ms
    
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    latencies = []
    results_ids_list = []
    
    for q in query_vectors:
        t_q_start = time.perf_counter()
        res = brute_force_search(store, q, top_k=top_k)
        latencies.append((time.perf_counter() - t_q_start) * 1000.0)  # ms
        results_ids_list.append([r.id for r in res])
        
    avg_latency = np.mean(latencies)
    throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    
    return {
        "indexing_time_ms": indexing_time,
        "memory_usage_kb": peak_memory / 1024.0,
        "avg_latency_ms": avg_latency,
        "throughput_qps": throughput,
        "results": results_ids_list
    }

def measure_ivf(
    base_vectors: np.ndarray, 
    vector_ids: List[str], 
    query_vectors: np.ndarray, 
    n_clusters: int, 
    top_k: int,
    bf_results: List[List[str]]
) -> Dict[str, Any]:
    """Measure indexing and query performance of IVF index."""
    gc.collect()
    tracemalloc.start()
    
    t_start = time.perf_counter()
    ivf = IVFIndex(n_clusters=n_clusters, dimension=base_vectors.shape[1])
    ivf.train(base_vectors)
    ivf.add_vectors(vector_ids, base_vectors)
    indexing_time = (time.perf_counter() - t_start) * 1000.0  # ms
    
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    latencies = []
    recalls = []
    
    for idx, q in enumerate(query_vectors):
        t_q_start = time.perf_counter()
        res = ivf.search(q, top_k=top_k)
        latencies.append((time.perf_counter() - t_q_start) * 1000.0)  # ms
        
        ivf_ids = set([r.id for r in res])
        bf_ids = set(bf_results[idx])
        
        intersection = ivf_ids.intersection(bf_ids)
        expected_size = min(top_k, len(bf_ids))
        recall = len(intersection) / expected_size if expected_size > 0 else 1.0
        recalls.append(recall)
        
    avg_latency = np.mean(latencies)
    avg_recall = np.mean(recalls)
    throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    
    return {
        "indexing_time_ms": indexing_time,
        "memory_usage_kb": peak_memory / 1024.0,
        "avg_latency_ms": avg_latency,
        "recall": avg_recall,
        "throughput_qps": throughput
    }

def measure_hnsw(
    base_vectors: np.ndarray, 
    vector_ids: List[str], 
    query_vectors: np.ndarray, 
    ef_search: int, 
    top_k: int,
    bf_results: List[List[str]]
) -> Dict[str, Any]:
    """Measure indexing and query performance of HNSW index."""
    gc.collect()
    tracemalloc.start()
    
    t_start = time.perf_counter()
    hnsw = HNSWIndex(
        dimension=base_vectors.shape[1], 
        max_elements=len(vector_ids),
        M=16,
        ef_construction=200,
        ef_search=ef_search
    )
    hnsw.add_vectors(vector_ids, base_vectors)
    indexing_time = (time.perf_counter() - t_start) * 1000.0  # ms
    
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    latencies = []
    recalls = []
    
    for idx, q in enumerate(query_vectors):
        t_q_start = time.perf_counter()
        res = hnsw.search(q, top_k=top_k)
        latencies.append((time.perf_counter() - t_q_start) * 1000.0)  # ms
        
        hnsw_ids = set([r.id for r in res])
        bf_ids = set(bf_results[idx])
        
        intersection = hnsw_ids.intersection(bf_ids)
        expected_size = min(top_k, len(bf_ids))
        recall = len(intersection) / expected_size if expected_size > 0 else 1.0
        recalls.append(recall)
        
    avg_latency = np.mean(latencies)
    avg_recall = np.mean(recalls)
    throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    
    return {
        "indexing_time_ms": indexing_time,
        "memory_usage_kb": peak_memory / 1024.0,
        "avg_latency_ms": avg_latency,
        "recall": avg_recall,
        "throughput_qps": throughput
    }

def run():
    print("Generating synthetic dataset (10,000 vectors, 64 dims)...")
    base_vectors, vector_ids, query_vectors = generate_synthetic_data(
        n_vectors=10000,
        n_queries=100,
        dimension=64,
        seed=42
    )
    
    top_k = 10
    print(f"Running baseline Brute Force search (top_k={top_k})...")
    bf_metrics = measure_brute_force(base_vectors, vector_ids, query_vectors, top_k=top_k)
    
    # IVF Configurations
    ivf_configs = [5, 20, 50, 100]
    ivf_results = []
    for clusters in ivf_configs:
        print(f"Running IVF Index search (n_clusters={clusters}, top_k={top_k})...")
        metrics = measure_ivf(
            base_vectors, 
            vector_ids, 
            query_vectors, 
            n_clusters=clusters, 
            top_k=top_k, 
            bf_results=bf_metrics["results"]
        )
        ivf_results.append((clusters, metrics))
        
    # HNSW Configurations (varying ef_search)
    hnsw_configs = [5, 10, 50, 100]
    hnsw_results = []
    for ef in hnsw_configs:
        print(f"Running HNSW Index search (ef_search={ef}, top_k={top_k})...")
        metrics = measure_hnsw(
            base_vectors,
            vector_ids,
            query_vectors,
            ef_search=ef,
            top_k=top_k,
            bf_results=bf_metrics["results"]
        )
        hnsw_results.append((ef, metrics))
        
    # Print Console Summary
    print("\n==========================================================================================================")
    print("                                         BENCHMARK SUMMARY                                                ")
    print("==========================================================================================================")
    print(f"{'Index Type':<25} | {'Latency':<12} | {'Throughput':<12} | {'Recall @ 10':<12} | {'Indexing Time':<15} | {'Memory (Peak)':<12}")
    print("-" * 106)
    print(f"{'Brute Force (Exact)':<25} | {bf_metrics['avg_latency_ms']:8.4f} ms | {bf_metrics['throughput_qps']:8.1f} QPS | {'1.0000':<12} | {bf_metrics['indexing_time_ms']:12.2f} ms | {bf_metrics['memory_usage_kb']:10.1f} KB")
    
    print("-" * 106)
    for clusters, res in ivf_results:
        print(f"IVF (n_clusters={clusters:<3})          | {res['avg_latency_ms']:8.4f} ms | {res['throughput_qps']:8.1f} QPS | {res['recall']:12.4f} | {res['indexing_time_ms']:12.2f} ms | {res['memory_usage_kb']:10.1f} KB")
        
    print("-" * 106)
    for ef, res in hnsw_results:
        print(f"HNSW (ef_search={ef:<3})          | {res['avg_latency_ms']:8.4f} ms | {res['throughput_qps']:8.1f} QPS | {res['recall']:12.4f} | {res['indexing_time_ms']:12.2f} ms | {res['memory_usage_kb']:10.1f} KB")
    print("==========================================================================================================\n")

    # Generate Markdown report in the project directory
    report_path = os.path.join(os.path.dirname(__file__), "..", "benchmark_report.md")
    
    # Format markdown contents
    markdown_content = f"""# Vector Search Benchmark Report: Brute Force vs IVF vs HNSW

This report provides performance and quality evaluations comparing the **Exact Brute-Force Cosine Similarity**, **Inverted File Index (IVF)**, and **HNSW Graph Approximate Nearest Neighbor** indexes.

## Evaluation Setup
- **Dataset Size**: 10,000 vectors
- **Vector Dimension**: 64 dimensions (L2-normalized)
- **Query Batch**: 100 random queries
- **Top-K Retrieval**: $k = 10$
- **Hardware Profile**: macOS Python environment

## Performance Summary Table

| Index Type | Avg Latency | Throughput (QPS) | Recall @ 10 | Indexing Time (Train+Add) | Peak Memory Allocated |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Brute Force (Baseline)** | {bf_metrics['avg_latency_ms']:.4f} ms | {bf_metrics['throughput_qps']:.1f} QPS | 1.0000 | {bf_metrics['indexing_time_ms']:.2f} ms | {bf_metrics['memory_usage_kb']:.1f} KB |
"""
    for clusters, res in ivf_results:
        markdown_content += f"| **IVF (n_clusters={clusters})** | {res['avg_latency_ms']:.4f} ms | {res['throughput_qps']:.1f} QPS | {res['recall']:.4f} | {res['indexing_time_ms']:.2f} ms | {res['memory_usage_kb']:.1f} KB |\n"

    for ef, res in hnsw_results:
        markdown_content += f"| **HNSW (ef_search={ef})** | {res['avg_latency_ms']:.4f} ms | {res['throughput_qps']:.1f} QPS | {res['recall']:.4f} | {res['indexing_time_ms']:.2f} ms | {res['memory_usage_kb']:.1f} KB |\n"

    markdown_content += """
## Key Observations

1. **Exact vs Approximate Search**:
   - **Brute Force** calculates similarity scores against all indexed vectors, ensuring perfect recall (1.0000) at the expense of higher lookup latency (low QPS).
   - **IVF (Inverted File Index)** achieves faster query times by partitioning data into clusters via KMeans and searching only within the query's nearest cluster bucket. However, vectors residing in other buckets are missed, reducing recall.
   - **HNSW (Hierarchical Navigable Small World)** organizes vectors into a multi-layered graph. It yields excellent lookup throughput (extremely high QPS) while maintaining high recall (usually > 0.95 with proper configuration), offering the best overall trade-off.

2. **HNSW Hyperparameter tuning (`ef_search`)**:
   - Tuning `ef_search` allows you to control the search size on the graph.
   - Higher values of `ef_search` increase graph coverage during queries, yielding near-perfect recall at the cost of higher query execution time (lower QPS).
   - Lower values of `ef_search` optimize for maximum search speed (throughput) with slightly lower recall.

3. **Memory & Build Time Overhead**:
   - **Brute Force** requires almost no build time and very minimal memory footprint.
   - **IVF** requires clustering (KMeans training), which introduces moderate build time and low index overhead.
   - **HNSW** constructs an extensive proximity graph. Constructing the links takes longer (longer build time) and requires more memory to store the graph structures (higher peak memory).
"""
    
    with open(report_path, "w") as f:
        f.write(markdown_content)
        
    print(f"Generated benchmark report at: {os.path.abspath(report_path)}")

if __name__ == "__main__":
    run()
