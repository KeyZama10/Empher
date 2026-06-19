import os
import matplotlib.pyplot as plt
import numpy as np

from vector_engine.benchmarks.run_benchmarks import (
    generate_synthetic_data,
    measure_brute_force,
    measure_ivf,
    measure_hnsw
)

def run_visualization():
    print("Generating synthetic dataset (10,000 vectors, 64 dims)...")
    base_vectors, vector_ids, query_vectors = generate_synthetic_data(
        n_vectors=10000,
        n_queries=100,
        dimension=64,
        seed=42
    )
    
    top_k = 10
    print("Running Brute Force Baseline...")
    bf = measure_brute_force(base_vectors, vector_ids, query_vectors, top_k=top_k)
    
    # Run IVF configs
    ivf_clusters = [5, 10, 20, 50, 100, 200]
    ivf_recalls = []
    ivf_latencies = []
    ivf_qps = []
    
    for c in ivf_clusters:
        print(f"Running IVF with n_clusters={c}...")
        res = measure_ivf(base_vectors, vector_ids, query_vectors, n_clusters=c, top_k=top_k, bf_results=bf["results"])
        ivf_recalls.append(res["recall"])
        ivf_latencies.append(res["avg_latency_ms"])
        ivf_qps.append(res["throughput_qps"])
        
    # Run HNSW configs
    hnsw_efs = [2, 5, 10, 20, 50, 100, 200]
    hnsw_recalls = []
    hnsw_latencies = []
    hnsw_qps = []
    
    for ef in hnsw_efs:
        print(f"Running HNSW with ef_search={ef}...")
        res = measure_hnsw(base_vectors, vector_ids, query_vectors, ef_search=ef, top_k=top_k, bf_results=bf["results"])
        hnsw_recalls.append(res["recall"])
        hnsw_latencies.append(res["avg_latency_ms"])
        hnsw_qps.append(res["throughput_qps"])
        
    # Create Side-by-Side Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Recall vs Latency (Lower latency is better, higher recall is better)
    ax1.plot(ivf_recalls, ivf_latencies, 'o-', color='#3182bd', label='IVF (varying n_clusters)', linewidth=2, markersize=8)
    ax1.plot(hnsw_recalls, hnsw_latencies, 's-', color='#e6550d', label='HNSW (varying ef_search)', linewidth=2, markersize=8)
    ax1.scatter([1.0], [bf["avg_latency_ms"]], color='#2ca02c', marker='*', s=200, label='Brute Force (Exact)', zorder=5)
    
    ax1.set_title("Recall vs Latency (Lower is Better)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Recall @ 10", fontsize=11)
    ax1.set_ylabel("Average Query Latency (ms)", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)
    
    # Plot 2: Recall vs Throughput (Higher Throughput/QPS is better, higher recall is better)
    ax2.plot(ivf_recalls, ivf_qps, 'o-', color='#3182bd', label='IVF (varying n_clusters)', linewidth=2, markersize=8)
    ax2.plot(hnsw_recalls, hnsw_qps, 's-', color='#e6550d', label='HNSW (varying ef_search)', linewidth=2, markersize=8)
    ax2.scatter([1.0], [bf["throughput_qps"]], color='#2ca02c', marker='*', s=200, label='Brute Force (Exact)', zorder=5)
    
    ax2.set_title("Recall vs Throughput (Higher is Better)", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Recall @ 10", fontsize=11)
    ax2.set_ylabel("Throughput (Queries Per Second)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save the output image
    output_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "search_performance.png")
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\nSaved search performance visualization plot to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    run_visualization()
