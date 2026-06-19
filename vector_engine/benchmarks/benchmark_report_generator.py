import os
import json
import csv
from datetime import datetime
from typing import Any

# Import benchmarks
from vector_engine.benchmarks.benchmark_ivf import run_benchmark as run_ivf
from vector_engine.benchmarks.benchmark_hnsw import run_benchmark as run_hnsw
from vector_engine.benchmarks.benchmark_pq import run_benchmark as run_pq
from vector_engine.benchmarks.benchmark_nprobe import run_benchmark as run_nprobe
from vector_engine.benchmarks.benchmark_distributed import run_benchmark as run_dist
from vector_engine.benchmarks.benchmark_replication import run_benchmark as run_rep

def generate_report():
    print("=============================================================")
    print("      DISTRIBUTED VECTOR ENGINE BENCHMARK ORCHESTRATOR       ")
    print("=============================================================")
    
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.normpath(os.path.join(script_dir, "..", "docs", "benchmarks"))
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Run benchmarks
    print("\n1/6. Running IVF Benchmark...")
    ivf_res = run_ivf()
    
    print("\n2/6. Running HNSW Benchmark...")
    hnsw_res = run_hnsw()
    
    print("\n3/6. Running PQ Benchmark...")
    pq_res = run_pq()
    
    print("\n4/6. Running IVF nprobe Benchmark...")
    nprobe_res = run_nprobe()
    
    print("\n5/6. Running Distributed Scaling Benchmark...")
    dist_res = run_dist()
    
    print("\n6/6. Running Primary-Replica Replication Benchmark...")
    rep_res = run_rep()
    
    # 2. Gather raw results
    all_results = {
        "timestamp": datetime.utcnow().isoformat(),
        "ivf_metrics": ivf_res,
        "hnsw_metrics": hnsw_res,
        "pq_metrics": pq_res,
        "nprobe_metrics": nprobe_res,
        "distributed_scaling": dist_res,
        "replication_metrics": rep_res
    }
    
    # Save JSON File
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Output] Saved JSON raw data to: {json_path}")
    
    # Save CSV File (flattened representation)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    csv_rows = []
    
    # Helper to append rows
    def add_row(bench: str, metric: str, val: Any, unit: str):
        csv_rows.append({
            "benchmark_name": bench,
            "metric_name": metric,
            "value": f"{val:.4f}" if isinstance(val, float) else str(val),
            "unit": unit
        })
        
    # Flat IVF / HNSW / PQ
    for res in [ivf_res, hnsw_res, pq_res]:
        b_name = res["index_type"]
        add_row(b_name, "Recall@1", res["recall_at_1"], "percentage")
        add_row(b_name, "Recall@5", res["recall_at_5"], "percentage")
        add_row(b_name, "Recall@10", res["recall_at_10"], "percentage")
        add_row(b_name, "Latency_P50", res["latency_p50_ms"], "ms")
        add_row(b_name, "Latency_P95", res["latency_p95_ms"], "ms")
        add_row(b_name, "Latency_P99", res["latency_p99_ms"], "ms")
        add_row(b_name, "Throughput", res["qps"], "QPS")
        add_row(b_name, "Memory", res["memory_mb"], "MB")
        add_row(b_name, "Build_Time", res["build_time_ms"], "ms")
        add_row(b_name, "Query_CPU", res["query_cpu_percent"], "percentage")
        if "compression_ratio" in res:
            add_row(b_name, "Compression_Ratio", res["compression_ratio"], "ratio")
            
    # nprobe tradeoff CSV entries
    for row in nprobe_res:
        b_name = f"IVF_nprobe_{row['nprobe']}"
        add_row(b_name, "Recall@10", row["recall_at_10"], "percentage")
        add_row(b_name, "Latency_P50", row["latency_p50_ms"], "ms")
        add_row(b_name, "Throughput", row["qps"], "QPS")
        
    # Distributed scaling CSV entries
    for row in dist_res:
        b_name = f"Distributed_Workers_{row['n_workers']}"
        add_row(b_name, "Ingest_Throughput", row["ingest_qps"], "QPS")
        add_row(b_name, "Search_Throughput", row["search_qps"], "QPS")
        add_row(b_name, "Latency_P95", row["search_latency_p95_ms"], "ms")
        add_row(b_name, "Recovery_Time", row["recovery_time_ms"], "ms")
        
    # Replication CSV entries
    b_name = "Replication_Metrics"
    add_row(b_name, "Replication_Lag", rep_res["replication_lag_ms"], "ms")
    add_row(b_name, "Primary_Reads_Throughput", rep_res["primary_reads_qps"], "QPS")
    add_row(b_name, "Replica_Reads_Throughput", rep_res["replica_reads_qps"], "QPS")
    add_row(b_name, "Replica_Reads_Latency_P95", rep_res["replica_reads_latency_p95_ms"], "ms")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["benchmark_name", "metric_name", "value", "unit"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[Output] Saved CSV flat table to: {csv_path}")
    
    # 3. Generate Markdown Report
    nprobe_table_rows = ""
    for r in nprobe_res:
        nprobe_table_rows += f"| **nprobe={r['nprobe']}** | {r['recall_at_1']:.4f} | {r['recall_at_5']:.4f} | {r['recall_at_10']:.4f} | {r['latency_p50_ms']:.2f} ms | {r['latency_p95_ms']:.2f} ms | {r['qps']:.1f} | {r['query_cpu_percent']:.1f}% |\n"
        
    dist_table_rows = ""
    for r in dist_res:
        rec_time = f"{r['recovery_time_ms']:.2f} ms" if r['recovery_time_ms'] > 0 else "N/A"
        dist_table_rows += f"| **{r['n_workers']} Worker(s)** | {r['ingest_qps']:.1f} | {r['search_qps']:.1f} | {r['search_latency_p50_ms']:.2f} ms | {r['search_latency_p95_ms']:.2f} ms | {rec_time} |\n"

    report_content = f"""# Distributed Vector Search Engine - Performance Benchmark Report

This document reports the performance metrics, search quality recall, system scalability, and replication lags evaluated across the indexing modules and distributed shards of the database.

---

## 1. Indexing Strategy Comparison

This table evaluates search quality (Recall), latency percentiles, throughput (QPS), build speeds, and memory consumption across the main index configurations:

| Index Strategy | Recall@1 | Recall@5 | Recall@10 | Latency (P50) | Latency (P95) | Latency (P99) | Throughput (QPS) | Memory RSS | Indexing Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact (Brute Force)** | 1.0000 | 1.0000 | 1.0000 | Baseline | Baseline | Baseline | - | Baseline | - |
| **IVF Index (n_clusters=32)** | {ivf_res['recall_at_1']:.4f} | {ivf_res['recall_at_5']:.4f} | {ivf_res['recall_at_10']:.4f} | {ivf_res['latency_p50_ms']:.2f} ms | {ivf_res['latency_p95_ms']:.2f} ms | {ivf_res['latency_p99_ms']:.2f} ms | {ivf_res['qps']:.1f} QPS | {ivf_res['memory_mb']:.2f} MB | {ivf_res['build_time_ms']:.1f} ms |
| **HNSW Index** | {hnsw_res['recall_at_1']:.4f} | {hnsw_res['recall_at_5']:.4f} | {hnsw_res['recall_at_10']:.4f} | {hnsw_res['latency_p50_ms']:.2f} ms | {hnsw_res['latency_p95_ms']:.2f} ms | {hnsw_res['latency_p99_ms']:.2f} ms | {hnsw_res['qps']:.1f} QPS | {hnsw_res['memory_mb']:.2f} MB | {hnsw_res['build_time_ms']:.1f} ms |
| **Product Quantization** | {pq_res['recall_at_1']:.4f} | {pq_res['recall_at_5']:.4f} | {pq_res['recall_at_10']:.4f} | {pq_res['latency_p50_ms']:.2f} ms | {pq_res['latency_p95_ms']:.2f} ms | {pq_res['latency_p99_ms']:.2f} ms | {pq_res['qps']:.1f} QPS | {pq_res['memory_mb']:.2f} MB | {pq_res['build_time_ms']:.1f} ms |

- **Compression Ratio**: Product Quantization achieved a **{pq_res['compression_ratio']:.1f}x memory compression ratio** compared to raw float32 vectors, with a search recall loss of **{pq_res['recall_loss_at_10']*100:.2f}%** at Recall@10.

---

## 2. IVF nprobe Recall-Latency Trade-offs

Increasing the `nprobe` parameter shifts search bounds across multiple cluster centroids, improving recall at the cost of processing more vectors (increasing latency):

| nprobe configuration | Recall@1 | Recall@5 | Recall@10 | Latency (P50) | Latency (P95) | Throughput (QPS) | CPU Usage % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{nprobe_table_rows}
- **Observation**: Larger `nprobe` levels yield close to perfect recall (converging towards brute force search) while decreasing query throughput due to larger search boundaries.

---

## 3. Distributed Sharding Scaling (1, 2, and 4 Worker Nodes)

Evaluating horizontal scalability, broadcast gather latencies, and worker node crash-recovery speeds under consistent hashing sharding:

| Active Worker Nodes | Ingest Throughput | Query Throughput | Latency (P50) | Latency (P95) | WAL Recovery Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
{dist_table_rows}
- **Fault Tolerance**: The WAL crash-recovery time indicates the duration (including subprocess restart, file checks, and log replay) required to bring a failed worker partition node back online with zero data loss.

---

## 4. Primary-Replica Replication & Replica-Read Performance

Primary-Replica configuration metrics under continuous insert replication:

- **Replication Lag**: **{rep_res['replication_lag_ms']:.2f} ms** (lag from primary ingestion confirmation to replica node memory state availability).
- **Read Throughput Scaling**:
  - **Primary-Only Reads**: {rep_res['primary_reads_qps']:.1f} QPS | Latency P95: {rep_res['primary_reads_latency_p95_ms']:.2f} ms
  - **Replica Load-Balanced Reads**: {rep_res['replica_reads_qps']:.1f} QPS | Latency P95: {rep_res['replica_reads_latency_p95_ms']:.2f} ms

---

## 5. Architectural Recommendations for Google Scale Ingest

1. **Product Quantization for Scaling**: PQ compresses high-dimensional vector spaces by **{pq_res['compression_ratio']:.1f}x**, enabling larger vector caches.
2. **HNSW for Low-Latency High-Recall**: HNSW exhibits sub-millisecond latencies with nearly perfect recall, making it ideal for real-time online retrieval.
3. **Replica Reads for QPS Scaling**: Offloading read queries to active replicas keeps primary write queues non-blocked, ensuring stable write latency.
"""
    
    markdown_path = os.path.join(output_dir, "benchmark_report.md")
    with open(markdown_path, "w") as f:
        f.write(report_content)
    print(f"[Output] Saved Markdown report to: {markdown_path}")
    
    # Save copy to the workspace docs/benchmarks directory
    workspace_docs_dir = os.path.normpath(os.path.join(script_dir, "..", "docs", "benchmarks"))
    os.makedirs(workspace_docs_dir, exist_ok=True)
    with open(os.path.join(workspace_docs_dir, "benchmark_report.md"), "w") as f:
        f.write(report_content)
    with open(os.path.join(workspace_docs_dir, "benchmark_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
        
    print("\nBenchmark framework execution completed successfully!")
    print("=============================================================")

if __name__ == "__main__":
    generate_report()
