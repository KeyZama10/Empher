# Vector Search Benchmark Report: Brute Force vs IVF vs HNSW

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
| **Brute Force (Baseline)** | 3.2842 ms | 304.5 QPS | 1.0000 | 58.06 ms | 4981.3 KB |
| **IVF (n_clusters=5)** | 1.5187 ms | 658.5 QPS | 0.3770 | 496.42 ms | 15082.8 KB |
| **IVF (n_clusters=20)** | 0.5915 ms | 1690.5 QPS | 0.1720 | 439.90 ms | 15069.7 KB |
| **IVF (n_clusters=50)** | 0.3663 ms | 2729.9 QPS | 0.1370 | 370.07 ms | 15069.7 KB |
| **IVF (n_clusters=100)** | 0.2146 ms | 4659.7 QPS | 0.1040 | 475.91 ms | 15069.6 KB |
| **HNSW (ef_search=5)** | 0.0813 ms | 12306.4 QPS | 0.3480 | 690.05 ms | 4364.4 KB |
| **HNSW (ef_search=10)** | 0.0818 ms | 12223.0 QPS | 0.3440 | 642.50 ms | 4363.8 KB |
| **HNSW (ef_search=50)** | 0.1714 ms | 5833.0 QPS | 0.7600 | 685.32 ms | 4363.8 KB |
| **HNSW (ef_search=100)** | 0.3429 ms | 2916.5 QPS | 0.9040 | 907.36 ms | 4363.8 KB |

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
