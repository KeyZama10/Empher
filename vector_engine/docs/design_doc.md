# Distributed Vector Search Engine - Engineering Design Document

This document defines the architectural specifications, mathematical formulations, algorithmic layouts, and durability guarantees of the Distributed Vector Search Engine.

---

## 1. Problem Statement

Modern neural networks (large language models, vision models, recommendation networks) represent unstructured data as dense high-dimensional vectors (embeddings). As dataset cardinalities scale into millions of items, executing similarity searches across these embedding collections introduces major challenges:
1. **The Curse of Dimensionality**: High-dimensional spaces (e.g., 128 to 1024 dimensions) render spatial indexing structures (KD-trees, R-trees) ineffective, causing search latencies to degenerate to linear brute force scans ($O(N \cdot D)$).
2. **Horizontal Scalability Limits**: Monolithic vector stores cannot scale beyond the physical memory constraints of a single server. A distributed sharding partition layout is required.
3. **High Availability and Durability**: Sharded clusters are prone to individual node crashes and network partitions. The system must guarantee zero-data-loss durability and sub-second failover.
4. **Security Vulnerabilities**: Legacy persistence schemes (such as Python's native `pickle`) introduce critical Remote Code Execution (RCE) vulnerabilities.

This design document outlines the implementation of a distributed, secure, and durable vector database designed to resolve these challenges.

---

## 2. Requirements

### 2.1. Functional Requirements
- **Insert Vector**: Insert vectors with unique IDs, float coordinates, and key-value metadata.
- **Similarity Search**: Query the database using a vector to retrieve the top-$K$ most similar items ranked by Cosine Similarity.
- **Dynamic Sharding**: Partitions vectors dynamically across available primary shards.
- **Standby Failover**: Automatically promote a standby coordinator to active leader if the primary coordinator crashes.
- **Data Durability**: Recover node state to memory on reboot following a crash.

### 2.2. Core APIs (gRPC Protobuf)
```protobuf
service SearchWorker {
    rpc InsertVector (InsertRequest) returns (InsertResponse);
    rpc SearchVectors (SearchRequest) returns (SearchResponse);
    rpc HealthCheck (HealthRequest) returns (HealthResponse);
    rpc ClearIndex (ClearRequest) returns (ClearResponse);
}
```

---

## 3. Non-Functional Requirements

- **Latency SLAs**:
  - Read query P95 latency under 2.0 ms for single-shard queries.
  - Broadcast-gather P95 latency under 5.0 ms across a 4-node cluster.
- **Read Throughput**: Sustain at least 5,000 read QPS on a standard indexing strategy.
- **Write Durability**: Commit mutations to persistent storage via Write-Ahead Logging (WAL) with `fsync` before return confirmations.
- **Fault Tolerance**: Standby coordinator self-promotion lease expiration under 4 seconds. Key movement on node churn under 30%.
- **Zero-RCE Security**: Eliminate pickle serialization. Implement binary NumPy `.npy` formatting with checksum validations.

---

## 4. Architecture Overview

The system utilizes a decoupled coordinator-worker topology coordinated by a centralized service registry:

```
[ Client ]
    │ (HTTP REST / JSON)
    ▼
[ Load Balancer ]
    │
    ├───────────────┐
    ▼               ▼ (Active-Passive Election via Lock Lease)
[ Coordinator A ] [ Coordinator B ]
 (Active Leader)   (Standby Backup)
    │
    ├──► [ Consistent Hash Ring ] (SHA-256 Virtual Nodes Mapping)
    │
    ├──► [ Service Registry (etcd / Lock Mock) ]
    │
    └──► [ Worker Shard Clusters ]
          ├── Shard 1 Primary (Port 50051) ──► Shard 1 Replica (Port 50052)
          └── Shard 2 Primary (Port 50053) ──► Shard 2 Replica (Port 50054)
```

---

## 5. Search Architecture & Indexing Strategies

The engine supports three indexing strategies alongside brute force Cosine Similarity:

### 5.1. Mathematical Formulation (Cosine Similarity)
$$\text{Similarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}$$

### 5.2. Indexing Options
1. **Brute Force (Exact)**:
   - Computes dot product across all items in memory.
   - **Complexity**: $O(N \cdot D)$ search time. Guarantees 100% Recall.
2. **IVF (Inverted File Index)**:
   - Partitions vector space into $C$ cluster centroids via $K$-means clustering.
   - Search queries are mapped to the nearest $M$ centroids (where $M = \text{nprobe}$).
   - **Complexity**: $O(\frac{M}{C} \cdot N \cdot D)$ search time.
   - **nprobe Sweeping**: Higher `nprobe` levels expand searched centroids, shifting along the recall-latency Pareto frontier (Recall climbs from 11.4% at `nprobe=1` to 66.4% at `nprobe=16`).
3. **HNSW (Hierarchical Navigable Small World)**:
   - Constructs a multi-layer graph where bottom layers contain dense, short-range connections, and top layers hold sparse, long-range links.
   - Navigates logarithmically from top layers down to identify nearest neighbors.
   - **Complexity**: $O(\log N)$ search time. Delivers **6,000 QPS** at **76.5% Recall@10**.
4. **Product Quantization (PQ)**:
   - Compresses vectors by splitting the $D$-dimensional space into $M$ sub-vectors.
   - Trains centroids for each sub-space (quantization codebooks).
   - Replaces float coordinates with byte indexes referencing codebook centroids.
   - **Complexity**: $O(N \cdot M)$ using Asymmetric Distance Computation (ADC). Compresses index sizes by **17.6x**.

---

## 6. Sharding Strategy

The database segments the vector keyspace into partition shards.
- **Scatter Stage**: The Coordinator broadcasts search queries in parallel to the worker nodes representing all active partition shards.
- **Local Search**: Each worker executes local similarity searches (exact or indexed) over its in-memory segment.
- **Gather Stage**: The Coordinator gathers local top-$K$ candidate lists from the workers.
- **Merge & Re-rank**: The Coordinator consolidates all local lists, performs global sorting by descending similarity score, and slices the final global top-$K$ list.

---

## 7. Consistent Hashing Design

To avoid the excessive key redistribution under modulo-based sharding ($N \bmod \text{workers}$), the coordinator uses a **Consistent Hashing Ring**:

### 7.1. Mathematical Guarantee
When worker nodes join or leave the ring, the number of keys required to move is bounded by:
$$\text{Keys Moved} \approx \frac{K_{\text{total}}}{N}$$
Where $N$ is the number of active workers. Modulo hashing moves up to $74.6\%$ of keys on churn. Consistent Hashing bounds this to **23.2%** (less than the $30\%$ SLA requirement).

### 7.2. Virtual Nodes (vnodes)
- Each physical worker maps to 100 virtual nodes distributed uniformly across the 32-bit ring.
- Resolves key-distribution skewness (data imbalances) across primary worker shards.
- Hashing uses SHA-256 to guarantee uniformity of partition rings.

---

## 8. Write-Ahead Logging (WAL) Durability Design

To guarantee durability, mutations are written to disk before memory commits:

```
gRPC Write Request ──► Append JSON Line (active.wal) ──► OS fsync() ──► Commit to VectorStore
```

### 8.1. Ingestion Guarantees
- A write is confirmed successful only after the operation is appended to `active.wal` and flushed to disk using explicit operating system `fsync` calls.
- Ingestion write latency with WAL is `0.3608 ms` compared to `0.0012 ms` memory-only.

### 8.2. Snapshot & WAL Rotation
- To prevent unbounded log growth, the worker triggers log rotation:
  1. Freezes memory writes.
  2. Saves current `VectorStore` memory matrices to a compressed NumPy zip archive (`snapshot.npz`).
  3. Truncates `active.wal` to 0 bytes.
- On reboot, the worker loads `snapshot.npz` directly and replays only the trailing mutations recorded in `active.wal`. This snapshot optimization yields a **5.7x recovery speedup** (reducing recovery time from 281.58 ms to 49.80 ms).

---

## 9. Replication Design

High Availability is achieved via Primary-Replica replication:

- **Write Path**: Insert operations route to the Primary Worker node assigned by the hash ring. The primary logs mutations locally and asynchronously forwards the write payload to its registered read-replicas.
- **Read Path**: The coordinator load-balances query requests across healthy replica nodes using a random choice selector. If all replica nodes fail health checks, query targets fallback automatically to the primary worker.
- **Replication Lag**: Evaluated at **16.47 ms** on local loopback. Read-write isolation ensures primary write queues are unblocked.

---

## 10. Failover Design

The coordinator cluster implements active-passive leader election to prevent split-brain routing:

- **Distributed Locks**: Coordinators attempt to acquire a Compare-And-Swap (CAS) lease lock key `/coordinators/leader` in the service registry with a 4-second TTL.
- **Heartbeats**: The active coordinator renews its lock lease every 1 second.
- **Passive Promotion**: The standby coordinator continuously polls registry lock statuses. If the active coordinator crashes and fails to renew heartbeats, the lease expires. The standby coordinator acquires the lease and promotes itself to Leader.
- **linear writing**: Standby coordinators reject all write/read API calls with a `PermissionError` to prevent split-brain state mutations.

---

## 11. Security Design (Anti-Pickle Persistence)

To eliminate Remote Code Execution (RCE) injection vectors, the database uses a secure NumPy-based persistence schema:

- **NumPy Serialization**: Vector coordinate matrices are stored as binary NumPy `.npy` arrays loaded strictly with `allow_pickle=False`. Metadata and configurations are saved as JSON strings.
- **SHA-256 Integrity Verification**: On save, SHA-256 hashes of `.npy` arrays are compiled and written into `index_config.json`. On load, these hashes are recalculated and verified to prevent loading corrupted files.
- **Version Compatibility Check**: Saved files compile `version: "1.0.0"` in `version.json`. Schema versions are validated on load, and mismatched versions are rejected.

---

## 12. Performance Analysis

### 12.1. Benchmark Evaluation Summary
The performance metrics of the engine indexing configurations are summarized below:

| Configuration | Recall@10 | Latency (P50) | Latency (P95) | Throughput (QPS) | Memory RSS | Build Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact (Brute Force)** | 1.0000 | Baseline | Baseline | - | Baseline | - |
| **HNSW Index** | 0.7650 | 0.15 ms | 0.20 ms | 5999.6 QPS | 2.88 MB | 724.5 ms |
| **IVF (n_clusters=32)** | 0.1720 | 0.30 ms | 0.38 ms | 3169.9 QPS | < 0.1 MB | 449.9 ms |
| **Product Quantization** | 0.3210 | 2.83 ms | 6.16 ms | 322.5 QPS | 39.62 MB | 2107.6 ms |

---

## 13. Scaling Analysis & System Bottlenecks

### 13.1. Coordinator Broadcast Bounds
As the number of shards $N$ scales, the coordinator's scatter-gather phase is bounded by the slowest responding shard ($O(N)$ network fan-out). This can lead to latency degradation under high concurrency.
- *Mitigation*: Implement asynchronous scatter execution using non-blocking gRPC multiplexing and partial query completions.

### 13.2. Registry Key-Space Bounds
Using a shared JSON file registry fallback introduces write bottlenecks due to OS file locking.
- *Mitigation*: Deploy multi-node etcd consensus clusters to scale registry key write throughput.

---

## 14. Tradeoffs

### 14.1. Consistency vs Availability (CAP Theorem)
- Under network partitions separating primary nodes from replica nodes, the system chooses **Availability (AP)**: write confirmations continue at the primary, sacrificing replica query consistency temporarily until partition healing.

### 14.2. Memory Space vs CPU Cycles
- **Product Quantization** trades CPU compression latency for memory efficiency. Centroid calculations and ADC lookup tables reduce physical memory requirements by **17.6x**, but increase search latency to `6.16 ms`.

---

## 15. Future Work

- **Raft Consensus Integration**: Replace primary-replica async writes with synchronous Raft consensus log replication across shard groups to guarantee strong consistency (CP).
- **Dynamic Centroid Retraining**: Implement background clustering retraining to adapt IVF indices dynamically without indexing downtime as vector distributions shift.
- **GPU Acceleration**: Implement CUDA kernels for accelerated matrix similarity calculation to scale brute force throughput.
