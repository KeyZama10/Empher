import os
import shutil
import time
import numpy as np
from typing import List, Dict, Tuple
from vector_engine.app.vector_store import VectorStore
from vector_engine.app.storage.wal import WALManager

def generate_synthetic_data(n_vectors: int, dimension: int) -> Tuple[List[str], np.ndarray]:
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    ids = [f"vec_{i}" for i in range(n_vectors)]
    return ids, vectors

def run_benchmarks() -> None:
    print("=============================================================")
    print("                 WAL PERFORMANCE BENCHMARKS                  ")
    print("=============================================================")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_dir = os.path.join(script_dir, "wal_benchmark_temp")
    if os.path.exists(benchmark_dir):
        shutil.rmtree(benchmark_dir)
    os.makedirs(benchmark_dir, exist_ok=True)
    
    dimension = 128
    
    # 1. Benchmark Insert Latency Overhead
    print("\n1. Evaluating Insert Latency...")
    n_inserts = 1000
    ids, vectors = generate_synthetic_data(n_inserts, dimension)
    
    # Baseline: Memory Only
    store_mem = VectorStore(dimension=dimension)
    t_start = time.perf_counter()
    for i in range(n_inserts):
        store_mem.add_vector(ids[i], vectors[i])
    mem_total_time = (time.perf_counter() - t_start) * 1000.0  # ms
    mem_avg_latency = mem_total_time / n_inserts
    print(f"   Memory Only:   Avg Latency = {mem_avg_latency:8.4f} ms | Total = {mem_total_time:8.2f} ms")
    
    # WAL Enabled (with fsync)
    store_wal = VectorStore(dimension=dimension)
    wal_mgr = WALManager(benchmark_dir)
    t_start = time.perf_counter()
    for i in range(n_inserts):
        wal_mgr.append_insert(ids[i], vectors[i].tolist())
        store_wal.add_vector(ids[i], vectors[i])
    wal_total_time = (time.perf_counter() - t_start) * 1000.0  # ms
    wal_avg_latency = wal_total_time / n_inserts
    wal_mgr.close()
    print(f"   WAL (fsync):   Avg Latency = {wal_avg_latency:8.4f} ms | Total = {wal_total_time:8.2f} ms")
    
    # 2. Benchmark Recovery Speeds (Scale of operations)
    print("\n2. Evaluating Recovery Times (Replaying WAL logs)...")
    sizes = [100, 1000, 5000]
    recovery_times = {}
    
    for size in sizes:
        size_dir = os.path.join(benchmark_dir, f"size_{size}")
        os.makedirs(size_dir, exist_ok=True)
        
        wal = WALManager(size_dir)
        size_ids, size_vectors = generate_synthetic_data(size, dimension)
        for i in range(size):
            wal.append_insert(size_ids[i], size_vectors[i].tolist())
        wal.close()
        
        # Measure replay time on fresh store
        rec_store = VectorStore(dimension=dimension)
        t_start = time.perf_counter()
        rec_wal = WALManager(size_dir)
        rec_wal.replay(rec_store)
        replay_time_ms = (time.perf_counter() - t_start) * 1000.0
        rec_wal.close()
        
        recovery_times[size] = replay_time_ms
        print(f"   Replaying {size:5} entries: Recovery Time = {replay_time_ms:8.2f} ms")
        
    # 3. Benchmark Snapshot vs Pure WAL Recovery
    print("\n3. Comparing Snapshot vs Pure WAL Recovery...")
    total_ops = 5500
    snapshot_ops = 5000
    rem_ops = 500
    
    ids_all, vectors_all = generate_synthetic_data(total_ops, dimension)
    
    # Case A: Pure WAL recovery of 5,500 operations
    pure_dir = os.path.join(benchmark_dir, "pure_wal")
    os.makedirs(pure_dir, exist_ok=True)
    wal_pure = WALManager(pure_dir)
    for i in range(total_ops):
        wal_pure.append_insert(ids_all[i], vectors_all[i].tolist())
    wal_pure.close()
    
    t_start = time.perf_counter()
    rec_store_pure = VectorStore(dimension=dimension)
    wal_rec_pure = WALManager(pure_dir)
    wal_rec_pure.replay(rec_store_pure)
    pure_recovery_ms = (time.perf_counter() - t_start) * 1000.0
    wal_rec_pure.close()
    
    # Case B: Snapshot at 5,000 + replay 500 mutations
    snap_dir = os.path.join(benchmark_dir, "snap_wal")
    os.makedirs(snap_dir, exist_ok=True)
    
    # Index 5,000 and rotate
    wal_snap = WALManager(snap_dir)
    store_snap = VectorStore(dimension=dimension)
    for i in range(snapshot_ops):
        wal_snap.append_insert(ids_all[i], vectors_all[i].tolist())
        store_snap.add_vector(ids_all[i], vectors_all[i])
    wal_snap.rotate(store_snap)
    
    # Index remaining 500
    for i in range(snapshot_ops, total_ops):
        wal_snap.append_insert(ids_all[i], vectors_all[i].tolist())
    wal_snap.close()
    
    # Recover: loads 5,000 from snapshot, replays 500 from active.wal
    t_start = time.perf_counter()
    rec_store_snap = VectorStore(dimension=dimension)
    wal_rec_snap = WALManager(snap_dir)
    wal_rec_snap.replay(rec_store_snap)
    snap_recovery_ms = (time.perf_counter() - t_start) * 1000.0
    wal_rec_snap.close()
    
    print(f"   Pure WAL recovery (5,500 logs):     {pure_recovery_ms:8.2f} ms")
    print(f"   Snapshot + WAL recovery (5k+500):    {snap_recovery_ms:8.2f} ms")
    
    # 4. Generate the Markdown Report
    report_path = os.path.normpath(os.path.join(script_dir, "..", "wal_benchmark_report.md"))
    
    report_content = f"""# Write-Ahead Logging (WAL) Benchmark & Architecture Report

This report documents the architecture design, crash recovery workflow, and performance benchmarks of the **Write-Ahead Logging (WAL)** durability framework implemented in the Distributed Vector Search Engine.

---

## 1. WAL Architecture Design

The system employs a Write-Ahead Logging (WAL) protocol to guarantee **durability** and **crash recovery**. Before any mutation (vector insertion, deletion, or index clearing) is committed to the in-memory `VectorStore`, it must be written to an append-only WAL log and forced to disk using `fsync`.

### 1.1. Ingestion Flow (Durability Constraint)
```mermaid
graph TD
    Client[Client / Coordinator] -->|Mutation Request| Worker[Search Worker Node]
    subgraph Worker Node
        Worker -->|1. Append JSON Line| WAL[Write-Ahead Log: active.wal]
        WAL -->|2. Force sync to disk| Disk[(Persistent Disk)]
        Disk -->|3. Success Ack| Memory[VectorStore: In-Memory Index]
        Memory -->|4. Return Success| Worker
    end
```

### 1.2. Snapshot & WAL Rotation
To prevent unbounded log growth and fast restart times, the index supports WAL Rotation:
1. The active memory state of the `VectorStore` is saved to a compressed, secure ZIP snapshot (`snapshot.npz`).
2. The active WAL log (`active.wal`) is truncated atomically to `0` bytes.
3. Subsequent operations continue appending to the fresh, empty `active.wal`.

---

## 2. Crash Recovery Sequence

On node start, the worker reconstructs its state by:
1. Re-loading the last compact snapshot file (`snapshot.npz`) if present.
2. Replaying any operations recorded in the active WAL log (`active.wal`) that occurred post-snapshot.

```mermaid
sequenceDiagram
    autonumber
    actor System as System Restart / Process Boot
    participant Worker as Search Worker Node
    participant WAL as WALManager
    participant Disk as Persistent Disk
    participant Store as VectorStore

    System->>Worker: Starts Search Worker Process
    Worker->>WAL: Initialize and request state recovery (replay)
    WAL->>Disk: Check if snapshot.npz exists
    alt snapshot.npz exists
        Disk->>WAL: Return snapshot.npz binary data
        WAL->>Store: Load snapshot state (np.load)
    else snapshot.npz not found
        WAL->>Store: Initialize empty VectorStore
    end
    WAL->>Disk: Read active.wal JSON lines
    loop Each logged mutation in active.wal
        Disk->>WAL: Parse JSON line entry (timestamp, op, data)
        alt Checksum / JSON Valid
            alt op is insert
                WAL->>Store: add_vector(id, vector, metadata)
            else op is delete
                WAL->>Store: remove_vector(id)
            else op is clear
                WAL->>Store: reset memory structures
            end
        else Line Corrupted / Invalid Checksum
            WAL->>WAL: Skip line, log warning to stderr
        end
    end
    WAL->>Worker: Recovery complete (VectorStore rebuilt)
    Worker->>System: Node Ready (Health Check -> Healthy)
```

---

## 3. Recovery Performance Benchmarks

### 3.1. Evaluation Platform
- **Disk Sync**: fsync enabled per record mutation
- **Vector Space Dimension**: {dimension}

### 3.2. Ingestion Latency Overhead
Logging mutations to disk with synchronous `fsync` introduces storage bottlenecks. Below is the comparative insert latency across 1,000 insert mutations:

| Configuration | Total Ingestion Time | Average Latency per Insert | Ingestion Overhead |
| :--- | :---: | :---: | :---: |
| **Memory-Only Ingest** | {mem_total_time:.2f} ms | {mem_avg_latency:.4f} ms | Baseline |
| **WAL Enabled (fsync)** | {wal_total_time:.2f} ms | {wal_avg_latency:.4f} ms | {wal_avg_latency / mem_avg_latency:.1f}x |

*Production Recommendation*: In high-throughput architectures, the overhead of individual synchronous disk flushes can be mitigated using client-side batching or asynchronous buffer flushes with group-commit scheduling.

### 3.3. Replay & State Reconstruction Speeds
Measuring the time to read, parse, and commit mutations from the active WAL log onto a fresh `VectorStore` instance:

| Number of Log Mutations | Recovery Time (ms) | Recovery Speed (ops/sec) |
| :---: | :---: | :---: |
| **100** | {recovery_times[100]:.2f} ms | {100 / (recovery_times[100] / 1000.0):.1f} ops/sec |
| **1,000** | {recovery_times[1000]:.2f} ms | {1000 / (recovery_times[1000] / 1000.0):.1f} ops/sec |
| **5,000** | {recovery_times[5000]:.2f} ms | {5000 / (recovery_times[5000] / 1000.0):.1f} ops/sec |

### 3.4. Snapshot Optimization Comparison
Reconstructing a partition state of {total_ops:,} vectors:
1. **Full Log Replay**: Parsing and loading all {total_ops:,} JSON lines from the active log.
2. **Snapshot + Replay**: Loading a compressed snapshot (`snapshot.npz`) containing {snapshot_ops:,} vectors and replaying only the remaining {rem_ops} mutations from the active log.

| Recovery Strategy | Snapshot Size | Replay Size | Recovery Time (ms) | Speed Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Full Log Replay** | 0 vectors | {total_ops:,} vectors | {pure_recovery_ms:.2f} ms | Baseline |
| **Snapshot + WAL** | {snapshot_ops:,} vectors | {rem_ops} vectors | {snap_recovery_ms:.2f} ms | **{pure_recovery_ms / snap_recovery_ms:.2f}x faster** |

*Analysis*: Utilizing pre-built NumPy ZIP matrices bypassing line parses, JSON parsing, and validation logic during recovery yields a **{pure_recovery_ms / snap_recovery_ms:.1f}x speedup**, keeping boot times sub-second even for large indexes.
"""
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"\nSaved benchmark report to: {os.path.abspath(report_path)}")
    
    # Save a copy to the artifact directory
    artifact_dir = "/Users/irfanahmedshaikh/.gemini/antigravity-ide/brain/bead855d-04a9-494a-a262-9393a67ee79e"
    artifact_path = os.path.join(artifact_dir, "wal_benchmark_report.md")
    os.makedirs(artifact_dir, exist_ok=True)
    with open(artifact_path, "w") as f:
        f.write(report_content)
    print(f"Saved copy to artifact path: {artifact_path}")
    
    # Clean up benchmark directory
    if os.path.exists(benchmark_dir):
        shutil.rmtree(benchmark_dir)
        print("Cleaned up benchmark logs directory.")

if __name__ == "__main__":
    run_benchmarks()
