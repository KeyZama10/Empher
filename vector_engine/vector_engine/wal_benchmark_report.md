# Write-Ahead Logging (WAL) Benchmark Report

This document reports the performance metrics of the **Write-Ahead Logging (WAL)** system, measuring insert latency overhead and state reconstruction recovery speeds.

## Evaluation Platform & Hardware
- **Disk Sync**: fsync enabled per record mutation
- **Vector Space Dimension**: 128

---

## 1. Insert Latency Overhead

Adding WAL logging with synchronous flushing introduces disk I/O overhead. This is evaluated by performing 1,000 insert mutations:

| Configuration | Total Time | Average Latency per Insert | Write Overhead Factor |
| :--- | :---: | :---: | :---: |
| **Memory-Only Ingest** | 2.47 ms | 0.0025 ms | Baseline |
| **WAL Enabled (fsync)** | 160.68 ms | 0.1607 ms | 65.1x |

*Analysis*: Enabling WAL with synchronous `fsync` raises latency due to disk write constraints. In high-concurrency systems, network throughput can be recovered using batching (InsertVectors gRPC stream) or buffer writing with periodic group commits.

---

## 2. Replay & Recovery Times

Replay time is evaluated by reading and executing mutations from the active WAL log onto a fresh VectorStore instance:

| Number of Log Mutations | Recovery Time (ms) | Recovery Speed (ops/sec) |
| :---: | :---: | :---: |
| **100** | 5.29 ms | 18891.5 ops/sec |
| **1,000** | 50.99 ms | 19611.5 ops/sec |
| **5,000** | 256.43 ms | 19498.1 ops/sec |

---

## 3. Snapshot Rotation vs. Full Log Replay

We compare recovery times when restoring a cluster partition of 5,500 vectors:
1. **Full Log Replay**: Reading and parsing all 5,500 JSON lines.
2. **Snapshot + Replay**: Loading a snapshot of 5,000 vectors (pre-built NPZ matrix) and replaying only the 500 subsequent mutations.

| Recovery Strategy | Snapshot Size | Replay Size | Recovery Time (ms) | Speed Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Full Log Replay** | 0 vectors | 5,500 vectors | 284.85 ms | Baseline |
| **Snapshot + WAL** | 5,000 vectors | 500 vectors | 47.97 ms | 5.94x faster |

*Analysis*: Snapshot rotation dramatically increases recovery speeds (yielding a **5.9x speedup**). Loading pre-built NumPy matrix structures bypasses loop parses and Pydantic/JSON validation logic, keeping startup fast.
