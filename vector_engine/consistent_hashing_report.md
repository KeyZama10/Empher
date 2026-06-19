# Consistent Hashing Benchmark Report

This report documents the sharding behavior, key distribution uniformity, and key reallocation patterns during failover events. It compares **Modulo-based Sharding** against **Consistent Hashing** with virtual nodes.

## Failure Simulation Setup
- **Keys Evaluated**: 10,000 synthetic vector identifiers (`vec_0` to `vec_9999`)
- **Initial Nodes**: 4 workers (`worker_1`, `worker_2`, `worker_3`, `worker_4`)
- **Action**: Simulate sudden failure/removal of `worker_4`, reducing cluster node count to 3 workers.
- **Virtual Nodes**: 100 virtual nodes per physical worker node.

---

## 1. Key Distribution Uniformity

The table below shows the initial key distribution across all 4 worker nodes under both sharding strategies:

| Worker Node | Modulo Key Count | Modulo Share % | Consistent Hashing Key Count | Consistent Hashing Share % |
| :--- | :---: | :---: | :---: | :---: |
| **worker_1** | 2,492 | 24.92% | 2,636 | 26.36% |
| **worker_2** | 2,480 | 24.80% | 2,319 | 23.19% |
| **worker_3** | 2,506 | 25.06% | 2,723 | 27.23% |
| **worker_4** | 2,522 | 25.22% | 2,322 | 23.22% |
| **Std Dev** | **15.68** | - | **182.12** | - |

- **Observation**: Both Modulo and Consistent Hashing (using 100 virtual nodes per worker) achieve a highly uniform key distribution, with key shares hovering closely around the ideal **25.00%** mark.

---

## 2. Key Movement comparison on Failover

When `worker_4` fails, keys must be reallocated. Consistent Hashing limits movement *only* to keys that resided on the failed worker.

| Hashing Strategy | Initial Nodes | Remaining Nodes | Keys Moved | Key Movement % | Target Met |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Modulo Hashing** | 4 | 3 | 7,460 | 74.60% | ❌ No (Limit: <30%) |
| **Consistent Hashing** | 4 | 3 | 2,322 | 23.22% |  Yes (< 30%) |

---

## 3. Visualizations

Here is a side-by-side visualization of initial load distribution and key movement compared to the allowable target threshold (30%):

![Consistent Hashing Benchmark Visualization](file:///Users/irfanahmedshaikh/.gemini/antigravity-ide/brain/bead855d-04a9-494a-a262-9393a67ee79e/consistent_hashing_benchmarks.png)

---

## 4. Production Impact & Trade-offs

1. **Modulo Hashing Inefficiency**: Removing or adding a node changes the hash partition formula ($hash(key) \pmod N$). This forces **74.60%** of the database index/cache keys to change locations. In a large production vector database, this creates massive network/IO storms and massive lookup latency spikes (cache miss storms) as workers reshuffle data.
2. **Consistent Hashing Minimization**: Consistent Hashing maps keys and workers onto a circular ring. Removing `worker_4` only moves the keys belonging to `worker_4` to its successor on the ring. Key movement is limited to **23.22%**, closely matching the theoretical minimum of $1/N = 25.00\%$. All other keys remain routed to their original healthy workers.
3. **Virtual Nodes Customization**: By setting `replicas=100` (virtual nodes per physical node), we smooth out partition boundaries on the ring. If virtual nodes are set too low (e.g., < 10), standard deviation increases, leading to hotspots. Setting it to 100-200 vnodes ensures stable load uniformity while preserving low metadata lookup overhead.
