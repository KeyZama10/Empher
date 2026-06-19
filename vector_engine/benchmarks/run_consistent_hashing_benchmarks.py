import os
import sys
import hashlib
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from typing import List, Dict
from vector_engine.app.distributed.hash_ring import HashRing

def get_modulo_node(key: str, nodes: List[str]) -> str:
    """Standard Modulo-based sharding key router."""
    h = int(hashlib.sha256(key.encode('utf-8')).hexdigest(), 16)
    idx = h % len(nodes)
    return nodes[idx]

def run_simulation() -> None:
    print("=============================================================")
    print("      CONSISTENT HASHING KEY MOVEMENT BENCHMARK SIMULATION   ")
    print("=============================================================")
    
    # Configuration
    n_keys = 10000
    keys = [f"vec_{i}" for i in range(n_keys)]
    
    initial_nodes = ["worker_1", "worker_2", "worker_3", "worker_4"]
    remaining_nodes = ["worker_1", "worker_2", "worker_3"]
    removed_node = "worker_4"
    
    # 1. Simulate Modulo Hashing
    print("\n1. Running Modulo Hashing Ingestion...")
    modulo_initial = {key: get_modulo_node(key, initial_nodes) for key in keys}
    
    print("   Simulating worker_4 failure...")
    modulo_post = {key: get_modulo_node(key, remaining_nodes) for key in keys}
    
    modulo_moved = sum(1 for key in keys if modulo_initial[key] != modulo_post[key])
    modulo_pct = (modulo_moved / n_keys) * 100.0
    print(f"   Modulo Hashing Key Movement: {modulo_moved} keys ({modulo_pct:.2f}%)")
    
    # Calculate initial distribution for Modulo
    modulo_counts = Counter(modulo_initial.values())
    modulo_dist = {node: modulo_counts.get(node, 0) for node in initial_nodes}
    modulo_std = np.std(list(modulo_dist.values()))
    
    # 2. Simulate Consistent Hashing
    print("\n2. Running Consistent Hashing Ingestion (100 replicas)...")
    ring = HashRing(replicas=100)
    for node in initial_nodes:
        ring.add_node(node)
        
    ring_initial = {key: ring.get_node(key) for key in keys}
    
    print("   Simulating worker_4 failure...")
    ring.remove_node(removed_node)
    
    ring_post = {key: ring.get_node(key) for key in keys}
    
    ring_moved = sum(1 for key in keys if ring_initial[key] != ring_post[key])
    ring_pct = (ring_moved / n_keys) * 100.0
    print(f"   Consistent Hashing Key Movement: {ring_moved} keys ({ring_pct:.2f}%)")
    
    # Calculate initial distribution for Consistent Hashing
    # Recreate ring with all nodes to check initial distribution
    full_ring = HashRing(replicas=100)
    for node in initial_nodes:
        full_ring.add_node(node)
    ring_initial_counts = Counter(full_ring.get_node(key) for key in keys)
    ring_dist = {node: ring_initial_counts.get(node, 0) for node in initial_nodes}
    ring_std = np.std(list(ring_dist.values()))
    
    print("\n=============================================================")
    print("                        COMPARATIVE SUMMARY                  ")
    print("=============================================================")
    print(f"{'Hashing Strategy':<25} | {'Initial':<8} | {'Post-Fail':<9} | {'Keys Moved':<10} | {'Movement %':<10}")
    print("-" * 72)
    print(f"{'Modulo Hashing':<25} | {len(initial_nodes):<8} | {len(remaining_nodes):<9} | {modulo_moved:<10} | {modulo_pct:.2f}%")
    print(f"{'Consistent Hashing':<25} | {len(initial_nodes):<8} | {len(remaining_nodes):<9} | {ring_moved:<10} | {ring_pct:.2f}%")
    print("=============================================================")
    
    # 3. Generate Visualizations
    print("\n3. Generating Consistent Hashing Visualization plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Key Distribution Across Workers
    x = np.arange(len(initial_nodes))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, [modulo_dist[node] for node in initial_nodes], width, label=f'Modulo Hashing (std={modulo_std:.1f})', color='#e06666')
    rects2 = ax1.bar(x + width/2, [ring_dist[node] for node in initial_nodes], width, label=f'Consistent Hashing (std={ring_std:.1f})', color='#6fa8dc')
    
    ax1.set_ylabel('Number of Keys Allocated', fontsize=11)
    ax1.set_title('Initial Key Distribution (10,000 Keys)', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(initial_nodes, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=10)
    
    # Plot 2: Key Movement Comparison
    strategies = ['Modulo Hashing', 'Consistent Hashing']
    movements = [modulo_pct, ring_pct]
    colors = ['#cc0000', '#3d85c6']
    
    bars = ax2.bar(strategies, movements, color=colors, width=0.4)
    ax2.set_ylabel('Key Movement %', fontsize=11)
    ax2.set_title('Key Movement % on Worker Failover', fontsize=13, fontweight='bold', pad=10)
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Target threshold line (30%)
    ax2.axhline(y=30.0, color='r', linestyle='--', linewidth=2, label='Target Threshold (<30%)')
    ax2.legend(fontsize=10)
    
    # Values on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    
    # Save the output images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_png = os.path.join(script_dir, "consistent_hashing_benchmarks.png")
    plt.savefig(output_png, dpi=150)
    
    artifact_dir = "/Users/irfanahmedshaikh/.gemini/antigravity-ide/brain/bead855d-04a9-494a-a262-9393a67ee79e"
    artifact_png = os.path.join(artifact_dir, "consistent_hashing_benchmarks.png")
    os.makedirs(artifact_dir, exist_ok=True)
    plt.savefig(artifact_png, dpi=150)
    plt.close()
    
    print(f"   Saved benchmark visualization to: {output_png}")
    print(f"   Saved artifact visualization to: {artifact_png}")
    
    # 4. Generate the Markdown Report
    report_path = os.path.normpath(os.path.join(script_dir, "..", "consistent_hashing_report.md"))
    
    report_content = fr"""# Consistent Hashing Benchmark Report

This report documents the sharding behavior, key distribution uniformity, and key reallocation patterns during failover events. It compares **Modulo-based Sharding** against **Consistent Hashing** with virtual nodes.

## Failure Simulation Setup
- **Keys Evaluated**: {n_keys:,} synthetic vector identifiers (`vec_0` to `vec_{n_keys-1}`)
- **Initial Nodes**: 4 workers (`worker_1`, `worker_2`, `worker_3`, `worker_4`)
- **Action**: Simulate sudden failure/removal of `worker_4`, reducing cluster node count to 3 workers.
- **Virtual Nodes**: 100 virtual nodes per physical worker node.

---

## 1. Key Distribution Uniformity

The table below shows the initial key distribution across all 4 worker nodes under both sharding strategies:

| Worker Node | Modulo Key Count | Modulo Share % | Consistent Hashing Key Count | Consistent Hashing Share % |
| :--- | :---: | :---: | :---: | :---: |
| **worker_1** | {modulo_dist['worker_1']:,} | {modulo_dist['worker_1']/n_keys*100:.2f}% | {ring_dist['worker_1']:,} | {ring_dist['worker_1']/n_keys*100:.2f}% |
| **worker_2** | {modulo_dist['worker_2']:,} | {modulo_dist['worker_2']/n_keys*100:.2f}% | {ring_dist['worker_2']:,} | {ring_dist['worker_2']/n_keys*100:.2f}% |
| **worker_3** | {modulo_dist['worker_3']:,} | {modulo_dist['worker_3']/n_keys*100:.2f}% | {ring_dist['worker_3']:,} | {ring_dist['worker_3']/n_keys*100:.2f}% |
| **worker_4** | {modulo_dist['worker_4']:,} | {modulo_dist['worker_4']/n_keys*100:.2f}% | {ring_dist['worker_4']:,} | {ring_dist['worker_4']/n_keys*100:.2f}% |
| **Std Dev** | **{modulo_std:.2f}** | - | **{ring_std:.2f}** | - |

- **Observation**: Both Modulo and Consistent Hashing (using 100 virtual nodes per worker) achieve a highly uniform key distribution, with key shares hovering closely around the ideal **25.00%** mark.

---

## 2. Key Movement comparison on Failover

When `worker_4` fails, keys must be reallocated. Consistent Hashing limits movement *only* to keys that resided on the failed worker.

| Hashing Strategy | Initial Nodes | Remaining Nodes | Keys Moved | Key Movement % | Target Met |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Modulo Hashing** | 4 | 3 | {modulo_moved:,} | {modulo_pct:.2f}% | ❌ No (Limit: <30%) |
| **Consistent Hashing** | 4 | 3 | {ring_moved:,} | {ring_pct:.2f}% |  Yes (< 30%) |

---

## 3. Visualizations

Here is a side-by-side visualization of initial load distribution and key movement compared to the allowable target threshold (30%):

![Consistent Hashing Benchmark Visualization](file://{artifact_png})

---

## 4. Production Impact & Trade-offs

1. **Modulo Hashing Inefficiency**: Removing or adding a node changes the hash partition formula ($hash(key) \pmod N$). This forces **{modulo_pct:.2f}%** of the database index/cache keys to change locations. In a large production vector database, this creates massive network/IO storms and massive lookup latency spikes (cache miss storms) as workers reshuffle data.
2. **Consistent Hashing Minimization**: Consistent Hashing maps keys and workers onto a circular ring. Removing `worker_4` only moves the keys belonging to `worker_4` to its successor on the ring. Key movement is limited to **{ring_pct:.2f}%**, closely matching the theoretical minimum of $1/N = 25.00\%$. All other keys remain routed to their original healthy workers.
3. **Virtual Nodes Customization**: By setting `replicas=100` (virtual nodes per physical node), we smooth out partition boundaries on the ring. If virtual nodes are set too low (e.g., < 10), standard deviation increases, leading to hotspots. Setting it to 100-200 vnodes ensures stable load uniformity while preserving low metadata lookup overhead.
"""
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"\nSaved benchmark report to: {os.path.abspath(report_path)}")
    
    # Assert target constraint
    assert ring_pct < 30.0, f"Error: Consistent Hashing key movement exceeded target: {ring_pct:.2f}%"

if __name__ == "__main__":
    run_simulation()
