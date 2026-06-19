import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Path configurations
RESULTS_JSON_PATH = "docs/benchmarks/benchmark_results.json"
PLOTS_DIR = "docs/benchmarks/plots"

def load_telemetry():
    if not os.path.exists(RESULTS_JSON_PATH):
        raise FileNotFoundError(f"Benchmark results file not found at {RESULTS_JSON_PATH}")
    with open(RESULTS_JSON_PATH, "r") as f:
        return json.load(f)

def setup_style():
    # Apply clean professional styling parameters
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["grid.color"] = "#EBEBEB"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.8
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["figure.titlesize"] = 14

def save_fig(fig, filename):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path}")

def plot_recall_vs_latency(data):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.grid(True)
    
    # 1. Extract HNSW, IVF, PQ
    hnsw = data["hnsw_metrics"]
    ivf = data["ivf_metrics"]
    pq = data["pq_metrics"]
    
    ax.scatter(hnsw["recall_at_10"], hnsw["latency_p95_ms"], color="#1F77B4", marker="o", s=150, label="HNSW Index", zorder=5)
    ax.scatter(ivf["recall_at_10"], ivf["latency_p95_ms"], color="#2CA02C", marker="s", zorder=5, s=120, label="IVF Index (nprobe=2)")
    ax.scatter(pq["recall_at_10"], pq["latency_p95_ms"], color="#9467BD", marker="D", zorder=5, s=120, label="Product Quantization")
    
    # 2. Extract nprobe sweep points
    nprobe_recalls = [pt["recall_at_10"] for pt in data["nprobe_metrics"]]
    nprobe_latencies = [pt["latency_p95_ms"] for pt in data["nprobe_metrics"]]
    nprobes = [pt["nprobe"] for pt in data["nprobe_metrics"]]
    
    ax.plot(nprobe_recalls, nprobe_latencies, color="#2CA02C", linestyle="--", alpha=0.7, zorder=3)
    ax.scatter(nprobe_recalls, nprobe_latencies, color="#2CA02C", marker="^", s=80, label="IVF nprobe Sweep", zorder=4)
    
    # Annotate points
    ax.annotate("HNSW", (hnsw["recall_at_10"], hnsw["latency_p95_ms"]), textcoords="offset points", xytext=(-10,12), ha='center', weight='bold', color="#1F77B4")
    ax.annotate("PQ", (pq["recall_at_10"], pq["latency_p95_ms"]), textcoords="offset points", xytext=(12,-15), ha='center', weight='bold', color="#9467BD")
    
    for recall, latency, nprobe in zip(nprobe_recalls, nprobe_latencies, nprobes):
        ax.annotate(f"nprobe={nprobe}", (recall, latency), textcoords="offset points", xytext=(8,5), ha='left', fontsize=8, color="#1B5E20")

    ax.set_title("Search Quality (Recall@10) vs Query Latency", pad=15)
    ax.set_xlabel("Recall@10 (Higher is Better)")
    ax.set_ylabel("Query Latency P95 (ms, Log Scale)")
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.05)
    
    # Clean up yticks to look natural
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    
    save_fig(fig, "recall_vs_latency.png")

def plot_hnsw_vs_ivf(data):
    hnsw = data["hnsw_metrics"]
    ivf = data["ivf_metrics"]
    
    metrics = ["Recall@10", "P50 Latency (ms)", "Throughput (QPS)"]
    hnsw_vals = [hnsw["recall_at_10"], hnsw["latency_p50_ms"], hnsw["qps"]]
    ivf_vals = [ivf["recall_at_10"], ivf["latency_p50_ms"], ivf["qps"]]
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    colors = ["#1F77B4", "#2CA02C"]
    
    # Subplot 1: Recall
    axes[0].grid(True, axis="y")
    axes[0].bar(["HNSW", "IVF (n=32)"], [hnsw_vals[0], ivf_vals[0]], color=colors, width=0.5, zorder=3)
    axes[0].set_title("Recall@10 (Higher is Better)")
    axes[0].set_ylim(0.0, 1.0)
    for i, v in enumerate([hnsw_vals[0], ivf_vals[0]]):
        axes[0].text(i, v + 0.02, f"{v:.4f}", ha="center", weight="bold")
        
    # Subplot 2: Latency
    axes[1].grid(True, axis="y")
    axes[1].bar(["HNSW", "IVF (n=32)"], [hnsw_vals[1], ivf_vals[1]], color=colors, width=0.5, zorder=3)
    axes[1].set_title("P50 Latency (Lower is Better)")
    axes[1].set_ylabel("Latency (ms)")
    for i, v in enumerate([hnsw_vals[1], ivf_vals[1]]):
        axes[1].text(i, v + 0.005, f"{v:.3f} ms", ha="center", weight="bold")
        
    # Subplot 3: Throughput
    axes[2].grid(True, axis="y")
    axes[2].bar(["HNSW", "IVF (n=32)"], [hnsw_vals[2], ivf_vals[2]], color=colors, width=0.5, zorder=3)
    axes[2].set_title("Throughput (Higher is Better)")
    axes[2].set_ylabel("Queries Per Second (QPS)")
    for i, v in enumerate([hnsw_vals[2], ivf_vals[2]]):
        axes[2].text(i, v + 100, f"{v:.1f}", ha="center", weight="bold")
        
    plt.suptitle("HNSW vs IVF Index Core Performance Comparison", y=1.02)
    plt.tight_layout()
    
    save_fig(fig, "hnsw_vs_ivf.png")

def plot_pq_compression(data):
    pq = data["pq_metrics"]
    ratio = pq["compression_ratio"]
    
    # Assume 100,000 vectors of 128-dimensions float32
    raw_size_mb = 100000 * 128 * 4 / (1024 * 1024)  # ~48.8 MB
    pq_size_mb = raw_size_mb / ratio
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.grid(True, axis="y")
    
    bars = ax.bar(["Raw Contiguous Float32", "Product Quantization (PQ)"], [raw_size_mb, pq_size_mb], color=["#7F7F7F", "#9467BD"], width=0.5, zorder=3)
    ax.set_title("Memory Consumption (100K Vectors, 128-Dim)")
    ax.set_ylabel("Memory Space (MB)")
    
    # Annotate bars
    ax.text(0, raw_size_mb + 1.0, f"{raw_size_mb:.2f} MB\n(1.0x Baseline)", ha="center", weight="bold")
    ax.text(1, pq_size_mb + 1.0, f"{pq_size_mb:.2f} MB\n({ratio:.1f}x Compressed)", ha="center", weight="bold", color="#7B1FA2")
    
    ax.set_ylim(0, raw_size_mb * 1.15)
    
    save_fig(fig, "pq_compression.png")

def plot_nprobe_tradeoff(data):
    nprobes = [pt["nprobe"] for pt in data["nprobe_metrics"]]
    recalls = [pt["recall_at_10"] for pt in data["nprobe_metrics"]]
    latencies = [pt["latency_p95_ms"] for pt in data["nprobe_metrics"]]
    
    fig, ax1 = plt.subplots(figsize=(7, 5))
    
    color = "#1F77B4"
    ax1.set_xlabel("IVF nprobe parameter")
    ax1.set_ylabel("Recall@10 (Higher is Better)", color=color)
    ax1.grid(True)
    line1 = ax1.plot(nprobes, recalls, color=color, marker="o", linewidth=2.5, label="Recall@10")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0.0, 1.0)
    
    ax2 = ax1.twinx()
    color = "#FF7F0E"
    ax2.set_ylabel("Query Latency P95 (ms, Lower is Better)", color=color)
    line2 = ax2.plot(nprobes, latencies, color=color, marker="s", linewidth=2.5, linestyle="--", label="P95 Latency")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.grid(False) # Turn off grid for twin to avoid overlap
    
    # Combined legend
    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right", frameon=True, facecolor="white")
    
    plt.title("IVF nprobe Parameter Recall-Latency Trade-off Curves", pad=15)
    ax1.set_xticks(nprobes)
    
    save_fig(fig, "nprobe_tradeoff.png")

def plot_distributed_scaling(data):
    scaling = data["distributed_scaling"]
    workers = [pt["n_workers"] for pt in scaling]
    ingest_qps = [pt["ingest_qps"] for pt in scaling]
    search_qps = [pt["search_qps"] for pt in scaling]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.grid(True)
    
    ax.plot(workers, ingest_qps, color="#D62728", marker="o", linewidth=2.5, label="Ingest Throughput (QPS)")
    ax.plot(workers, search_qps, color="#1F77B4", marker="s", linewidth=2.5, label="Search Throughput (QPS)")
    
    ax.set_title("Distributed Cluster Scaling Throughput Performance", pad=15)
    ax.set_xlabel("Number of Shard Worker Nodes")
    ax.set_ylabel("Throughput (Queries / Ingests Per Second)")
    ax.set_xticks(workers)
    
    ax.set_ylim(0, max(ingest_qps + search_qps) * 1.25)
    ax.legend(loc="lower left", frameon=True, facecolor="white")
    
    # Label the coordinates
    for x, y in zip(workers, ingest_qps):
        ax.annotate(f"{y:.0f} QPS", (x, y), textcoords="offset points", xytext=(0,10), ha="center", weight="bold", color="#B71C1C")
    for x, y in zip(workers, search_qps):
        ax.annotate(f"{y:.0f} QPS", (x, y), textcoords="offset points", xytext=(0,-15), ha="center", weight="bold", color="#0D47A1")
        
    save_fig(fig, "distributed_scaling.png")

def plot_replica_scaling(data):
    rep = data["replication_metrics"]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Subplot 1: Read QPS
    axes[0].grid(True, axis="y")
    bars_qps = axes[0].bar(["Primary-Only", "Replica Load-Balanced"], [rep["primary_reads_qps"], rep["replica_reads_qps"]], color=["#7F7F7F", "#1F77B4"], width=0.5, zorder=3)
    axes[0].set_title("Read Throughput Scaling (QPS)")
    axes[0].set_ylabel("Throughput (QPS)")
    for i, v in enumerate([rep["primary_reads_qps"], rep["replica_reads_qps"]]):
        axes[0].text(i, v + 50, f"{v:.1f}", ha="center", weight="bold")
    axes[0].set_ylim(0, max(rep["primary_reads_qps"], rep["replica_reads_qps"]) * 1.15)
        
    # Subplot 2: Latency P95
    axes[1].grid(True, axis="y")
    bars_lat = axes[1].bar(["Primary-Only", "Replica Load-Balanced"], [rep["primary_reads_latency_p95_ms"], rep["replica_reads_latency_p95_ms"]], color=["#7F7F7F", "#FF7F0E"], width=0.5, zorder=3)
    axes[1].set_title("P95 Read Latency Scaling (ms)")
    axes[1].set_ylabel("Latency (ms)")
    for i, v in enumerate([rep["primary_reads_latency_p95_ms"], rep["replica_reads_latency_p95_ms"]]):
        axes[1].text(i, v + 0.05, f"{v:.3f} ms", ha="center", weight="bold")
    axes[1].set_ylim(0, max(rep["primary_reads_latency_p95_ms"], rep["replica_reads_latency_p95_ms"]) * 1.2)
        
    plt.suptitle("Primary-Replica Load-Balancing Performance Impact", y=1.02)
    plt.tight_layout()
    
    save_fig(fig, "replica_scaling.png")

def plot_recovery_time(data):
    # From WAL architectural recovery log metrics
    log_sizes = [100, 1000, 5000]
    replay_times = [5.19, 52.02, 252.12]
    
    strategies = ["Full Log Replay\n(5,500 entries)", "Snapshot + WAL\n(5,000 snap + 500 log)"]
    strategy_times = [281.58, 49.80]
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    # Subplot A: Replay time vs Log size
    axes[0].grid(True, axis="y")
    axes[0].bar([str(s) for s in log_sizes], replay_times, color="#2CA02C", width=0.4, zorder=3)
    axes[0].set_title("WAL Log Replay State Reconstruction Time")
    axes[0].set_xlabel("Number of Log Mutations")
    axes[0].set_ylabel("Recovery Time (ms)")
    for i, v in enumerate(replay_times):
        axes[0].text(i, v + 5, f"{v:.2f} ms", ha="center", weight="bold")
    axes[0].set_ylim(0, max(replay_times) * 1.15)
        
    # Subplot B: Snapshot vs Raw
    axes[1].grid(True, axis="y")
    axes[1].bar(strategies, strategy_times, color=["#E377C2", "#FF7F0E"], width=0.4, zorder=3)
    axes[1].set_title("Recovery Strategy Optimization comparison")
    axes[1].set_ylabel("Recovery Time (ms)")
    for i, v in enumerate(strategy_times):
        axes[1].text(i, v + 8, f"{v:.2f} ms", ha="center", weight="bold")
    axes[1].set_ylim(0, max(strategy_times) * 1.15)
    
    # Highlight speedup
    speedup = strategy_times[0] / strategy_times[1]
    axes[1].text(0.5, (strategy_times[0] + strategy_times[1])/2, f"{speedup:.2f}x Speedup", ha="center", weight="bold", color="#D32F2F", bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
    
    plt.suptitle("Crash Recovery Durability Speeds & Optimization", y=1.02)
    plt.tight_layout()
    
    save_fig(fig, "recovery_time.png")

def main():
    setup_style()
    data = load_telemetry()
    
    plot_recall_vs_latency(data)
    plot_hnsw_vs_ivf(data)
    plot_pq_compression(data)
    plot_nprobe_tradeoff(data)
    plot_distributed_scaling(data)
    plot_replica_scaling(data)
    plot_recovery_time(data)
    
    print("All benchmark plots generated successfully!")

if __name__ == "__main__":
    main()
