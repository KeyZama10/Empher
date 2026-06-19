import os
import sys
import time
import subprocess
import shutil
import asyncio
import psutil
import numpy as np
from typing import List, Dict, Any, Tuple
from vector_engine.app.coordinator import Coordinator
from vector_engine.app.proto import engine_pb2, engine_pb2_grpc

def generate_data(n_vectors: int, dimension: int) -> Tuple[List[str], List[List[float]]]:
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    
    ids = [f"vec_{i}" for i in range(n_vectors)]
    return ids, vectors.tolist()

async def run_replication_benchmark_async() -> Dict[str, Any]:
    print("   [Replication Benchmark] Booting Primary (50111) and Replica (50112)...")
    
    ports = [50111, 50112]
    for port in ports:
        path = f"vector_engine/data/worker_{port}"
        if os.path.exists(path):
            shutil.rmtree(path)
            
    mock_file = "vector_engine/data/service_discovery_mock.json"
    if os.path.exists(mock_file):
        try:
            os.remove(mock_file)
        except Exception:
            pass

    # Start Replica first
    p_rep = subprocess.Popen([
        sys.executable, "-m", "vector_engine.app.worker",
        "--port", "50112",
        "--role", "replica",
        "--primary-addr", "localhost:50111"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Start Primary configured with replica 50112
    p_pri = subprocess.Popen([
        sys.executable, "-m", "vector_engine.app.worker",
        "--port", "50111",
        "--role", "primary",
        "--replicas", "localhost:50112"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    await asyncio.sleep(3.5) # Wait for registry boot
    
    coord = Coordinator()
    coord.start()
    
    await asyncio.sleep(2.5) # Wait for sync
    
    ids, vectors = generate_data(100, 64)
    
    # 1. Measure Replication Lag
    t_write_start = time.perf_counter()
    await coord.insert_vector(ids[0], vectors[0])
    
    # Poll replica until it has the vector
    import grpc.aio
    channel_rep = grpc.aio.insecure_channel("localhost:50112")
    stub_rep = engine_pb2_grpc.SearchWorkerStub(channel_rep)
    
    rep_size = 0
    t_replicated = time.perf_counter()
    for _ in range(50):
        try:
            resp = await stub_rep.HealthCheck(engine_pb2.HealthRequest(), timeout=0.1)
            if resp.size == 1:
                t_replicated = time.perf_counter()
                rep_size = 1
                break
        except Exception:
            pass
        await asyncio.sleep(0.01)
        
    await channel_rep.close()
    replication_lag_ms = (t_replicated - t_write_start) * 1000.0
    
    # Ingest remaining vectors
    for i in range(1, 100):
        await coord.insert_vector(ids[i], vectors[i])
    await asyncio.sleep(1.0) # Wait for replication
    
    # Generate search queries
    n_queries = 100
    queries = np.random.randn(n_queries, 64).astype(np.float32)
    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1.0
    queries = (queries / q_norms).tolist()
    
    # 2. Benchmark Case A: Replica Reads (balanced reads)
    latencies_replica = []
    t_start = time.perf_counter()
    for q in queries:
        t_q0 = time.perf_counter()
        await coord.search(q, top_k=5)
        latencies_replica.append((time.perf_counter() - t_q0) * 1000.0)
    total_time_rep = time.perf_counter() - t_start
    replica_qps = n_queries / total_time_rep
    rep_p50 = np.percentile(latencies_replica, 50)
    rep_p95 = np.percentile(latencies_replica, 95)
    rep_p99 = np.percentile(latencies_replica, 99)
    
    # 3. Benchmark Case B: Primary-Only Reads (simulate replica offline, forces primary routing)
    coord.workers["localhost:50112"]["is_active"] = False
    
    latencies_primary = []
    t_start = time.perf_counter()
    for q in queries:
        t_q0 = time.perf_counter()
        await coord.search(q, top_k=5)
        latencies_primary.append((time.perf_counter() - t_q0) * 1000.0)
    total_time_pri = time.perf_counter() - t_start
    primary_qps = n_queries / total_time_pri
    pri_p50 = np.percentile(latencies_primary, 50)
    pri_p95 = np.percentile(latencies_primary, 95)
    pri_p99 = np.percentile(latencies_primary, 99)
    
    # Teardown
    await coord.close_all_channels()
    p_pri.terminate()
    p_rep.terminate()
    p_pri.wait()
    p_rep.wait()
    
    for port in ports:
        path = f"vector_engine/data/worker_{port}"
        if os.path.exists(path):
            shutil.rmtree(path)
            
    if os.path.exists(mock_file):
        try:
            os.remove(mock_file)
        except Exception:
            pass
            
    return {
        "replication_lag_ms": replication_lag_ms if rep_size == 1 else -1,
        "primary_reads_qps": primary_qps,
        "primary_reads_latency_p50_ms": pri_p50,
        "primary_reads_latency_p95_ms": pri_p95,
        "primary_reads_latency_p99_ms": pri_p99,
        "replica_reads_qps": replica_qps,
        "replica_reads_latency_p50_ms": rep_p50,
        "replica_reads_latency_p95_ms": rep_p95,
        "replica_reads_latency_p99_ms": rep_p99
    }

def run_benchmark() -> Dict[str, Any]:
    return asyncio.run(run_replication_benchmark_async())

if __name__ == "__main__":
    res = run_benchmark()
    import json
    print(json.dumps(res, indent=2))
