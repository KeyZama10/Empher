import os
import sys
import time
import psutil
import subprocess
import shutil
import asyncio
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

async def run_distributed_nodes_benchmark(n_workers: int) -> Dict[str, Any]:
    print(f"   [Scaling Benchmark] Evaluating {n_workers} Workers...")
    
    # 1. Start worker nodes on ports 50101 to 50100+N
    ports = [50100 + i for i in range(1, n_workers + 1)]
    processes = []
    
    # Clear old WAL logs
    for port in ports:
        path = f"vector_engine/data/worker_{port}"
        if os.path.exists(path):
            shutil.rmtree(path)
            
    # Clear mock file
    mock_file = "vector_engine/data/service_discovery_mock.json"
    if os.path.exists(mock_file):
        try:
            os.remove(mock_file)
        except Exception:
            pass

    for port in ports:
        p = subprocess.Popen([
            sys.executable, "-m", "vector_engine.app.worker",
            "--port", str(port),
            "--role", "primary"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)
        
    await asyncio.sleep(3.0) # Wait for registry boot
    
    coord = Coordinator()
    coord.start()
    
    await asyncio.sleep(2.0) # Wait for sync
    
    # Generate data
    n_vectors = 1000
    ids, vectors = generate_data(n_vectors, 64)
    
    # Ingestion Benchmark
    process = psutil.Process()
    process.cpu_percent(interval=None)
    t0 = time.perf_counter()
    for i in range(n_vectors):
        await coord.insert_vector(ids[i], vectors[i])
    ingest_time = time.perf_counter() - t0
    ingest_qps = n_vectors / ingest_time
    ingest_cpu = process.cpu_percent(interval=None)
    
    # Query Benchmark (100 queries)
    n_queries = 100
    queries = np.random.randn(n_queries, 64).astype(np.float32)
    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1.0
    queries = (queries / q_norms).tolist()
    
    latencies = []
    process.cpu_percent(interval=None)
    t_start = time.perf_counter()
    for q in queries:
        t_q0 = time.perf_counter()
        await coord.search(q, top_k=5)
        latencies.append((time.perf_counter() - t_q0) * 1000.0) # ms
    total_query_time = time.perf_counter() - t_start
    query_cpu = process.cpu_percent(interval=None)
    
    qps = n_queries / total_query_time
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    # Measure WAL Recovery Time for 1 node failure
    recovery_time_ms = 0.0
    if n_workers > 1:
        # Kill last worker
        killed_port = ports[-1]
        processes[-1].terminate()
        processes[-1].wait()
        
        # Start it back up and time the recovery (re-loading and replaying)
        t_rec_start = time.perf_counter()
        p_new = subprocess.Popen([
            sys.executable, "-m", "vector_engine.app.worker",
            "--port", str(killed_port),
            "--role", "primary"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes[-1] = p_new
        
        # Poll health endpoint until healthy and has recovered its size
        import grpc.aio
        channel = grpc.aio.insecure_channel(f"localhost:{killed_port}")
        stub = engine_pb2_grpc.SearchWorkerStub(channel)
        
        for _ in range(50):
            try:
                resp = await stub.HealthCheck(engine_pb2.HealthRequest(), timeout=0.2)
                if resp.status_code == "healthy" and resp.size > 0:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.1)
            
        recovery_time_ms = (time.perf_counter() - t_rec_start) * 1000.0
        await channel.close()
        
    # Teardown
    await coord.close_all_channels()
    for p in processes:
        p.terminate()
        p.wait()
        
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
        "n_workers": n_workers,
        "ingest_qps": ingest_qps,
        "ingest_cpu": ingest_cpu,
        "search_qps": qps,
        "search_latency_p50_ms": p50,
        "search_latency_p95_ms": p95,
        "search_latency_p99_ms": p99,
        "query_cpu": query_cpu,
        "recovery_time_ms": recovery_time_ms
    }

async def run_benchmark_async() -> List[Dict[str, Any]]:
    results = []
    # Test configurations of 1, 2, and 4 nodes
    for n in [1, 2, 4]:
        res = await run_distributed_nodes_benchmark(n)
        results.append(res)
    return results

def run_benchmark() -> List[Dict[str, Any]]:
    return asyncio.run(run_benchmark_async())

if __name__ == "__main__":
    res = run_benchmark()
    import json
    print(json.dumps(res, indent=2))
