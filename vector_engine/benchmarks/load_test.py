import asyncio
import time
import random
import numpy as np
import httpx
from typing import List

BASE_URL = "http://localhost:8000"
DIMENSION = 64
N_INSERTS = 100
N_QUERIES = 500
CONCURRENCY = 10

def generate_vector(dim: int) -> List[float]:
    vec = np.random.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()

async def insert_task(client: httpx.AsyncClient, index: int) -> bool:
    payload = {
        "id": f"load_vec_{index}",
        "vector": generate_vector(DIMENSION),
        "metadata": {"source": "load_tester", "index": index}
    }
    try:
        response = await client.post(f"{BASE_URL}/insert", json=payload, timeout=5.0)
        return response.status_code == 201
    except Exception:
        return False

async def query_task(client: httpx.AsyncClient) -> tuple:
    payload = {
        "vector": generate_vector(DIMENSION),
        "top_k": random.choice([5, 10, 15])
    }
    t_start = time.perf_counter()
    try:
        response = await client.post(f"{BASE_URL}/search", json=payload, timeout=5.0)
        latency = (time.perf_counter() - t_start) * 1000.0  # ms
        success = response.status_code == 200
        return success, latency
    except Exception:
        latency = (time.perf_counter() - t_start) * 1000.0
        return False, latency

async def run_load_test():
    async with httpx.AsyncClient() as client:
        # Step 1: Health check
        try:
            print("Checking if API service is running...")
            res = await client.get(f"{BASE_URL}/health", timeout=2.0)
            if res.status_code != 200:
                print("Health check failed. Ensure the server is running on port 8000.")
                return
            print("API service is healthy. Beginning load test setup...")
        except Exception as e:
            print(f"Failed to connect to API service: {e}")
            print("Please run the FastAPI server (e.g. uvicorn vector_engine.app.api:app) before launching the load test.")
            return

        # Step 2: Ingest initial vectors
        print(f"\n1. Ingesting {N_INSERTS} initial vectors...")
        tasks = [insert_task(client, i) for i in range(N_INSERTS)]
        insert_results = await asyncio.gather(*tasks)
        successful_inserts = sum(1 for r in insert_results if r)
        print(f"   Successfully ingested: {successful_inserts}/{N_INSERTS} vectors.")

        # Step 3: Run concurrent query load
        print(f"\n2. Executing {N_QUERIES} search queries with concurrency limit of {CONCURRENCY}...")
        
        # Concurrency control semaphores
        sem = asyncio.Semaphore(CONCURRENCY)
        
        async def worker():
            async with sem:
                return await query_task(client)

        t_test_start = time.perf_counter()
        
        # Dispatch queries
        query_tasks = [worker() for _ in range(N_QUERIES)]
        query_results = await asyncio.gather(*query_tasks)
        
        total_time_sec = time.perf_counter() - t_test_start
        
        # Calculate statistics
        successes = sum(1 for r in query_results if r[0])
        failures = N_QUERIES - successes
        latencies = [r[1] for r in query_results if r[0]]
        
        avg_latency = np.mean(latencies) if latencies else 0.0
        p95_latency = np.percentile(latencies, 95) if latencies else 0.0
        qps = N_QUERIES / total_time_sec
        
        print("\n==================================================")
        print("                 LOAD TEST RESULTS                ")
        print("==================================================")
        print(f"Total Queries Executed:  {N_QUERIES}")
        print(f"Successful Queries:      {successes}")
        print(f"Failed Queries:          {failures} ({failures/N_QUERIES*100.0:.2f}%)")
        print(f"Total Execution Time:    {total_time_sec:.2f} seconds")
        print(f"Throughput (QPS):        {qps:.2f} queries/sec")
        if latencies:
            print(f"Average Latency:         {avg_latency:.2f} ms")
            print(f"95th Percentile Latency: {p95_latency:.2f} ms")
        print("==================================================\n")

if __name__ == "__main__":
    asyncio.run(run_load_test())
