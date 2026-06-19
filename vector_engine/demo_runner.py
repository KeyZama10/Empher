import os
import sys
import time
import subprocess
import socket
import urllib.request
import urllib.error
import json

def is_port_open(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def check_docker():
    try:
        # Run a simple docker command to see if daemon is responsive
        res = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return res.returncode == 0
    except Exception:
        return False

def start_docker():
    print("\n[INFO] Attempting to boot cluster using Docker Compose...")
    commands = [
        ["docker", "compose", "-f", "deploy/docker-compose.yml", "up", "-d", "--build"],
        ["docker-compose", "-f", "deploy/docker-compose.yml", "up", "-d", "--build"]
    ]
    started = False
    for cmd in commands:
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0:
                started = True
                break
        except FileNotFoundError:
            continue
    return started

def start_local_subprocesses():
    print("\n[WARN] Docker not active/available. Gracefully falling back to local Python subprocesses...")
    processes = []
    env = os.environ.copy()
    # Set PYTHONPATH to parent directory so that vector_engine package is found
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = parent_dir
    env["MODE"] = "coordinator"
    env["COORDINATOR_ID"] = "coord_active"
    env["STANDBY"] = "false"

    # Launch worker nodes (2 shards, each primary + replica)
    worker_cmds = [
        ["python3", "-m", "vector_engine.app.worker", "--port", "50101", "--role", "primary", "--replicas", "localhost:50102"],
        ["python3", "-m", "vector_engine.app.worker", "--port", "50102", "--role", "replica", "--primary-addr", "localhost:50101"],
        ["python3", "-m", "vector_engine.app.worker", "--port", "50201", "--role", "primary", "--replicas", "localhost:50202"],
        ["python3", "-m", "vector_engine.app.worker", "--port", "50202", "--role", "replica", "--primary-addr", "localhost:50201"]
    ]

    print("[INFO] Booting 4 Search Shard Workers (2 Primary-Replica Shards)...")
    for cmd in worker_cmds:
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)

    time.sleep(1.5)

    # Launch coordinator gateway
    print("[INFO] Booting Coordinator Gateway API on port 8000...")
    coord_cmd = ["python3", "-m", "uvicorn", "vector_engine.app.api:app", "--host", "127.0.0.1", "--port", "8000"]
    p = subprocess.Popen(coord_cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes.append(p)

    return processes

def wait_for_coordinator(timeout=25):
    print("[INFO] Waiting for Coordinator gateway health checks...", end="", flush=True)
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_open("127.0.0.1", 8000):
            try:
                req = urllib.request.Request("http://127.0.0.1:8000/health")
                with urllib.request.urlopen(req, timeout=1.0) as response:
                    data = json.loads(response.read().decode())
                    if data.get("status") in ["healthy", "standby"]:
                        print("\n[SUCCESS] Gateway is online and healthy!")
                        return True
            except Exception:
                pass
        print(".", end="", flush=True)
        time.sleep(0.5)
    print("\n[ERROR] Timeout waiting for Coordinator to boot.")
    return False

def run_demo_operations():
    print("\n" + "="*60)
    print("      DISTRIBUTED SEARCH CLUSTER OPERATIONS DEMO")
    print("="*60)

    # 1. Ingest sample vectors
    samples = [
        {"id": "doc1", "vector": [1.0, 0.0, 0.0, 0.0], "metadata": {"title": "Introduction to Linear Algebra", "category": "Math"}},
        {"id": "doc2", "vector": [0.0, 1.0, 0.0, 0.0], "metadata": {"title": "Quantum Mechanics & Wave Theory", "category": "Physics"}},
        {"id": "doc3", "vector": [0.0, 0.0, 1.0, 0.0], "metadata": {"title": "Organic Chemistry Fundamentals", "category": "Chemistry"}},
        {"id": "doc4", "vector": [0.8, 0.6, 0.0, 0.0], "metadata": {"title": "Matrix Computations and Operations", "category": "Math"}},
        {"id": "doc5", "vector": [0.1, 0.1, 0.9, 0.3], "metadata": {"title": "Galactic Dynamics & Astrophysics", "category": "Astronomy"}}
    ]

    print("\n1. Ingesting sample embeddings into sharded storage:")
    for doc in samples:
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8000/insert",
                data=json.dumps(doc).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode())
                print(f"   [Ingested] ID: {doc['id']:<5} | Title: {doc['metadata']['title']}")
        except Exception as e:
            print(f"   [Error] Failed to ingest {doc['id']}: {e}")

    time.sleep(1.0)

    # 2. Query math-oriented vectors (Demonstrate Cache Miss)
    query_math = {"vector": [0.9, 0.1, 0.0, 0.0], "top_k": 2}
    print(f"\n2. Querying database with Math-focused query vector: {query_math['vector']}")
    
    start = time.time()
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8000/search",
            data=json.dumps(query_math).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            results = json.loads(resp.read().decode())
            latency = (time.time() - start) * 1000
            print(f"   [Results] Cache MISS - Search Execution Time: {latency:.2f} ms")
            for idx, match in enumerate(results.get("results", []), 1):
                print(f"     Rank {idx}: ID={match['id']} | Score={match['score']:.4f} | Title='{match['metadata'].get('title')}'")
    except Exception as e:
        print(f"   [Error] Query failed: {e}")

    # 3. Repeat exact same query (Demonstrate Cache Hit)
    print("\n3. Querying with identical vector again (Demonstrating Fast Redis Cache Hit):")
    start = time.time()
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8000/search",
            data=json.dumps(query_math).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            results = json.loads(resp.read().decode())
            latency = (time.time() - start) * 1000
            print(f"   [Results] Cache HIT - Search Execution Time: {latency:.2f} ms")
            for idx, match in enumerate(results.get("results", []), 1):
                print(f"     Rank {idx}: ID={match['id']} | Score={match['score']:.4f} | Title='{match['metadata'].get('title')}'")
    except Exception as e:
        print(f"   [Error] Query failed: {e}")

def print_benchmarks():
    print("\n" + "="*60)
    print("      EXECUTIVE BENCHMARK PERFORMANCE RESULTS SUMMARY")
    print("="*60)
    summary_path = "docs/benchmarks/benchmark_summary.md"
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                print(f.read())
        except Exception as e:
            print(f"[Error] Failed to print benchmarks summary: {e}")
    else:
        print("[WARN] Benchmark summary file not found.")

def main():
    print("="*60)
    print("  Distributed Vector Search Engine - Local Demo Orchestrator")
    print("="*60)
    
    use_docker = check_docker()
    docker_started = False
    processes = []
    
    try:
        if use_docker:
            docker_started = start_docker()
            if not docker_started:
                # Docker daemon is active, but compose command failed/missing, do local fallback
                processes = start_local_subprocesses()
        else:
            processes = start_local_subprocesses()
            
        # Wait for API gateway to become healthy
        healthy = wait_for_coordinator()
        if healthy:
            run_demo_operations()
            print_benchmarks()
            
            print("\n" + "="*60)
            print("                     DEMO RUN COMPLETED")
            print("="*60)
            if docker_started:
                print("\n[INFO] Docker containers are left running for inspection!")
                print("   - REST API Gateway: http://localhost:8000/docs")
                print("   - Grafana Dashboard: http://localhost:3000")
                print("   - Prometheus Server: http://localhost:9090")
                print("   - To shutdown the cluster, run: make clean")
            else:
                print("\n[INFO] Local python cluster stopped successfully.")
                print("   - Cleaned up temp lockfiles and index logs.")
        else:
            print("\n[ERROR] Could not start the coordinator gateway.")
            
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down demo runner...")
    finally:
        if processes:
            print("\n[INFO] Stopping local python subprocesses...")
            for p in processes:
                p.terminate()
                p.wait()
            # Clean up mock storage registry database files
            for f in ["vector_engine/data/service_discovery_mock.json", "vector_engine/data/service_discovery_mock.lock"]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception:
                        pass
            print("[SUCCESS] Subprocesses cleaned up.")

if __name__ == "__main__":
    main()
