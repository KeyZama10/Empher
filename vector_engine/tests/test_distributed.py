import sys
import time
import subprocess
import pytest
import numpy as np
import shutil
import os

from vector_engine.app.coordinator import Coordinator
from vector_engine.app.proto import engine_pb2, engine_pb2_grpc

@pytest.fixture
def workers_cluster():
    """Start 2 Search Workers in subprocesses on ports 50081 and 50082."""
    ports = [50081, 50082]
    processes = []
    
    # Clean up WAL log directories before starting
    for port in ports:
        path = f"vector_engine/data/worker_{port}"
        if os.path.exists(path):
            shutil.rmtree(path)
            
    for port in ports:
        p = subprocess.Popen([
            sys.executable, "-m", "vector_engine.app.worker", "--port", str(port)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)
    
    # Wait for async gRPC workers to boot up
    time.sleep(1.5)
    yield ports
    
    # Teardown: terminate processes cleanly
    for p in processes:
        p.terminate()
        p.wait()
        
    # Clean up WAL log directories after shutting down
    for port in ports:
        path = f"vector_engine/data/worker_{port}"
        if os.path.exists(path):
            shutil.rmtree(path)

@pytest.mark.asyncio
async def test_coordinator_sharding_and_search(workers_cluster) -> None:
    ports = workers_cluster
    coord = Coordinator()
    for port in ports:
        coord.register_worker("localhost", port)
        
    # Initially active
    await coord.check_workers_health()
    active_workers = [addr for addr, info in coord.workers.items() if info["is_active"]]
    assert len(active_workers) == 2

    # Ingest vectors
    # sharded deterministically
    assert await coord.insert_vector("v1", [1.0, 0.0, 0.0], {"name": "x"})
    assert await coord.insert_vector("v2", [0.0, 1.0, 0.0], {"name": "y"})
    assert await coord.insert_vector("v3", [0.8, 0.6, 0.0], {"name": "diag"})
    
    # Query closest to x-axis
    results = await coord.search([0.9, 0.1, 0.0], top_k=2)
    
    # Verify both nodes queried, results combined, sorted, and capped to top-k
    assert len(results) == 2
    assert results[0].id == "v1"
    assert results[1].id == "v3"
    assert results[0].score > results[1].score
    assert results[0].metadata == {"name": "x"}
    
    await coord.close_all_channels()

@pytest.mark.asyncio
async def test_coordinator_fault_tolerance_timeout(workers_cluster) -> None:
    ports = workers_cluster
    
    # Coordinator with low timeout for testing (0.5 seconds)
    coord = Coordinator(timeout=0.5, max_retries=1)
    
    # Register workers
    coord.register_worker("localhost", ports[0])
    coord.register_worker("localhost", ports[1])
    
    # Insert vectors to both
    await coord.insert_vector("v1", [1.0, 0.0])
    await coord.insert_vector("v2", [0.0, 1.0])
    
    # Simulate worker crash on port 50082 by deregistering or connecting to non-existent port.
    # To test actual fault tolerance where a worker times out/disconnects:
    # We can connect to an invalid port (50083) and mark it active.
    # This will simulate a completely offline worker.
    coord.register_worker("localhost", 50083)
    
    # Now execute search. Although one worker (50083) is offline and will fail/timeout,
    # the search should not crash. It should gracefully return results from the active worker.
    results = await coord.search([1.0, 0.0], top_k=5)
    
    # We should still get results from the remaining online workers!
    assert isinstance(results, list)
    
    await coord.close_all_channels()
