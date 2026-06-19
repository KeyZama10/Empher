import os
import sys
import time
import subprocess
import pytest
import shutil
import numpy as np
import asyncio

from vector_engine.app.coordinator import Coordinator
from vector_engine.app.proto import engine_pb2, engine_pb2_grpc

@pytest.fixture
def replica_cluster():
    """Start 1 Primary Worker (port 50091) and 1 Replica Worker (port 50092) in subprocesses."""
    # Ensure WAL clean start
    ports = [50091, 50092]
    for port in ports:
        path = f"vector_engine/data/worker_{port}"
        if os.path.exists(path):
            shutil.rmtree(path)
            
    # Clean up mock service discovery file before starting workers
    mock_file = "vector_engine/data/service_discovery_mock.json"
    if os.path.exists(mock_file):
        try:
            os.remove(mock_file)
        except Exception:
            pass

    # Start Replica node first
    log_replica = open("replica_debug.log", "w")
    p_replica = subprocess.Popen([
        sys.executable, "-m", "vector_engine.app.worker",
        "--port", "50092",
        "--role", "replica",
        "--primary-addr", "localhost:50091"
    ], stdout=log_replica, stderr=log_replica)
    
    # Start Primary node configured with replica localhost:50092
    log_primary = open("primary_debug.log", "w")
    p_primary = subprocess.Popen([
        sys.executable, "-m", "vector_engine.app.worker",
        "--port", "50091",
        "--role", "primary",
        "--replicas", "localhost:50092"
    ], stdout=log_primary, stderr=log_primary)
    
    # Wait for servers to start
    time.sleep(3.5)
    
    yield
    
    # Teardown
    p_primary.terminate()
    p_replica.terminate()
    p_primary.wait()
    p_replica.wait()
    
    log_primary.close()
    log_replica.close()
    
    # Clean up WAL logs
    for port in ports:
        path = f"vector_engine/data/worker_{port}"
        if os.path.exists(path):
            shutil.rmtree(path)
            
    # Clean up mock file after shutdown
    if os.path.exists(mock_file):
        try:
            os.remove(mock_file)
        except Exception:
            pass

@pytest.mark.asyncio
async def test_replication_and_replica_reads(replica_cluster):
    """Verify primary replicates mutations and coordinator load-balances reads to healthy replicas."""
    # Initialize registry
    from vector_engine.app.distributed.service_discovery import EtcdServiceRegistry
    registry = EtcdServiceRegistry(use_mock_fallback=True)
    
    # Give heartbeats/sync a moment to establish
    coord = Coordinator()
    coord.start()
    
    await asyncio.sleep(3.5) # Wait for registry to sync nodes
    
    # 1. Insert vector via coordinator (routes to localhost:50091 primary)
    success = await coord.insert_vector("rv1", [1.0, 0.0, 0.0], {"lbl": "replica_test"})
    assert success is True
    
    # Allow small window for async replication to replica node
    await asyncio.sleep(1.0)
    
    # 2. Verify replication: query replica store directly via gRPC
    import grpc.aio
    channel_rep = grpc.aio.insecure_channel("localhost:50092")
    stub_rep = engine_pb2_grpc.SearchWorkerStub(channel_rep)
    health_rep = await stub_rep.HealthCheck(engine_pb2.HealthRequest())
    assert health_rep.size == 1  # Should have successfully replicated the vector
    await channel_rep.close()
    
    # 3. Search via coordinator: routes to replica (localhost:50092)
    results = await coord.search([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 1
    assert results[0].id == "rv1"
    
    # 4. Fallback: simulate replica failure by marking it inactive in health check
    if "localhost:50092" in coord.workers:
        coord.workers["localhost:50092"]["is_active"] = False
    else:
        raise KeyError(f"localhost:50092 not in workers: {list(coord.workers.keys())}")
    
    # Perform search again. Coordinator should fall back to primary node localhost:50091
    results_post = await coord.search([1.0, 0.0, 0.0], top_k=2)
    assert len(results_post) == 1
    assert results_post[0].id == "rv1"
    
    await coord.close_all_channels()
