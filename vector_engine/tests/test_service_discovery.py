import time
import pytest
import os
from vector_engine.app.distributed.service_discovery import EtcdServiceRegistry

def test_mock_registry_registration():
    """Test that EtcdServiceRegistry registers and retrieves workers dynamically."""
    # Prevent test pollution by clearing shared mock registry file
    mock_file = "vector_engine/data/service_discovery_mock.json"
    if os.path.exists(mock_file):
        try:
            os.remove(mock_file)
        except Exception:
            pass

    registry = EtcdServiceRegistry(use_mock_fallback=True)
    assert registry.use_mock is True  # Should fall back to mock since no etcd server is running
    
    # 1. Register nodes
    registry.register_worker("w1", "localhost", 50051, role="primary", ttl=3)
    registry.register_worker("w2", "localhost", 50052, role="replica", primary_addr="localhost:50051", ttl=3)
    
    # Retrieve active nodes
    active = registry.get_active_workers()
    assert len(active) == 2
    
    ids = {node["worker_id"] for node in active}
    assert ids == {"w1", "w2"}
    
    roles = {node["worker_id"]: node["role"] for node in active}
    assert roles["w1"] == "primary"
    assert roles["w2"] == "replica"
    
    # 2. Heartbeat keeps node alive
    time.sleep(1.5)
    registry.heartbeat("w1")
    
    # Sleep another 1.6 seconds (total 3.1s since w2 registered)
    time.sleep(1.6)
    
    # w2 should have expired (ttl=3), but w1 was renewed by heartbeat so it stays active
    active_post = registry.get_active_workers()
    assert len(active_post) == 1
    assert active_post[0]["worker_id"] == "w1"
    
    # 3. Clean deregister
    registry.deregister_worker("w1")
    assert len(registry.get_active_workers()) == 0
