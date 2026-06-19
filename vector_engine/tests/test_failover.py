import asyncio
import pytest
import os
from vector_engine.app.coordinator import Coordinator

@pytest.mark.asyncio
async def test_coordinator_failover_flow():
    """Verify that multiple coordinators perform failover election via the registry."""
    # Prevent test pollution by clearing shared mock registry file
    mock_file = "vector_engine/data/service_discovery_mock.json"
    if os.path.exists(mock_file):
        try:
            os.remove(mock_file)
        except Exception:
            pass

    # Initialize registry and clear locks
    from vector_engine.app.distributed.service_discovery import EtcdServiceRegistry
    registry = EtcdServiceRegistry(use_mock_fallback=True)
    registry.release_leader_lock("coord_a")
    registry.release_leader_lock("coord_b")
    
    # 1. Initialize two coordinators in standby mode
    coord_a = Coordinator(coordinator_id="coord_a", standby=True)
    coord_b = Coordinator(coordinator_id="coord_b", standby=True)
    
    # Verify both start as non-leaders
    assert coord_a.is_leader is False
    assert coord_b.is_leader is False
    
    # Start both election loops
    coord_a.start()
    coord_b.start()
    
    # Wait for election to resolve (coord_a starts loop first and should lock)
    await asyncio.sleep(1.5)
    
    # One and only one must be elected leader
    assert coord_a.is_leader != coord_b.is_leader
    
    leader = "coord_a" if coord_a.is_leader else "coord_b"
    standby = coord_b if leader == "coord_a" else coord_a
    active = coord_a if leader == "coord_a" else coord_b
    
    # 2. Standby coordinator operations should raise PermissionError
    with pytest.raises(PermissionError, match="is in STANDBY mode"):
        await standby.search([1.0, 0.0], top_k=2)
        
    with pytest.raises(PermissionError, match="is in STANDBY mode"):
        await standby.insert_vector("v1", [1.0, 0.0])
        
    # 3. Simulate Active Coordinator Failover
    await active.close_all_channels()
    
    # Allow time for active lease to expire and standby to detect/acquire lock
    await asyncio.sleep(4.5)
    
    # Standby must have promoted itself to leader
    assert standby.is_leader is True
    
    # Clean up standby coordinator channels
    await standby.close_all_channels()
