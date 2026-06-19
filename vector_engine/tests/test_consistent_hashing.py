import pytest
import numpy as np
from collections import Counter
from vector_engine.app.distributed.hash_ring import HashRing

def test_hash_ring_basic():
    """Test basic node insertion, removal, and routing logic on the HashRing."""
    ring = HashRing(replicas=10)
    
    # Empty ring returns None
    assert ring.get_node("key_1") is None
    
    # Add a single node
    ring.add_node("node_1")
    assert ring.get_node("key_1") == "node_1"
    assert ring.get_node("key_2") == "node_1"
    
    # Add second node
    ring.add_node("node_2")
    # Verify routing resolves to either node_1 or node_2
    nodes = {ring.get_node(f"key_{i}") for i in range(100)}
    assert nodes.issubset({"node_1", "node_2"})
    
    # Remove node_1
    ring.remove_node("node_1")
    for i in range(10):
        assert ring.get_node(f"test_key_{i}") == "node_2"
        
    # Remove node_2
    ring.remove_node("node_2")
    assert ring.get_node("key_1") is None

def test_hash_ring_failure_simulation():
    """Simulate a worker node crash and verify key movement is minimized (< 30%)."""
    ring = HashRing(replicas=100)
    nodes = ["node_1", "node_2", "node_3", "node_4"]
    for node in nodes:
        ring.add_node(node)
        
    # Generate 1,000 test keys
    keys = [f"vec_{i}" for i in range(1000)]
    
    # Initial assignments
    initial_mapping = {key: ring.get_node(key) for key in keys}
    
    # Simulate worker crash on node_4
    ring.remove_node("node_4")
    
    # New assignments
    post_failure_mapping = {key: ring.get_node(key) for key in keys}
    
    # Count how many keys moved
    moved_count = 0
    node_4_keys_count = 0
    for key in keys:
        init_node = initial_mapping[key]
        post_node = post_failure_mapping[key]
        
        if init_node == "node_4":
            node_4_keys_count += 1
            # These keys MUST move since node_4 is gone
            assert post_node != "node_4"
        else:
            # For keys mapped to other active nodes, they should NOT move
            if init_node != post_node:
                moved_count += 1
                
    # In consistent hashing, the key movement of non-failed node keys should be exactly 0
    assert moved_count == 0, "Consistent Hashing moved keys that belonged to healthy nodes!"
    
    # Total key movement percentage should equal the fraction of keys originally on node_4
    total_movement_percent = (node_4_keys_count / len(keys)) * 100.0
    print(f"Consistent Hashing Key Movement: {total_movement_percent:.2f}%")
    assert total_movement_percent < 30.0, f"Key movement was too high: {total_movement_percent:.2f}%"

def test_hash_ring_load_distribution():
    """Verify that virtual nodes distribute keys evenly across physical nodes."""
    ring = HashRing(replicas=200)
    nodes = ["node_1", "node_2", "node_3", "node_4"]
    for node in nodes:
        ring.add_node(node)
        
    # Assign 10,000 keys
    keys = [f"key_{i}" for i in range(10000)]
    assignments = [ring.get_node(key) for key in keys]
    counts = Counter(assignments)
    
    # Expected key count per node: 2,500
    expected = len(keys) / len(nodes)
    
    print("\nLoad Distribution:")
    for node in nodes:
        cnt = counts[node]
        pct = (cnt / len(keys)) * 100.0
        print(f"  {node}: {cnt} keys ({pct:.2f}%)")
        
        # Verify that each node handles roughly its share (within 30% margin)
        assert abs(cnt - expected) / expected < 0.30, f"Imbalanced load on {node}: {cnt} keys"
