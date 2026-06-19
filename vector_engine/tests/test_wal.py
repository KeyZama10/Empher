import os
import shutil
import pytest
import numpy as np
from vector_engine.app.vector_store import VectorStore
from vector_engine.app.storage.wal import WALManager

@pytest.fixture
def temp_wal_dir():
    """Create a temporary directory for WAL tests and clean up on teardown."""
    path = "vector_engine/data/test_wal"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)

def test_wal_basic_recovery(temp_wal_dir):
    """Verify that WAL manager logs insert mutations and recovers them cleanly."""
    wal = WALManager(temp_wal_dir)
    store = VectorStore(dimension=3)
    
    # Insert vectors through WAL and store
    data = [
        ("v1", [1.0, 0.0, 0.0], {"name": "x"}),
        ("v2", [0.0, 1.0, 0.0], {"name": "y"}),
        ("v3", [0.0, 0.0, 1.0], {"name": "z"})
    ]
    
    for vid, vec, meta in data:
        wal.append_insert(vid, vec, meta)
        store.add_vector(vid, np.array(vec, dtype=np.float32), meta)
        
    wal.close()
    
    # Verify state matches
    assert store.size == 3
    assert "v1" in store._ids_set
    
    # Create a fresh store and replay WAL
    new_store = VectorStore(dimension=3)
    new_wal = WALManager(temp_wal_dir)
    new_wal.replay(new_store)
    
    # Verify exact state recovery
    assert new_store.size == 3
    assert np.allclose(new_store.vectors[0], [1.0, 0.0, 0.0])
    assert new_store.metadata[0] == {"name": "x"}
    assert new_store.metadata[1] == {"name": "y"}
    new_wal.close()

def test_wal_rotation_and_snapshot(temp_wal_dir):
    """Test WAL rotation (snapshot writing) and subsequent log replay."""
    wal = WALManager(temp_wal_dir)
    store = VectorStore(dimension=2)
    
    # Add initial batch
    wal.append_insert("v1", [1.0, 0.0], {"lbl": "v1"})
    store.add_vector("v1", np.array([1.0, 0.0]), {"lbl": "v1"})
    
    # Perform rotation (creates snapshot, truncates WAL)
    wal.rotate(store)
    
    # Verify active WAL is truncated and snapshot is written
    assert os.path.exists(wal.snapshot_path)
    assert os.path.exists(wal.wal_path)
    assert os.path.getsize(wal.wal_path) == 0
    
    # Add post-rotation batch (writes to the fresh active.wal)
    wal.append_insert("v2", [0.0, 1.0], {"lbl": "v2"})
    store.add_vector("v2", np.array([0.0, 1.0]), {"lbl": "v2"})
    wal.close()
    
    # Recover state on a fresh store from snapshot + active.wal combined
    new_store = VectorStore(dimension=2)
    new_wal = WALManager(temp_wal_dir)
    new_wal.replay(new_store)
    
    assert new_store.size == 2
    assert "v1" in new_store._ids_set
    assert "v2" in new_store._ids_set
    assert new_store.metadata[0] == {"lbl": "v1"}
    assert new_store.metadata[1] == {"lbl": "v2"}
    new_wal.close()

def test_wal_corrupt_entries_handling(temp_wal_dir):
    """Verify that replaying a WAL with corrupted lines skips them and recovers healthy ones."""
    wal_path = os.path.join(temp_wal_dir, "active.wal")
    os.makedirs(temp_wal_dir, exist_ok=True)
    
    # Manually write valid, partial/corrupted, and malformed JSON entries
    entries = [
        '{"op": "insert", "id": "v1", "vector": [1.0, 0.0], "metadata": {"name": "ok1"}}',
        '{"op": "insert", "id": "v2", "vector": [0.0, 1.0]', # Truncated mid-line (crash simulation)
        '{"op": "insert", "invalid_json_here":::: }',         # Bad syntax JSON
        '{"op": "insert", "id": "v3", "vector": [0.5, 0.5], "metadata": {"name": "ok2"}}'
    ]
    
    with open(wal_path, "w") as f:
        for entry in entries:
            f.write(entry + "\n")
            
    # Replay on fresh store
    store = VectorStore(dimension=2)
    wal = WALManager(temp_wal_dir)
    wal.replay(store)
    
    # Verify that the two valid records (v1 and v3) were recovered, and corrupt ones skipped
    assert store.size == 2
    assert "v1" in store._ids_set
    assert "v3" in store._ids_set
    assert "v2" not in store._ids_set
    
    # Verify we can continue to write to the re-opened active WAL
    wal.append_insert("v4", [0.7, 0.7], {"name": "ok3"})
    store.add_vector("v4", np.array([0.7, 0.7]), {"name": "ok3"})
    wal.close()
    
    # Re-recovery check
    final_store = VectorStore(dimension=2)
    final_wal = WALManager(temp_wal_dir)
    final_wal.replay(final_store)
    
    assert final_store.size == 3
    assert "v4" in final_store._ids_set
    final_wal.close()

def test_wal_delete_recovery(temp_wal_dir):
    """Verify that deleting a vector is correctly logged to WAL and replayed."""
    wal = WALManager(temp_wal_dir)
    store = VectorStore(dimension=2)
    
    # 1. Insert vectors
    wal.append_insert("v1", [1.0, 0.0], {"lbl": "v1"})
    store.add_vector("v1", np.array([1.0, 0.0]), {"lbl": "v1"})
    
    wal.append_insert("v2", [0.0, 1.0], {"lbl": "v2"})
    store.add_vector("v2", np.array([0.0, 1.0]), {"lbl": "v2"})
    
    # 2. Delete a vector
    wal.append_delete("v1")
    store.remove_vector("v1")
    wal.close()
    
    # Verify memory store state
    assert store.size == 1
    assert "v2" in store._ids_set
    assert "v1" not in store._ids_set
    
    # 3. Recover on a new store
    new_store = VectorStore(dimension=2)
    new_wal = WALManager(temp_wal_dir)
    new_wal.replay(new_store)
    
    # Verify recovered store state has v2 but not v1
    assert new_store.size == 1
    assert "v2" in new_store._ids_set
    assert "v1" not in new_store._ids_set
    new_wal.close()
