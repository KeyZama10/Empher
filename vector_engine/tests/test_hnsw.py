import os
import pytest
import numpy as np

from vector_engine.app.hnsw_index import HNSWIndex

def test_hnsw_init_validation() -> None:
    with pytest.raises(ValueError, match="dimension must be positive"):
        HNSWIndex(dimension=0)
        
    with pytest.raises(ValueError, match="max_elements must be positive"):
        HNSWIndex(dimension=4, max_elements=0)

def test_hnsw_add_and_search_success() -> None:
    # Dimension 3, capacity 10
    index = HNSWIndex(dimension=3, max_elements=10)
    assert index.size == 0
    
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    ids = ["v1", "v2", "v3"]
    metadatas = [{"axis": "x"}, {"axis": "y"}, {"axis": "z"}]
    
    index.add_vectors(ids, vectors, metadatas)
    assert index.size == 3
    
    # Query closest to x-axis
    results = index.search(np.array([0.9, 0.1, 0.0]), top_k=2)
    assert len(results) == 2
    assert results[0].id == "v1"
    assert results[0].metadata == {"axis": "x"}
    assert results[0].score > results[1].score

def test_hnsw_dimension_validation() -> None:
    index = HNSWIndex(dimension=3)
    
    # Add dimension mismatch
    with pytest.raises(ValueError, match="Dimension mismatch"):
        index.add_vectors(["v1"], np.array([[1.0, 0.0]]))
        
    index.add_vectors(["v1"], np.array([[1.0, 0.0, 0.0]]))
    
    # Search dimension mismatch
    with pytest.raises(ValueError, match="Query dimension .* does not match"):
        index.search(np.array([1.0, 0.0]))

def test_hnsw_capacity_exceeded() -> None:
    index = HNSWIndex(dimension=2, max_elements=2)
    
    # Add 2 elements
    index.add_vectors(["v1", "v2"], np.array([[1.0, 0.0], [0.0, 1.0]]))
    
    # Adding 1 more element should raise ValueError due to capacity limit
    with pytest.raises(ValueError, match="exceeds max capacity"):
        index.add_vectors(["v3"], np.array([[1.0, 1.0]]))

def test_hnsw_duplicate_id_prevention() -> None:
    index = HNSWIndex(dimension=2, max_elements=5)
    index.add_vectors(["v1"], np.array([[1.0, 0.0]]))
    
    with pytest.raises(ValueError, match="already exists"):
        index.add_vectors(["v1"], np.array([[0.0, 1.0]]))

def test_hnsw_save_load(tmp_path) -> None:
    filepath = str(tmp_path / "hnsw_test.bin")
    
    index = HNSWIndex(dimension=4, max_elements=10)
    vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ], dtype=np.float32)
    index.add_vectors(["v1", "v2"], vectors, [{"tag": "first"}, {"tag": "second"}])
    
    index.save(filepath)
    assert os.path.exists(filepath)
    assert os.path.exists(filepath + ".meta.json")
    
    # Load into new index instance
    loaded_index = HNSWIndex(dimension=4, max_elements=10) # params will be overwritten by load
    loaded_index.load(filepath)
    
    assert loaded_index.size == 2
    assert loaded_index.dimension == 4
    
    # Verify search after load
    results = loaded_index.search(np.array([1.0, 0.1, 0.0, 0.0]), top_k=1)
    assert len(results) == 1
    assert results[0].id == "v1"
    assert results[0].metadata == {"tag": "first"}
