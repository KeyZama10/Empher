import os
import pytest
import numpy as np

from vector_engine.app.vector_store import VectorStore

def test_vector_store_init() -> None:
    store = VectorStore(dimension=8)
    assert store.dimension == 8
    assert store.size == 0
    assert store.vectors.shape == (0, 8)

def test_vector_store_inferred_dimension() -> None:
    store = VectorStore()
    assert store.dimension is None
    
    # First added vector should set the dimension
    store.add_vector("v1", np.array([1.0, 2.0, 3.0]))
    assert store.dimension == 3
    assert store.size == 1
    
    # Adding incorrect dimension vector should fail
    with pytest.raises(ValueError, match="Dimension mismatch"):
        store.add_vector("v2", np.array([1.0, 2.0]))

def test_vector_store_duplicate_id() -> None:
    store = VectorStore(dimension=2)
    store.add_vector("v1", np.array([1.0, 2.0]))
    
    with pytest.raises(ValueError, match="Vector ID 'v1' already exists"):
        store.add_vector("v1", np.array([3.0, 4.0]))

def test_vector_store_batch_add() -> None:
    store = VectorStore(dimension=3)
    ids = ["v1", "v2"]
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    metadatas = [{"tag": "x"}, {"tag": "y"}]
    
    store.add_vectors(ids, vectors, metadatas)
    
    assert store.size == 2
    assert store.ids == ids
    assert store.metadata == metadatas
    np.testing.assert_allclose(store.vectors, vectors)

def test_vector_store_batch_add_validation() -> None:
    store = VectorStore(dimension=2)
    
    # Length mismatch IDs vs vectors
    with pytest.raises(ValueError, match="Mismatch"):
        store.add_vectors(["v1"], np.ones((2, 2)))
        
    # Length mismatch metadata vs vectors
    with pytest.raises(ValueError, match="Mismatch"):
        store.add_vectors(["v1", "v2"], np.ones((2, 2)), metadatas=[{"tag": "x"}])

def test_vector_store_save_load(tmp_path) -> None:
    filepath = str(tmp_path / "test_store.npz")
    
    store = VectorStore(dimension=4)
    store.add_vector("v1", np.array([1.0, 0.0, 0.0, 0.0]), {"info": "first"})
    store.add_vector("v2", np.array([0.0, 1.0, 0.0, 0.0]), {"info": "second"})
    
    store.save(filepath)
    assert os.path.exists(filepath)
    
    # Load into a new store
    loaded_store = VectorStore()
    loaded_store.load(filepath)
    
    assert loaded_store.size == 2
    assert loaded_store.dimension == 4
    assert loaded_store.ids == ["v1", "v2"]
    assert loaded_store.metadata == [{"info": "first"}, {"info": "second"}]
    np.testing.assert_allclose(
        loaded_store.vectors,
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    )
