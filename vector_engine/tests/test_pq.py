import os
import pytest
import numpy as np

from vector_engine.app.pq import ProductQuantizer

def test_pq_init_validation() -> None:
    # Centroids must be between 1 and 256
    with pytest.raises(ValueError, match="n_centroids must be between"):
        ProductQuantizer(n_centroids=0)
    with pytest.raises(ValueError, match="n_centroids must be between"):
        ProductQuantizer(n_centroids=257)
    with pytest.raises(ValueError, match="n_subspaces must be positive"):
        ProductQuantizer(n_subspaces=0)

def test_pq_untrained_errors() -> None:
    pq = ProductQuantizer(n_subspaces=2, n_centroids=4)
    assert not pq.is_trained
    
    with pytest.raises(ValueError, match="must be trained before encoding"):
        pq.encode(np.ones((5, 4)))
        
    with pytest.raises(ValueError, match="must be trained before decoding"):
        pq.decode(np.zeros((5, 2), dtype=np.uint8))

def test_pq_training_validation() -> None:
    pq = ProductQuantizer(n_subspaces=3, n_centroids=4)
    
    # 4 dimensions is not divisible by 3 subspaces
    with pytest.raises(ValueError, match="must be divisible by"):
        pq.train(np.ones((10, 4)))
        
    # Insufficient training samples (< n_centroids=4)
    with pytest.raises(ValueError, match="must be at least the number of centroids"):
        pq.train(np.ones((3, 6)))

def test_pq_encode_decode_success() -> None:
    # 2 subspaces, 4 centroids (dimension D = 4, subspace d = 2)
    pq = ProductQuantizer(n_subspaces=2, n_centroids=4)
    
    # Generate mock vectors
    train_vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0]
    ], dtype=np.float32)
    
    pq.train(train_vectors)
    assert pq.is_trained
    assert pq.dimension == 4
    assert pq.subspace_dim == 2
    assert pq.codebooks.shape == (2, 4, 2)
    
    # Encode vectors
    codes = pq.encode(train_vectors)
    assert codes.shape == (6, 2)
    assert codes.dtype == np.uint8
    
    # Decode back
    reconstructed = pq.decode(codes)
    assert reconstructed.shape == (6, 4)
    assert reconstructed.dtype == np.float32

def test_pq_save_load(tmp_path) -> None:
    filepath = str(tmp_path / "pq_model.npz")
    
    pq = ProductQuantizer(n_subspaces=2, n_centroids=2)
    train_vectors = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    pq.train(train_vectors)
    pq.save(filepath)
    assert os.path.exists(filepath)
    
    # Load into new instance
    loaded_pq = ProductQuantizer()
    loaded_pq.load(filepath)
    
    assert loaded_pq.is_trained
    assert loaded_pq.n_subspaces == 2
    assert loaded_pq.n_centroids == 2
    assert loaded_pq.dimension == 4
    assert loaded_pq.subspace_dim == 2
    np.testing.assert_allclose(loaded_pq.codebooks, pq.codebooks)
