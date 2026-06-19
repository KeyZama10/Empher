import pytest
import numpy as np
from vector_engine.app.ivf_index import IVFIndex

def test_ivf_nprobe_routing():
    """Verify that nprobe allows searching multiple centroids and returns boundary candidates."""
    # 4 clusters, dimension 2
    index = IVFIndex(n_clusters=4, dimension=2)
    
    # Training data mapping to 4 distinct regions/centroids
    train_vectors = np.array([
        [10.0, 0.0],   # Centroid 0
        [-10.0, 0.0],  # Centroid 1
        [0.0, 10.0],   # Centroid 2
        [0.0, -10.0]   # Centroid 3
    ])
    
    index.train(train_vectors)
    
    # Add vectors to specific centroids
    # Add v0 near centroid 0, v1 near centroid 1, etc.
    index.add_vectors(
        ["v0", "v1", "v2", "v3"],
        np.array([
            [10.1, 0.0],   # Bucket 0
            [-10.1, 0.0],  # Bucket 1
            [0.0, 10.1],   # Bucket 2
            [0.0, -10.1]   # Bucket 3
        ])
    )
    
    # Query point is exactly in the middle of Bucket 0 and Bucket 2, but slightly closer to Bucket 0
    query = np.array([5.1, 5.0]) # closest to [10.1, 0.0], second closest to [0.0, 10.1]
    
    # nprobe = 1: should ONLY search the single closest centroid (0) -> returns only v0
    results_nprobe_1 = index.search(query, top_k=4, nprobe=1)
    ids_1 = {r.id for r in results_nprobe_1}
    assert "v0" in ids_1
    assert "v2" not in ids_1
    
    # nprobe = 2: should search the 2 closest centroids (0 and 2) -> returns both v0 and v2
    results_nprobe_2 = index.search(query, top_k=4, nprobe=2)
    ids_2 = {r.id for r in results_nprobe_2}
    assert "v0" in ids_2
    assert "v2" in ids_2
    
    # nprobe = 4: should search all centroids -> returns all index elements
    results_nprobe_4 = index.search(query, top_k=4, nprobe=4)
    assert len(results_nprobe_4) == 4
    
    # Validation checks
    with pytest.raises(ValueError, match="nprobe must be a positive integer"):
        index.search(query, nprobe=0)
    with pytest.raises(ValueError, match="nprobe must be a positive integer"):
        index.search(query, nprobe=-1)
