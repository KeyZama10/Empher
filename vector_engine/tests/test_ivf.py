import pytest
import numpy as np

from vector_engine.app.ivf_index import IVFIndex

def test_ivf_init_validation() -> None:
    # Cannot initialize with 0 or negative clusters
    with pytest.raises(ValueError, match="must be a positive integer"):
        IVFIndex(n_clusters=0)
        
    with pytest.raises(ValueError, match="must be a positive integer"):
        IVFIndex(n_clusters=-5)

def test_ivf_untrained_errors() -> None:
    index = IVFIndex(n_clusters=2)
    assert not index.is_trained
    
    # Adding vectors before training raises error
    with pytest.raises(ValueError, match="must be trained before adding vectors"):
        index.add_vectors(["v1"], np.array([[1.0, 2.0]]))
        
    # Searching before training raises error
    with pytest.raises(ValueError, match="must be trained before performing searches"):
        index.search(np.array([1.0, 2.0]))

def test_ivf_training_insufficient_samples() -> None:
    index = IVFIndex(n_clusters=4)
    # 3 samples is less than n_clusters=4, should raise ValueError
    samples = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])
    with pytest.raises(ValueError, match="must be at least the number of clusters"):
        index.train(samples)

def test_ivf_training_and_search_success() -> None:
    # 4 clusters, dimension 2
    index = IVFIndex(n_clusters=4, dimension=2)
    
    # Train vectors
    train_vectors = np.array([
        [1.0, 0.0],  # Centroid 1 area
        [-1.0, 0.0], # Centroid 2 area
        [0.0, 1.0],  # Centroid 3 area
        [0.0, -1.0]  # Centroid 4 area
    ])
    
    index.train(train_vectors)
    assert index.is_trained
    assert index.dimension == 2
    
    # Add vectors
    ids = ["v1", "v2", "v3", "v4"]
    index.add_vectors(ids, train_vectors)
    
    assert index.total_vectors == 4
    
    # Query closest to v1 [1.0, 0.0]
    results = index.search(np.array([0.9, 0.1]), top_k=2)
    
    # Check that search returned the item in the closest bucket
    assert len(results) >= 1
    assert results[0].id == "v1"
    assert results[0].score == pytest.approx(1.0, abs=1e-2)

def test_ivf_empty_bucket() -> None:
    # Initialize index
    index = IVFIndex(n_clusters=2, dimension=2)
    train_vectors = np.array([
        [1.0, 0.0],
        [-1.0, 0.0]
    ])
    index.train(train_vectors)
    
    # Add vector only to one cluster/area.
    # [1.0, 0.0] will map to cluster 0 (or whichever KMeans assigns)
    # Let's predict using the fit kmeans to see assignments
    assignments = index.kmeans.predict(train_vectors)
    
    # Add only one vector
    index.add_vectors([f"vec_{assignments[0]}"], train_vectors[0:1])
    
    # Query closest to train_vectors[1] (which should route to the empty cluster/bucket)
    empty_query = train_vectors[1]
    
    results = index.search(empty_query)
    # The routed bucket should be empty, so results should be empty
    assert results == []

def test_ivf_dimension_validation() -> None:
    index = IVFIndex(n_clusters=2, dimension=2)
    train_vectors = np.array([
        [1.0, 0.0],
        [-1.0, 0.0]
    ])
    index.train(train_vectors)
    
    # Dimension mismatch on add
    with pytest.raises(ValueError, match="Dimension mismatch"):
        index.add_vectors(["v1"], np.array([[1.0, 0.0, 0.0]]))
        
    # Dimension mismatch on search
    with pytest.raises(ValueError, match="Query dimension .* does not match"):
        index.search(np.array([1.0, 0.0, 0.0]))

def test_ivf_duplicate_id_prevention() -> None:
    index = IVFIndex(n_clusters=2, dimension=2)
    train_vectors = np.array([
        [1.0, 0.0],
        [-1.0, 0.0]
    ])
    index.train(train_vectors)
    
    index.add_vectors(["v1"], train_vectors[0:1])
    with pytest.raises(ValueError, match="already exists"):
        index.add_vectors(["v1"], train_vectors[1:2])
