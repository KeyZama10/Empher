import pytest
import numpy as np

from vector_engine.app.vector_store import VectorStore
from vector_engine.app.search import brute_force_search

def test_brute_force_search_correctness() -> None:
    store = VectorStore(dimension=3)
    # Add vectors
    store.add_vector("doc_x", np.array([1.0, 0.0, 0.0]), {"title": "X Direction"})
    store.add_vector("doc_y", np.array([0.0, 1.0, 0.0]), {"title": "Y Direction"})
    store.add_vector("doc_diag", np.array([1.0, 1.0, 0.0]), {"title": "Diagonal"})
    
    # Query very close to doc_x
    query = np.array([1.0, 0.1, 0.0])
    
    # Search top 2
    results = brute_force_search(store, query, top_k=2)
    
    assert len(results) == 2
    # doc_x should be rank 1, doc_diag rank 2, scores should be sorted descending
    assert results[0].id == "doc_x"
    assert results[1].id == "doc_diag"
    assert results[0].score > results[1].score

def test_brute_force_search_top_k_larger_than_store() -> None:
    store = VectorStore(dimension=2)
    store.add_vector("v1", np.array([1.0, 0.0]))
    
    # Querying top_k = 5 on a store of size 1 should return only 1 result
    results = brute_force_search(store, np.array([1.0, 0.0]), top_k=5)
    assert len(results) == 1
    assert results[0].id == "v1"

def test_brute_force_search_empty_store_error() -> None:
    store = VectorStore(dimension=2)
    with pytest.raises(ValueError, match="Cannot perform search on an empty VectorStore"):
        brute_force_search(store, np.array([1.0, 0.0]), top_k=2)

def test_brute_force_search_invalid_top_k() -> None:
    store = VectorStore(dimension=2)
    store.add_vector("v1", np.array([1.0, 0.0]))
    
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        brute_force_search(store, np.array([1.0, 0.0]), top_k=0)
        
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        brute_force_search(store, np.array([1.0, 0.0]), top_k=-2)

def test_brute_force_search_query_dim_mismatch() -> None:
    store = VectorStore(dimension=3)
    store.add_vector("v1", np.array([1.0, 0.0, 0.0]))
    
    # VectorStore has dimension 3, query has dimension 2
    with pytest.raises(ValueError, match="Query dimension .* does not match"):
        brute_force_search(store, np.array([1.0, 0.0]), top_k=1)
