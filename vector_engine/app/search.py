from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from vector_engine.app.vector_store import VectorStore
from vector_engine.app.utils import cosine_similarity_matrix

@dataclass(frozen=True)
class SearchResult:
    """Represents a single search result match."""
    id: str
    score: float
    metadata: Dict[str, Any]

def brute_force_search(
    vector_store: "VectorStore", 
    query_vector: np.ndarray, 
    top_k: int = 5
) -> List[SearchResult]:
    """Perform a brute-force nearest neighbor search using cosine similarity.

    Args:
        vector_store: The VectorStore database containing target vectors.
        query_vector: The 1-D query vector of shape (D,).
        top_k: The maximum number of nearest neighbors to return.

    Returns:
        A list of SearchResult objects ordered from highest to lowest similarity score.

    Raises:
        ValueError: If vector_store is empty, query_vector dimension mismatches, or top_k <= 0.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer. Got {top_k}.")

    if vector_store.size == 0:
        raise ValueError("Cannot perform search on an empty VectorStore.")

    # Convert query_vector to numpy float32
    if not isinstance(query_vector, np.ndarray):
        query_vector = np.array(query_vector, dtype=np.float32)
    else:
        query_vector = query_vector.astype(np.float32)

    if query_vector.ndim != 1:
        raise ValueError(f"Query vector must be 1-D. Got shape {query_vector.shape}.")

    if vector_store.dimension is not None and query_vector.shape[0] != vector_store.dimension:
        raise ValueError(
            f"Query dimension ({query_vector.shape[0]}) does not match "
            f"VectorStore dimension ({vector_store.dimension})."
        )

    # Compute similarities for all stored vectors
    similarities = cosine_similarity_matrix(query_vector, vector_store.vectors)

    # Determine how many items to return
    k = min(top_k, vector_store.size)

    # Perform partition or sort to get the indices of top k largest similarities.
    # For large datasets, np.argpartition is O(N) which is faster than O(N log N) argsort.
    if k < vector_store.size:
        # Get indices of the k largest elements (not necessarily sorted)
        unsorted_top_k_indices = np.argpartition(similarities, -k)[-k:]
        # Now sort only those top k elements descending
        sorted_top_k_indices = unsorted_top_k_indices[
            np.argsort(similarities[unsorted_top_k_indices])[::-1]
        ]
    else:
        # If we need all elements, just sort all of them descending
        sorted_top_k_indices = np.argsort(similarities)[::-1]

    # Retrieve matching records
    ids = vector_store.ids
    metadata = vector_store.metadata

    results = []
    for idx in sorted_top_k_indices:
        results.append(
            SearchResult(
                id=ids[idx],
                score=float(similarities[idx]),
                metadata=metadata[idx]
            )
        )

    return results
