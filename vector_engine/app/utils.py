import numpy as np

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate the cosine similarity between two 1-D vectors.

    Args:
        v1: First vector of shape (D,).
        v2: Second vector of shape (D,).

    Returns:
        The cosine similarity score in range [-1.0, 1.0].
        Returns 0.0 if either vector has a norm of 0.
    
    Raises:
        ValueError: If inputs are not 1-D or shapes don't match.
    """
    if v1.ndim != 1 or v2.ndim != 1:
        raise ValueError(f"Vectors must be 1-D. Got dimensions {v1.ndim} and {v2.ndim}.")
    if v1.shape != v2.shape:
        raise ValueError(f"Vector shapes must match. Got {v1.shape} and {v2.shape}.")

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0.0 or norm_v2 == 0.0:
        return 0.0
        
    dot_prod = np.dot(v1, v2)
    similarity = dot_prod / (norm_v1 * norm_v2)
    return float(np.clip(similarity, -1.0, 1.0))

def cosine_similarity_matrix(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Calculate the cosine similarity between a query vector and a matrix of vectors.

    Args:
        query: 1-D query vector of shape (D,).
        vectors: 2-D array of vectors of shape (N, D).

    Returns:
        A 1-D array of cosine similarity scores of shape (N,).
        Returns 0.0 for elements where a vector has norm 0, or if query norm is 0.

    Raises:
        ValueError: If query is not 1-D, vectors is not 2-D, or dimensions do not match.
    """
    if query.ndim != 1:
        raise ValueError(f"Query vector must be 1-D. Got dimension {query.ndim}.")
    if vectors.ndim != 2:
        raise ValueError(f"Vectors matrix must be 2-D. Got dimension {vectors.ndim}.")
    if query.shape[0] != vectors.shape[1]:
        raise ValueError(
            f"Query dimension ({query.shape[0]}) must match vector dimension ({vectors.shape[1]})."
        )

    query_norm = np.linalg.norm(query)
    if query_norm == 0.0:
        return np.zeros(vectors.shape[0], dtype=np.float64)

    vector_norms = np.linalg.norm(vectors, axis=1)
    dot_products = np.dot(vectors, query)

    # Avoid division by zero by replacing zero norms with 1.0.
    # We will then zero out the similarity for these vectors.
    zero_mask = (vector_norms == 0.0)
    norms_product = vector_norms * query_norm
    safe_norms_product = np.where(zero_mask, 1.0, norms_product)

    similarities = dot_products / safe_norms_product
    similarities[zero_mask] = 0.0

    return np.clip(similarities, -1.0, 1.0)
