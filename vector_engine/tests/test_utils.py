import pytest
import numpy as np

from vector_engine.app.utils import cosine_similarity, cosine_similarity_matrix

def test_cosine_similarity_identical() -> None:
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([1.0, 2.0, 3.0])
    # Similarity should be 1.0 (or very close due to floating point precision)
    assert pytest.approx(cosine_similarity(v1, v2)) == 1.0

def test_cosine_similarity_orthogonal() -> None:
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    assert pytest.approx(cosine_similarity(v1, v2)) == 0.0

def test_cosine_similarity_opposite() -> None:
    v1 = np.array([1.0, -1.0])
    v2 = np.array([-1.0, 1.0])
    assert pytest.approx(cosine_similarity(v1, v2)) == -1.0

def test_cosine_similarity_zero_norm() -> None:
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 2.0])
    # Norm of v1 is zero, should return 0.0
    assert cosine_similarity(v1, v2) == 0.0
    assert cosine_similarity(v2, v1) == 0.0

def test_cosine_similarity_validation() -> None:
    # Dimension mismatch
    with pytest.raises(ValueError, match="Vector shapes must match"):
        cosine_similarity(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))

    # Dimensions not 1-D
    with pytest.raises(ValueError, match="Vectors must be 1-D"):
        cosine_similarity(np.ones((2, 2)), np.ones((2, 2)))

def test_cosine_similarity_matrix() -> None:
    query = np.array([1.0, 0.0])
    vectors = np.array([
        [1.0, 0.0],  # Parallel (1.0)
        [0.0, 1.0],  # Orthogonal (0.0)
        [-1.0, 0.0], # Opposite (-1.0)
        [0.0, 0.0]   # Zero-norm (0.0)
    ])
    
    expected = np.array([1.0, 0.0, -1.0, 0.0])
    similarities = cosine_similarity_matrix(query, vectors)
    
    np.testing.assert_allclose(similarities, expected, atol=1e-7)

def test_cosine_similarity_matrix_zero_query() -> None:
    query = np.array([0.0, 0.0])
    vectors = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    similarities = cosine_similarity_matrix(query, vectors)
    assert np.all(similarities == 0.0)

def test_cosine_similarity_matrix_validation() -> None:
    # Query must be 1-D
    with pytest.raises(ValueError, match="Query vector must be 1-D"):
        cosine_similarity_matrix(np.ones((2, 2)), np.ones((2, 2)))
        
    # Vectors must be 2-D
    with pytest.raises(ValueError, match="Vectors matrix must be 2-D"):
        cosine_similarity_matrix(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    # Dimension mismatch
    with pytest.raises(ValueError, match="Query dimension .* must match vector dimension"):
        cosine_similarity_matrix(np.array([1.0, 2.0]), np.ones((5, 3)))
