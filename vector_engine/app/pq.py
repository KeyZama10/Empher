import os
import numpy as np
from typing import Optional

class ProductQuantizer:
    """Product Quantizer (PQ) for vector compression using subspace KMeans clustering."""

    def __init__(self, n_subspaces: int = 8, n_centroids: int = 256):
        """Initialize the ProductQuantizer.

        Args:
            n_subspaces: Number of subspaces (M) to partition vectors into.
            n_centroids: Number of cluster centroids (K) per subspace. Max 256.
        """
        if n_centroids <= 0 or n_centroids > 256:
            raise ValueError(f"n_centroids must be between 1 and 256. Got {n_centroids}.")
        if n_subspaces <= 0:
            raise ValueError(f"n_subspaces must be positive. Got {n_subspaces}.")

        self.n_subspaces: int = n_subspaces
        self.n_centroids: int = n_centroids
        self.dimension: Optional[int] = None
        self.subspace_dim: Optional[int] = None

        # Codebooks: shape (M, K, d) containing centroids for each subspace
        self.codebooks: Optional[np.ndarray] = None
        self._is_trained: bool = False

    @property
    def is_trained(self) -> bool:
        """Returns whether the quantizer has been trained."""
        return self._is_trained

    def train(self, vectors: np.ndarray) -> None:
        """Train codebooks for all subspaces using KMeans.

        Args:
            vectors: 2-D array of shape (N, D) containing training vectors.

        Raises:
            ValueError: If dimensions mismatch, not 2-D, or samples < n_centroids.
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
        else:
            vectors = vectors.astype(np.float32)

        if vectors.ndim != 2:
            raise ValueError(f"Training vectors must be 2-D. Got shape {vectors.shape}")

        n_samples, D = vectors.shape
        if D % self.n_subspaces != 0:
            raise ValueError(
                f"Vector dimension ({D}) must be divisible by "
                f"n_subspaces ({self.n_subspaces})."
            )

        if n_samples < self.n_centroids:
            raise ValueError(
                f"Number of training samples ({n_samples}) must be at least the "
                f"number of centroids ({self.n_centroids})."
            )

        self.dimension = D
        self.subspace_dim = D // self.n_subspaces

        # Initialize codebooks of shape (M, K, d)
        self.codebooks = np.zeros(
            (self.n_subspaces, self.n_centroids, self.subspace_dim), 
            dtype=np.float32
        )

        from sklearn.cluster import KMeans

        for m in range(self.n_subspaces):
            # Extract sub-vectors for subspace m
            sub_vectors = vectors[:, m * self.subspace_dim : (m + 1) * self.subspace_dim]
            
            # Fit KMeans to identify centroids
            kmeans = KMeans(
                n_clusters=self.n_centroids, 
                random_state=42 + m, 
                n_init="auto"
            )
            kmeans.fit(sub_vectors)
            self.codebooks[m] = kmeans.cluster_centers_

        self._is_trained = True

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Quantize vectors into compact byte codes.

        Args:
            vectors: 2-D array of shape (N, D).

        Returns:
            2-D array of shape (N, M) of type uint8 containing quantized codes.

        Raises:
            ValueError: If quantizer is not trained or dimensions mismatch.
        """
        if not self._is_trained or self.codebooks is None:
            raise ValueError("ProductQuantizer must be trained before encoding.")

        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
        else:
            vectors = vectors.astype(np.float32)

        if vectors.ndim != 2:
            raise ValueError(f"Vectors must be 2-D. Got shape {vectors.shape}")

        n_vectors, D = vectors.shape
        if D != self.dimension:
            raise ValueError(
                f"Dimension mismatch. Expected {self.dimension}, got {D}."
            )

        codes = np.zeros((n_vectors, self.n_subspaces), dtype=np.uint8)

        # Vectorized nearest-centroid search per subspace
        for m in range(self.n_subspaces):
            sub_vectors = vectors[:, m * self.subspace_dim : (m + 1) * self.subspace_dim]
            centroids = self.codebooks[m]

            # Compute squared Euclidean distances in pure NumPy:
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x . y^T
            sq_norms_x = np.sum(sub_vectors ** 2, axis=1, keepdims=True)  # (N, 1)
            sq_norms_y = np.sum(centroids ** 2, axis=1)                  # (K,)
            dot_products = np.dot(sub_vectors, centroids.T)              # (N, K)

            dists = sq_norms_x + sq_norms_y - 2 * dot_products

            # Assign to closest centroid
            codes[:, m] = np.argmin(dists, axis=1)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruct vectors from compact byte codes.

        Args:
            codes: 2-D array of shape (N, M) of type uint8.

        Returns:
            Reconstructed 2-D array of shape (N, D) containing float32 vectors.

        Raises:
            ValueError: If quantizer is not trained or code columns mismatch.
        """
        if not self._is_trained or self.codebooks is None:
            raise ValueError("ProductQuantizer must be trained before decoding.")

        if codes.ndim != 2:
            raise ValueError(f"Codes must be 2-D. Got shape {codes.shape}")

        n_vectors, M = codes.shape
        if M != self.n_subspaces:
            raise ValueError(
                f"Mismatch: Got codes with {M} columns, expected {self.n_subspaces}."
            )

        reconstructed = np.zeros((n_vectors, self.dimension), dtype=np.float32)

        # Map codes back to centroids and concatenate
        for m in range(self.n_subspaces):
            reconstructed[:, m * self.subspace_dim : (m + 1) * self.subspace_dim] = \
                self.codebooks[m][codes[:, m]]

        return reconstructed

    def save(self, filepath: str) -> None:
        """Save the ProductQuantizer parameters and codebooks to disk.

        Args:
            filepath: Target destination path.
        """
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        np.savez_compressed(
            filepath,
            n_subspaces=np.array([self.n_subspaces]),
            n_centroids=np.array([self.n_centroids]),
            dimension=np.array([self.dimension or 0]),
            subspace_dim=np.array([self.subspace_dim or 0]),
            codebooks=self.codebooks if self.codebooks is not None else np.empty(0),
            is_trained=np.array([self._is_trained])
        )

    def load(self, filepath: str) -> None:
        """Load the ProductQuantizer parameters and codebooks from disk.

        Args:
            filepath: Source file path.

        Raises:
            FileNotFoundError: If filepath does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with np.load(filepath) as data:
            self.n_subspaces = int(data["n_subspaces"][0])
            self.n_centroids = int(data["n_centroids"][0])
            dim = int(data["dimension"][0])
            self.dimension = dim if dim > 0 else None
            sub_dim = int(data["subspace_dim"][0])
            self.subspace_dim = sub_dim if sub_dim > 0 else None
            
            codebooks = data["codebooks"]
            self.codebooks = codebooks if codebooks.size > 0 else None
            self._is_trained = bool(data["is_trained"][0])
