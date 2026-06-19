import json
import os
from typing import List, Dict, Any, Optional
import numpy as np
from vector_engine.app.base_index import BaseVectorIndex
from vector_engine.app.search import SearchResult

class VectorStore(BaseVectorIndex):
    """A production-grade in-memory Vector Database that persists to disk using NumPy."""

    def __init__(self, dimension: Optional[int] = None):
        """Initialize the VectorStore.

        Args:
            dimension: The fixed dimensionality of vectors in this store.
                       If None, the dimension is inferred from the first added vector.
        """
        self.dimension: Optional[int] = dimension
        self._vectors_list: List[np.ndarray] = []
        self._ids_list: List[str] = []
        self._metadata_list: List[Dict[str, Any]] = []
        
        # Cached 2-D numpy matrix of vectors
        self._vectors_matrix: Optional[np.ndarray] = None
        # Set to track ID uniqueness in O(1)
        self._ids_set = set()

    @property
    def size(self) -> int:
        """Return the number of vectors stored."""
        return len(self._ids_list)

    @property
    def vectors(self) -> np.ndarray:
        """Return the vector database as a 2-D NumPy array of shape (N, D)."""
        if self._vectors_matrix is None:
            if not self._vectors_list:
                dim = self.dimension or 0
                return np.empty((0, dim), dtype=np.float32)
            self._vectors_matrix = np.stack(self._vectors_list).astype(np.float32)
        return self._vectors_matrix

    @property
    def ids(self) -> List[str]:
        """Return a copy of the list of vector IDs."""
        return list(self._ids_list)

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        """Return a copy of the list of vector metadata dictionaries."""
        return [dict(m) for m in self._metadata_list]

    def add_vector(
        self, 
        vector_id: str, 
        vector: np.ndarray, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a single vector to the store.

        Args:
            vector_id: Unique string identifier for the vector.
            vector: 1-D NumPy array or sequence of numbers.
            metadata: Optional dictionary containing metadata associated with the vector.

        Raises:
            ValueError: If vector_id already exists, dimension mismatch, or vector shape is invalid.
        """
        if not isinstance(vector_id, str):
            raise ValueError(f"vector_id must be a string. Got type {type(vector_id)}")
        
        if vector_id in self._ids_set:
            raise ValueError(f"Vector ID '{vector_id}' already exists in the store.")

        # Ensure vector is a 1-D numpy array
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        else:
            vector = vector.astype(np.float32)

        if vector.ndim != 1:
            raise ValueError(f"Vector must be 1-D. Got shape {vector.shape}")

        # Inferred dimension logic
        if self.dimension is None:
            self.dimension = vector.shape[0]
        elif vector.shape[0] != self.dimension:
            raise ValueError(
                f"Dimension mismatch. Expected {self.dimension}, got {vector.shape[0]}."
            )

        self._vectors_list.append(vector)
        self._ids_list.append(vector_id)
        self._ids_set.add(vector_id)
        self._metadata_list.append(metadata if metadata is not None else {})
        
        # Invalidate cached matrix
        self._vectors_matrix = None

    def add_vectors(
        self, 
        vector_ids: List[str], 
        vectors: np.ndarray, 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add multiple vectors in batch to the store.

        Args:
            vector_ids: List of unique string identifiers.
            vectors: 2-D array (or convertible) of shape (N, D).
            metadatas: Optional list of dictionaries containing metadata.

        Raises:
            ValueError: If lengths of inputs mismatch, IDs are duplicate, or vector dimension is incorrect.
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
        else:
            vectors = vectors.astype(np.float32)

        if vectors.ndim != 2:
            raise ValueError(f"Vectors must be a 2-D array. Got shape {vectors.shape}")

        n_vectors = vectors.shape[0]
        if len(vector_ids) != n_vectors:
            raise ValueError(
                f"Mismatch: Got {len(vector_ids)} IDs for {n_vectors} vectors."
            )

        if metadatas is not None and len(metadatas) != n_vectors:
            raise ValueError(
                f"Mismatch: Got {len(metadatas)} metadatas for {n_vectors} vectors."
            )

        # Check dimension
        if self.dimension is None:
            self.dimension = vectors.shape[1]
        elif vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension mismatch. Expected {self.dimension}, got {vectors.shape[1]}."
            )

        # Validate unique IDs
        for vid in vector_ids:
            if not isinstance(vid, str):
                raise ValueError(f"vector_id must be a string. Got type {type(vid)}")
            if vid in self._ids_set:
                raise ValueError(f"Vector ID '{vid}' already exists in the store.")

        # Batch insert
        for i, vid in enumerate(vector_ids):
            self._vectors_list.append(vectors[i])
            self._ids_list.append(vid)
            self._ids_set.add(vid)
            meta = metadatas[i] if metadatas is not None else {}
            self._metadata_list.append(meta if meta is not None else {})

        # Invalidate cached matrix
        self._vectors_matrix = None

    def remove_vector(self, vector_id: str) -> None:
        """Remove a single vector from the store by its ID.

        Args:
            vector_id: The unique string identifier of the vector to remove.

        Raises:
            KeyError: If vector_id does not exist in the store.
        """
        if vector_id not in self._ids_set:
            raise KeyError(f"Vector ID '{vector_id}' not found in the store.")
        
        idx = self._ids_list.index(vector_id)
        self._vectors_list.pop(idx)
        self._ids_list.pop(idx)
        self._ids_set.remove(vector_id)
        self._metadata_list.pop(idx)
        
        # Invalidate cached matrix
        self._vectors_matrix = None

    def save(self, filepath: str) -> None:
        """Save the vector store to a secure ZIP archive container.

        Args:
            filepath: Path to the target file.
        """
        import zipfile
        import io
        import hashlib
        
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # 1. Prepare vectors.npy
        vectors_np = self.vectors
        vectors_buffer = io.BytesIO()
        np.save(vectors_buffer, vectors_np)
        vectors_bytes = vectors_buffer.getvalue()
        
        # Calculate SHA-256 checksum of raw vectors.npy bytes
        checksum = hashlib.sha256(vectors_bytes).hexdigest()

        # 2. Prepare metadata.json and index_config.json
        version_data = {"version": "1.0.0"}
        config_data = {
            "dimension": self.dimension or 0,
            "size": self.size,
            "vectors_checksum": checksum
        }
        
        ids_meta_data = {
            "ids": self._ids_list,
            "metadata": self._metadata_list
        }

        # 3. Write all to zip archive
        with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("version.json", json.dumps(version_data, indent=2))
            archive.writestr("index_config.json", json.dumps(config_data, indent=2))
            archive.writestr("metadata.json", json.dumps(ids_meta_data, indent=2))
            archive.writestr("vectors.npy", vectors_bytes)

    def load(self, filepath: str) -> None:
        """Load the vector store from a secure ZIP archive container.

        Args:
            filepath: Path to the ZIP archive.

        Raises:
            FileNotFoundError: If filepath does not exist.
            ValueError: If file content is invalid, corrupted, or version mismatch.
        """
        import zipfile
        import io
        import hashlib
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            with zipfile.ZipFile(filepath, "r") as archive:
                # 1. Read and validate version
                if "version.json" not in archive.namelist():
                    raise ValueError("Invalid format: missing version.json")
                version_data = json.loads(archive.read("version.json").decode("utf-8"))
                if version_data.get("version") != "1.0.0":
                    raise ValueError(f"Unsupported schema version: {version_data.get('version')}")

                # 2. Read index config
                if "index_config.json" not in archive.namelist():
                    raise ValueError("Invalid format: missing index_config.json")
                config_data = json.loads(archive.read("index_config.json").decode("utf-8"))
                expected_checksum = config_data.get("vectors_checksum")

                # 3. Read vectors.npy and check corruption
                if "vectors.npy" not in archive.namelist():
                    raise ValueError("Invalid format: missing vectors.npy")
                vectors_bytes = archive.read("vectors.npy")
                
                # Checksum validation
                actual_checksum = hashlib.sha256(vectors_bytes).hexdigest()
                if actual_checksum != expected_checksum:
                    raise ValueError("Corruption detected: vectors.npy checksum mismatch.")

                vectors_matrix = np.load(io.BytesIO(vectors_bytes)).astype(np.float32)

                # 4. Read metadata and IDs
                if "metadata.json" not in archive.namelist():
                    raise ValueError("Invalid format: missing metadata.json")
                ids_meta_data = json.loads(archive.read("metadata.json").decode("utf-8"))
                ids_list = ids_meta_data.get("ids", [])
                metadata_list = ids_meta_data.get("metadata", [])

                # Verify dimension and counts match
                dim = config_data.get("dimension")
                if dim > 0:
                    self.dimension = dim
                    if vectors_matrix.shape[0] > 0 and vectors_matrix.shape[1] != self.dimension:
                        raise ValueError(f"Dimension mismatch: expected {self.dimension}, got {vectors_matrix.shape[1]}")
                else:
                    self.dimension = vectors_matrix.shape[1] if vectors_matrix.shape[0] > 0 else None

                if len(ids_list) != vectors_matrix.shape[0]:
                    raise ValueError(f"Inconsistent file size: got {len(ids_list)} IDs for {vectors_matrix.shape[0]} vectors.")

                # Reinitialize internal state
                self._vectors_matrix = vectors_matrix
                self._vectors_list = list(vectors_matrix)
                self._ids_list = ids_list
                self._ids_set = set(ids_list)
                self._metadata_list = metadata_list

        except zipfile.BadZipFile as e:
            raise ValueError(f"Invalid ZIP archive or corrupted file structure: {e}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise ValueError(f"Failed to load VectorStore: {e}")

    def search(self, query: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """Perform nearest neighbor search using brute-force cosine similarity.

        Args:
            query: 1-D query vector of shape (D,).
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            Ranked list of SearchResult matches.
        """
        from vector_engine.app.search import brute_force_search, SearchResult
        return brute_force_search(self, query, top_k)
