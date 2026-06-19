import os
import json
import numpy as np
import hnswlib
from typing import List, Dict, Any, Optional

from vector_engine.app.base_index import BaseVectorIndex
from vector_engine.app.search import SearchResult

class HNSWIndex(BaseVectorIndex):
    """Approximate Nearest Neighbor Search index using HNSW (hierarchical navigable small world) graph."""

    def __init__(
        self, 
        dimension: int, 
        max_elements: int = 10000, 
        M: int = 16, 
        ef_construction: int = 200, 
        ef_search: int = 50,
        space: str = "cosine"
    ):
        """Initialize HNSW Index.

        Args:
            dimension: Dimensionality of vectors.
            max_elements: The maximum capacity of the index.
            M: Number of bi-directional links created for every new element during construction.
            ef_construction: Size of the dynamic candidate list during construction.
            ef_search: Size of the dynamic candidate list during query search.
            space: Metric space ('cosine', 'l2', or 'ip').
        """
        if dimension <= 0:
            raise ValueError(f"dimension must be positive. Got {dimension}.")
        if max_elements <= 0:
            raise ValueError(f"max_elements must be positive. Got {max_elements}.")

        self.dimension: int = dimension
        self.max_elements: int = max_elements
        self.M: int = M
        self.ef_construction: int = ef_construction
        self.ef_search: int = ef_search
        self.space: str = space

        # Create the hnswlib instance
        self.index = hnswlib.Index(space=self.space, dim=self.dimension)
        self.index.init_index(
            max_elements=self.max_elements, 
            ef_construction=self.ef_construction, 
            M=self.M
        )
        self.index.set_ef(self.ef_search)

        # Mapping string IDs <-> unique integer IDs required by hnswlib
        self._str_to_int: Dict[str, int] = {}
        self._int_to_str: Dict[int, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._current_int_id: int = 0

    @property
    def size(self) -> int:
        """Returns the number of indexed elements."""
        return self.index.element_count

    def add_vectors(
        self, 
        vector_ids: List[str], 
        vectors: np.ndarray, 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add multiple vectors to the HNSW index.

        Args:
            vector_ids: List of unique string identifiers.
            vectors: 2-D array of shape (N, D).
            metadatas: Optional list of dictionaries containing metadata.

        Raises:
            ValueError: If dimensions mismatch, lengths mismatch, capacity exceeded, or ID duplicate.
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
        else:
            vectors = vectors.astype(np.float32)

        if vectors.ndim != 2:
            raise ValueError(f"Vectors must be 2-D. Got shape {vectors.shape}")

        n_vectors, d = vectors.shape
        if d != self.dimension:
            raise ValueError(
                f"Dimension mismatch. Expected {self.dimension}, got vector dimension {d}."
            )

        if len(vector_ids) != n_vectors:
            raise ValueError(
                f"Mismatch: Got {len(vector_ids)} IDs for {n_vectors} vectors."
            )

        if metadatas is not None and len(metadatas) != n_vectors:
            raise ValueError(
                f"Mismatch: Got {len(metadatas)} metadatas for {n_vectors} vectors."
            )

        # Check capacity
        if self.index.element_count + n_vectors > self.max_elements:
            raise ValueError(
                f"Cannot add {n_vectors} vectors. Index size ({self.index.element_count}) "
                f"plus new vectors exceeds max capacity of {self.max_elements}."
            )

        # Check for duplicate IDs
        for vid in vector_ids:
            if not isinstance(vid, str):
                raise ValueError(f"vector_id must be a string. Got type {type(vid)}")
            if vid in self._str_to_int:
                raise ValueError(f"Vector ID '{vid}' already exists in the HNSW Index.")

        int_ids = []
        for i, vid in enumerate(vector_ids):
            int_id = self._current_int_id
            self._current_int_id += 1
            
            self._str_to_int[vid] = int_id
            self._int_to_str[int_id] = vid
            meta = metadatas[i] if metadatas is not None else {}
            self._metadata[vid] = meta if meta is not None else {}
            
            int_ids.append(int_id)

        # Add to hnswlib index
        int_ids_np = np.array(int_ids, dtype=np.int64)
        self.index.add_items(vectors, int_ids_np)

    def search(self, query: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """Search the HNSW index for the nearest neighbors.

        Args:
            query: 1-D query vector of shape (D,).
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            Ranked list of SearchResult matches.

        Raises:
            ValueError: If index is empty, query shape is invalid, or top_k <= 0.
        """
        if self.index.element_count == 0:
            raise ValueError("Cannot perform search on an empty HNSW Index.")

        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer. Got {top_k}.")

        if not isinstance(query, np.ndarray):
            query = np.array(query, dtype=np.float32)
        else:
            query = query.astype(np.float32)

        if query.ndim != 1:
            raise ValueError(f"Query vector must be 1-D. Got shape {query.shape}.")

        if query.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension ({query.shape[0]}) does not match "
                f"Index dimension ({self.dimension})."
            )

        # hnswlib requires top_k <= index size
        k = min(top_k, self.index.element_count)
        if k <= 0:
            return []

        # Reshape query for query run: (1, D)
        query_2d = query.reshape(1, -1)
        
        # Query the graph
        labels, distances = self.index.knn_query(query_2d, k=k)

        results = []
        for int_id, dist in zip(labels[0], distances[0]):
            int_id = int(int_id)
            str_id = self._int_to_str[int_id]
            
            # hnswlib cosine distance = 1.0 - cosine_similarity
            score = float(1.0 - dist)
            
            results.append(
                SearchResult(
                    id=str_id,
                    score=score,
                    metadata=self._metadata.get(str_id, {})
                )
            )

        return results

    def save(self, filepath: str) -> None:
        """Save the HNSW index graph and its associated ID/metadata mappings.

        Args:
            filepath: Target graph file destination.
        """
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # 1. Save HNSW binary graph
        self.index.save_index(filepath)

        # 2. Save metadata mapping
        meta_filepath = filepath + ".meta.json"
        meta_data = {
            "dimension": self.dimension,
            "max_elements": self.max_elements,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "space": self.space,
            "current_int_id": self._current_int_id,
            "str_to_int": self._str_to_int,
            # JSON keys must be strings
            "int_to_str": {str(k): v for k, v in self._int_to_str.items()},
            "metadata": self._metadata
        }
        with open(meta_filepath, "w") as f:
            json.dump(meta_data, f, indent=2)

    def load(self, filepath: str) -> None:
        """Load the HNSW index graph and associated ID/metadata mappings.

        Args:
            filepath: Source graph file.

        Raises:
            FileNotFoundError: If graph or metadata mapping file is missing.
        """
        meta_filepath = filepath + ".meta.json"
        if not os.path.exists(filepath) or not os.path.exists(meta_filepath):
            raise FileNotFoundError(
                f"Missing HNSWIndex files at '{filepath}' or '{meta_filepath}'."
            )

        # 1. Load metadata mapping
        with open(meta_filepath, "r") as f:
            meta_data = json.load(f)

        self.dimension = meta_data["dimension"]
        self.max_elements = meta_data["max_elements"]
        self.M = meta_data["M"]
        self.ef_construction = meta_data["ef_construction"]
        self.ef_search = meta_data["ef_search"]
        self.space = meta_data["space"]
        self._current_int_id = meta_data["current_int_id"]
        self._str_to_int = meta_data["str_to_int"]
        self._int_to_str = {int(k): v for k, v in meta_data["int_to_str"].items()}
        self._metadata = meta_data["metadata"]

        # 2. Re-initialize and load HNSW binary index
        self.index = hnswlib.Index(space=self.space, dim=self.dimension)
        self.index.load_index(filepath, max_elements=self.max_elements)
        self.index.set_ef(self.ef_search)
