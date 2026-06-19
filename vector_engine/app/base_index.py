from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any, Optional
from vector_engine.app.search import SearchResult

class BaseVectorIndex(ABC):
    """Abstract Base Class defining the interface for all vector index implementations."""

    @abstractmethod
    def add_vectors(
        self, 
        vector_ids: List[str], 
        vectors: np.ndarray, 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add multiple vectors to the index.

        Args:
            vector_ids: List of unique string identifiers.
            vectors: 2-D array of shape (N, D) containing the vectors.
            metadatas: Optional list of dictionaries containing metadata.
        """
        pass

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """Perform nearest neighbor search.

        Args:
            query: 1-D query vector of shape (D,).
            top_k: The number of nearest neighbors to return.

        Returns:
            A list of SearchResult objects sorted by score descending.
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Persist the index state to disk.

        Args:
            filepath: Target destination path.
        """
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load the index state from disk.

        Args:
            filepath: Source file path.
        """
        pass
