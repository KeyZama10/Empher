import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Any, Optional

from vector_engine.app.base_index import BaseVectorIndex
from vector_engine.app.search import SearchResult
from vector_engine.app.utils import cosine_similarity_matrix

class IVFIndex(BaseVectorIndex):
    """Inverted File Index (IVF) vector search engine using KMeans clustering."""

    def __init__(self, n_clusters: int = 4, dimension: Optional[int] = None):
        """Initialize the IVF Index.

        Args:
            n_clusters: Configurable number of clusters (centroids).
            dimension: The dimensionality of the vectors. If None, it will be set upon training.
        """
        if n_clusters <= 0:
            raise ValueError(f"n_clusters must be a positive integer. Got {n_clusters}.")
        
        self.n_clusters: int = n_clusters
        self.dimension: Optional[int] = dimension
        self.centroids: Optional[np.ndarray] = None
        self.kmeans: Optional[KMeans] = None
        
        # Inverted lists: centroid index -> Dict with 'vectors', 'ids', and 'metadata' lists
        self.buckets: Dict[int, Dict[str, Any]] = {}
        self._ids_set = set()
        self._is_trained: bool = False

    @property
    def is_trained(self) -> bool:
        """Returns whether the index is trained."""
        return self._is_trained

    @property
    def total_vectors(self) -> int:
        """Returns the total number of indexed vectors."""
        return len(self._ids_set)

    def train(self, vectors: np.ndarray) -> None:
        """Train the index (fit KMeans centroids) using a set of training vectors.

        Args:
            vectors: 2-D array of vectors of shape (N, D).

        Raises:
            ValueError: If vectors is not 2-D, dimension mismatch, or number of vectors < n_clusters.
        """
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float64)
        else:
            vectors = vectors.astype(np.float64)

        if vectors.ndim != 2:
            raise ValueError(f"Training vectors must be a 2-D array. Got shape {vectors.shape}")

        n_samples, d = vectors.shape

        if n_samples < self.n_clusters:
            raise ValueError(
                f"Number of training vectors ({n_samples}) must be at least the "
                f"number of clusters ({self.n_clusters}) to train KMeans."
            )

        if self.dimension is None:
            self.dimension = d
        elif d != self.dimension:
            raise ValueError(
                f"Dimension mismatch. Expected {self.dimension}, got training dimension {d}."
            )

        # Initialize and fit KMeans clustering
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=42, 
            n_init="auto"
        )
        self.kmeans.fit(vectors)
        self.centroids = self.kmeans.cluster_centers_
        self._is_trained = True

        # Initialize empty inverted lists (buckets)
        self.buckets = {
            i: {
                "vectors": [],
                "ids": [],
                "metadata": []
            }
            for i in range(self.n_clusters)
        }
        self._ids_set.clear()

    def add_vectors(
        self, 
        vector_ids: List[str], 
        vectors: np.ndarray, 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Index a set of vectors by assigning them to their nearest cluster.

        Args:
            vector_ids: List of unique string identifiers.
            vectors: 2-D array of shape (N, D) containing vectors to index.
            metadatas: Optional list of dictionaries containing metadata.

        Raises:
            ValueError: If index is not trained, dimension mismatch, or duplicate ID.
        """
        if not self._is_trained or self.centroids is None:
            raise ValueError("IVFIndex must be trained before adding vectors.")

        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float64)
        else:
            vectors = vectors.astype(np.float64)

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

        # Check for duplicates first to ensure transactional consistency
        for vid in vector_ids:
            if not isinstance(vid, str):
                raise ValueError(f"vector_id must be a string. Got type {type(vid)}")
            if vid in self._ids_set:
                raise ValueError(f"Vector ID '{vid}' already exists in the IVF Index.")

        # Assign vectors to nearest cluster centroids using Euclidean distance broadcasting
        dists = np.sum((vectors[:, np.newaxis, :] - self.centroids) ** 2, axis=2)
        cluster_assignments = np.argmin(dists, axis=1)

        # Put vectors into their assigned buckets
        for idx, cluster_idx in enumerate(cluster_assignments):
            vid = vector_ids[idx]
            cluster_idx = int(cluster_idx)
            
            self.buckets[cluster_idx]["vectors"].append(vectors[idx])
            self.buckets[cluster_idx]["ids"].append(vid)
            
            meta = metadatas[idx] if metadatas is not None else {}
            self.buckets[cluster_idx]["metadata"].append(meta if meta is not None else {})
            self._ids_set.add(vid)

    def search(self, query: np.ndarray, top_k: int = 5, nprobe: int = 1) -> List[SearchResult]:
        """Query the IVF index by locating the nearest nprobe centroids and searching those clusters.

        Args:
            query: 1-D query vector of shape (D,).
            top_k: Maximum number of search results to return.
            nprobe: Number of nearest centroids to search.

        Returns:
            Ranked list of SearchResult matches.

        Raises:
            ValueError: If index not trained, query shape is invalid, top_k <= 0, or nprobe <= 0.
        """
        if not self._is_trained or self.centroids is None:
            raise ValueError("IVFIndex must be trained before performing searches.")

        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer. Got {top_k}.")

        if nprobe <= 0:
            raise ValueError(f"nprobe must be a positive integer. Got {nprobe}.")

        if not isinstance(query, np.ndarray):
            query = np.array(query, dtype=np.float64)
        else:
            query = query.astype(np.float64)

        if query.ndim != 1:
            raise ValueError(f"Query vector must be 1-D. Got shape {query.shape}.")

        if query.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension ({query.shape[0]}) does not match "
                f"Index dimension ({self.dimension})."
            )

        # 1. Predict nearest nprobe cluster centroids using Euclidean distance
        dists = np.sum((self.centroids - query) ** 2, axis=1)
        actual_nprobe = min(nprobe, self.n_clusters)
        closest_centroids = np.argsort(dists)[:actual_nprobe]

        # 2. Extract and merge cluster contents
        bucket_vectors = []
        bucket_ids = []
        bucket_meta = []
        for cluster_idx in closest_centroids:
            bucket = self.buckets[int(cluster_idx)]
            bucket_vectors.extend(bucket["vectors"])
            bucket_ids.extend(bucket["ids"])
            bucket_meta.extend(bucket["metadata"])

        if not bucket_vectors:
            # All selected clusters are empty
            return []

        # 3. Perform brute-force search only on the vectors in these clusters
        vectors_matrix = np.stack(bucket_vectors)
        similarities = cosine_similarity_matrix(query, vectors_matrix)

        # Sort the subset of vectors
        n_elements = len(bucket_ids)
        k = min(top_k, n_elements)

        if k < n_elements:
            unsorted_top_k = np.argpartition(similarities, -k)[-k:]
            sorted_indices = unsorted_top_k[
                np.argsort(similarities[unsorted_top_k])[::-1]
            ]
        else:
            sorted_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in sorted_indices:
            results.append(
                SearchResult(
                    id=bucket_ids[idx],
                    score=float(similarities[idx]),
                    metadata=bucket_meta[idx]
                )
            )

        return results

    def save(self, filepath: str) -> None:
        """Save the IVFIndex to a secure ZIP archive container.

        Args:
            filepath: Path to the target file.
        """
        import zipfile
        import io
        import hashlib
        import json

        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        # 1. Gather all vectors and their assignments from buckets
        all_vectors = []
        all_ids = []
        all_metadata = []
        bucket_assignments = []
        
        for centroid_idx in range(self.n_clusters):
            if centroid_idx in self.buckets:
                b = self.buckets[centroid_idx]
                for idx in range(len(b["ids"])):
                    all_vectors.append(b["vectors"][idx])
                    all_ids.append(b["ids"][idx])
                    all_metadata.append(b["metadata"][idx])
                    bucket_assignments.append(centroid_idx)

        # Build buffers for numpy arrays
        centroids_bytes = b""
        vectors_bytes = b""
        assignments_bytes = b""
        
        centroids_checksum = ""
        vectors_checksum = ""
        assignments_checksum = ""

        # 2. Serialize centroids if trained
        if self._is_trained and self.centroids is not None:
            centroids_np = self.centroids.astype(np.float64)
            c_buf = io.BytesIO()
            np.save(c_buf, centroids_np)
            centroids_bytes = c_buf.getvalue()
            centroids_checksum = hashlib.sha256(centroids_bytes).hexdigest()

        # 3. Serialize vectors and bucket assignments if populated
        if all_vectors:
            vectors_np = np.stack(all_vectors).astype(np.float64)
            v_buf = io.BytesIO()
            np.save(v_buf, vectors_np)
            vectors_bytes = v_buf.getvalue()
            vectors_checksum = hashlib.sha256(vectors_bytes).hexdigest()

            assignments_np = np.array(bucket_assignments, dtype=np.int64)
            a_buf = io.BytesIO()
            np.save(a_buf, assignments_np)
            assignments_bytes = a_buf.getvalue()
            assignments_checksum = hashlib.sha256(assignments_bytes).hexdigest()

        # 4. Prepare JSON configurations
        version_data = {"version": "1.0.0"}
        config_data = {
            "n_clusters": self.n_clusters,
            "dimension": self.dimension or 0,
            "is_trained": self._is_trained,
            "centroids_checksum": centroids_checksum,
            "vectors_checksum": vectors_checksum,
            "assignments_checksum": assignments_checksum
        }
        
        metadata_data = {
            "ids": all_ids,
            "metadata": all_metadata
        }

        # 5. Write everything to zip archive
        with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("version.json", json.dumps(version_data, indent=2))
            archive.writestr("index_config.json", json.dumps(config_data, indent=2))
            archive.writestr("metadata.json", json.dumps(metadata_data, indent=2))
            if centroids_bytes:
                archive.writestr("centroids.npy", centroids_bytes)
            if vectors_bytes:
                archive.writestr("vectors.npy", vectors_bytes)
                archive.writestr("bucket_assignments.npy", assignments_bytes)

    def load(self, filepath: str) -> None:
        """Load the IVFIndex from a secure ZIP archive container.

        Args:
            filepath: Path to the ZIP archive.

        Raises:
            FileNotFoundError: If filepath does not exist.
            ValueError: If file content is invalid, corrupted, or version mismatch.
        """
        import zipfile
        import io
        import hashlib
        import json

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
                
                self.n_clusters = config_data["n_clusters"]
                dim = config_data.get("dimension")
                self.dimension = dim if dim > 0 else None
                self._is_trained = config_data["is_trained"]

                # 3. Read centroids if trained
                if self._is_trained:
                    if "centroids.npy" not in archive.namelist():
                        raise ValueError("Corrupted index: missing centroids.npy for trained index.")
                    centroids_bytes = archive.read("centroids.npy")
                    
                    # Validate checksum
                    expected_c_sum = config_data.get("centroids_checksum")
                    if hashlib.sha256(centroids_bytes).hexdigest() != expected_c_sum:
                        raise ValueError("Corruption detected: centroids.npy checksum mismatch.")

                    self.centroids = np.load(io.BytesIO(centroids_bytes))
                else:
                    self.centroids = None

                # Initialize empty buckets
                self.buckets = {
                    i: {
                        "vectors": [],
                        "ids": [],
                        "metadata": []
                    }
                    for i in range(self.n_clusters)
                }
                self._ids_set = set()

                # 4. Read vectors and bucket assignments if present
                if "vectors.npy" in archive.namelist():
                    if "bucket_assignments.npy" not in archive.namelist() or "metadata.json" not in archive.namelist():
                        raise ValueError("Corrupted index: incomplete datasets in archive.")
                    
                    vectors_bytes = archive.read("vectors.npy")
                    assignments_bytes = archive.read("bucket_assignments.npy")
                    
                    # Validate checksums
                    expected_v_sum = config_data.get("vectors_checksum")
                    expected_a_sum = config_data.get("assignments_checksum")
                    if hashlib.sha256(vectors_bytes).hexdigest() != expected_v_sum:
                        raise ValueError("Corruption detected: vectors.npy checksum mismatch.")
                    if hashlib.sha256(assignments_bytes).hexdigest() != expected_a_sum:
                        raise ValueError("Corruption detected: bucket_assignments.npy checksum mismatch.")

                    vectors = np.load(io.BytesIO(vectors_bytes))
                    assignments = np.load(io.BytesIO(assignments_bytes))
                    
                    metadata_data = json.loads(archive.read("metadata.json").decode("utf-8"))
                    ids_list = metadata_data.get("ids", [])
                    metadata_list = metadata_data.get("metadata", [])

                    # Verification checks
                    if len(ids_list) != vectors.shape[0] or len(assignments) != vectors.shape[0]:
                        raise ValueError("Corrupted index: inconsistent data array lengths.")

                    # Reconstruct buckets state
                    for idx, bucket_idx in enumerate(assignments):
                        bucket_idx = int(bucket_idx)
                        self.buckets[bucket_idx]["vectors"].append(vectors[idx])
                        self.buckets[bucket_idx]["ids"].append(ids_list[idx])
                        self.buckets[bucket_idx]["metadata"].append(metadata_list[idx])
                        self._ids_set.add(ids_list[idx])

        except zipfile.BadZipFile as e:
            raise ValueError(f"Invalid ZIP archive or corrupted file structure: {e}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise ValueError(f"Failed to load IVFIndex: {e}")
