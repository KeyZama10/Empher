import os
import shutil
import pickle
import zipfile
import json
import pytest
import numpy as np
from sklearn.cluster import KMeans
from vector_engine.app.ivf_index import IVFIndex
from vector_engine.app.vector_store import VectorStore
from vector_engine.app.storage.migration import migrate_legacy_ivf

class LegacyKMeans:
    def __init__(self, centers):
        self.cluster_centers_ = centers

@pytest.fixture
def temp_dir():
    path = "vector_engine/data/test_security"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)

def test_legacy_migration_and_compatibility(temp_dir):
    """Test migrating a legacy pickle-based IVFIndex to the new ZIP format."""
    legacy_file = os.path.join(temp_dir, "legacy_ivf.pkl")
    new_file = os.path.join(temp_dir, "migrated_ivf.idx")
    
    # 1. Train and construct a valid IVFIndex using the legacy structure
    index = IVFIndex(n_clusters=2, dimension=2)
    training_data = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
    index.train(training_data)
    
    # Add vectors
    index.add_vectors(
        ["v1", "v2", "v3"],
        np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float64),
        [{"lbl": "x"}, {"lbl": "y"}, {"lbl": "diag"}]
    )
    
    legacy_state = {
        "n_clusters": index.n_clusters,
        "dimension": index.dimension,
        "is_trained": index._is_trained,
        "ids_set": index._ids_set,
        "kmeans": LegacyKMeans(index.centroids),
        "buckets": index.buckets
    }
    
    with open(legacy_file, "wb") as f:
        pickle.dump(legacy_state, f)
        
    # Verify that calling load() directly on a legacy file fails (backward compatibility barrier)
    index_fail = IVFIndex(n_clusters=2, dimension=2)
    with pytest.raises(ValueError) as excinfo:
        index_fail.load(legacy_file)
    assert "Invalid ZIP archive" in str(excinfo.value)
    
    # 2. Perform Migration
    migrate_legacy_ivf(legacy_file, new_file)
    assert os.path.exists(new_file)
    
    # 3. Load migrated index
    migrated_index = IVFIndex(n_clusters=2, dimension=2)
    migrated_index.load(new_file)
    
    # Verify properties and search functionality
    assert migrated_index.is_trained
    assert migrated_index.total_vectors == 3
    
    results = migrated_index.search(np.array([1.0, 0.0]), top_k=2)
    assert len(results) >= 1
    assert results[0].id == "v1"
    assert results[0].metadata == {"lbl": "x"}

def test_schema_version_validation(temp_dir):
    """Verify loading rejects unknown or unsupported schema versions."""
    valid_file = os.path.join(temp_dir, "valid_store.idx")
    bad_version_file = os.path.join(temp_dir, "bad_version_store.idx")
    
    store = VectorStore(dimension=2)
    store.add_vector("v1", np.array([1.0, 0.0]), {"lbl": "x"})
    store.save(valid_file)
    
    # Corrupt the version.json inside the ZIP
    with zipfile.ZipFile(valid_file, "r") as src:
        with zipfile.ZipFile(bad_version_file, "w") as dest:
            for name in src.namelist():
                if name == "version.json":
                    # Write an unsupported version
                    dest.writestr("version.json", json.dumps({"version": "2.0.0"}))
                else:
                    dest.writestr(name, src.read(name))
                    
    # Attempt to load the index with invalid version
    new_store = VectorStore()
    with pytest.raises(ValueError) as excinfo:
        new_store.load(bad_version_file)
    assert "Unsupported schema version: 2.0.0" in str(excinfo.value)

def test_data_corruption_detection(temp_dir):
    """Verify checksum checks trigger failure on corrupted vectors.npy file."""
    valid_file = os.path.join(temp_dir, "valid_store.idx")
    corrupt_file = os.path.join(temp_dir, "corrupt_store.idx")
    
    store = VectorStore(dimension=2)
    store.add_vector("v1", np.array([1.0, 0.0]), {"lbl": "x"})
    store.save(valid_file)
    
    # Modify data inside vectors.npy to simulate silent bit-rot/corruption
    with zipfile.ZipFile(valid_file, "r") as src:
        with zipfile.ZipFile(corrupt_file, "w") as dest:
            for name in src.namelist():
                if name == "vectors.npy":
                    # Write modified/corrupted array bytes
                    dest.writestr(name, b"corrupted bytes here 12345")
                else:
                    dest.writestr(name, src.read(name))
                    
    # Attempt to load should fail checksum check
    new_store = VectorStore()
    with pytest.raises(ValueError) as excinfo:
        new_store.load(corrupt_file)
    assert "checksum mismatch" in str(excinfo.value)
