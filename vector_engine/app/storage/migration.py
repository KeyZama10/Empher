import os
import pickle
from typing import Any
from vector_engine.app.ivf_index import IVFIndex

def migrate_legacy_ivf(legacy_filepath: str, new_filepath: str) -> None:
    """Migrate a legacy pickle-based IVFIndex file to the new secure ZIP format.

    Args:
        legacy_filepath: Path to the legacy pickle index file.
        new_filepath: Target path to write the migrated index file.

    Raises:
        FileNotFoundError: If the legacy file is missing.
        ValueError: If the legacy file is corrupted or does not contain required states.
    """
    if not os.path.exists(legacy_filepath):
        raise FileNotFoundError(f"Legacy pickle file not found: {legacy_filepath}")

    # Load using legacy pickle specifically for migration
    try:
        with open(legacy_filepath, "rb") as f:
            state = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse legacy pickle file: {e}")

    # Validate state structure
    required_keys = ["n_clusters", "dimension", "is_trained", "ids_set", "kmeans", "buckets"]
    for key in required_keys:
        if key not in state:
            raise ValueError(f"Corrupted legacy state: missing key '{key}'")

    # Reconstruct temporary IVFIndex from state
    index = IVFIndex(n_clusters=state["n_clusters"], dimension=state["dimension"])
    index._is_trained = state["is_trained"]
    index._ids_set = set(state["ids_set"])
    if state["is_trained"] and state["kmeans"] is not None:
        index.centroids = state["kmeans"].cluster_centers_
    else:
        index.centroids = None
    index.buckets = state["buckets"]

    # Save to the new secure ZIP format
    index.save(new_filepath)
