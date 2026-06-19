import os
import numpy as np

from vector_engine.app.vector_store import VectorStore
from vector_engine.app.search import brute_force_search

def run_demo() -> None:
    print("==================================================")
    print("  Distributed Vector Search Engine - Phase 1 Demo  ")
    print("==================================================")

    # 1. Initialize VectorStore with dimension 4
    dim = 4
    print(f"\n1. Initializing VectorStore with dimension: {dim}")
    store = VectorStore(dimension=dim)

    # 2. Add some test vectors
    print("\n2. Adding vectors to the store:")
    data = [
        ("v1", [1.0, 0.0, 0.0, 0.0], {"name": "X-Axis Unit Vector", "category": "math"}),
        ("v2", [0.8, 0.6, 0.0, 0.0], {"name": "Diagonal First Quadrant", "category": "math"}),
        ("v3", [0.0, 0.0, 1.0, 0.0], {"name": "Z-Axis Unit Vector", "category": "science"}),
        ("v4", [0.0, 0.0, 0.7, 0.7], {"name": "Diagonal Plane Z-W", "category": "science"}),
    ]
    
    for vid, vec, meta in data:
        vec_arr = np.array(vec, dtype=np.float32)
        store.add_vector(vector_id=vid, vector=vec_arr, metadata=meta)
        print(f"   Added '{vid}': {vec} | Metadata: {meta}")

    print(f"   Total stored vectors: {store.size}")

    # 3. Save the vector store to disk
    filepath = "vector_db_demo.npz"
    print(f"\n3. Saving VectorStore to disk at '{filepath}'...")
    store.save(filepath)

    # 4. Load the database into a new VectorStore instance
    print("\n4. Loading database into a NEW VectorStore instance...")
    new_store = VectorStore()
    new_store.load(filepath)
    print(f"   Loaded store size: {new_store.size}")
    print(f"   Loaded store dimension: {new_store.dimension}")

    # 5. Perform brute-force nearest neighbor search
    query = np.array([1.0, 0.2, 0.0, 0.0], dtype=np.float32)
    top_k = 3
    print(f"\n5. Performing brute-force nearest neighbor search:")
    print(f"   Query vector: {query.tolist()}")
    print(f"   Top-K requested: {top_k}")

    results = brute_force_search(new_store, query, top_k=top_k)

    print("\n   Results (ordered by cosine similarity descending):")
    for rank, res in enumerate(results, start=1):
        print(f"   [{rank}] ID: {res.id: <4} | Score: {res.score:.5f} | Metadata: {res.metadata}")

    # Clean up the file created for the demo
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"\n6. Cleaned up demo file '{filepath}'.")

    print("\nDemo completed successfully!")
    print("==================================================")

if __name__ == "__main__":
    run_demo()
