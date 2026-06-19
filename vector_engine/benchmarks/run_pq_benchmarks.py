import os
import time
import numpy as np
from typing import Dict, Any

from vector_engine.app.pq import ProductQuantizer
from vector_engine.app.vector_store import VectorStore
from vector_engine.app.search import brute_force_search

def generate_synthetic_data(
    n_vectors: int, 
    n_queries: int, 
    dimension: int, 
    seed: int = 42
) -> tuple:
    """Generate reproducible L2-normalized synthetic dataset."""
    np.random.seed(seed)
    base_vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    norms = np.linalg.norm(base_vectors, axis=1, keepdims=True)
    base_vectors = base_vectors / np.where(norms == 0.0, 1.0, norms)
    
    vector_ids = [f"vec_{i}" for i in range(n_vectors)]
    
    query_vectors = np.random.randn(n_queries, dimension).astype(np.float32)
    q_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    query_vectors = query_vectors / np.where(q_norms == 0.0, 1.0, q_norms)
    
    return base_vectors, vector_ids, query_vectors

def run_pq_benchmarks():
    print("Generating synthetic dataset (10,000 vectors, 128 dims)...")
    base_vectors, vector_ids, query_vectors = generate_synthetic_data(
        n_vectors=10000,
        n_queries=100,
        dimension=128,
        seed=42
    )
    
    # 1. Split into training and evaluation sets
    # We use 2,000 vectors for codebook training, and evaluate on all 10,000
    train_vectors = base_vectors[:2000]
    
    # Initialize ProductQuantizer
    # Split 128 dimensions into 16 subspaces (each subspace is 8 dimensions)
    # n_centroids = 256 allows 1-byte representation per subspace code
    n_subspaces = 16
    n_centroids = 256
    pq = ProductQuantizer(n_subspaces=n_subspaces, n_centroids=n_centroids)
    
    print(f"Training ProductQuantizer on {len(train_vectors)} vectors (M={n_subspaces}, K={n_centroids})...")
    t_train_start = time.perf_counter()
    pq.train(train_vectors)
    training_time = (time.perf_counter() - t_train_start)
    print(f"Quantizer trained in {training_time:.2f} seconds.")
    
    # 2. Encode all vectors
    print("Encoding 10,000 vectors into compact byte codes...")
    t_enc_start = time.perf_counter()
    codes = pq.encode(base_vectors)
    encoding_time = (time.perf_counter() - t_enc_start) * 1000.0  # ms
    print(f"Encoded in {encoding_time:.2f} ms ({encoding_time/len(base_vectors):.4f} ms per vector).")
    
    # 3. Decode vectors
    print("Decoding byte codes back to reconstructed float32 vectors...")
    t_dec_start = time.perf_counter()
    reconstructed_vectors = pq.decode(codes)
    decoding_time = (time.perf_counter() - t_dec_start) * 1000.0  # ms
    print(f"Decoded in {decoding_time:.2f} ms.")
    
    # 4. Measure size details
    # Original: 10,000 vectors * 128 dims * 4 bytes
    original_bytes = base_vectors.nbytes
    # PQ: 10,000 vectors * 16 subspaces * 1 byte (uint8) + codebooks overhead
    # Codebooks: 16 subspaces * 256 centroids * 8 dimensions * 4 bytes
    codebooks_bytes = pq.codebooks.nbytes if pq.codebooks is not None else 0
    pq_codes_bytes = codes.nbytes
    total_pq_bytes = pq_codes_bytes + codebooks_bytes
    
    compression_ratio = original_bytes / pq_codes_bytes
    memory_savings = (1 - (pq_codes_bytes / original_bytes)) * 100.0
    
    # 5. Measure Recall Loss
    # We run top_k=10 searches on the original vectors vs reconstructed vectors
    print("Evaluating recall loss...")
    
    # Exact Baseline Store
    exact_store = VectorStore(dimension=128)
    exact_store.add_vectors(vector_ids, base_vectors)
    
    # Reconstructed PQ Store
    pq_store = VectorStore(dimension=128)
    pq_store.add_vectors(vector_ids, reconstructed_vectors)
    
    recalls = []
    top_k = 10
    
    for idx, q in enumerate(query_vectors):
        res_exact = brute_force_search(exact_store, q, top_k=top_k)
        res_pq = brute_force_search(pq_store, q, top_k=top_k)
        
        exact_ids = set([r.id for r in res_exact])
        pq_ids = set([r.id for r in res_pq])
        
        intersection = exact_ids.intersection(pq_ids)
        recall = len(intersection) / top_k
        recalls.append(recall)
        
    avg_recall = np.mean(recalls)
    recall_loss = 1.0 - avg_recall
    
    print("\n==========================================================================")
    print("                             PQ BENCHMARK SUMMARY                         ")
    print("==========================================================================")
    print(f"Original Byte Size:           {original_bytes / 1024.0:.1f} KB")
    print(f"PQ Codes Byte Size:           {pq_codes_bytes / 1024.0:.1f} KB")
    print(f"Codebooks Overhead:           {codebooks_bytes / 1024.0:.1f} KB")
    print(f"Compression Ratio (Codes):    {compression_ratio:.1f}x")
    print(f"Memory Savings % (Codes):     {memory_savings:.3f}%")
    print(f"Average Recall @ 10:          {avg_recall:.4f}")
    print(f"Recall Loss @ 10:             {recall_loss:.4f} ({recall_loss * 100.0:.2f}%)")
    print("==========================================================================\n")
    
    # Generate Markdown Report
    report_path = os.path.join(os.path.dirname(__file__), "..", "pq_benchmark_report.md")
    
    report_content = f"""# Product Quantization (PQ) Benchmark & Scaling Report

This report evaluates **Product Quantization (PQ)** for high-dimensional vector compression and discusses architectural designs required to scale the system to **100 Million vectors**.

## Evaluation Parameters
- **Dataset Size**: 10,000 vectors
- **Vector Dimension**: 128 dimensions (float32)
- **Subspaces ($M$)**: 16 subspaces (subspace dimension $d = 8$)
- **Centroids per Subspace ($K$)**: 256 centroids (represented as 1-byte `uint8` indexes)
- **Training Pool**: 2,000 vectors

## Performance Metrics

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Original Memory Size** | {original_bytes / 1024.0:.2f} KB | Raw float32 array size |
| **Quantized Codes Size** | {pq_codes_bytes / 1024.0:.2f} KB | Array of compressed uint8 codes |
| **Codebook Overhead** | {codebooks_bytes / 1024.0:.2f} KB | Codebook centroids stored in float32 |
| **Compression Ratio (Codes)** | {compression_ratio:.1f}x | Size reduction factor of vector array |
| **Memory Savings** | {memory_savings:.3f}% | Percentage of memory footprint reduced |
| **Average Recall @ 10** | {avg_recall:.4f} | Intersection rate of top-10 neighbors |
| **Recall Loss** | {recall_loss * 100.0:.2f}% | Loss in recall due to lossy quantization |

---

## Design and Scaling Architecture to 100 Million Vectors

Scaling a vector search engine to **100M vectors** introduces substantial compute, memory, and disk bottlenecks. Product Quantization serves as a core enabling technology. Below is the technical architecture designed to support this scale.

### 1. Memory Budget Estimation
- **Raw Vectors**: 100,000,000 vectors * 128 dimensions * 4 bytes = 51.2 GB of RAM.
- **PQ Codes (M=16)**: 100,000,000 vectors * 16 subspaces * 1 byte = 1.6 GB of RAM.
*Conclusion*: PQ compression allows the index to easily fit into standard system memory, reducing infrastructure cost by **96.8%**.

### 2. Sample-Based Training
Training KMeans on 100M vectors is computationally infeasible. Our design:
- Decouples training from indexing.
- Extracts a random, representative subset of **20,000 to 50,000 vectors** from the corpus.
- Fits the codebooks on this subset, then freezes the centroids.
- Quantizes the remaining 100M vectors in parallel using the frozen codebooks.

### 3. Streaming & Batch Processing (Memory-Constrained Systems)
For systems where the initial uncompressed dataset cannot fit into memory:
- **Streaming Pipeline**: Load raw vectors from disk (e.g., CSV, parquet, or binary stream) in chunks of 50,000.
- **Incremental Encoding**: Quantize each chunk and append the resulting `uint8` codes to a memory-mapped file (`np.memmap`) on disk.
- This ensures a constant memory overhead (under 100 MB) during the entire ingestion pipeline.

### 4. Asymmetric Distance Computation (ADC)
To avoid decoding vectors at search time (which is slow), we should implement **ADC**:
1. When a query vector q arrives, split it into M subspaces.
2. For each subspace, precompute the distance from q_m to all K=256 centroids in the codebook. This creates a lookup table of shape (M, 256) in O(M * K) time.
3. For each indexed vector (represented by M bytes), sum the precomputed distances from the lookup table. This requires only M table lookups and additions per vector, avoiding floating-point dimension math entirely.
4. Scale search throughput using CPU SIMD vectorization (AVX-512) and multiprocessing.

"""
    
    with open(report_path, "w") as f:
        f.write(report_content)
        
    print(f"Generated PQ benchmark report at: {os.path.abspath(report_path)}")

if __name__ == "__main__":
    run_pq_benchmarks()
