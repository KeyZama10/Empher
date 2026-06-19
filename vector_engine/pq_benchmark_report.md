# Product Quantization (PQ) Benchmark & Scaling Report

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
| **Original Memory Size** | 5000.00 KB | Raw float32 array size |
| **Quantized Codes Size** | 156.25 KB | Array of compressed uint8 codes |
| **Codebook Overhead** | 128.00 KB | Codebook centroids stored in float32 |
| **Compression Ratio (Codes)** | 32.0x | Size reduction factor of vector array |
| **Memory Savings** | 96.875% | Percentage of memory footprint reduced |
| **Average Recall @ 10** | 0.2940 | Intersection rate of top-10 neighbors |
| **Recall Loss** | 70.60% | Loss in recall due to lossy quantization |

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

