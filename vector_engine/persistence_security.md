# Vector Engine - Persistence Security & Storage Layout

This document details the security model, storage specifications, and migration path implemented to completely eliminate **Remote Code Execution (RCE)** risks in the Distributed Vector Search Engine.

---

## Why Python Pickle is Dangerous in Production

Python's standard `pickle` serialization module is **highly insecure** when used to read untrusted or external files. It was designed for internal object copying, not as a safe database storage format.

### The Mechanics of Pickle Remote Code Execution (RCE)
The primary hazard arises because `pickle` executes arbitrary Python code during deserialization.
When `pickle.load()` parses a file, it reads bytecode instructions. The bytecode can contain the `GLOBAL` instruction (represented by the `c` opcode in the pickle virtual machine). This allows the deserializer to import *any* module available in the Python path (e.g. `os`, `sys`, `subprocess`) and call *any* constructor or function with arbitrary arguments.

For example, a malicious actor can construct a payload like:
```python
import pickle
import os

class MaliciousPayload:
    def __reduce__(self):
        # Executes this shell command when pickle.loads() is called
        return (os.system, ("rm -rf /tmp/important_data || echo 'RCE Executed!'",))

payload = pickle.dumps(MaliciousPayload())
```
When this file is loaded, Python immediately triggers `os.system(...)` in the context of the running application process. If the vector database runs as a privileged user in a Kubernetes container or local cluster, this leads to complete system compromise.

---

## Secure ZIP-Based Storage Layout

To eliminate this vulnerability, the engine replaces all `pickle` and `allow_pickle=True` actions with a **secure, declarative ZIP archive container**. The container holds raw data formats that are parsed strictly using native primitives (JSON and binary NumPy matrix buffers) without code evaluation.

```
index_archive.idx (ZIP Container)
├── version.json             <-- Version checking schema
├── index_config.json        <-- Non-pickle hyperparameters & checksums
├── metadata.json            <-- Vector ID mappings (JSON string lists)
├── vectors.npy              <-- Raw numeric binary floats (NumPy array)
├── centroids.npy            <-- Raw KMeans centroids array (IVFIndex only)
└── bucket_assignments.npy   <-- Shard node index assignments (IVFIndex only)
```

### File Details & Schema

#### 1. `version.json`
Ensures that the client and engine are aligned on schema versioning, rejecting files that lack a matching version string:
```json
{
  "version": "1.0.0"
}
```

#### 2. `index_config.json`
Holds hyperparameters, state flags, and calculated SHA-256 hashes of the NumPy files to detect data corruption:
```json
{
  "n_clusters": 4,
  "dimension": 128,
  "is_trained": true,
  "centroids_checksum": "a5e8f...",
  "vectors_checksum": "b3f91...",
  "assignments_checksum": "f8d1c..."
}
```

#### 3. `metadata.json`
Stores the association of vector IDs to key-value metadata:
```json
{
  "ids": ["vec_1", "vec_2"],
  "metadata": [
    {"name": "unit_vector_x"},
    {"name": "unit_vector_y"}
  ]
}
```

#### 4. `.npy` files
Raw NumPy binary files (`vectors.npy`, `centroids.npy`, and `bucket_assignments.npy`) are loaded strictly as contiguous C-arrays using `np.load` without passing `allow_pickle=True`, completely avoiding code execution.

---

## Integrity Check & Corruption Detection

1. **SHA-256 Checksum Validation**: During saving, the raw byte representation of `.npy` arrays is hashed. When loading, the engine re-calculates the SHA-256 of the loaded file and verifies it matches the hash recorded in `index_config.json`. Any bit-rot, truncation, or malicious editing of the array is caught immediately.
2. **Dimension Consistency**: Rejects file loads if the shape of `vectors.npy` does not match the configured dimensions.
3. **Count Consistency**: Validates that the number of vector rows matches the length of the ID list and metadata lists.
