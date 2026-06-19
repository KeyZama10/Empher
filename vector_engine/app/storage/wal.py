import os
import sys
import json
import time
import threading
import numpy as np
from typing import List, Dict, Any, Optional

class WALManager:
    """Manages Write-Ahead Logging (WAL) and snapshotting for worker node durability."""

    def __init__(self, log_dir: str):
        """Initialize WAL Manager.

        Args:
            log_dir: The directory where WAL logs and snapshots are stored.
        """
        self.log_dir: str = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.wal_path: str = os.path.join(self.log_dir, "active.wal")
        self.snapshot_path: str = os.path.join(self.log_dir, "snapshot.npz")
        
        self._lock = threading.Lock()
        # Open in append mode (will create the file if it does not exist)
        self._wal_file = open(self.wal_path, "a")

    def _write_entry(self, entry: Dict[str, Any]) -> None:
        """Helper to write, flush and fsync a WAL entry in a thread-safe manner."""
        line = json.dumps(entry) + "\n"
        with self._lock:
            if self._wal_file and not self._wal_file.closed:
                self._wal_file.write(line)
                self._wal_file.flush()
                # Force OS to write to physical disk
                try:
                    os.fsync(self._wal_file.fileno())
                except OSError:
                    # Ignore if the filesystem doesn't support fsync (e.g. some mock environments)
                    pass

    def append_insert(self, vector_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Append an insert mutation to the log.

        Args:
            vector_id: Unique identifier of the vector.
            vector: 1-D float list representing the vector.
            metadata: Associated metadata dictionary.
        """
        entry = {
            "op": "insert",
            "id": vector_id,
            "vector": list(vector),
            "metadata": metadata if metadata is not None else {},
            "timestamp": time.time()
        }
        self._write_entry(entry)

    def append_delete(self, vector_id: str) -> None:
        """Append a delete mutation to the log.

        Args:
            vector_id: Unique identifier of the vector to delete.
        """
        entry = {
            "op": "delete",
            "id": vector_id,
            "timestamp": time.time()
        }
        self._write_entry(entry)

    def append_clear(self) -> None:
        """Append a index clear mutation to the log."""
        entry = {
            "op": "clear",
            "timestamp": time.time()
        }
        self._write_entry(entry)

    def replay(self, store) -> None:
        """Replay snapshot and WAL mutations to restore the VectorStore state.

        Args:
            store: The VectorStore instance to rebuild.
        """
        with self._lock:
            # Close active.wal file handle temporarily to avoid reads during updates
            if self._wal_file and not self._wal_file.closed:
                self._wal_file.close()

            # 1. Load Snapshot if it exists
            if os.path.exists(self.snapshot_path):
                try:
                    store.load(self.snapshot_path)
                except Exception as e:
                    print(f"Warning: Failed to load snapshot: {e}", file=sys.stderr)

            # 2. Replay subsequent WAL entries line-by-line
            if os.path.exists(self.wal_path):
                with open(self.wal_path, "r") as f:
                    for line_num, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            op = entry.get("op")
                            if op == "insert":
                                vector_id = entry["id"]
                                vector = np.array(entry["vector"], dtype=np.float32)
                                metadata = entry.get("metadata", {})
                                
                                # Overwrite if the key already exists (replayed state correction)
                                if vector_id in store._ids_set:
                                    idx = store._ids_list.index(vector_id)
                                    store._vectors_list.pop(idx)
                                    store._ids_list.pop(idx)
                                    store._ids_set.remove(vector_id)
                                    store._metadata_list.pop(idx)
                                    store._vectors_matrix = None
                                    
                                store.add_vector(vector_id, vector, metadata)
                            elif op == "delete":
                                vector_id = entry["id"]
                                if vector_id in store._ids_set:
                                    store.remove_vector(vector_id)
                            elif op == "clear":
                                # Reset store state
                                store.dimension = None
                                store._vectors_list = []
                                store._ids_list = []
                                store._ids_set = set()
                                store._metadata_list = []
                                store._vectors_matrix = None
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            # Log warning and skip malformed entries
                            print(f"Warning: Corrupted WAL entry at line {line_num}: {e}. Skipping.", file=sys.stderr)

            # Re-open active.wal in append mode
            self._wal_file = open(self.wal_path, "a")

    def rotate(self, store) -> None:
        """Create a compact snapshot of the memory state and truncate the WAL log."""
        with self._lock:
            # 1. Close current WAL file
            if self._wal_file and not self._wal_file.closed:
                self._wal_file.flush()
                try:
                    os.fsync(self._wal_file.fileno())
                except OSError:
                    pass
                self._wal_file.close()

            # 2. Save current store state to temporary snapshot
            # NumPy save adds '.npz' automatically, so we make the temporary path end in '.npz'
            tmp_snapshot = self.snapshot_path + ".tmp.npz"
            store.save(tmp_snapshot)
            
            # Atomic swap of snapshot files
            if os.path.exists(self.snapshot_path):
                os.remove(self.snapshot_path)
            os.rename(tmp_snapshot, self.snapshot_path)

            # 3. Truncate active.wal log
            with open(self.wal_path, "w") as f:
                f.truncate(0)

            # 4. Re-open active.wal log in append mode
            self._wal_file = open(self.wal_path, "a")

    def close(self) -> None:
        """Close the active WAL file handle cleanly."""
        with self._lock:
            if self._wal_file and not self._wal_file.closed:
                self._wal_file.flush()
                try:
                    os.fsync(self._wal_file.fileno())
                except OSError:
                    pass
                self._wal_file.close()
