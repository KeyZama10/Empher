import hashlib
import bisect
import threading
from typing import List, Dict, Optional

class HashRing:
    """A thread-safe Consistent Hashing Ring implementation using SHA-256."""

    def __init__(self, replicas: int = 100):
        """Initialize Consistent Hash Ring.

        Args:
            replicas: The number of virtual nodes per physical node.
        """
        if replicas <= 0:
            raise ValueError(f"replicas must be a positive integer. Got {replicas}.")
        
        self.replicas: int = replicas
        self._lock = threading.Lock()
        
        # Mapping: hash (int) -> physical node string (e.g. "localhost:50051")
        self.ring: Dict[int, str] = {}
        
        # Keep sorted list of virtual node hashes
        self.sorted_keys: List[int] = []

    def _hash(self, key: str) -> int:
        """Compute the SHA-256 hash of a string key, returning an integer."""
        h = hashlib.sha256(key.encode('utf-8')).hexdigest()
        return int(h, 16)

    def add_node(self, node_id: str) -> None:
        """Add a physical node to the hash ring by creating its virtual nodes.

        Args:
            node_id: Unique string identifier of the physical node.
        """
        with self._lock:
            # Prevent duplicate inserts that would pollute sorted_keys
            # We check if any virtual node for this node_id is already in the ring
            for i in range(self.replicas):
                vnode_key = f"{node_id}#vnode_{i}"
                vnode_hash = self._hash(vnode_key)
                
                if vnode_hash not in self.ring:
                    self.ring[vnode_hash] = node_id
                    # Insert in sorted order
                    bisect.insort(self.sorted_keys, vnode_hash)

    def remove_node(self, node_id: str) -> None:
        """Remove all virtual nodes belonging to a physical node from the ring.

        Args:
            node_id: Unique string identifier of the physical node to remove.
        """
        with self._lock:
            for i in range(self.replicas):
                vnode_key = f"{node_id}#vnode_{i}"
                vnode_hash = self._hash(vnode_key)
                
                if vnode_hash in self.ring:
                    del self.ring[vnode_hash]
                    # Find index and remove from sorted_keys
                    idx = bisect.bisect_left(self.sorted_keys, vnode_hash)
                    if idx < len(self.sorted_keys) and self.sorted_keys[idx] == vnode_hash:
                        self.sorted_keys.pop(idx)

    def get_node(self, key: str) -> Optional[str]:
        """Find the physical node responsible for the given key.

        Binary searches the ring clockwise to find the first virtual node 
        whose hash is >= the key's hash. Wraps around if none is found.

        Args:
            key: The string key to route (e.g. vector ID).

        Returns:
            The physical node ID mapping to the key, or None if the ring is empty.
        """
        with self._lock:
            if not self.sorted_keys:
                return None
            
            key_hash = self._hash(key)
            # Find the position of the first hash >= key_hash
            idx = bisect.bisect_right(self.sorted_keys, key_hash)
            
            # Wrap around the ring if key_hash is greater than all nodes
            if idx == len(self.sorted_keys):
                idx = 0
                
            return self.ring[self.sorted_keys[idx]]
