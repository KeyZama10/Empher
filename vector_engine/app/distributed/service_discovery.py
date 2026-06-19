import os
import json
import base64
import time
import logging
import threading
import httpx
from typing import List, Dict, Any, Optional

logger = logging.getLogger("service_discovery")

LOCK_FILE_PATH = "vector_engine/data/service_discovery_mock.lock"
MOCK_FILE_PATH = "vector_engine/data/service_discovery_mock.json"

try:
    import fcntl
except ImportError:
    fcntl = None

class ProcessLock:
    def __init__(self):
        self._thread_lock = threading.Lock()
        self.fd = None

    def __enter__(self):
        self._thread_lock.acquire()
        if fcntl:
            try:
                os.makedirs(os.path.dirname(LOCK_FILE_PATH), exist_ok=True)
                self.fd = open(LOCK_FILE_PATH, "w")
                fcntl.flock(self.fd, fcntl.LOCK_EX)
            except Exception:
                self.fd = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
                self.fd.close()
            except Exception:
                pass
        self._thread_lock.release()

# Cross-process and cross-thread lock for mock registry
_mock_thread_lock = ProcessLock()

def _load_mock_data() -> Dict[str, Any]:
    os.makedirs(os.path.dirname(MOCK_FILE_PATH), exist_ok=True)
    if not os.path.exists(MOCK_FILE_PATH):
        return {"registry": {}, "locks": {}}
    for attempt in range(5):
        try:
            with open(MOCK_FILE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            time.sleep(0.05)
    return {"registry": {}, "locks": {}}

def _save_mock_data(data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(MOCK_FILE_PATH), exist_ok=True)
    for attempt in range(5):
        try:
            tmp_path = MOCK_FILE_PATH + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f)
            os.replace(tmp_path, MOCK_FILE_PATH)
            return
        except Exception:
            time.sleep(0.05)

class EtcdServiceRegistry:
    """Dynamic service discovery and leader election coordinator using etcd v3 or shared file mock."""

    def __init__(self, etcd_endpoints: Optional[str] = None, use_mock_fallback: bool = True):
        self.etcd_endpoints = etcd_endpoints or os.getenv("ETCD_ENDPOINTS", "http://localhost:2379")
        self.use_mock_fallback = use_mock_fallback
        self.use_mock = False
        
        # Test connection to etcd
        try:
            resp = httpx.post(
                f"{self.etcd_endpoints}/v3/kv/range", 
                json={"key": base64.b64encode(b"ping").decode("utf-8")}, 
                timeout=0.2
            )
            if resp.status_code == 200:
                logger.info("Successfully connected to etcd cluster.")
            else:
                raise ConnectionError()
        except Exception:
            if self.use_mock_fallback:
                self.use_mock = True
                logger.info("etcd cluster not reachable. Falling back to Shared File-Backed Mock Registry.")
            else:
                raise ConnectionError("Could not connect to etcd server.")

    def _to_b64(self, val: str) -> str:
        return base64.b64encode(val.encode("utf-8")).decode("utf-8")

    def _from_b64(self, val: str) -> str:
        return base64.b64decode(val.encode("utf-8")).decode("utf-8")

    def register_worker(
        self, 
        worker_id: str, 
        host: str, 
        port: int, 
        role: str = "primary", 
        primary_addr: str = "", 
        ttl: int = 5
    ) -> None:
        """Register a worker node to the registry."""
        data = {
            "worker_id": worker_id,
            "host": host,
            "port": port,
            "role": role,
            "primary_addr": primary_addr,
            "ttl": ttl,
            "last_seen": time.time()
        }
        
        if self.use_mock:
            with _mock_thread_lock:
                state = _load_mock_data()
                state["registry"][worker_id] = data
                _save_mock_data(state)
            return

        key = f"workers/{worker_id}"
        val = json.dumps(data)
        
        try:
            payload = {
                "key": self._to_b64(key),
                "value": self._to_b64(val)
            }
            resp = httpx.post(f"{self.etcd_endpoints}/v3/kv/put", json=payload, timeout=1.0)
            if resp.status_code != 200:
                raise Exception(f"etcd register failed: {resp.text}")
        except Exception as e:
            logger.error(f"Error registering worker to etcd: {e}. Falling back to mock.")
            self.use_mock = True
            with _mock_thread_lock:
                state = _load_mock_data()
                state["registry"][worker_id] = data
                _save_mock_data(state)

    def deregister_worker(self, worker_id: str) -> None:
        """Remove a worker node from the registry."""
        if self.use_mock:
            with _mock_thread_lock:
                state = _load_mock_data()
                state["registry"].pop(worker_id, None)
                _save_mock_data(state)
            return

        key = f"workers/{worker_id}"
        try:
            payload = {
                "key": self._to_b64(key)
            }
            httpx.post(f"{self.etcd_endpoints}/v3/kv/deleterange", json=payload, timeout=1.0)
        except Exception as e:
            logger.error(f"Error deregistering worker from etcd: {e}")

    def heartbeat(self, worker_id: str) -> None:
        """Update last_seen timestamp to keep registration alive."""
        if self.use_mock:
            with _mock_thread_lock:
                state = _load_mock_data()
                if worker_id in state["registry"]:
                    state["registry"][worker_id]["last_seen"] = time.time()
                    _save_mock_data(state)
            return

        key = f"workers/{worker_id}"
        try:
            payload = {"key": self._to_b64(key)}
            resp = httpx.post(f"{self.etcd_endpoints}/v3/kv/range", json=payload, timeout=1.0)
            kvs = resp.json().get("kvs", [])
            if kvs:
                data = json.loads(self._from_b64(kvs[0]["value"]))
                data["last_seen"] = time.time()
                put_payload = {
                    "key": self._to_b64(key),
                    "value": self._to_b64(json.dumps(data))
                }
                httpx.post(f"{self.etcd_endpoints}/v3/kv/put", json=put_payload, timeout=1.0)
        except Exception as e:
            logger.error(f"Error sending heartbeat to etcd: {e}")

    def get_active_workers(self) -> List[Dict[str, Any]]:
        """Retrieve all currently healthy worker configurations."""
        now = time.time()
        
        if self.use_mock:
            with _mock_thread_lock:
                state = _load_mock_data()
                active = []
                expired = []
                for wid, info in state["registry"].items():
                    if now - info["last_seen"] <= info["ttl"]:
                        active.append(info)
                    else:
                        expired.append(wid)
                
                if expired:
                    for wid in expired:
                        state["registry"].pop(wid, None)
                    _save_mock_data(state)
                return active

        key_prefix = "workers/"
        range_end = "workers0"
        try:
            payload = {
                "key": self._to_b64(key_prefix),
                "range_end": self._to_b64(range_end)
            }
            resp = httpx.post(f"{self.etcd_endpoints}/v3/kv/range", json=payload, timeout=1.0)
            kvs = resp.json().get("kvs", [])
            
            active = []
            for kv in kvs:
                info = json.loads(self._from_b64(kv["value"]))
                if now - info["last_seen"] <= info["ttl"]:
                    active.append(info)
                else:
                    self.deregister_worker(info["worker_id"])
            return active
        except Exception as e:
            logger.error(f"Error fetching active workers from etcd: {e}. Returning mock values.")
            self.use_mock = True
            return self.get_active_workers()

    def acquire_leader_lock(self, coordinator_id: str, ttl: int = 5) -> bool:
        """Attempt to acquire primary coordinator leader lock."""
        now = time.time()
        lock_key = "coordinators/leader"
        lock_data = {
            "coordinator_id": coordinator_id,
            "acquired_at": now,
            "ttl": ttl
        }
        
        if self.use_mock:
            with _mock_thread_lock:
                state = _load_mock_data()
                current_lock = state["locks"].get(lock_key)
                if current_lock:
                    if now - current_lock["acquired_at"] > current_lock["ttl"]:
                        state["locks"][lock_key] = lock_data
                        _save_mock_data(state)
                        return True
                    if current_lock["coordinator_id"] == coordinator_id:
                        current_lock["acquired_at"] = now
                        _save_mock_data(state)
                        return True
                    return False
                else:
                    state["locks"][lock_key] = lock_data
                    _save_mock_data(state)
                    return True

        try:
            payload = {"key": self._to_b64(lock_key)}
            resp = httpx.post(f"{self.etcd_endpoints}/v3/kv/range", json=payload, timeout=1.0)
            kvs = resp.json().get("kvs", [])
            
            if kvs:
                current = json.loads(self._from_b64(kvs[0]["value"]))
                if now - current["acquired_at"] <= current["ttl"]:
                    if current["coordinator_id"] == coordinator_id:
                        put_payload = {
                            "key": self._to_b64(lock_key),
                            "value": self._to_b64(json.dumps(lock_data))
                        }
                        httpx.post(f"{self.etcd_endpoints}/v3/kv/put", json=put_payload, timeout=1.0)
                        return True
                    return False
            
            put_payload = {
                "key": self._to_b64(lock_key),
                "value": self._to_b64(json.dumps(lock_data))
            }
            httpx.post(f"{self.etcd_endpoints}/v3/kv/put", json=put_payload, timeout=1.0)
            return True
        except Exception as e:
            logger.error(f"Error acquiring leader lock: {e}. Falling back to mock lock.")
            self.use_mock = True
            return self.acquire_leader_lock(coordinator_id, ttl)

    def release_leader_lock(self, coordinator_id: str) -> None:
        """Release the leader lock."""
        lock_key = "coordinators/leader"
        if self.use_mock:
            with _mock_thread_lock:
                state = _load_mock_data()
                current_lock = state["locks"].get(lock_key)
                if current_lock and current_lock["coordinator_id"] == coordinator_id:
                    state["locks"].pop(lock_key, None)
                    _save_mock_data(state)
            return

        try:
            payload = {"key": self._to_b64(lock_key)}
            resp = httpx.post(f"{self.etcd_endpoints}/v3/kv/range", json=payload, timeout=1.0)
            kvs = resp.json().get("kvs", [])
            if kvs:
                current = json.loads(self._from_b64(kvs[0]["value"]))
                if current["coordinator_id"] == coordinator_id:
                    del_payload = {"key": self._to_b64(lock_key)}
                    httpx.post(f"{self.etcd_endpoints}/v3/kv/deleterange", json=del_payload, timeout=1.0)
        except Exception as e:
            logger.error(f"Error releasing leader lock: {e}")

    def check_leader(self) -> Optional[str]:
        """Query who the active leader is."""
        now = time.time()
        lock_key = "coordinators/leader"
        
        if self.use_mock:
            with _mock_thread_lock:
                state = _load_mock_data()
                current_lock = state["locks"].get(lock_key)
                if current_lock and now - current_lock["acquired_at"] <= current_lock["ttl"]:
                    return current_lock["coordinator_id"]
                return None

        try:
            payload = {"key": self._to_b64(lock_key)}
            resp = httpx.post(f"{self.etcd_endpoints}/v3/kv/range", json=payload, timeout=1.0)
            kvs = resp.json().get("kvs", [])
            if kvs:
                current = json.loads(self._from_b64(kvs[0]["value"]))
                if now - current["acquired_at"] <= current["ttl"]:
                    return current["coordinator_id"]
            return None
        except Exception:
            self.use_mock = True
            return self.check_leader()
