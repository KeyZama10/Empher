import asyncio
import json
import logging
import random
import grpc
from typing import List, Dict, Any, Optional

from vector_engine.app.proto import engine_pb2, engine_pb2_grpc
from vector_engine.app.search import SearchResult
from vector_engine.app.distributed.hash_ring import HashRing

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] Coordinator: %(message)s")
logger = logging.getLogger("coordinator")

class Coordinator:
    """Manages sharding, worker node pools, gRPC broadcasts, merges, replica routing, and standby failovers."""

    def __init__(
        self, 
        coordinator_id: str = "coordinator_1", 
        standby: bool = False, 
        timeout: float = 2.0, 
        max_retries: int = 3
    ):
        """Initialize Coordinator.

        Args:
            coordinator_id: Unique string identifying this coordinator instance.
            standby: If True, boots as standby and performs active-passive leader election.
            timeout: Maximum timeout in seconds for gRPC worker queries.
            max_retries: Max retry attempts for transient worker request failures.
        """
        self.coordinator_id: str = coordinator_id
        self.is_standby: bool = standby
        self.is_leader: bool = not standby
        self.timeout: float = timeout
        self.max_retries: int = max_retries
        
        # worker_addr ("host:port") -> dict
        self.workers: Dict[str, Dict[str, Any]] = {}
        # primary_addr -> list of replica_addr
        self.replicas_map: Dict[str, List[str]] = {}
        
        self._health_check_task: Optional[asyncio.Task] = None
        self._leader_election_task: Optional[asyncio.Task] = None
        self.hash_ring = HashRing(replicas=100)

        # Service discovery registry
        from vector_engine.app.distributed.service_discovery import EtcdServiceRegistry
        self.registry = EtcdServiceRegistry()

    def start(self) -> None:
        """Start the background loops (election and worker sync)."""
        loop = asyncio.get_event_loop()
        if self.is_standby:
            self._leader_election_task = loop.create_task(self._leader_election_loop())
            # Start sync workers loop in background even when standby to keep stubs ready
            self.start_health_check_loop(2.0)
        else:
            self.start_health_check_loop(2.0)

    async def _leader_election_loop(self) -> None:
        """Lock acquisition loop for active-passive standby failover."""
        while True:
            try:
                acquired = self.registry.acquire_leader_lock(self.coordinator_id, ttl=4)
                if acquired:
                    if not self.is_leader:
                        logger.info(f"Coordinator {self.coordinator_id} successfully acquired leader lock! Promoting to ACTIVE LEADER.")
                        self.is_leader = True
                else:
                    if self.is_leader:
                        logger.warning(f"Coordinator {self.coordinator_id} lost leader lock! Demoting to STANDBY.")
                        self.is_leader = False
            except Exception as e:
                logger.error(f"Error in leader election loop: {e}")
            await asyncio.sleep(1.0)

    async def sync_workers_from_registry(self) -> None:
        """Fetch registered nodes from dynamic discovery and sync local connections."""
        try:
            active_nodes = self.registry.get_active_workers()
            active_addrs = set()
            
            replicas_by_primary = {}
            primaries = []
            
            for node in active_nodes:
                addr = f"{node['host']}:{node['port']}"
                active_addrs.add(addr)
                
                if addr not in self.workers:
                    channel = grpc.aio.insecure_channel(addr)
                    stub = engine_pb2_grpc.SearchWorkerStub(channel)
                    self.workers[addr] = {
                        "channel": channel,
                        "stub": stub,
                        "is_active": True,
                        "role": node["role"],
                        "primary_addr": node["primary_addr"]
                    }
                else:
                    self.workers[addr]["role"] = node["role"]
                    self.workers[addr]["primary_addr"] = node["primary_addr"]
                    
                if node["role"] == "primary":
                    primaries.append(addr)
                elif node["role"] == "replica":
                    p_addr = node["primary_addr"]
                    if p_addr:
                        replicas_by_primary.setdefault(p_addr, []).append(addr)
            
            self.replicas_map = replicas_by_primary

            # Clean up disconnected nodes
            to_remove = []
            for addr in list(self.workers.keys()):
                if addr not in active_addrs:
                    to_remove.append(addr)
            
            for addr in to_remove:
                channel = self.workers[addr]["channel"]
                await channel.close()
                self.workers.pop(addr)
                self.hash_ring.remove_node(addr)
                
            # Keep hash ring aligned with active primaries
            for p_addr in primaries:
                if p_addr not in self.hash_ring.ring.values():
                    self.hash_ring.add_node(p_addr)
                    
        except Exception as e:
            logger.error(f"Error syncing workers from registry: {e}")

    def register_worker(self, host: str, port: int) -> None:
        """Register worker statically (manual integration fallback)."""
        addr = f"{host}:{port}"
        if addr in self.workers:
            return
        channel = grpc.aio.insecure_channel(addr)
        stub = engine_pb2_grpc.SearchWorkerStub(channel)
        self.workers[addr] = {
            "channel": channel,
            "stub": stub,
            "is_active": True,
            "role": "primary",
            "primary_addr": ""
        }
        self.hash_ring.add_node(addr)
        logger.info(f"Statically registered worker node: {addr}")

    async def remove_worker(self, host: str, port: int) -> None:
        """Remove worker and close channels."""
        addr = f"{host}:{port}"
        if addr in self.workers:
            channel = self.workers[addr]["channel"]
            await channel.close()
            self.workers.pop(addr)
            self.hash_ring.remove_node(addr)
            logger.info(f"Deregistered worker node: {addr}")

    def start_health_check_loop(self, interval: float = 2.0) -> None:
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_loop(interval))

    def stop_health_check_loop(self) -> None:
        if self._health_check_task is not None and not self._health_check_task.done():
            self._health_check_task.cancel()

    async def _health_loop(self, interval: float) -> None:
        while True:
            try:
                # Sync dynamically from etcd registry
                await self.sync_workers_from_registry()
                await self.check_workers_health()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

    async def check_workers_health(self) -> None:
        for addr, info in list(self.workers.items()):
            stub = info["stub"]
            try:
                await stub.HealthCheck(engine_pb2.HealthRequest(), timeout=1.0)
                if not info["is_active"]:
                    info["is_active"] = True
                    if info["role"] == "primary":
                        self.hash_ring.add_node(addr)
            except Exception:
                if info["is_active"]:
                    info["is_active"] = False
                    if info["role"] == "primary":
                        self.hash_ring.remove_node(addr)

    async def _call_with_retry(self, stub_method, request) -> Any:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                return await stub_method(request, timeout=self.timeout)
            except grpc.RpcError as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(0.1 * (2 ** attempt))
        raise last_err

    async def insert_vector(
        self, 
        vector_id: str, 
        vector: List[float], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Route insert operation strictly to the primary node of the partition shard."""
        if not self.is_leader:
            raise PermissionError(f"Coordinator {self.coordinator_id} is in STANDBY mode and cannot perform insertions.")

        target_addr = self.hash_ring.get_node(vector_id)
        if not target_addr:
            raise ValueError("No active search worker nodes available to index vector.")
        
        stub = self.workers[target_addr]["stub"]
        meta_json = json.dumps(metadata) if metadata is not None else ""
        request = engine_pb2.InsertRequest(
            id=vector_id, 
            vector=vector, 
            metadata_json=meta_json
        )
        
        try:
            response = await self._call_with_retry(stub.InsertVector, request)
            return response.success
        except Exception as e:
            logger.error(f"Failed to shard vector '{vector_id}' to primary worker {target_addr}: {e}")
            return False

    async def search(self, query: List[float], top_k: int = 5) -> List[SearchResult]:
        """Query shard partitions by load-balancing search reads across healthy replica nodes."""
        if not self.is_leader:
            raise PermissionError(f"Coordinator {self.coordinator_id} is in STANDBY mode and cannot perform queries.")

        active_primaries = list(set(self.hash_ring.ring.values()))
        if not active_primaries:
            raise ValueError("No active search worker nodes available to perform queries.")

        # Resolve read targets (load-balance reads to replicas when available)
        query_targets = []
        for p_addr in active_primaries:
            replicas = self.replicas_map.get(p_addr, [])
            active_replicas = [r for r in replicas if r in self.workers and self.workers[r]["is_active"]]
            if active_replicas:
                # Load balance reads across healthy replica nodes
                selected_replica = random.choice(active_replicas)
                query_targets.append(selected_replica)
            else:
                # Fallback to primary if replicas are offline
                if p_addr in self.workers and self.workers[p_addr]["is_active"]:
                    query_targets.append(p_addr)

        if not query_targets:
            raise ValueError("No healthy query targets (primaries or replicas) available.")

        request = engine_pb2.SearchRequest(vector=query, top_k=top_k)
        
        async def query_node(addr: str, stub) -> Optional[engine_pb2.SearchResponse]:
            try:
                return await self._call_with_retry(stub.SearchVectors, request)
            except Exception as e:
                logger.error(f"Failed query on node {addr}: {e}")
                return None

        coroutines = [
            query_node(addr, self.workers[addr]["stub"]) 
            for addr in query_targets
        ]
        
        responses = await asyncio.gather(*coroutines)
        
        merged_results = []
        for resp in responses:
            if resp is None:
                continue
            for r in resp.results:
                meta = json.loads(r.metadata_json) if r.metadata_json else {}
                merged_results.append(
                    SearchResult(
                        id=r.id,
                        score=r.score,
                        metadata=meta
                    )
                )

        merged_results.sort(key=lambda x: x.score, reverse=True)
        return merged_results[:top_k]

    async def clear_all_indexes(self) -> None:
        if not self.is_leader:
            raise PermissionError(f"Coordinator {self.coordinator_id} is in STANDBY mode.")

        active_primaries = list(set(self.hash_ring.ring.values()))
        if not active_primaries:
            return

        request = engine_pb2.ClearRequest()
        coroutines = [
            self._call_with_retry(self.workers[addr]["stub"].ClearIndex, request)
            for addr in active_primaries
        ]
        await asyncio.gather(*coroutines, return_exceptions=True)

    async def close_all_channels(self) -> None:
        self.stop_health_check_loop()
        if self._leader_election_task:
            self._leader_election_task.cancel()
            try:
                await self._leader_election_task
            except asyncio.CancelledError:
                pass
                
        # Release election lock
        if self.is_leader:
            try:
                self.registry.release_leader_lock(self.coordinator_id)
            except Exception:
                pass

        addrs = list(self.workers.keys())
        for addr in addrs:
            host, port = addr.split(":")
            await self.remove_worker(host, int(port))
