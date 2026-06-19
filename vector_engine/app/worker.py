import asyncio
import argparse
import json
import logging
import os
import grpc
import numpy as np
from typing import List, Dict, Any, Optional

from vector_engine.app.proto import engine_pb2, engine_pb2_grpc
from vector_engine.app.vector_store import VectorStore
from vector_engine.app.search import brute_force_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] Worker: %(message)s")
logger = logging.getLogger("worker")

class SearchWorkerServicer(engine_pb2_grpc.SearchWorkerServicer):
    """Implementation of the gRPC SearchWorker service with primary-replica replication."""

    def __init__(
        self, 
        port: int = 50051, 
        role: str = "primary", 
        primary_addr: str = "", 
        replica_addresses: Optional[List[str]] = None,
        host: Optional[str] = None
    ):
        self.port = port
        self.role = role
        self.primary_addr = primary_addr
        self.replica_addresses = replica_addresses or []
        self.host = host or os.getenv("WORKER_HOST", "localhost")
        self.store = VectorStore()
        
        # Set up WAL directory specific to the port
        self.storage_dir = f"vector_engine/data/worker_{port}"
        from vector_engine.app.storage.wal import WALManager
        self.wal = WALManager(self.storage_dir)
        # Replay WAL on startup to restore state
        self.wal.replay(self.store)

        # Dynamic worker client channels for replication
        self.replica_stubs: Dict[str, engine_pb2_grpc.SearchWorkerStub] = {}
        self.replica_channels: List[grpc.aio.Channel] = []
        
        for r_addr in self.replica_addresses:
            channel = grpc.aio.insecure_channel(r_addr)
            self.replica_channels.append(channel)
            self.replica_stubs[r_addr] = engine_pb2_grpc.SearchWorkerStub(channel)

        # Service Registry setup
        from vector_engine.app.distributed.service_discovery import EtcdServiceRegistry
        self.registry = EtcdServiceRegistry()
        self.worker_id = f"worker_{port}"
        
        # Register worker in discovery registry
        self.registry.register_worker(
            worker_id=self.worker_id,
            host=self.host,
            port=port,
            role=self.role,
            primary_addr=self.primary_addr,
            ttl=6
        )

        # Heartbeat loop configuration
        self.heartbeat_task = None
        try:
            loop = asyncio.get_running_loop()
            self.heartbeat_task = loop.create_task(self._heartbeat_loop())
        except RuntimeError:
            pass

    async def _heartbeat_loop(self) -> None:
        """Background loop to send heartbeats to registry."""
        while True:
            try:
                self.registry.heartbeat(self.worker_id)
            except Exception as e:
                logger.error(f"Heartbeat failed for {self.worker_id}: {e}")
            await asyncio.sleep(2.0)

    async def _replicate_insert(self, addr: str, stub: engine_pb2_grpc.SearchWorkerStub, request: engine_pb2.InsertRequest) -> None:
        try:
            resp = await stub.InsertVector(request, timeout=2.0)
            if not resp.success:
                logger.warning(f"Replication failed on replica {addr}: {resp.message}")
        except Exception as e:
            logger.error(f"Failed to replicate insert to replica {addr}: {e}")

    async def _replicate_clear(self, addr: str, stub: engine_pb2_grpc.SearchWorkerStub, request: engine_pb2.ClearRequest) -> None:
        try:
            resp = await stub.ClearIndex(request, timeout=2.0)
            if not resp.success:
                logger.warning(f"Replication failed on replica {addr}: {resp.message}")
        except Exception as e:
            logger.error(f"Failed to replicate clear to replica {addr}: {e}")

    async def InsertVector(
        self, 
        request: engine_pb2.InsertRequest, 
        context: grpc.aio.ServicerContext
    ) -> engine_pb2.InsertResponse:
        logger.info(f"Received InsertVector request [Role: {self.role}]: ID={request.id}")
        try:
            vector_np = np.array(request.vector, dtype=np.float32)
            metadata = json.loads(request.metadata_json) if request.metadata_json else {}
            
            # Validation checks before committing to WAL
            if request.id in self.store._ids_set:
                raise ValueError(f"Vector ID '{request.id}' already exists in the store.")
            
            if self.store.dimension is not None and len(vector_np) != self.store.dimension:
                raise ValueError(
                    f"Dimension mismatch. Expected {self.store.dimension}, got {len(vector_np)}."
                )

            # 1. Write mutation to WAL first (durability constraint)
            self.wal.append_insert(request.id, request.vector, metadata)
            
            # 2. Update memory store
            self.store.add_vector(
                vector_id=request.id, 
                vector=vector_np, 
                metadata=metadata
            )

            # 3. Propagate to Replicas asynchronously if primary
            if self.role == "primary" and self.replica_stubs:
                for addr, stub in self.replica_stubs.items():
                    asyncio.create_task(self._replicate_insert(addr, stub, request))

            return engine_pb2.InsertResponse(success=True, message="Success")
        except ValueError as e:
            logger.warning(f"Validation failure during insert: {e}")
            return engine_pb2.InsertResponse(success=False, message=str(e))
        except Exception as e:
            logger.error(f"Internal error during insert: {e}")
            return engine_pb2.InsertResponse(success=False, message=str(e))

    async def SearchVectors(
        self, 
        request: engine_pb2.SearchRequest, 
        context: grpc.aio.ServicerContext
    ) -> engine_pb2.SearchResponse:
        logger.info(f"Received SearchVectors request: top_k={request.top_k}")
        try:
            if self.store.size == 0:
                return engine_pb2.SearchResponse(results=[])

            query_np = np.array(request.vector, dtype=np.float32)
            results = brute_force_search(
                vector_store=self.store, 
                query_vector=query_np, 
                top_k=request.top_k
            )
            
            pb_results = []
            for r in results:
                pb_results.append(
                    engine_pb2.SearchResult(
                        id=r.id,
                        score=r.score,
                        metadata_json=json.dumps(r.metadata)
                    )
                )
            return engine_pb2.SearchResponse(results=pb_results)
        except Exception as e:
            logger.error(f"Internal error during search: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return engine_pb2.SearchResponse()

    async def HealthCheck(
        self, 
        request: engine_pb2.HealthRequest, 
        context: grpc.aio.ServicerContext
    ) -> engine_pb2.HealthResponse:
        return engine_pb2.HealthResponse(
            status="healthy",
            size=self.store.size
        )

    async def ClearIndex(
        self, 
        request: engine_pb2.ClearRequest, 
        context: grpc.aio.ServicerContext
    ) -> engine_pb2.ClearResponse:
        logger.info("Clearing local index shard...")
        # Write mutation to WAL first (durability constraint)
        self.wal.append_clear()
        
        # Reset local store variables
        self.store.dimension = None
        self.store._vectors_list = []
        self.store._ids_list = []
        self.store._ids_set = set()
        self.store._metadata_list = []
        self.store._vectors_matrix = None

        # Propagate Clear to Replicas if primary
        if self.role == "primary" and self.replica_stubs:
            for addr, stub in self.replica_stubs.items():
                asyncio.create_task(self._replicate_clear(addr, stub, request))

        return engine_pb2.ClearResponse(success=True)

    async def cleanup(self) -> None:
        """Perform resource cleanup on shutdown."""
        logger.info(f"Cleaning up worker {self.worker_id}...")
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        try:
            self.registry.deregister_worker(self.worker_id)
        except Exception as e:
            logger.error(f"Deregister failed: {e}")
            
        for channel in self.replica_channels:
            await channel.close()

async def serve(port: int, role: str = "primary", primary_addr: str = "", replicas: str = "", host: Optional[str] = None) -> None:
    server = grpc.aio.server()
    replica_list = [addr.strip() for addr in replicas.split(",") if addr.strip()] if replicas else []
    servicer = SearchWorkerServicer(
        port=port, 
        role=role, 
        primary_addr=primary_addr, 
        replica_addresses=replica_list,
        host=host
    )
    engine_pb2_grpc.add_SearchWorkerServicer_to_server(servicer, server)
    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)
    logger.info(f"Starting async gRPC worker server on port {port} [Role: {role}]...")
    await server.start()
    try:
        await server.wait_for_termination()
    finally:
        await servicer.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Vector Engine - Search Worker")
    parser.add_argument("--port", type=int, default=50051, help="gRPC Server Listen Port")
    parser.add_argument("--role", type=str, default="primary", choices=["primary", "replica"], help="Replication Role")
    parser.add_argument("--primary-addr", type=str, default="", help="Host:port of primary if replica")
    parser.add_argument("--replicas", type=str, default="", help="Comma-separated host:port list of replicas if primary")
    parser.add_argument("--host", type=str, default=None, help="Hostname/IP registered in etcd discovery")
    args = parser.parse_args()
    
    try:
        asyncio.run(serve(args.port, args.role, args.primary_addr, args.replicas, args.host))
    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
