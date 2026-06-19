import os
import json
import logging
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import redis.asyncio as aioredis
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import psutil
import grpc

from vector_engine.app.vector_store import VectorStore
from vector_engine.app.search import brute_force_search
from vector_engine.app.proto import engine_pb2, engine_pb2_grpc

# --- Custom Structured JSON Logging ---
class StructuredFormatter(logging.Formatter):
    """Formats log records as JSON objects for structured logging."""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

# Configure logging to use stdout and the StructuredFormatter
log_handler = logging.StreamHandler()
log_handler.setFormatter(StructuredFormatter())

root_logger = logging.getLogger()
# Clear existing handlers
for h in list(root_logger.handlers):
    root_logger.removeHandler(h)
root_logger.addHandler(log_handler)
root_logger.setLevel(logging.INFO)

logger = logging.getLogger("vector_engine_api")

# --- Prometheus Metrics definitions ---
REQUEST_COUNT = Counter(
    "vector_search_requests_total",
    "Total requests to the vector search service",
    ["endpoint"]
)
SEARCH_LATENCY = Histogram(
    "vector_search_latency_seconds",
    "Time spent performing searches"
)
CACHE_HITS = Counter(
    "vector_search_cache_hits_total",
    "Total search cache hits"
)
CACHE_MISSES = Counter(
    "vector_search_cache_misses_total",
    "Total search cache misses"
)
MEMORY_USAGE = Gauge(
    "vector_search_memory_bytes",
    "Process resident memory usage (RSS) in bytes"
)

# Initialize FastAPI App
app = FastAPI(
    title="Distributed Vector Search Engine",
    description="Deployment gateway supporting REST and gRPC routing.",
    version="1.0.0"
)

# Global in-memory VectorStore (fallback local mode)
store = VectorStore()

# Global Coordinator (distributed mode)
coordinator: Optional[Any] = None

# Global Redis Client
redis_client: Optional[aioredis.Redis] = None

# Global gRPC Server
grpc_server: Optional[grpc.aio.Server] = None


class CoordinatorGRPCServicer(engine_pb2_grpc.SearchWorkerServicer):
    """Bridges Coordinator distributed logic to external gRPC clients."""
    def __init__(self, coordinator_instance):
        self.coord = coordinator_instance

    async def InsertVector(self, request: engine_pb2.InsertRequest, context: grpc.aio.ServicerContext) -> engine_pb2.InsertResponse:
        logger.info(f"gRPC Insert Request: {request.id}")
        try:
            success = await self.coord.insert_vector(
                vector_id=request.id,
                vector=list(request.vector),
                metadata=json.loads(request.metadata_json) if request.metadata_json else None
            )
            return engine_pb2.InsertResponse(success=success, message="Success" if success else "Failed sharding")
        except PermissionError as e:
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details(str(e))
            return engine_pb2.InsertResponse(success=False, message=str(e))
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return engine_pb2.InsertResponse(success=False, message=str(e))

    async def SearchVectors(self, request: engine_pb2.SearchRequest, context: grpc.aio.ServicerContext) -> engine_pb2.SearchResponse:
        logger.info(f"gRPC Search Request: top_k={request.top_k}")
        try:
            results = await self.coord.search(
                query=list(request.vector),
                top_k=request.top_k
            )
            pb_results = [
                engine_pb2.SearchResult(
                    id=r.id,
                    score=r.score,
                    metadata_json=json.dumps(r.metadata)
                )
                for r in results
            ]
            return engine_pb2.SearchResponse(results=pb_results)
        except PermissionError as e:
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details(str(e))
            return engine_pb2.SearchResponse()
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return engine_pb2.SearchResponse()

    async def HealthCheck(self, request: engine_pb2.HealthRequest, context: grpc.aio.ServicerContext) -> engine_pb2.HealthResponse:
        return engine_pb2.HealthResponse(
            status="healthy" if self.coord.is_leader else "standby",
            size=len(self.coord.workers)
        )

    async def ClearIndex(self, request: engine_pb2.ClearRequest, context: grpc.aio.ServicerContext) -> engine_pb2.ClearResponse:
        try:
            await self.coord.clear_all_indexes()
            return engine_pb2.ClearResponse(success=True)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return engine_pb2.ClearResponse(success=False)


# --- Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    global redis_client, coordinator, grpc_server
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        redis_client = aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        # Test connection
        await redis_client.ping()
        logger.info(f"Connected to Redis at {redis_url}")
    except Exception as e:
        logger.warning(f"Could not connect to Redis at {redis_url}: {e}. Caching is disabled.")
        redis_client = None

    if os.getenv("MODE") == "coordinator":
        coordinator_id = os.getenv("COORDINATOR_ID", "coordinator_1")
        standby = os.getenv("STANDBY", "false").lower() == "true"
        from vector_engine.app.coordinator import Coordinator
        coordinator = Coordinator(coordinator_id=coordinator_id, standby=standby)
        coordinator.start()
        logger.info(f"Coordinator started in mode coordinator (id={coordinator_id}, standby={standby})")

        # Launch Coordinator gRPC Server
        grpc_server = grpc.aio.server()
        servicer = CoordinatorGRPCServicer(coordinator)
        engine_pb2_grpc.add_SearchWorkerServicer_to_server(servicer, grpc_server)
        grpc_server.add_insecure_port("[::]:50050")
        await grpc_server.start()
        logger.info("Coordinator gRPC Server running on port 50050")

@app.on_event("shutdown")
async def shutdown_event():
    global redis_client, coordinator, grpc_server
    if redis_client is not None:
        await redis_client.close()
        logger.info("Closed Redis connection.")
    if coordinator is not None:
        await coordinator.close_all_channels()
        logger.info("Coordinator service channels closed.")
    if grpc_server is not None:
        await grpc_server.stop(grace=1.0)
        logger.info("Coordinator gRPC server stopped.")

# --- Pydantic Schemas ---

class InsertRequest(BaseModel):
    id: str = Field(..., min_length=1, description="Unique string identifier for the vector")
    vector: List[float] = Field(..., min_length=1, description="1-D list of floats")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")

class InsertResponse(BaseModel):
    status: str
    id: str

class SearchRequest(BaseModel):
    vector: List[float] = Field(..., min_length=1, description="1-D query vector")
    top_k: int = Field(default=5, gt=0, description="Max nearest neighbors to return")

class SearchMatch(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[SearchMatch]

class HealthResponse(BaseModel):
    status: str
    size: int
    dimension: Optional[int]

# --- Exception Handlers ---

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    logger.warning(f"Domain Validation Error: Path: {request.url.path} | Detail: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )

@app.exception_handler(PermissionError)
async def permission_error_handler(request: Request, exc: PermissionError) -> JSONResponse:
    logger.warning(f"Standby Failover Permission Check Blocked: Path: {request.url.path} | Detail: {exc}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled Exception: Path: {request.url.path}", exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal Server Error"}
    )

# --- Endpoints ---

@app.post("/insert", response_model=InsertResponse, status_code=status.HTTP_201_CREATED)
async def insert_vector(request_data: InsertRequest) -> InsertResponse:
    REQUEST_COUNT.labels(endpoint="/insert").inc()
    logger.info(f"Inserting vector ID: {request_data.id}")
    
    if coordinator is not None:
        success = await coordinator.insert_vector(
            vector_id=request_data.id,
            vector=request_data.vector,
            metadata=request_data.metadata
        )
        if not success:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to insert sharded vector")
    else:
        vector_np = np.array(request_data.vector, dtype=np.float32)
        store.add_vector(
            vector_id=request_data.id, 
            vector=vector_np, 
            metadata=request_data.metadata
        )
    
    logger.info(f"Inserted vector ID: {request_data.id}")
    return InsertResponse(status="success", id=request_data.id)

@app.post("/search", response_model=SearchResponse)
async def search_vectors(request_data: SearchRequest) -> SearchResponse:
    REQUEST_COUNT.labels(endpoint="/search").inc()
    
    # 1. Compute Cache Key using query params hash
    query_str = f"{request_data.vector}_{request_data.top_k}"
    cache_key = f"search:{hashlib.sha256(query_str.encode('utf-8')).hexdigest()}"
    
    # 2. Query Cache
    cached_val = None
    if redis_client is not None:
        try:
            cached_val = await redis_client.get(cache_key)
        except Exception as e:
            logger.warning(f"Redis cache look up error: {e}")

    if cached_val is not None:
        CACHE_HITS.inc()
        logger.info(f"Cache HIT for key: {cache_key}")
        matches = [SearchMatch(**item) for item in json.loads(cached_val)]
        return SearchResponse(results=matches)

    # 3. Cache Miss - Execute search
    CACHE_MISSES.inc()
    logger.info(f"Cache MISS for key: {cache_key}")

    with SEARCH_LATENCY.time():
        if coordinator is not None:
            results = await coordinator.search(
                query=request_data.vector,
                top_k=request_data.top_k
            )
            matches = [
                SearchMatch(id=r.id, score=r.score, metadata=r.metadata)
                for r in results
            ]
        else:
            query_np = np.array(request_data.vector, dtype=np.float32)
            search_results = brute_force_search(
                vector_store=store, 
                query_vector=query_np, 
                top_k=request_data.top_k
            )
            matches = [
                SearchMatch(
                    id=res.id, 
                    score=res.score, 
                    metadata=res.metadata
                )
                for res in search_results
            ]

    # 4. Save to Cache with 60 seconds TTL
    if redis_client is not None:
        try:
            serialized_matches = json.dumps([m.dict() for m in matches])
            await redis_client.setex(cache_key, 60, serialized_matches)
        except Exception as e:
            logger.warning(f"Failed to populate Redis cache: {e}")

    return SearchResponse(results=matches)

@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    REQUEST_COUNT.labels(endpoint="/health").inc()
    if coordinator is not None:
        return HealthResponse(
            status="healthy" if coordinator.is_leader else "standby",
            size=len(coordinator.workers),
            dimension=None
        )
    return HealthResponse(
        status="healthy",
        size=store.size,
        dimension=store.dimension
    )

@app.get("/metrics", include_in_schema=False)
async def metrics():
    # Update memory statistics dynamically
    process = psutil.Process()
    MEMORY_USAGE.set(process.memory_info().rss)
    
    return Response(
        generate_latest(), 
        media_type=CONTENT_TYPE_LATEST
    )

