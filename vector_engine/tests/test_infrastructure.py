import json
import logging
import pytest
from fastapi.testclient import TestClient

from vector_engine.app.api import app, StructuredFormatter, store

# Initialize test client
client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_store():
    """Reset the global vector store state before each test run."""
    store.dimension = None
    store._vectors_list = []
    store._ids_list = []
    store._ids_set = set()
    store._metadata_list = []
    store._vectors_matrix = None
    yield

def test_prometheus_metrics_endpoint() -> None:
    # 1. Trigger health check to increment metrics
    client.get("/health")
    
    # 2. Get metrics page
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    
    # Verify metric lines are present in the Prometheus output
    metrics_text = response.text
    assert "vector_search_requests_total" in metrics_text
    assert "vector_search_memory_bytes" in metrics_text
    assert "# TYPE vector_search_requests_total counter" in metrics_text

def test_structured_logging_formatter() -> None:
    # Instantiate formatter
    formatter = StructuredFormatter()
    
    # Create a mock LogRecord
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_path.py",
        lineno=10,
        msg="Structured logging message test",
        args=(),
        exc_info=None
    )
    
    # Format and verify JSON
    formatted_log = formatter.format(record)
    log_json = json.loads(formatted_log)
    
    assert "timestamp" in log_json
    assert log_json["level"] == "INFO"
    assert log_json["logger"] == "test_logger"
    assert log_json["message"] == "Structured logging message test"

def test_redis_cache_bypass_failsafe() -> None:
    # Set the global api.redis_client to None to mock Redis down/unavailable
    import vector_engine.app.api as api
    original_client = api.redis_client
    api.redis_client = None
    
    try:
        # Populate store
        client.post("/insert", json={"id": "v1", "vector": [1.0, 0.0]})
        
        # Query: since Redis is None, it should bypass cache and proceed with brute force search
        response = client.post("/search", json={"vector": [1.0, 0.0], "top_k": 1})
        
        # Search should execute successfully
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 1
        assert results[0]["id"] == "v1"
        
    finally:
        # Restore client
        api.redis_client = original_client
