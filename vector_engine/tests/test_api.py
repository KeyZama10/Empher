import pytest
from fastapi.testclient import TestClient

from vector_engine.app.api import app, store

# Initialize FastAPI test client
client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_store():
    """Reset the global vector store state before each test run for test isolation."""
    store.dimension = None
    store._vectors_list = []
    store._ids_list = []
    store._ids_set = set()
    store._metadata_list = []
    store._vectors_matrix = None
    yield

def test_health_endpoint_empty() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["size"] == 0
    assert data["dimension"] is None

def test_insert_vector_success() -> None:
    payload = {
        "id": "doc_1",
        "vector": [1.0, 2.0, 3.0],
        "metadata": {"source": "test_suite"}
    }
    # 1. Insert vector
    response = client.post("/insert", json=payload)
    assert response.status_code == 201
    assert response.json() == {"status": "success", "id": "doc_1"}
    
    # 2. Check health statistics are updated
    health_response = client.get("/health")
    assert health_response.status_code == 200
    health_data = health_response.json()
    assert health_data["size"] == 1
    assert health_data["dimension"] == 3

def test_insert_vector_duplicate_id() -> None:
    payload = {
        "id": "doc_1",
        "vector": [1.0, 2.0, 3.0]
    }
    # First insert is successful
    res1 = client.post("/insert", json=payload)
    assert res1.status_code == 201
    
    # Second insert with duplicate ID fails with HTTP 400
    res2 = client.post("/insert", json=payload)
    assert res2.status_code == 400
    assert "already exists" in res2.json()["detail"]

def test_insert_vector_dimension_mismatch() -> None:
    # First insert sets dimension to 3
    client.post("/insert", json={"id": "v1", "vector": [1.0, 2.0, 3.0]})
    
    # Second insert has 2 dimensions, should fail with HTTP 400
    payload = {
        "id": "v2",
        "vector": [4.0, 5.0]
    }
    response = client.post("/insert", json=payload)
    assert response.status_code == 400
    assert "Dimension mismatch" in response.json()["detail"]

def test_pydantic_validation_insert() -> None:
    # Missing required 'vector' field
    payload = {"id": "v1"}
    response = client.post("/insert", json=payload)
    assert response.status_code == 422
    
    # Empty string ID is invalid under validation (min_length=1)
    payload_empty_id = {"id": "", "vector": [1.0, 2.0]}
    response = client.post("/insert", json=payload_empty_id)
    assert response.status_code == 422

def test_search_endpoint_success() -> None:
    # Populate vectors
    client.post("/insert", json={"id": "v1", "vector": [1.0, 0.0, 0.0], "metadata": {"name": "x"}})
    client.post("/insert", json={"id": "v2", "vector": [0.0, 1.0, 0.0], "metadata": {"name": "y"}})
    client.post("/insert", json={"id": "v3", "vector": [1.0, 1.0, 0.0], "metadata": {"name": "diag"}})
    
    search_payload = {
        "vector": [1.0, 0.1, 0.0],
        "top_k": 2
    }
    
    response = client.post("/search", json=search_payload)
    assert response.status_code == 200
    results = response.json()["results"]
    
    assert len(results) == 2
    # Verify ranked sorting (v1 closest, then v3)
    assert results[0]["id"] == "v1"
    assert results[1]["id"] == "v3"
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["metadata"] == {"name": "x"}

def test_search_endpoint_empty_db() -> None:
    # Attempting to search an empty DB should trigger ValueError mapped to HTTP 400
    search_payload = {
        "vector": [1.0, 0.0, 0.0],
        "top_k": 5
    }
    response = client.post("/search", json=search_payload)
    assert response.status_code == 400
    assert "empty VectorStore" in response.json()["detail"]

def test_search_endpoint_dimension_mismatch() -> None:
    client.post("/insert", json={"id": "v1", "vector": [1.0, 0.0, 0.0]})
    
    # Query has dimension 2, store dimension is 3
    search_payload = {
        "vector": [1.0, 0.0],
        "top_k": 1
    }
    response = client.post("/search", json=search_payload)
    assert response.status_code == 400
    assert "does not match" in response.json()["detail"]

def test_search_validation_errors() -> None:
    # Invalid top_k <= 0
    search_payload = {
        "vector": [1.0, 0.0, 0.0],
        "top_k": 0
    }
    response = client.post("/search", json=search_payload)
    assert response.status_code == 422
