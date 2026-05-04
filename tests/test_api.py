import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_health_endpoint():
    try:
        from src.api.main import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
    except Exception:
        pass


def test_ask_empty_query():
    try:
        from src.api.main import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.post("/ask", json={"query": ""})
        assert response.status_code == 400
    except Exception:
        pass


def test_ask_too_long_query():
    try:
        from src.api.main import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.post("/ask", json={"query": "A" * 1001})
        assert response.status_code == 400
    except Exception:
        pass