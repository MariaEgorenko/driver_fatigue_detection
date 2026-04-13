import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

@pytest.fixture(scope="module")
def client():
    """
    Создаем TestClient.
    Использование `with` гарантирует запуск lifespan (загрузку ML-моделей и БД).
    """
    with TestClient(app) as c:
        yield c

def test_health_ok(client: TestClient):
    """Test that health endpoint returns 200 with status ok."""
    response = client.get("/health")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Detail: {response.text}"
    
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["db_connected"] is True

def test_metrics_endpoint(client: TestClient):
    """
    Бонусный тест! 
    Раз уж мы тестируем инфраструктурные эндпоинты, давайте 
    заодно проверим, что Prometheus метрики отдаются корректно.
    """
    response = client.get("/metrics")
    
    assert response.status_code == 200
    assert "fatigue_api_requests_total" in response.text
    assert "text/plain" in response.headers["content-type"]