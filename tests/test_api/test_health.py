"""
Tests for Health API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.api.main import app


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
def mock_health_checker():
    """Mock health checker"""
    with patch('src.api.dependencies.get_health_checker') as mock:
        mock_checker = AsyncMock()
        mock_checker.check_database_health.return_value = True
        mock_checker.check_triton_health.return_value = True
        mock_checker.check_storage_health.return_value = True
        mock_checker.check_celery_health.return_value = True
        mock.return_value = mock_checker
        yield mock_checker


class TestBasicHealth:
    """Test basic health endpoint"""
    
    def test_health_endpoint_success(self, client):
        """Test basic health endpoint returns success"""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["status"] == "healthy"
        assert "timestamp" in data["data"]
        assert "version" in data["data"]
    
    def test_health_endpoint_format(self, client):
        """Test health endpoint response format"""
        response = client.get("/api/v1/health")
        data = response.json()
        
        # Check response structure
        assert "success" in data
        assert "data" in data
        assert "timestamp" in data
        
        # Check health data structure
        health_data = data["data"]
        assert "status" in health_data
        assert "timestamp" in health_data
        assert "uptime" in health_data
        assert "version" in health_data


class TestDetailedHealth:
    """Test detailed health endpoint"""
    
    def test_detailed_health_success(self, client, mock_health_checker):
        """Test detailed health endpoint with all services healthy"""
        response = client.get("/api/v1/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        health_data = data["data"]
        assert health_data["status"] == "healthy"
        assert "components" in health_data
        assert "system_info" in health_data
    
    def test_detailed_health_component_failure(self, client):
        """Test detailed health with component failure"""
        with patch('src.api.dependencies.get_health_checker') as mock_get_checker:
            mock_checker = AsyncMock()
            mock_checker.check_database_health.return_value = False
            mock_checker.check_triton_health.return_value = True
            mock_checker.check_storage_health.return_value = True
            mock_checker.check_celery_health.return_value = True
            mock_get_checker.return_value = mock_checker
            
            response = client.get("/api/v1/health/detailed")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            health_data = data["data"]
            assert health_data["status"] == "unhealthy"
            assert "components" in health_data
    
    def test_detailed_health_components_structure(self, client, mock_health_checker):
        """Test detailed health components structure"""
        response = client.get("/api/v1/health/detailed")
        data = response.json()
        
        components = data["data"]["components"]
        expected_components = ["database", "triton", "storage", "celery"]
        
        for component in expected_components:
            assert component in components
            assert "healthy" in components[component]


class TestReadinessProbe:
    """Test Kubernetes readiness probe"""
    
    def test_readiness_success(self, client, mock_health_checker):
        """Test readiness probe when services are ready"""
        response = client.get("/api/v1/health/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
    
    def test_readiness_failure(self, client):
        """Test readiness probe when services are not ready"""
        with patch('src.api.dependencies.get_health_checker') as mock_get_checker:
            mock_checker = AsyncMock()
            mock_checker.check_database_health.return_value = False
            mock_checker.check_storage_health.return_value = False
            mock_get_checker.return_value = mock_checker
            
            response = client.get("/api/v1/health/ready")
            
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "not ready"


class TestLivenessProbe:
    """Test Kubernetes liveness probe"""
    
    def test_liveness_success(self, client):
        """Test liveness probe success"""
        response = client.get("/api/v1/health/live")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert "timestamp" in data


class TestMetricsEndpoint:
    """Test metrics endpoint"""
    
    def test_metrics_success(self, client):
        """Test metrics endpoint"""
        with patch('src.api.dependencies.get_metrics_collector') as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.get_metrics.return_value = {
                "requests_total": 100,
                "requests_failed": 5,
                "response_time_avg": 0.1
            }
            mock_get_collector.return_value = mock_collector
            
            response = client.get("/api/v1/metrics")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert "data" in data
            metrics = data["data"]
            assert "requests_total" in metrics
            assert "requests_failed" in metrics
            assert "response_time_avg" in metrics
    
    def test_metrics_format(self, client):
        """Test metrics response format"""
        with patch('src.api.dependencies.get_metrics_collector') as mock_get_collector:
            mock_collector = AsyncMock()
            mock_collector.get_metrics.return_value = {}
            mock_get_collector.return_value = mock_collector
            
            response = client.get("/api/v1/metrics")
            data = response.json()
            
            # Check response structure
            assert "success" in data
            assert "data" in data
            assert "timestamp" in data


class TestVersionEndpoint:
    """Test version information endpoint"""
    
    def test_version_success(self, client):
        """Test version endpoint"""
        response = client.get("/api/v1/version")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        version_data = data["data"]
        assert "version" in version_data
        assert "build_date" in version_data
        assert "git_commit" in version_data
    
    def test_version_format(self, client):
        """Test version information format"""
        response = client.get("/api/v1/version")
        data = response.json()
        
        version_data = data["data"]
        expected_fields = ["version", "build_date", "git_commit", "python_version"]
        
        for field in expected_fields:
            assert field in version_data


class TestHealthErrorHandling:
    """Test health endpoint error handling"""
    
    def test_health_exception_handling(self, client):
        """Test health endpoint handles exceptions gracefully"""
        with patch('src.api.dependencies.get_health_checker') as mock_get_checker:
            mock_get_checker.side_effect = Exception("Database connection failed")
            
            response = client.get("/api/v1/health/detailed")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert "error" in data
    
    def test_metrics_exception_handling(self, client):
        """Test metrics endpoint handles exceptions gracefully"""
        with patch('src.api.dependencies.get_metrics_collector') as mock_get_collector:
            mock_get_collector.side_effect = Exception("Metrics collection failed")
            
            response = client.get("/api/v1/metrics")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert "error" in data


@pytest.mark.integration
class TestHealthIntegration:
    """Integration tests for health endpoints"""
    
    @pytest.mark.asyncio
    async def test_health_with_real_dependencies(self):
        """Test health endpoint with real dependencies (if available)"""
        # This test would run with actual database/redis connections
        # Skip if dependencies are not available
        pytest.skip("Integration test - requires real dependencies")
    
    @pytest.mark.slow
    def test_health_performance(self, client):
        """Test health endpoint performance"""
        import time
        
        start_time = time.time()
        response = client.get("/api/v1/health")
        end_time = time.time()
        
        assert response.status_code == 200
        # Health check should be fast (< 1 second)
        assert (end_time - start_time) < 1.0


@pytest.mark.parametrize("endpoint", [
    "/api/v1/health",
    "/api/v1/health/detailed",
    "/api/v1/health/ready",
    "/api/v1/health/live",
    "/api/v1/metrics",
    "/api/v1/version"
])
def test_health_endpoints_accessibility(client, endpoint):
    """Test that all health endpoints are accessible"""
    response = client.get(endpoint)
    # All health endpoints should return some response (not 404)
    assert response.status_code != 404


def test_health_endpoints_cors(client):
    """Test CORS headers on health endpoints"""
    response = client.options("/api/v1/health")
    
    # Should handle OPTIONS request for CORS
    assert response.status_code in [200, 204]


def test_health_content_type(client):
    """Test content type of health responses"""
    response = client.get("/api/v1/health")
    
    assert response.status_code == 200
    assert "application/json" in response.headers.get("content-type", "")