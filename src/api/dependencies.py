"""
API Dependencies
Dependency injection for FastAPI endpoints
"""
import logging
from functools import lru_cache
from typing import Generator

from src.services.orchestrator_service import OrchestratorService
from src.repositories.job_repository import JobRepository
from src.repositories.audio_repository import AudioRepository
from src.repositories.result_repository import ResultRepository
from src.models.triton_client import TritonClient
from src.config.app_config import app_config

logger = logging.getLogger(__name__)

# Global service instances (initialized once)
_orchestrator_service = None
_job_repository = None
_audio_repository = None
_result_repository = None
_triton_client = None


async def get_orchestrator_service() -> OrchestratorService:
    """Get orchestrator service instance"""
    global _orchestrator_service
    
    if _orchestrator_service is None:
        _orchestrator_service = OrchestratorService()
        await _orchestrator_service.initialize()
    
    return _orchestrator_service


async def get_job_repository() -> JobRepository:
    """Get job repository instance"""
    global _job_repository
    
    if _job_repository is None:
        _job_repository = JobRepository()
        await _job_repository.initialize()
    
    return _job_repository


async def get_audio_repository() -> AudioRepository:
    """Get audio repository instance"""
    global _audio_repository
    
    if _audio_repository is None:
        _audio_repository = AudioRepository()
        await _audio_repository.initialize()
    
    return _audio_repository


async def get_result_repository() -> ResultRepository:
    """Get result repository instance"""
    global _result_repository
    
    if _result_repository is None:
        _result_repository = ResultRepository()
        await _result_repository.initialize()
    
    return _result_repository


async def get_triton_client() -> TritonClient:
    """Get Triton client instance"""
    global _triton_client
    
    if _triton_client is None and app_config.use_triton:
        _triton_client = TritonClient()
        await _triton_client.initialize()
    
    return _triton_client


@lru_cache()
def get_settings():
    """Get application settings (cached)"""
    return app_config


async def cleanup_dependencies():
    """Cleanup all global dependencies"""
    global _orchestrator_service, _job_repository, _audio_repository, _result_repository, _triton_client
    
    try:
        if _orchestrator_service:
            await _orchestrator_service.cleanup()
            _orchestrator_service = None
        
        if _job_repository:
            await _job_repository.cleanup()
            _job_repository = None
        
        if _audio_repository:
            await _audio_repository.cleanup()
            _audio_repository = None
        
        if _result_repository:
            await _result_repository.cleanup()
            _result_repository = None
        
        if _triton_client:
            await _triton_client.cleanup()
            _triton_client = None
        
        logger.info("All dependencies cleaned up")
        
    except Exception as e:
        logger.exception(f"Error during dependency cleanup: {e}")


# Authentication dependencies (placeholder for future implementation)
async def get_current_user():
    """Get current authenticated user (placeholder)"""
    # TODO: Implement actual authentication
    return {"user_id": "anonymous", "permissions": ["read", "write"]}


def require_permission(permission: str):
    """Require specific permission (placeholder)"""
    def dependency(user=get_current_user):
        # TODO: Implement permission checking
        return user
    return dependency


# Rate limiting dependencies
class RateLimiter:
    """Simple rate limiter implementation"""
    
    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.requests = {}
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed"""
        import time
        
        now = time.time()
        
        # Clean old entries
        self.requests = {
            k: timestamps for k, timestamps in self.requests.items()
            if any(t > now - self.period for t in timestamps)
        }
        
        # Filter timestamps within period
        if key in self.requests:
            self.requests[key] = [
                t for t in self.requests[key]
                if t > now - self.period
            ]
        else:
            self.requests[key] = []
        
        # Check rate limit
        if len(self.requests[key]) >= self.calls:
            return False
        
        # Add current request
        self.requests[key].append(now)
        return True


# Global rate limiter instances
_api_rate_limiter = RateLimiter(calls=100, period=60)  # 100 calls per minute
_upload_rate_limiter = RateLimiter(calls=10, period=60)  # 10 uploads per minute


async def check_api_rate_limit(request):
    """Check API rate limit"""
    client_ip = request.client.host
    
    if not _api_rate_limiter.is_allowed(client_ip):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Too many requests."
        )
    
    return True


async def check_upload_rate_limit(request):
    """Check upload rate limit"""
    client_ip = request.client.host
    
    if not _upload_rate_limiter.is_allowed(client_ip):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=429,
            detail="Upload rate limit exceeded. Please wait before uploading again."
        )
    
    return True


# Database session dependency
async def get_db_session():
    """Get database session (placeholder)"""
    # TODO: Implement actual database session management
    # This would typically return a database session from connection pool
    pass


# Request context dependencies
class RequestContext:
    """Request context for tracking and logging"""
    
    def __init__(self, request_id: str, user_id: str = None, client_ip: str = None):
        self.request_id = request_id
        self.user_id = user_id
        self.client_ip = client_ip
        self.start_time = None
        self.metadata = {}
    
    def set_metadata(self, key: str, value: any):
        """Set metadata for the request"""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default=None):
        """Get metadata for the request"""
        return self.metadata.get(key, default)


async def get_request_context(request) -> RequestContext:
    """Get request context"""
    import uuid
    import time
    
    request_id = str(uuid.uuid4())
    client_ip = request.client.host
    
    context = RequestContext(
        request_id=request_id,
        client_ip=client_ip
    )
    context.start_time = time.time()
    
    return context


# Health check dependencies
class HealthChecker:
    """Health check utilities"""
    
    @staticmethod
    async def check_database_health() -> bool:
        """Check database connectivity"""
        try:
            job_repo = await get_job_repository()
            # Simple connectivity test
            return job_repo.pool is not None
        except Exception as e:
            logger.exception(f"Database health check failed: {e}")
            return False
    
    @staticmethod
    async def check_triton_health() -> bool:
        """Check Triton server connectivity"""
        try:
            if not app_config.use_triton:
                return True  # Not using Triton
            
            triton_client = await get_triton_client()
            return triton_client is not None and triton_client._initialized
        except Exception as e:
            logger.exception(f"Triton health check failed: {e}")
            return False
    
    @staticmethod
    async def check_storage_health() -> bool:
        """Check storage connectivity"""
        try:
            audio_repo = await get_audio_repository()
            # Check if storage paths are accessible
            return audio_repo.local_storage_path.exists()
        except Exception as e:
            logger.exception(f"Storage health check failed: {e}")
            return False
    
    @staticmethod
    async def check_celery_health() -> bool:
        """Check Celery worker connectivity"""
        try:
            import os
            from src.config.app_config import app_config
            
            # If Celery is disabled via environment variable or app config, return True
            use_celery = os.getenv('USE_CELERY', 'true').lower()
            if use_celery == 'false':
                logger.info("Celery is disabled, marking as healthy")
                return True
            
            # Check if the config indicates we should use Celery
            if hasattr(app_config, 'use_celery') and not app_config.use_celery:
                logger.info("Celery disabled in config, marking as healthy")
                return True
                
            from src.tasks.celery_app import celery_app
            
            # Check if workers are available
            inspect = celery_app.control.inspect()
            stats = inspect.stats()
            
            return stats is not None and len(stats) > 0
        except Exception as e:
            logger.warning(f"Celery health check failed: {e}")
            # If Celery is not configured or available, but USE_CELERY=false, still return True
            import os
            if os.getenv('USE_CELERY', 'true').lower() == 'false':
                return True
            return False


async def get_health_checker() -> HealthChecker:
    """Get health checker instance"""
    return HealthChecker()


# Validation dependencies
class RequestValidator:
    """Request validation utilities"""
    
    @staticmethod
    def validate_audio_file(file) -> bool:
        """Validate audio file"""
        if not file:
            return False
        
        # Check file size
        if file.size > app_config.max_file_size:
            return False
        
        # Check content type
        if not file.content_type.startswith('audio/'):
            return False
        
        # Check file extension
        allowed_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        file_extension = file.filename.lower().split('.')[-1]
        if f'.{file_extension}' not in allowed_extensions:
            return False
        
        return True
    
    @staticmethod
    def validate_languages(languages: list) -> bool:
        """Validate language list"""
        supported_languages = [
            'english', 'hindi', 'punjabi', 'bengali', 'nepali', 'dogri'
        ]
        
        if not languages:
            return False
        
        return all(lang.lower() in supported_languages for lang in languages)


async def get_request_validator() -> RequestValidator:
    """Get request validator instance"""
    return RequestValidator()


# Error handling dependencies
class ErrorHandler:
    """Error handling utilities"""
    
    @staticmethod
    async def handle_processing_error(job_id: str, error: Exception):
        """Handle processing error"""
        try:
            job_repo = await get_job_repository()
            await job_repo.update_job(job_id, {
                'status': 'FAILED',
                'error_msg': str(error)
            })
            
            logger.error(f"Processing error for job {job_id}: {error}")
        except Exception as e:
            logger.exception(f"Error handling processing error: {e}")
    
    @staticmethod
    async def handle_validation_error(error: Exception, context: dict = None):
        """Handle validation error"""
        logger.warning(f"Validation error: {error}, context: {context}")
    
    @staticmethod
    async def handle_system_error(error: Exception, component: str = "unknown"):
        """Handle system error"""
        logger.error(f"System error in {component}: {error}")


async def get_error_handler() -> ErrorHandler:
    """Get error handler instance"""
    return ErrorHandler()


# Cache dependencies
class CacheManager:
    """Simple cache manager"""
    
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
    
    def get(self, key: str, default=None):
        """Get value from cache"""
        return self._cache.get(key, default)
    
    def set(self, key: str, value, ttl: int = 300):
        """Set value in cache with TTL"""
        import time
        
        self._cache[key] = value
        self._timestamps[key] = time.time() + ttl
    
    def delete(self, key: str):
        """Delete key from cache"""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
    
    def cleanup(self):
        """Clean up expired entries"""
        import time
        
        now = time.time()
        expired_keys = [
            key for key, expire_time in self._timestamps.items()
            if expire_time < now
        ]
        
        for key in expired_keys:
            self.delete(key)


# Global cache instance
_cache_manager = CacheManager()


async def get_cache_manager() -> CacheManager:
    """Get cache manager instance"""
    return _cache_manager


# Metrics dependencies
class MetricsCollector:
    """Simple metrics collector"""
    
    def __init__(self):
        self.counters = {}
        self.timers = {}
        self.gauges = {}
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment counter metric"""
        self.counters[name] = self.counters.get(name, 0) + value
    
    def record_timer(self, name: str, value: float):
        """Record timer metric"""
        if name not in self.timers:
            self.timers[name] = []
        self.timers[name].append(value)
    
    def set_gauge(self, name: str, value: float):
        """Set gauge metric"""
        self.gauges[name] = value
    
    def get_metrics(self) -> dict:
        """Get all metrics"""
        import numpy as np
        
        metrics = {
            'counters': self.counters,
            'gauges': self.gauges,
            'timers': {}
        }
        
        # Calculate timer statistics
        for name, values in self.timers.items():
            if values:
                metrics['timers'][name] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'p50': np.percentile(values, 50),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                }
        
        return metrics


# Global metrics instance
_metrics_collector = MetricsCollector()


async def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector instance"""
    return _metrics_collector