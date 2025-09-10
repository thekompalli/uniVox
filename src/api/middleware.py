"""
API Middleware
Custom middleware for FastAPI application
"""
import logging
import time
import uuid
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.status import HTTP_429_TOO_MANY_REQUESTS, HTTP_500_INTERNAL_SERVER_ERROR

from src.config.app_config import app_config

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Start timing
        start_time = time.time()
        
        # Log incoming request
        logger.info(
            f"Request {request_id}: {request.method} {request.url.path} "
            f"from {request.client.host}"
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response {request_id}: {response.status_code} "
                f"({process_time:.3f}s)"
            )
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log error
            logger.exception(
                f"Error processing request {request_id}: {e} "
                f"({process_time:.3f}s)"
            )
            
            # Return error response
            return JSONResponse(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                    "request_id": request_id
                },
                headers={"X-Request-ID": request_id}
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.requests = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limit
        if not self._is_allowed(client_id):
            logger.warning(f"Rate limit exceeded for {client_id}")
            
            return JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate Limit Exceeded",
                    "message": f"Maximum {self.calls} requests per {self.period} seconds allowed",
                    "retry_after": self.period
                },
                headers={"Retry-After": str(self.period)}
            )
        
        return await call_next(request)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Try to get user ID from headers (if authenticated)
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return f"user:{user_id}"
        
        # Fall back to IP address
        return f"ip:{request.client.host}"
    
    def _is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed under rate limit"""
        now = time.time()
        
        # Clean old entries
        cutoff_time = now - self.period
        self.requests = {
            k: [t for t in timestamps if t > cutoff_time]
            for k, timestamps in self.requests.items()
        }
        
        # Remove empty entries
        self.requests = {k: v for k, v in self.requests.items() if v}
        
        # Check current client
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Check rate limit
        if len(self.requests[client_id]) >= self.calls:
            return False
        
        # Record this request
        self.requests[client_id].append(now)
        return True


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security headers and basic security middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        if app_config.environment == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting request metrics"""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.request_times = []
        self.error_count = 0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        self.request_count += 1
        
        try:
            response = await call_next(request)
            
            # Record metrics
            process_time = time.time() - start_time
            self.request_times.append(process_time)
            
            # Keep only recent times (last 1000 requests)
            if len(self.request_times) > 1000:
                self.request_times = self.request_times[-1000:]
            
            # Count errors
            if response.status_code >= 400:
                self.error_count += 1
            
            return response
            
        except Exception as e:
            self.error_count += 1
            raise
    
    def get_metrics(self) -> dict:
        """Get collected metrics"""
        import numpy as np
        
        if not self.request_times:
            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "avg_response_time": 0.0,
                "p95_response_time": 0.0
            }
        
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "avg_response_time": np.mean(self.request_times),
            "p95_response_time": np.percentile(self.request_times, 95),
            "p99_response_time": np.percentile(self.request_times, 99)
        }


class CORSMiddleware:
    """Custom CORS middleware for more control"""
    
    def __init__(
        self,
        allow_origins: list = None,
        allow_methods: list = None,
        allow_headers: list = None,
        allow_credentials: bool = False
    ):
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = allow_headers or ["*"]
        self.allow_credentials = allow_credentials
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
            response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
        else:
            response = await call_next(request)
        
        # Add CORS headers
        origin = request.headers.get("origin")
        if origin and (self.allow_origins == ["*"] or origin in self.allow_origins):
            response.headers["Access-Control-Allow-Origin"] = origin
        elif self.allow_origins == ["*"]:
            response.headers["Access-Control-Allow-Origin"] = "*"
        
        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to limit request body size"""
    
    def __init__(self, app, max_size: int = 100 * 1024 * 1024):  # 100MB default
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check Content-Length header
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_size:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "Payload Too Large",
                            "message": f"Request body size {size} exceeds maximum allowed size {self.max_size}",
                            "max_size": self.max_size
                        }
                    )
            except ValueError:
                pass
        
        return await call_next(request)


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to timeout long-running requests"""
    
    def __init__(self, app, timeout_seconds: int = 300):  # 5 minutes default
        super().__init__(app)
        self.timeout_seconds = timeout_seconds
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        import asyncio
        
        try:
            # Run request with timeout
            response = await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout_seconds
            )
            return response
            
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout after {self.timeout_seconds}s: {request.url.path}")
            
            return JSONResponse(
                status_code=408,
                content={
                    "error": "Request Timeout",
                    "message": f"Request took longer than {self.timeout_seconds} seconds",
                    "timeout": self.timeout_seconds
                }
            )


class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware for response compression"""
    
    def __init__(self, app, minimum_size: int = 1000):
        super().__init__(app)
        self.minimum_size = minimum_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Check if client accepts gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding.lower():
            return response
        
        # Check if response should be compressed
        if not self._should_compress(response):
            return response
        
        # Compress response body
        try:
            import gzip
            
            # Get response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            # Compress if size is above threshold
            if len(body) >= self.minimum_size:
                compressed_body = gzip.compress(body)
                
                # Create new response with compressed body
                response.headers["Content-Encoding"] = "gzip"
                response.headers["Content-Length"] = str(len(compressed_body))
                
                from starlette.responses import Response
                return Response(
                    content=compressed_body,
                    status_code=response.status_code,
                    headers=response.headers,
                    media_type=response.media_type
                )
            
            return response
            
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return response
    
    def _should_compress(self, response: Response) -> bool:
        """Check if response should be compressed"""
        # Don't compress already compressed content
        if response.headers.get("content-encoding"):
            return False
        
        # Only compress text-based content
        content_type = response.headers.get("content-type", "")
        compressible_types = [
            "application/json",
            "application/javascript",
            "application/xml",
            "text/",
            "application/x-javascript"
        ]
        
        return any(ct in content_type for ct in compressible_types)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
            
        except ValueError as e:
            logger.warning(f"Validation error: {e}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Bad Request",
                    "message": str(e),
                    "type": "validation_error"
                }
            )
        
        except FileNotFoundError as e:
            logger.warning(f"File not found: {e}")
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Not Found",
                    "message": "The requested resource was not found",
                    "type": "file_not_found"
                }
            )
        
        except PermissionError as e:
            logger.warning(f"Permission denied: {e}")
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Forbidden",
                    "message": "You don't have permission to access this resource",
                    "type": "permission_error"
                }
            )
        
        except Exception as e:
            logger.exception(f"Unhandled error: {e}")
            
            error_details = {
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "type": "internal_error"
            }
            
            # Include error details in debug mode
            if app_config.debug:
                error_details["details"] = str(e)
                error_details["traceback"] = self._get_traceback()
            
            return JSONResponse(
                status_code=500,
                content=error_details
            )
    
    def _get_traceback(self) -> str:
        """Get formatted traceback"""
        import traceback
        return traceback.format_exc()


# Global middleware instances (for metrics collection)
metrics_middleware = MetricsMiddleware(None)


def get_middleware_metrics() -> dict:
    """Get metrics from middleware"""
    return metrics_middleware.get_metrics()