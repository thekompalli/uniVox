"""
FastAPI Main Application
Entry point for PS-06 Competition System
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.config.app_config import app_config, logging_config
from src.api.controllers.process_controller import ProcessController
from src.api.controllers.health_controller import HealthController
from src.api.middleware import (
    RequestLoggingMiddleware,
    RateLimitMiddleware,
    SecurityMiddleware
)
from src.api.dependencies import get_orchestrator_service
from src.services.orchestrator_service import OrchestratorService

# Configure logging
logging.config.dictConfig(logging_config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info(f"Starting {app_config.app_name} v{app_config.version}")
    
    # Initialize services
    try:
        orchestrator = OrchestratorService()
        await orchestrator.initialize()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down application")
    try:
        await orchestrator.cleanup()
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


# Create FastAPI application
app = FastAPI(
    title=app_config.app_name,
    version=app_config.version,
    description="Language Agnostic Speaker Identification & Diarization System for PS-06 Competition",
    docs_url="/api/docs" if app_config.debug else None,
    redoc_url="/api/redoc" if app_config.debug else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if app_config.debug else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(SecurityMiddleware)

# Global exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP_EXCEPTION",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred",
            "detail": str(exc) if app_config.debug else None
        }
    )

# Initialize controllers
process_controller = ProcessController()
health_controller = HealthController()

# Include routers
app.include_router(
    process_controller.router,
    prefix=app_config.api_prefix,
    tags=["processing"]
)

app.include_router(
    health_controller.router,
    prefix=app_config.api_prefix,
    tags=["health"]
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": app_config.app_name,
        "version": app_config.version,
        "status": "healthy",
        "docs_url": "/api/docs" if app_config.debug else None
    }

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "name": app_config.app_name,
        "version": app_config.version,
        "api_version": "v1",
        "endpoints": {
            "process": f"{app_config.api_prefix}/process",
            "status": f"{app_config.api_prefix}/status/{{job_id}}",
            "result": f"{app_config.api_prefix}/result/{{job_id}}",
            "health": f"{app_config.api_prefix}/health"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=app_config.api_host,
        port=app_config.api_port,
        reload=app_config.debug,
        log_config=logging_config.LOGGING_CONFIG
    )