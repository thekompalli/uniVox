"""
Health Controller
System health and monitoring endpoints
"""
import logging
import psutil
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from src.api.schemas.process_schemas import HealthResponse, APIResponse
from src.api.dependencies import (
    get_health_checker, get_metrics_collector, get_orchestrator_service,
    get_job_repository, get_audio_repository, get_result_repository
)
from src.api.middleware import get_middleware_metrics
from src.config.app_config import app_config
from src.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


class HealthController:
    """Health monitoring and system status controller"""
    
    def __init__(self):
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup health monitoring routes"""
        
        @self.router.get("/health", response_model=APIResponse)
        async def health_check():
            """
            Basic health check endpoint
            Returns overall system health status
            """
            try:
                health_status = await self._get_basic_health()
                
                return APIResponse(
                    success=True,
                    data=health_status
                )
                
            except Exception as e:
                logger.exception(f"Health check failed: {e}")
                return APIResponse(
                    success=False,
                    error={
                        "error": "HEALTH_CHECK_FAILED",
                        "message": "Health check encountered an error",
                        "detail": str(e)
                    }
                )
        
        @self.router.get("/health/detailed", response_model=APIResponse)
        async def detailed_health_check(health_checker=Depends(get_health_checker)):
            """
            Detailed health check with component status
            """
            try:
                health_status = await self._get_detailed_health(health_checker)
                
                return APIResponse(
                    success=True,
                    data=health_status
                )
                
            except Exception as e:
                logger.exception(f"Detailed health check failed: {e}")
                return APIResponse(
                    success=False,
                    error={
                        "error": "DETAILED_HEALTH_CHECK_FAILED",
                        "message": "Detailed health check failed",
                        "detail": str(e)
                    }
                )
        
        @self.router.get("/health/ready")
        async def readiness_check():
            """
            Kubernetes readiness probe endpoint
            """
            try:
                is_ready = await self._check_readiness()
                
                if is_ready:
                    return JSONResponse(
                        status_code=200,
                        content={"status": "ready"}
                    )
                else:
                    return JSONResponse(
                        status_code=503,
                        content={"status": "not ready"}
                    )
                    
            except Exception as e:
                logger.exception(f"Readiness check failed: {e}")
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "error",
                        "message": str(e)
                    }
                )
        
        @self.router.get("/health/live")
        async def liveness_check():
            """
            Kubernetes liveness probe endpoint
            """
            try:
                # Simple liveness check - just return OK if service is running
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "alive",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
            except Exception as e:
                logger.exception(f"Liveness check failed: {e}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "message": str(e)
                    }
                )
        
        @self.router.get("/metrics", response_model=APIResponse)
        async def get_metrics(metrics_collector=Depends(get_metrics_collector)):
            """
            Get system metrics
            """
            try:
                metrics = await self._collect_all_metrics(metrics_collector)
                
                return APIResponse(
                    success=True,
                    data=metrics
                )
                
            except Exception as e:
                logger.exception(f"Metrics collection failed: {e}")
                return APIResponse(
                    success=False,
                    error={
                        "error": "METRICS_COLLECTION_FAILED",
                        "message": "Failed to collect metrics",
                        "detail": str(e)
                    }
                )
        
        @self.router.get("/status", response_model=APIResponse)
        async def system_status(
            orchestrator=Depends(get_orchestrator_service),
            job_repo=Depends(get_job_repository)
        ):
            """
            Get comprehensive system status
            """
            try:
                status = await self._get_system_status(orchestrator, job_repo)
                
                return APIResponse(
                    success=True,
                    data=status
                )
                
            except Exception as e:
                logger.exception(f"System status failed: {e}")
                return APIResponse(
                    success=False,
                    error={
                        "error": "SYSTEM_STATUS_FAILED",
                        "message": "Failed to get system status",
                        "detail": str(e)
                    }
                )
        
        @self.router.get("/version", response_model=APIResponse)
        async def get_version():
            """Get application version information"""
            try:
                version_info = {
                    "version": app_config.version,
                    "name": app_config.app_name,
                    "environment": app_config.environment,
                    "build_time": getattr(app_config, 'build_time', None),
                    "git_commit": getattr(app_config, 'git_commit', None),
                    "python_version": self._get_python_version()
                }
                
                return APIResponse(
                    success=True,
                    data=version_info
                )
                
            except Exception as e:
                logger.exception(f"Version info failed: {e}")
                return APIResponse(
                    success=False,
                    error={
                        "error": "VERSION_INFO_FAILED",
                        "message": "Failed to get version information",
                        "detail": str(e)
                    }
                )
    
    async def _get_basic_health(self) -> Dict[str, Any]:
        """Get basic health status"""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": self._get_uptime(),
            "version": app_config.version
        }
    
    async def _get_detailed_health(self, health_checker) -> Dict[str, Any]:
        """Get detailed health status"""
        # Check all components
        checks = await asyncio.gather(
            health_checker.check_database_health(),
            health_checker.check_triton_health(),
            health_checker.check_storage_health(),
            health_checker.check_celery_health(),
            return_exceptions=True
        )
        
        component_status = {
            "database": self._format_check_result(checks[0]),
            "triton": self._format_check_result(checks[1]),
            "storage": self._format_check_result(checks[2]),
            "celery": self._format_check_result(checks[3])
        }
        
        # Overall health
        overall_healthy = all(
            status["healthy"] for status in component_status.values()
        )
        
        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": self._get_uptime(),
            "version": app_config.version,
            "components": component_status,
            "system_info": await self._get_system_info()
        }
    
    async def _check_readiness(self) -> bool:
        """Check if system is ready to serve requests"""
        try:
            # Check critical components
            from src.api.dependencies import get_health_checker
            health_checker = await get_health_checker()
            
            db_healthy = await health_checker.check_database_health()
            storage_healthy = await health_checker.check_storage_health()
            
            return db_healthy and storage_healthy
            
        except Exception as e:
            logger.exception(f"Readiness check error: {e}")
            return False
    
    async def _collect_all_metrics(self, metrics_collector) -> Dict[str, Any]:
        """Collect all system metrics"""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "application": metrics_collector.get_metrics(),
            "middleware": get_middleware_metrics(),
            "system": await self._get_system_metrics(),
            "celery": await self._get_celery_metrics()
        }
        
        return metrics
    
    async def _get_system_status(self, orchestrator, job_repo) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Get job statistics
            job_stats = await job_repo.get_job_statistics()
            
            # Get system resources
            system_resources = await self._get_system_metrics()
            
            # Get processing status
            processing_status = {
                "active_jobs": job_stats.get("queued", 0) + 
                              job_stats.get("preprocessing", 0) + 
                              job_stats.get("diarization", 0) + 
                              job_stats.get("transcription", 0) + 
                              job_stats.get("translation", 0),
                "completed_jobs_24h": job_stats.get("completed", 0),
                "failed_jobs_24h": job_stats.get("failed", 0),
                "total_jobs_24h": job_stats.get("total_jobs", 0)
            }
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": "operational",
                "job_statistics": job_stats,
                "processing_status": processing_status,
                "system_resources": system_resources,
                "uptime": self._get_uptime()
            }
            
        except Exception as e:
            logger.exception(f"Error getting system status: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": "error",
                "error": str(e)
            }
    
    async def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            import platform
            import os
            
            # Get disk usage for the current drive (Windows-compatible)
            disk_path = os.path.abspath(os.sep) if os.name != 'nt' else os.path.abspath(os.getcwd())[:3]
            
            return {
                "platform": platform.system(),
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "disk_total_gb": round(psutil.disk_usage(disk_path).total / (1024**3), 2),
                "python_version": self._get_python_version(),
                "hostname": self._get_hostname()
            }
            
        except Exception as e:
            logger.warning(f"Error getting system info: {e}")
            return {"error": str(e)}
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics (if available)
            try:
                network = psutil.net_io_counters()
                network_metrics = {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
            except:
                network_metrics = {"error": "Network metrics unavailable"}
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "per_core": cpu_per_core,
                    "count": psutil.cpu_count()
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                },
                "network": network_metrics
            }
            
        except Exception as e:
            logger.warning(f"Error getting system metrics: {e}")
            return {"error": str(e)}
    
    async def _get_celery_metrics(self) -> Dict[str, Any]:
        """Get Celery worker metrics"""
        try:
            inspect = celery_app.control.inspect()
            
            # Get active tasks
            active_tasks = inspect.active()
            active_count = sum(len(tasks) for tasks in (active_tasks or {}).values()) if active_tasks else 0
            
            # Get scheduled tasks
            scheduled_tasks = inspect.scheduled()
            scheduled_count = sum(len(tasks) for tasks in (scheduled_tasks or {}).values()) if scheduled_tasks else 0
            
            # Get worker stats
            stats = inspect.stats()
            worker_count = len(stats) if stats else 0
            
            return {
                "workers": worker_count,
                "active_tasks": active_count,
                "scheduled_tasks": scheduled_count,
                "worker_stats": stats or {}
            }
            
        except Exception as e:
            logger.warning(f"Error getting Celery metrics: {e}")
            return {"error": str(e)}
    
    def _format_check_result(self, result) -> Dict[str, Any]:
        """Format health check result"""
        if isinstance(result, Exception):
            return {
                "healthy": False,
                "error": str(result),
                "checked_at": datetime.utcnow().isoformat()
            }
        elif isinstance(result, bool):
            return {
                "healthy": result,
                "checked_at": datetime.utcnow().isoformat()
            }
        else:
            return {
                "healthy": False,
                "error": f"Unexpected result type: {type(result)}",
                "checked_at": datetime.utcnow().isoformat()
            }
    
    def _get_uptime(self) -> str:
        """Get application uptime"""
        try:
            # This would ideally track from application start
            # For now, use boot time as approximation
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            uptime_timedelta = timedelta(seconds=uptime_seconds)
            
            return str(uptime_timedelta)
            
        except Exception:
            return "unknown"
    
    def _get_python_version(self) -> str:
        """Get Python version"""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _get_hostname(self) -> str:
        """Get system hostname"""
        try:
            import socket
            return socket.gethostname()
        except Exception:
            return "unknown"