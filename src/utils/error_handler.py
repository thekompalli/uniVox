"""
Error Handler
Centralized error handling and recovery utilities
"""
import logging
import traceback
import asyncio
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from enum import Enum
import json

from src.config.app_config import app_config

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories"""
    SYSTEM = "system"
    VALIDATION = "validation"
    PROCESSING = "processing"
    EXTERNAL = "external"
    NETWORK = "network"
    STORAGE = "storage"
    MODEL = "model"
    USER = "user"


class ErrorHandler:
    """Centralized error handling system"""
    
    def __init__(self):
        self.error_counts = {}
        self.error_history = []
        self.max_history_size = 1000
        self.retry_strategies = {}
        self._setup_retry_strategies()
    
    def _setup_retry_strategies(self):
        """Setup default retry strategies for different error types"""
        self.retry_strategies = {
            # Network errors - exponential backoff
            ConnectionError: {
                'max_retries': 3,
                'base_delay': 1.0,
                'backoff_factor': 2.0,
                'max_delay': 30.0
            },
            TimeoutError: {
                'max_retries': 2,
                'base_delay': 5.0,
                'backoff_factor': 1.5,
                'max_delay': 60.0
            },
            # File system errors
            FileNotFoundError: {
                'max_retries': 2,
                'base_delay': 0.5,
                'backoff_factor': 1.0,
                'max_delay': 2.0
            },
            PermissionError: {
                'max_retries': 1,
                'base_delay': 1.0,
                'backoff_factor': 1.0,
                'max_delay': 1.0
            },
            # Model/processing errors
            RuntimeError: {
                'max_retries': 2,
                'base_delay': 2.0,
                'backoff_factor': 2.0,
                'max_delay': 10.0
            }
        }
    
    async def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        job_id: Optional[str] = None,
        retry_func: Optional[Callable] = None,
        retry_args: Optional[tuple] = None,
        retry_kwargs: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Handle error with logging, tracking, and optional retry
        
        Args:
            error: The exception that occurred
            context: Additional context information
            severity: Error severity level
            category: Error category
            job_id: Related job ID if applicable
            retry_func: Function to retry if applicable
            retry_args: Arguments for retry function
            retry_kwargs: Keyword arguments for retry function
            
        Returns:
            Error handling result
        """
        try:
            # Create error record
            error_record = self._create_error_record(
                error, context, severity, category, job_id
            )
            
            # Log error
            self._log_error(error_record)
            
            # Track error
            self._track_error(error_record)
            
            # Handle job-specific errors
            if job_id:
                await self._handle_job_error(job_id, error_record)
            
            # Attempt retry if specified
            retry_result = None
            if retry_func:
                retry_result = await self._attempt_retry(
                    error, retry_func, retry_args, retry_kwargs
                )
            
            # Send alerts if necessary
            await self._send_alerts(error_record)
            
            return {
                'error_id': error_record['id'],
                'handled': True,
                'retry_attempted': retry_result is not None,
                'retry_successful': retry_result.get('success', False) if retry_result else False,
                'recovery_actions': error_record.get('recovery_actions', [])
            }
            
        except Exception as handler_error:
            logger.exception(f"Error in error handler: {handler_error}")
            return {
                'error_id': None,
                'handled': False,
                'error': str(handler_error)
            }
    
    def _create_error_record(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]],
        severity: ErrorSeverity,
        category: ErrorCategory,
        job_id: Optional[str]
    ) -> Dict[str, Any]:
        """Create structured error record"""
        import uuid
        
        error_record = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'type': type(error).__name__,
            'message': str(error),
            'severity': severity.value,
            'category': category.value,
            'job_id': job_id,
            'context': context or {},
            'traceback': traceback.format_exc(),
            'recovery_actions': self._get_recovery_actions(error, category)
        }
        
        return error_record
    
    def _log_error(self, error_record: Dict[str, Any]):
        """Log error with appropriate level"""
        severity = error_record['severity']
        error_msg = f"[{error_record['id']}] {error_record['type']}: {error_record['message']}"
        
        if error_record['job_id']:
            error_msg += f" (Job: {error_record['job_id']})"
        
        if severity == ErrorSeverity.CRITICAL.value:
            logger.critical(error_msg, extra={'error_record': error_record})
        elif severity == ErrorSeverity.HIGH.value:
            logger.error(error_msg, extra={'error_record': error_record})
        elif severity == ErrorSeverity.MEDIUM.value:
            logger.warning(error_msg, extra={'error_record': error_record})
        else:
            logger.info(error_msg, extra={'error_record': error_record})
    
    def _track_error(self, error_record: Dict[str, Any]):
        """Track error for metrics and analysis"""
        error_type = error_record['type']
        
        # Update error counts
        if error_type not in self.error_counts:
            self.error_counts[error_type] = {
                'count': 0,
                'last_seen': None,
                'severity_counts': {}
            }
        
        self.error_counts[error_type]['count'] += 1
        self.error_counts[error_type]['last_seen'] = error_record['timestamp']
        
        severity = error_record['severity']
        if severity not in self.error_counts[error_type]['severity_counts']:
            self.error_counts[error_type]['severity_counts'][severity] = 0
        self.error_counts[error_type]['severity_counts'][severity] += 1
        
        # Add to history
        self.error_history.append(error_record)
        
        # Trim history if too long
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    async def _handle_job_error(self, job_id: str, error_record: Dict[str, Any]):
        """Handle job-specific error actions"""
        try:
            # Import here to avoid circular imports
            from src.api.dependencies import get_job_repository
            
            job_repo = await get_job_repository()
            
            # Update job status
            await job_repo.update_job(job_id, {
                'status': 'FAILED',
                'error_msg': error_record['message'],
                'updated_at': datetime.utcnow()
            })
            
            logger.info(f"Updated job {job_id} status to FAILED due to error {error_record['id']}")
            
        except Exception as e:
            logger.exception(f"Error updating job status: {e}")
    
    async def _attempt_retry(
        self,
        error: Exception,
        retry_func: Callable,
        retry_args: Optional[tuple] = None,
        retry_kwargs: Optional[dict] = None
    ) -> Dict[str, Any]:
        """Attempt to retry failed operation"""
        try:
            error_type = type(error)
            strategy = self.retry_strategies.get(error_type)
            
            if not strategy:
                logger.debug(f"No retry strategy for {error_type.__name__}")
                return {'success': False, 'reason': 'no_strategy'}
            
            max_retries = strategy['max_retries']
            base_delay = strategy['base_delay']
            backoff_factor = strategy['backoff_factor']
            max_delay = strategy['max_delay']
            
            retry_args = retry_args or ()
            retry_kwargs = retry_kwargs or {}
            
            for attempt in range(max_retries):
                # Calculate delay
                delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                
                logger.info(f"Retrying operation (attempt {attempt + 1}/{max_retries}) after {delay}s delay")
                
                # Wait before retry
                await asyncio.sleep(delay)
                
                try:
                    # Attempt retry
                    if asyncio.iscoroutinefunction(retry_func):
                        result = await retry_func(*retry_args, **retry_kwargs)
                    else:
                        result = retry_func(*retry_args, **retry_kwargs)
                    
                    logger.info(f"Retry successful after {attempt + 1} attempts")
                    return {
                        'success': True,
                        'attempts': attempt + 1,
                        'result': result
                    }
                    
                except Exception as retry_error:
                    logger.warning(f"Retry attempt {attempt + 1} failed: {retry_error}")
                    
                    # If this is the last attempt, don't continue
                    if attempt == max_retries - 1:
                        break
            
            logger.error(f"All {max_retries} retry attempts failed")
            return {
                'success': False,
                'attempts': max_retries,
                'reason': 'max_retries_exceeded'
            }
            
        except Exception as e:
            logger.exception(f"Error during retry attempt: {e}")
            return {
                'success': False,
                'attempts': 0,
                'reason': 'retry_error',
                'error': str(e)
            }
    
    def _get_recovery_actions(
        self,
        error: Exception,
        category: ErrorCategory
    ) -> List[str]:
        """Get suggested recovery actions for error"""
        error_type = type(error)
        actions = []
        
        # Generic recovery actions by error type
        if error_type == ConnectionError:
            actions.extend([
                "Check network connectivity",
                "Verify service endpoints are accessible",
                "Check firewall settings"
            ])
        elif error_type == TimeoutError:
            actions.extend([
                "Increase timeout settings",
                "Check system load and performance",
                "Verify external service availability"
            ])
        elif error_type == FileNotFoundError:
            actions.extend([
                "Verify file path exists",
                "Check file permissions",
                "Ensure storage is mounted"
            ])
        elif error_type == PermissionError:
            actions.extend([
                "Check file/directory permissions",
                "Verify user has required access",
                "Review security policies"
            ])
        elif error_type == MemoryError:
            actions.extend([
                "Check available memory",
                "Reduce batch sizes",
                "Optimize memory usage",
                "Scale up resources"
            ])
        
        # Category-specific actions
        if category == ErrorCategory.MODEL:
            actions.extend([
                "Check model files integrity",
                "Verify model configuration",
                "Ensure GPU memory availability"
            ])
        elif category == ErrorCategory.STORAGE:
            actions.extend([
                "Check disk space",
                "Verify storage service status",
                "Clean up temporary files"
            ])
        elif category == ErrorCategory.EXTERNAL:
            actions.extend([
                "Check external service status",
                "Verify API credentials",
                "Review rate limiting"
            ])
        
        return list(set(actions))  # Remove duplicates
    
    async def _send_alerts(self, error_record: Dict[str, Any]):
        """Send alerts for critical errors"""
        try:
            severity = error_record['severity']
            
            # Only alert on high/critical errors
            if severity not in [ErrorSeverity.HIGH.value, ErrorSeverity.CRITICAL.value]:
                return
            
            # Check if we should throttle alerts
            if self._should_throttle_alert(error_record):
                return
            
            # Send alert (placeholder - would integrate with actual alerting system)
            alert_data = {
                'title': f"PS-06 System Error: {error_record['type']}",
                'message': error_record['message'],
                'severity': severity,
                'timestamp': error_record['timestamp'],
                'job_id': error_record.get('job_id'),
                'recovery_actions': error_record['recovery_actions']
            }
            
            # Log alert (in real system, would send to Slack, email, PagerDuty, etc.)
            logger.critical(f"ALERT: {alert_data['title']} - {alert_data['message']}")
            
        except Exception as e:
            logger.exception(f"Error sending alert: {e}")
    
    def _should_throttle_alert(self, error_record: Dict[str, Any]) -> bool:
        """Check if alert should be throttled to avoid spam"""
        error_type = error_record['type']
        
        # Check recent errors of same type
        recent_errors = [
            e for e in self.error_history
            if e['type'] == error_type and
            datetime.fromisoformat(e['timestamp']) > datetime.utcnow() - timedelta(minutes=5)
        ]
        
        # Throttle if more than 3 errors of same type in last 5 minutes
        return len(recent_errors) > 3
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        try:
            total_errors = sum(data['count'] for data in self.error_counts.values())
            
            # Recent errors (last hour)
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            recent_errors = [
                e for e in self.error_history
                if datetime.fromisoformat(e['timestamp']) > cutoff_time
            ]
            
            # Severity distribution
            severity_counts = {}
            for error in self.error_history:
                severity = error['severity']
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Category distribution
            category_counts = {}
            for error in self.error_history:
                category = error['category']
                category_counts[category] = category_counts.get(category, 0) + 1
            
            return {
                'total_errors': total_errors,
                'unique_error_types': len(self.error_counts),
                'recent_errors_1h': len(recent_errors),
                'error_rate_1h': len(recent_errors) / 60.0,  # errors per minute
                'top_errors': sorted(
                    self.error_counts.items(),
                    key=lambda x: x[1]['count'],
                    reverse=True
                )[:10],
                'severity_distribution': severity_counts,
                'category_distribution': category_counts,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.exception(f"Error getting error statistics: {e}")
            return {'error': str(e)}
    
    def clear_error_history(self):
        """Clear error history (for testing/maintenance)"""
        self.error_history.clear()
        self.error_counts.clear()
        logger.info("Error history cleared")
    
    def export_error_log(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Export error log for analysis"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            return [
                error for error in self.error_history
                if datetime.fromisoformat(error['timestamp']) > cutoff_time
            ]
            
        except Exception as e:
            logger.exception(f"Error exporting error log: {e}")
            return []
    
    async def recover_failed_job(self, job_id: str) -> Dict[str, Any]:
        """Attempt to recover a failed job"""
        try:
            from src.api.dependencies import get_job_repository, get_orchestrator_service
            
            job_repo = await get_job_repository()
            orchestrator = await get_orchestrator_service()
            
            # Get job details
            job_data = await job_repo.get_job(job_id)
            if not job_data:
                return {'success': False, 'reason': 'job_not_found'}
            
            if job_data['status'] != 'FAILED':
                return {'success': False, 'reason': 'job_not_failed'}
            
            # Reset job status and retry
            await job_repo.update_job(job_id, {
                'status': 'QUEUED',
                'error_msg': None,
                'progress': 0.0,
                'updated_at': datetime.utcnow()
            })
            
            # Requeue processing
            from src.tasks.audio_processing_tasks import process_audio_pipeline
            request_params = job_data.get('request_params', {})
            
            task = process_audio_pipeline.delay(job_id, request_params)
            await job_repo.update_job(job_id, {"task_id": task.id})
            
            logger.info(f"Recovery initiated for job {job_id}")
            
            return {
                'success': True,
                'message': f'Job {job_id} recovery initiated',
                'new_task_id': task.id
            }
            
        except Exception as e:
            logger.exception(f"Error recovering job {job_id}: {e}")
            return {
                'success': False,
                'reason': 'recovery_error',
                'error': str(e)
            }


# Global error handler instance
error_handler = ErrorHandler()


# Convenience functions
async def handle_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to handle errors"""
    return await error_handler.handle_error(
        error, context, severity, category, job_id
    )


async def handle_job_error(job_id: str, error: Exception) -> Dict[str, Any]:
    """Convenience function to handle job-specific errors"""
    return await error_handler.handle_error(
        error,
        context={'job_id': job_id},
        severity=ErrorSeverity.HIGH,
        category=ErrorCategory.PROCESSING,
        job_id=job_id
    )


def get_error_statistics() -> Dict[str, Any]:
    """Get error statistics"""
    return error_handler.get_error_statistics()