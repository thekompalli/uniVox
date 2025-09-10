"""
Celery Application Configuration
Async task processing for PS-06 system
"""
import logging
from celery import Celery
from kombu import Queue

from src.config.app_config import app_config

logger = logging.getLogger(__name__)

# Create Celery application
celery_app = Celery(
    'ps06_system',
    broker=app_config.celery_broker_url,
    backend=app_config.celery_result_backend,
    include=['src.tasks.audio_processing_tasks']
)

# Celery configuration
celery_app.conf.update(
    # Task serialization
    task_serializer=app_config.celery_task_serializer,
    result_serializer=app_config.celery_result_serializer,
    accept_content=['json'],
    
    # Task routing
    task_routes={
        'src.tasks.audio_processing_tasks.process_audio_pipeline': 'high_priority',
        'src.tasks.audio_processing_tasks.preprocess_audio': 'normal_priority',
        'src.tasks.audio_processing_tasks.diarize_speakers': 'gpu_queue',
        'src.tasks.audio_processing_tasks.transcribe_audio': 'gpu_queue',
        'src.tasks.audio_processing_tasks.translate_text': 'normal_priority',
    },
    
    # Queue definitions
    task_default_queue='normal_priority',
    task_queues=(
        Queue('high_priority', routing_key='high_priority'),
        Queue('normal_priority', routing_key='normal_priority'),
        Queue('low_priority', routing_key='low_priority'),
        Queue('gpu_queue', routing_key='gpu_queue'),
    ),
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=50,
    
    # Task execution
    task_time_limit=3600,  # 1 hour
    task_soft_time_limit=3300,  # 55 minutes
    
    # Result backend settings
    result_expires=86400,  # 24 hours
    result_backend_max_retries=10,
    
    # Retry configuration
    task_retry_jitter=True,
    task_retry_delay_max=300,  # 5 minutes
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Beat schedule (for periodic tasks)
    beat_schedule={
        'cleanup-old-jobs': {
            'task': 'src.tasks.audio_processing_tasks.cleanup_old_jobs',
            'schedule': 3600.0,  # Every hour
        },
        'system-health-check': {
            'task': 'src.tasks.audio_processing_tasks.system_health_check',
            'schedule': 300.0,  # Every 5 minutes
        },
    },
    timezone='UTC',
)

# Task failure handler
@celery_app.task(bind=True)
def task_failure_handler(self, task_id, error, exception_type):
    """Handle task failures"""
    logger.error(f"Task {task_id} failed: {exception_type}: {error}")
    # Could send notifications, update database, etc.

# Celery signals
from celery.signals import worker_ready, worker_shutdown, task_prerun, task_postrun, task_retry, task_failure

@worker_ready.connect
def worker_ready_handler(sender, **kwargs):
    """Worker ready signal handler"""
    logger.info(f"Celery worker ready: {sender}")

@worker_shutdown.connect  
def worker_shutdown_handler(sender, **kwargs):
    """Worker shutdown signal handler"""
    logger.info(f"Celery worker shutting down: {sender}")

@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Task prerun signal handler"""
    logger.info(f"Task {task.name} [{task_id}] starting")

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, 
                 retval=None, state=None, **kwds):
    """Task postrun signal handler"""
    logger.info(f"Task {task.name} [{task_id}] completed with state: {state}")

@task_retry.connect
def task_retry_handler(sender=None, task_id=None, reason=None, einfo=None, **kwargs):
    """Task retry signal handler"""
    logger.warning(f"Task {task_id} retrying: {reason}")

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, einfo=None, **kwargs):
    """Task failure signal handler"""
    logger.error(f"Task {task_id} failed: {exception}")

if __name__ == '__main__':
    celery_app.start()