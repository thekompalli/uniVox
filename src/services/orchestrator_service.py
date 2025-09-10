"""
Orchestrator Service
Main business logic coordinator for PS-06 system
"""
import logging
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path

from src.api.schemas.process_schemas import (
    ProcessRequest, JobStatus, ProcessResult, JobState, AudioSpecs,
    BatchProcessRequest, BatchProcessResult
)
from src.repositories.job_repository import JobRepository
from src.repositories.audio_repository import AudioRepository
from src.repositories.result_repository import ResultRepository
from src.tasks.audio_processing_tasks import process_audio_pipeline
from src.config.app_config import app_config
from src.utils.audio_utils import AudioUtils
from src.utils.error_handler import ErrorHandler

logger = logging.getLogger(__name__)


class OrchestratorService:
    """Main orchestrator for audio processing pipeline"""
    
    def __init__(self):
        self.job_repo = JobRepository()
        self.audio_repo = AudioRepository()
        self.result_repo = ResultRepository()
        self.audio_utils = AudioUtils()
        self.error_handler = ErrorHandler()
        self._initialized = False
    
    async def initialize(self):
        """Initialize the orchestrator service"""
        try:
            logger.info("Initializing orchestrator service")
            
            # Initialize repositories
            await self.job_repo.initialize()
            await self.audio_repo.initialize()
            await self.result_repo.initialize()
            
            # Start cleanup task
            asyncio.create_task(self._cleanup_task())
            
            self._initialized = True
            logger.info("Orchestrator service initialized successfully")
            
        except Exception as e:
            logger.exception(f"Failed to initialize orchestrator service: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            logger.info("Cleaning up orchestrator service")
            
            await self.job_repo.cleanup()
            await self.audio_repo.cleanup()
            await self.result_repo.cleanup()
            
            logger.info("Orchestrator service cleanup completed")
            
        except Exception as e:
            logger.exception(f"Error during orchestrator cleanup: {e}")
    
    async def create_job(self, audio_file, request: ProcessRequest) -> JobStatus:
        """
        Create a new processing job
        
        Args:
            audio_file: Uploaded audio file
            request: Processing request parameters
            
        Returns:
            Job status with job_id
        """
        try:
            # Generate job ID
            job_id = str(uuid.uuid4())
            
            # Validate and save audio file
            audio_specs = await self._validate_and_save_audio(job_id, audio_file)
            
            # Create job record
            job_data = {
                "job_id": job_id,
                "status": JobState.QUEUED,
                "progress": 0.0,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "audio_specs": audio_specs.dict(),
                "request_params": request.dict()
            }
            
            await self.job_repo.create_job(job_id, request)
            
            # Queue processing task
            task = process_audio_pipeline.delay(job_id, request.dict())
            
            logger.info(f"Created job {job_id} with task {task.id}")
            
            return JobStatus(
                job_id=job_id,
                status=JobState.QUEUED,
                progress=0.0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                estimated_completion=datetime.utcnow() + timedelta(
                    seconds=self._estimate_processing_time(audio_specs.duration)
                )
            )
            
        except Exception as e:
            logger.exception(f"Error creating job: {e}")
            await self.error_handler.handle_error(e, context={"job_creation": True})
            raise
    
    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """
        Get job processing status
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status or None if not found
        """
        try:
            job_status = await self.job_repo.get_job(job_id)
            if not job_status:
                return None
            
            # job_repo.get_job already returns a JobStatus object
            return job_status
            
        except Exception as e:
            logger.exception(f"Error getting job status: {e}")
            return None
    
    async def get_result(self, job_id: str) -> Optional[ProcessResult]:
        """
        Get processing results
        
        Args:
            job_id: Job identifier
            
        Returns:
            Processing result or None if not found/incomplete
        """
        try:
            # Check job status
            job_status = await self.job_repo.get_job(job_id)
            if not job_status or job_status.status != JobState.COMPLETED:
                return None
            
            # Get result data from database
            result_data = await self.result_repo.get_result(job_id)
            if result_data:
                payload = result_data['result_data']
                try:
                    normalized = await self._normalize_result_payload(job_id, payload)
                    return ProcessResult(**normalized)
                except Exception as norm_err:
                    logger.warning(
                        f"Result normalization failed for {job_id} (DB payload): {norm_err}. "
                        f"Falling back to raw payload."
                    )
                    return ProcessResult(**payload)
            
            # Fallback: Load from file system (use configured data_dir)
            logger.info(f"Result not found in database for job {job_id}, trying file system fallback")
            from pathlib import Path
            import json
            
            # Use app-configured data directory to be robust to CWD
            result_file = app_config.data_dir / "results" / job_id / "final_result.json"
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                
                logger.info(f"Loaded result from file system for job {job_id}")
                normalized = await self._normalize_result_payload(job_id, file_data)
                return ProcessResult(**normalized)
            
            return None
            
        except Exception as e:
            logger.exception(f"Error getting result: {e}")
            return None

    async def _normalize_result_payload(self, job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure payload conforms to ProcessResult schema (fills audio_specs if incomplete)."""
        try:
            # Fast path: if audio_specs appears complete, return as-is
            audio_specs = payload.get('audio_specs', {}) or {}
            required_keys = {'sample_rate', 'channels', 'duration', 'format', 'bit_depth', 'file_size'}
            if isinstance(audio_specs, dict) and required_keys.issubset(audio_specs.keys()):
                return payload

            # Try to reconstruct audio_specs from preprocessing artifacts
            from pathlib import Path
            import json as _json

            prep_path = app_config.data_dir / 'audio' / job_id / 'preprocessing_results.json'
            processed_path = None
            if prep_path.exists():
                try:
                    with open(prep_path, 'r', encoding='utf-8') as f:
                        prep = _json.load(f)
                    processed_path = prep.get('processed_path') or prep.get('original_path')
                except Exception:
                    processed_path = None

            # If we have an audio file path, compute specs
            if processed_path:
                try:
                    specs = await self.audio_utils.get_audio_specs(Path(processed_path))
                    payload['audio_specs'] = specs.dict()
                    return payload
                except Exception:
                    pass

            # Last resort: fill reasonable defaults if duration present
            duration = audio_specs.get('duration') or 0.0
            payload['audio_specs'] = {
                'sample_rate': app_config.target_sample_rate,
                'channels': app_config.target_channels,
                'duration': duration,
                'format': 'wav',
                'bit_depth': 16,
                'file_size': 0,
            }
            return payload
        except Exception as e:
            # On any normalization error, return original payload
            logger.warning(f"Failed to normalize result payload for {job_id}: {e}")
            return payload
    
    async def get_result_file_path(self, job_id: str, file_type: str) -> Optional[Path]:
        """
        Get path to specific result file
        
        Args:
            job_id: Job identifier
            file_type: Type of file (sid_csv, sd_csv, etc.)
            
        Returns:
            File path or None if not found
        """
        try:
            result = await self.get_result(job_id)
            if not result:
                return None
            
            file_path = getattr(result, file_type, None)
            if file_path and Path(file_path).exists():
                return Path(file_path)
            
            return None
            
        except Exception as e:
            logger.exception(f"Error getting result file path: {e}")
            return None
    
    async def create_batch_job(self, request: BatchProcessRequest) -> BatchProcessResult:
        """
        Create batch processing job
        
        Args:
            request: Batch processing request
            
        Returns:
            Batch processing result
        """
        try:
            batch_id = str(uuid.uuid4())
            results = []
            
            # Process each file
            for file_path in request.files:
                try:
                    # Create individual job (simplified for batch)
                    job_status = await self._create_batch_file_job(
                        file_path, request.common_settings
                    )
                    results.append(job_status)
                    
                except Exception as e:
                    logger.exception(f"Error processing file {file_path}: {e}")
                    continue
            
            return BatchProcessResult(
                batch_id=batch_id,
                total_files=len(request.files),
                completed=0,
                failed=0,
                results=[],
                batch_metrics={}
            )
            
        except Exception as e:
            logger.exception(f"Error creating batch job: {e}")
            raise
    
    async def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get batch processing status"""
        # Implementation would track batch jobs in database
        # For now, return placeholder
        return {
            "batch_id": batch_id,
            "status": "processing",
            "total_files": 0,
            "completed": 0,
            "failed": 0
        }
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a processing job
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if cancelled successfully
        """
        try:
            job_data = await self.job_repo.get_job(job_id)
            if not job_data:
                return False
            
            # Can only cancel queued or processing jobs
            if job_data["status"] in [JobState.COMPLETED.value, JobState.FAILED.value]:
                return False
            
            # Note: Task cancellation not implemented as task_id is not stored in database
            # Jobs will continue processing but results will be marked as cancelled
            
            # Update job status
            await self.job_repo.update_job(job_id, {
                "status": JobState.FAILED.value,
                "error_msg": "Job cancelled by user",
                "updated_at": datetime.utcnow()
            })
            
            logger.info(f"Cancelled job {job_id}")
            return True
            
        except Exception as e:
            logger.exception(f"Error cancelling job: {e}")
            return False
    
    async def list_jobs(
        self, 
        limit: int = 50, 
        offset: int = 0, 
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List processing jobs with pagination
        
        Args:
            limit: Maximum jobs to return
            offset: Pagination offset
            status: Filter by status
            
        Returns:
            Paginated job list
        """
        try:
            jobs = await self.job_repo.list_jobs(
                limit=limit,
                offset=offset,
                status=status
            )
            
            return {
                "jobs": jobs,
                "total": len(jobs),  # Would be actual total from database
                "limit": limit,
                "offset": offset
            }
            
        except Exception as e:
            logger.exception(f"Error listing jobs: {e}")
            return {"jobs": [], "total": 0, "limit": limit, "offset": offset}
    
    async def _validate_and_save_audio(self, job_id: str, audio_file) -> AudioSpecs:
        """Validate and save uploaded audio file"""
        try:
            # Read file content
            content = await audio_file.read()
            
            # Validate file format
            if not any(audio_file.filename.lower().endswith(f'.{fmt}') 
                      for fmt in app_config.supported_formats):
                raise ValueError(f"Unsupported format. Supported: {app_config.supported_formats}")
            
            # Save to storage
            file_path = await self.audio_repo.save_audio_file(job_id, content, audio_file.filename)
            
            # Extract audio specifications
            audio_specs = await self.audio_utils.get_audio_specs(file_path)
            
            logger.info(f"Saved audio file for job {job_id}: {audio_specs.format}, "
                       f"{audio_specs.duration:.2f}s, {audio_specs.sample_rate}Hz")
            
            return audio_specs
            
        except Exception as e:
            logger.exception(f"Error validating and saving audio: {e}")
            raise
    
    def _estimate_processing_time(self, duration: float) -> int:
        """Estimate processing time based on audio duration"""
        # Rough estimate: 2x real-time for balanced mode
        base_time = duration * 2
        
        # Add overhead for different stages
        overhead = 30  # seconds
        
        return int(base_time + overhead)
    
    async def _create_batch_file_job(
        self, 
        file_path: str, 
        settings: ProcessRequest
    ) -> JobStatus:
        """Create job for single file in batch"""
        # Simplified implementation for batch processing
        # Would need actual file handling logic
        job_id = str(uuid.uuid4())
        
        return JobStatus(
            job_id=job_id,
            status=JobState.QUEUED,
            progress=0.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    async def _cleanup_task(self):
        """Background task to cleanup old jobs and files"""
        while True:
            try:
                await asyncio.sleep(app_config.cleanup_interval)
                
                # Clean up old completed/failed jobs
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                await self.job_repo.cleanup_old_jobs(cutoff_time)
                
                # Clean up temporary files
                await self.audio_repo.cleanup_temp_files(cutoff_time)
                
                logger.info("Completed cleanup task")
                
            except Exception as e:
                logger.exception(f"Error in cleanup task: {e}")
                continue
