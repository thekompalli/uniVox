"""
Process Controller
Main API endpoints for audio processing
"""
import logging
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
import aiofiles

from src.api.schemas.process_schemas import (
    ProcessRequest, JobStatus, ProcessResult, CompetitionOutput,
    BatchProcessRequest, BatchProcessResult, APIResponse
)
from src.services.orchestrator_service import OrchestratorService
from src.api.dependencies import get_orchestrator_service
from src.config.app_config import app_config
from src.utils.format_utils import FormatUtils

logger = logging.getLogger(__name__)


class ProcessController:
    """Main processing controller"""
    
    def __init__(self):
        self.router = APIRouter()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.router.post("/process", response_model=APIResponse)
        async def process_audio(
            audio_file: UploadFile = File(...),
            languages: str = Form("english,hindi,punjabi"),
            speaker_gallery: Optional[str] = Form(None),
            quality_mode: str = Form("balanced"),
            enable_overlaps: bool = Form(True),
            min_segment_duration: float = Form(0.5),
            translate: bool = Form(True),
            orchestrator: OrchestratorService = Depends(get_orchestrator_service)
        ):
            """
            Process audio file for speaker diarization, identification, and transcription
            
            Args:
                audio_file: Audio file to process
                languages: Comma-separated list of expected languages
                speaker_gallery: Optional comma-separated speaker IDs
                quality_mode: Processing quality (fast/balanced/high)
                enable_overlaps: Enable overlap detection
                min_segment_duration: Minimum segment duration in seconds
            
            Returns:
                Job status with job_id for tracking
            """
            try:
                # Validate file
                if not audio_file.content_type.startswith('audio/'):
                    raise HTTPException(400, "File must be an audio file")
                
                if audio_file.size > app_config.max_file_size:
                    raise HTTPException(400, f"File too large. Max size: {app_config.max_file_size} bytes")
                
                # Parse parameters
                language_list = [lang.strip() for lang in languages.split(',')]
                gallery_list = None
                if speaker_gallery:
                    gallery_list = [spk.strip() for spk in speaker_gallery.split(',')]
                
                # Create request
                request = ProcessRequest(
                    languages=language_list,
                    speaker_gallery=gallery_list,
                    quality_mode=quality_mode,
                    enable_overlaps=enable_overlaps,
                    min_segment_duration=min_segment_duration,
                    translate=translate
                )
                
                # Create job
                job_status = await orchestrator.create_job(
                    audio_file=audio_file,
                    request=request
                )
                
                logger.info(f"Created processing job: {job_status.job_id}")
                
                return APIResponse(
                    success=True,
                    data=job_status
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.exception(f"Error processing audio: {e}")
                raise HTTPException(500, f"Processing failed: {str(e)}")
        
        @self.router.get("/status/{job_id}", response_model=APIResponse)
        async def get_job_status(
            job_id: str,
            orchestrator: OrchestratorService = Depends(get_orchestrator_service)
        ):
            """
            Get job processing status
            
            Args:
                job_id: Job identifier
            
            Returns:
                Current job status and progress
            """
            try:
                status = await orchestrator.get_job_status(job_id)
                if not status:
                    raise HTTPException(404, f"Job {job_id} not found")
                
                return APIResponse(
                    success=True,
                    data=status
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.exception(f"Error getting job status: {e}")
                raise HTTPException(500, f"Status retrieval failed: {str(e)}")
        
        @self.router.get("/result/{job_id}", response_model=APIResponse)
        async def get_result(
            job_id: str,
            format: Optional[str] = "json",
            orchestrator: OrchestratorService = Depends(get_orchestrator_service)
        ):
            """
            Get processing results
            
            Args:
                job_id: Job identifier
                format: Result format (json/competition)
            
            Returns:
                Processing results in requested format
            """
            try:
                result = await orchestrator.get_result(job_id)
                if not result:
                    raise HTTPException(404, f"Results for job {job_id} not found")
                
                if format == "competition":
                    # Return competition format
                    competition_result = await self._format_for_competition(result)
                    return APIResponse(
                        success=True,
                        data=competition_result
                    )
                
                return APIResponse(
                    success=True,
                    data=result
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.exception(f"Error getting result: {e}")
                raise HTTPException(500, f"Result retrieval failed: {str(e)}")
        
        @self.router.get("/download/{job_id}/{file_type}")
        async def download_file(
            job_id: str,
            file_type: str,
            orchestrator: OrchestratorService = Depends(get_orchestrator_service)
        ):
            """
            Download specific result file
            
            Args:
                job_id: Job identifier
                file_type: File type (sid_csv, sd_csv, lid_csv, asr_trn, nmt_txt)
            
            Returns:
                File download response
            """
            try:
                file_path = await orchestrator.get_result_file_path(job_id, file_type)
                if not file_path or not file_path.exists():
                    raise HTTPException(404, f"File {file_type} not found for job {job_id}")
                
                return FileResponse(
                    path=file_path,
                    filename=f"{job_id}_{file_type}",
                    media_type='application/octet-stream'
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.exception(f"Error downloading file: {e}")
                raise HTTPException(500, f"File download failed: {str(e)}")
        
        @self.router.post("/batch", response_model=APIResponse)
        async def process_batch(
            request: BatchProcessRequest,
            orchestrator: OrchestratorService = Depends(get_orchestrator_service)
        ):
            """
            Process multiple audio files in batch
            
            Args:
                request: Batch processing request
            
            Returns:
                Batch job status
            """
            try:
                batch_result = await orchestrator.create_batch_job(request)
                
                logger.info(f"Created batch job: {batch_result.batch_id}")
                
                return APIResponse(
                    success=True,
                    data=batch_result
                )
                
            except Exception as e:
                logger.exception(f"Error processing batch: {e}")
                raise HTTPException(500, f"Batch processing failed: {str(e)}")
        
        @self.router.get("/batch/{batch_id}", response_model=APIResponse)
        async def get_batch_status(
            batch_id: str,
            orchestrator: OrchestratorService = Depends(get_orchestrator_service)
        ):
            """
            Get batch processing status
            
            Args:
                batch_id: Batch identifier
            
            Returns:
                Batch status and progress
            """
            try:
                status = await orchestrator.get_batch_status(batch_id)
                if not status:
                    raise HTTPException(404, f"Batch {batch_id} not found")
                
                return APIResponse(
                    success=True,
                    data=status
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.exception(f"Error getting batch status: {e}")
                raise HTTPException(500, f"Batch status retrieval failed: {str(e)}")

        @self.router.get("/steps/{job_id}", response_model=APIResponse)
        async def list_step_artifacts(job_id: str):
            """List available per-step artifact files for a job"""
            try:
                from pathlib import Path
                import json as _json
                base = app_config.data_dir / "audio" / job_id
                mapping = {
                    'preprocessing': base / 'preprocessing_results.json',
                    'diarization': base / 'diarization_results.json',
                    'speaker_identification': base / 'speaker_identification_results.json',
                    'language_identification': base / 'language_identification_results.json',
                    'asr': base / 'asr_results.json',
                    'translation': base / 'translation_results.json',
                }
                items = {
                    k: {
                        'exists': p.exists(),
                        'path': str(p) if p.exists() else None,
                        'size': (p.stat().st_size if p.exists() else 0)
                    }
                    for k, p in mapping.items()
                }
                return APIResponse(success=True, data={'job_id': job_id, 'artifacts': items})
            except Exception as e:
                logger.exception(f"Error listing step artifacts: {e}")
                raise HTTPException(500, f"Artifacts listing failed: {str(e)}")

        @self.router.get("/steps/{job_id}/{stage}", response_model=APIResponse)
        async def get_step_artifact(job_id: str, stage: str):
            """Fetch the JSON content of a per-step artifact"""
            try:
                from pathlib import Path
                import json as _json
                base = app_config.data_dir / "audio" / job_id
                filename_map = {
                    'preprocessing': 'preprocessing_results.json',
                    'diarization': 'diarization_results.json',
                    'speaker_identification': 'speaker_identification_results.json',
                    'language_identification': 'language_identification_results.json',
                    'asr': 'asr_results.json',
                    'translation': 'translation_results.json',
                }
                key = stage.lower()
                if key not in filename_map:
                    raise HTTPException(400, f"Unknown stage: {stage}")
                path = base / filename_map[key]
                if not path.exists():
                    raise HTTPException(404, f"Artifact not found for stage {stage}")
                async with aiofiles.open(path, 'r') as f:
                    content = await f.read()
                data = _json.loads(content)
                return APIResponse(success=True, data={'job_id': job_id, 'stage': key, 'artifact': data})
            except HTTPException:
                raise
            except Exception as e:
                logger.exception(f"Error reading step artifact: {e}")
                raise HTTPException(500, f"Artifact read failed: {str(e)}")
        
        @self.router.delete("/job/{job_id}")
        async def cancel_job(
            job_id: str,
            orchestrator: OrchestratorService = Depends(get_orchestrator_service)
        ):
            """
            Cancel a processing job
            
            Args:
                job_id: Job identifier
            
            Returns:
                Cancellation status
            """
            try:
                success = await orchestrator.cancel_job(job_id)
                
                return APIResponse(
                    success=success,
                    data={"message": f"Job {job_id} cancelled" if success else "Job could not be cancelled"}
                )
                
            except Exception as e:
                logger.exception(f"Error cancelling job: {e}")
                raise HTTPException(500, f"Job cancellation failed: {str(e)}")
        
        @self.router.get("/jobs")
        async def list_jobs(
            limit: int = 50,
            offset: int = 0,
            status: Optional[str] = None,
            orchestrator: OrchestratorService = Depends(get_orchestrator_service)
        ):
            """
            List processing jobs
            
            Args:
                limit: Maximum number of jobs to return
                offset: Offset for pagination
                status: Filter by job status
            
            Returns:
                List of jobs with pagination
            """
            try:
                jobs = await orchestrator.list_jobs(
                    limit=limit,
                    offset=offset,
                    status=status
                )
                
                return APIResponse(
                    success=True,
                    data=jobs
                )
                
            except Exception as e:
                logger.exception(f"Error listing jobs: {e}")
                raise HTTPException(500, f"Job listing failed: {str(e)}")
    
    async def _format_for_competition(self, result: ProcessResult) -> CompetitionOutput:
        """Format result for competition submission"""
        try:
            format_utils = FormatUtils()
            
            # Read competition format files
            sid_content = await self._read_file(result.sid_csv)
            sd_content = await self._read_file(result.sd_csv)
            lid_content = await self._read_file(result.lid_csv)
            asr_content = await self._read_file(result.asr_trn)
            nmt_content = await self._read_file(result.nmt_txt)
            
            # Generate solution hash
            solution_hash = format_utils.generate_solution_hash(result.job_id)
            
            return CompetitionOutput(
                evaluation_id="01",  # This would be provided by competition
                audio_files=[],  # List of processed files
                sid_results=sid_content,
                sd_results=sd_content,
                lid_results=lid_content,
                asr_results=asr_content,
                nmt_results=nmt_content,
                solution_hash=solution_hash,
                performance_summary=result.metrics
            )
            
        except Exception as e:
            logger.exception(f"Error formatting for competition: {e}")
            raise
    
    async def _read_file(self, file_path: str) -> str:
        """Read file content asynchronously"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                return await f.read()
        except Exception:
            return ""
