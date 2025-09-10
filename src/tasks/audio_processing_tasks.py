"""
Audio Processing Tasks
Celery tasks for async audio processing pipeline
"""
import logging
import traceback
from datetime import datetime, timedelta
import asyncio
from typing import Dict, Any, List

from celery import chain, group
from src.tasks.celery_app import celery_app
from src.api.schemas.process_schemas import JobState, ProcessRequest
from src.services.audio_processing_service import AudioProcessingService
from pathlib import Path
import json
from src.services.diarization_service import DiarizationService
from src.services.speaker_service import SpeakerIdentificationService
from src.services.language_service import LanguageIdentificationService
from src.services.asr_service import ASRService
from src.services.translation_service import TranslationService
from src.services.format_service import FormatService
from src.repositories.celery_job_repository import get_celery_job_repo
from src.repositories.result_repository import ResultRepository
from src.utils.error_handler import ErrorHandler

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='process_audio_pipeline', ignore_result=True)
def process_audio_pipeline(self, job_id: str, request_params: Dict[str, Any]):
    """
    Main orchestration task for complete audio processing pipeline
    
    Args:
        job_id: Unique job identifier
        request_params: Processing request parameters
    
    Returns:
        Complete processing result
    """
    try:
        logger.info(f"Starting audio processing pipeline for job {job_id}")
        
        # Update job status
        _update_job_status(job_id, JobState.PREPROCESSING, 0.1, "Starting audio preprocessing")
        
        # Create processing chain (sequential to avoid blocking on group semantics)
        pipeline = chain(
            preprocess_audio.si(job_id, request_params),
            diarize_speakers.si(job_id),
            identify_languages.si(job_id, request_params),
            transcribe_audio.si(job_id, request_params),
            translate_text.si(job_id, request_params),
            generate_outputs.si(job_id, request_params)
        )
        
        # Execute pipeline without waiting for result (avoid result.get())
        result = pipeline.apply_async()
        
        # Return the task ID instead of blocking on result
        logger.info(f"Audio processing pipeline started for job {job_id}, task_id: {result.id}")
        return {"task_id": result.id, "status": "pipeline_started"}
        
    except Exception as exc:
        logger.exception(f"Error in audio processing pipeline for job {job_id}: {exc}")
        
        # Update job with error
        _update_job_status(
            job_id, 
            JobState.FAILED, 
            0.0, 
            f"Pipeline failed: {str(exc)}",
            error_msg=str(exc)
        )
        
        # Retry logic
        if self.request.retries < 3:
            logger.info(f"Retrying job {job_id} (attempt {self.request.retries + 1})")
            raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
        
        raise


@celery_app.task(bind=True, ignore_result=True)
def preprocess_audio(self, job_id: str, request_params: Dict[str, Any]):
    """
    Audio preprocessing task
    
    Args:
        job_id: Job identifier
        request_params: Processing parameters
    
    Returns:
        Preprocessing results
    """
    try:
        logger.info(f"Starting audio preprocessing for job {job_id}")
        
        _update_job_status(job_id, JobState.PREPROCESSING, 0.2, "Processing audio file")
        
        # Initialize service
        audio_service = AudioProcessingService()
        job_repo = get_celery_job_repo()
        
        # Get job data to find audio file path
        job_data = job_repo.get_job_sync(job_id)
        if not job_data:
            raise ValueError(f"Job {job_id} not found")
        
        # Find audio file in job directory (since audio_file_path is not stored in DB)
        from pathlib import Path
        audio_dir = Path("data/audio") / job_id
        
        if not audio_dir.exists():
            raise ValueError(f"Audio directory not found for job {job_id}: {audio_dir}")
        
        # Find the audio file (should start with "original_")
        audio_files = list(audio_dir.glob("original_*"))
        if not audio_files:
            raise ValueError(f"No audio file found in directory for job {job_id}: {audio_dir}")
        
        audio_file_path = str(audio_files[0])  # Use first match
        logger.info(f"Found audio file for job {job_id}: {audio_file_path}")
        
        # Process audio
        result = audio_service.process_audio_sync(job_id, audio_file_path)

        # Persist preprocessing results for downstream tasks to consume
        try:
            job_audio_dir = Path("data/audio") / job_id
            job_audio_dir.mkdir(parents=True, exist_ok=True)
            with open(job_audio_dir / "preprocessing_results.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"Saved preprocessing results for job {job_id}")
        except Exception as save_exc:
            logger.warning(f"Could not save preprocessing results for {job_id}: {save_exc}")
        
        _update_job_status(job_id, JobState.PREPROCESSING, 0.3, "Audio preprocessing completed")
        
        logger.info(f"Audio preprocessing completed for job {job_id}")
        return result
        
    except Exception as exc:
        logger.exception(f"Error in audio preprocessing for job {job_id}: {exc}")
        raise self.retry(exc=exc, countdown=60, max_retries=3)


@celery_app.task(bind=True, ignore_result=True)
def diarize_speakers(self, job_id: str):
    """
    Speaker diarization task
    
    Args:
        job_id: Job identifier
    
    Returns:
        Diarization results
    """
    try:
        logger.info(f"Starting speaker diarization for job {job_id}")
        
        _update_job_status(job_id, JobState.DIARIZATION, 0.4, "Performing speaker diarization")
        
        # Initialize services
        diarization_service = DiarizationService()
        speaker_service = SpeakerIdentificationService()
        
        # Load processed audio data
        processed_data = diarization_service.load_processed_data_sync(job_id)
        
        # Perform diarization
        diarization_result = diarization_service.diarize_audio_sync(
            processed_data['audio_data'],
            processed_data['sample_rate']
        )
        
        # Speaker identification
        identification_result = speaker_service.identify_speakers_sync(
            job_id, 
            diarization_result['segments']
        )
        
        # Combine results
        result = {
            'diarization': diarization_result,
            'identification': identification_result
        }

        # Persist results for downstream/debugging
        try:
            job_audio_dir = Path("data/audio") / job_id
            job_audio_dir.mkdir(parents=True, exist_ok=True)
            with open(job_audio_dir / "diarization_results.json", "w", encoding="utf-8") as f:
                json.dump(diarization_result, f, indent=2, default=str, ensure_ascii=False)
            with open(job_audio_dir / "speaker_identification_results.json", "w", encoding="utf-8") as f:
                json.dump(identification_result, f, indent=2, default=str, ensure_ascii=False)
            logger.info(f"Saved diarization and speaker ID results for job {job_id}")
        except Exception as save_exc:
            logger.warning(f"Could not save diarization/ID results for {job_id}: {save_exc}")
        
        _update_job_status(job_id, JobState.DIARIZATION, 0.5, "Speaker diarization completed")
        
        logger.info(f"Speaker diarization completed for job {job_id}")
        # Avoid returning large/non-serializable objects to Celery backend
        return None
        
    except Exception as exc:
        logger.exception(f"Error in speaker diarization for job {job_id}: {exc}")
        raise self.retry(exc=exc, countdown=60, max_retries=3)


@celery_app.task(bind=True, ignore_result=True)
def identify_languages(self, job_id: str, request_params: Dict[str, Any] = None):
    """
    Language identification task
    
    Args:
        job_id: Job identifier
    
    Returns:
        Language identification results
    """
    try:
        logger.info(f"Starting language identification for job {job_id}")
        
        _update_job_status(job_id, JobState.LANGUAGE_ID, 0.5, "Identifying languages")
        
        # Initialize service
        language_service = LanguageIdentificationService()
        
        # Load processed audio data
        processed_data = language_service.load_processed_data_sync(job_id)

        # Prefer diarization segments if available
        segments = processed_data.get('speech_segments', [])
        try:
            diar_path = Path("data/audio") / job_id / "diarization_results.json"
            if diar_path.exists():
                import json as _json
                with open(diar_path, 'r', encoding='utf-8') as _f:
                    diar_data = _json.load(_f)
                    if isinstance(diar_data, dict) and 'segments' in diar_data:
                        segments = diar_data['segments']
        except Exception:
            pass

        # Perform language identification
        expected_langs = None
        try:
            if request_params and isinstance(request_params, dict):
                expected_langs = request_params.get('languages')
        except Exception:
            expected_langs = None

        result = language_service.identify_languages_sync(
            job_id,
            processed_data['audio_data'],
            processed_data['sample_rate'],
            segments,
            expected_languages=expected_langs
        )

        # Persist language identification results
        try:
            job_audio_dir = Path("data/audio") / job_id
            job_audio_dir.mkdir(parents=True, exist_ok=True)
            with open(job_audio_dir / "language_identification_results.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str, ensure_ascii=False)
            logger.info(f"Saved language identification results for job {job_id}")
        except Exception as save_exc:
            logger.warning(f"Could not save language results for {job_id}: {save_exc}")
        
        _update_job_status(job_id, JobState.LANGUAGE_ID, 0.6, "Language identification completed")
        
        logger.info(f"Language identification completed for job {job_id}")
        return result
        
    except Exception as exc:
        logger.exception(f"Error in language identification for job {job_id}: {exc}")
        # Don't block the pipeline; continue with empty language results
        _update_job_status(job_id, JobState.LANGUAGE_ID, 0.6, "Language identification skipped due to error", str(exc))
        return {'segments': [], 'languages_detected': [], 'total_segments': 0}


@celery_app.task(bind=True, ignore_result=True)
def transcribe_audio(self, job_id: str, request_params: Dict[str, Any]):
    """
    Speech recognition task
    
    Args:
        job_id: Job identifier
        request_params: Processing parameters
    
    Returns:
        Transcription results
    """
    try:
        logger.info(f"Starting speech recognition for job {job_id}")
        
        # Breadcrumb 1: starting ASR and loading model
        _update_job_status(job_id, JobState.TRANSCRIPTION, 0.70, "Loading ASR model")
        
        # Initialize service
        asr_service = ASRService()
        _update_job_status(job_id, JobState.TRANSCRIPTION, 0.72, "ASR model ready")
        
        # Load processing results
        diarization_data = asr_service.load_diarization_results_sync(job_id)
        language_data = asr_service.load_language_results_sync(job_id)
        _update_job_status(job_id, JobState.TRANSCRIPTION, 0.74, "Preparing segments for transcription")
        
        # Perform transcription
        _update_job_status(job_id, JobState.TRANSCRIPTION, 0.75, "Transcribing segments")
        result = asr_service.transcribe_segments_sync(
            job_id,
            diarization_data.get('segments', []),
            language_data.get('segments', []),
            request_params.get('languages', ['english'])
        )
        
        _update_job_status(job_id, JobState.TRANSCRIPTION, 0.8, "Speech recognition completed")
        
        logger.info(f"Speech recognition completed for job {job_id}")

        # Attempt to refine language identification using transcribed text
        try:
            from src.services.language_service import LanguageIdentificationService
            lid_service = LanguageIdentificationService()
            trans_segments = result.get('transcription_segments', result.get('segments', []))
            if trans_segments:
                if lid_service.refine_with_transcripts_sync(job_id, trans_segments):
                    logger.info(f"Refined language identification using ASR transcripts for job {job_id}")
        except Exception as refine_exc:
            logger.warning(f"Could not refine language identification for {job_id}: {refine_exc}")

        return result
        
    except Exception as exc:
        logger.exception(f"Error in speech recognition for job {job_id}: {exc}")
        raise self.retry(exc=exc, countdown=60, max_retries=3)


@celery_app.task(bind=True, ignore_result=True)
def translate_text(self, job_id: str, request_params: Dict[str, Any]):
    """
    Neural machine translation task
    
    Args:
        job_id: Job identifier
        request_params: Processing parameters
    
    Returns:
        Translation results
    """
    try:
        logger.info(f"Starting translation for job {job_id}")

        # Allow disabling translation (Whisper transcribe-only mode)
        try:
            if request_params is not None and request_params.get('translate') is False:
                logger.info(f"Translation disabled by request for job {job_id}; generating passthrough results")
                _update_job_status(job_id, JobState.TRANSLATION, 0.9, "Translation skipped (passthrough)")
                # Build no-translation results from ASR output
                translation_service = TranslationService()
                transcription_data = translation_service.load_transcription_results_sync(job_id)
                segments = []
                for seg in transcription_data.get('segments', []):
                    segments.append({
                        'start': seg.get('start'),
                        'end': seg.get('end'),
                        'speaker': seg.get('speaker'),
                        'source_language': seg.get('language', 'unknown'),
                        'target_language': 'english',
                        'source_text': seg.get('text', ''),
                        'translated_text': seg.get('text', ''),
                        'translation_confidence': 1.0,
                        'translation_method': 'no_translation'
                    })
                # Persist and return in the same format as normal translation (sync context)
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(
                        translation_service._save_translation_results(job_id, {
                            'segments': segments,
                            'translation_segments': segments,
                            'quality_metrics': translation_service._calculate_translation_quality(segments) if segments else {},
                            'total_segments': len(segments),
                            'languages_translated': list({s.get('source_language','unknown') for s in segments})
                        })
                    )
                finally:
                    try:
                        loop.close()
                    except Exception:
                        pass
                logger.info(f"Translation passthrough completed for job {job_id}")
                return {'segments': segments, 'quality_metrics': {}, 'total_segments': len(segments)}
        except Exception as guard_exc:
            logger.warning(f"Translation disable guard failed; proceeding with normal translation: {guard_exc}")

        _update_job_status(job_id, JobState.TRANSLATION, 0.9, "Translating text")
        
        # Initialize service
        translation_service = TranslationService()
        
        # Load transcription results
        transcription_data = translation_service.load_transcription_results_sync(job_id)
        
        # Perform translation
        result = translation_service.translate_segments_sync(
            job_id,
            transcription_data['segments'],
            target_language='english'
        )
        
        logger.info(f"Translation completed for job {job_id}")
        return result
        
    except Exception as exc:
        logger.exception(f"Error in translation for job {job_id}: {exc}")
        raise self.retry(exc=exc, countdown=60, max_retries=3)


@celery_app.task(bind=True, ignore_result=True)
def generate_outputs(self, job_id: str, request_params: Dict[str, Any]):
    """
    Generate competition output formats
    
    Args:
        job_id: Job identifier
        request_params: Processing parameters
    
    Returns:
        Generated output file paths
    """
    try:
        logger.info(f"Starting output generation for job {job_id}")
        
        _update_job_status(job_id, JobState.POSTPROCESSING, 0.95, "Generating output files")
        
        # Initialize service
        format_service = FormatService()
        result_repo = ResultRepository()
        
        # Load all processing results
        all_results = format_service.load_all_results_sync(job_id)
        
        # Generate competition format outputs
        output_files = format_service.generate_competition_outputs_sync(
            job_id, all_results
        )
        
        # Calculate performance metrics
        metrics = format_service.calculate_performance_metrics_sync(all_results)
        
        # Create final result in ProcessResult format
        audio_specs = all_results.get('audio_specs', {})
        if not audio_specs.get('duration'):
            # Get duration from segments if not in specs
            segments = all_results.get('final_segments', [])
            if segments:
                audio_specs['duration'] = max(seg.get('end', 0.0) for seg in segments)
        
        # Extract speaker and language information
        speakers_detected = 0
        languages_detected = []
        
        if 'diarization' in all_results:
            speakers_detected = all_results['diarization'].get('num_speakers', 0)
        if 'language_identification' in all_results:
            languages_detected = all_results['language_identification'].get('languages_detected', [])
        
        final_result = {
            'job_id': job_id,
            'audio_specs': audio_specs,
            'processing_time': _calculate_processing_time(job_id),
            
            # Competition output file paths
            'sid_csv': output_files.get('sid_csv', ''),
            'sd_csv': output_files.get('sd_csv', ''),
            'lid_csv': output_files.get('lid_csv', ''),
            'asr_trn': output_files.get('asr_trn', ''),
            'nmt_txt': output_files.get('nmt_txt', ''),
            
            # Processing results
            'segments': all_results.get('final_segments', []),
            'speakers_detected': speakers_detected,
            'languages_detected': languages_detected,
            
            # Performance metrics
            'metrics': metrics,
            
            # Additional metadata
            'completed_at': datetime.utcnow().isoformat()
        }
        
        result_repo.save_result_sync(job_id, final_result)
        
        # Mark job as completed
        _update_job_status(job_id, JobState.COMPLETED, 1.0, "Processing completed successfully")
        
        logger.info(f"Output generation completed for job {job_id}")
        return final_result
        
    except Exception as exc:
        logger.exception(f"Error in output generation for job {job_id}: {exc}")
        raise self.retry(exc=exc, countdown=60, max_retries=3)


@celery_app.task
def cleanup_old_jobs():
    """Periodic task to cleanup old jobs and files"""
    try:
        logger.info("Starting periodic cleanup task")
        
        job_repo = JobRepository()
        
        # Clean up jobs older than 7 days
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        cleaned_count = job_repo.cleanup_old_jobs_sync(cutoff_time)
        
        logger.info(f"Cleaned up {cleaned_count} old jobs")
        return {"cleaned_jobs": cleaned_count}
        
    except Exception as exc:
        logger.exception(f"Error in cleanup task: {exc}")
        raise


@celery_app.task
def system_health_check():
    """Periodic system health check task"""
    try:
        logger.info("Running system health check")
        
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'celery_status': 'healthy',
            'queue_sizes': _get_queue_sizes(),
            'active_jobs': _get_active_job_count(),
            'system_resources': _get_system_resources()
        }
        
        logger.info("System health check completed")
        return health_status
        
    except Exception as exc:
        logger.exception(f"Error in health check: {exc}")
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'celery_status': 'error',
            'error': str(exc)
        }


def _update_job_status(
    job_id: str, 
    status: JobState, 
    progress: float, 
    stage: str,
    error_msg: str = None
):
    """Update job status in database"""
    try:
        job_repo = get_celery_job_repo()
        
        update_data = {
            'status': status,
            'progress': progress,
            'current_stage': stage,
            'updated_at': datetime.utcnow()
        }
        
        if error_msg:
            update_data['error_msg'] = error_msg
        
        result = job_repo.update_job_sync(job_id, update_data)
        if not result:
            logger.warning(f"Failed to update job status for {job_id}")
        
    except Exception as e:
        logger.exception(f"Error updating job status: {e}")


def _calculate_processing_time(job_id: str) -> float:
    """Calculate total processing time for job"""
    try:
        job_repo = get_celery_job_repo()
        job_data = job_repo.get_job_sync(job_id)
        
        if job_data:
            start_time = job_data.get('created_at')
            end_time = datetime.utcnow()
            
            if start_time:
                # Handle datetime strings from JSON storage
                if isinstance(start_time, str):
                    from datetime import datetime
                    start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                return (end_time - start_time).total_seconds()
        
        return 0.0
        
    except Exception:
        return 0.0


def _get_queue_sizes() -> Dict[str, int]:
    """Get current queue sizes"""
    try:
        inspect = celery_app.control.inspect()
        active_queues = inspect.active_queues()
        
        if active_queues:
            # This is a simplified version
            # In reality, you'd need to check each queue size
            return {
                'high_priority': 0,
                'normal_priority': 0,
                'low_priority': 0,
                'gpu_queue': 0
            }
        
        return {}
        
    except Exception:
        return {}


def _get_active_job_count() -> int:
    """Get number of currently active jobs"""
    try:
        # For now, return a placeholder count
        # Would need to implement count_active_jobs_sync in CeleryJobRepository
        return 0
        
    except Exception:
        return 0


def _get_system_resources() -> Dict[str, Any]:
    """Get current system resource usage"""
    try:
        import psutil
        
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
        }
        
    except ImportError:
        return {'error': 'psutil not available'}
    except Exception as e:
        return {'error': str(e)}
