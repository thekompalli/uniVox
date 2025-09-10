"""
Audio Repository
Storage operations for audio files and processing results
"""
import logging
import asyncio
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import soundfile as sf
import aiofiles
from minio import Minio
from minio.error import S3Error

from src.config.app_config import app_config

logger = logging.getLogger(__name__)


class AudioRepository:
    """Repository for audio file storage operations"""
    
    def __init__(self):
        self.minio_client = None
        self.local_storage_path = app_config.data_dir / "audio"
        self.local_storage_path.mkdir(exist_ok=True)
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize storage connections"""
        try:
            async with self._lock:
                if self.minio_client is None:
                    self.minio_client = Minio(
                        app_config.minio_endpoint,
                        access_key=app_config.minio_access_key,
                        secret_key=app_config.minio_secret_key,
                        secure=app_config.minio_secure
                    )
                    
                    # Create buckets if they don't exist
                    await self._create_buckets()
                    
            logger.info("Audio repository initialized")
            
        except Exception as e:
            logger.exception(f"Error initializing audio repository: {e}")
            # Fall back to local storage only
            logger.warning("Falling back to local storage only")
    
    async def cleanup(self):
        """Cleanup storage connections"""
        try:
            # MinIO client doesn't need explicit cleanup
            logger.info("Audio repository cleanup completed")
            
        except Exception as e:
            logger.exception(f"Error during audio repository cleanup: {e}")
    
    async def _create_buckets(self):
        """Create MinIO buckets if they don't exist"""
        try:
            buckets = [
                app_config.audio_bucket,
                app_config.results_bucket,
                app_config.models_bucket
            ]
            
            def create_bucket_sync(bucket_name):
                try:
                    if not self.minio_client.bucket_exists(bucket_name):
                        self.minio_client.make_bucket(bucket_name)
                        logger.info(f"Created bucket: {bucket_name}")
                except S3Error as e:
                    logger.exception(f"Error creating bucket {bucket_name}: {e}")
                    raise
            
            # Run bucket creation in thread pool
            loop = asyncio.get_event_loop()
            for bucket in buckets:
                await loop.run_in_executor(None, create_bucket_sync, bucket)
                
        except Exception as e:
            logger.exception(f"Error creating buckets: {e}")
            raise
    
    async def save_audio_file(
        self, 
        job_id: str, 
        audio_content: bytes, 
        original_filename: str
    ) -> Path:
        """Save uploaded audio file"""
        try:
            # Create job directory
            job_dir = self.local_storage_path / job_id
            job_dir.mkdir(exist_ok=True)
            
            # Save to local storage first
            file_path = job_dir / f"original_{original_filename}"
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(audio_content)
            
            # Also save to MinIO if available
            if self.minio_client:
                try:
                    await self._save_to_minio(
                        app_config.audio_bucket,
                        f"{job_id}/original_{original_filename}",
                        file_path
                    )
                except Exception as e:
                    logger.warning(f"Failed to save to MinIO: {e}")
            
            logger.info(f"Saved audio file for job {job_id}: {file_path}")
            return file_path
            
        except Exception as e:
            logger.exception(f"Error saving audio file: {e}")
            raise
    
    async def save_processed_audio(
        self,
        job_id: str,
        audio_data: np.ndarray,
        sample_rate: int,
        filename: str
    ) -> Path:
        """Save processed audio data"""
        try:
            job_dir = self.local_storage_path / job_id
            job_dir.mkdir(exist_ok=True)
            
            file_path = job_dir / filename
            
            # Save using soundfile
            sf.write(str(file_path), audio_data, sample_rate)
            
            # Also save to MinIO if available
            if self.minio_client:
                try:
                    await self._save_to_minio(
                        app_config.audio_bucket,
                        f"{job_id}/{filename}",
                        file_path
                    )
                except Exception as e:
                    logger.warning(f"Failed to save processed audio to MinIO: {e}")
            
            logger.info(f"Saved processed audio: {file_path}")
            return file_path
            
        except Exception as e:
            logger.exception(f"Error saving processed audio: {e}")
            raise
    
    async def load_audio_file(self, job_id: str, filename: str) -> Path:
        """Load audio file by job ID and filename"""
        try:
            # Check local storage first
            local_path = self.local_storage_path / job_id / filename
            if local_path.exists():
                return local_path
            
            # Try to download from MinIO
            if self.minio_client:
                try:
                    await self._download_from_minio(
                        app_config.audio_bucket,
                        f"{job_id}/{filename}",
                        local_path
                    )
                    if local_path.exists():
                        return local_path
                except Exception as e:
                    logger.warning(f"Failed to download from MinIO: {e}")
            
            raise FileNotFoundError(f"Audio file not found: {job_id}/{filename}")
            
        except Exception as e:
            logger.exception(f"Error loading audio file: {e}")
            raise
    
    async def save_processing_results(
        self,
        job_id: str,
        stage: str,
        results: Dict[str, Any]
    ):
        """Save processing results for a specific stage"""
        try:
            job_dir = self.local_storage_path / job_id
            job_dir.mkdir(exist_ok=True)
            
            # Save as JSON
            results_file = job_dir / f"{stage}_results.json"
            async with aiofiles.open(results_file, 'w', encoding='utf-8') as f:
                # Write UTF-8 with non-ASCII preserved for human readability (e.g., Devanagari)
                await f.write(json.dumps(results, default=str, indent=2, ensure_ascii=False))
            
            # Save to MinIO if available
            if self.minio_client:
                try:
                    await self._save_to_minio(
                        app_config.results_bucket,
                        f"{job_id}/{stage}_results.json",
                        results_file
                    )
                except Exception as e:
                    logger.warning(f"Failed to save results to MinIO: {e}")
            
            logger.info(f"Saved {stage} results for job {job_id}")
            
        except Exception as e:
            logger.exception(f"Error saving processing results: {e}")
            raise
    
    async def load_processing_results(
        self,
        job_id: str,
        stage: str
    ) -> Optional[Dict[str, Any]]:
        """Load processing results for a specific stage"""
        try:
            # Check local storage first
            local_path = self.local_storage_path / job_id / f"{stage}_results.json"
            
            if not local_path.exists() and self.minio_client:
                # Try to download from MinIO
                try:
                    await self._download_from_minio(
                        app_config.results_bucket,
                        f"{job_id}/{stage}_results.json",
                        local_path
                    )
                except Exception as e:
                    logger.warning(f"Failed to download results from MinIO: {e}")
            
            if local_path.exists():
                async with aiofiles.open(local_path, 'r') as f:
                    content = await f.read()
                    return json.loads(content)
            
            return None
            
        except Exception as e:
            logger.exception(f"Error loading processing results: {e}")
            return None
    
    async def cleanup_job_files(self, job_id: str):
        """Clean up all files for a job"""
        try:
            # Clean up local files
            job_dir = self.local_storage_path / job_id
            if job_dir.exists():
                import shutil
                shutil.rmtree(job_dir)
                logger.info(f"Cleaned up local files for job {job_id}")
            
            # Clean up MinIO files
            if self.minio_client:
                try:
                    await self._cleanup_minio_prefix(app_config.audio_bucket, f"{job_id}/")
                    await self._cleanup_minio_prefix(app_config.results_bucket, f"{job_id}/")
                    logger.info(f"Cleaned up MinIO files for job {job_id}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup MinIO files: {e}")
            
        except Exception as e:
            logger.exception(f"Error cleaning up job files: {e}")
    
    async def cleanup_temp_files(self, cutoff_time: datetime):
        """Clean up temporary files older than cutoff time"""
        try:
            cleaned_count = 0
            
            for job_dir in self.local_storage_path.iterdir():
                if job_dir.is_dir():
                    # Check directory modification time
                    mod_time = datetime.fromtimestamp(job_dir.stat().st_mtime)
                    if mod_time < cutoff_time:
                        # Check if job is still active (simplified check)
                        results_exist = any(
                            f.name.endswith('_results.json') 
                            for f in job_dir.iterdir() 
                            if f.is_file()
                        )
                        
                        if not results_exist:
                            import shutil
                            shutil.rmtree(job_dir)
                            cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} temporary job directories")
            return cleaned_count
            
        except Exception as e:
            logger.exception(f"Error cleaning up temp files: {e}")
            return 0
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            stats = {
                'local_storage': {},
                'minio_storage': {}
            }
            
            # Local storage stats
            if self.local_storage_path.exists():
                total_size = 0
                file_count = 0
                
                for item in self.local_storage_path.rglob('*'):
                    if item.is_file():
                        total_size += item.stat().st_size
                        file_count += 1
                
                stats['local_storage'] = {
                    'total_size_mb': total_size / (1024 * 1024),
                    'file_count': file_count,
                    'directory_count': len(list(self.local_storage_path.iterdir()))
                }
            
            # MinIO stats (simplified)
            if self.minio_client:
                stats['minio_storage'] = {
                    'buckets': [
                        app_config.audio_bucket,
                        app_config.results_bucket,
                        app_config.models_bucket
                    ],
                    'status': 'connected'
                }
            
            return stats
            
        except Exception as e:
            logger.exception(f"Error getting storage stats: {e}")
            return {}
    
    async def _save_to_minio(
        self, 
        bucket: str, 
        object_name: str, 
        file_path: Path
    ):
        """Save file to MinIO"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.minio_client.fput_object,
                bucket,
                object_name,
                str(file_path)
            )
        except Exception as e:
            logger.exception(f"Error saving to MinIO: {e}")
            raise
    
    async def _download_from_minio(
        self, 
        bucket: str, 
        object_name: str, 
        file_path: Path
    ):
        """Download file from MinIO"""
        try:
            # Create parent directory
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.minio_client.fget_object,
                bucket,
                object_name,
                str(file_path)
            )
        except Exception as e:
            logger.exception(f"Error downloading from MinIO: {e}")
            raise
    
    async def _cleanup_minio_prefix(self, bucket: str, prefix: str):
        """Clean up all objects with given prefix in MinIO"""
        try:
            def cleanup_sync():
                objects = self.minio_client.list_objects(bucket, prefix=prefix, recursive=True)
                for obj in objects:
                    self.minio_client.remove_object(bucket, obj.object_name)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, cleanup_sync)
            
        except Exception as e:
            logger.exception(f"Error cleaning up MinIO prefix: {e}")
            raise
    
    def save_processing_results_sync(
        self, 
        job_id: str, 
        stage: str, 
        results: Dict[str, Any]
    ):
        """Synchronous version for Celery tasks"""
        try:
            job_dir = self.local_storage_path / job_id
            job_dir.mkdir(exist_ok=True)
            
            results_file = job_dir / f"{stage}_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, default=str, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {stage} results for job {job_id} (sync)")
            
        except Exception as e:
            logger.exception(f"Error saving processing results sync: {e}")
            raise
