"""
Result Repository
Storage and retrieval of processing results
"""
import logging
import asyncio
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import asyncpg

from src.config.app_config import app_config

logger = logging.getLogger(__name__)


class ResultRepository:
    """Repository for processing results storage and retrieval"""
    
    def __init__(self):
        self.pool = None
        self.file_storage_path = app_config.data_dir / "results"
        self.file_storage_path.mkdir(exist_ok=True)
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize database connection and storage"""
        try:
            async with self._lock:
                if self.pool is None:
                    self.pool = await asyncpg.create_pool(
                        app_config.database_url,
                        min_size=2,
                        max_size=app_config.database_pool_size
                    )
                    
                    # Create tables if they don't exist
                    await self._create_tables()
                    
            logger.info("Result repository initialized")
            
        except Exception as e:
            logger.exception(f"Error initializing result repository: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup database connections"""
        try:
            if self.pool:
                await self.pool.close()
                self.pool = None
            logger.info("Result repository cleanup completed")
            
        except Exception as e:
            logger.exception(f"Error during result repository cleanup: {e}")
    
    async def _create_tables(self):
        """Create database tables for results"""
        try:
            async with self.pool.acquire() as connection:
                # Results table
                await connection.execute("""
                    CREATE TABLE IF NOT EXISTS results (
                        job_id TEXT PRIMARY KEY,
                        result_type TEXT NOT NULL,
                        result_data JSONB,
                        file_paths JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        status TEXT DEFAULT 'active',
                        metadata JSONB DEFAULT '{}'
                    );
                """)
                
                # Performance metrics table
                await connection.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        job_id TEXT PRIMARY KEY,
                        speaker_accuracy FLOAT,
                        diarization_error_rate FLOAT,
                        language_accuracy FLOAT,
                        word_error_rate FLOAT,
                        bleu_score FLOAT,
                        processing_time FLOAT,
                        rtf FLOAT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        FOREIGN KEY (job_id) REFERENCES results(job_id) ON DELETE CASCADE
                    );
                """)
                
                # Validation results table
                await connection.execute("""
                    CREATE TABLE IF NOT EXISTS validation_results (
                        job_id TEXT PRIMARY KEY,
                        sid_valid BOOLEAN DEFAULT FALSE,
                        sd_valid BOOLEAN DEFAULT FALSE,
                        lid_valid BOOLEAN DEFAULT FALSE,
                        asr_valid BOOLEAN DEFAULT FALSE,
                        nmt_valid BOOLEAN DEFAULT FALSE,
                        validation_errors JSONB DEFAULT '[]',
                        validated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        FOREIGN KEY (job_id) REFERENCES results(job_id) ON DELETE CASCADE
                    );
                """)
                
                # Create indices
                await connection.execute("""
                    CREATE INDEX IF NOT EXISTS idx_results_created_at ON results(created_at);
                """)
                await connection.execute("""
                    CREATE INDEX IF NOT EXISTS idx_results_status ON results(status);
                """)
                await connection.execute("""
                    CREATE INDEX IF NOT EXISTS idx_results_type ON results(result_type);
                """)
                
            logger.info("Result database tables created/verified")
            
        except Exception as e:
            logger.exception(f"Error creating result tables: {e}")
            raise
    
    async def save_result(
        self,
        job_id: str,
        result_type: str,
        result_data: Dict[str, Any],
        file_paths: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save processing result
        
        Args:
            job_id: Job identifier
            result_type: Type of result (diarization, asr, etc.)
            result_data: Result data dictionary
            file_paths: Optional file paths dictionary
            metadata: Optional metadata
            
        Returns:
            Success status
        """
        try:
            # Save to database
            async with self.pool.acquire() as connection:
                await connection.execute("""
                    INSERT INTO results (job_id, result_type, result_data, file_paths, metadata, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (job_id) 
                    DO UPDATE SET 
                        result_type = EXCLUDED.result_type,
                        result_data = EXCLUDED.result_data,
                        file_paths = EXCLUDED.file_paths,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at
                """, 
                    job_id,
                    result_type,
                    json.dumps(result_data, default=str),
                    json.dumps(file_paths or {}, default=str),
                    json.dumps(metadata or {}, default=str),
                    datetime.utcnow()
                )
            
            # Also save to file system for backup
            await self._save_result_to_file(job_id, result_type, result_data, file_paths)
            
            logger.info(f"Saved {result_type} result for job {job_id}")
            return True
            
        except Exception as e:
            logger.exception(f"Error saving result: {e}")
            return False
    
    async def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete result by job ID
        
        Args:
            job_id: Job identifier
            
        Returns:
            Complete result data or None if not found
        """
        try:
            async with self.pool.acquire() as connection:
                row = await connection.fetchrow("""
                    SELECT * FROM results WHERE job_id = $1
                """, job_id)
                
                if row:
                    result = dict(row)
                    # Parse JSON fields
                    if result.get('result_data'):
                        result['result_data'] = json.loads(result['result_data'])
                    if result.get('file_paths'):
                        result['file_paths'] = json.loads(result['file_paths'])
                    if result.get('metadata'):
                        result['metadata'] = json.loads(result['metadata'])
                    
                    return result
                
                return None
                
        except Exception as e:
            logger.exception(f"Error getting result {job_id}: {e}")
            # Try to load from file system as fallback
            return await self._load_result_from_file(job_id)
    
    async def get_result_by_type(
        self, 
        job_id: str, 
        result_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get specific result type for a job"""
        try:
            result = await self.get_result(job_id)
            if result and result.get('result_type') == result_type:
                return result.get('result_data', {})
            
            return None
            
        except Exception as e:
            logger.exception(f"Error getting result by type: {e}")
            return None
    
    async def save_performance_metrics(
        self,
        job_id: str,
        metrics: Dict[str, float]
    ) -> bool:
        """Save performance metrics for a job"""
        try:
            async with self.pool.acquire() as connection:
                await connection.execute("""
                    INSERT INTO performance_metrics (
                        job_id, speaker_accuracy, diarization_error_rate,
                        language_accuracy, word_error_rate, bleu_score,
                        processing_time, rtf
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (job_id)
                    DO UPDATE SET
                        speaker_accuracy = EXCLUDED.speaker_accuracy,
                        diarization_error_rate = EXCLUDED.diarization_error_rate,
                        language_accuracy = EXCLUDED.language_accuracy,
                        word_error_rate = EXCLUDED.word_error_rate,
                        bleu_score = EXCLUDED.bleu_score,
                        processing_time = EXCLUDED.processing_time,
                        rtf = EXCLUDED.rtf
                """,
                    job_id,
                    metrics.get('speaker_accuracy', 0.0),
                    metrics.get('diarization_error_rate', 1.0),
                    metrics.get('language_accuracy', 0.0),
                    metrics.get('word_error_rate', 1.0),
                    metrics.get('bleu_score', 0.0),
                    metrics.get('processing_time', 0.0),
                    metrics.get('rtf', 0.0)
                )
            
            logger.info(f"Saved performance metrics for job {job_id}")
            return True
            
        except Exception as e:
            logger.exception(f"Error saving performance metrics: {e}")
            return False
    
    async def get_performance_metrics(self, job_id: str) -> Optional[Dict[str, float]]:
        """Get performance metrics for a job"""
        try:
            async with self.pool.acquire() as connection:
                row = await connection.fetchrow("""
                    SELECT * FROM performance_metrics WHERE job_id = $1
                """, job_id)
                
                if row:
                    return {
                        'speaker_accuracy': row['speaker_accuracy'],
                        'diarization_error_rate': row['diarization_error_rate'],
                        'language_accuracy': row['language_accuracy'],
                        'word_error_rate': row['word_error_rate'],
                        'bleu_score': row['bleu_score'],
                        'processing_time': row['processing_time'],
                        'rtf': row['rtf']
                    }
                
                return None
                
        except Exception as e:
            logger.exception(f"Error getting performance metrics: {e}")
            return None
    
    async def save_validation_results(
        self,
        job_id: str,
        validation_data: Dict[str, Any]
    ) -> bool:
        """Save validation results for output files"""
        try:
            async with self.pool.acquire() as connection:
                await connection.execute("""
                    INSERT INTO validation_results (
                        job_id, sid_valid, sd_valid, lid_valid, asr_valid, nmt_valid,
                        validation_errors, validated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (job_id)
                    DO UPDATE SET
                        sid_valid = EXCLUDED.sid_valid,
                        sd_valid = EXCLUDED.sd_valid,
                        lid_valid = EXCLUDED.lid_valid,
                        asr_valid = EXCLUDED.asr_valid,
                        nmt_valid = EXCLUDED.nmt_valid,
                        validation_errors = EXCLUDED.validation_errors,
                        validated_at = EXCLUDED.validated_at
                """,
                    job_id,
                    validation_data.get('sid_csv', {}).get('valid', False),
                    validation_data.get('sd_csv', {}).get('valid', False),
                    validation_data.get('lid_csv', {}).get('valid', False),
                    validation_data.get('asr_trn', {}).get('valid', False),
                    validation_data.get('nmt_txt', {}).get('valid', False),
                    json.dumps([
                        error for file_data in validation_data.values()
                        for error in file_data.get('errors', [])
                    ]),
                    datetime.utcnow()
                )
            
            logger.info(f"Saved validation results for job {job_id}")
            return True
            
        except Exception as e:
            logger.exception(f"Error saving validation results: {e}")
            return False
    
    async def get_validation_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get validation results for a job"""
        try:
            async with self.pool.acquire() as connection:
                row = await connection.fetchrow("""
                    SELECT * FROM validation_results WHERE job_id = $1
                """, job_id)
                
                if row:
                    return {
                        'sid_valid': row['sid_valid'],
                        'sd_valid': row['sd_valid'],
                        'lid_valid': row['lid_valid'],
                        'asr_valid': row['asr_valid'],
                        'nmt_valid': row['nmt_valid'],
                        'validation_errors': json.loads(row['validation_errors']),
                        'validated_at': row['validated_at'],
                        'all_valid': all([
                            row['sid_valid'], row['sd_valid'], row['lid_valid'],
                            row['asr_valid'], row['nmt_valid']
                        ])
                    }
                
                return None
                
        except Exception as e:
            logger.exception(f"Error getting validation results: {e}")
            return None
    
    async def list_results(
        self,
        limit: int = 50,
        offset: int = 0,
        result_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List results with filtering and pagination"""
        try:
            query = "SELECT job_id, result_type, created_at, updated_at, status FROM results"
            params = []
            conditions = []
            param_count = 1
            
            if result_type:
                conditions.append(f"result_type = ${param_count}")
                params.append(result_type)
                param_count += 1
            
            if status:
                conditions.append(f"status = ${param_count}")
                params.append(status)
                param_count += 1
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += f" ORDER BY created_at DESC LIMIT ${param_count} OFFSET ${param_count + 1}"
            params.extend([limit, offset])
            
            async with self.pool.acquire() as connection:
                rows = await connection.fetch(query, *params)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.exception(f"Error listing results: {e}")
            return []
    
    async def delete_result(self, job_id: str) -> bool:
        """Delete result and associated data"""
        try:
            async with self.pool.acquire() as connection:
                # Delete from all tables (cascading will handle related data)
                result = await connection.execute("""
                    DELETE FROM results WHERE job_id = $1
                """, job_id)
                
                deleted = result.split()[-1] == '1'
            
            # Also delete file system data
            if deleted:
                await self._delete_result_files(job_id)
            
            logger.info(f"Deleted result for job {job_id}")
            return deleted
            
        except Exception as e:
            logger.exception(f"Error deleting result: {e}")
            return False
    
    async def cleanup_old_results(self, cutoff_time: datetime) -> int:
        """Clean up old results"""
        try:
            async with self.pool.acquire() as connection:
                # Get job IDs to delete files
                rows = await connection.fetch("""
                    SELECT job_id FROM results 
                    WHERE created_at < $1 AND status = 'completed'
                """, cutoff_time)
                
                job_ids = [row['job_id'] for row in rows]
                
                # Delete from database
                result = await connection.execute("""
                    DELETE FROM results 
                    WHERE created_at < $1 AND status = 'completed'
                """, cutoff_time)
                
                deleted_count = int(result.split()[-1])
            
            # Clean up file system
            for job_id in job_ids:
                await self._delete_result_files(job_id)
            
            logger.info(f"Cleaned up {deleted_count} old results")
            return deleted_count
            
        except Exception as e:
            logger.exception(f"Error cleaning up old results: {e}")
            return 0
    
    async def get_results_statistics(self) -> Dict[str, Any]:
        """Get results statistics"""
        try:
            async with self.pool.acquire() as connection:
                # Basic counts
                basic_stats = await connection.fetchrow("""
                    SELECT 
                        COUNT(*) as total_results,
                        COUNT(*) FILTER (WHERE status = 'active') as active_results,
                        COUNT(*) FILTER (WHERE status = 'completed') as completed_results,
                        COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as last_24h
                    FROM results
                """)
                
                # Performance averages
                perf_stats = await connection.fetchrow("""
                    SELECT 
                        AVG(speaker_accuracy) as avg_speaker_accuracy,
                        AVG(diarization_error_rate) as avg_diarization_error,
                        AVG(language_accuracy) as avg_language_accuracy,
                        AVG(word_error_rate) as avg_word_error,
                        AVG(bleu_score) as avg_bleu_score,
                        AVG(processing_time) as avg_processing_time,
                        AVG(rtf) as avg_rtf
                    FROM performance_metrics
                    WHERE created_at > NOW() - INTERVAL '7 days'
                """)
                
                # Result type distribution
                type_stats = await connection.fetch("""
                    SELECT result_type, COUNT(*) as count
                    FROM results
                    GROUP BY result_type
                """)
                
                return {
                    'total_results': basic_stats['total_results'],
                    'active_results': basic_stats['active_results'],
                    'completed_results': basic_stats['completed_results'],
                    'last_24h': basic_stats['last_24h'],
                    'avg_performance': {
                        'speaker_accuracy': float(perf_stats['avg_speaker_accuracy'] or 0),
                        'diarization_error_rate': float(perf_stats['avg_diarization_error'] or 0),
                        'language_accuracy': float(perf_stats['avg_language_accuracy'] or 0),
                        'word_error_rate': float(perf_stats['avg_word_error'] or 0),
                        'bleu_score': float(perf_stats['avg_bleu_score'] or 0),
                        'processing_time': float(perf_stats['avg_processing_time'] or 0),
                        'rtf': float(perf_stats['avg_rtf'] or 0)
                    },
                    'result_types': {row['result_type']: row['count'] for row in type_stats}
                }
                
        except Exception as e:
            logger.exception(f"Error getting results statistics: {e}")
            return {}
    
    async def _save_result_to_file(
        self,
        job_id: str,
        result_type: str,
        result_data: Dict[str, Any],
        file_paths: Optional[Dict[str, str]] = None
    ):
        """Save result to file system as backup"""
        try:
            job_dir = self.file_storage_path / job_id
            job_dir.mkdir(exist_ok=True)
            
            # Save main result data
            result_file = job_dir / f"{result_type}_result.json"
            with open(result_file, 'w') as f:
                json.dump(result_data, f, default=str, indent=2)
            
            # Save file paths if provided
            if file_paths:
                paths_file = job_dir / f"{result_type}_paths.json"
                with open(paths_file, 'w') as f:
                    json.dump(file_paths, f, indent=2)
            
        except Exception as e:
            logger.exception(f"Error saving result to file: {e}")
    
    async def _load_result_from_file(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load result from file system"""
        try:
            job_dir = self.file_storage_path / job_id
            if not job_dir.exists():
                return None
            
            # Find result files
            result_files = list(job_dir.glob("*_result.json"))
            if not result_files:
                return None
            
            # Load the most recent result file
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                result_data = json.load(f)
            
            # Try to load file paths
            result_type = latest_file.stem.replace('_result', '')
            paths_file = job_dir / f"{result_type}_paths.json"
            file_paths = {}
            
            if paths_file.exists():
                with open(paths_file, 'r') as f:
                    file_paths = json.load(f)
            
            return {
                'job_id': job_id,
                'result_type': result_type,
                'result_data': result_data,
                'file_paths': file_paths,
                'created_at': datetime.fromtimestamp(latest_file.stat().st_ctime),
                'updated_at': datetime.fromtimestamp(latest_file.stat().st_mtime)
            }
            
        except Exception as e:
            logger.exception(f"Error loading result from file: {e}")
            return None
    
    async def _delete_result_files(self, job_id: str):
        """Delete result files from file system"""
        try:
            job_dir = self.file_storage_path / job_id
            if job_dir.exists():
                import shutil
                shutil.rmtree(job_dir)
                
        except Exception as e:
            logger.exception(f"Error deleting result files: {e}")
    
    # Synchronous methods for Celery tasks
    def save_result_sync(self, job_id: str, result_data: Dict[str, Any]) -> bool:
        """Save result synchronously"""
        try:
            # Simplified sync implementation
            job_dir = self.file_storage_path / job_id
            job_dir.mkdir(exist_ok=True)
            
            result_file = job_dir / "final_result.json"
            with open(result_file, 'w') as f:
                json.dump(result_data, f, default=str, indent=2)
            
            return True
            
        except Exception as e:
            logger.exception(f"Error saving result sync: {e}")
            return False