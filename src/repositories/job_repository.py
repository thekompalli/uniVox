"""
Job Repository
Database operations for job tracking and management
"""
import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import aiofiles
import asyncpg
from contextlib import asynccontextmanager

from src.api.schemas.process_schemas import JobState, ProcessRequest, JobStatus
from src.config.database_config import DatabaseConfig
from src.utils.error_handler import ErrorHandler

logger = logging.getLogger(__name__)


class JobRepository:
    """Repository for job management operations"""
    
    def __init__(self):
        self.pool = None
        self.error_handler = ErrorHandler()
        self.db_config = DatabaseConfig()
        self._initialized = False
        
        # Fallback to file-based storage if no database
        self.fallback_storage = Path("data/jobs")
        self.fallback_storage.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialize database connection pool"""
        try:
            logger.info("Initializing job repository")
            
            # Try to connect to PostgreSQL
            try:
                self.pool = await asyncpg.create_pool(
                    dsn=self.db_config.get_postgres_url(),
                    **self.db_config.get_connection_pool_config()["postgresql"]
                )
                
                # Create tables if they don't exist
                await self._create_tables()
                
                logger.info("Connected to PostgreSQL database")
                
            except Exception as db_error:
                logger.warning(f"PostgreSQL connection failed: {db_error}")
                logger.info("Falling back to file-based storage")
                self.pool = None
            
            self._initialized = True
            logger.info("Job repository initialized")
            
        except Exception as e:
            logger.exception(f"Error initializing job repository: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup database connections"""
        try:
            if self.pool:
                await self.pool.close()
                logger.info("Database connections closed")
                
        except Exception as e:
            logger.exception(f"Error during job repository cleanup: {e}")
    
    async def _create_tables(self):
        """Create database tables"""
        if not self.pool:
            return
        
        create_jobs_table = """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id VARCHAR(50) PRIMARY KEY,
            state VARCHAR(20) NOT NULL,
            progress FLOAT DEFAULT 0.0,
            status_message TEXT,
            task_id TEXT,
            request_params JSONB,
            result JSONB,
            error_details JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            expires_at TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
        CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
        CREATE INDEX IF NOT EXISTS idx_jobs_expires_at ON jobs(expires_at);
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(create_jobs_table)
            # Ensure task_id exists on previously created tables
            await conn.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS task_id TEXT")
    
    async def create_job(
        self, 
        job_id: str, 
        request_params: ProcessRequest
    ) -> JobStatus:
        """Create a new job"""
        try:
            job_data = {
                "job_id": job_id,
                "state": JobState.QUEUED,
                "progress": 0.0,
                "status_message": "Job queued for processing",
                "request_params": request_params.dict(),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "expires_at": datetime.utcnow() + timedelta(hours=24)
            }
            
            if self.pool:
                await self._create_job_db(job_data)
            else:
                await self._create_job_file(job_data)
            
            return JobStatus(
                job_id=job_id,
                status=JobState.QUEUED,
                progress=0.0,
                current_stage="Job queued for processing",
                created_at=job_data["created_at"],
                updated_at=job_data["updated_at"]
            )
            
        except Exception as e:
            logger.exception(f"Error creating job {job_id}: {e}")
            raise
    
    async def _create_job_db(self, job_data: Dict[str, Any]):
        """Create job in database"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO jobs (
                    job_id, state, progress, status_message, 
                    request_params, created_at, updated_at, expires_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                job_data["job_id"],
                job_data["state"],
                job_data["progress"],
                job_data["status_message"],
                json.dumps(job_data["request_params"]),
                job_data["created_at"],
                job_data["updated_at"],
                job_data["expires_at"]
            )
    
    async def _create_job_file(self, job_data: Dict[str, Any]):
        """Create job in file storage"""
        job_file = self.fallback_storage / f"{job_data['job_id']}.json"
        
        # Convert datetime objects to ISO strings for JSON serialization
        serializable_data = {
            **job_data,
            "created_at": job_data["created_at"].isoformat(),
            "updated_at": job_data["updated_at"].isoformat(),
            "expires_at": job_data["expires_at"].isoformat()
        }
        
        async with aiofiles.open(job_file, 'w') as f:
            await f.write(json.dumps(serializable_data, indent=2))
    
    async def get_job(self, job_id: str) -> Optional[JobStatus]:
        """Get job by ID"""
        try:
            if self.pool:
                return await self._get_job_db(job_id)
            else:
                return await self._get_job_file(job_id)
                
        except Exception as e:
            logger.exception(f"Error getting job {job_id}: {e}")
            return None
    
    async def _get_job_db(self, job_id: str) -> Optional[JobStatus]:
        """Get job from database"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM jobs WHERE job_id = $1", job_id
            )
            
            if not row:
                return None
            
            return JobStatus(
                job_id=row["job_id"],
                status=JobState(row["state"]),
                progress=row["progress"],
                current_stage=row["status_message"],
                error_msg=row.get("error_details"),
                created_at=row["created_at"],
                updated_at=row["updated_at"]
            )
    
    async def _get_job_file(self, job_id: str) -> Optional[JobStatus]:
        """Get job from file storage"""
        job_file = self.fallback_storage / f"{job_id}.json"
        
        if not job_file.exists():
            return None
        
        try:
            async with aiofiles.open(job_file, 'r') as f:
                data = json.loads(await f.read())
            
            return JobStatus(
                job_id=data["job_id"],
                status=JobState(data["state"]),
                progress=data["progress"],
                current_stage=data["status_message"],
                error_msg=data.get("error_details"),
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"])
            )
            
        except Exception as e:
            logger.exception(f"Error reading job file {job_file}: {e}")
            return None
    
    async def update_job_status(
        self,
        job_id: str,
        state: Optional[JobState] = None,
        progress: Optional[float] = None,
        status_message: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update job status"""
        try:
            if self.pool:
                return await self._update_job_db(job_id, state, progress, status_message, error_details)
            else:
                return await self._update_job_file(job_id, state, progress, status_message, error_details)
                
        except Exception as e:
            logger.exception(f"Error updating job {job_id}: {e}")
            return False
    
    async def update_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update job with arbitrary fields"""
        try:
            if self.pool:
                return await self._update_job_db_generic(job_id, updates)
            else:
                return await self._update_job_file_generic(job_id, updates)
                
        except Exception as e:
            logger.exception(f"Error updating job {job_id}: {e}")
            return False
    
    async def _update_job_db_generic(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update job in database with generic fields"""
        try:
            if not updates:
                return True
                
            set_clauses = []
            params = []
            param_count = 1
            # Only allow known columns to avoid UndefinedColumn errors
            allowed_fields = {
                'state', 'progress', 'status_message', 'task_id',
                'request_params', 'result', 'error_details',
                'created_at', 'updated_at', 'completed_at', 'expires_at'
            }
            
            for key, value in updates.items():
                if key not in allowed_fields:
                    # Skip unknown columns silently
                    continue
                set_clauses.append(f"{key} = ${param_count}")
                params.append(value)
                param_count += 1
            
            set_clauses.append(f"updated_at = ${param_count}")
            params.append(datetime.utcnow())
            params.append(job_id)
            
            query = f"UPDATE jobs SET {', '.join(set_clauses)} WHERE job_id = ${param_count + 1}"
            
            async with self.pool.acquire() as connection:
                result = await connection.execute(query, *params)
                return result.split()[-1] == '1'
                
        except Exception as e:
            logger.exception(f"Error updating job {job_id} in database: {e}")
            return False
    
    async def _update_job_file_generic(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update job in file with generic fields"""
        try:
            job_file = self.fallback_storage / f"{job_id}.json"
            
            if job_file.exists():
                async with aiofiles.open(job_file, 'r') as f:
                    job_data = json.loads(await f.read())
                
                # Update the fields
                for key, value in updates.items():
                    job_data[key] = value
                job_data['updated_at'] = datetime.utcnow().isoformat()
                
                async with aiofiles.open(job_file, 'w') as f:
                    await f.write(json.dumps(job_data, indent=2, default=str))
                
                return True
            
            return False
            
        except Exception as e:
            logger.exception(f"Error updating job {job_id} in file: {e}")
            return False
    
    async def _update_job_db(
        self,
        job_id: str,
        state: Optional[JobState],
        progress: Optional[float],
        status_message: Optional[str],
        error_details: Optional[Dict[str, Any]]
    ) -> bool:
        """Update job in database"""
        updates = ["updated_at = $2"]
        params = [job_id, datetime.utcnow()]
        param_count = 2
        
        if state is not None:
            param_count += 1
            updates.append(f"state = ${param_count}")
            params.append(state.value)
            
            if state == JobState.COMPLETED:
                param_count += 1
                updates.append(f"completed_at = ${param_count}")
                params.append(datetime.utcnow())
        
        if progress is not None:
            param_count += 1
            updates.append(f"progress = ${param_count}")
            params.append(progress)
        
        if status_message is not None:
            param_count += 1
            updates.append(f"status_message = ${param_count}")
            params.append(status_message)
        
        if error_details is not None:
            param_count += 1
            updates.append(f"error_details = ${param_count}")
            params.append(json.dumps(error_details))
        
        query = f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = $1"
        
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, *params)
            return result == "UPDATE 1"
    
    async def _update_job_file(
        self,
        job_id: str,
        state: Optional[JobState],
        progress: Optional[float],
        status_message: Optional[str],
        error_details: Optional[Dict[str, Any]]
    ) -> bool:
        """Update job in file storage"""
        job_file = self.fallback_storage / f"{job_id}.json"
        
        if not job_file.exists():
            return False
        
        try:
            async with aiofiles.open(job_file, 'r') as f:
                data = json.loads(await f.read())
            
            # Update fields
            data["updated_at"] = datetime.utcnow().isoformat()
            
            if state is not None:
                data["state"] = state.value
                if state == JobState.COMPLETED:
                    data["completed_at"] = datetime.utcnow().isoformat()
            
            if progress is not None:
                data["progress"] = progress
            
            if status_message is not None:
                data["status_message"] = status_message
            
            if error_details is not None:
                data["error_details"] = error_details
            
            async with aiofiles.open(job_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
            
            return True
            
        except Exception as e:
            logger.exception(f"Error updating job file {job_file}: {e}")
            return False
    
    async def set_job_result(self, job_id: str, result: Dict[str, Any]) -> bool:
        """Set job result"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE jobs 
                        SET result = $2, state = $3, completed_at = $4, updated_at = $5 
                        WHERE job_id = $1
                        """,
                        job_id,
                        json.dumps(result),
                        JobState.COMPLETED.value,
                        datetime.utcnow(),
                        datetime.utcnow()
                    )
            else:
                job_file = self.fallback_storage / f"{job_id}.json"
                if job_file.exists():
                    async with aiofiles.open(job_file, 'r') as f:
                        data = json.loads(await f.read())
                    
                    data["result"] = result
                    data["state"] = JobState.COMPLETED.value
                    data["completed_at"] = datetime.utcnow().isoformat()
                    data["updated_at"] = datetime.utcnow().isoformat()
                    
                    async with aiofiles.open(job_file, 'w') as f:
                        await f.write(json.dumps(data, indent=2))
            
            return True
            
        except Exception as e:
            logger.exception(f"Error setting job result {job_id}: {e}")
            return False
    
    async def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job result"""
        try:
            if self.pool:
                async with self.pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT result FROM jobs WHERE job_id = $1", job_id
                    )
                    
                    if row and row["result"]:
                        return json.loads(row["result"])
            else:
                job_file = self.fallback_storage / f"{job_id}.json"
                if job_file.exists():
                    async with aiofiles.open(job_file, 'r') as f:
                        data = json.loads(await f.read())
                    
                    return data.get("result")
            
            return None
            
        except Exception as e:
            logger.exception(f"Error getting job result {job_id}: {e}")
            return None
    
    async def list_jobs(
        self,
        state: Optional[JobState] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[JobStatus]:
        """List jobs with optional filtering"""
        try:
            if self.pool:
                return await self._list_jobs_db(state, limit, offset)
            else:
                return await self._list_jobs_file(state, limit, offset)
                
        except Exception as e:
            logger.exception(f"Error listing jobs: {e}")
            return []
    
    async def _list_jobs_db(
        self,
        state: Optional[JobState],
        limit: int,
        offset: int
    ) -> List[JobStatus]:
        """List jobs from database"""
        async with self.pool.acquire() as conn:
            if state:
                rows = await conn.fetch(
                    """
                    SELECT * FROM jobs 
                    WHERE state = $1 
                    ORDER BY created_at DESC 
                    LIMIT $2 OFFSET $3
                    """,
                    state.value, limit, offset
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM jobs 
                    ORDER BY created_at DESC 
                    LIMIT $1 OFFSET $2
                    """,
                    limit, offset
                )
            
            jobs = []
            for row in rows:
                jobs.append(JobStatus(
                    job_id=row["job_id"],
                    state=JobState(row["state"]),
                    progress=row["progress"],
                    status_message=row["status_message"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    completed_at=row["completed_at"],
                    error_details=json.loads(row["error_details"]) if row["error_details"] else None
                ))
            
            return jobs
    
    async def _list_jobs_file(
        self,
        state: Optional[JobState],
        limit: int,
        offset: int
    ) -> List[JobStatus]:
        """List jobs from file storage"""
        jobs = []
        
        # Get all job files
        job_files = list(self.fallback_storage.glob("*.json"))
        job_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for job_file in job_files[offset:]:
            try:
                async with aiofiles.open(job_file, 'r') as f:
                    data = json.loads(await f.read())
                
                job_state = JobState(data["state"])
                
                # Filter by state if specified
                if state and job_state != state:
                    continue
                
                jobs.append(JobStatus(
                    job_id=data["job_id"],
                    state=job_state,
                    progress=data["progress"],
                    status_message=data["status_message"],
                    created_at=datetime.fromisoformat(data["created_at"]),
                    updated_at=datetime.fromisoformat(data["updated_at"]),
                    completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
                    error_details=data.get("error_details")
                ))
                
                if len(jobs) >= limit:
                    break
                    
            except Exception as e:
                logger.warning(f"Error reading job file {job_file}: {e}")
                continue
        
        return jobs
    
    async def cleanup_expired_jobs(self) -> int:
        """Clean up expired jobs"""
        try:
            if self.pool:
                return await self._cleanup_expired_jobs_db()
            else:
                return await self._cleanup_expired_jobs_file()
                
        except Exception as e:
            logger.exception(f"Error cleaning up expired jobs: {e}")
            return 0
    
    async def _cleanup_expired_jobs_db(self) -> int:
        """Clean up expired jobs from database"""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM jobs WHERE expires_at < $1",
                datetime.utcnow()
            )
            
            # Extract count from result string like "DELETE 5"
            deleted_count = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired jobs")
            
            return deleted_count
    
    async def _cleanup_expired_jobs_file(self) -> int:
        """Clean up expired jobs from file storage"""
        deleted_count = 0
        current_time = datetime.utcnow()
        
        job_files = list(self.fallback_storage.glob("*.json"))
        
        for job_file in job_files:
            try:
                async with aiofiles.open(job_file, 'r') as f:
                    data = json.loads(await f.read())
                
                expires_at = datetime.fromisoformat(data["expires_at"])
                
                if expires_at < current_time:
                    job_file.unlink()  # Delete file
                    deleted_count += 1
                    
            except Exception as e:
                logger.warning(f"Error processing job file {job_file}: {e}")
                continue
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} expired job files")
        
        return deleted_count
    
    def update_job_sync(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Synchronous version of update_job for Celery tasks"""
        try:
            import asyncio
            import threading
            
            # Check if we're already in an event loop (async context)
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, need to run in thread pool
                def sync_update():
                    # Create new event loop for this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self.update_job(job_id, updates))
                    finally:
                        new_loop.close()
                
                # Run in thread pool to avoid blocking current event loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(sync_update)
                    return future.result(timeout=30)  # 30 second timeout
                    
            except RuntimeError:
                # No event loop running, we can run async directly
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.update_job(job_id, updates))
                finally:
                    loop.close()
                    
        except Exception as e:
            logger.exception(f"Error in sync update_job for {job_id}: {e}")
            return False
    
    def get_job_sync(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous version of get_job for Celery tasks"""
        try:
            import asyncio
            
            # Check if we're already in an event loop (async context)
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, need to run in thread pool
                def sync_get():
                    # Create new event loop for this thread
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        job_status = new_loop.run_until_complete(self.get_job(job_id))
                        if job_status:
                            return {
                                "job_id": job_status.job_id,
                                "status": job_status.status.value,
                                "progress": job_status.progress,
                                "current_stage": job_status.current_stage,
                                "error_msg": job_status.error_msg,
                                "created_at": job_status.created_at,
                                "updated_at": job_status.updated_at
                            }
                        return None
                    finally:
                        new_loop.close()
                
                # Run in thread pool to avoid blocking current event loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(sync_get)
                    return future.result(timeout=30)  # 30 second timeout
                    
            except RuntimeError:
                # No event loop running, we can run async directly
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    job_status = loop.run_until_complete(self.get_job(job_id))
                    if job_status:
                        return {
                            "job_id": job_status.job_id,
                            "status": job_status.status.value,
                            "progress": job_status.progress,
                            "current_stage": job_status.current_stage,
                            "error_msg": job_status.error_msg,
                            "created_at": job_status.created_at,
                            "updated_at": job_status.updated_at
                        }
                    return None
                finally:
                    loop.close()
                    
        except Exception as e:
            logger.exception(f"Error in sync get_job for {job_id}: {e}")
            return None
