"""
Synchronous Job Repository for Celery Tasks
Uses psycopg2 instead of asyncpg to avoid event loop conflicts
"""
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import Dict, Any, Optional
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from src.api.schemas.process_schemas import JobState
from src.config.database_config import database_config

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class CeleryJobRepository:
    """Synchronous job repository specifically for Celery tasks"""
    
    def __init__(self):
        self.db_config = database_config
        self._connection = None
        self.data_dir = Path("data/jobs")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def get_connection(self):
        """Get or create a database connection"""
        if self._connection is None or self._connection.closed:
            try:
                # Use psycopg2 for synchronous connection
                db_url = os.getenv("DATABASE_URL", "")
                
                if db_url.startswith("postgresql://"):
                    # Parse PostgreSQL URL
                    import urllib.parse
                    result = urllib.parse.urlparse(db_url)
                    
                    self._connection = psycopg2.connect(
                        host=result.hostname,
                        port=result.port or 5432,
                        database=result.path[1:],  # Remove leading slash
                        user=result.username,
                        password=result.password,
                        cursor_factory=RealDictCursor
                    )
                    logger.info("Connected to PostgreSQL database (sync)")
                    
                else:
                    # Fallback to file-based storage
                    logger.info("Using file-based storage for Celery tasks")
                    self._connection = None
                    
            except Exception as e:
                logger.exception(f"Error connecting to database: {e}")
                self._connection = None
                
        return self._connection
        
    def close_connection(self):
        """Close database connection"""
        if self._connection and not self._connection.closed:
            self._connection.close()
            self._connection = None
            
    def update_job_sync(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Synchronously update job status"""
        try:
            conn = self.get_connection()
            
            if conn:
                # Database update
                return self._update_job_db_sync(job_id, updates)
            else:
                # File-based update
                return self._update_job_file_sync(job_id, updates)
                
        except Exception as e:
            logger.exception(f"Error in sync update_job for {job_id}: {e}")
            return False
            
    def get_job_sync(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Synchronously get job data"""
        try:
            conn = self.get_connection()
            
            if conn:
                # Database retrieval
                return self._get_job_db_sync(job_id)
            else:
                # File-based retrieval
                return self._get_job_file_sync(job_id)
                
        except Exception as e:
            logger.exception(f"Error in sync get_job for {job_id}: {e}")
            return None
            
    def _update_job_db_sync(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update job in PostgreSQL database synchronously"""
        try:
            conn = self.get_connection()
            if not conn:
                return False
                
            # Map field names to database schema
            field_mapping = {
                'status': 'state',
                'current_stage': 'status_message',
                'error_msg': 'error_details'
            }
            
            # Build UPDATE query
            set_clauses = []
            values = []
            
            for key, value in updates.items():
                # Map field names
                db_field = field_mapping.get(key, key)
                
                if key == 'updated_at' and isinstance(value, datetime):
                    set_clauses.append(f"{db_field} = %s")
                    values.append(value)
                elif key == 'status' and hasattr(value, 'value'):
                    # Convert JobState enum to value
                    set_clauses.append(f"{db_field} = %s")
                    values.append(value.value)
                elif key in ['current_stage']:
                    set_clauses.append(f"{db_field} = %s")
                    values.append(str(value))
                elif key == 'error_msg':
                    # Store error message in error_details JSON field
                    set_clauses.append(f"{db_field} = %s")
                    values.append(json.dumps({"message": str(value)}) if value else None)
                elif key in ['progress']:
                    set_clauses.append(f"{db_field} = %s")
                    values.append(float(value))
                else:
                    set_clauses.append(f"{db_field} = %s")
                    values.append(value)
                    
            if not set_clauses:
                return True
                
            query = f"""
                UPDATE jobs 
                SET {', '.join(set_clauses)}
                WHERE job_id = %s
            """
            values.append(job_id)
            
            with conn.cursor() as cursor:
                cursor.execute(query, values)
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.debug(f"Updated job {job_id} in database")
                    return True
                else:
                    logger.warning(f"Job {job_id} not found in database for update")
                    return False
                    
        except Exception as e:
            logger.exception(f"Error updating job {job_id} in database: {e}")
            if conn:
                conn.rollback()
            return False
            
    def _get_job_db_sync(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job from PostgreSQL database synchronously"""
        try:
            conn = self.get_connection()
            if not conn:
                return None
                
            query = """
                SELECT job_id, state, progress, status_message, error_details, 
                       created_at, updated_at
                FROM jobs 
                WHERE job_id = %s
            """
            
            with conn.cursor() as cursor:
                cursor.execute(query, (job_id,))
                row = cursor.fetchone()
                
                if row:
                    # Map database fields to expected format
                    job_data = dict(row)
                    
                    # Map field names back
                    mapped_data = {
                        'job_id': job_data['job_id'],
                        'status': job_data['state'],
                        'progress': job_data.get('progress', 0.0),
                        'current_stage': job_data.get('status_message', ''),
                        'created_at': job_data.get('created_at'),
                        'updated_at': job_data.get('updated_at'),
                    }
                    
                    # Extract error message from error_details JSON if present
                    error_details = job_data.get('error_details')
                    if error_details:
                        try:
                            if isinstance(error_details, str):
                                error_data = json.loads(error_details)
                            else:
                                error_data = error_details
                            mapped_data['error_msg'] = error_data.get('message', '')
                        except (json.JSONDecodeError, TypeError):
                            mapped_data['error_msg'] = str(error_details)
                    else:
                        mapped_data['error_msg'] = None
                    
                    return mapped_data
                    
        except Exception as e:
            logger.exception(f"Error getting job {job_id} from database: {e}")
            
        return None
        
    def _update_job_file_sync(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update job in file system synchronously.

        Ensure keys match file-backed schema expected by JobRepository
        (state/status_message) rather than API object field names
        (status/current_stage).
        """
        try:
            job_file = self.data_dir / f"{job_id}.json"

            # Read existing data (may have been created by async repository)
            job_data = {}
            if job_file.exists():
                with open(job_file, 'r', encoding='utf-8') as f:
                    try:
                        job_data = json.load(f)
                    except json.JSONDecodeError:
                        # Corrupt file; start fresh but keep minimal identity
                        job_data = {"job_id": job_id}

            # Map API-style fields to file schema
            mapped_updates: Dict[str, Any] = {}
            for key, value in updates.items():
                file_key = key
                if key == 'status':
                    file_key = 'state'
                    # Enum value handling
                    value = value.value if hasattr(value, 'value') else value
                elif key == 'current_stage':
                    file_key = 'status_message'
                elif key == 'error_msg':
                    file_key = 'error_details'

                # Serialize datetimes
                if isinstance(value, datetime):
                    mapped_updates[file_key] = value.isoformat()
                else:
                    mapped_updates[file_key] = value

            # Apply updates over existing data
            job_data.update(mapped_updates)

            # Preserve job_id if known
            job_data.setdefault('job_id', job_id)

            # Write back
            with open(job_file, 'w', encoding='utf-8') as f:
                json.dump(job_data, f, indent=2, default=str)

            logger.debug(f"Updated job {job_id} in file system")
            return True

        except Exception as e:
            logger.exception(f"Error updating job {job_id} in file system: {e}")
            return False
            
    def _get_job_file_sync(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job from file system synchronously"""
        try:
            job_file = self.data_dir / f"{job_id}.json"
            
            if job_file.exists():
                with open(job_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
                    
        except Exception as e:
            logger.exception(f"Error getting job {job_id} from file system: {e}")
            
        return None


# Global instance for Celery tasks to avoid connection conflicts
_celery_job_repo = None

def get_celery_job_repo() -> CeleryJobRepository:
    """Get singleton instance of CeleryJobRepository"""
    global _celery_job_repo
    if _celery_job_repo is None:
        _celery_job_repo = CeleryJobRepository()
    return _celery_job_repo
