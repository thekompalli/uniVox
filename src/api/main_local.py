"""
Local FastAPI Main Application for PS-06 System
Simplified version for local testing without external dependencies
"""
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import uuid
from datetime import datetime
from typing import Optional, List
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import mock services
try:
    from src.services.mock_orchestrator_service import MockOrchestratorService
except ImportError:
    print("Mock services not found. Run local_setup.py first.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="PS-06 Competition System (Local)",
    version="1.0.0-local",
    description="Language Agnostic Speaker Identification & Diarization System - Local Testing Version",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize mock orchestrator
orchestrator = MockOrchestratorService()

# Pydantic models for requests
class ProcessRequest:
    def __init__(self, languages: str = "english", quality_mode: str = "balanced"):
        self.languages = languages.split(',') if languages else ["english"]
        self.quality_mode = quality_mode
    
    def dict(self):
        return {
            "languages": self.languages,
            "quality_mode": self.quality_mode
        }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "PS-06 Competition System (Local)",
        "version": "1.0.0-local",
        "status": "running",
        "docs_url": "/docs",
        "description": "Local testing version"
    }

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0-local",
        "services": {
            "api": "running",
            "orchestrator": "running",
            "storage": "local"
        }
    }

@app.post("/api/v1/process")
async def process_audio(
    audio_file: UploadFile = File(...),
    languages: str = Form("english"),
    quality_mode: str = Form("balanced")
):
    """
    Process audio file through the complete pipeline
    """
    try:
        # Validate file
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file extension
        allowed_extensions = ['wav', 'mp3', 'ogg', 'flac', 'm4a']
        file_extension = audio_file.filename.split('.')[-1].lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported: {', '.join(allowed_extensions)}"
            )
        
        # Check file size (100MB limit for local testing)
        max_size = 100 * 1024 * 1024  # 100MB
        content = await audio_file.read()
        if len(content) > max_size:
            raise HTTPException(status_code=400, detail="File too large. Max size: 100MB")
        
        # Reset file pointer
        await audio_file.seek(0)
        
        # Create processing request
        request = ProcessRequest(languages=languages, quality_mode=quality_mode)
        
        # Process with mock orchestrator
        logger.info(f"Processing audio file: {audio_file.filename}")
        job_status = await orchestrator.create_job(audio_file, request)
        
        return {
            "success": True,
            "message": "Audio processing completed",
            "data": {
                "job_id": job_status["job_id"],
                "status": job_status["status"],
                "progress": job_status["progress"],
                "created_at": job_status["created_at"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/v1/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Get processing job status
    """
    try:
        status = await orchestrator.get_job_status(job_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "job_id": job_id,
            "status": status["status"],
            "progress": status["progress"],
            "created_at": status["created_at"],
            "updated_at": status["updated_at"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting job status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@app.get("/api/v1/result/{job_id}")
async def get_result(job_id: str):
    """
    Get processing results
    """
    try:
        # Check if job exists
        job_status = await orchestrator.get_job_status(job_id)
        if not job_status:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job_status["status"] != "completed":
            raise HTTPException(status_code=400, detail="Job not completed yet")
        
        # Get results
        results = await orchestrator.get_result(job_id)
        if not results:
            raise HTTPException(status_code=404, detail="Results not found")
        
        return {
            "success": True,
            "job_id": job_id,
            "data": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting results: {e}")
        raise HTTPException(status_code=500, detail=f"Result retrieval failed: {str(e)}")

@app.get("/api/v1/jobs")
async def list_jobs(limit: int = 10, offset: int = 0):
    """
    List processing jobs
    """
    try:
        # For local testing, return all jobs
        all_jobs = list(orchestrator.jobs.values())
        
        # Apply pagination
        paginated_jobs = all_jobs[offset:offset + limit]
        
        return {
            "jobs": paginated_jobs,
            "total": len(all_jobs),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.exception(f"Error listing jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Job listing failed: {str(e)}")

@app.delete("/api/v1/job/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its results
    """
    try:
        if job_id not in orchestrator.jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Remove job
        del orchestrator.jobs[job_id]
        
        # Clean up files
        data_dir = Path("data")
        for file_pattern in [f"{job_id}.*", f"{job_id}_*"]:
            for file_path in data_dir.glob(file_pattern):
                file_path.unlink(missing_ok=True)
        
        return {
            "success": True,
            "message": f"Job {job_id} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting job: {e}")
        raise HTTPException(status_code=500, detail=f"Job deletion failed: {str(e)}")

@app.get("/api/v1/models/info")
async def get_model_info():
    """
    Get information about loaded models
    """
    return {
        "models": {
            "asr": {
                "name": "Whisper Base (Mock)",
                "languages": ["english", "hindi", "punjabi"],
                "status": "loaded"
            },
            "diarization": {
                "name": "pyannote.audio (Mock)",
                "status": "loaded"
            },
            "speaker_id": {
                "name": "Mock Speaker ID",
                "status": "loaded"
            },
            "language_id": {
                "name": "Mock Language ID",
                "status": "loaded"
            },
            "translation": {
                "name": "Mock Translator",
                "status": "loaded"
            }
        },
        "supported_languages": ["english", "hindi", "punjabi"],
        "supported_formats": ["wav", "mp3", "ogg", "flac"]
    }

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": "HTTP_EXCEPTION",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred",
            "detail": str(exc)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)