#!/usr/bin/env python3
"""
Local setup script for PS-06 system
Simplified setup without external dependencies
"""
import os
import sys
import subprocess
from pathlib import Path

def install_basic_requirements():
    """Install only essential packages for local testing"""
    basic_packages = [
        'fastapi==0.104.1',
        'uvicorn[standard]==0.24.0',
        'pydantic==2.5.0',
        'numpy==1.24.3',
        'soundfile==0.12.1',
        'python-multipart==0.0.6',
        'aiofiles==23.2.1',
        'requests==2.31.0'
    ]
    
    print("📦 Installing basic packages for local testing...")
    for package in basic_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
    
    # Optional ML packages (will install if possible)
    optional_packages = [
        'torch==2.1.0',
        'transformers==4.35.2',
        'openai-whisper==20231117'
    ]
    
    print("\n🔬 Installing optional ML packages...")
    for package in optional_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install {package} (optional)")

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'logs', 'temp', 'models']
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 Created directory: {dir_name}")

def create_mock_services():
    """Create mock versions of complex services for local testing"""
    
    # Create a simplified orchestrator service
    mock_orchestrator_content = '''"""
Mock Orchestrator Service for Local Testing
Simplified version without external dependencies
"""
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import json
import numpy as np
import soundfile as sf
from pathlib import Path

class MockOrchestratorService:
    """Simplified orchestrator for local testing"""
    
    def __init__(self):
        self.jobs = {}
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
    
    async def create_job(self, audio_file, request) -> Dict[str, Any]:
        """Create a mock processing job"""
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        audio_data = await audio_file.read()
        audio_path = self.data_dir / f"{job_id}.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_data)
        
        # Create job record
        job = {
            "job_id": job_id,
            "status": "completed",  # Mock as completed immediately
            "progress": 1.0,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "audio_path": str(audio_path),
            "request_params": request.dict() if hasattr(request, 'dict') else dict(request)
        }
        
        self.jobs[job_id] = job
        
        # Create mock results
        await self._create_mock_results(job_id, audio_path)
        
        return {
            "job_id": job_id,
            "status": "completed",
            "progress": 1.0,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        return self.jobs.get(job_id)
    
    async def get_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get processing results"""
        if job_id not in self.jobs:
            return None
        
        results_path = self.data_dir / f"{job_id}_results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                return json.load(f)
        
        return None
    
    async def _create_mock_results(self, job_id: str, audio_path: Path):
        """Create mock processing results"""
        try:
            # Try to read audio file for basic info
            try:
                audio_data, sample_rate = sf.read(audio_path)
                duration = len(audio_data) / sample_rate
                num_samples = len(audio_data)
            except:
                duration = 10.0  # Default duration
                sample_rate = 16000
                num_samples = 160000
            
            # Mock results
            mock_results = {
                "job_id": job_id,
                "audio_specs": {
                    "duration": duration,
                    "sample_rate": sample_rate,
                    "channels": 1,
                    "format": "wav"
                },
                "diarization": {
                    "segments": [
                        {
                            "start": 0.0,
                            "end": duration * 0.6,
                            "speaker": "speaker_1",
                            "confidence": 0.9
                        },
                        {
                            "start": duration * 0.4,
                            "end": duration,
                            "speaker": "speaker_2", 
                            "confidence": 0.85
                        }
                    ],
                    "num_speakers": 2
                },
                "speaker_identification": {
                    "segments": [
                        {
                            "start": 0.0,
                            "end": duration * 0.6,
                            "speaker_id": "speaker_1",
                            "confidence": 0.8
                        },
                        {
                            "start": duration * 0.4,
                            "end": duration,
                            "speaker_id": "speaker_2",
                            "confidence": 0.75
                        }
                    ]
                },
                "language_identification": {
                    "segments": [
                        {
                            "start": 0.0,
                            "end": duration,
                            "language": "english",
                            "confidence": 0.9
                        }
                    ],
                    "languages_detected": ["english"]
                },
                "transcription": {
                    "segments": [
                        {
                            "start": 0.0,
                            "end": duration * 0.6,
                            "speaker": "speaker_1",
                            "language": "english",
                            "text": "Hello, this is a sample transcription for speaker one.",
                            "confidence": 0.85
                        },
                        {
                            "start": duration * 0.4,
                            "end": duration,
                            "speaker": "speaker_2",
                            "language": "english", 
                            "text": "And this is speaker two responding to the conversation.",
                            "confidence": 0.8
                        }
                    ]
                },
                "translation": {
                    "segments": [
                        {
                            "start": 0.0,
                            "end": duration * 0.6,
                            "speaker": "speaker_1",
                            "source_text": "Hello, this is a sample transcription for speaker one.",
                            "translated_text": "Hello, this is a sample transcription for speaker one.",
                            "source_language": "english",
                            "target_language": "english",
                            "confidence": 1.0
                        },
                        {
                            "start": duration * 0.4,
                            "end": duration,
                            "speaker": "speaker_2",
                            "source_text": "And this is speaker two responding to the conversation.",
                            "translated_text": "And this is speaker two responding to the conversation.",
                            "source_language": "english",
                            "target_language": "english",
                            "confidence": 1.0
                        }
                    ]
                },
                "processing_time": 2.5,
                "quality_metrics": {
                    "speaker_identification_accuracy": 0.82,
                    "diarization_error_rate": 0.15,
                    "word_error_rate": 0.12,
                    "bleu_score": 0.85
                }
            }
            
            # Save results
            results_path = self.data_dir / f"{job_id}_results.json"
            with open(results_path, 'w') as f:
                json.dump(mock_results, f, indent=2)
                
        except Exception as e:
            print(f"Error creating mock results: {e}")
'''
    
    # Write mock service file
    mock_service_path = Path("src/services/mock_orchestrator_service.py")
    mock_service_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mock_service_path, 'w') as f:
        f.write(mock_orchestrator_content)
    
    print("🎭 Created mock orchestrator service")

def main():
    print("🏠 PS-06 Local Setup")
    print("=" * 30)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return
    
    print(f"✅ Python {sys.version}")
    
    # Install packages
    install_basic_requirements()
    
    # Create directories
    create_directories()
    
    # Create mock services
    create_mock_services()
    
    print("\n🎉 Local setup complete!")
    print("\nNext steps:")
    print("1. Run: python run_local.py")
    print("2. Open browser: http://localhost:8000/docs")
    print("3. Test with Postman: POST http://localhost:8000/api/v1/process")

if __name__ == "__main__":
    main()