"""
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
