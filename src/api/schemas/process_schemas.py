"""
API Schemas for PS-06 Competition System
Pydantic models for request/response validation
"""
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import uuid


class JobState(str, Enum):
    """Job processing states"""
    QUEUED = "QUEUED"
    PREPROCESSING = "PREPROCESSING"
    VAD_PROCESSING = "VAD_PROCESSING"
    DIARIZATION = "DIARIZATION"
    LANGUAGE_ID = "LANGUAGE_ID"
    TRANSCRIPTION = "TRANSCRIPTION"
    TRANSLATION = "TRANSLATION"
    POSTPROCESSING = "POSTPROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class AudioSpecs(BaseModel):
    """Audio file specifications"""
    sample_rate: int = Field(ge=8000, le=48000, description="Sample rate in Hz")
    channels: int = Field(ge=1, le=2, description="Number of audio channels")
    duration: float = Field(gt=0, description="Duration in seconds")
    format: str = Field(description="Audio format")
    bit_depth: int = Field(ge=8, le=32, description="Bit depth")
    file_size: int = Field(gt=0, description="File size in bytes")


class ProcessRequest(BaseModel):
    """Audio processing request"""
    languages: List[str] = Field(
        default=["english", "hindi", "punjabi"],
        description="Expected languages in the audio"
    )
    speaker_gallery: Optional[List[str]] = Field(
        default=None,
        description="List of known speaker IDs for identification"
    )
    quality_mode: str = Field(
        default="balanced",
        pattern="^(fast|balanced|high)$",
        description="Processing quality mode"
    )
    enable_overlaps: bool = Field(
        default=True,
        description="Enable overlap detection in diarization"
    )
    min_segment_duration: float = Field(
        default=0.5,
        ge=0.1,
        le=10.0,
        description="Minimum segment duration in seconds"
    )
    translate: bool = Field(
        default=True,
        description="Run translation stage (true) or skip (false)"
    )
    num_speakers: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Optional hint for diarization number of speakers"
    )
    
    @validator('languages')
    def validate_languages(cls, v):
        supported = ["english", "hindi", "punjabi", "bengali", "nepali", "dogri"]
        for lang in v:
            if lang not in supported:
                raise ValueError(f"Unsupported language: {lang}")
        return v


class SegmentData(BaseModel):
    """Individual segment information"""
    start: float = Field(ge=0, description="Start time in seconds")
    end: float = Field(gt=0, description="End time in seconds")
    speaker: Optional[str] = Field(None, description="Speaker identifier")
    language: Optional[str] = Field(None, description="Detected language")
    text: Optional[str] = Field(None, description="Transcribed text")
    translation: Optional[str] = Field(None, description="English translation")
    confidence: float = Field(ge=0, le=1, description="Confidence score")
    
    @validator('end')
    def end_after_start(cls, v, values):
        if 'start' in values and v <= values['start']:
            raise ValueError('End time must be after start time')
        return v


class JobStatus(BaseModel):
    """Job status response"""
    job_id: str = Field(description="Unique job identifier")
    status: JobState = Field(description="Current processing status")
    progress: float = Field(ge=0, le=1, description="Processing progress (0-1)")
    current_stage: Optional[str] = Field(None, description="Current processing stage")
    error_msg: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(description="Job creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    estimated_completion: Optional[datetime] = Field(
        None, description="Estimated completion time"
    )
    processing_time: Optional[float] = Field(
        None, description="Total processing time in seconds"
    )


class ProcessResult(BaseModel):
    """Complete processing result"""
    job_id: str = Field(description="Job identifier")
    audio_specs: AudioSpecs = Field(description="Input audio specifications")
    processing_time: float = Field(description="Total processing time in seconds")
    
    # Competition output file paths
    sid_csv: str = Field(description="Speaker identification CSV file path")
    sd_csv: str = Field(description="Speaker diarization CSV file path") 
    lid_csv: str = Field(description="Language identification CSV file path")
    asr_trn: str = Field(description="ASR transcription TRN file path")
    nmt_txt: str = Field(description="Translation TXT file path")
    
    # Processing results
    segments: List[SegmentData] = Field(description="Processed audio segments")
    speakers_detected: int = Field(description="Number of unique speakers")
    languages_detected: List[str] = Field(description="Detected languages")
    
    # Performance metrics
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Performance and quality metrics"
    )


class HealthResponse(BaseModel):
    """System health response"""
    status: str = Field(description="Overall system status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(description="System version")
    services: Dict[str, str] = Field(description="Service status")
    models: Dict[str, str] = Field(description="Model status")
    system_info: Dict[str, Any] = Field(description="System information")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    job_id: Optional[str] = Field(None, description="Related job ID if applicable")


class CompetitionOutput(BaseModel):
    """Competition-specific output format"""
    evaluation_id: str = Field(description="Competition evaluation ID")
    audio_files: List[str] = Field(description="Processed audio files")
    
    # Required output files for competition
    sid_results: str = Field(description="SID_XX.csv file content")
    sd_results: str = Field(description="SD_XX.csv file content")
    lid_results: str = Field(description="LID_XX.csv file content")
    asr_results: str = Field(description="ASR_XX.trn file content")
    nmt_results: str = Field(description="NMT_XX.txt file content")
    
    # Solution hash for verification
    solution_hash: str = Field(description="MD5 hash of solution")
    
    # Performance summary
    performance_summary: Dict[str, float] = Field(
        description="Performance metrics summary"
    )


class BatchProcessRequest(BaseModel):
    """Batch processing request for multiple files"""
    files: List[str] = Field(description="List of audio file paths")
    common_settings: ProcessRequest = Field(description="Common processing settings")
    priority: int = Field(default=0, description="Processing priority")


class BatchProcessResult(BaseModel):
    """Batch processing result"""
    batch_id: str = Field(description="Batch identifier")
    total_files: int = Field(description="Total number of files")
    completed: int = Field(description="Number of completed files")
    failed: int = Field(description="Number of failed files")
    results: List[ProcessResult] = Field(description="Individual file results")
    batch_metrics: Dict[str, Any] = Field(description="Batch-level metrics")


# Request/Response wrapper models
class APIResponse(BaseModel):
    """Generic API response wrapper"""
    success: bool = Field(description="Request success status")
    data: Optional[Any] = Field(None, description="Response data")
    error: Optional[ErrorResponse] = Field(None, description="Error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginatedResponse(BaseModel):
    """Paginated response model"""
    items: List[Any] = Field(description="Response items")
    total: int = Field(description="Total number of items")
    page: int = Field(description="Current page number")
    size: int = Field(description="Items per page")
    pages: int = Field(description="Total number of pages")
