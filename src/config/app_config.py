"""
Application Configuration
Main configuration for PS-06 competition system
"""
import os
from typing import List, Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import validator
from pathlib import Path


class AppConfig(BaseSettings):
    """Main application configuration"""
    
    # Application Settings
    app_name: str = "PS-06 Competition System"
    version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    
    # Database Settings
    database_url: str = "postgresql://ps06:password@localhost:5432/ps06_db"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Redis Settings
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    
    # Celery Settings
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    celery_task_serializer: str = "json"
    celery_result_serializer: str = "json"
    
    # Storage Settings
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_secure: bool = False
    
    # Storage Buckets
    audio_bucket: str = "ps06-audio"
    results_bucket: str = "ps06-results"
    models_bucket: str = "ps06-models"
    
    # File Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    models_dir: Path = base_dir / "models"
    data_dir: Path = base_dir / "data"
    logs_dir: Path = base_dir / "logs"
    temp_dir: Path = base_dir / "temp"
    
    # Audio Processing Settings
    max_file_size: int = 500 * 1024 * 1024  # 500MB
    supported_formats: List[str] = ["wav", "mp3", "ogg", "flac", "m4a"]
    target_sample_rate: int = 16000
    target_channels: int = 1
    chunk_duration: int = 30  # seconds
    
    # Model Settings
    triton_url: str = "localhost:8001"
    use_triton: bool = False
    device: str = "cuda"
    batch_size: int = 4
    
    # Competition Languages
    stage1_languages: List[str] = ["english", "hindi", "punjabi"]
    stage2_languages: List[str] = ["english", "hindi", "punjabi", "bengali", "nepali", "dogri"]
    
    # Processing Thresholds
    vad_threshold: float = 0.5
    speaker_threshold: float = 0.75
    language_confidence_threshold: float = 0.8
    min_segment_duration: float = 0.5  # seconds
    
    # Performance Settings
    max_concurrent_jobs: int = 5
    job_timeout: int = 3600  # 1 hour
    cleanup_interval: int = 86400  # 24 hours

    # Developer/Testing Aids
    # Limit ASR to first N seconds for quick functional tests (None disables)
    asr_quick_test_seconds: Optional[int] = None
    
    # Competition Settings
    competition_weights: Dict[str, float] = {
        "speaker_identification": 0.15,
        "speaker_diarization": 0.20,
        "language_identification": 0.20,
        "speech_recognition": 0.30,
        "machine_translation": 0.15
    }
    
    @validator('models_dir', 'data_dir', 'logs_dir', 'temp_dir')
    def create_directories(cls, v):
        """Create directories if they don't exist"""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"
        # Avoid picking up unrelated global env vars like DEBUG from the OS
        env_prefix = "PS06_"


class ModelConfig(BaseSettings):
    """Model-specific configuration"""
    
    # Whisper Configuration
    whisper_model: str = "large-v3"
    whisper_language: Optional[str] = None
    whisper_task: str = "transcribe"
    whisper_beam_size: int = 5
    whisper_best_of: int = 5
    whisper_temperature: float = 0.0
    
    # pyannote Configuration
    pyannote_model: str = "pyannote/speaker-diarization-3.1"
    pyannote_clustering_method: str = "centroid"
    pyannote_min_cluster_size: int = 12
    pyannote_threshold: float = 0.7045654963945799
    
    # WeSpeaker Configuration
    wespeaker_model: str = "voxceleb_resnet34"
    wespeaker_embedding_dim: int = 256
    wespeaker_similarity_metric: str = "cosine"
    
    # SpeechBrain Configuration
    speechbrain_model: str = "speechbrain/lang-id-voxlingua107-ecapa"
    speechbrain_threshold: float = 0.8
    
    # IndicTrans Configuration
    indictrans_model: str = "ai4bharat/indictrans2-en-indic-1B"
    indictrans_batch_size: int = 8
    indictrans_beam_size: int = 4
    
    # NLLB Configuration
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    nllb_max_length: int = 400
    
    # Silero VAD Configuration
    silero_model: str = "silero_vad"
    silero_threshold: float = 0.5
    silero_min_speech_duration: float = 0.1
    silero_max_speech_duration: float = float('inf')
    
    class Config:
        env_file = ".env"
        extra = "ignore"
        env_prefix = "PS06_"


class LoggingConfig:
    """Logging configuration"""
    
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_LEVEL = "INFO"
    LOG_FILE = "ps06_system.log"
    
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": LOG_FORMAT,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "formatter": "default",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": LOG_FILE,
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
        },
        # Explicit loggers so uvicorn/fastapi access lines show up in console
        "loggers": {
            # Uvicorn core logger
            "uvicorn": {
                "handlers": ["default", "file"],
                "level": LOG_LEVEL,
                "propagate": False,
            },
            # Uvicorn error logger (application errors)
            "uvicorn.error": {
                "handlers": ["default", "file"],
                "level": LOG_LEVEL,
                "propagate": False,
            },
            # Uvicorn access logger (HTTP GET/POST lines)
            "uvicorn.access": {
                "handlers": ["default"],
                "level": LOG_LEVEL,
                "propagate": False,
            },
            # FastAPI logger (optional)
            "fastapi": {
                "handlers": ["default", "file"],
                "level": LOG_LEVEL,
                "propagate": False,
            },
        },
        "root": {
            "level": LOG_LEVEL,
            "handlers": ["default", "file"],
        },
    }


# Global configuration instances
app_config = AppConfig()
model_config = ModelConfig()
logging_config = LoggingConfig()
