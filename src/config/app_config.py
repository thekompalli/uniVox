"""
Application Configuration
Main configuration for PS-06 competition system
"""
import os
from typing import List, Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import validator
from pathlib import Path
from typing import ClassVar
from enum import Enum
import os
from typing import List, Dict, Any, Optional, ClassVar
from pydantic_settings import BaseSettings
from pydantic import validator, Field  # Add Field here
from pathlib import Path
from enum import Enum


class NoiseReductionMethodEnum(str, Enum):
    """Noise reduction method options"""
    AUTO = "auto"
    ADVANCED_SPECTRAL = "advanced_spectral"
    ADAPTIVE_WIENER = "adaptive_wiener"
    MULTI_BAND = "multi_band"
    HYBRID = "hybrid"
    BASIC = "basic"

class NoiseRemovalConfig(BaseSettings):
    """Configuration for noise removal functionality"""
    
    # Default noise removal settings
    default_method: NoiseReductionMethodEnum = NoiseReductionMethodEnum.AUTO
    default_strength: float = Field(default=0.8, ge=0.0, le=1.0, description="Noise reduction strength")
    preserve_speech: bool = Field(default=True, description="Prioritize speech preservation")
    auto_detect_method: bool = Field(default=True, description="Automatically detect best method")
    
    # SNR thresholds for triggering noise removal
    snr_threshold_advanced: float = Field(default=15.0, description="Use advanced methods below this SNR")
    snr_threshold_basic: float = Field(default=10.0, description="Use basic methods below this SNR")
    
    # Method selection parameters
    very_noisy_snr_threshold: float = Field(default=5.0, description="Use multi-band for very noisy audio")
    speech_heavy_ratio: float = Field(default=0.7, description="Use adaptive Wiener for speech-heavy audio")
    low_freq_noise_ratio: float = Field(default=0.4, description="Use multi-band for low-frequency noise")
    
    # Processing parameters
    stft_n_fft: int = Field(default=1024, description="STFT window size")
    stft_hop_length: int = Field(default=256, description="STFT hop length")
    noise_estimation_frames: int = Field(default=10, description="Frames for noise estimation")
    
    # Quality assessment thresholds
    excellent_snr_improvement: float = Field(default=3.0, description="SNR improvement for excellent rating")
    good_snr_improvement: float = Field(default=1.0, description="SNR improvement for good rating")
    
    class Config:
        env_prefix = "NOISE_REMOVAL_"
        case_sensitive = False

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
    
       # Audio Processing Settings (existing)
    max_file_size: int = 500 * 1024 * 1024  # 500MB
    supported_formats: List[str] = ["wav", "mp3", "ogg", "flac", "m4a"]
    target_sample_rate: int = 16000
    target_channels: int = 1
    chunk_duration: int = 30  # seconds
    
    # ADD THESE NEW NOISE REMOVAL FIELDS:
    # Noise Removal Settings
    enable_advanced_noise_removal: bool = Field(default=True, description="Enable advanced noise removal")
    default_noise_removal_method: NoiseReductionMethodEnum = Field(
        default=NoiseReductionMethodEnum.AUTO, 
        description="Default noise removal method"
    )
    noise_removal_strength: float = Field(
        default=0.8, 
        ge=0.0, 
        le=1.0, 
        description="Default noise reduction strength"
    )
    noise_removal_preserve_speech: bool = Field(
        default=True, 
        description="Prioritize speech preservation during noise removal"
    )
    noise_removal_auto_detect: bool = Field(
        default=True, 
        description="Automatically detect best noise removal method"
    )
    
    # Noise removal thresholds
    noise_removal_snr_threshold_advanced: float = Field(
        default=15.0, 
        description="SNR threshold for advanced noise removal"
    )
    noise_removal_snr_threshold_basic: float = Field(
        default=10.0, 
        description="SNR threshold for basic noise removal"
    )
    
    # Advanced noise removal parameters
    noise_removal_stft_n_fft: int = Field(default=1024, description="STFT window size for noise removal")
    noise_removal_stft_hop_length: int = Field(default=256, description="STFT hop length for noise removal")
    noise_removal_estimation_frames: int = Field(default=10, description="Frames for noise estimation")
    
    # Keep all your existing fields and methods...
    # Model Settings
    triton_url: str = "localhost:8001"
    use_triton: bool = False
    device: str = "cuda"
    batch_size: int = 4

    @property
    def noise_removal_config(self) -> Dict[str, Any]:
        """Get noise removal configuration as a dictionary"""
        return {
            'enabled': self.enable_advanced_noise_removal,
            'method': self.default_noise_removal_method.value,
            'strength': self.noise_removal_strength,
            'preserve_speech': self.noise_removal_preserve_speech,
            'auto_detect_method': self.noise_removal_auto_detect,
            'snr_threshold_advanced': self.noise_removal_snr_threshold_advanced,
            'snr_threshold_basic': self.noise_removal_snr_threshold_basic,
            'stft_n_fft': self.noise_removal_stft_n_fft,
            'stft_hop_length': self.noise_removal_stft_hop_length,
            'estimation_frames': self.noise_removal_estimation_frames
        }

    
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

    # Faster-Whisper Configuration
    whisper_backend: str = "openai"  # "openai" or "fast"
    fast_whisper_model_path: Optional[str] = None
    fast_whisper_device: str = "cuda"
    fast_whisper_compute_type: str = "float16"
    fast_whisper_vad_filter: bool = False

    # Custom Model Configuration
    use_custom_hindi_model: bool = False  # Use Hindi2Hinglish model for Hindi audio
    hindi2hinglish_model_path: Optional[str] = None
    custom_model_languages: List[str] = ["hindi", "hinglish"]  # Languages that should use custom model
    
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

    # Cache Configuration
    huggingface_cache_dir: str = "./models/huggingface"
    
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
    
class NoiseRemovalConfigHelper:
    """Helper class to provide noise removal configuration"""
    
    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
    
    @property
    def default_method(self) -> str:
        return self.app_config.default_noise_removal_method.value
    
    @property
    def default_strength(self) -> float:
        return self.app_config.noise_removal_strength
    
    @property
    def preserve_speech(self) -> bool:
        return self.app_config.noise_removal_preserve_speech
    
    @property
    def auto_detect_method(self) -> bool:
        return self.app_config.noise_removal_auto_detect
    
    @property
    def snr_threshold_advanced(self) -> float:
        return self.app_config.noise_removal_snr_threshold_advanced
    
    @property
    def snr_threshold_basic(self) -> float:
        return self.app_config.noise_removal_snr_threshold_basic    

# Global configuration instances
app_config = AppConfig()
model_config = ModelConfig()
logging_config = LoggingConfig()
# Add this line:
noise_removal_helper = NoiseRemovalConfigHelper(app_config)