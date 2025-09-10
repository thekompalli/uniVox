"""
Common Schemas
Shared data models and validation schemas
"""
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class StatusEnum(str, Enum):
    """Generic status enumeration"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PriorityEnum(str, Enum):
    """Priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class QualityModeEnum(str, Enum):
    """Processing quality modes"""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH = "high"
    ULTRA = "ultra"


class LanguageEnum(str, Enum):
    """Supported languages for PS-06"""
    ENGLISH = "english"
    HINDI = "hindi"
    PUNJABI = "punjabi"
    BENGALI = "bengali"
    NEPALI = "nepali"
    DOGRI = "dogri"


class BaseTimestampedModel(BaseModel):
    """Base model with timestamps"""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    size: int = Field(default=20, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$", description="Sort order")


class FilterParams(BaseModel):
    """Generic filter parameters"""
    status: Optional[StatusEnum] = None
    priority: Optional[PriorityEnum] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None


class ValidationError(BaseModel):
    """Validation error detail"""
    field: str = Field(description="Field name that failed validation")
    message: str = Field(description="Validation error message")
    code: str = Field(description="Error code")
    value: Optional[Any] = Field(None, description="Invalid value")


class Coordinates(BaseModel):
    """2D coordinates"""
    x: float = Field(description="X coordinate")
    y: float = Field(description="Y coordinate")


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    top_left: Coordinates = Field(description="Top-left corner")
    bottom_right: Coordinates = Field(description="Bottom-right corner")
    
    @validator('bottom_right')
    def validate_coordinates(cls, v, values):
        if 'top_left' in values:
            if v.x <= values['top_left'].x or v.y <= values['top_left'].y:
                raise ValueError("Bottom-right must be greater than top-left coordinates")
        return v


class TimeRange(BaseModel):
    """Time range with start and end"""
    start: float = Field(ge=0, description="Start time in seconds")
    end: float = Field(gt=0, description="End time in seconds")
    
    @validator('end')
    def validate_time_range(cls, v, values):
        if 'start' in values and v <= values['start']:
            raise ValueError("End time must be greater than start time")
        return v


class FileMetadata(BaseModel):
    """File metadata information"""
    filename: str = Field(description="Original filename")
    size: int = Field(ge=0, description="File size in bytes")
    mime_type: str = Field(description="MIME type")
    checksum: Optional[str] = Field(None, description="File checksum (MD5)")
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)


class PerformanceMetrics(BaseModel):
    """Performance metrics"""
    processing_time: float = Field(ge=0, description="Processing time in seconds")
    memory_usage: Optional[float] = Field(None, ge=0, description="Memory usage in MB")
    cpu_usage: Optional[float] = Field(None, ge=0, le=100, description="CPU usage percentage")
    gpu_usage: Optional[float] = Field(None, ge=0, le=100, description="GPU usage percentage")
    throughput: Optional[float] = Field(None, ge=0, description="Items processed per second")


class SystemInfo(BaseModel):
    """System information"""
    platform: str = Field(description="Platform name")
    python_version: str = Field(description="Python version")
    total_memory: float = Field(description="Total system memory in GB")
    available_memory: float = Field(description="Available memory in GB")
    cpu_count: int = Field(description="Number of CPU cores")
    gpu_count: int = Field(default=0, description="Number of GPUs")


class ModelInfo(BaseModel):
    """Model information"""
    name: str = Field(description="Model name")
    version: str = Field(description="Model version")
    type: str = Field(description="Model type")
    size: Optional[float] = Field(None, description="Model size in MB")
    accuracy: Optional[float] = Field(None, ge=0, le=1, description="Model accuracy")
    supported_languages: List[LanguageEnum] = Field(default_factory=list)
    requires_gpu: bool = Field(default=False)


class CompetitionStage(str, Enum):
    """Competition stages"""
    STAGE_1 = "stage_1"
    STAGE_2 = "stage_2"
    STAGE_3 = "stage_3"


class EvaluationMetrics(BaseModel):
    """Competition evaluation metrics"""
    speaker_identification_accuracy: Optional[float] = Field(None, ge=0, le=1)
    diarization_error_rate: Optional[float] = Field(None, ge=0)
    language_identification_accuracy: Optional[float] = Field(None, ge=0, le=1)
    word_error_rate: Optional[float] = Field(None, ge=0)
    bleu_score: Optional[float] = Field(None, ge=0)
    real_time_factor: Optional[float] = Field(None, ge=0)


class ResourceUsage(BaseModel):
    """Resource usage information"""
    cpu_cores: int = Field(description="Number of CPU cores used")
    memory_gb: float = Field(description="Memory usage in GB")
    gpu_memory_gb: Optional[float] = Field(None, description="GPU memory usage in GB")
    storage_gb: float = Field(description="Storage usage in GB")
    processing_time_seconds: float = Field(description="Total processing time")


class HealthStatus(BaseModel):
    """Health status information"""
    service_name: str = Field(description="Service name")
    status: str = Field(description="Health status")
    last_check: datetime = Field(description="Last health check time")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")


class LogEntry(BaseModel):
    """Log entry model"""
    timestamp: datetime = Field(description="Log timestamp")
    level: str = Field(description="Log level")
    logger: str = Field(description="Logger name")
    message: str = Field(description="Log message")
    extra: Optional[Dict[str, Any]] = Field(None, description="Extra log data")


class CacheEntry(BaseModel):
    """Cache entry model"""
    key: str = Field(description="Cache key")
    value: Any = Field(description="Cached value")
    ttl: int = Field(description="Time to live in seconds")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(description="Expiration time")


class TaskProgress(BaseModel):
    """Task progress information"""
    task_id: str = Field(description="Task identifier")
    progress: float = Field(ge=0, le=1, description="Progress percentage (0-1)")
    current_step: str = Field(description="Current processing step")
    total_steps: int = Field(description="Total number of steps")
    completed_steps: int = Field(description="Number of completed steps")
    estimated_remaining: Optional[float] = Field(None, description="Estimated remaining time in seconds")


class NotificationSettings(BaseModel):
    """Notification settings"""
    email_enabled: bool = Field(default=False)
    email_address: Optional[str] = Field(None)
    webhook_enabled: bool = Field(default=False)
    webhook_url: Optional[str] = Field(None)
    notification_types: List[str] = Field(default_factory=list)


# Common response wrappers
class SuccessResponse(BaseModel):
    """Success response wrapper"""
    success: bool = Field(default=True)
    message: str = Field(description="Success message")
    data: Optional[Any] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response wrapper"""
    success: bool = Field(default=False)
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request identifier")


class ListResponse(BaseModel):
    """List response with pagination"""
    items: List[Any] = Field(description="List items")
    total: int = Field(description="Total number of items")
    page: int = Field(description="Current page")
    size: int = Field(description="Items per page")
    pages: int = Field(description="Total pages")
    has_next: bool = Field(description="Has next page")
    has_prev: bool = Field(description="Has previous page")