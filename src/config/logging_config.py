"""
Logging Configuration
Centralized logging setup for PS-06 system
"""
import os
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional


class LoggingConfig:
    """Logging configuration for the PS-06 system"""
    
    # Base configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s")
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # File logging
    LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
    LOG_FILE = LOG_DIR / "ps06_system.log"
    ERROR_LOG_FILE = LOG_DIR / "ps06_errors.log"
    ACCESS_LOG_FILE = LOG_DIR / "ps06_access.log"
    
    # Rotation settings
    MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(50 * 1024 * 1024)))  # 50MB
    BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "10"))
    
    # Performance logging
    SLOW_QUERY_THRESHOLD = float(os.getenv("SLOW_QUERY_THRESHOLD", "1.0"))
    LOG_SQL_QUERIES = os.getenv("LOG_SQL_QUERIES", "false").lower() == "true"
    
    @classmethod
    def setup_directories(cls):
        """Create log directories if they don't exist"""
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_logging_config(cls) -> Dict[str, Any]:
        """Get complete logging configuration dictionary"""
        cls.setup_directories()
        
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": cls.LOG_FORMAT,
                    "datefmt": cls.LOG_DATE_FORMAT,
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s",
                    "datefmt": cls.LOG_DATE_FORMAT,
                },
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(filename)s %(lineno)d %(funcName)s %(message)s",
                },
                "access": {
                    "format": "%(asctime)s - %(remote_addr)s - %(method)s %(url)s - %(status_code)s - %(process_time).3fs",
                    "datefmt": cls.LOG_DATE_FORMAT,
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": cls.LOG_LEVEL,
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "detailed",
                    "filename": str(cls.LOG_FILE),
                    "maxBytes": cls.MAX_BYTES,
                    "backupCount": cls.BACKUP_COUNT,
                    "encoding": "utf8",
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": "detailed",
                    "filename": str(cls.ERROR_LOG_FILE),
                    "maxBytes": cls.MAX_BYTES,
                    "backupCount": cls.BACKUP_COUNT,
                    "encoding": "utf8",
                },
                "access_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "access",
                    "filename": str(cls.ACCESS_LOG_FILE),
                    "maxBytes": cls.MAX_BYTES,
                    "backupCount": cls.BACKUP_COUNT,
                    "encoding": "utf8",
                },
                "json_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "json",
                    "filename": str(cls.LOG_DIR / "ps06_structured.log"),
                    "maxBytes": cls.MAX_BYTES,
                    "backupCount": cls.BACKUP_COUNT,
                    "encoding": "utf8",
                },
            },
            "loggers": {
                # Root logger
                "": {
                    "level": cls.LOG_LEVEL,
                    "handlers": ["console", "file", "error_file"],
                    "propagate": False,
                },
                # Application loggers
                "src": {
                    "level": cls.LOG_LEVEL,
                    "handlers": ["console", "file", "error_file"],
                    "propagate": False,
                },
                "src.api": {
                    "level": cls.LOG_LEVEL,
                    "handlers": ["console", "file", "access_file"],
                    "propagate": False,
                },
                "src.services": {
                    "level": cls.LOG_LEVEL,
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                "src.models": {
                    "level": cls.LOG_LEVEL,
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                "src.tasks": {
                    "level": cls.LOG_LEVEL,
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                # Third-party loggers
                "uvicorn": {
                    "level": "INFO",
                    "handlers": ["console", "access_file"],
                    "propagate": False,
                },
                "uvicorn.access": {
                    "level": "INFO",
                    "handlers": ["access_file"],
                    "propagate": False,
                },
                "fastapi": {
                    "level": "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                "celery": {
                    "level": "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                "celery.task": {
                    "level": "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                "sqlalchemy": {
                    "level": "WARNING",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                "sqlalchemy.engine": {
                    "level": "INFO" if cls.LOG_SQL_QUERIES else "WARNING",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                "transformers": {
                    "level": "WARNING",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                "torch": {
                    "level": "WARNING",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                "whisper": {
                    "level": "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
                "pyannote": {
                    "level": "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False,
                },
            },
        }
    
    @classmethod
    def configure_logging(cls):
        """Configure logging using the logging configuration"""
        import logging.config
        
        config = cls.get_logging_config()
        logging.config.dictConfig(config)
        
        # Log the configuration
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured - Level: {cls.LOG_LEVEL}, Directory: {cls.LOG_DIR}")
    
    @classmethod
    def get_performance_logger(cls) -> logging.Logger:
        """Get performance-specific logger"""
        logger = logging.getLogger("performance")
        if not logger.handlers:
            handler = logging.handlers.RotatingFileHandler(
                cls.LOG_DIR / "ps06_performance.log",
                maxBytes=cls.MAX_BYTES,
                backupCount=cls.BACKUP_COUNT
            )
            formatter = logging.Formatter(
                "%(asctime)s - %(message)s",
                datefmt=cls.LOG_DATE_FORMAT
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    @classmethod
    def get_audit_logger(cls) -> logging.Logger:
        """Get audit-specific logger"""
        logger = logging.getLogger("audit")
        if not logger.handlers:
            handler = logging.handlers.RotatingFileHandler(
                cls.LOG_DIR / "ps06_audit.log",
                maxBytes=cls.MAX_BYTES,
                backupCount=cls.BACKUP_COUNT
            )
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt=cls.LOG_DATE_FORMAT
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    @classmethod
    def log_performance_metric(
        cls, 
        operation: str, 
        duration: float, 
        success: bool = True,
        **kwargs
    ):
        """Log performance metrics"""
        logger = cls.get_performance_logger()
        
        metric_data = {
            "operation": operation,
            "duration": duration,
            "success": success,
            **kwargs
        }
        
        import json
        logger.info(json.dumps(metric_data))
    
    @classmethod
    def log_audit_event(
        cls,
        event: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        **kwargs
    ):
        """Log audit events"""
        logger = cls.get_audit_logger()
        
        audit_data = {
            "event": event,
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "timestamp": logging.Formatter().formatTime(logging.LogRecord("", 0, "", 0, "", (), None)),
            **kwargs
        }
        
        import json
        logger.info(json.dumps(audit_data))


# Global instance
logging_config = LoggingConfig()

# Convenience function
def setup_logging():
    """Setup logging configuration"""
    logging_config.configure_logging()