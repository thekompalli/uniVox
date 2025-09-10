"""
Database Configuration
PostgreSQL and Redis configuration settings
"""
import os
from typing import Dict, Any, Optional
from urllib.parse import urlparse

from src.config.app_config import app_config


class DatabaseConfig:
    """Database configuration settings"""
    
    # ===== POSTGRESQL SETTINGS =====
    # Connection settings
    postgres_host = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db = os.getenv("POSTGRES_DB", "ps06_db")
    postgres_user = os.getenv("POSTGRES_USER", "ps06_user")
    postgres_password = os.getenv("POSTGRES_PASSWORD", "ps06_password")
    postgres_schema = os.getenv("POSTGRES_SCHEMA", "public")
    
    # SSL settings
    postgres_ssl_mode = os.getenv("POSTGRES_SSL_MODE", "prefer")
    postgres_ssl_cert = os.getenv("POSTGRES_SSL_CERT")
    postgres_ssl_key = os.getenv("POSTGRES_SSL_KEY")
    postgres_ssl_root_cert = os.getenv("POSTGRES_SSL_ROOT_CERT")
    
    # Connection pool settings
    postgres_pool_size = int(os.getenv("POSTGRES_POOL_SIZE", "20"))
    postgres_max_overflow = int(os.getenv("POSTGRES_MAX_OVERFLOW", "30"))
    postgres_pool_timeout = int(os.getenv("POSTGRES_POOL_TIMEOUT", "30"))
    postgres_pool_recycle = int(os.getenv("POSTGRES_POOL_RECYCLE", "3600"))
    postgres_pool_pre_ping = os.getenv("POSTGRES_POOL_PRE_PING", "true").lower() == "true"
    
    # Query settings
    postgres_statement_timeout = int(os.getenv("POSTGRES_STATEMENT_TIMEOUT", "300000"))  # 5 minutes
    postgres_idle_timeout = int(os.getenv("POSTGRES_IDLE_TIMEOUT", "600"))  # 10 minutes
    postgres_command_timeout = int(os.getenv("POSTGRES_COMMAND_TIMEOUT", "60"))
    
    # ===== REDIS SETTINGS =====
    # Connection settings
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db = int(os.getenv("REDIS_DB", "0"))
    redis_password = os.getenv("REDIS_PASSWORD")
    redis_username = os.getenv("REDIS_USERNAME")
    
    # SSL settings for Redis
    redis_ssl = os.getenv("REDIS_SSL", "false").lower() == "true"
    redis_ssl_cert_reqs = os.getenv("REDIS_SSL_CERT_REQS", "none")
    redis_ssl_ca_certs = os.getenv("REDIS_SSL_CA_CERTS")
    redis_ssl_certfile = os.getenv("REDIS_SSL_CERTFILE")
    redis_ssl_keyfile = os.getenv("REDIS_SSL_KEYFILE")
    
    # Connection pool settings
    redis_max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
    redis_socket_timeout = int(os.getenv("REDIS_SOCKET_TIMEOUT", "30"))
    redis_socket_connect_timeout = int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "30"))
    redis_health_check_interval = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))
    redis_retry_on_timeout = os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
    
    # Cache settings
    redis_default_ttl = int(os.getenv("REDIS_DEFAULT_TTL", "3600"))  # 1 hour
    redis_key_prefix = os.getenv("REDIS_KEY_PREFIX", "ps06:")
    
    # ===== BACKUP SETTINGS =====
    # PostgreSQL backup
    backup_enabled = os.getenv("BACKUP_ENABLED", "true").lower() == "true"
    backup_schedule = os.getenv("BACKUP_SCHEDULE", "0 2 * * *")  # Daily at 2 AM
    backup_retention_days = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))
    backup_location = os.getenv("BACKUP_LOCATION", "/data/backups")
    backup_compression = os.getenv("BACKUP_COMPRESSION", "gzip")
    
    # Point-in-time recovery
    wal_archiving_enabled = os.getenv("WAL_ARCHIVING_ENABLED", "false").lower() == "true"
    wal_archive_location = os.getenv("WAL_ARCHIVE_LOCATION", "/data/wal_archive")
    
    # ===== MONITORING SETTINGS =====
    # Database monitoring
    slow_query_threshold = float(os.getenv("SLOW_QUERY_THRESHOLD", "1.0"))  # seconds
    log_slow_queries = os.getenv("LOG_SLOW_QUERIES", "true").lower() == "true"
    log_all_queries = os.getenv("LOG_ALL_QUERIES", "false").lower() == "true"
    
    # Connection monitoring
    connection_warning_threshold = float(os.getenv("CONNECTION_WARNING_THRESHOLD", "0.8"))
    connection_critical_threshold = float(os.getenv("CONNECTION_CRITICAL_THRESHOLD", "0.95"))
    
    # Performance monitoring
    enable_query_stats = os.getenv("ENABLE_QUERY_STATS", "true").lower() == "true"
    stats_reset_interval = int(os.getenv("STATS_RESET_INTERVAL", "86400"))  # 24 hours
    
    @classmethod
    def get_postgres_url(cls, include_password: bool = True) -> str:
        """Get PostgreSQL connection URL"""
        password_part = f":{cls.postgres_password}" if include_password and cls.postgres_password else ""
        return (
            f"postgresql://{cls.postgres_user}{password_part}@"
            f"{cls.postgres_host}:{cls.postgres_port}/{cls.postgres_db}"
        )
    
    @classmethod
    def get_async_postgres_url(cls, include_password: bool = True) -> str:
        """Get async PostgreSQL connection URL"""
        password_part = f":{cls.postgres_password}" if include_password and cls.postgres_password else ""
        return (
            f"postgresql+asyncpg://{cls.postgres_user}{password_part}@"
            f"{cls.postgres_host}:{cls.postgres_port}/{cls.postgres_db}"
        )
    
    @classmethod
    def get_redis_url(cls, include_password: bool = True) -> str:
        """Get Redis connection URL"""
        auth_part = ""
        if cls.redis_username and cls.redis_password:
            if include_password:
                auth_part = f"{cls.redis_username}:{cls.redis_password}@"
            else:
                auth_part = f"{cls.redis_username}:***@"
        elif cls.redis_password:
            if include_password:
                auth_part = f":{cls.redis_password}@"
            else:
                auth_part = ":***@"
        
        protocol = "rediss" if cls.redis_ssl else "redis"
        return f"{protocol}://{auth_part}{cls.redis_host}:{cls.redis_port}/{cls.redis_db}"
    
    @classmethod
    def get_postgres_config(cls) -> Dict[str, Any]:
        """Get PostgreSQL configuration dictionary"""
        config = {
            "host": cls.postgres_host,
            "port": cls.postgres_port,
            "database": cls.postgres_db,
            "user": cls.postgres_user,
            "password": cls.postgres_password,
            "command_timeout": cls.postgres_command_timeout,
            "server_settings": {
                "application_name": app_config.app_name,
                "search_path": cls.postgres_schema,
                "statement_timeout": f"{cls.postgres_statement_timeout}ms",
                "idle_in_transaction_session_timeout": f"{cls.postgres_idle_timeout}s"
            }
        }
        
        # Add SSL configuration if specified
        if cls.postgres_ssl_mode != "disable":
            ssl_config = {"sslmode": cls.postgres_ssl_mode}
            
            if cls.postgres_ssl_cert:
                ssl_config["sslcert"] = cls.postgres_ssl_cert
            if cls.postgres_ssl_key:
                ssl_config["sslkey"] = cls.postgres_ssl_key
            if cls.postgres_ssl_root_cert:
                ssl_config["sslrootcert"] = cls.postgres_ssl_root_cert
            
            config.update(ssl_config)
        
        return config
    
    @classmethod
    def get_redis_config(cls) -> Dict[str, Any]:
        """Get Redis configuration dictionary"""
        config = {
            "host": cls.redis_host,
            "port": cls.redis_port,
            "db": cls.redis_db,
            "socket_timeout": cls.redis_socket_timeout,
            "socket_connect_timeout": cls.redis_socket_connect_timeout,
            "max_connections": cls.redis_max_connections,
            "health_check_interval": cls.redis_health_check_interval,
            "retry_on_timeout": cls.redis_retry_on_timeout,
            "decode_responses": True
        }
        
        # Add authentication if specified
        if cls.redis_username:
            config["username"] = cls.redis_username
        if cls.redis_password:
            config["password"] = cls.redis_password
        
        # Add SSL configuration if enabled
        if cls.redis_ssl:
            ssl_config = {"ssl": True}
            
            if cls.redis_ssl_cert_reqs:
                ssl_config["ssl_cert_reqs"] = cls.redis_ssl_cert_reqs
            if cls.redis_ssl_ca_certs:
                ssl_config["ssl_ca_certs"] = cls.redis_ssl_ca_certs
            if cls.redis_ssl_certfile:
                ssl_config["ssl_certfile"] = cls.redis_ssl_certfile
            if cls.redis_ssl_keyfile:
                ssl_config["ssl_keyfile"] = cls.redis_ssl_keyfile
            
            config.update(ssl_config)
        
        return config
    
    @classmethod
    def get_connection_pool_config(cls) -> Dict[str, Any]:
        """Get connection pool configuration"""
        return {
            "postgresql": {
                "min_size": max(1, cls.postgres_pool_size // 4),
                "max_size": cls.postgres_pool_size,
                "command_timeout": cls.postgres_command_timeout,
            },
            "redis": {
                "max_connections": cls.redis_max_connections,
                "socket_timeout": cls.redis_socket_timeout,
                "health_check_interval": cls.redis_health_check_interval
            }
        }
    
    @classmethod
    def get_backup_config(cls) -> Dict[str, Any]:
        """Get backup configuration"""
        return {
            "enabled": cls.backup_enabled,
            "schedule": cls.backup_schedule,
            "retention_days": cls.backup_retention_days,
            "location": cls.backup_location,
            "compression": cls.backup_compression,
            "wal_archiving": {
                "enabled": cls.wal_archiving_enabled,
                "location": cls.wal_archive_location
            }
        }
    
    @classmethod
    def get_monitoring_config(cls) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return {
            "slow_query_threshold": cls.slow_query_threshold,
            "log_slow_queries": cls.log_slow_queries,
            "log_all_queries": cls.log_all_queries,
            "connection_thresholds": {
                "warning": cls.connection_warning_threshold,
                "critical": cls.connection_critical_threshold
            },
            "performance": {
                "enable_stats": cls.enable_query_stats,
                "stats_reset_interval": cls.stats_reset_interval
            }
        }
    
    @classmethod
    def validate_config(cls) -> list:
        """Validate database configuration"""
        warnings = []
        
        # Check required PostgreSQL settings
        if not cls.postgres_host:
            warnings.append("PostgreSQL host is not configured")
        
        if not cls.postgres_db:
            warnings.append("PostgreSQL database name is not configured")
        
        if not cls.postgres_user:
            warnings.append("PostgreSQL user is not configured")
        
        # Check Redis settings
        if not cls.redis_host:
            warnings.append("Redis host is not configured")
        
        # Validate pool sizes
        if cls.postgres_pool_size <= 0:
            warnings.append("PostgreSQL pool size must be positive")
        
        if cls.redis_max_connections <= 0:
            warnings.append("Redis max connections must be positive")
        
        # Check backup settings
        if cls.backup_enabled:
            if not os.path.exists(os.path.dirname(cls.backup_location)):
                warnings.append(f"Backup location directory does not exist: {cls.backup_location}")
        
        return warnings
    
    @classmethod
    def test_connections(cls) -> Dict[str, bool]:
        """Test database connections"""
        results = {"postgresql": False, "redis": False}
        
        # Test PostgreSQL connection
        try:
            import asyncpg
            import asyncio
            
            async def test_postgres():
                try:
                    conn = await asyncpg.connect(
                        host=cls.postgres_host,
                        port=cls.postgres_port,
                        database=cls.postgres_db,
                        user=cls.postgres_user,
                        password=cls.postgres_password,
                        command_timeout=5
                    )
                    await conn.execute("SELECT 1")
                    await conn.close()
                    return True
                except Exception:
                    return False
            
            results["postgresql"] = asyncio.run(test_postgres())
            
        except ImportError:
            pass  # asyncpg not available
        except Exception:
            pass
        
        # Test Redis connection
        try:
            import redis
            
            r = redis.Redis(**cls.get_redis_config())
            r.ping()
            results["redis"] = True
            
        except ImportError:
            pass  # redis not available
        except Exception:
            pass
        
        return results
    
    @classmethod
    def get_database_info(cls) -> Dict[str, Any]:
        """Get database configuration information"""
        return {
            "postgresql": {
                "url": cls.get_postgres_url(include_password=False),
                "host": cls.postgres_host,
                "port": cls.postgres_port,
                "database": cls.postgres_db,
                "user": cls.postgres_user,
                "pool_size": cls.postgres_pool_size,
                "ssl_mode": cls.postgres_ssl_mode
            },
            "redis": {
                "url": cls.get_redis_url(include_password=False),
                "host": cls.redis_host,
                "port": cls.redis_port,
                "db": cls.redis_db,
                "max_connections": cls.redis_max_connections,
                "ssl_enabled": cls.redis_ssl
            },
            "backup": cls.get_backup_config(),
            "monitoring": cls.get_monitoring_config()
        }


# Create global instance
database_config = DatabaseConfig()

# Validate configuration on import
config_warnings = database_config.validate_config()
if config_warnings:
    import logging
    logger = logging.getLogger(__name__)
    for warning in config_warnings:
        logger.warning(f"Database config warning: {warning}")