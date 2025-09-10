"""
Cache Manager
Advanced caching utilities with multiple backends
"""
import logging
import asyncio
import time
import json
import hashlib
import pickle
from typing import Any, Optional, Dict, List, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

import redis.asyncio as redis
from src.config.app_config import app_config

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    ttl: int
    created_at: float
    expires_at: float
    access_count: int = 0
    last_accessed: float = 0
    size_bytes: int = 0


class CacheBackend(ABC):
    """Abstract cache backend interface"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set value with TTL"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend with LRU eviction"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0
        }
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            await self._cleanup_expired()
            
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            current_time = time.time()
            
            if current_time > entry.expires_at:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                self._stats['misses'] += 1
                return None
            
            # Update access info
            entry.access_count += 1
            entry.last_accessed = current_time
            
            # Move to end (most recently used)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            self._stats['hits'] += 1
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        if ttl is None:
            ttl = self.default_ttl
        
        async with self._lock:
            current_time = time.time()
            
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = len(str(value))
            
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl,
                created_at=current_time,
                expires_at=current_time + ttl,
                size_bytes=size_bytes
            )
            
            # Remove existing entry if present
            if key in self._cache:
                if key in self._access_order:
                    self._access_order.remove(key)
            
            # Check if we need to evict
            while len(self._cache) >= self.max_size:
                await self._evict_lru()
            
            self._cache[key] = entry
            self._access_order.append(key)
            self._stats['sets'] += 1
            
            return True
    
    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                self._stats['deletes'] += 1
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        async with self._lock:
            if key not in self._cache:
                return False
            
            entry = self._cache[key]
            if time.time() > entry.expires_at:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return False
            
            return True
    
    async def clear(self) -> bool:
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            return True
    
    async def get_stats(self) -> Dict[str, Any]:
        async with self._lock:
            await self._cleanup_expired()
            
            total_size = sum(entry.size_bytes for entry in self._cache.values())
            
            return {
                **self._stats,
                'size': len(self._cache),
                'max_size': self.max_size,
                'total_size_bytes': total_size,
                'hit_rate': self._stats['hits'] / max(1, self._stats['hits'] + self._stats['misses'])
            }
    
    async def _cleanup_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time > entry.expires_at
        ]
        
        for key in expired_keys:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
    
    async def _evict_lru(self):
        """Evict least recently used entry"""
        if self._access_order:
            lru_key = self._access_order.pop(0)
            if lru_key in self._cache:
                del self._cache[lru_key]
                self._stats['evictions'] += 1


class RedisCacheBackend(CacheBackend):
    """Redis cache backend"""
    
    def __init__(self, redis_url: str = None, prefix: str = "ps06:cache:"):
        self.redis_url = redis_url or app_config.redis_url
        self.prefix = prefix
        self.redis_client: Optional[redis.Redis] = None
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis cache backend")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key"""
        return f"{self.prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        if not self.redis_client:
            return None
        
        try:
            prefixed_key = self._make_key(key)
            data = await self.redis_client.get(prefixed_key)
            
            if data is None:
                self._stats['misses'] += 1
                return None
            
            # Deserialize
            value = pickle.loads(data)
            self._stats['hits'] += 1
            return value
            
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self._stats['errors'] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        if not self.redis_client:
            return False
        
        try:
            prefixed_key = self._make_key(key)
            data = pickle.dumps(value)
            
            await self.redis_client.setex(prefixed_key, ttl, data)
            self._stats['sets'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            self._stats['errors'] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        if not self.redis_client:
            return False
        
        try:
            prefixed_key = self._make_key(key)
            result = await self.redis_client.delete(prefixed_key)
            
            if result > 0:
                self._stats['deletes'] += 1
                return True
            return False
            
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            self._stats['errors'] += 1
            return False
    
    async def exists(self, key: str) -> bool:
        if not self.redis_client:
            return False
        
        try:
            prefixed_key = self._make_key(key)
            return await self.redis_client.exists(prefixed_key) > 0
            
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    async def clear(self) -> bool:
        if not self.redis_client:
            return False
        
        try:
            pattern = f"{self.prefix}*"
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                await self.redis_client.delete(*keys)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        stats = dict(self._stats)
        
        if self.redis_client:
            try:
                info = await self.redis_client.info()
                stats.update({
                    'connected': True,
                    'used_memory': info.get('used_memory', 0),
                    'connected_clients': info.get('connected_clients', 0),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0)
                })
            except Exception as e:
                logger.error(f"Redis stats error: {e}")
                stats['connected'] = False
        else:
            stats['connected'] = False
        
        return stats


class CacheManager:
    """Advanced cache manager with multiple backends and features"""
    
    def __init__(self, backend: CacheBackend = None, enable_metrics: bool = True):
        self.backend = backend or MemoryCacheBackend()
        self.enable_metrics = enable_metrics
        self._locks: Dict[str, asyncio.Lock] = {}
        self._function_cache: Dict[str, Any] = {}
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        try:
            value = await self.backend.get(key)
            return value if value is not None else default
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return default
    
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set value in cache"""
        try:
            return await self.backend.set(key, value, ttl)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            return await self.backend.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def get_or_set(
        self, 
        key: str, 
        callable_func: Callable, 
        ttl: int = 300,
        *args, 
        **kwargs
    ) -> Any:
        """Get from cache or compute and set"""
        # Try to get from cache first
        value = await self.get(key)
        if value is not None:
            return value
        
        # Use lock to prevent multiple computations
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        
        async with self._locks[key]:
            # Double-check after acquiring lock
            value = await self.get(key)
            if value is not None:
                return value
            
            # Compute value
            try:
                if asyncio.iscoroutinefunction(callable_func):
                    value = await callable_func(*args, **kwargs)
                else:
                    value = callable_func(*args, **kwargs)
                
                # Cache the result
                await self.set(key, value, ttl)
                return value
                
            except Exception as e:
                logger.error(f"Error computing cached value: {e}")
                raise
    
    def cached(
        self, 
        ttl: int = 300, 
        key_func: Optional[Callable] = None,
        namespace: str = "func"
    ):
        """Decorator for caching function results"""
        def decorator(func: Callable):
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Generate key from function name and arguments
                    func_name = f"{func.__module__}.{func.__name__}"
                    args_str = str(args) + str(sorted(kwargs.items()))
                    cache_key = hashlib.md5(args_str.encode()).hexdigest()
                    cache_key = f"{namespace}:{func_name}:{cache_key}"
                
                return await self.get_or_set(cache_key, func, ttl, *args, **kwargs)
            
            return wrapper
        return decorator
    
    async def bulk_get(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple keys at once"""
        result = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        return result
    
    async def bulk_set(self, items: Dict[str, Any], ttl: int = 300) -> Dict[str, bool]:
        """Set multiple key-value pairs"""
        results = {}
        for key, value in items.items():
            results[key] = await self.set(key, value, ttl)
        return results
    
    async def bulk_delete(self, keys: List[str]) -> Dict[str, bool]:
        """Delete multiple keys"""
        results = {}
        for key in keys:
            results[key] = await self.delete(key)
        return results
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return await self.backend.get_stats()
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        return await self.backend.clear()
    
    async def cleanup_expired(self):
        """Cleanup expired entries (if supported by backend)"""
        if hasattr(self.backend, '_cleanup_expired'):
            await self.backend._cleanup_expired()
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate keys matching pattern (placeholder for future implementation)"""
        # This would require backend-specific implementation
        pass


# Global cache manager instance
cache_manager = CacheManager()


# Convenience functions
async def get_cache_manager(backend_type: str = "memory") -> CacheManager:
    """Get configured cache manager"""
    if backend_type == "redis":
        backend = RedisCacheBackend()
        await backend.connect()
        return CacheManager(backend)
    else:
        return CacheManager(MemoryCacheBackend())


# Cache decorators
def cached(ttl: int = 300, key_func: Optional[Callable] = None):
    """Convenience decorator using global cache manager"""
    return cache_manager.cached(ttl, key_func)


async def cache_get(key: str, default: Any = None) -> Any:
    """Convenience function for cache get"""
    return await cache_manager.get(key, default)


async def cache_set(key: str, value: Any, ttl: int = 300) -> bool:
    """Convenience function for cache set"""
    return await cache_manager.set(key, value, ttl)


async def cache_delete(key: str) -> bool:
    """Convenience function for cache delete"""
    return await cache_manager.delete(key)