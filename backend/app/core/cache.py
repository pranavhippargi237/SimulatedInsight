"""
Redis-based caching layer for query responses and simulations.

Provides fast caching with TTL support for:
- Chat query responses (5 min TTL)
- Simulation results (1 hour TTL)
- Correlation calculations (10 min TTL)
- Bottleneck detection (5 min TTL)

Gracefully falls back to in-memory cache if Redis is unavailable.
"""
import logging
import json
import hashlib
from typing import Optional, Any, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Redis imports with graceful fallback
REDIS_AVAILABLE = False
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    logger.debug("Redis not available - using in-memory cache fallback")
    redis = None

# In-memory fallback cache
_memory_cache: Dict[str, Dict[str, Any]] = {}


async def cache_get(key: str) -> Optional[Any]:
    """
    Get value from cache (Redis or in-memory fallback).
    
    Args:
        key: Cache key
        
    Returns:
        Cached value or None if not found/expired
    """
    if REDIS_AVAILABLE:
        try:
            # Try to get Redis connection from settings
            from app.core.config import settings
            redis_client = redis.Redis(
                host=getattr(settings, 'REDIS_HOST', 'localhost'),
                port=getattr(settings, 'REDIS_PORT', 6379),
                db=0,
                decode_responses=True
            )
            value = await redis_client.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except Exception as e:
            logger.debug(f"Redis cache_get failed: {e}, using memory fallback")
            # Fall through to memory cache
    
    # In-memory fallback
    if key in _memory_cache:
        entry = _memory_cache[key]
        expires_at = entry.get("expires_at")
        if expires_at and datetime.utcnow() < expires_at:
            return entry.get("value")
        else:
            # Expired, remove it
            del _memory_cache[key]
    return None


async def cache_set(key: str, value: Any, ttl: int = 300) -> bool:
    """
    Set value in cache (Redis or in-memory fallback).
    
    Args:
        key: Cache key
        value: Value to cache (must be JSON-serializable)
        ttl: Time to live in seconds (default: 5 minutes)
        
    Returns:
        True if cached successfully, False otherwise
    """
    try:
        # Serialize value
        if isinstance(value, (dict, list)):
            serialized = json.dumps(value, default=str)
        else:
            serialized = json.dumps(value, default=str)
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to serialize cache value: {e}")
        return False
    
    if REDIS_AVAILABLE:
        try:
            from app.core.config import settings
            redis_client = redis.Redis(
                host=getattr(settings, 'REDIS_HOST', 'localhost'),
                port=getattr(settings, 'REDIS_PORT', 6379),
                db=0,
                decode_responses=False  # We're storing JSON strings
            )
            await redis_client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.debug(f"Redis cache_set failed: {e}, using memory fallback")
            # Fall through to memory cache
    
    # In-memory fallback
    expires_at = datetime.utcnow() + timedelta(seconds=ttl)
    _memory_cache[key] = {
        "value": value,  # Store original value, not serialized
        "expires_at": expires_at
    }
    
    # Clean up expired entries periodically (keep cache size reasonable)
    if len(_memory_cache) > 1000:
        now = datetime.utcnow()
        expired_keys = [
            k for k, v in _memory_cache.items()
            if v.get("expires_at") and v["expires_at"] < now
        ]
        for k in expired_keys:
            del _memory_cache[k]
    
    return True


async def cache_delete(key: str) -> bool:
    """Delete a key from cache."""
    if REDIS_AVAILABLE:
        try:
            from app.core.config import settings
            redis_client = redis.Redis(
                host=getattr(settings, 'REDIS_HOST', 'localhost'),
                port=getattr(settings, 'REDIS_PORT', 6379),
                db=0
            )
            await redis_client.delete(key)
            return True
        except Exception:
            pass
    
    # In-memory fallback
    if key in _memory_cache:
        del _memory_cache[key]
        return True
    return False


async def cache_clear(pattern: str = None) -> int:
    """
    Clear cache entries matching pattern (or all if pattern is None).
    
    Args:
        pattern: Optional pattern to match (e.g., "sim_*")
        
    Returns:
        Number of keys deleted
    """
    if REDIS_AVAILABLE:
        try:
            from app.core.config import settings
            redis_client = redis.Redis(
                host=getattr(settings, 'REDIS_HOST', 'localhost'),
                port=getattr(settings, 'REDIS_PORT', 6379),
                db=0
            )
            if pattern:
                keys = await redis_client.keys(pattern)
                if keys:
                    await redis_client.delete(*keys)
                    return len(keys)
            else:
                await redis_client.flushdb()
                return -1  # Unknown count
        except Exception:
            pass
    
    # In-memory fallback
    if pattern:
        deleted = 0
        keys_to_delete = [k for k in _memory_cache.keys() if pattern.replace('*', '') in k]
        for k in keys_to_delete:
            del _memory_cache[k]
            deleted += 1
        return deleted
    else:
        count = len(_memory_cache)
        _memory_cache.clear()
        return count
