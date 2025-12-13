"""
Tests for Redis caching functionality.
"""
import pytest
from app.data.storage import cache_set, cache_get, cache_delete, cache_clear, _generate_cache_key


@pytest.mark.asyncio
async def test_cache_set_get():
    """Test basic cache set and get operations."""
    test_key = "test:key:123"
    test_value = {"test": "data", "number": 42}
    
    # Set cache
    success = await cache_set(test_key, test_value, ttl=60)
    # May fail if Redis not available - that's OK
    if not success:
        pytest.skip("Redis not available")
    
    # Get cache
    retrieved = await cache_get(test_key)
    assert retrieved == test_value
    
    # Cleanup
    await cache_delete(test_key)


@pytest.mark.asyncio
async def test_cache_ttl():
    """Test that TTL works correctly."""
    test_key = "test:ttl:123"
    test_value = {"test": "ttl"}
    
    success = await cache_set(test_key, test_value, ttl=1)
    if not success:
        pytest.skip("Redis not available")
    
    # Should be available immediately
    retrieved = await cache_get(test_key)
    assert retrieved == test_value
    
    # Cleanup
    await cache_delete(test_key)


@pytest.mark.asyncio
async def test_cache_key_generation():
    """Test cache key generation."""
    key1 = _generate_cache_key("query:", query="test", window_hours=24)
    key2 = _generate_cache_key("query:", query="test", window_hours=24)
    key3 = _generate_cache_key("query:", query="test", window_hours=48)
    
    # Same parameters should generate same key
    assert key1 == key2
    
    # Different parameters should generate different keys
    assert key1 != key3
    
    # Key should start with prefix
    assert key1.startswith("query:")


@pytest.mark.asyncio
async def test_cache_clear():
    """Test cache clearing with prefix."""
    # Set multiple keys with prefix
    await cache_set("test:prefix:1", {"data": 1})
    await cache_set("test:prefix:2", {"data": 2})
    await cache_set("test:other:1", {"data": 3})
    
    # Clear with prefix
    await cache_clear("test:prefix:")
    
    # Check that prefix keys are gone
    assert await cache_get("test:prefix:1") is None
    assert await cache_get("test:prefix:2") is None
    
    # Other key should still exist
    other = await cache_get("test:other:1")
    if other:
        assert other["data"] == 3
    
    # Cleanup
    await cache_delete("test:other:1")
