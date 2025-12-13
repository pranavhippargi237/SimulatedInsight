"""
Storage layer using SQLite for persistence and Redis for caching (optional).

Provides:
- SQLite: Primary data storage for events, KPIs, and staffing
- Redis: High-performance caching for query responses, simulation results, and correlation calculations
- Graceful fallback: If Redis is unavailable, caching is disabled (no-op functions)
"""
import logging
import os
import sqlite3
import json
import hashlib
from typing import List, Optional, Dict, Any
from datetime import datetime
from app.core.config import settings

logger = logging.getLogger(__name__)

# Globals
sqlite_conn: Optional[sqlite3.Connection] = None
redis_client = None

DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/ed.sqlite")

# Redis connection (optional)
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
    logger.debug("Redis available for caching")
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    logger.debug("Redis not available - caching disabled")


async def init_storage():
    """Initialize SQLite and Redis (if available)."""
    global sqlite_conn, redis_client

    # Initialize SQLite
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        sqlite_conn = sqlite3.connect(DB_PATH, check_same_thread=False, detect_types=sqlite3.PARSE_DECLTYPES)
        sqlite_conn.execute("PRAGMA journal_mode=WAL;")
        sqlite_conn.execute("PRAGMA synchronous=NORMAL;")
        await create_tables()
        logger.info("SQLite connection established.")
        print("✅ SQLite database initialized successfully")
    except Exception as e:
        logger.error(f"SQLite connection failed: {e}")
        print(f"❌ SQLite initialization failed: {e}")
        sqlite_conn = None
    
    # Initialize Redis (optional)
    if REDIS_AVAILABLE:
        try:
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
            redis_client = await redis.from_url(redis_url, decode_responses=True)
            # Test connection
            await redis_client.ping()
            logger.info("Redis connection established for caching.")
            print("✅ Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e} - caching disabled")
            redis_client = None
    else:
        logger.info("Redis not available - caching disabled")


async def create_tables():
    """Create SQLite tables if they don't exist."""
    if not sqlite_conn:
        logger.warning("SQLite not initialized - skipping table creation")
        return
    try:
        cur = sqlite_conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ed_events (
                timestamp TEXT,
                event_type TEXT,
                patient_id TEXT,
                stage TEXT,
                resource_type TEXT,
                resource_id TEXT,
                duration_minutes REAL,
                esi INTEGER,
                metadata TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON ed_events(timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON ed_events(event_type)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS ed_kpis (
                timestamp TEXT,
                door_to_doctor_minutes REAL,
                length_of_stay_minutes REAL,
                lwbs_rate REAL,
                bed_utilization REAL,
                queue_length INTEGER
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_kpis_ts ON ed_kpis(timestamp)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS ed_staffing (
                timestamp TEXT,
                resource_type TEXT,
                count INTEGER,
                department TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_staff_ts ON ed_staffing(timestamp)")

        sqlite_conn.commit()
        logger.info("SQLite tables created/verified.")
    except Exception as e:
        logger.error(f"Failed to create SQLite tables: {e}", exc_info=True)
        raise


async def insert_events(events: List[Dict[str, Any]]):
    """Insert events into SQLite."""
    if not events:
        return

    try:
        _insert_events_sqlite(events)
    except Exception as e:
        logger.error(f"Failed to insert events: {e}", exc_info=True)
        raise


def _insert_events_sqlite(events: List[Dict[str, Any]]):
    if not sqlite_conn:
        raise ValueError("SQLite not available")
    cur = sqlite_conn.cursor()
    rows = []
    for e in events:
        ts = e.get("timestamp")
        if ts is not None:
            if hasattr(ts, "isoformat"):
                ts = ts.isoformat()
            else:
                ts = str(ts)
        rows.append((
            ts,
            e.get("event_type"),
            e.get("patient_id"),
            e.get("stage"),
            e.get("resource_type"),
            e.get("resource_id"),
            e.get("duration_minutes"),
            e.get("esi"),
            str(e.get("metadata", {}))
        ))
    cur.executemany("""
        INSERT INTO ed_events (
            timestamp, event_type, patient_id, stage, resource_type,
            resource_id, duration_minutes, esi, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    sqlite_conn.commit()
    logger.info(f"Inserted {len(rows)} events into SQLite.")
    print(f"📥 Stored {len(rows)} events in SQLite database")


def insert_kpis_sqlite(kpis: List[Dict[str, Any]]):
    """Insert KPI rows into SQLite."""
    if not sqlite_conn:
        raise ValueError("SQLite not available")
    if not kpis:
        return
    cur = sqlite_conn.cursor()
    rows = []
    for k in kpis:
        ts = k.get("timestamp")
        if ts is not None:
            if hasattr(ts, "isoformat"):
                ts = ts.isoformat()
            else:
                ts = str(ts)
        rows.append((
            ts,
            k.get("dtd", 0),
            k.get("los", 0),
            k.get("lwbs", 0),
            k.get("bed_utilization", 0),
            k.get("queue_length", 0)
        ))
    cur.executemany("""
        INSERT INTO ed_kpis (
            timestamp, door_to_doctor_minutes, length_of_stay_minutes,
            lwbs_rate, bed_utilization, queue_length
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, rows)
    sqlite_conn.commit()
    logger.info(f"Inserted {len(rows)} KPI rows into SQLite.")
    print(f"💾 Stored {len(rows)} KPI records in SQLite database")


async def get_events(
    start_time: datetime,
    end_time: datetime,
    event_types: Optional[List[str]] = None,
    raise_if_empty: bool = False
) -> List[Dict[str, Any]]:
    """Retrieve events from SQLite."""
    if not sqlite_conn:
        return []
    
    try:
        events = _get_events_sqlite(start_time, end_time, event_types)
        if raise_if_empty and not events:
            raise ValueError("No events available in storage")
        return events
    except Exception as e:
        logger.error(f"Failed to get events: {e}", exc_info=True)
        return []


def _get_events_sqlite(
    start_time: datetime,
    end_time: datetime,
    event_types: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    cur = sqlite_conn.cursor()
    params = [start_time.isoformat(), end_time.isoformat()]
    query = """
        SELECT timestamp, event_type, patient_id, stage, resource_type, resource_id, duration_minutes, esi, metadata
        FROM ed_events
        WHERE timestamp >= ? AND timestamp <= ?
    """
    if event_types:
        placeholders = ",".join("?" * len(event_types))
        query += f" AND event_type IN ({placeholders})"
        params.extend(event_types)
    query += " ORDER BY timestamp LIMIT 10000"
    rows = cur.execute(query, params).fetchall()
    events = []
    for row in rows:
        events.append({
            "timestamp": datetime.fromisoformat(row[0]) if row[0] else None,
            "event_type": row[1],
            "patient_id": row[2],
            "stage": row[3],
            "resource_type": row[4],
            "resource_id": row[5],
            "duration_minutes": row[6],
            "esi": row[7],
            "metadata": eval(row[8]) if row[8] else {}
        })
    return events


async def get_kpis(
    start_time: datetime,
    end_time: datetime,
    window_minutes: int = 60,
    raise_if_empty: bool = False
) -> List[Dict[str, Any]]:
    """Retrieve KPIs from SQLite."""
    if not sqlite_conn:
        return []
    
    try:
        kpis = _get_kpis_sqlite(start_time, end_time)
        if raise_if_empty and (not kpis or len(kpis) == 0):
            raise ValueError("No KPIs available in storage")
        return kpis
    except Exception as e:
        logger.error(f"Failed to get KPIs: {e}", exc_info=True)
        return []


def _get_kpis_sqlite(
    start_time: datetime,
    end_time: datetime
) -> List[Dict[str, Any]]:
    cur = sqlite_conn.cursor()
    rows = cur.execute("""
        SELECT timestamp, door_to_doctor_minutes, length_of_stay_minutes, lwbs_rate, bed_utilization, queue_length
        FROM ed_kpis
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp
        LIMIT 1000
    """, (start_time.isoformat(), end_time.isoformat())).fetchall()
    kpis = []
    for row in rows:
        kpis.append({
            "timestamp": datetime.fromisoformat(row[0]) if row[0] else None,
            "dtd": row[1],
            "los": row[2],
            "lwbs": row[3],
            "bed_utilization": row[4],
            "queue_length": row[5]
        })
    return kpis


def get_all_kpis(limit: int = 1000) -> List[Dict[str, Any]]:
    """Fetch all KPIs up to limit."""
    if not sqlite_conn:
        return []
    cur = sqlite_conn.cursor()
    rows = cur.execute(f"""
        SELECT timestamp, door_to_doctor_minutes, length_of_stay_minutes, lwbs_rate, bed_utilization, queue_length
        FROM ed_kpis
        ORDER BY timestamp
        LIMIT {limit}
    """).fetchall()
    kpis = []
    for row in rows:
        kpis.append({
            "timestamp": datetime.fromisoformat(row[0]) if row[0] else None,
            "dtd": row[1],
            "los": row[2],
            "lwbs": row[3],
            "bed_utilization": row[4],
            "queue_length": row[5]
        })
    return kpis


async def cache_set(key: str, value: Any, ttl: Optional[int] = None):
    """
    Cache a value in Redis with optional TTL.
    
    Args:
        key: Cache key
        value: Value to cache (will be JSON-serialized)
        ttl: Time-to-live in seconds (default: 3600 = 1 hour)
    
    Returns:
        True if cached successfully, False otherwise
    """
    if not redis_client:
        return False
    
    try:
        # Serialize value to JSON
        serialized = json.dumps(value, default=str)
        
        # Default TTL: 1 hour for query responses, 6 hours for simulations
        if ttl is None:
            ttl = 3600  # 1 hour default
        
        # Set with TTL
        await redis_client.setex(key, ttl, serialized)
        logger.debug(f"Cached key: {key} (TTL: {ttl}s)")
        return True
    except Exception as e:
        logger.warning(f"Failed to cache key {key}: {e}")
        return False


async def cache_get(key: str) -> Optional[Any]:
    """
    Retrieve a cached value from Redis.
    
    Args:
        key: Cache key
    
    Returns:
        Cached value (deserialized from JSON) or None if not found
    """
    if not redis_client:
        return None
    
    try:
        serialized = await redis_client.get(key)
        if serialized is None:
            return None
        
        # Deserialize from JSON
        value = json.loads(serialized)
        logger.debug(f"Cache hit: {key}")
        return value
    except Exception as e:
        logger.warning(f"Failed to retrieve cache key {key}: {e}")
        return None


async def cache_delete(key: str):
    """Delete a cached value from Redis."""
    if not redis_client:
        return
    
    try:
        await redis_client.delete(key)
        logger.debug(f"Deleted cache key: {key}")
    except Exception as e:
        logger.warning(f"Failed to delete cache key {key}: {e}")


async def cache_clear(prefix: str = ""):
    """
    Clear cached values matching a prefix.
    
    Args:
        prefix: Key prefix to match (empty string = clear all)
    """
    if not redis_client:
        return
    
    try:
        if prefix:
            # Delete keys matching prefix
            keys = await redis_client.keys(f"{prefix}*")
            if keys:
                await redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache keys with prefix: {prefix}")
        else:
            # Clear all keys (use with caution)
            await redis_client.flushdb()
            logger.warning("Cleared all cache keys")
    except Exception as e:
        logger.warning(f"Failed to clear cache with prefix {prefix}: {e}")


def _generate_cache_key(prefix: str, **kwargs) -> str:
    """
    Generate a cache key from prefix and keyword arguments.
    
    Args:
        prefix: Key prefix (e.g., "query:", "simulation:")
        **kwargs: Key-value pairs to include in hash
    
    Returns:
        Cache key string
    """
    # Sort kwargs for consistent hashing
    sorted_kwargs = sorted(kwargs.items())
    # Create hash from kwargs
    hash_str = json.dumps(sorted_kwargs, default=str, sort_keys=True)
    hash_obj = hashlib.md5(hash_str.encode())
    hash_hex = hash_obj.hexdigest()[:12]  # Use first 12 chars
    
    return f"{prefix}{hash_hex}"


def reset_sqlite():
    """Truncate all data tables."""
    if not sqlite_conn:
        return
    cur = sqlite_conn.cursor()
    for table in ["ed_events", "ed_kpis", "ed_staffing"]:
        try:
            cur.execute(f"DELETE FROM {table}")
        except Exception as e:
            logger.warning(f"Could not clear table {table}: {e}")
    sqlite_conn.commit()

