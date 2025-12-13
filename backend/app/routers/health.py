"""
Health check endpoint.
"""
from fastapi import APIRouter
from datetime import datetime
import app.data.storage as storage
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint with storage status."""
    import time
    start_time = time.time()
    
    try:
        logger.info("🏥 Health check request received")
        print(f"[{time.strftime('%H:%M:%S')}] 🏥 Health check request received")
        
        from app.data.storage import init_storage
        
        # Ensure storage initialized
        logger.info("📦 Checking storage initialization...")
        print(f"[{time.strftime('%H:%M:%S')}] 📦 Checking storage initialization...")
        
        if storage.sqlite_conn is None:
            logger.info("🔧 Initializing storage...")
            print(f"[{time.strftime('%H:%M:%S')}] 🔧 Initializing storage...")
            await init_storage()
            logger.info("✅ Storage initialized")
            print(f"[{time.strftime('%H:%M:%S')}] ✅ Storage initialized")

        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": "ED Bottleneck Engine API",
            "storage": {
                "sqlite": storage.sqlite_conn is not None
            }
        }
        
        # Test SQLite connection if available
        if storage.sqlite_conn:
            logger.info("🔍 Testing SQLite connection...")
            print(f"[{time.strftime('%H:%M:%S')}] 🔍 Testing SQLite connection...")
            try:
                storage.sqlite_conn.execute("SELECT 1")
                health_status["storage"]["sqlite_status"] = "connected"
                logger.info("✅ SQLite connection test passed")
                print(f"[{time.strftime('%H:%M:%S')}] ✅ SQLite connection test passed")
            except Exception as e:
                health_status["storage"]["sqlite_status"] = f"error: {str(e)}"
                health_status["status"] = "degraded"
                logger.error(f"❌ SQLite connection test failed: {e}")
                print(f"[{time.strftime('%H:%M:%S')}] ❌ SQLite connection test failed: {e}")
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Health check completed in {elapsed:.3f}s")
        print(f"[{time.strftime('%H:%M:%S')}] ✅ Health check completed in {elapsed:.3f}s")
        
        return health_status
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ Health check failed after {elapsed:.3f}s: {e}", exc_info=True)
        print(f"[{time.strftime('%H:%M:%S')}] ❌ Health check failed after {elapsed:.3f}s: {e}")
        raise

