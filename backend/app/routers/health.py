"""
Health check endpoint.
"""
from fastapi import APIRouter
from datetime import datetime
import app.data.storage as storage

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint with storage status."""
    from app.data.storage import init_storage
    # Ensure storage initialized
    if storage.sqlite_conn is None:
        await init_storage()

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
        try:
            storage.sqlite_conn.execute("SELECT 1")
            health_status["storage"]["sqlite_status"] = "connected"
        except Exception as e:
            health_status["storage"]["sqlite_status"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
    
    return health_status

