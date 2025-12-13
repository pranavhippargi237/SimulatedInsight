"""
Main FastAPI application entry point for ED Bottleneck Engine.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.routers import ingest, metrics, detect, simulate, optimize, health, advisor, intelligent, chat, insights, flow
from app.core.config import settings

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Log startup
logger.info("=" * 60)
logger.info("🚀 Starting ED Bottleneck Engine API")
logger.info("=" * 60)
print("=" * 60)
print("🚀 Starting ED Bottleneck Engine API")
print("=" * 60)

# Initialize FastAPI app
app = FastAPI(
    title="ED Bottleneck Engine API",
    description="Real-time Emergency Department bottleneck detection and simulation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - allow environment variable override for production
import os
cors_origins = settings.CORS_ORIGINS.copy()
if os.getenv("CORS_ORIGINS"):
    # Allow comma-separated list from environment
    cors_origins.extend([origin.strip() for origin in os.getenv("CORS_ORIGINS").split(",")])
cors_origins.extend(["http://localhost:3000", "http://127.0.0.1:3000"])  # Always include localhost for dev

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(ingest.router, prefix="/api", tags=["Ingestion"])
app.include_router(metrics.router, prefix="/api", tags=["Metrics"])
app.include_router(detect.router, prefix="/api", tags=["Detection"])
app.include_router(simulate.router, prefix="/api", tags=["Simulation"])
app.include_router(optimize.router, prefix="/api", tags=["Optimization"])
app.include_router(advisor.router, prefix="/api", tags=["Advisor"])
app.include_router(intelligent.router, prefix="/api", tags=["Intelligent"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
# Also mount chat router at /api/v1 for versioned streaming endpoint
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(insights.router, tags=["Insights"])
app.include_router(flow.router, prefix="/api", tags=["Flow"])


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."}
    )


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    import time
    startup_start = time.time()
    
    logger.info("🔄 Starting API initialization...")
    print(f"[{time.strftime('%H:%M:%S')}] 🔄 Starting API initialization...")
    
    try:
        # Initialize database connections, etc.
        logger.info("📦 Initializing storage...")
        print(f"[{time.strftime('%H:%M:%S')}] 📦 Initializing storage...")
        from app.data.storage import init_storage
        await init_storage()
        
        elapsed = time.time() - startup_start
        logger.info(f"✅ API startup complete in {elapsed:.3f}s")
        print(f"[{time.strftime('%H:%M:%S')}] ✅ API startup complete in {elapsed:.3f}s")
        print("=" * 60)
        print("🌐 API is ready and listening on http://localhost:8000")
        print("📚 API docs available at http://localhost:8000/docs")
        print("=" * 60)
    except Exception as e:
        elapsed = time.time() - startup_start
        logger.error(f"❌ API startup failed after {elapsed:.3f}s: {e}", exc_info=True)
        print(f"[{time.strftime('%H:%M:%S')}] ❌ API startup failed after {elapsed:.3f}s: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    import time
    logger.info("🛑 Shutting down ED Bottleneck Engine API...")
    print(f"[{time.strftime('%H:%M:%S')}] 🛑 Shutting down ED Bottleneck Engine API...")

