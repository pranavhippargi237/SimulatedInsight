"""
Main FastAPI application entry point for ED Bottleneck Engine.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.routers import ingest, metrics, detect, simulate, optimize, health, advisor, intelligent, chat, insights, flow
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ED Bottleneck Engine API",
    description="Real-time Emergency Department bottleneck detection and simulation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS + ["http://localhost:3000", "http://127.0.0.1:3000"],  # Explicitly include localhost
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
    logger.info("Starting ED Bottleneck Engine API...")
    # Initialize database connections, etc.
    from app.data.storage import init_storage
    await init_storage()
    logger.info("API startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down ED Bottleneck Engine API...")

