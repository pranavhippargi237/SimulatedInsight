"""
Application configuration using Pydantic settings.
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    API_V1_PREFIX: str = "/api"
    PROJECT_NAME: str = "ED Bottleneck Engine"
    VERSION: str = "1.0.0"
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ]
    
    # Environment
    ENVIRONMENT: str = "development"
    
    # Storage: SQLite only (no external dependencies)
    # Removed: ClickHouse and Redis - using SQLite for all data storage
    
    # OpenAI Settings
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 10
    
    # Simulation Settings
    SIMULATION_ITERATIONS: int = 100
    SIMULATION_SEED: int = 42
    
    # Detection Settings
    DETECTION_WINDOW_HOURS: int = 24
    ANOMALY_Z_SCORE_THRESHOLD: float = 2.0
    BOTTLENECK_ALERT_THRESHOLD_DTD: float = 45.0  # minutes
    
    # Data Settings
    MAX_EVENTS_PER_MINUTE: int = 1000
    DATA_RETENTION_DAYS: int = 7
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

