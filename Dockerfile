# Root-level Dockerfile that points to backend
# This is for Render.com when Root Directory is not set
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from backend
COPY backend/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy entire backend directory (including app/data)
COPY backend/ ./backend/
WORKDIR /app/backend

# Debug: Verify app/data exists and list contents
RUN echo "=== Checking app structure ===" && \
    ls -la app/ && \
    echo "=== Checking app/data ===" && \
    ls -la app/data/ 2>/dev/null && \
    echo "=== app/data Python files ===" && \
    ls -la app/data/*.py 2>/dev/null || (echo "ERROR: app/data/*.py not found!" && find . -name "*.py" -path "*/data/*" | head -5)

# Set PYTHONPATH so Python can find the app module
ENV PYTHONPATH=/app/backend

# Expose port (Render sets PORT env var)
EXPOSE ${PORT:-8000}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/health || exit 1

# Run the application
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
