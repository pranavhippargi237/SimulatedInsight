#!/bin/bash

# Kill any existing uvicorn processes
echo "🛑 Stopping any existing backend processes..."
pkill -f "uvicorn.*main" 2>/dev/null || true
sleep 1

# Navigate to backend directory
cd "$(dirname "$0")/backend" || exit 1

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Start the backend server
echo "🚀 Starting backend server on port 8000..."
echo "============================================================"
uvicorn app.main:app --reload --port 8000 --log-level info
