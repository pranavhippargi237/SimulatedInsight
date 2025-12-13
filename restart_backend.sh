#!/bin/bash

# Script to restart the backend server with extensive logging

echo "=========================================="
echo "🔄 Restarting Backend Server"
echo "=========================================="

# Kill any existing backend processes
echo "🛑 Stopping any existing backend processes..."
pkill -f "uvicorn app.main:app" || echo "No existing processes found"

# Wait a moment for processes to stop
sleep 2

# Navigate to backend directory
cd "$(dirname "$0")/backend" || exit 1

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🐍 Activating virtual environment..."
    source venv/bin/activate
fi

# Start the backend with verbose logging
echo "🚀 Starting backend server on http://localhost:8000..."
echo "📚 API docs will be available at http://localhost:8000/docs"
echo "=========================================="
echo ""

uvicorn app.main:app --reload --port 8000 --log-level info
