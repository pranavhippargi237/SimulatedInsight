#!/bin/bash

echo "🔄 Restarting ED Bottleneck Engine Application..."
echo ""

# Kill all existing processes
echo "🛑 Stopping all processes..."
pkill -9 -f "uvicorn" 2>/dev/null
pkill -9 -f "vite" 2>/dev/null
pkill -9 -f "npm.*dev" 2>/dev/null
sleep 2

# Clear ports
echo "🧹 Clearing ports..."
lsof -ti:8000,3000,5173 2>/dev/null | xargs kill -9 2>/dev/null
sleep 1

# Start backend
echo "🚀 Starting backend..."
cd "$(dirname "$0")/backend" || exit 1
if [ -d "venv" ]; then
    source venv/bin/activate
fi
nohup uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload --log-level info > backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > backend.pid
sleep 3

# Verify backend
if curl -s --max-time 3 http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "✅ Backend is running (PID: $BACKEND_PID)"
else
    echo "❌ Backend failed to start. Check backend/backend.log"
    exit 1
fi

# Start frontend
echo "🎨 Starting frontend..."
cd "$(dirname "$0")/frontend" || exit 1
nohup npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > frontend.pid
sleep 5

# Verify frontend
FRONTEND_URL=""
if curl -s --max-time 2 http://localhost:5173 > /dev/null 2>&1; then
    FRONTEND_URL="http://localhost:5173"
elif curl -s --max-time 2 http://localhost:3000 > /dev/null 2>&1; then
    FRONTEND_URL="http://localhost:3000"
fi

if [ -n "$FRONTEND_URL" ]; then
    echo "✅ Frontend is running (PID: $FRONTEND_PID)"
    echo ""
    echo "=========================================="
    echo "✅ Application restarted successfully!"
    echo "=========================================="
    echo "🌐 Backend:  http://localhost:8000"
    echo "🌐 Frontend: $FRONTEND_URL"
    echo "📚 API Docs: http://localhost:8000/docs"
    echo ""
    echo "📝 Logs:"
    echo "   Backend:  backend/backend.log"
    echo "   Frontend: frontend/frontend.log"
    echo ""
else
    echo "⚠️  Frontend may still be starting. Check frontend/frontend.log"
fi
