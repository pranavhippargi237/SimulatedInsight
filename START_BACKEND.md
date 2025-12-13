# 🚀 Start Backend Server - Quick Guide

## The Problem
The backend server is **NOT running**. That's why all requests are timing out.

## ✅ Solution: Start the Backend

### Option 1: Using the Restart Script (Easiest)

Open a **NEW terminal window** and run:

```bash
cd "/Users/pranavhippargi/Desktop/Simulated Insights"
./restart_backend.sh
```

### Option 2: Manual Start

Open a **NEW terminal window** and run:

```bash
# Navigate to backend directory
cd "/Users/pranavhippargi/Desktop/Simulated Insights/backend"

# Activate virtual environment (if you have one)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start the server
uvicorn app.main:app --reload --port 8000
```

## ✅ What You Should See

When the backend starts successfully, you'll see:

```
============================================================
🚀 Starting ED Bottleneck Engine API
============================================================
[HH:MM:SS] 🔄 Starting API initialization...
[HH:MM:SS] 📦 Initializing storage...
[HH:MM:SS] ✅ API startup complete in X.XXXs
============================================================
🌐 API is ready and listening on http://localhost:8000
📚 API docs available at http://localhost:8000/docs
============================================================
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

## 🔍 Verify It's Working

In another terminal, test the health endpoint:

```bash
curl http://localhost:8000/api/health
```

You should get a JSON response like:
```json
{
  "status": "healthy",
  "timestamp": "2024-...",
  "service": "ED Bottleneck Engine API",
  "storage": {
    "sqlite": true,
    "sqlite_status": "connected"
  }
}
```

## ⚠️ Common Issues

### Port 8000 Already in Use
If you see "Address already in use", kill the existing process:
```bash
lsof -ti:8000 | xargs kill -9
```

### Python/Module Not Found
Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

### Import Errors
Make sure you're in the backend directory and running:
```bash
uvicorn app.main:app --reload --port 8000
```

## 📊 Once Backend is Running

1. **Refresh your browser** - The frontend should now connect
2. **Check browser console** - You should see `[API] ✅ Health check succeeded`
3. **Try uploading a file** - It should work now!

## 🎯 Keep the Terminal Open

**IMPORTANT**: Keep the terminal window with the backend server **OPEN**. If you close it, the server stops!
