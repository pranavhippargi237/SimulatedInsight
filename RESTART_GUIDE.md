# 🔄 Restart Guide with Extensive Logging

This guide will help you restart both backend and frontend with extensive logging enabled.

## 🛑 Step 1: Stop All Running Services

First, stop any existing processes:

```bash
# Kill backend processes
pkill -f "uvicorn app.main:app"

# Kill frontend processes (if running in terminal)
pkill -f "vite"
```

## 🚀 Step 2: Start Backend Server

### Option A: Using the restart script (Recommended)
```bash
./restart_backend.sh
```

### Option B: Manual start
```bash
cd backend

# Activate virtual environment (if using one)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start with verbose logging
uvicorn app.main:app --reload --port 8000 --log-level info
```

**Expected Backend Output:**
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

**Verify Backend is Running:**
```bash
curl http://localhost:8000/api/health
```

You should see a JSON response with status "healthy".

## 🎨 Step 3: Start Frontend Server

### Option A: Using the restart script
```bash
# In a NEW terminal window
./restart_frontend.sh
```

### Option B: Manual start
```bash
cd frontend
npm run dev
```

**Expected Frontend Output:**
```
  VITE vX.X.X  ready in XXX ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

## 📊 Step 4: Monitor Logs

### Backend Logs Will Show:
- ✅ Startup sequence with timestamps
- 🏥 Health check requests with timing
- 📤 File upload requests with file details
- 📊 Processing progress for large files
- ⏱️ Timing information for all operations
- ❌ Any errors with full stack traces

### Frontend Console Will Show:
- `[Chat]` - Chat page operations
- `[API]` - API call details
- Timing information for all requests
- Detailed error information

## 🔍 Troubleshooting

### Backend Not Starting?
1. Check if port 8000 is already in use:
   ```bash
   lsof -i :8000
   ```

2. Check Python version (needs 3.11+):
   ```bash
   python --version
   ```

3. Check if dependencies are installed:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

### Frontend Not Connecting?
1. Check browser console for detailed error messages
2. Verify backend is running: `curl http://localhost:8000/api/health`
3. Check CORS settings in backend logs

### Upload Still Failing?
1. Check backend logs for detailed error messages
2. Check file size (should be reasonable)
3. Verify file format matches expected CSV structure
4. Look for timeout messages in both frontend and backend logs

## 📝 Log Locations

- **Backend**: Terminal where you ran `uvicorn`
- **Frontend**: Browser console (F12 → Console tab)
- **Network**: Browser DevTools → Network tab (shows all API calls)

## ✅ Success Indicators

**Backend:**
- ✅ "API startup complete" message
- ✅ "Uvicorn running on http://127.0.0.1:8000"
- ✅ Health check returns 200 OK

**Frontend:**
- ✅ No backend connection errors in console
- ✅ Health check succeeds
- ✅ Can upload files without timeout
