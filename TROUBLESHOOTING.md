# Troubleshooting Guide

## Common Issues and Solutions

### 1. Dashboard Shows "Loading..." Forever

**Symptoms:**
- Dashboard stuck on "Loading..." message
- Console shows "timeout of 30000ms exceeded" errors

**Causes:**
- Backend server not running
- ClickHouse database not running
- Redis not running
- Network connectivity issues

**Solutions:**

1. **Check if backend is running:**
   ```bash
   curl http://localhost:8000/api/health
   ```
   Should return: `{"status":"healthy",...}`

2. **Start backend server:**
   ```bash
   cd backend
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Start storage services (ClickHouse + Redis):**
   ```bash
   docker-compose up -d
   ```

4. **Verify storage connections:**
   - Check ClickHouse: `docker ps | grep clickhouse`
   - Check Redis: `docker ps | grep redis`

### 2. Chat Shows "Thinking..." and Never Responds

**Symptoms:**
- Chat interface shows "Thinking..." bubble
- No response after 30+ seconds
- Console shows timeout errors

**Causes:**
- Backend timeout (30s limit)
- Deep analysis taking too long
- Causal analysis hanging
- OpenAI API issues

**Solutions:**

1. **Check backend logs** for errors:
   ```bash
   # In backend terminal, look for error messages
   ```

2. **Try simpler queries first:**
   - "What are my bottlenecks?" (simpler)
   - Avoid complex queries until basic ones work

3. **Check if data is loaded:**
   - Upload a small CSV file first
   - Wait for upload to complete
   - Then try chat queries

4. **Disable heavy features temporarily:**
   - Deep analysis is now conditional (only for metric queries)
   - Causal analysis limited to top 1 bottleneck

### 3. Data Upload Hangs

**Symptoms:**
- Upload shows "Thinking..." 
- No completion message
- Console shows timeout errors

**Causes:**
- Large file taking too long
- ClickHouse not responding
- Network issues

**Solutions:**

1. **Try smaller files first** (< 1000 rows)

2. **Check ClickHouse status:**
   ```bash
   docker ps | grep clickhouse
   docker logs <clickhouse-container-id>
   ```

3. **Use advanced data generator instead:**
   ```bash
   curl -X POST "http://localhost:8000/api/ingest/generate-advanced?num_patients=100&days=1"
   ```

### 4. All Endpoints Timing Out

**Root Cause:** Backend server or storage services not running

**Quick Fix:**
```bash
# Terminal 1: Start storage
docker-compose up -d

# Terminal 2: Start backend
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 3: Start frontend (if not already running)
cd frontend
npm run dev
```

### 5. Storage Connection Issues

**Check storage health:**
```bash
curl http://localhost:8000/api/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "storage": {
    "clickhouse": true,
    "redis": true,
    "clickhouse_status": "connected",
    "redis_status": "connected"
  }
}
```

**If storage shows as false:**
- ClickHouse: Check `docker-compose.yml` and ensure ClickHouse container is running
- Redis: Check Redis container status
- Update `.env` file with correct connection details

## Performance Optimizations Applied

1. **Timeouts Added:**
   - Chat: 30 seconds global timeout
   - Upload: 60 seconds
   - KPI calculation: 30 seconds
   - Detection: 20 seconds
   - Storage queries: 10 seconds

2. **Caching:**
   - Causal analysis: 5-minute cache
   - Deep analysis: 5-minute cache
   - Metrics: 5-second cache

3. **Batch Processing:**
   - Events inserted in chunks of 1000
   - Query limits added (10,000 events max)

4. **Graceful Degradation:**
   - App starts even if ClickHouse/Redis unavailable
   - Endpoints return empty results instead of errors
   - Frontend shows helpful error messages

## Quick Health Check

Run this to verify everything is working:

```bash
# 1. Check backend
curl http://localhost:8000/api/health

# 2. Check if you can get metrics (should return empty if no data)
curl http://localhost:8000/api/metrics?window=24h

# 3. Check if you can detect bottlenecks (should return empty if no data)
curl -X POST http://localhost:8000/api/detect?window_hours=24&top_n=3
```

If all three work, the backend is running correctly. If they timeout, check storage services.
