# API Documentation

## Base URL

```
http://localhost:8000/api
```

## Authentication

Currently, the MVP does not require authentication. For production, implement JWT or OAuth2.

## Endpoints

### Health Check

**GET** `/health`

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "service": "ED Bottleneck Engine API"
}
```

---

### Ingest Data

#### CSV Upload

**POST** `/ingest/csv`

Upload ED events from CSV file.

**Request:**
- Content-Type: `multipart/form-data`
- Body: CSV file

**Response:**
```json
{
  "status": "ok",
  "processed": 500,
  "invalid": 0,
  "total": 500
}
```

#### JSON Upload

**POST** `/ingest/json`

Upload ED events as JSON array.

**Request Body:**
```json
[
  {
    "timestamp": "2024-01-15T10:30:00Z",
    "event_type": "arrival",
    "patient_id": "anon_patient_123",
    "stage": "triage",
    "resource_type": "nurse",
    "duration_minutes": 5.0
  }
]
```

**Response:**
```json
{
  "status": "ok",
  "processed": 1,
  "invalid": 0,
  "total": 1
}
```

---

### Get Metrics

**GET** `/metrics`

Get real-time KPIs and historical trends.

**Query Parameters:**
- `window` (string, default: "24h"): Time window (e.g., "24h", "48h")
- `include_anomalies` (boolean, default: true): Include anomaly detection

**Response:**
```json
{
  "status": "ok",
  "window_hours": 24,
  "current_metrics": {
    "dtd": 35.0,
    "los": 180.0,
    "lwbs": 0.08,
    "bed_utilization": 0.75,
    "queue_length": 5
  },
  "historical_kpis": [...],
  "anomalies": [
    {
      "metric": "dtd",
      "severity": "high",
      "value": 55.0,
      "z_score": 2.5,
      "timestamp": "2024-01-15T14:00:00Z"
    }
  ],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

### Detect Bottlenecks

**POST** `/detect`

Detect bottlenecks in ED operations.

**Query Parameters:**
- `window_hours` (int, default: 24): Time window to analyze
- `top_n` (int, default: 3): Number of top bottlenecks to return

**Response:**
```json
{
  "status": "ok",
  "window_hours": 24,
  "bottlenecks": [
    {
      "bottleneck_name": "Imaging Queue",
      "stage": "imaging",
      "impact_score": 0.25,
      "current_wait_time_minutes": 45.0,
      "causes": ["staff: -2", "equipment: unavailable"],
      "severity": "high",
      "recommendations": ["Add 1 imaging tech", "Cross-train staff"]
    }
  ],
  "count": 1
}
```

---

### Run Simulation

#### Structured Request

**POST** `/simulate`

Run simulation with structured scenario.

**Request Body:**
```json
{
  "scenario": {
    "action": "add",
    "resource_type": "nurse",
    "quantity": 2,
    "time_start": "14:00",
    "time_end": "18:00",
    "day": "Saturday"
  },
  "simulation_hours": 24,
  "iterations": 100
}
```

**Response:**
```json
{
  "scenario_id": "sim_abc123",
  "baseline_metrics": {
    "dtd": 35.0,
    "los": 180.0,
    "lwbs": 0.08
  },
  "predicted_metrics": {
    "dtd": 28.0,
    "los": 165.0,
    "lwbs": 0.05
  },
  "deltas": {
    "dtd_change": -20.0,
    "los_change": -8.3,
    "lwbs_drop": -37.5
  },
  "confidence": 0.85,
  "execution_time_seconds": 8.5
}
```

#### Natural Language

**POST** `/simulate/nlp`

Run simulation from natural language query.

**Request Body:**
```json
{
  "query": "What if we add two nurses during peak hours on weekends?"
}
```

**Response:**
```json
{
  "status": "ok",
  "original_query": "What if we add two nurses during peak hours on weekends?",
  "parsed_scenario": {
    "scenario": {...},
    "confidence": 0.9,
    "original_query": "...",
    "suggestions": []
  },
  "simulation_result": {...}
}
```

---

### Optimize

**POST** `/optimize`

Generate optimization suggestions.

**Request Body:**
```json
{
  "bottlenecks": ["Imaging Queue", "Triage Queue"],
  "constraints": {
    "staff_max": 10,
    "budget": 1000
  },
  "objective": "minimize_dtd"
}
```

**Response:**
```json
{
  "status": "ok",
  "suggestions": [
    {
      "priority": 1,
      "action": "add",
      "resource_type": "nurse",
      "quantity": 2,
      "expected_impact": {
        "dtd_reduction": -15.0,
        "lwbs_drop": -10.0
      },
      "cost": 100.0,
      "confidence": 0.8
    }
  ],
  "count": 1
}
```

---

## Error Responses

All endpoints return standard HTTP status codes:

- `200`: Success
- `422`: Validation error
- `500`: Internal server error

**Error Format:**
```json
{
  "detail": "Error message here"
}
```

---

## Rate Limiting

- Default: 10 requests per minute per IP
- Configurable via `RATE_LIMIT_PER_MINUTE` environment variable

---

## Data Schemas

See `backend/app/data/schemas.py` for full Pydantic schemas.

### Event Types

- `arrival`: Patient arrives
- `triage`: Triage assessment
- `bed_assign`: Bed assignment
- `doctor_visit`: Doctor consultation
- `imaging`: Imaging procedure
- `discharge`: Patient discharge
- `lwbs`: Left Without Being Seen

### Resource Types

- `nurse`: Nursing staff
- `doctor`: Physician
- `tech`: Technician (imaging, lab, etc.)
- `bed`: Bed resource

---

## Examples

### cURL Examples

```bash
# Health check
curl http://localhost:8000/api/health

# Get metrics
curl "http://localhost:8000/api/metrics?window=24h"

# Detect bottlenecks
curl -X POST "http://localhost:8000/api/detect?window_hours=24&top_n=3"

# Run NLP simulation
curl -X POST "http://localhost:8000/api/simulate/nlp" \
  -H "Content-Type: application/json" \
  -d '{"query": "Add 2 nurses from 2-6 PM"}'
```

---

For interactive API documentation, visit http://localhost:8000/docs


