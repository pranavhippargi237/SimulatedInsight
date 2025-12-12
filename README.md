# ED Bottleneck Engine - Real-Time MVP

A natural-language-first system for detecting, simulating, and optimizing Emergency Department (ED) bottlenecks in real-time.

## 🎯 Overview

The ED Bottleneck Engine empowers hospital operations teams to:
- **Detect bottlenecks** in real-time using queueing models and anomaly detection
- **Simulate scenarios** using natural language queries (e.g., "What if we add two nurses during peak hours?")
- **Optimize operations** with AI-powered suggestions based on constraints

**MVP Success Metrics**:
- Simulations complete in <10s
- Real-time metrics update every 5s
- Bottleneck detection sensitivity >85%
- 90% natural-language query parsing accuracy

## 🏗️ Architecture

```
Frontend (React) → API Gateway (FastAPI) → Core Engines
                                         ├── Bottleneck Detector
                                         ├── Simulation Engine (SimPy)
                                         └── Optimization Layer
                                         
Data Flow: ClickHouse (OLAP) + Redis (Cache)
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for local development)
- Node.js 18+ (for local frontend development)
- OpenAI API key (optional, for enhanced NLP parsing)

### Running with Docker Compose

1. **Clone and navigate to the project**:
   ```bash
   cd "Simulated Insights"
   ```

2. **Set environment variables** (optional):
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

3. **Start all services**:
   ```bash
   docker-compose up --build
   ```

4. **Access the application**:
   - Frontend: http://localhost:3000
   - API Docs: http://localhost:8000/docs
   - ClickHouse: http://localhost:8123

### Local Development

#### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start ClickHouse and Redis (via Docker)
docker-compose up clickhouse redis -d

# Run the API
uvicorn app.main:app --reload --port 8000
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

## 📊 Data Ingestion

### Generate Sample Data

```bash
cd backend
python generate_sample_data.py
# Creates sample_data.csv with 1000 synthetic events
```

### Upload Data

1. **Via Frontend**: Use the Chat page → "Upload CSV Data" button
2. **Via API**:
   ```bash
   curl -X POST "http://localhost:8000/api/ingest/csv" \
     -F "file=@sample_data.csv"
   ```

### CSV Format

```csv
timestamp,event_type,patient_id,stage,resource_type,resource_id,duration_minutes
2024-01-15T10:30:00Z,arrival,anon_patient_123,triage,,,
2024-01-15T10:35:00Z,triage,anon_patient_123,triage,nurse,nurse_1,5.0
```

## 🎮 Usage

### Natural Language Queries

Use the **Chat** page to ask questions in plain English:

- "What if we add two nurses during peak hours on weekends?"
- "Simulate adding a triage nurse from 2-6 PM"
- "What happens if we add one doctor on Saturday?"

The system will:
1. Parse your query into a structured scenario
2. Run a discrete-event simulation
3. Show predicted impacts on DTD, LOS, and LWBS

### Dashboard

The **Dashboard** shows:
- Real-time KPIs (DTD, LOS, LWBS, Bed Utilization)
- Historical trends with anomaly alerts
- Detected bottlenecks with root causes and recommendations

### History

The **History** page tracks all simulations with export to CSV.

## 🔌 API Endpoints

### Health Check
```bash
GET /api/health
```

### Ingest Data
```bash
POST /api/ingest/csv
POST /api/ingest/json
```

### Get Metrics
```bash
GET /api/metrics?window=24h&include_anomalies=true
```

### Detect Bottlenecks
```bash
POST /api/detect?window_hours=24&top_n=3
```

### Run Simulation
```bash
POST /api/simulate
POST /api/simulate/nlp  # Natural language input
```

### Optimize
```bash
POST /api/optimize
```

See full API documentation at http://localhost:8000/docs

## 🧪 Testing

```bash
cd backend
pytest tests/ -v --cov=app
```

## 📁 Project Structure

```
ed-bottleneck-engine/
├── backend/
│   ├── app/
│   │   ├── core/          # Detection, simulation, optimization, NLP
│   │   ├── data/          # Ingestion, schemas, storage
│   │   ├── routers/       # API endpoints
│   │   └── main.py        # FastAPI app
│   ├── tests/             # Test suite
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── pages/         # Dashboard, Chat, History
│   │   └── services/      # API client
│   └── package.json
├── docker-compose.yml
└── README.md
```

## 🔧 Configuration

Environment variables (set in `.env` or docker-compose):

- `OPENAI_API_KEY`: For enhanced NLP parsing (optional)
- `CLICKHOUSE_HOST`: ClickHouse host (default: localhost)
- `REDIS_HOST`: Redis host (default: localhost)
- `CORS_ORIGINS`: Allowed CORS origins

## 📈 Performance

- **Simulation**: <10s for 100 Monte Carlo iterations
- **Detection**: <5s for 24h window analysis
- **Metrics**: Real-time updates every 5s
- **Cache**: Redis TTL = 1 hour (bottlenecks), 5s (metrics)

## 🛠️ Extending

### Add New ED Stage

1. Update `EDSimulation` in `backend/app/core/simulation.py`
2. Add stage to `PatientGenerator.process_patient()`
3. Update detection logic in `BottleneckDetector`

### Custom NLP Parsing

Modify `backend/app/core/nlp.py` to add domain-specific parsing rules.

## 📝 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📧 Support

For issues or questions, please open a GitHub issue.

---

**Built with**: FastAPI, React, SimPy, ClickHouse, Redis, OpenAI

