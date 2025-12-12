# Master Cursor Prompt: ED Bottleneck Engine MVP (Production-Ready)

## Context
This is a **production-ready MVP** for a Real-Time ED Bottleneck Engine with a Natural-Language-First Interface. The system is already 80% built with advanced features including causal inference, RL optimization, and ML-calibrated simulations. This prompt serves as both documentation and a blueprint for enhancements.

### Quick Implementation Status
- ✅ **Fully Implemented**: Detection, Simulation, Basic Optimization, Chat Interface, Data Generation, ROI Calculator
- ⚠️ **Implemented but Disabled**: Causal Inference (performance), Deep Analysis/InsightEngine (performance)
- 🔧 **Partially Implemented**: RL Optimization (simplified PPO-style, not full Stable-Baselines3), Equity-Aware Optimization (partial SDOH integration)
- 📋 **Planned/Placeholder**: Full RL (Stable-Baselines3), Predictive Forecasting (N-BEATS/TSiT+), Advanced Visualizations (ReactFlow, Sankey)

## Current System Status

### ✅ Implemented Features
- **Backend**: FastAPI with async endpoints, Pydantic schemas, ClickHouse + Redis
- **Frontend**: React 18 + Tailwind CSS, Vite build, responsive UI
- **Detection**: Real wait-time calculation from patient journeys, anomaly detection (IsolationForest)
- **Simulation**: SimPy DES with ML calibration, ESI-based acuity, realistic flows (labs, imaging, beds)
- **Optimization**: 
  - **Basic**: Linear Programming (PuLP) with realistic constraints (max 2 doctors, 3 nurses, 2 techs)
  - **Advanced**: Simplified PPO-style RL optimizer (`advanced_optimization.py`) with policy weights, Monte Carlo rollouts
- **Causal Inference**: DoWhy DAGs, pgmpy Bayesian networks, SHAP attributions (fully implemented, currently disabled for performance)
- **Chat Interface**: ChatGPT-like conversational AI with context management, intent recognition
- **Data Generation**: 
  - Basic generator (`generate_sample_data.py`)
  - Advanced generator (`generate_sample_data_advanced.py`) with SDOH integration, iterative validation
- **ROI Calculator**: Quantified impact, payback periods, daily cost/savings
- **Insight Engine**: Scalable analysis framework for all metrics (LWBS, DTD, LOS, throughput, bed utilization)
- **Operational Analysis**: Patient journey tracing, throughput analysis, resource utilization patterns

### ⚠️ Currently Disabled (For Performance)
- **Causal Analysis**: Temporarily disabled (`ENABLE_CAUSAL_ANALYSIS = False` in `detection.py:67`)
  - **Reason**: Timeout issues with complex causal graphs
  - **Re-enable**: Set to `True` after adding proper caching and timeout handling
- **Deep Analysis (InsightEngine)**: Temporarily disabled (`ENABLE_DEEP_ANALYSIS = False` in `conversational_ai.py:212`)
  - **Reason**: Performance optimization for chat responsiveness
  - **Re-enable**: Set to `True` after optimizing InsightEngine queries

### 🎯 Enhancement Opportunities
- **Full RL Optimization**: Upgrade simplified PPO-style to full Stable-Baselines3 PPO with Gymnasium environment
- **Predictive Forecasting**: Add N-BEATS/TSiT+ for 2-4 hour surge prediction
- **Real-time Visualization**: ReactFlow for causal DAGs, Sankey flows for patient cascades
- **Equity-Aware Optimization**: Full SDOH stratification in optimization (currently partial in `advanced_optimization.py`)
- **Federated Learning**: Multi-site aggregation for causal graphs and simulation parameters
- **Agent-Based Modeling**: Mesa integration for staff awareness states and emergent behavior

---

## Phase 1: Repository Structure & Data Layer

### Current Structure
```
ed-bottleneck-engine/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, CORS, routers
│   │   ├── core/
│   │   │   ├── config.py        # Settings (ClickHouse, Redis, OpenAI)
│   │   │   ├── detection.py     # BottleneckDetector (real wait times)
│   │   │   ├── simulation.py   # EDSimulation (SimPy DES + ML calibration)
│   │   │   ├── optimization.py  # Optimizer (LP with realistic constraints)
│   │   │   ├── causal_inference.py  # CausalInferenceEngine (DoWhy + pgmpy)
│   │   │   ├── causal_narrative.py # LLM-powered narrative generation
│   │   │   ├── conversational_ai.py # ChatGPT-like interface
│   │   │   ├── conversation_manager.py # Context management
│   │   │   ├── insight_engine.py    # Scalable analysis framework
│   │   │   ├── lwbs_analysis.py     # Deep LWBS insights
│   │   │   ├── roi_calculator.py    # ROI calculations
│   │   │   ├── advanced_detection.py  # Multivariate patterns, rare anomalies
│   │   │   ├── advanced_optimization.py # RL-style optimizer (simplified PPO)
│   │   │   ├── ml_calibration.py    # Bayesian parameter estimation, GP surrogates
│   │   │   ├── operational_analysis.py # Patient journey analysis, throughput
│   │   │   ├── root_cause_analysis.py  # Multi-level RCA (immediate/underlying/systemic)
│   │   │   ├── data_validation.py   # KS-tests, distribution matching
│   │   │   ├── intelligent_router.py # OpenAI-powered intent classification
│   │   │   └── nlp.py              # Natural language processing utilities
│   │   ├── data/
│   │   │   ├── schemas.py       # Pydantic models (EDEvent, Bottleneck, etc.)
│   │   │   ├── storage.py       # ClickHouse + Redis operations
│   │   │   └── ingestion.py    # CSV/JSON ingestion, KPI calculation
│   │   └── routers/
│   │       ├── ingest.py        # /ingest/csv, /ingest/generate-advanced
│   │       ├── metrics.py       # /metrics (real-time KPIs)
│   │       ├── detect.py        # /detect (bottleneck detection)
│   │       ├── simulate.py      # /simulate (what-if scenarios)
│   │       ├── optimize.py      # /optimize (resource recommendations)
│   │       ├── chat.py          # /chat (conversational AI)
│   │       └── advisor.py       # /advisor (action plans)
│   ├── generate_sample_data.py         # Basic data generator
│   ├── generate_sample_data_advanced.py # Advanced generator (SDOH, validation)
│   └── requirements.txt         # Dependencies (FastAPI, SimPy, DoWhy, etc.)
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # React Router setup
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx   # Main dashboard (human-readable text)
│   │   │   └── Chat.jsx         # Conversational interface
│   │   ├── components/
│   │   │   ├── SimulationResult.jsx  # Simulation results display
│   │   │   ├── ActionPlan.jsx        # Detailed action plans
│   │   │   └── BottleneckReport.jsx   # Bottleneck details
│   │   └── services/
│   │       └── api.js           # Axios client for backend
│   └── package.json
├── docker-compose.yml           # Multi-container setup
└── README.md
```

### Data Schema (Pydantic)
```python
# app/data/schemas.py
class EDEvent(BaseModel):
    timestamp: datetime
    event_type: EventType  # arrival, triage, doctor_visit, labs, imaging, bed_assign, discharge, lwbs
    patient_id: str
    stage: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    duration_minutes: Optional[float]
    esi: Optional[int]  # Emergency Severity Index (1-5)
    metadata: Optional[Dict[str, Any]] = None

class Bottleneck(BaseModel):
    bottleneck_name: str
    stage: str
    impact_score: float  # 0-1
    current_wait_time_minutes: float
    causes: List[str]
    severity: str  # low, medium, high, critical
    recommendations: List[str]
    metadata: Optional[Dict[str, Any]] = None  # Causal analysis, narratives

class SimulationRequest(BaseModel):
    scenario: List[ScenarioChange]  # add/remove/shift resources
    window_hours: int = 24

class OptimizationRequest(BaseModel):
    constraints: Dict[str, Any]  # budget, staff_max, max_doctors, max_nurses, max_techs
    target_metrics: List[str]  # ["dtd", "lwbs", "los"]
```

### Data Ingestion
- **CSV Format**: `timestamp,event_type,patient_id,stage,resource_type,resource_id,duration_minutes`
- **Storage**: ClickHouse (time-series queries), Redis (hot cache)
- **KPI Calculation**: Hourly aggregates (DTD, LOS, LWBS, bed_utilization, queue_length)
- **Advanced Generator**: `generate_sample_data_advanced.py` with:
  - SDOH integration (transport delays, access scores)
  - Iterative validation (KS-tests)
  - Tuned parameters (LWBS 1.1-1.8%, LOS 4-5h)
  - Behavioral health tails (5% extended stays)

---

## Phase 2: Core Engines (Current Implementation)

### 1. Detection Engine (`app/core/detection.py`)

**Current Implementation:**
- **Real Wait-Time Calculation**: Tracks patient journeys (arrival → triage → doctor → labs/imaging → bed → discharge)
- **P95 Wait Times**: Uses 95th percentile from actual data (capped at 2 hours)
- **Anomaly Detection**: 
  - Z-score based (DTD, LOS, LWBS spikes)
  - IsolationForest for multivariate anomalies (already implemented)
- **Stages**: Triage, Doctor, Labs, Imaging, Bed queues
- **Caching**: Redis cache for bottleneck results (5-minute TTL)

**Advanced Detection (`app/core/advanced_detection.py`):**
- **Multivariate Pattern Detection**: SHAP-like feature attribution
- **Rare Anomaly Detection**: Ensemble methods (IsolationForest + LocalOutlierFactor)
- **Placeholders**: Causal inference integration, predictive foresight

**Enhancement Opportunities:**
```python
# Add predictive forecasting (not yet implemented)
from tsai.all import TSiTPlus
forecast = TSiTPlus.forecast(kpis, horizon=2)  # 2-hour ahead

# Note: IsolationForest is already implemented in detection.py
# from sklearn.ensemble import IsolationForest
# self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
```

### 2. Simulation Engine (`app/core/simulation.py`)

**Current Implementation:**
- **SimPy DES**: Discrete-event simulation with realistic patient flow
- **ML Calibration**: Bayesian parameter estimation, GP surrogates for service times
- **ESI-Based Acuity**: Different service times and routing by ESI (1-5)
- **Realistic Flows**: Labs, imaging, bed assignment, LWBS logic
- **Baseline Comparison**: Runs baseline simulation, compares deltas

**Key Features:**
- Poisson arrivals with time-of-day patterns
- Log-normal service times (calibrated via GP)
- Resource efficiency: More resources → faster service
- Monte Carlo iterations (reduced for speed)

**Enhancement Opportunities:**
```python
# Add agent-based modeling (Mesa)
from mesa import Agent, Model
class StaffAgent(Agent):
    def step(self):
        if self.awareness < 0.5:
            self.delay_multiplier = 1.15  # 15% delay if low awareness
```

### 3. Optimization Engine

**Basic Optimizer (`app/core/optimization.py`):**
- **Linear Programming (PuLP)**: Constraint-based optimization
- **Realistic Constraints**: Max 2 doctors, 3 nurses, 2 techs per suggestion
- **ROI Integration**: Cost/savings calculations
- **Budget Constraints**: Respects total budget limits

**Advanced Optimizer (`app/core/advanced_optimization.py`):**
- **Simplified PPO-Style RL**: Policy weights for resource allocation decisions
- **Monte Carlo Rollouts**: Stochastic optimization with variance estimation
- **Equity-Aware**: SDOH penalties, LWBS disparity tracking (partial implementation)
- **Historical Learning**: Policy updates from past simulation results
- **Realistic Limits**: Enforces max_doctors=2, max_nurses=3, max_techs=2

**Current RL Implementation (Simplified):**
```python
# Simplified policy weights (not full PPO)
self.policy_weights = {
    "nurse": {"dtd": -15.0, "lwbs": -10.0, "equity": 0.1},
    "doctor": {"dtd": -20.0, "lwbs": -15.0, "equity": 0.15},
    "tech": {"dtd": -8.0, "los": -5.0, "equity": 0.05}
}
```

**Enhancement to Full RL:**
```python
# Upgrade to full Stable-Baselines3 PPO
from stable_baselines3 import PPO
from gymnasium import Env, spaces

class EDOptimizationEnv(Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,))
        self.action_space = spaces.MultiDiscrete([3, 3, 3])  # +1/0/-1 for each resource
        self.state = kpis  # Current KPIs
    
    def step(self, action):
        # Apply action, run sim, calculate reward
        reward = -dtd_change + throughput_gain - cost
        return next_state, reward, done, info

model = PPO("MlpPolicy", env)
model.learn(total_timesteps=10000)
```

### 4. Causal Inference Engine (`app/core/causal_inference.py`)

**Current Implementation (Fully Built, Disabled for Performance):**
- **DoWhy DAGs**: Domain-specific causal graphs (imaging, labs, bed, doctor)
- **ATE Estimation**: Average Treatment Effect with confidence intervals (95% CI)
- **Bayesian Networks (pgmpy)**: Probabilistic inference with VariableElimination
- **SHAP Attributions**: Feature importance percentages
- **Counterfactuals**: "What if" scenarios with ROI calculations
- **Equity Analysis**: Disparity detection by acuity/SDOH (high vs. low acuity wait times)
- **Variance Explained**: R² decomposition for each factor
- **Confidence Scoring**: Overall and per-component confidence scores

**Current Status:**
- **Location**: `detection.py:67` - `ENABLE_CAUSAL_ANALYSIS = False`
- **Timeout**: 5 seconds for causal analysis, 3 seconds for narrative generation
- **Limitation**: Only analyzes top 2 bottlenecks to avoid timeout

**Re-enable with Optimizations:**
```python
# In detection.py, set:
ENABLE_CAUSAL_ANALYSIS = True  # After adding optimizations

# Add caching for causal results (recommended)
from functools import lru_cache
@lru_cache(maxsize=10)
def cached_causal_analysis(bottleneck_hash):
    # Cache results for 5 minutes
    pass

# Add async timeout protection (already implemented)
import asyncio
causal_analysis = await asyncio.wait_for(
    causal_engine.analyze_bottleneck_causality(...),
    timeout=5.0
)
```

### 5. Conversational AI (`app/core/conversational_ai.py`)

**Current Implementation:**
- **Context Management**: Multi-turn conversations via `ConversationManager`
- **Intent Recognition**: OpenAI-powered query parsing with tool schemas
- **Action Execution**: Routes to simulation, detection, advisor, bottleneck analysis
- **Response Generation**: Natural language with quantified insights, root cause analysis
- **Fallback Handling**: Graceful degradation when OpenAI is unavailable
- **JSON Safety**: Cleans float('inf') and NaN values for JSON compliance

**Components:**
- **ConversationManager** (`conversation_manager.py`): Maintains message history, intent recognition
- **IntelligentRouter** (`intelligent_router.py`): OpenAI-powered query classification
- **InsightEngine** (`insight_engine.py`): Scalable deep analysis (currently disabled)

**Current Status:**
- **Deep Analysis**: Disabled (`ENABLE_DEEP_ANALYSIS = False` in `conversational_ai.py:212`)
- **Reason**: Performance optimization for chat responsiveness
- **Fallback**: Uses basic bottleneck detection and simulation when deep analysis is disabled

**Enhancement Opportunities:**
- Add function calling for structured tool use (OpenAI Functions API)
- Implement streaming responses for long analyses (Server-Sent Events)
- Add conversation export (PDF/CSV) with `jsPDF` or `react-csv`
- Re-enable InsightEngine with optimized queries and caching

---

## Phase 3: Frontend & Visualization

### Current Implementation
- **Chat Interface**: Natural language input, message history
- **Dashboard**: Human-readable text reports (no charts per user request)
- **Simulation Results**: Detailed explanations with "Why this happens"
- **Action Plans**: Structured recommendations with ROI

### Enhancement Opportunities

#### 1. Interactive Visualizations
```typescript
// Add ReactFlow for causal DAGs
import ReactFlow from 'reactflow';

function CausalGraph({ graphDOT }) {
  const nodes = parseDOT(graphDOT);
  return <ReactFlow nodes={nodes} edges={edges} />;
}

// Add Sankey for flow cascades
import { Sankey } from 'recharts';

function FlowCascade({ bottlenecks }) {
  return <Sankey data={buildFlowData(bottlenecks)} />;
}
```

#### 2. Real-Time Updates
```typescript
// WebSocket for live KPI updates
const ws = new WebSocket('ws://localhost:8000/ws/metrics');
ws.onmessage = (event) => {
  setMetrics(JSON.parse(event.data));
};
```

#### 3. Export Functionality
```typescript
// PDF export for reports
import jsPDF from 'jspdf';
function exportToPDF(report) {
  const doc = new jsPDF();
  doc.text(report, 10, 10);
  doc.save('ed-report.pdf');
}
```

---

## Phase 4: Advanced Features (Future Enhancements)

### 1. Predictive Forecasting
```python
# app/core/forecasting.py
from tsai.all import TSiTPlus, InceptionTime

class EDForecaster:
    def __init__(self):
        self.model = TSiTPlus(d_model=64, n_heads=4)
    
    async def forecast_surge(self, horizon_hours=2):
        # Forecast patient arrivals 2 hours ahead
        # Integrate with weather/seasonal data
        pass
```

### 2. Full RL Optimization (Upgrade from Simplified PPO-Style)
**Current**: Simplified PPO-style with policy weights in `advanced_optimization.py`
**Target**: Full Stable-Baselines3 PPO with Gymnasium environment

```python
# app/core/rl_optimization.py (NEW FILE)
from stable_baselines3 import PPO
from gymnasium import Env, spaces

class EDOptimizationEnv(Env):
    """RL environment for dynamic staffing decisions."""
    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,))
        self.action_space = spaces.MultiDiscrete([3, 3, 3])  # +1/0/-1 for each resource
        
    def step(self, action):
        # Apply action, run simulation, calculate reward
        reward = -dtd_change + throughput_gain - cost
        return next_state, reward, done, info

# Training
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("ed_optimization_ppo")
```

**Dependencies to Add:**
```bash
# Add to requirements.txt
stable-baselines3==2.2.1
gymnasium==0.29.1
```

### 3. Equity-Aware Optimization
```python
# app/core/equity_optimizer.py
class EquityAwareOptimizer:
    def optimize(self, request, equity_mode=True):
        if equity_mode:
            # Penalize solutions that increase LWBS disparity
            # Track SDOH proxies (access scores, transport delays)
            # Ensure low-SES patients benefit proportionally
            pass
```

### 4. Federated Learning (Multi-Site)
```python
# app/core/federated_learning.py
class FederatedAggregator:
    async def aggregate_models(self, site_models):
        # Aggregate causal graphs, simulation parameters
        # Preserve PHI safety (aggregates only)
        pass
```

---

## Phase 5: Testing & Documentation

### Current Test Coverage
- Unit tests for simulation logic
- Integration tests for API endpoints
- Data validation tests

### Enhancement Opportunities
```python
# tests/test_simulation.py
def test_simulation_accuracy():
    """Simulation should be within 10% of historical baselines."""
    result = run_simulation(scenario)
    assert abs(result.dtd_delta - expected) < 0.1

# tests/test_causal_inference.py
def test_ate_confidence_intervals():
    """ATE should have valid 95% CI."""
    ate = estimate_ate(data, treatment, outcome)
    assert ate.ci_lower < ate.value < ate.ci_upper
```

---

## Quick Start Commands

### Generate Realistic Data
```bash
# Option 1: Use advanced generator script
./backend/generate_advanced_data.sh

# Option 2: Use API endpoint
curl -X POST "http://localhost:8000/api/ingest/generate-advanced?num_patients=500&days=2"

# Option 3: Upload existing CSV
# Via frontend: Upload CSV Data button
# Or via curl:
curl -X POST "http://localhost:8000/api/ingest/csv" -F "file=@backend/sample_data_advanced.csv"
```

### Run System
```bash
docker compose up -d
# Backend: http://localhost:8000
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

### Test Chat
```bash
# Ask questions like:
# - "What are my current bottlenecks?"
# - "What if we add 2 nurses?"
# - "What should I do to reduce wait times?"
```

---

## Key Configuration

### Environment Variables (`.env`)
```
OPENAI_API_KEY=sk-proj-...
CLICKHOUSE_HOST=clickhouse
CLICKHOUSE_PORT=8123
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=
REDIS_HOST=redis
REDIS_PORT=6379
```

### Performance Tuning
```python
# In detection.py:
ENABLE_CAUSAL_ANALYSIS = False  # Set to True after optimization
CAUSAL_TIMEOUT = 5.0  # seconds

# In conversational_ai.py:
ENABLE_DEEP_ANALYSIS = False  # Set to True after optimization
DEEP_ANALYSIS_TIMEOUT = 10.0  # seconds
```

---

## Enhancement Roadmap

### Immediate (Week 1-2)
1. **Re-enable Causal Analysis** with proper timeouts and caching
2. **Add Predictive Forecasting** (N-BEATS/TSiT+ for 2-4h ahead)
3. **Enhance Visualizations** (ReactFlow DAGs, Sankey flows)

### Short-term (Month 1)
4. **RL Optimization** (PPO for dynamic staffing)
5. **Equity-Aware Optimization** (SDOH stratification, disparity tracking)
6. **Real-Time WebSocket Updates** (live KPI streaming)

### Long-term (Month 2-3)
7. **Federated Learning** (multi-site aggregation)
8. **EHR Integration** (Epic FHIR hooks)
9. **Advanced Visualizations** (interactive DAGs, flow cascades)

---

## Cursor Chat Examples

After generating the base system, use these prompts for iterative improvements:

```
"Add SHAP visualization to the causal inference results in the frontend"
"Implement N-BEATS forecasting for 2-hour surge prediction"
"Add PPO RL optimization with Stable-Baselines3"
"Create ReactFlow component to visualize causal DAGs"
"Add equity-aware constraints to optimization (penalize LWBS disparity)"
"Implement WebSocket for real-time KPI updates"
"Add PDF export for action plans and bottleneck reports"
"Create federated learning aggregator for multi-site deployment"
```

---

## Success Metrics

### Performance Targets
- **Simulation**: <10s per scenario
- **Detection**: <5s for 48h window
- **Chat Response**: <30s for complex queries
- **Accuracy**: Simulation within 10% of historical baselines

### Business Targets
- **DTD Reduction**: 10-20%
- **LOS Reduction**: 10-20%
- **LWBS Reduction**: >20% (from baseline to <2%)
- **ROI**: Positive payback within 6 months

---

## Notes for Cursor

1. **Start with Phase 1**: Generate skeleton, then iterate
2. **Use Existing Code**: Reference current implementations for patterns
3. **Focus on Integration**: Ensure all components work together
4. **Test Incrementally**: Validate each phase before moving forward
5. **Document Decisions**: Add comments explaining design choices

This prompt provides a complete blueprint for building and enhancing the ED Bottleneck Engine. Use it iteratively with Cursor to add features, fix bugs, and optimize performance.

