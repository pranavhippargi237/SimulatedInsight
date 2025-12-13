# SimulatedInsight: Improvement Roadmap to 15/10

## Current State: 6-7/10 MVP

**Strengths:**
- Clean monorepo structure
- ED-specific features (ESI branching, LWBS tracking)
- Docker-ready deployment
- Type-hinted, async-friendly code
- Basic simulation and detection working

**Weaknesses:**
- Rigid NL handling (fixed prompts)
- Shallow causal analysis (rules > DoWhy)
- Basic simulations (no RL/ABM)
- Limited error handling
- Sparse test coverage (~40-50%)

## Prioritized Roadmap

### Phase 1: Code Quality & Reliability (2-3 Days) - HIGH PRIORITY

**Goal:** Transform brittle MVP to reliable beta

#### 1.1 Comprehensive Error Handling
- [x] Add try/except blocks in simulation loops
- [x] Handle division by zero (empty data → safe defaults)
- [x] Validate input parameters
- [ ] Add graceful fallbacks for missing calibration data
- [ ] Implement retry logic for transient failures

**Files to Update:**
- `backend/app/core/simulation.py` - Add error handling in `run_simulation()`
- `backend/app/core/detection.py` - Handle empty KPI lists gracefully
- `backend/app/core/optimization.py` - Validate LP constraints

**Impact:** Prevents crashes on edge cases (zero patients, missing data)

#### 1.2 Enhanced Code Comments
- [x] Add module-level docstrings explaining algorithms
- [x] Document NHAMCS data sources and assumptions
- [ ] Add inline comments for complex logic
- [ ] Document parameter ranges and defaults

**Impact:** 2x faster onboarding for new developers

#### 1.3 Data-Driven Configurations
- [ ] Replace hardcoded deltas (e.g., -15% DTD/nurse) with fitted models
- [ ] Load calibration parameters from synth CSV via scikit-learn
- [ ] Make impact multipliers configurable via environment variables

**Impact:** More accurate simulations based on actual data patterns

---

### Phase 2: NL Dynamism (3-4 Days) - HIGH PRIORITY

**Goal:** True "ask anything" capability

#### 2.1 LangChain ReAct Agent Enhancement
- [x] Implement ReAct agent with dynamic tool selection
- [x] Add reflection loops ("Unclear filter? Suggest weekend?")
- [ ] Add multi-step tool chaining (corr → sim → opt)
- [ ] Implement query clarification prompts

**Files:**
- `backend/app/core/nl_agent.py` - Enhanced ReAct prompt
- `backend/app/core/conversational_ai.py` - Agent prioritization

**Impact:** Handles 90% of ED queries organically (vs. 30% currently)

#### 2.2 Advanced Correlation Tool
- [x] Add Polars-based correlation calculations
- [x] Return confidence intervals
- [ ] Support multiple metric correlations (psych → LWBS, LOS, DTD)
- [ ] Add temporal correlation (lag analysis)

**Files:**
- `backend/app/core/nl_agent.py` - `correlate_ed_metrics_tool`
- `backend/app/core/correlation_analysis.py` - Enhanced with Polars

**Impact:** Answers queries like "Psych corr to beds?" with quantified results

---

### Phase 3: Analysis Depth (4-5 Days) - HIGH PRIORITY

**Goal:** From generic recs to playbooks

#### 3.1 DoWhy Causal Integration
- [ ] Integrate DoWhy into `detection.py` for DAG-based RCA
- [ ] Add SHAP feature attribution (% variance explained)
- [ ] Calculate ATE with confidence intervals
- [ ] Identify confounders automatically

**Files:**
- `backend/app/core/detection.py` - Add `_causal_analysis_dowhy()`
- `backend/app/core/causal_inference.py` - Enhance existing module

**Impact:** 40% deeper root cause analysis (ATE -20 min, CI [-25,-15])

#### 3.2 Stable-Baselines3 PPO RL Upgrade
- [ ] Create Gym environment for sequential optimization
- [ ] Train PPO agent on historical ED data
- [ ] Generate RL-based recommendations (vs. rule-based)
- [ ] Add ROI calculations for RL actions

**Files:**
- `backend/app/core/rl_environment.py` - New Gym environment
- `backend/app/core/optimization.py` - Integrate RL recommendations

**Impact:** 2x better simulation ROI (RL learns optimal policies)

---

### Phase 4: UX & Scalability (2-3 Days) - MEDIUM PRIORITY

**Goal:** Production-ready performance

#### 4.1 Dynamic Rendering
- [ ] LangGraph node for visualization decisions
- [ ] Auto-generate Sankey diagrams for flow analysis
- [ ] Interactive DAGs for causal chains
- [ ] Export playbooks (PDF/CSV) for huddles

**Files:**
- `frontend/src/components/` - Enhanced visualization components
- `backend/app/core/conversational_ai.py` - LangGraph integration

**Impact:** Sub-5s responses; exportable insights

#### 4.2 Redis Caching
- [ ] Cache query responses (TTL: 5 min)
- [ ] Cache simulation results (TTL: 1 hour)
- [ ] Cache correlation calculations
- [ ] Add cache invalidation on data upload

**Files:**
- `backend/app/core/cache.py` - New caching module
- `backend/app/routers/chat.py` - Add cache layer

**Impact:** 10x faster repeated queries

#### 4.3 Test Coverage to 80%
- [ ] Add pytest tests for NL agent (20+ query scenarios)
- [ ] Test simulation edge cases (empty data, extreme values)
- [ ] Test correlation calculations
- [ ] Integration tests for full query → response flow

**Files:**
- `backend/tests/test_nl_agent.py` - New comprehensive tests
- `backend/tests/test_simulation.py` - Enhanced edge case tests

**Impact:** Confidence in production deployments

---

## Moonshot Vision: 15/10 Product

### Full RAG Ecosystem
- LangGraph + FAISS vectors on aggregates/NHAMCS
- Auto-retrieve benchmarks for equity queries
- Multi-hop reasoning ("Equity disparities in psych holds?" → benchmarks + sim)

### Live Integrations
- FHIR Epic hooks (real-time KPIs, PHI-scrubbed)
- xAI Grok API for forecasts
- Webhook support for alerts

### Federated Learning
- Flower lib for cross-hospital model training
- Anonymous aggregate sharing
- 20% accuracy boost on rare events

### UX Moonshot
- Voice NL (Whisper + Grok voice)
- AR overlays (HoloLens bed heatmaps)
- Multi-agent collaboration

### Monetization
- Freemium model (basic sims free, premium $5k/yr)
- Target: 15% LOS cuts, $1M/hospital ROI
- Pilot with 5 EDs

---

## Implementation Status

- [x] Phase 1.1: Error handling (partial)
- [x] Phase 1.2: Code comments (partial)
- [x] Phase 2.1: LangChain ReAct agent (complete)
- [x] Phase 2.2: Correlation tool enhancement (partial)
- [ ] Phase 3.1: DoWhy integration (pending)
- [ ] Phase 3.2: RL upgrade (pending)
- [ ] Phase 4.1: Dynamic rendering (pending)
- [ ] Phase 4.2: Redis caching (pending)
- [ ] Phase 4.3: Test coverage (pending)

---

## Next Steps

1. **Immediate (This Week):**
   - Complete error handling across all modules
   - Finish DoWhy integration in detection.py
   - Add Polars correlation calculations

2. **Short-term (Next 2 Weeks):**
   - RL environment setup
   - Redis caching implementation
   - Test coverage expansion

3. **Long-term (1-2 Months):**
   - RAG ecosystem
   - Live integrations
   - Federated learning

---

## Success Metrics

**Current (6-7/10):**
- Handles basic queries: ✅
- Detects bottlenecks: ✅
- Runs simulations: ✅
- NL feels scripted: ❌
- Causal depth shallow: ❌

**Target (10/10):**
- Handles 90% of queries organically: 🎯
- Deep causal analysis with ATE: 🎯
- RL-based recommendations: 🎯
- Sub-5s response times: 🎯
- 80% test coverage: 🎯

**Moonshot (15/10):**
- Prescient simulations (GANs + TSiT+): 🚀
- Equity engine (Fairlearn + SDOH): 🚀
- Ecosystem lock (xAI/Grok): 🚀
- $1M/hospital ROI: 🚀


## Current State: 6-7/10 MVP

**Strengths:**
- Clean monorepo structure
- ED-specific features (ESI branching, LWBS tracking)
- Docker-ready deployment
- Type-hinted, async-friendly code
- Basic simulation and detection working

**Weaknesses:**
- Rigid NL handling (fixed prompts)
- Shallow causal analysis (rules > DoWhy)
- Basic simulations (no RL/ABM)
- Limited error handling
- Sparse test coverage (~40-50%)

## Prioritized Roadmap

### Phase 1: Code Quality & Reliability (2-3 Days) - HIGH PRIORITY

**Goal:** Transform brittle MVP to reliable beta

#### 1.1 Comprehensive Error Handling
- [x] Add try/except blocks in simulation loops
- [x] Handle division by zero (empty data → safe defaults)
- [x] Validate input parameters
- [ ] Add graceful fallbacks for missing calibration data
- [ ] Implement retry logic for transient failures

**Files to Update:**
- `backend/app/core/simulation.py` - Add error handling in `run_simulation()`
- `backend/app/core/detection.py` - Handle empty KPI lists gracefully
- `backend/app/core/optimization.py` - Validate LP constraints

**Impact:** Prevents crashes on edge cases (zero patients, missing data)

#### 1.2 Enhanced Code Comments
- [x] Add module-level docstrings explaining algorithms
- [x] Document NHAMCS data sources and assumptions
- [ ] Add inline comments for complex logic
- [ ] Document parameter ranges and defaults

**Impact:** 2x faster onboarding for new developers

#### 1.3 Data-Driven Configurations
- [ ] Replace hardcoded deltas (e.g., -15% DTD/nurse) with fitted models
- [ ] Load calibration parameters from synth CSV via scikit-learn
- [ ] Make impact multipliers configurable via environment variables

**Impact:** More accurate simulations based on actual data patterns

---

### Phase 2: NL Dynamism (3-4 Days) - HIGH PRIORITY

**Goal:** True "ask anything" capability

#### 2.1 LangChain ReAct Agent Enhancement
- [x] Implement ReAct agent with dynamic tool selection
- [x] Add reflection loops ("Unclear filter? Suggest weekend?")
- [ ] Add multi-step tool chaining (corr → sim → opt)
- [ ] Implement query clarification prompts

**Files:**
- `backend/app/core/nl_agent.py` - Enhanced ReAct prompt
- `backend/app/core/conversational_ai.py` - Agent prioritization

**Impact:** Handles 90% of ED queries organically (vs. 30% currently)

#### 2.2 Advanced Correlation Tool
- [x] Add Polars-based correlation calculations
- [x] Return confidence intervals
- [ ] Support multiple metric correlations (psych → LWBS, LOS, DTD)
- [ ] Add temporal correlation (lag analysis)

**Files:**
- `backend/app/core/nl_agent.py` - `correlate_ed_metrics_tool`
- `backend/app/core/correlation_analysis.py` - Enhanced with Polars

**Impact:** Answers queries like "Psych corr to beds?" with quantified results

---

### Phase 3: Analysis Depth (4-5 Days) - HIGH PRIORITY

**Goal:** From generic recs to playbooks

#### 3.1 DoWhy Causal Integration
- [ ] Integrate DoWhy into `detection.py` for DAG-based RCA
- [ ] Add SHAP feature attribution (% variance explained)
- [ ] Calculate ATE with confidence intervals
- [ ] Identify confounders automatically

**Files:**
- `backend/app/core/detection.py` - Add `_causal_analysis_dowhy()`
- `backend/app/core/causal_inference.py` - Enhance existing module

**Impact:** 40% deeper root cause analysis (ATE -20 min, CI [-25,-15])

#### 3.2 Stable-Baselines3 PPO RL Upgrade
- [ ] Create Gym environment for sequential optimization
- [ ] Train PPO agent on historical ED data
- [ ] Generate RL-based recommendations (vs. rule-based)
- [ ] Add ROI calculations for RL actions

**Files:**
- `backend/app/core/rl_environment.py` - New Gym environment
- `backend/app/core/optimization.py` - Integrate RL recommendations

**Impact:** 2x better simulation ROI (RL learns optimal policies)

---

### Phase 4: UX & Scalability (2-3 Days) - MEDIUM PRIORITY

**Goal:** Production-ready performance

#### 4.1 Dynamic Rendering
- [ ] LangGraph node for visualization decisions
- [ ] Auto-generate Sankey diagrams for flow analysis
- [ ] Interactive DAGs for causal chains
- [ ] Export playbooks (PDF/CSV) for huddles

**Files:**
- `frontend/src/components/` - Enhanced visualization components
- `backend/app/core/conversational_ai.py` - LangGraph integration

**Impact:** Sub-5s responses; exportable insights

#### 4.2 Redis Caching
- [ ] Cache query responses (TTL: 5 min)
- [ ] Cache simulation results (TTL: 1 hour)
- [ ] Cache correlation calculations
- [ ] Add cache invalidation on data upload

**Files:**
- `backend/app/core/cache.py` - New caching module
- `backend/app/routers/chat.py` - Add cache layer

**Impact:** 10x faster repeated queries

#### 4.3 Test Coverage to 80%
- [ ] Add pytest tests for NL agent (20+ query scenarios)
- [ ] Test simulation edge cases (empty data, extreme values)
- [ ] Test correlation calculations
- [ ] Integration tests for full query → response flow

**Files:**
- `backend/tests/test_nl_agent.py` - New comprehensive tests
- `backend/tests/test_simulation.py` - Enhanced edge case tests

**Impact:** Confidence in production deployments

---

## Moonshot Vision: 15/10 Product

### Full RAG Ecosystem
- LangGraph + FAISS vectors on aggregates/NHAMCS
- Auto-retrieve benchmarks for equity queries
- Multi-hop reasoning ("Equity disparities in psych holds?" → benchmarks + sim)

### Live Integrations
- FHIR Epic hooks (real-time KPIs, PHI-scrubbed)
- xAI Grok API for forecasts
- Webhook support for alerts

### Federated Learning
- Flower lib for cross-hospital model training
- Anonymous aggregate sharing
- 20% accuracy boost on rare events

### UX Moonshot
- Voice NL (Whisper + Grok voice)
- AR overlays (HoloLens bed heatmaps)
- Multi-agent collaboration

### Monetization
- Freemium model (basic sims free, premium $5k/yr)
- Target: 15% LOS cuts, $1M/hospital ROI
- Pilot with 5 EDs

---

## Implementation Status

- [x] Phase 1.1: Error handling (partial)
- [x] Phase 1.2: Code comments (partial)
- [x] Phase 2.1: LangChain ReAct agent (complete)
- [x] Phase 2.2: Correlation tool enhancement (partial)
- [ ] Phase 3.1: DoWhy integration (pending)
- [ ] Phase 3.2: RL upgrade (pending)
- [ ] Phase 4.1: Dynamic rendering (pending)
- [ ] Phase 4.2: Redis caching (pending)
- [ ] Phase 4.3: Test coverage (pending)

---

## Next Steps

1. **Immediate (This Week):**
   - Complete error handling across all modules
   - Finish DoWhy integration in detection.py
   - Add Polars correlation calculations

2. **Short-term (Next 2 Weeks):**
   - RL environment setup
   - Redis caching implementation
   - Test coverage expansion

3. **Long-term (1-2 Months):**
   - RAG ecosystem
   - Live integrations
   - Federated learning

---

## Success Metrics

**Current (6-7/10):**
- Handles basic queries: ✅
- Detects bottlenecks: ✅
- Runs simulations: ✅
- NL feels scripted: ❌
- Causal depth shallow: ❌

**Target (10/10):**
- Handles 90% of queries organically: 🎯
- Deep causal analysis with ATE: 🎯
- RL-based recommendations: 🎯
- Sub-5s response times: 🎯
- 80% test coverage: 🎯

**Moonshot (15/10):**
- Prescient simulations (GANs + TSiT+): 🚀
- Equity engine (Fairlearn + SDOH): 🚀
- Ecosystem lock (xAI/Grok): 🚀
- $1M/hospital ROI: 🚀

