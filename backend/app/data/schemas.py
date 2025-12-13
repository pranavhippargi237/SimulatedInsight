"""
Pydantic schemas for data validation and serialization.
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    """ED event types."""
    ARRIVAL = "arrival"
    TRIAGE = "triage"
    BED_ASSIGN = "bed_assign"
    DOCTOR_VISIT = "doctor_visit"
    IMAGING = "imaging"
    LABS = "labs"  # Laboratory tests
    DISCHARGE = "discharge"
    LWBS = "lwbs"  # Left Without Being Seen


class EDEvent(BaseModel):
    """Schema for ED events."""
    timestamp: datetime
    event_type: EventType
    patient_id: str = Field(..., description="Anonymized patient identifier")
    stage: Optional[str] = None
    resource_type: Optional[str] = None  # e.g., "nurse", "doctor", "bed"
    resource_id: Optional[str] = None
    duration_minutes: Optional[float] = None
    esi: Optional[int] = Field(None, ge=1, le=5, description="Emergency Severity Index (1-5)")
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T10:30:00Z",
                "event_type": "arrival",
                "patient_id": "anon_patient_123",
                "stage": "triage",
                "resource_type": "nurse",
                "duration_minutes": 5.0
            }
        }


class StaffingEvent(BaseModel):
    """Schema for staffing changes."""
    timestamp: datetime
    resource_type: str = Field(..., description="e.g., 'nurse', 'doctor', 'tech'")
    count: int = Field(..., gt=0)
    department: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T14:00:00Z",
                "resource_type": "nurse",
                "count": 3,
                "department": "triage"
            }
        }


class KPI(BaseModel):
    """Key Performance Indicators."""
    timestamp: datetime
    door_to_doctor_minutes: float = Field(..., alias="dtd")
    length_of_stay_minutes: float = Field(..., alias="los")
    lwbs_rate: float = Field(..., ge=0, le=1, alias="lwbs")
    bed_utilization: float = Field(..., ge=0, le=1)
    queue_length: int = Field(..., ge=0)

    class Config:
        populate_by_name = True


class Bottleneck(BaseModel):
    """Bottleneck detection result with comprehensive insights."""
    bottleneck_name: str
    stage: str
    impact_score: float = Field(..., ge=0, le=1, description="Impact on overall flow (0-1)")
    current_wait_time_minutes: float
    causes: List[str] = Field(default_factory=list)
    severity: str = Field(..., description="low, medium, high, critical")
    recommendations: List[str] = Field(default_factory=list)
    # OpenAI-generated analysis fields
    where: Optional[str] = Field(None, description="AI-generated description of where the bottleneck is occurring")
    why: Optional[str] = Field(None, description="AI-generated root cause analysis explaining why the bottleneck happened")
    first_order_effects: Optional[List[str]] = Field(None, description="AI-generated list of direct, immediate effects")
    second_order_effects: Optional[List[str]] = Field(None, description="AI-generated list of downstream, cascading effects")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional analysis data including root cause analysis")
    
    # Enhanced quantitative metrics
    p95_wait_time_minutes: Optional[float] = Field(None, description="95th percentile wait time")
    queue_length: Optional[int] = Field(None, description="Current queue length")
    throughput_drag_percent: Optional[float] = Field(None, description="Percentage reduction in throughput")
    lwbs_impact_percent: Optional[float] = Field(None, description="Additional LWBS percentage caused by this bottleneck")
    peak_hours: Optional[List[int]] = Field(None, description="Peak hours when bottleneck is most severe")
    peak_time_range: Optional[str] = Field(None, description="Peak time range (e.g., '19:00-22:00')")
    
    # Causal breakdown
    causal_breakdown: Optional[Dict[str, Any]] = Field(None, description="Causal analysis with SHAP, ATE, CI, correlations")
    equity_analysis: Optional[Dict[str, Any]] = Field(None, description="Equity impact analysis by patient groups")
    
    # Simulated actions
    simulated_actions: Optional[List[Dict[str, Any]]] = Field(None, description="Prioritized actions with Delta, ROI, Confidence, Equity Lift")
    
    # Forecasting
    forecast: Optional[Dict[str, Any]] = Field(None, description="Forecasted volume and risk")
    
    # Operational examples
    operational_example: Optional[str] = Field(None, description="Real patient journey scenario")
    
    # Patient flow cascade
    flow_cascade: Optional[Dict[str, Any]] = Field(None, description="Patient flow cascade showing how bottlenecks propagate through the system")

    class Config:
        json_schema_extra = {
            "example": {
                "bottleneck_name": "Imaging Queue",
                "stage": "imaging",
                "impact_score": 0.25,
                "current_wait_time_minutes": 45.0,
                "causes": ["staff: -2", "equipment: unavailable"],
                "severity": "high",
                "recommendations": ["Add 1 imaging tech", "Cross-train staff"]
            }
        }


class ScenarioChange(BaseModel):
    """Schema for simulation scenario changes."""
    action: str = Field(..., description="add, remove, shift, modify")
    resource_type: str = Field(..., description="nurse, doctor, bed, tech")
    quantity: int = Field(..., description="Number of resources")
    time_start: Optional[str] = Field(None, description="HH:MM format")
    time_end: Optional[str] = Field(None, description="HH:MM format")
    day: Optional[str] = None
    department: Optional[str] = None

    @validator("action")
    def validate_action(cls, v):
        allowed = ["add", "remove", "shift", "modify"]
        if v not in allowed:
            raise ValueError(f"Action must be one of {allowed}")
        return v


class SimulationRequest(BaseModel):
    """Request schema for simulation."""
    scenario: List[ScenarioChange] = Field(..., description="List of scenario changes (add/remove/shift resources)")
    simulation_hours: int = Field(24, ge=1, le=48, alias="window_hours")
    iterations: Optional[int] = None
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "scenario": [
                    {
                        "action": "add",
                        "resource_type": "nurse",
                        "quantity": 2,
                        "time_start": "14:00",
                        "time_end": "18:00",
                        "day": "Saturday"
                    }
                ],
                "simulation_hours": 24,
                "iterations": 100
            }
        }


class SimulationResult(BaseModel):
    """Simulation output."""
    scenario_id: str
    baseline_metrics: Dict[str, float]
    predicted_metrics: Dict[str, float]
    deltas: Dict[str, float] = Field(..., description="Percentage changes")
    confidence: float = Field(..., ge=0, le=1)
    execution_time_seconds: float
    traces: Optional[List[Dict[str, Any]]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "scenario_id": "sim_001",
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
                    "dtd_reduction": -20.0,
                    "los_reduction": -8.3,
                    "lwbs_drop": -37.5
                },
                "confidence": 0.85,
                "execution_time_seconds": 8.5
            }
        }


class OptimizationRequest(BaseModel):
    """Request schema for optimization."""
    constraints: Dict[str, Any] = Field(default_factory=dict, description="budget, staff_max, max_doctors, max_nurses, max_techs")
    target_metrics: List[str] = Field(default_factory=lambda: ["dtd", "lwbs", "los"], description="Target metrics to optimize")

    class Config:
        json_schema_extra = {
            "example": {
                "bottlenecks": ["Imaging Queue", "Triage Queue"],
                "constraints": {
                    "staff_max": 10,
                    "budget": 1000
                },
                "objective": "minimize_dtd"
            }
        }


class OptimizationSuggestion(BaseModel):
    """Optimization suggestion result."""
    priority: int
    action: str
    resource_type: str
    quantity: int
    expected_impact: Dict[str, float]
    cost: Optional[float] = None
    confidence: float = Field(..., ge=0, le=1)


class NLPQuery(BaseModel):
    """Natural language query from frontend."""
    query: str = Field(..., min_length=1, max_length=500)
    context: Optional[Dict[str, Any]] = None


class ParsedScenario(BaseModel):
    """Parsed scenario from NLP."""
    scenario: ScenarioChange
    confidence: float = Field(..., ge=0, le=1)
    original_query: str
    suggestions: Optional[List[str]] = None


class OptimizationSuggestion(BaseModel):
    """Optimization suggestion result."""
    priority: int
    action: str
    resource_type: str
    quantity: int
    expected_impact: Dict[str, float]
    cost: Optional[float] = None
    confidence: float = Field(..., ge=0, le=1)


class NLPQuery(BaseModel):
    """Natural language query from frontend."""
    query: str = Field(..., min_length=1, max_length=500)
    context: Optional[Dict[str, Any]] = None


class ParsedScenario(BaseModel):
    """Parsed scenario from NLP."""
    scenario: ScenarioChange
    confidence: float = Field(..., ge=0, le=1)
    original_query: str
    suggestions: Optional[List[str]] = None

