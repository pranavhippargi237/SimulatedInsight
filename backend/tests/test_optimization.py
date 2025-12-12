"""
Tests for optimization engine.
"""
import pytest
from app.core.optimization import Optimizer
from app.data.schemas import OptimizationRequest, Bottleneck


@pytest.mark.asyncio
async def test_optimization_basic():
    """Test basic optimization."""
    optimizer = Optimizer()
    
    bottlenecks = [
        Bottleneck(
            bottleneck_name="Triage Queue",
            stage="triage",
            impact_score=0.8,
            current_wait_time_minutes=45.0,
            causes=["Insufficient staffing"],
            severity="high",
            recommendations=[]
        )
    ]
    
    request = OptimizationRequest(
        bottlenecks=[],
        constraints={"budget": 1000, "staff_max": 10},
        objective="minimize_dtd"
    )
    
    suggestions = await optimizer.optimize(
        request=request,
        bottlenecks=bottlenecks,
        historical_sims=None
    )
    
    assert isinstance(suggestions, list)
    if suggestions:
        assert suggestions[0].priority > 0
        assert suggestions[0].expected_impact is not None

