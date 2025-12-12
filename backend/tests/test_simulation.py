"""
Tests for simulation engine.
"""
import pytest
from app.core.simulation import EDSimulation
from app.data.schemas import SimulationRequest, ScenarioChange


@pytest.mark.asyncio
async def test_simulation_basic():
    """Test basic simulation."""
    simulator = EDSimulation(seed=42)
    
    scenario = ScenarioChange(
        action="add",
        resource_type="nurse",
        quantity=1
    )
    
    request = SimulationRequest(
        scenario=scenario,
        simulation_hours=1,
        iterations=10
    )
    
    result = await simulator.run_simulation(request)
    
    assert result.scenario_id is not None
    assert result.baseline_metrics is not None
    assert result.predicted_metrics is not None
    assert result.confidence >= 0
    assert result.confidence <= 1
    assert result.execution_time_seconds > 0


@pytest.mark.asyncio
async def test_simulation_multiple_iterations():
    """Test simulation with multiple iterations."""
    simulator = EDSimulation(seed=42)
    
    scenario = ScenarioChange(
        action="add",
        resource_type="doctor",
        quantity=1
    )
    
    request = SimulationRequest(
        scenario=scenario,
        simulation_hours=2,
        iterations=5
    )
    
    result = await simulator.run_simulation(request)
    
    assert result.confidence > 0
    assert "dtd" in result.predicted_metrics
    assert "los" in result.predicted_metrics

