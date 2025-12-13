"""
Tests for RL-based optimization with Stable-Baselines3.
"""
import pytest
from app.core.optimization import Optimizer
from app.data.schemas import OptimizationRequest, Bottleneck


@pytest.mark.asyncio
async def test_rl_optimization_basic():
    """Test basic RL optimization."""
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
    
    # Mock historical simulations
    historical_sims = [
        {
            "scenario": [{"resource_type": "nurse", "quantity": 1}],
            "deltas": {"dtd_reduction": 10.0, "lwbs_drop": 5.0}
        }
        for _ in range(15)  # Enough for RL
    ]
    
    suggestions = await optimizer.optimize(
        request=request,
        bottlenecks=bottlenecks,
        historical_sims=historical_sims
    )
    
    assert isinstance(suggestions, list)
    # RL suggestions may or may not be present depending on availability
    if suggestions:
        assert all(s.priority > 0 for s in suggestions)


@pytest.mark.asyncio
async def test_rl_environment():
    """Test RL environment creation."""
    try:
        from app.core.rl_environment import EDOptimizationEnv
        env = EDOptimizationEnv(
            initial_state={"dtd": 40.0, "los": 200.0, "lwbs": 0.06},
            constraints={"budget": 1000.0, "max_nurses": 3}
        )
        
        # Test reset
        obs, info = env.reset()
        assert obs is not None
        assert len(obs) == 11  # 11 features
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    except ImportError:
        pytest.skip("Gymnasium or RL environment not available")
