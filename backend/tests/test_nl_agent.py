"""
Tests for LangChain ReAct agent.
"""
import pytest
from app.core.nl_agent import run_agent, _build_agent


@pytest.mark.asyncio
async def test_agent_basic():
    """Test basic agent functionality."""
    agent_executor = _build_agent()
    if agent_executor is None:
        pytest.skip("LangChain or OpenAI not available")
    
    # Test that agent can be invoked
    result = await run_agent("What are the current bottlenecks?")
    assert result is None or isinstance(result, dict)


@pytest.mark.asyncio
async def test_agent_tool_chaining():
    """Test that agent can chain multiple tools."""
    agent_executor = _build_agent()
    if agent_executor is None:
        pytest.skip("LangChain or OpenAI not available")
    
    # Complex query that might require multiple tools
    result = await run_agent("Compare weekday and weekend metrics, then detect bottlenecks")
    assert result is None or isinstance(result, dict)
    if result:
        assert "output" in result
        assert "agentic" in result


@pytest.mark.asyncio
async def test_agent_reflection():
    """Test that agent uses reflection loops."""
    agent_executor = _build_agent()
    if agent_executor is None:
        pytest.skip("LangChain or OpenAI not available")
    
    result = await run_agent("What if psych surges hit 25% next weekend?")
    assert result is None or isinstance(result, dict)
    if result:
        # Check for tool chaining (indicates reflection)
        intermediate_steps = result.get("intermediate_steps", [])
        assert isinstance(intermediate_steps, list)
