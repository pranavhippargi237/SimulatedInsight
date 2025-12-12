"""
Tests for NLP parsing.
"""
import pytest
from app.core.nlp import NLParser
from app.data.schemas import NLPQuery


@pytest.mark.asyncio
async def test_nlp_rule_based():
    """Test rule-based NLP parsing."""
    parser = NLParser()
    
    query = NLPQuery(query="Add 2 nurses from 2-6 PM on Saturday")
    result = await parser.parse_query(query)
    
    assert result.scenario.action == "add"
    assert result.scenario.resource_type == "nurse"
    assert result.scenario.quantity == 2
    assert result.confidence > 0


@pytest.mark.asyncio
async def test_nlp_extract_time():
    """Test time extraction from query."""
    parser = NLParser()
    
    query = NLPQuery(query="Add doctor from 14:00 to 18:00")
    result = await parser.parse_query(query)
    
    assert result.scenario.time_start is not None
    assert result.scenario.time_end is not None

