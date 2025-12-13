"""
Intelligent router endpoint - single entry point for all queries.
Uses OpenAI to understand intent and orchestrates actions.
"""
from fastapi import APIRouter, HTTPException
from app.core.intelligent_router import IntelligentRouter
from app.data.schemas import NLPQuery

router = APIRouter()
intelligent_router = IntelligentRouter()


@router.post("/intelligent/query")
async def process_intelligent_query(query: NLPQuery):
    """
    Intelligent query processing endpoint.
    
    This is the main entry point that:
    1. Understands user intent using OpenAI
    2. Decides what actions to take (simulate, plan, both, etc.)
    3. Executes actions
    4. Generates a natural language response
    
    Examples:
    - "What if we add 2 nurses?" → Runs simulation
    - "What should I do?" → Generates action plan
    - "What if we add nurses and what should I do?" → Runs simulation AND generates plan
    - "Compare adding 2 vs 3 nurses" → Runs multiple simulations and compares
    
    Args:
        query: Natural language query
        
    Returns:
        Comprehensive response with results and natural language explanation
    """
    try:
        result = await intelligent_router.process_query(query.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


