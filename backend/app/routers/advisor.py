"""
Advisor endpoints for conversational "what should I do" queries.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.core.advisor import EDAdvisor
from app.core.nlp import NLParser

router = APIRouter()
advisor = EDAdvisor()
nlp_parser = NLParser()


@router.post("/advisor/plan")
async def get_action_plan(
    window_hours: int = Query(48, description="Time window for analysis"),
    top_n: int = Query(5, description="Number of top recommendations"),
    include_roi: bool = Query(True, description="Include ROI calculations"),
    equity_mode: bool = Query(True, description="Enable equity-aware recommendations")
):
    """
    Generate a detailed action plan based on current ED state.
    
    This endpoint analyzes current bottlenecks, generates optimization suggestions,
    calculates ROI, and provides a comprehensive implementation plan.
    
    Args:
        window_hours: Time window for analysis (default 48h)
        top_n: Number of top recommendations (default 5)
        include_roi: Include ROI calculations (default True)
        equity_mode: Enable equity-aware recommendations (default True)
        
    Returns:
        Detailed action plan with priorities, ROI, and implementation steps
    """
    try:
        plan = await advisor.generate_action_plan(
            window_hours=window_hours,
            top_n=top_n,
            include_roi=include_roi,
            equity_mode=equity_mode
        )
        return plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate action plan: {str(e)}")


@router.post("/advisor/ask")
async def ask_advisor(query: str = Query(..., description="Natural language query")):
    """
    Conversational advisor endpoint - answers "what should I do" type queries.
    
    Uses NLP to parse the query and generates appropriate recommendations.
    
    Examples:
    - "What should I do?"
    - "What are my options?"
    - "Give me a plan"
    - "What's the best way to reduce wait times?"
    
    Args:
        query: Natural language query (as query parameter)
        
    Returns:
        Action plan or specific recommendations based on query
    """
    try:
        query_lower = query.lower()
        
        # Check if it's a "what should I do" type query
        if any(phrase in query_lower for phrase in [
            "what should i do", "what do you recommend", "what are my options",
            "give me a plan", "what's the best", "how should i", "what can i do",
            "help me", "advice", "recommendations", "suggestions"
        ]):
            # Generate full action plan
            plan = await advisor.generate_action_plan(
                window_hours=48,
                top_n=5,
                include_roi=True,
                equity_mode=True
            )
            
            return {
                "status": "ok",
                "query": query,
                "response_type": "action_plan",
                "plan": plan
            }
        
        # Otherwise, try to parse as a specific scenario query
        from app.data.schemas import NLPQuery
        nlp_query = NLPQuery(query=query)
        parsed = await nlp_parser.parse_query(nlp_query)
        
        if parsed.confidence > 0.7:
            # It's a specific scenario, provide targeted advice
            return {
                "status": "ok",
                "query": query,
                "response_type": "targeted_advice",
                "parsed_scenario": parsed.dict(),
                "message": f"Based on your query, I can help you simulate: {parsed.scenario.action} {parsed.scenario.quantity} {parsed.scenario.resource_type}(s). Use the simulation endpoint to see the impact."
            }
        else:
            # Generate general action plan
            plan = await advisor.generate_action_plan(
                window_hours=48,
                top_n=5,
                include_roi=True,
                equity_mode=True
            )
            
            return {
                "status": "ok",
                "query": query,
                "response_type": "action_plan",
                "plan": plan,
                "message": "I've generated a comprehensive action plan based on your current ED state."
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process advisor query: {str(e)}")

