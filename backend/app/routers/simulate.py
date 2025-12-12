"""
Simulation endpoints.
"""
from fastapi import APIRouter, HTTPException
from app.data.schemas import SimulationRequest, SimulationResult, NLPQuery
from app.core.simulation import EDSimulation
from app.core.nlp import NLParser

router = APIRouter()
simulator = EDSimulation()
nlp_parser = NLParser()


@router.post("/simulate")
async def run_simulation(request: SimulationRequest) -> SimulationResult:
    """
    Run a simulation scenario.
    
    Accepts structured simulation request with scenario changes.
    """
    try:
        result = await simulator.run_simulation(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@router.post("/simulate/nlp")
async def run_simulation_nlp(query: NLPQuery) -> dict:
    """
    Run simulation from natural language query.
    
    First parses the query, then runs simulation.
    """
    try:
        # Parse query
        parsed = await nlp_parser.parse_query(query)
        
        if parsed.confidence < 0.7:
            return {
                "status": "warning",
                "message": "Low confidence parsing. Please refine your query.",
                "parsed_scenario": parsed.dict(),
                "suggestions": parsed.suggestions
            }
        
        # Create simulation request
        sim_request = SimulationRequest(
            scenario=[parsed.scenario],  # List of scenario changes
            simulation_hours=24,
            iterations=100
        )
        
        # Run simulation
        result = await simulator.run_simulation(sim_request)
        
        return {
            "status": "ok",
            "original_query": query.query,
            "parsed_scenario": parsed.dict(),
            "simulation_result": result.dict()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

