"""
Intelligent Query Router: Uses OpenAI to understand user intent and orchestrate actions.
This is the "brain" that decides whether to simulate, generate plans, or do both.
"""
import logging
import json
from typing import Dict, Any, Optional, List
import openai
from app.core.config import settings
from app.core.simulation import EDSimulation
from app.core.advisor import EDAdvisor
from app.core.nlp import NLParser
from app.data.schemas import NLPQuery, SimulationRequest, ParsedScenario

logger = logging.getLogger(__name__)


class IntelligentRouter:
    """
    Intelligent router that understands user intent and orchestrates actions.
    
    Uses OpenAI to:
    1. Understand what the user wants (simulation, plan, both, or something else)
    2. Parse the query intelligently
    3. Decide which actions to take
    4. Execute actions in the right order
    5. Combine results into a coherent response
    """
    
    def __init__(self):
        if settings.OPENAI_API_KEY:
            self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        else:
            self.client = None
            logger.warning("OpenAI API key not set - intelligent routing disabled")
        
        self.simulator = EDSimulation()
        self.advisor = EDAdvisor()
        self.nlp_parser = NLParser()
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query intelligently.
        
        This is the main entry point that:
        1. Understands user intent using OpenAI
        2. Parses the query to extract parameters
        3. Decides which actions to take (simulate, plan, both, etc.)
        4. Executes actions
        5. Combines results into a coherent response
        
        Args:
            query: Natural language query from user
            
        Returns:
            Comprehensive response with simulation results, plans, and explanations
        """
        if not self.client:
            # Fallback to simple routing
            return await self._fallback_routing(query)
        
        try:
            # Step 1: Understand intent using OpenAI
            intent = await self._understand_intent(query)
            
            logger.info(f"Intent detected: {intent['intent_type']} (confidence: {intent['confidence']})")
            
            # Step 2: Parse query to extract parameters
            parsed = await self._parse_query_intelligently(query, intent)
            
            # Step 3: Execute actions based on intent
            results = await self._execute_intent(intent, parsed, query)
            
            # Step 4: Generate natural language response
            response = await self._generate_response(intent, results, query)
            
            return {
                "status": "ok",
                "intent": intent,
                "parsed": parsed.dict() if isinstance(parsed, ParsedScenario) else parsed,
                "results": results,
                "response": response,
                "actions_taken": intent.get("actions", [])
            }
        
        except Exception as e:
            logger.error(f"Intelligent routing failed: {e}", exc_info=True)
            return await self._fallback_routing(query)
    
    async def _understand_intent(self, query: str) -> Dict[str, Any]:
        """
        Use OpenAI to understand what the user wants.
        
        Returns intent classification with actions to take.
        """
        prompt = f"""
You are an intelligent assistant for an Emergency Department operations system. 
Analyze the user's query and determine their intent.

User Query: "{query}"

Classify the intent into one of these categories:
1. "simulation" - User wants to simulate a specific scenario (e.g., "What if we add 2 nurses?")
2. "plan" - User wants recommendations/action plan (e.g., "What should I do?", "Give me a plan")
3. "both" - User wants both simulation AND plan (e.g., "What if we add nurses and what should I do?")
4. "analysis" - User wants to understand current state (e.g., "What are my bottlenecks?")
5. "comparison" - User wants to compare scenarios (e.g., "Compare adding 2 vs 3 nurses")
6. "explanation" - User wants explanation of results (e.g., "Why did DTD increase?")

Output JSON:
{{
    "intent_type": "simulation|plan|both|analysis|comparison|explanation",
    "confidence": 0.0-1.0,
    "actions": ["simulate", "generate_plan", "detect_bottlenecks", etc.],
    "reasoning": "Why you classified it this way",
    "requires_simulation": true/false,
    "requires_plan": true/false,
    "requires_analysis": true/false
}}

Analyze the query now:
"""
        
        response = self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert at understanding user intent for ED operations systems."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    
    async def _parse_query_intelligently(self, query: str, intent: Dict[str, Any]) -> ParsedScenario:
        """
        Parse query using NLP parser (which uses OpenAI if available).
        """
        nlp_query = NLPQuery(query=query)
        parsed = await self.nlp_parser.parse_query(nlp_query)
        return parsed
    
    async def _execute_intent(
        self,
        intent: Dict[str, Any],
        parsed: ParsedScenario,
        original_query: str
    ) -> Dict[str, Any]:
        """
        Execute actions based on intent.
        
        This orchestrates:
        - Running simulations
        - Generating action plans
        - Detecting bottlenecks
        - Combining results
        """
        results = {
            "simulation": None,
            "action_plan": None,
            "bottlenecks": None,
            "analysis": None
        }
        
        actions = intent.get("actions", [])
        
        # Run simulation if needed
        if intent.get("requires_simulation") or "simulate" in actions:
            if parsed.confidence >= 0.7:
                try:
                    sim_request = SimulationRequest(
                        scenario=[parsed.scenario],  # List of scenario changes
                        simulation_hours=24,
                        iterations=100
                    )
                    sim_result = await self.simulator.run_simulation(sim_request)
                    results["simulation"] = sim_result.dict()
                except Exception as e:
                    logger.error(f"Simulation failed: {e}")
                    results["simulation"] = {"error": str(e)}
        
        # Generate action plan if needed
        if intent.get("requires_plan") or "generate_plan" in actions:
            try:
                plan = await self.advisor.generate_action_plan(
                    window_hours=48,
                    top_n=5,
                    include_roi=True,
                    equity_mode=True
                )
                results["action_plan"] = plan
            except Exception as e:
                logger.error(f"Action plan generation failed: {e}")
                results["action_plan"] = {"error": str(e)}
        
        # Detect bottlenecks if needed
        if intent.get("requires_analysis") or "detect_bottlenecks" in actions:
            try:
                from app.core.detection import BottleneckDetector
                detector = BottleneckDetector()
                bottlenecks = await detector.detect_bottlenecks(window_hours=48, top_n=5)
                results["bottlenecks"] = [b.dict() for b in bottlenecks]
            except Exception as e:
                logger.error(f"Bottleneck detection failed: {e}")
                results["bottlenecks"] = {"error": str(e)}
        
        return results
    
    async def _generate_response(
        self,
        intent: Dict[str, Any],
        results: Dict[str, Any],
        original_query: str
    ) -> str:
        """
        Use OpenAI to generate a natural language response that explains:
        - What was done
        - What the results mean
        - What actions to take next
        """
        if not self.client:
            # Fallback to simple response
            return self._simple_response(intent, results)
        
        # Build context for response generation
        context = {
            "intent": intent,
            "simulation": results.get("simulation"),
            "action_plan": results.get("action_plan"),
            "bottlenecks": results.get("bottlenecks")
        }
        
        prompt = f"""
You are an intelligent assistant explaining ED operations results to a hospital director.

Original Query: "{original_query}"

Intent: {intent.get('intent_type')} - {intent.get('reasoning')}

Results Available:
- Simulation: {bool(context['simulation'])}
- Action Plan: {bool(context['action_plan'])}
- Bottlenecks: {bool(context['bottlenecks'])}

Generate a clear, professional response that:
1. Acknowledges what the user asked
2. Explains what was done
3. Highlights key findings
4. Provides actionable next steps
5. Uses natural, conversational language

Keep it concise but comprehensive. Focus on actionable insights.

Response:
"""
        
        try:
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert ED operations consultant explaining results to hospital directors."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._simple_response(intent, results)
    
    def _simple_response(self, intent: Dict[str, Any], results: Dict[str, Any]) -> str:
        """Fallback simple response."""
        intent_type = intent.get("intent_type", "unknown")
        
        if intent_type == "simulation" and results.get("simulation"):
            return "I've run the simulation. Here are the results:"
        elif intent_type == "plan" and results.get("action_plan"):
            return "I've generated an action plan based on your current ED state."
        elif intent_type == "both":
            return "I've run the simulation and generated an action plan."
        else:
            return "I've processed your query. Here are the results:"
    
    async def _fallback_routing(self, query: str) -> Dict[str, Any]:
        """Fallback routing when OpenAI is not available."""
        query_lower = query.lower()
        
        # Simple pattern matching
        if any(phrase in query_lower for phrase in [
            "what should i do", "what do you recommend", "give me a plan"
        ]):
            plan = await self.advisor.generate_action_plan()
            return {
                "status": "ok",
                "intent": {"intent_type": "plan", "confidence": 0.8},
                "results": {"action_plan": plan},
                "response": "I've generated an action plan based on your current ED state."
            }
        else:
            # Try to parse and simulate
            nlp_query = NLPQuery(query=query)
            parsed = await self.nlp_parser.parse_query(nlp_query)
            
            if parsed.confidence >= 0.7:
                sim_request = SimulationRequest(
                    scenario=[parsed.scenario],  # List of scenario changes
                    simulation_hours=24,
                    iterations=100
                )
                sim_result = await self.simulator.run_simulation(sim_request)
                return {
                    "status": "ok",
                    "intent": {"intent_type": "simulation", "confidence": 0.8},
                    "parsed": parsed.dict(),
                    "results": {"simulation": sim_result.dict()},
                    "response": "I've run the simulation. Here are the results:"
                }
            else:
                return {
                    "status": "error",
                    "message": "Could not understand query. Please try rephrasing."
                }

