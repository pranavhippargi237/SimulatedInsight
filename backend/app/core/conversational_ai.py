"""
Conversational AI: Main orchestrator for ChatGPT-like natural language interaction.
Combines conversation management, intent understanding, and action execution.
"""
import logging
import math
from typing import Dict, Any, Optional
from app.core.conversation_manager import ConversationManager
from app.core.simulation import EDSimulation
from app.core.advisor import EDAdvisor
from app.core.detection import BottleneckDetector
from app.core.config import settings
from app.data.schemas import SimulationRequest, ScenarioChange

logger = logging.getLogger(__name__)


class ConversationalAI:
    """
    Main conversational AI that handles natural language interaction.
    
    This is the ChatGPT-like interface that:
    1. Maintains conversation context
    2. Understands natural language queries
    3. Executes actions (simulation, planning, analysis)
    4. Generates natural language responses
    5. Handles follow-up questions
    """
    
    def __init__(self):
        self.conversation_manager = ConversationManager()
        self.simulator = EDSimulation()
        self.advisor = EDAdvisor()
        self.detector = BottleneckDetector()
    
    async def chat(
        self,
        query: str,
        conversation_id: str = "default",
        system_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main chat interface - process user query and return response.
        
        Args:
            query: User's natural language query
            conversation_id: Conversation session ID
            system_context: Additional system context (current ED state, etc.)
            
        Returns:
            Complete response with natural language text and results
        """
        import asyncio
        
        try:
            # Add global timeout of 30 seconds to prevent hanging
            return await asyncio.wait_for(
                self._chat_internal(query, conversation_id, system_context),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.error("Chat processing timed out after 30 seconds")
            return {
                "status": "error",
                "response": "I'm taking too long to process your query. Please try a simpler question or check back in a moment.",
                "conversation_id": conversation_id
            }
        except Exception as e:
            logger.error(f"Chat processing failed: {e}", exc_info=True)
            return {
                "status": "error",
                "response": f"I encountered an error processing your query: {str(e)}. Could you try rephrasing?",
                "conversation_id": conversation_id
            }
    
    async def _chat_internal(
        self,
        query: str,
        conversation_id: str = "default",
        system_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Internal chat processing without timeout wrapper."""
        # Store query for use in action execution
        self._last_query = query
        
        # Step 1: Process query with conversation context
        parsed = await self.conversation_manager.process_with_context(
            query=query,
            conversation_id=conversation_id,
            system_context=system_context
        )
        
        # Step 2: Execute actions based on parsed intent
        results = await self._execute_actions(parsed, conversation_id, query)
        
        # Step 3: Generate natural language response
        # Check if this is a comparison query for response generation
        query_lower = query.lower() if query else ""
        is_comparison_query = (
            "weekday" in query_lower or "weekend" in query_lower or 
            "compare" in query_lower or "difference" in query_lower or
            "different" in query_lower or "vs" in query_lower or "versus" in query_lower
        )
        
        response_text = await self._generate_response(
            query=query,
            parsed=parsed,
            results=results,
            conversation_id=conversation_id,
            is_comparison_query=is_comparison_query
        )
        
        return {
            "status": "ok",
            "response": response_text,
            "intent": parsed.get("intent"),
            "is_follow_up": parsed.get("is_follow_up", False),
            "results": results,
            "conversation_id": conversation_id
        }
    
    async def _execute_actions(
        self,
        parsed: Dict[str, Any],
        conversation_id: str,
        query: str = ""
    ) -> Dict[str, Any]:
        """Execute actions based on parsed intent."""
        results = {
            "simulation": None,
            "action_plan": None,
            "bottlenecks": None
        }
        
        actions = parsed.get("actions", [])
        parameters = parsed.get("parameters", {})
        intent = parsed.get("intent")
        query_lower = query.lower() if query else ""

        # LLM intent augmentation (if no intent or generic)
        if not intent or intent == "generic":
            try:
                from app.core.nl_intent import parse_intent_llm
                llm_intent = await parse_intent_llm(query or "")
                if llm_intent and isinstance(llm_intent, dict):
                    intent = llm_intent.get("intent") or intent
                    parsed.update(llm_intent)
                    # propagate commonly used fields
                    if "days" in llm_intent:
                        parameters["days"] = llm_intent["days"]
                    if "window_hours" in llm_intent and llm_intent["window_hours"]:
                        parameters["window_hours"] = llm_intent["window_hours"]
                    if "top_n" in llm_intent and llm_intent["top_n"]:
                        parameters["top_n"] = llm_intent["top_n"]
                    query_lower = query.lower() if query else ""
            except Exception as e:
                logger.debug(f"LLM intent augmentation failed: {e}")
        
        # LangChain ReAct Agent: PRIORITIZE for ALL queries - this is the truly agentic path
        # The agent uses ReAct (Reason + Act) loops to:
        # 1. Reason about what the user is asking
        # 2. Choose appropriate tools dynamically (not forced categories)
        # 3. Call tools and observe results
        # 4. Reflect on outputs and potentially chain multiple tools
        # 5. Generate natural, conversational responses
        # This replaces canned replies with dynamic, emergent responses
        agent_output = None
        agent_used = False
        agent_intermediate_steps = []
        
        # Always try LangChain agent first for agentic behavior
        # It can handle ANY query by reasoning through tools dynamically
        try:
            from app.core.nl_agent import run_agent
            agent_result = await run_agent(query or "")
            if agent_result and isinstance(agent_result, dict):
                agent_output = agent_result.get("output")
                agent_intermediate_steps = agent_result.get("intermediate_steps", [])
                is_agentic = agent_result.get("agentic", False)
                
                if agent_output and len(agent_output.strip()) > 50:
                    results["agent_output"] = agent_output
                    results["agent_intermediate_steps"] = agent_intermediate_steps
                    agent_used = True
                    logger.info(f"✅ LangChain ReAct agent generated dynamic response: {len(agent_output)} chars, {len(agent_intermediate_steps)} tool calls")
                elif is_agentic:
                    # Even if output is short, if agent ran, it's agentic
                    results["agent_output"] = agent_output or "Agent processed query but output was brief."
                    agent_used = True
        except Exception as e:
            logger.debug(f"LangChain agent skipped/failed (will use fallback): {e}")

        # Comparisons (day-of-week)
        # Check for weekday/weekend comparisons
        is_weekday_weekend_query = (
            "weekday" in query_lower or "weekend" in query_lower or "weekdays" in query_lower or "weekends" in query_lower
        ) and ("compare" in query_lower or "difference" in query_lower or "different" in query_lower or "vs" in query_lower or "versus" in query_lower)
        
        if intent == "compare_days" or is_weekday_weekend_query or ("compare" in query_lower and any(d in query_lower for d in ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"])):
            try:
                from app.core.comparisons import compare_days
                # Extract days from parsed or query
                days = parsed.get("days", [])
                if not days:
                    # Handle weekday/weekend queries
                    if is_weekday_weekend_query:
                        # Compare weekdays (Mon-Fri) vs weekends (Sat-Sun)
                        days = ["Weekday", "Weekend"]
                    else:
                        # simple parse for two days in query
                        day_names = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
                        found = [d.title() for d in day_names if d in query_lower]
                        days = found[:2] if found else ["Saturday","Monday"]
                results["comparison"] = await compare_days(days=days, window_hours=parsed.get("window_hours", 720))
            except Exception as e:
                logger.warning(f"Day comparison failed: {e}")
                results["comparison"] = {"error": str(e)}

        # Spike detection (volume surges)
        if intent == "spikes" or "spike" in query_lower or "surge" in query_lower:
            try:
                from app.core.comparisons import detect_spikes
                results["spikes"] = await detect_spikes(window_hours=parsed.get("window_hours", 168), top_n=parsed.get("top_n", 5))
            except Exception as e:
                logger.warning(f"Spike detection failed: {e}")
                results["spikes"] = {"error": str(e)}

        # Run simulation
        if "simulate" in actions or parsed.get("intent") == "simulation":
            if parameters.get("resource_type"):
                try:
                    scenario = ScenarioChange(
                        action=parameters.get("action", "add"),
                        resource_type=parameters.get("resource_type"),
                        quantity=parameters.get("quantity", 1),
                        time_start=parameters.get("time_start"),
                        time_end=parameters.get("time_end"),
                        day=parameters.get("day")
                    )
                    
                    sim_request = SimulationRequest(
                        scenario=[scenario],  # List of scenario changes
                        simulation_hours=24,
                        iterations=100
                    )
                    
                    sim_result = await self.simulator.run_simulation(sim_request)
                    # Clean simulation results to ensure JSON compliance
                    sim_dict = sim_result.dict()
                    # Recursively clean any inf/NaN values
                    def clean_dict(d):
                        if isinstance(d, dict):
                            return {k: clean_dict(v) for k, v in d.items()}
                        elif isinstance(d, list):
                            return [clean_dict(item) for item in d]
                        elif isinstance(d, float):
                            return d if math.isfinite(d) else 0.0
                        else:
                            return d
                    results["simulation"] = clean_dict(sim_dict)
                except Exception as e:
                    logger.error(f"Simulation failed: {e}")
                    results["simulation"] = {"error": str(e)}
        
        # Generate action plan
        if "generate_plan" in actions or parsed.get("intent") == "plan":
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
        
        # Always detect bottlenecks for comprehensive analysis
        # This provides context for deep analysis even if not explicitly requested
        try:
            bottlenecks = await self.detector.detect_bottlenecks(
                window_hours=48,
                top_n=5
            )
            # Clean bottleneck results to ensure JSON compliance
            def clean_bottleneck(b):
                import math
                b_dict = b.dict() if hasattr(b, 'dict') else b
                cleaned = {}
                for k, v in b_dict.items():
                    if isinstance(v, float):
                        cleaned[k] = v if math.isfinite(v) else 0.0
                    else:
                        cleaned[k] = v
                return cleaned
            results["bottlenecks"] = [clean_bottleneck(b) for b in bottlenecks]
        except Exception as e:
            logger.error(f"Bottleneck detection failed: {e}")
            results["bottlenecks"] = []
        
        # For ALL queries, get current metrics and perform deep analysis
        try:
            from app.data.storage import get_kpis, get_events
            from datetime import datetime, timedelta
            import math
            
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=48)
            kpis = await get_kpis(start_time, end_time)
            events = await get_events(start_time, end_time)
            
            current_metrics_dict = {}
            if kpis and len(kpis) > 0:
                latest_kpi = kpis[-1] if isinstance(kpis[-1], dict) else kpis[-1].dict() if hasattr(kpis[-1], 'dict') else {}
                # Ensure all values are finite and JSON-compliant
                def safe_float(val, default=0.0):
                    if val is None:
                        return default
                    try:
                        fval = float(val)
                        return fval if math.isfinite(fval) else default
                    except (ValueError, TypeError):
                        return default
                
                current_metrics_dict = {
                    "dtd": safe_float(latest_kpi.get("dtd", 0)),
                    "los": safe_float(latest_kpi.get("los", 0)),
                    "lwbs": safe_float(latest_kpi.get("lwbs", 0)),
                    "bed_utilization": safe_float(latest_kpi.get("bed_utilization", 0))
                }
                results["current_metrics"] = current_metrics_dict
        
            # For ALL queries, perform deep analysis using InsightEngine
            # Always enabled - uses caching and timeouts to prevent hanging
            ENABLE_DEEP_ANALYSIS = True
            
            # Always run deep analysis if we have events OR if bottlenecks were detected
            should_run_analysis = ENABLE_DEEP_ANALYSIS and (
                (events and len(events) > 0) or 
                (results.get("bottlenecks") and len(results.get("bottlenecks", [])) > 0)
            )
            
            if should_run_analysis:
                try:
                    from app.core.insight_engine import InsightEngine
                    from datetime import datetime, timedelta
                    import asyncio
                        
                    # Events already fetched above, reuse them
                    # Detect metric type from query - use actual metrics if available
                    metric_name = "dtd"  # Default to DTD
                    if "lwbs" in query_lower or "left without" in query_lower:
                        metric_name = "lwbs"
                    elif "dtd" in query_lower or "door to doctor" in query_lower or "door-to-doctor" in query_lower:
                        metric_name = "dtd"
                    elif "los" in query_lower or "length of stay" in query_lower:
                        metric_name = "los"
                    elif "throughput" in query_lower or "patient flow" in query_lower:
                        metric_name = "throughput"
                    elif "bed" in query_lower or "bed utilization" in query_lower or "bed occupancy" in query_lower:
                        metric_name = "bed"
                    
                    # If no specific metric in query, use the metric with actual data
                    if metric_name == "dtd" and current_metrics_dict:
                        if current_metrics_dict.get("dtd", 0) <= 0 and current_metrics_dict.get("los", 0) > 0:
                            metric_name = "los"
                        elif current_metrics_dict.get("dtd", 0) <= 0 and current_metrics_dict.get("los", 0) <= 0 and current_metrics_dict.get("lwbs", 0) > 0:
                            metric_name = "lwbs"
                    
                    insight_engine = InsightEngine()
                    # Increase timeout to 15 seconds for more comprehensive analysis
                    # Use events if available, otherwise use empty list
                    analysis_events = events if events and len(events) > 0 else []
                    analysis_kpis = kpis if kpis and len(kpis) > 0 else []
                    
                    deep_analysis = await asyncio.wait_for(
                        insight_engine.analyze(metric_name, analysis_events, analysis_kpis, window_hours=48),
                        timeout=15.0  # Increased timeout for better analysis
                    )
                
                    # Use actual metrics if available, otherwise use analysis results
                    actual_current_value = 0.0
                    actual_benchmark = 30.0
                    if current_metrics_dict:
                        if metric_name == "dtd" and current_metrics_dict.get("dtd", 0) > 0:
                            actual_current_value = current_metrics_dict.get("dtd", 0)
                            actual_benchmark = 30.0
                        elif metric_name == "los" and current_metrics_dict.get("los", 0) > 0:
                            actual_current_value = current_metrics_dict.get("los", 0)
                            actual_benchmark = 180.0
                        elif metric_name == "lwbs" and current_metrics_dict.get("lwbs", 0) > 0:
                            actual_current_value = current_metrics_dict.get("lwbs", 0) * 100  # Convert to percentage
                            actual_benchmark = 5.0
                        else:
                            # Use analysis results if no matching metric
                            actual_current_value = float(deep_analysis.current_value) if deep_analysis.current_value is not None else 0.0
                            actual_benchmark = float(deep_analysis.benchmark_value) if deep_analysis.benchmark_value is not None else 30.0
                    else:
                        # Use analysis results if no metrics available
                        actual_current_value = float(deep_analysis.current_value) if deep_analysis.current_value is not None else 0.0
                        actual_benchmark = float(deep_analysis.benchmark_value) if deep_analysis.benchmark_value is not None else 30.0
                
                    # Convert to dict for JSON serialization with enhanced formatting
                    results["deep_analysis"] = {
                        "metric_name": metric_name,  # Use the actual metric name, not "generic"
                        "current_value": actual_current_value,
                        "benchmark_value": actual_benchmark,
                        "excess": actual_current_value - actual_benchmark,
                        "insights": [
                            {
                                "insight_type": i.insight_type,
                                "type": i.insight_type,  # Alias for compatibility
                                "title": i.title,
                                "description": i.description,
                                "evidence": i.evidence if isinstance(i.evidence, dict) else {},
                                "impact_score": float(i.impact_score) if i.impact_score is not None else 0.0,
                                "confidence": float(i.confidence) if i.confidence is not None else 0.0,
                                "actionable": bool(i.actionable) if i.actionable is not None else False,
                                "recommendation": i.recommendation or "",
                                "unmet_need": i.unmet_need or ""
                            }
                            for i in deep_analysis.insights
                        ],
                        "patterns": deep_analysis.patterns if isinstance(deep_analysis.patterns, dict) else {},
                        "root_causes": deep_analysis.root_causes if isinstance(deep_analysis.root_causes, list) else [],
                        "unmet_needs": deep_analysis.unmet_needs if isinstance(deep_analysis.unmet_needs, list) else [],
                        "predictive_signals": deep_analysis.predictive_signals if isinstance(deep_analysis.predictive_signals, dict) else {},
                        "economic_impact": deep_analysis.economic_impact if isinstance(deep_analysis.economic_impact, dict) else None
                    }
                except asyncio.TimeoutError:
                    logger.warning("Deep analysis timed out - providing fallback analysis")
                    # Always provide fallback analysis
                    fallback_events = events if events else []
                    fallback_kpis = kpis if kpis else []
                    
                    # Generate insights from bottlenecks if available
                    if results.get("bottlenecks") and len(results.get("bottlenecks", [])) > 0:
                        top_bottleneck = results["bottlenecks"][0]
                        results["deep_analysis"] = {
                            "metric_name": f"{top_bottleneck.get('bottleneck_name', 'Bottleneck')} Analysis",
                            "current_value": float(top_bottleneck.get('current_wait_time_minutes', 0)),
                            "benchmark_value": 30.0,
                            "excess": float(top_bottleneck.get('current_wait_time_minutes', 0)) - 30.0,
                            "insights": [{
                                "insight_type": "bottleneck_analysis",
                                "type": "bottleneck_analysis",
                                "title": f"{top_bottleneck.get('bottleneck_name', 'Bottleneck')} - {top_bottleneck.get('severity', 'medium').upper()} Severity",
                                "description": f"Critical bottleneck at {top_bottleneck.get('stage', 'unknown')} stage: {top_bottleneck.get('current_wait_time_minutes', 0):.0f} min wait time, {top_bottleneck.get('impact_score', 0)*100:.0f}% impact on operations",
                                "evidence": {
                                    "bottleneck_name": top_bottleneck.get('bottleneck_name'),
                                    "stage": top_bottleneck.get('stage'),
                                    "wait_time": top_bottleneck.get('current_wait_time_minutes'),
                                    "impact": top_bottleneck.get('impact_score'),
                                    "severity": top_bottleneck.get('severity'),
                                    "causes": top_bottleneck.get('causes', [])
                                },
                                "impact_score": float(top_bottleneck.get('impact_score', 0)),
                                "confidence": 0.9,
                                "actionable": True,
                                "recommendation": top_bottleneck.get('recommendations', ['Review resource allocation'])[0] if top_bottleneck.get('recommendations') else "Review resource allocation at this stage",
                                "unmet_need": "Real-time bottleneck monitoring and intervention"
                            }],
                            "patterns": top_bottleneck.get('metadata', {}).get('temporal_analysis', {}),
                            "root_causes": top_bottleneck.get('causes', []) if top_bottleneck.get('causes') else [f"Bottleneck at {top_bottleneck.get('stage', 'unknown')} stage"],
                            "unmet_needs": ["Real-time bottleneck monitoring"],
                            "predictive_signals": {},
                            "economic_impact": None
                        }
                    elif fallback_events and len(fallback_events) > 0:
                        results["deep_analysis"] = {
                                "metric_name": "ED Operations",
                                "current_value": 0.0,
                                "benchmark_value": 30.0,
                                "excess": 0.0,
                                "insights": [{
                                    "insight_type": "analysis_timeout",
                                    "type": "analysis_timeout",
                                    "title": "Analysis Timeout",
                                    "description": "Deep analysis timed out. Basic analysis available - try asking about specific metrics like 'What are my bottlenecks?'",
                                    "evidence": {"timeout": True},
                                    "impact_score": 0.0,
                                    "confidence": 0.5,
                                    "actionable": False,
                                    "recommendation": "Try a more specific query or check data availability",
                                    "unmet_need": ""
                                }],
                                "patterns": {},
                                "root_causes": ["Analysis timed out - insufficient data or system overload"],
                                "unmet_needs": [],
                                "predictive_signals": {},
                            "economic_impact": None
                        }
                except Exception as e:
                    logger.warning(f"Deep analysis failed: {e} - providing fallback analysis", exc_info=True)
                    # Always provide fallback analysis
                    fallback_events = events if events else []
                    fallback_kpis = kpis if kpis else []
                    
                    # Generate insights from bottlenecks if available
                    if results.get("bottlenecks") and len(results.get("bottlenecks", [])) > 0:
                        top_bottleneck = results["bottlenecks"][0]
                        results["deep_analysis"] = {
                            "metric_name": f"{top_bottleneck.get('bottleneck_name', 'Bottleneck')} Analysis",
                            "current_value": float(top_bottleneck.get('current_wait_time_minutes', 0)),
                            "benchmark_value": 30.0,
                            "excess": float(top_bottleneck.get('current_wait_time_minutes', 0)) - 30.0,
                            "insights": [{
                                "insight_type": "bottleneck_analysis",
                                "type": "bottleneck_analysis",
                                "title": f"{top_bottleneck.get('bottleneck_name', 'Bottleneck')} - {top_bottleneck.get('severity', 'medium').upper()} Severity",
                                "description": f"Critical bottleneck at {top_bottleneck.get('stage', 'unknown')} stage: {top_bottleneck.get('current_wait_time_minutes', 0):.0f} min wait time, {top_bottleneck.get('impact_score', 0)*100:.0f}% impact",
                                "evidence": {
                                    "bottleneck_name": top_bottleneck.get('bottleneck_name'),
                                    "stage": top_bottleneck.get('stage'),
                                    "wait_time": top_bottleneck.get('current_wait_time_minutes'),
                                    "impact": top_bottleneck.get('impact_score'),
                                    "severity": top_bottleneck.get('severity'),
                                    "causes": top_bottleneck.get('causes', [])
                                },
                                "impact_score": float(top_bottleneck.get('impact_score', 0)),
                                "confidence": 0.9,
                                "actionable": True,
                                "recommendation": top_bottleneck.get('recommendations', ['Review resource allocation'])[0] if top_bottleneck.get('recommendations') else "Review resource allocation at this stage",
                                "unmet_need": "Real-time bottleneck monitoring"
                            }],
                            "patterns": top_bottleneck.get('metadata', {}).get('temporal_analysis', {}),
                            "root_causes": top_bottleneck.get('causes', []) if top_bottleneck.get('causes') else [f"Bottleneck at {top_bottleneck.get('stage', 'unknown')} stage"],
                            "unmet_needs": ["Real-time bottleneck monitoring"],
                            "predictive_signals": {},
                            "economic_impact": None
                        }
                    elif fallback_events and len(fallback_events) > 0:
                            results["deep_analysis"] = {
                                "metric_name": "ED Operations",
                                "current_value": 0.0,
                                "benchmark_value": 30.0,
                                "excess": 0.0,
                                "insights": [{
                                    "insight_type": "analysis_error",
                                    "type": "analysis_error",
                                    "title": "Analysis Error",
                                    "description": f"Analysis encountered an error: {str(e)[:100]}. Basic insights may still be available.",
                                    "evidence": {"error": str(e)[:200]},
                                    "impact_score": 0.0,
                                    "confidence": 0.3,
                                    "actionable": False,
                                    "recommendation": "Check data quality and try again",
                                    "unmet_need": ""
                                }],
                                "patterns": {},
                                "root_causes": [f"Analysis error: {str(e)[:100]}"],
                                "unmet_needs": [],
                                "predictive_signals": {},
                                "economic_impact": None
                            }
        except Exception as e:
            logger.debug(f"Could not fetch current metrics: {e}")
            # Even if metrics fail, try to generate basic analysis from bottlenecks
            if results.get("bottlenecks") and len(results.get("bottlenecks", [])) > 0:
                try:
                    from app.core.insight_engine import InsightEngine
                    import asyncio
                    
                    insight_engine = InsightEngine()
                    # Generate analysis from bottlenecks even without events
                    basic_analysis = await asyncio.wait_for(
                        insight_engine.analyze("generic", [], [], window_hours=48),
                        timeout=5.0
                    )
                    
                    results["deep_analysis"] = {
                            "metric_name": "Bottleneck Analysis",
                            "current_value": float(basic_analysis.current_value) if basic_analysis.current_value else 0.0,
                            "benchmark_value": float(basic_analysis.benchmark_value) if basic_analysis.benchmark_value else 30.0,
                            "excess": 0.0,
                            "insights": [
                                {
                                    "insight_type": i.insight_type,
                                    "type": i.insight_type,
                                    "title": i.title,
                                    "description": i.description,
                                    "evidence": i.evidence if isinstance(i.evidence, dict) else {},
                                    "impact_score": float(i.impact_score) if i.impact_score is not None else 0.0,
                                    "confidence": float(i.confidence) if i.confidence is not None else 0.0,
                                    "actionable": bool(i.actionable) if i.actionable is not None else False,
                                    "recommendation": i.recommendation or "",
                                    "unmet_need": i.unmet_need or ""
                                }
                                for i in basic_analysis.insights
                            ],
                            "patterns": basic_analysis.patterns if isinstance(basic_analysis.patterns, dict) else {},
                            "root_causes": basic_analysis.root_causes if isinstance(basic_analysis.root_causes, list) else [],
                            "unmet_needs": basic_analysis.unmet_needs if isinstance(basic_analysis.unmet_needs, list) else [],
                            "predictive_signals": basic_analysis.predictive_signals if isinstance(basic_analysis.predictive_signals, dict) else {},
                        "economic_impact": basic_analysis.economic_impact if isinstance(basic_analysis.economic_impact, dict) else None
                    }
                except Exception as analysis_error:
                    logger.debug(f"Fallback analysis also failed: {analysis_error}")
        
        # Final safety check: Ensure deep_analysis exists if we have bottlenecks
        # This guarantees deep analysis is always available when bottlenecks are detected
        if results.get("bottlenecks") and isinstance(results.get("bottlenecks"), list) and len(results["bottlenecks"]) > 0:
            if not results.get("deep_analysis") or not results["deep_analysis"].get("insights") or len(results["deep_analysis"]["insights"]) == 0:
                # Generate deep analysis from top bottleneck
                top_bottleneck = results["bottlenecks"][0]
                if isinstance(top_bottleneck, dict):
                    results["deep_analysis"] = {
                        "metric_name": f"{top_bottleneck.get('bottleneck_name', 'Bottleneck')} Deep Analysis",
                        "current_value": float(top_bottleneck.get('current_wait_time_minutes', 0)),
                        "benchmark_value": 30.0,
                        "excess": float(top_bottleneck.get('current_wait_time_minutes', 0)) - 30.0,
                        "insights": [{
                            "insight_type": "bottleneck_identified",
                            "type": "bottleneck_identified",
                            "title": f"{top_bottleneck.get('bottleneck_name', 'Bottleneck')} - {top_bottleneck.get('severity', 'medium').upper()} Severity",
                            "description": f"Critical bottleneck detected at {top_bottleneck.get('stage', 'unknown')} stage: {top_bottleneck.get('current_wait_time_minutes', 0):.0f} min wait time, {top_bottleneck.get('impact_score', 0)*100:.0f}% impact on overall ED operations. This is a {top_bottleneck.get('severity', 'medium')} severity issue requiring immediate attention.",
                            "evidence": {
                                "bottleneck_name": top_bottleneck.get('bottleneck_name'),
                                "stage": top_bottleneck.get('stage'),
                                "wait_time_minutes": float(top_bottleneck.get('current_wait_time_minutes', 0)),
                                "impact_score": float(top_bottleneck.get('impact_score', 0)),
                                "severity": top_bottleneck.get('severity'),
                                "causes": top_bottleneck.get('causes', []),
                                "recommendations": top_bottleneck.get('recommendations', [])
                            },
                            "impact_score": float(top_bottleneck.get('impact_score', 0)),
                            "confidence": 0.95,
                            "actionable": True,
                            "recommendation": top_bottleneck.get('recommendations', ['Review resource allocation and staffing at this stage'])[0] if top_bottleneck.get('recommendations') else "Review resource allocation and staffing at this stage",
                            "unmet_need": "Real-time bottleneck monitoring and automated intervention system"
                        }],
                        "patterns": top_bottleneck.get('metadata', {}).get('temporal_analysis', {}) if top_bottleneck.get('metadata') else {},
                        "root_causes": top_bottleneck.get('causes', []) if top_bottleneck.get('causes') else [
                            f"Bottleneck at {top_bottleneck.get('stage', 'unknown')} stage with {top_bottleneck.get('current_wait_time_minutes', 0):.0f} min wait time",
                            f"Impact score of {top_bottleneck.get('impact_score', 0)*100:.0f}% indicates significant operational disruption"
                        ],
                        "unmet_needs": ["Real-time bottleneck monitoring", "Automated intervention protocols"],
                        "predictive_signals": {},
                        "economic_impact": None
                    }
        
        return results
    
    async def _generate_response(
        self,
        query: str,
        parsed: Dict[str, Any],
        results: Dict[str, Any],
        conversation_id: str,
        is_comparison_query: bool = False
    ) -> str:
        """Generate natural language response using OpenAI."""
        conversation_manager = self.conversation_manager
        
        # Build a detailed response based on results
        if not conversation_manager.client:
            # Fallback: Generate detailed response from results
            response_parts = []
            
            if results.get("bottlenecks"):
                bottlenecks = results["bottlenecks"]
                if isinstance(bottlenecks, list) and len(bottlenecks) > 0:
                    response_parts.append(f"I've identified {len(bottlenecks)} bottleneck(s) in your ED:\n\n")
                    for i, b in enumerate(bottlenecks[:3], 1):
                        name = b.get("bottleneck_name", f"Bottleneck {i}")
                        wait = b.get("current_wait_time_minutes", 0)
                        severity = b.get("severity", "medium")
                        response_parts.append(f"{i}. **{name}**: {wait:.0f}-minute wait times ({severity} severity)")
                    if len(bottlenecks) > 3:
                        response_parts.append(f"\n...and {len(bottlenecks) - 3} more bottleneck(s)")
                else:
                    response_parts.append("I've analyzed your ED and found no significant bottlenecks. Your operations are running smoothly!")
            
            if results.get("current_metrics"):
                metrics = results["current_metrics"]
                response_parts.append(f"\n\n**Current Metrics:**\n")
                response_parts.append(f"- Door-to-Doctor: {metrics.get('dtd', 0):.1f} minutes\n")
                response_parts.append(f"- Length of Stay: {metrics.get('los', 0):.1f} minutes\n")
                response_parts.append(f"- LWBS Rate: {metrics.get('lwbs', 0)*100:.1f}%")
            
            if results.get("action_plan"):
                plan = results["action_plan"]
                recs = plan.get("recommendations", [])
                if recs:
                    response_parts.append(f"\n\nI've generated {len(recs)} prioritized recommendations to address these issues.")
            
            if not response_parts:
                return parsed.get("response", "I've processed your query. Here are the results.")
            
            return "".join(response_parts)
        
        # Build context for response generation
        context_parts = []
        
        if results.get("simulation"):
            sim = results["simulation"]
            context_parts.append(f"Simulation Results: DTD={sim.get('predicted_metrics', {}).get('dtd', 'N/A')} min, LOS={sim.get('predicted_metrics', {}).get('los', 'N/A')} min, LWBS={sim.get('predicted_metrics', {}).get('lwbs', 'N/A')}")
        
        if results.get("action_plan"):
            plan = results["action_plan"]
            context_parts.append(f"Action Plan: {len(plan.get('recommendations', []))} recommendations generated")
        
        if results.get("bottlenecks"):
            bottlenecks = results["bottlenecks"]
            if isinstance(bottlenecks, list) and len(bottlenecks) > 0:
                context_parts.append(f"Bottlenecks detected ({len(bottlenecks)} total):")
                for i, b in enumerate(bottlenecks[:5], 1):
                    name = b.get("bottleneck_name", f"Bottleneck {i}")
                    wait = b.get("current_wait_time_minutes", 0)
                    severity = b.get("severity", "medium")
                    impact = b.get("impact_score", 0)
                    stage = b.get("stage", "unknown")
                    causes = b.get("causes", [])
                    rca = b.get("metadata", {}).get("root_cause_analysis", {})
                    
                    context_parts.append(f"  {i}. {name} ({stage}): {wait:.0f} min wait, {severity} severity, {impact*100:.0f}% impact")
                    
                    # Add causal analysis (replaces rule-based RCA)
                    causal_analysis = b.get("metadata", {}).get("causal_analysis", {})
                    causal_narrative = b.get("metadata", {}).get("causal_narrative", "")
                    
                    if causal_analysis:
                        ate = causal_analysis.get('ate_estimates', {})
                        counterfactuals = causal_analysis.get('counterfactuals', [])
                        attributions = causal_analysis.get('feature_attributions', {}).get('attributions', {})
                        probabilities = causal_analysis.get('probabilistic_insights', {}).get('probabilities', {})
                        equity = causal_analysis.get('equity_analysis', {})
                        variance_explained = causal_analysis.get('variance_explained', {})
                        confidence_scores = causal_analysis.get('confidence_scores', {})
                        interactions = causal_analysis.get('interactions', [])
                        confounders = causal_analysis.get('confounders', [])
                        
                        # ATE with confidence intervals
                        if ate:
                            treatment = ate.get('treatment', '')
                            value = ate.get('value', 0)
                            ci_lower = ate.get('ci_lower')
                            ci_upper = ate.get('ci_upper')
                            if treatment:
                                ci_str = f" (95% CI [{ci_lower:.1f}, {ci_upper:.1f}])" if ci_lower is not None and ci_upper is not None else ""
                                context_parts.append(f"     CAUSAL ATE: {treatment} → {value:.1f} min wait{ci_str}")
                        
                        # Variance explained (R² decomposition)
                        if variance_explained:
                            top_vars = sorted(variance_explained.items(), key=lambda x: x[1].get('percentage', 0), reverse=True)[:2]
                            for var, data in top_vars:
                                pct = data.get('percentage', 0)
                                context_parts.append(f"     VARIANCE EXPLAINED: {var} explains {pct:.0f}% of wait time variance")
                        
                        # SHAP attributions
                        if attributions:
                            top_attr = sorted(attributions.items(), key=lambda x: x[1], reverse=True)[:2]
                            context_parts.append(f"     SHAP ATTRIBUTIONS: {', '.join([f'{k}: {v:.0f}%' for k, v in top_attr])}")
                        
                        # Probabilistic insights
                        if probabilities:
                            for prob_name, prob_val in list(probabilities.items())[:1]:
                                context_parts.append(f"     PROBABILITY: {prob_name} = {prob_val:.0%}")
                        
                        # Interactions
                        if interactions:
                            for interaction in interactions[:1]:
                                vars_str = ' × '.join(interaction.get('variables', []))
                                strength = interaction.get('strength', 0)
                                context_parts.append(f"     INTERACTION: {vars_str} amplifies impact by {strength*100:.0f}%")
                        
                        # Confounders
                        if confounders:
                            context_parts.append(f"     CONFOUNDERS: {', '.join(confounders[:3])}")
                        
                        # Counterfactuals with ROI
                        if counterfactuals:
                            cf = counterfactuals[0]
                            improvement = cf.get('improvement_pct', 0)
                            roi = cf.get('roi', {})
                            roi_pct = roi.get('roi_percentage', 0) if roi else 0
                            payback = roi.get('payback_days', 0) if roi else 0
                            context_parts.append(f"     COUNTERFACTUAL: {cf.get('scenario', 'Intervention')} → {improvement:.0f}% improvement, ROI: {roi_pct:.0f}%, Payback: {payback:.1f} days")
                        
                        # Equity analysis
                        if equity and equity.get('disparity_pct'):
                            disparity = equity.get('disparity_pct', 0)
                            concern = equity.get('concern', 'unknown')
                            context_parts.append(f"     EQUITY: {disparity:.0f}% disparity ({concern} concern) - High-acuity: {equity.get('high_acuity_wait', 0):.0f} min vs Low-acuity: {equity.get('low_acuity_wait', 0):.0f} min")
                        
                        # Confidence scores
                        if confidence_scores:
                            overall_conf = confidence_scores.get('overall_confidence', 0.5)
                            context_parts.append(f"     ANALYSIS CONFIDENCE: {overall_conf:.0%}")
                    
                    # Add causal narrative if available
                    if causal_narrative:
                        context_parts.append(f"     CAUSAL NARRATIVE: {causal_narrative[:300]}...")
                    
                    # Fallback to old RCA format if no causal analysis
                    rca = b.get("metadata", {}).get("root_cause_analysis", {})
                    if not causal_analysis and rca:
                        immediate = rca.get("immediate_causes", [])
                        underlying = rca.get("underlying_causes", [])
                        systemic = rca.get("systemic_causes", [])
                        causal_chain = rca.get("causal_chain", [])
                        contributing = rca.get("contributing_factors", {})
                        
                        if causal_chain:
                            context_parts.append(f"     Causal Chain: {' → '.join(causal_chain)}")
                        
                        if immediate:
                            context_parts.append(f"     Immediate Causes: {', '.join([c.get('description', '')[:80] for c in immediate[:2]])}")
                        
                        if underlying:
                            context_parts.append(f"     Underlying Causes: {', '.join([c.get('description', '')[:80] for c in underlying[:2]])}")
                        
                        if systemic:
                            context_parts.append(f"     Systemic Causes: {', '.join([c.get('description', '')[:80] for c in systemic[:1]])}")
                        
                        if contributing:
                            top_factors = sorted(contributing.items(), key=lambda x: x[1], reverse=True)[:3]
                            context_parts.append(f"     Contributing Factors: {', '.join([f'{f[0]} ({f[1]:.0f}%)' for f in top_factors])}")
                    
                    # Add operational mechanics (HOW operations cause issues)
                    op_mech = b.get("metadata", {}).get("operational_mechanics", {})
                    if op_mech:
                        examples = op_mech.get("data_driven_examples", [])
                        if examples:
                            example = examples[0]
                            if example.get("type") == "patient_journey":
                                data = example.get("data", {})
                                context_parts.append(f"     OPERATIONAL EXAMPLE: Patient {data.get('patient_id', 'N/A')} - "
                                                    f"Arrived {data.get('arrival_time', 'N/A')}, Total LOS: {data.get('total_los_minutes', 0):.0f} min, "
                                                    f"Wait at {bottleneck.stage}: {data.get('bottleneck_analysis', {}).get('wait_time_minutes', 0):.0f} min "
                                                    f"({data.get('bottleneck_analysis', {}).get('contribution_to_los', 0):.0f}% of total LOS)")
                        
                        throughput = op_mech.get("throughput_analysis", {})
                        if throughput:
                            context_parts.append(f"     THROUGHPUT: {throughput.get('average_processed_per_hour', 0):.1f} patients/hour processed vs "
                                              f"{throughput.get('average_arrivals_per_hour', 0):.1f} arrivals/hour "
                                              f"(efficiency: {throughput.get('throughput_efficiency', 0):.0%}, backlog: {throughput.get('average_backlog_per_hour', 0):.1f}/hour)")
                        
                        utilization = op_mech.get("utilization_analysis", {})
                        if utilization:
                            context_parts.append(f"     UTILIZATION: Peak {utilization.get('peak_utilization', 0):.0%} at {utilization.get('peak_window', 'N/A')}, "
                                              f"{utilization.get('peak_events_per_resource', 0):.1f} events/resource")
                    
                    elif causes:
                        context_parts.append(f"     Causes: {', '.join(causes[:3])}")
            else:
                context_parts.append("No bottlenecks detected - ED operating efficiently")
        
        # Add comparison results if available (weekday vs weekend, etc.)
        if results.get("comparison"):
            comparison = results["comparison"]
            if not comparison.get("error"):
                aggregates = comparison.get("aggregates", {})
                diffs = comparison.get("diffs", {})
                baseline = comparison.get("baseline", "")
                
                context_parts.append(f"DAY COMPARISON ({baseline} baseline):")
                for day, metrics in aggregates.items():
                    arrivals = metrics.get("arrivals", 0)
                    lwbs_rate = metrics.get("lwbs_rate", 0)
                    mean_waits = metrics.get("mean_waits", {})
                    
                    context_parts.append(f"  {day}:")
                    context_parts.append(f"    Arrivals: {arrivals} patients")
                    context_parts.append(f"    LWBS Rate: {lwbs_rate*100:.1f}%")
                    if mean_waits:
                        wait_strs = [f"{stage}: {wait:.1f} min" for stage, wait in mean_waits.items()]
                        context_parts.append(f"    Mean Waits: {', '.join(wait_strs)}")
                    
                    # Add differences if available
                    if day in diffs:
                        diff = diffs[day]
                        context_parts.append(f"    vs {baseline}:")
                        if diff.get("arrivals_delta"):
                            context_parts.append(f"      Arrivals: {diff['arrivals_delta']:+.0f} ({diff.get('arrivals_pct', 0):+.1f}%)")
                        if diff.get("lwbs_delta") is not None:
                            context_parts.append(f"      LWBS: {diff['lwbs_delta']*100:+.1f}% ({diff.get('lwbs_pct', 0):+.1f}% change)")
                        if diff.get("mean_waits_delta"):
                            wait_diffs = []
                            for stage, wait_diff in diff["mean_waits_delta"].items():
                                wait_diffs.append(f"{stage}: {wait_diff.get('delta', 0):+.1f} min ({wait_diff.get('pct', 0):+.1f}%)")
                            if wait_diffs:
                                context_parts.append(f"      Wait Times: {', '.join(wait_diffs)}")
        
        if results.get("current_metrics"):
            metrics = results["current_metrics"]
            import math
            dtd = metrics.get('dtd', 0)
            los = metrics.get('los', 0)
            lwbs = metrics.get('lwbs', 0)
            # Ensure values are finite before formatting
            dtd = dtd if math.isfinite(dtd) else 0
            los = los if math.isfinite(los) else 0
            lwbs = lwbs if math.isfinite(lwbs) else 0
            context_parts.append(f"Current Metrics: DTD={dtd:.1f} min, LOS={los:.1f} min, LWBS={lwbs*100:.1f}%")
        
        # Ensure metric_name_for_prompt is defined
        if 'metric_name_for_prompt' not in locals():
            metric_name_for_prompt = "metric"
        
        context = "\n".join(context_parts) if context_parts else "No results yet"
        
        # Get conversation history for context
        history = conversation_manager.get_conversation_history(conversation_id)
        
        # Build messages
        messages = [
            {
                "role": "system",
                "content": """You are a helpful ED operations assistant. Generate natural, conversational responses that:
1. Acknowledge what the user asked
2. Explain what you did
3. Highlight key findings
4. Provide actionable insights
5. Invite follow-up questions

Be conversational, helpful, and clear. Use natural language, not technical jargon."""
            }
        ]
        
        # Add recent history (last 5 messages)
        for msg in history[-5:]:
            if msg["role"] in ["user", "assistant"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current query and results
        # Use metric_name_for_prompt directly (already defined above)
        metric_display = metric_name_for_prompt
        prompt = f"""User asked: "{query}"

I analyzed their ED and found:
{context}

Generate a natural, conversational response that:
1. Acknowledges their question in a friendly, professional way
2. Explains what you analyzed (be specific - mention bottlenecks by name, wait times, metrics)

**FOR ALL METRIC QUERIES (if deep analysis is provided):**
3. **CRITICALLY IMPORTANT - Provide UNIQUE, DEEP INSIGHTS with QUANTIFIED DEPTH:**
   - Start with current value vs benchmark (e.g., "Your {metric_display} is X vs Y benchmark - Z excess")
   - **QUANTIFY EVERYTHING**: Use ATE with confidence intervals (e.g., "Staff shortage → +18.2 min wait, 95% CI [-22.1, -14.3]")
   - **VARIANCE EXPLAINED**: Show R² decomposition (e.g., "Staffing explains 45% of variance, boarding_lag explains 30%")
   - **PROBABILISTIC INSIGHTS**: Include probabilities (e.g., "P(wait spike | staff_short) = 72%")
   - **INTERACTIONS**: Show multivariate effects (e.g., "Staff shortage × surge amplifies impact by 65%")
   - **CONFOUNDERS**: Explain what's controlled for (e.g., "After controlling for boarding_lag, staffing effect = -15 min")
   - **ROI FOR EVERY REC**: Include ROI %, payback period, daily cost/savings (e.g., "Add 1 tech → ROI: 180%, Payback: 0.6 days, Daily net: $320")
   - **EQUITY STRATIFICATION**: Show disparities with numbers (e.g., "High-acuity wait 25% longer - 45 min vs 36 min")
   - **CONFIDENCE SCORES**: State analysis confidence (e.g., "Overall confidence: 78%")
   - **AVOID GENERIC STATEMENTS**: Never say "insufficient staffing" without quantifying (ATE, variance, ROI)
   - Highlight UNMET NEEDS revealed by the analysis (what systems/processes are missing?)
   - Explain WHY these specific patterns exist with data-driven evidence

4. **For all queries, provide DEEP ROOT CAUSE ANALYSIS:**
   - Explain the IMMEDIATE causes (what's happening right now)
   - Explain the UNDERLYING causes (process and resource issues)
   - Explain the SYSTEMIC causes (structural problems)
   - Show the CAUSAL CHAIN (how causes connect: systemic → underlying → immediate → bottleneck)
   - Highlight CONTRIBUTING FACTORS with percentages

5. **CRITICALLY IMPORTANT: Explain HOW operations cause these issues with concrete examples:**
   - Show specific patient journeys with timestamps and wait times
   - Explain operational sequences step-by-step
   - Show throughput analysis (arrivals vs processed per hour, backlog accumulation)
   - Show utilization patterns (events per resource, peak utilization windows)
   - Use the DATA-DRIVEN EXAMPLES provided to illustrate exactly how operations lead to bottlenecks
   - Explain the mechanics: "When X happens, it causes Y, which leads to Z bottleneck"

6. Highlights the KEY findings with actual numbers (wait times, DTD, LOS, LWBS rates)
7. Provides actionable insights about what these bottlenecks mean and WHY they're happening
8. Naturally invites follow-up questions

IMPORTANT: 
- **QUANTIFY EVERYTHING**: Use ATE with CIs, variance explained (R²), probabilities, ROI - NO generic statements
- **NEVER say "insufficient staffing" without numbers**: Use "Staff shortage → ATE: -18.2 min (CI [-22.1, -14.3]), explains 45% variance, ROI: 180%"
- **SHOW INTERACTIONS**: "Staff shortage × surge amplifies impact by 65%" (not just "both are problems")
- **INCLUDE CONFOUNDERS**: "After controlling for boarding_lag, staffing effect = -15 min" (shows causal depth)
- **ROI FOR ALL RECS**: Every recommendation must have ROI %, payback period, daily cost/savings
- **EQUITY STRATIFICATION**: Show disparities with actual numbers (high-acuity vs low-acuity wait times)
- **CONFIDENCE SCORES**: State analysis confidence (e.g., "78% confidence based on ATE CI width and Bayesian fit")
- **VARIANCE DECOMPOSITION**: Show what % each factor explains (e.g., "Staffing: 45%, Boarding: 30%, Surge: 25%")
- **AVOID HALLUCINATIONS**: Don't make up numbers - use only what's in the analysis
- **PROBABILISTIC REASONING**: Use probabilities (e.g., "68% chance LWBS spikes if bed holds persist")
- **MULTIVARIATE DEPTH**: Show how factors interact, not just list them separately
- Be conversational but data-driven and analytical
- Don't be generic - use SPECIFIC NUMBERS with confidence intervals
- Focus on QUANTIFIED ROOT CAUSES, OPERATIONAL MECHANICS, and UNMET NEEDS
- **Connect each insight to an unmet need with quantified impact**
- **Provide actionable recommendations with ROI calculations**"""
        
        messages.append({"role": "user", "content": prompt})
        
        # PRIORITIZE LangChain ReAct agent output - it's truly agentic and dynamic
        # The agent uses ReAct loops to reason through queries, call tools dynamically, and provide non-canned responses
        # This is the "ask anything" path - no pre-programmed categories, just reasoning + tools
        if results.get("agent_output") and len(results["agent_output"].strip()) > 30:
            agent_response = results["agent_output"]
            intermediate_steps = results.get("agent_intermediate_steps", [])
            
            logger.info(f"✅ Using LangChain ReAct agent response (agentic path, {len(intermediate_steps)} tool calls)")
            
            # The agent response is already natural and conversational
            # Optionally enhance with structured data if agent didn't include it
            # But prefer agent's natural response over forced structure
            enhancements = []
            
            # Only add structured data if agent response is very brief (might have missed context)
            if len(agent_response.strip()) < 200:
                if results.get("comparison") and not results["comparison"].get("error"):
                    enhancements.append(f"\n\n**Detailed Comparison:**\n{self._format_comparison(results['comparison'])}")
                if results.get("bottlenecks") and len(results.get("bottlenecks", [])) > 0:
                    bottlenecks = results["bottlenecks"][:3]
                    bottleneck_summary = "\n".join([
                        f"- {b.get('bottleneck_name', 'Bottleneck')}: {b.get('current_wait_time_minutes', 0):.0f} min wait"
                        for b in bottlenecks
                    ])
                    enhancements.append(f"\n\n**Bottlenecks:**\n{bottleneck_summary}")
            
            if enhancements:
                return f"{agent_response}{''.join(enhancements)}"
            
            # Return agent's natural response - it's already conversational and data-driven
            return agent_response
        
        try:
            response = conversation_manager.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=1500  # Increased for deeper analysis and comparisons
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            # Fallback: if we have comparison data, format it
            if results.get("comparison") and not results["comparison"].get("error"):
                return f"I've analyzed the differences between weekdays and weekends. {self._format_comparison(results['comparison'])}"
            return parsed.get("response", "I've processed your query. Here are the results.")
    
    def _format_comparison(self, comparison: Dict[str, Any]) -> str:
        """Format comparison results into readable text."""
        aggregates = comparison.get("aggregates", {})
        diffs = comparison.get("diffs", {})
        baseline = comparison.get("baseline", "")
        
        parts = []
        for day, metrics in aggregates.items():
            arrivals = metrics.get("arrivals", 0)
            lwbs_rate = metrics.get("lwbs_rate", 0)
            mean_waits = metrics.get("mean_waits", {})
            
            parts.append(f"\n**{day}:**")
            parts.append(f"- Arrivals: {arrivals} patients")
            parts.append(f"- LWBS Rate: {lwbs_rate*100:.1f}%")
            if mean_waits:
                wait_strs = [f"{stage.replace('_', ' ').title()}: {wait:.1f} min" for stage, wait in mean_waits.items()]
                parts.append(f"- Average Wait Times: {', '.join(wait_strs)}")
            
            # Add differences
            if day in diffs:
                diff = diffs[day]
                parts.append(f"\n**Differences vs {baseline}:**")
                if diff.get("arrivals_delta"):
                    parts.append(f"- Arrivals: {diff['arrivals_delta']:+.0f} ({diff.get('arrivals_pct', 0):+.1f}% change)")
                if diff.get("lwbs_delta") is not None:
                    parts.append(f"- LWBS Rate: {diff['lwbs_delta']*100:+.1f} percentage points ({diff.get('lwbs_pct', 0):+.1f}% change)")
                if diff.get("mean_waits_delta"):
                    for stage, wait_diff in diff["mean_waits_delta"].items():
                        stage_name = stage.replace('_', ' ').title()
                        parts.append(f"- {stage_name} Wait: {wait_diff.get('delta', 0):+.1f} min ({wait_diff.get('pct', 0):+.1f}% change)")
        
        return "\n".join(parts)

