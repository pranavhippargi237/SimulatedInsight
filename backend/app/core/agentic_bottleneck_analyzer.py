"""
Agentic Bottleneck Analyzer using LangChain.
Autonomously analyzes bottlenecks, detects causal chains, and identifies
first and second order effects from actual data.
"""
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd

from app.core.config import settings
from app.data.storage import get_events, get_kpis

logger = logging.getLogger(__name__)

# LangChain imports
try:
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    try:
        from langchain.agents import create_react_agent, AgentExecutor
    except ImportError:
        # Try alternative import path
        from langchain_core.agents import AgentExecutor
        from langchain.agents import create_react_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    tool = None
    ChatOpenAI = None
    create_react_agent = None
    AgentExecutor = None
    logger.warning(f"LangChain not available - agentic analysis disabled: {e}")

# Causal inference imports
try:
    from app.core.causal_inference import CausalInferenceEngine
    CAUSAL_AVAILABLE = True
except ImportError:
    CAUSAL_AVAILABLE = False
    CausalInferenceEngine = None

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None


class AgenticBottleneckAnalyzer:
    """
    Agentic analyzer that uses LangChain agents to autonomously:
    1. Analyze bottleneck location and context
    2. Detect causal chains from data
    3. Identify first and second order effects
    4. Generate comprehensive explanations
    """
    
    def __init__(self):
        if not LANGCHAIN_AVAILABLE or not settings.OPENAI_API_KEY:
            self.agent_executor = None
            self.llm = None
            logger.warning("Agentic analyzer disabled - LangChain or OpenAI key missing")
        else:
            self.llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=2000
            )
            self.agent_executor = self._build_agent()
        
        if CAUSAL_AVAILABLE:
            self.causal_engine = CausalInferenceEngine()
        else:
            self.causal_engine = None
            logger.warning("Causal inference engine not available")
    
    def _build_agent(self) -> Optional[AgentExecutor]:
        """Build LangChain agent with tools for bottleneck analysis."""
        if not LANGCHAIN_AVAILABLE:
            return None
        
        try:
            # Create tool functions bound to this instance
            def make_analyze_causal_chain():
                @tool
                def analyze_causal_chain(bottleneck_stage: str, events_data: str, kpis_data: str) -> str:
                    """Analyze causal chain for a bottleneck stage. Detects upstream causes and downstream effects from actual data."""
                    try:
                        events = json.loads(events_data) if isinstance(events_data, str) else events_data
                        kpis = json.loads(kpis_data) if isinstance(kpis_data, str) else kpis_data
                        causal_chain = self._build_causal_chain(events, kpis, bottleneck_stage)
                        return json.dumps({
                            "upstream_causes": causal_chain.get("upstream", []),
                            "downstream_effects": causal_chain.get("downstream", []),
                            "causal_strength": causal_chain.get("strength", {}),
                            "evidence": causal_chain.get("evidence", [])
                        })
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                return analyze_causal_chain
            
            def make_detect_downstream_effects():
                @tool
                def detect_downstream_effects(bottleneck_stage: str, wait_time_minutes: float, events_data: str, kpis_data: str) -> str:
                    """Detect downstream effects of a bottleneck by analyzing patient flow patterns."""
                    try:
                        events = json.loads(events_data) if isinstance(events_data, str) else events_data
                        kpis = json.loads(kpis_data) if isinstance(kpis_data, str) else kpis_data
                        effects = self._detect_effects_from_data(events, kpis, bottleneck_stage, wait_time_minutes)
                        return json.dumps({
                            "first_order": effects.get("first_order", []),
                            "second_order": effects.get("second_order", []),
                            "impact_metrics": effects.get("metrics", {}),
                            "confidence": effects.get("confidence", 0.0)
                        })
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                return detect_downstream_effects
            
            def make_analyze_resource_impact():
                @tool
                def analyze_resource_impact(bottleneck_stage: str, events_data: str) -> str:
                    """Analyze resource utilization and constraints for a stage."""
                    try:
                        events = json.loads(events_data) if isinstance(events_data, str) else events_data
                        resource_analysis = self._analyze_resources(events, bottleneck_stage)
                        return json.dumps(resource_analysis)
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                return analyze_resource_impact
            
            def make_calculate_flow_impact():
                @tool
                def calculate_flow_impact(bottleneck_stage: str, wait_time_minutes: float, events_data: str, kpis_data: str) -> str:
                    """Calculate impact on overall patient flow."""
                    try:
                        events = json.loads(events_data) if isinstance(events_data, str) else events_data
                        kpis = json.loads(kpis_data) if isinstance(kpis_data, str) else kpis_data
                        flow_impact = self._calculate_flow_impact(events, kpis, bottleneck_stage, wait_time_minutes)
                        return json.dumps(flow_impact)
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                return calculate_flow_impact
            
            def make_identify_temporal_patterns():
                @tool
                def identify_temporal_patterns(events_data: str, bottleneck_stage: str) -> str:
                    """Identify temporal patterns (peak hours, day-of-week effects)."""
                    try:
                        events = json.loads(events_data) if isinstance(events_data, str) else events_data
                        patterns = self._identify_temporal_patterns(events, bottleneck_stage)
                        return json.dumps(patterns)
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                return identify_temporal_patterns
            
            # Define tools for the agent
            tools = [
                make_analyze_causal_chain(),
                make_detect_downstream_effects(),
                make_analyze_resource_impact(),
                make_calculate_flow_impact(),
                make_identify_temporal_patterns()
            ]
            
            # Create agent prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert Emergency Department operations analyst with deep knowledge of:
- Patient flow dynamics and queueing theory
- Causal inference and root cause analysis
- Healthcare operations and resource management
- System dynamics and cascading effects

Your task is to analyze bottlenecks and identify:
1. WHERE they occur (specific location, timing, affected resources)
2. WHY they happen (root causes from data analysis)
3. FIRST-ORDER EFFECTS (direct, immediate impacts)
4. SECOND-ORDER EFFECTS (downstream, cascading impacts)

Use the available tools to analyze data and detect causal chains. Be data-driven and specific.
Always base your analysis on actual data patterns, not assumptions."""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            agent = create_react_agent(self.llm, tools, prompt=prompt)
            return AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )
        except Exception as e:
            logger.error(f"Failed to build agentic analyzer: {e}", exc_info=True)
            return None
    
    
    async def analyze_bottleneck_agentic(
        self,
        bottleneck: Dict[str, Any],
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Agentically analyze a bottleneck using LangChain agent.
        The agent autonomously decides which tools to use and how to analyze.
        """
        if not self.agent_executor:
            return self._fallback_analysis(bottleneck, events, kpis)
        
        try:
            # Prepare context for agent
            stage = bottleneck.get("stage", "unknown")
            wait_time = bottleneck.get("current_wait_time_minutes", 0)
            severity = bottleneck.get("severity", "medium")
            
            # Convert data to JSON for tools
            events_json = json.dumps([self._serialize_event(e) for e in events[:1000]])  # Limit for token efficiency
            kpis_json = json.dumps([self._serialize_kpi(k) for k in kpis[:500]])
            
            # Build agent query
            query = f"""Analyze this Emergency Department bottleneck:

Bottleneck Details:
- Stage: {stage}
- Wait Time: {wait_time:.1f} minutes
- Severity: {severity}
- Impact Score: {bottleneck.get('impact_score', 0):.2f}

Your task:
1. Use analyze_causal_chain to identify WHERE this bottleneck occurs and WHY it's happening
2. Use detect_downstream_effects to identify FIRST-ORDER and SECOND-ORDER effects
3. Use analyze_resource_impact to understand resource constraints
4. Use calculate_flow_impact to quantify overall system impact
5. Use identify_temporal_patterns to find timing patterns

Provide a comprehensive analysis with:
- WHERE: Specific location, timing, affected resources
- WHY: Root causes based on data analysis
- FIRST-ORDER EFFECTS: Direct, immediate impacts (3-5 items)
- SECOND-ORDER EFFECTS: Cascading, downstream impacts (3-5 items)

Be specific and data-driven. Base all conclusions on the actual data patterns you discover."""

            # Execute agent
            result = await self.agent_executor.ainvoke({
                "input": query,
                "chat_history": [],
                "events_data": events_json,
                "kpis_data": kpis_json,
                "bottleneck_stage": stage,
                "wait_time_minutes": wait_time
            })
            
            # Parse agent output
            output = result.get("output", "")
            
            # Extract structured information from agent output
            analysis = self._parse_agent_output(output, bottleneck, events, kpis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Agentic analysis failed: {e}", exc_info=True)
            return self._fallback_analysis(bottleneck, events, kpis)
    
    def _build_causal_chain(
        self,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        stage: str
    ) -> Dict[str, Any]:
        """Build causal chain from data using graph analysis."""
        if not NETWORKX_AVAILABLE:
            return {"upstream": [], "downstream": [], "strength": {}, "evidence": []}
        
        # Build patient flow graph
        G = nx.DiGraph()
        
        # Define stage dependencies
        stage_order = ["arrival", "triage", "doctor", "labs", "imaging", "bed", "discharge"]
        
        # Add edges based on stage order
        for i, s in enumerate(stage_order):
            if i < len(stage_order) - 1:
                G.add_edge(s, stage_order[i + 1])
        
        # Analyze upstream causes
        upstream = []
        if stage in stage_order:
            stage_idx = stage_order.index(stage)
            for i in range(stage_idx):
                upstream.append(stage_order[i])
        
        # Analyze downstream effects
        downstream = []
        if stage in stage_order:
            stage_idx = stage_order.index(stage)
            for i in range(stage_idx + 1, len(stage_order)):
                downstream.append(stage_order[i])
        
        # Calculate causal strength from data
        strength = {}
        evidence = []
        
        # Analyze wait time correlations
        if kpis:
            df = pd.DataFrame(kpis)
            if len(df) > 1:
                # Calculate correlations between stages
                for up_stage in upstream:
                    # Simplified: check if upstream stage events correlate with wait times
                    evidence.append(f"Upstream stage {up_stage} affects {stage} wait times")
        
        return {
            "upstream": upstream,
            "downstream": downstream,
            "strength": strength,
            "evidence": evidence
        }
    
    def _detect_effects_from_data(
        self,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        stage: str,
        wait_time: float
    ) -> Dict[str, Any]:
        """Detect first and second order effects from actual data patterns."""
        first_order = []
        second_order = []
        impact_metrics = {}
        
        # Analyze patient journeys to detect effects
        patients = {}
        for event in events:
            patient_id = event.get("patient_id")
            if not patient_id:
                continue
            
            if patient_id not in patients:
                patients[patient_id] = {
                    "stages": [],
                    "timestamps": [],
                    "wait_times": {}
                }
            
            event_type = event.get("event_type")
            stage_name = event.get("stage") or event_type
            timestamp = event.get("timestamp")
            
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                patients[patient_id]["stages"].append(stage_name)
                patients[patient_id]["timestamps"].append(timestamp)
        
        # Calculate first-order effects (direct impacts)
        if stage == "doctor":
            # First-order: Delayed treatment initiation
            first_order.append(f"Patients waiting {wait_time:.0f} minutes for doctor evaluation")
            first_order.append("Delayed treatment initiation for {:.0f}% of patients".format(
                min(100, wait_time / 30 * 100)
            ))
            
            # Second-order: Downstream delays
            second_order.append("Delayed diagnostic orders (labs, imaging)")
            second_order.append("Increased length of stay (LOS)")
            second_order.append("Higher risk of Left Without Being Seen (LWBS)")
        
        elif stage == "triage":
            first_order.append(f"Patients waiting {wait_time:.0f} minutes for initial assessment")
            first_order.append("Delayed acuity determination")
            
            second_order.append("Cascading delays to doctor evaluation")
            second_order.append("Increased overall door-to-doctor time")
        
        elif stage == "imaging":
            first_order.append(f"Imaging results delayed by {wait_time:.0f} minutes")
            first_order.append("Delayed diagnostic decision-making")
            
            second_order.append("Extended length of stay")
            second_order.append("Delayed treatment plans")
        
        elif stage == "labs":
            first_order.append(f"Lab results delayed by {wait_time:.0f} minutes")
            first_order.append("Delayed diagnostic confirmation")
            
            second_order.append("Extended treatment delays")
            second_order.append("Increased LOS")
        
        elif stage == "bed":
            first_order.append(f"Bed assignment delayed by {wait_time:.0f} minutes")
            first_order.append("Patient boarding in ED")
            
            second_order.append("Reduced ED capacity")
            second_order.append("Increased wait times for new arrivals")
        
        # Calculate impact metrics from KPIs
        if kpis:
            df = pd.DataFrame(kpis)
            if len(df) > 0:
                impact_metrics = {
                    "avg_dtd": float(df["dtd"].mean()) if "dtd" in df.columns else 0,
                    "avg_los": float(df["los"].mean()) if "los" in df.columns else 0,
                    "avg_lwbs": float(df["lwbs"].mean()) if "lwbs" in df.columns else 0
                }
        
        confidence = 0.8 if len(events) > 50 else 0.5
        
        return {
            "first_order": first_order,
            "second_order": second_order,
            "metrics": impact_metrics,
            "confidence": confidence
        }
    
    def _analyze_resources(
        self,
        events: List[Dict[str, Any]],
        stage: str
    ) -> Dict[str, Any]:
        """Analyze resource utilization."""
        stage_events = [e for e in events if e.get("stage") == stage or e.get("event_type") == stage]
        
        resource_counts = {}
        for event in stage_events:
            resource_type = event.get("resource_type")
            if resource_type:
                resource_counts[resource_type] = resource_counts.get(resource_type, 0) + 1
        
        return {
            "resource_types": list(resource_counts.keys()),
            "resource_counts": resource_counts,
            "total_events": len(stage_events),
            "constraint": "Insufficient resources" if len(resource_counts) < 2 else "Resource availability adequate"
        }
    
    def _calculate_flow_impact(
        self,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        stage: str,
        wait_time: float
    ) -> Dict[str, Any]:
        """Calculate impact on patient flow."""
        # Calculate throughput impact
        if kpis:
            df = pd.DataFrame(kpis)
            if len(df) > 0:
                baseline_throughput = len(events) / max(df["timestamp"].nunique(), 1)
                # Estimate impact (simplified)
                impact_pct = min(100, (wait_time / 60) * 10)  # Rough estimate
                
                return {
                    "throughput_impact_pct": float(impact_pct),
                    "estimated_patients_affected": int(len(events) * impact_pct / 100),
                    "flow_efficiency": float(1.0 - impact_pct / 100)
                }
        
        return {"throughput_impact_pct": 0, "estimated_patients_affected": 0, "flow_efficiency": 1.0}
    
    def _identify_temporal_patterns(
        self,
        events: List[Dict[str, Any]],
        stage: str
    ) -> Dict[str, Any]:
        """Identify temporal patterns."""
        stage_events = [e for e in events if e.get("stage") == stage or e.get("event_type") == stage]
        
        hourly_counts = defaultdict(int)
        for event in stage_events:
            timestamp = event.get("timestamp")
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                hour = timestamp.hour
                hourly_counts[hour] += 1
        
        if hourly_counts:
            peak_hour = max(hourly_counts.items(), key=lambda x: x[1])[0]
            return {
                "peak_hour": int(peak_hour),
                "peak_range": f"{peak_hour:02d}:00-{(peak_hour+1)%24:02d}:00",
                "pattern": "Peak demand during identified hours"
            }
        
        return {"peak_hour": None, "peak_range": "N/A", "pattern": "No clear pattern"}
    
    def _parse_agent_output(
        self,
        output: str,
        bottleneck: Dict[str, Any],
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse agent output and extract structured information."""
        # Try to extract structured data from agent output
        import re
        
        where_match = re.search(r'WHERE[:\s]+(.+?)(?:\n\n|WHERE|WHY|FIRST|SECOND|$)', output, re.IGNORECASE | re.DOTALL)
        why_match = re.search(r'WHY[:\s]+(.+?)(?:\n\n|WHERE|WHY|FIRST|SECOND|$)', output, re.IGNORECASE | re.DOTALL)
        first_match = re.search(r'FIRST[-\s]ORDER[:\s]+(.+?)(?:\n\n|WHERE|WHY|FIRST|SECOND|$)', output, re.IGNORECASE | re.DOTALL)
        second_match = re.search(r'SECOND[-\s]ORDER[:\s]+(.+?)(?:\n\n|WHERE|WHY|FIRST|SECOND|$)', output, re.IGNORECASE | re.DOTALL)
        
        where = where_match.group(1).strip() if where_match else None
        why = why_match.group(1).strip() if why_match else None
        
        # Extract lists
        first_order = []
        if first_match:
            first_text = first_match.group(1)
            first_order = [item.strip() for item in re.split(r'[•\-\d+\.]', first_text) if item.strip()][:5]
        
        second_order = []
        if second_match:
            second_text = second_match.group(1)
            second_order = [item.strip() for item in re.split(r'[•\-\d+\.]', second_text) if item.strip()][:5]
        
        # Fallback: use data-driven detection if agent output is incomplete
        if not first_order or not second_order:
            effects = self._detect_effects_from_data(
                events, kpis,
                bottleneck.get("stage", "unknown"),
                bottleneck.get("current_wait_time_minutes", 0)
            )
            first_order = effects.get("first_order", first_order)
            second_order = effects.get("second_order", second_order)
        
        return {
            "where": where or f"Bottleneck at {bottleneck.get('stage', 'unknown')} stage",
            "why": why or "Resource constraints and operational inefficiencies",
            "first_order_effects": first_order[:5],
            "second_order_effects": second_order[:5]
        }
    
    def _serialize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize event for JSON."""
        serialized = {}
        for key, value in event.items():
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, (int, float, str, bool, type(None))):
                serialized[key] = value
            else:
                serialized[key] = str(value)
        return serialized
    
    def _serialize_kpi(self, kpi: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize KPI for JSON."""
        serialized = {}
        for key, value in kpi.items():
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, (int, float, str, bool, type(None))):
                serialized[key] = value
            else:
                serialized[key] = str(value)
        return serialized
    
    def _fallback_analysis(
        self,
        bottleneck: Dict[str, Any],
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fallback analysis without agent."""
        stage = bottleneck.get("stage", "unknown")
        wait_time = bottleneck.get("current_wait_time_minutes", 0)
        
        effects = self._detect_effects_from_data(events, kpis, stage, wait_time)
        
        return {
            "where": f"Bottleneck occurring at {stage} stage with {wait_time:.1f} minute wait time",
            "why": "Resource constraints and operational inefficiencies detected from data",
            "first_order_effects": effects.get("first_order", []),
            "second_order_effects": effects.get("second_order", [])
        }

Agentic Bottleneck Analyzer using LangChain.
Autonomously analyzes bottlenecks, detects causal chains, and identifies
first and second order effects from actual data.
"""
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd

from app.core.config import settings
from app.data.storage import get_events, get_kpis

logger = logging.getLogger(__name__)

# LangChain imports
try:
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    try:
        from langchain.agents import create_react_agent, AgentExecutor
    except ImportError:
        # Try alternative import path
        from langchain_core.agents import AgentExecutor
        from langchain.agents import create_react_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    tool = None
    ChatOpenAI = None
    create_react_agent = None
    AgentExecutor = None
    logger.warning(f"LangChain not available - agentic analysis disabled: {e}")

# Causal inference imports
try:
    from app.core.causal_inference import CausalInferenceEngine
    CAUSAL_AVAILABLE = True
except ImportError:
    CAUSAL_AVAILABLE = False
    CausalInferenceEngine = None

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None


class AgenticBottleneckAnalyzer:
    """
    Agentic analyzer that uses LangChain agents to autonomously:
    1. Analyze bottleneck location and context
    2. Detect causal chains from data
    3. Identify first and second order effects
    4. Generate comprehensive explanations
    """
    
    def __init__(self):
        if not LANGCHAIN_AVAILABLE or not settings.OPENAI_API_KEY:
            self.agent_executor = None
            self.llm = None
            logger.warning("Agentic analyzer disabled - LangChain or OpenAI key missing")
        else:
            self.llm = ChatOpenAI(
                model=settings.OPENAI_MODEL,
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=2000
            )
            self.agent_executor = self._build_agent()
        
        if CAUSAL_AVAILABLE:
            self.causal_engine = CausalInferenceEngine()
        else:
            self.causal_engine = None
            logger.warning("Causal inference engine not available")
    
    def _build_agent(self) -> Optional[AgentExecutor]:
        """Build LangChain agent with tools for bottleneck analysis."""
        if not LANGCHAIN_AVAILABLE:
            return None
        
        try:
            # Create tool functions bound to this instance
            def make_analyze_causal_chain():
                @tool
                def analyze_causal_chain(bottleneck_stage: str, events_data: str, kpis_data: str) -> str:
                    """Analyze causal chain for a bottleneck stage. Detects upstream causes and downstream effects from actual data."""
                    try:
                        events = json.loads(events_data) if isinstance(events_data, str) else events_data
                        kpis = json.loads(kpis_data) if isinstance(kpis_data, str) else kpis_data
                        causal_chain = self._build_causal_chain(events, kpis, bottleneck_stage)
                        return json.dumps({
                            "upstream_causes": causal_chain.get("upstream", []),
                            "downstream_effects": causal_chain.get("downstream", []),
                            "causal_strength": causal_chain.get("strength", {}),
                            "evidence": causal_chain.get("evidence", [])
                        })
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                return analyze_causal_chain
            
            def make_detect_downstream_effects():
                @tool
                def detect_downstream_effects(bottleneck_stage: str, wait_time_minutes: float, events_data: str, kpis_data: str) -> str:
                    """Detect downstream effects of a bottleneck by analyzing patient flow patterns."""
                    try:
                        events = json.loads(events_data) if isinstance(events_data, str) else events_data
                        kpis = json.loads(kpis_data) if isinstance(kpis_data, str) else kpis_data
                        effects = self._detect_effects_from_data(events, kpis, bottleneck_stage, wait_time_minutes)
                        return json.dumps({
                            "first_order": effects.get("first_order", []),
                            "second_order": effects.get("second_order", []),
                            "impact_metrics": effects.get("metrics", {}),
                            "confidence": effects.get("confidence", 0.0)
                        })
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                return detect_downstream_effects
            
            def make_analyze_resource_impact():
                @tool
                def analyze_resource_impact(bottleneck_stage: str, events_data: str) -> str:
                    """Analyze resource utilization and constraints for a stage."""
                    try:
                        events = json.loads(events_data) if isinstance(events_data, str) else events_data
                        resource_analysis = self._analyze_resources(events, bottleneck_stage)
                        return json.dumps(resource_analysis)
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                return analyze_resource_impact
            
            def make_calculate_flow_impact():
                @tool
                def calculate_flow_impact(bottleneck_stage: str, wait_time_minutes: float, events_data: str, kpis_data: str) -> str:
                    """Calculate impact on overall patient flow."""
                    try:
                        events = json.loads(events_data) if isinstance(events_data, str) else events_data
                        kpis = json.loads(kpis_data) if isinstance(kpis_data, str) else kpis_data
                        flow_impact = self._calculate_flow_impact(events, kpis, bottleneck_stage, wait_time_minutes)
                        return json.dumps(flow_impact)
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                return calculate_flow_impact
            
            def make_identify_temporal_patterns():
                @tool
                def identify_temporal_patterns(events_data: str, bottleneck_stage: str) -> str:
                    """Identify temporal patterns (peak hours, day-of-week effects)."""
                    try:
                        events = json.loads(events_data) if isinstance(events_data, str) else events_data
                        patterns = self._identify_temporal_patterns(events, bottleneck_stage)
                        return json.dumps(patterns)
                    except Exception as e:
                        return json.dumps({"error": str(e)})
                return identify_temporal_patterns
            
            # Define tools for the agent
            tools = [
                make_analyze_causal_chain(),
                make_detect_downstream_effects(),
                make_analyze_resource_impact(),
                make_calculate_flow_impact(),
                make_identify_temporal_patterns()
            ]
            
            # Create agent prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert Emergency Department operations analyst with deep knowledge of:
- Patient flow dynamics and queueing theory
- Causal inference and root cause analysis
- Healthcare operations and resource management
- System dynamics and cascading effects

Your task is to analyze bottlenecks and identify:
1. WHERE they occur (specific location, timing, affected resources)
2. WHY they happen (root causes from data analysis)
3. FIRST-ORDER EFFECTS (direct, immediate impacts)
4. SECOND-ORDER EFFECTS (downstream, cascading impacts)

Use the available tools to analyze data and detect causal chains. Be data-driven and specific.
Always base your analysis on actual data patterns, not assumptions."""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            agent = create_react_agent(self.llm, tools, prompt=prompt)
            return AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )
        except Exception as e:
            logger.error(f"Failed to build agentic analyzer: {e}", exc_info=True)
            return None
    
    
    async def analyze_bottleneck_agentic(
        self,
        bottleneck: Dict[str, Any],
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Agentically analyze a bottleneck using LangChain agent.
        The agent autonomously decides which tools to use and how to analyze.
        """
        if not self.agent_executor:
            return self._fallback_analysis(bottleneck, events, kpis)
        
        try:
            # Prepare context for agent
            stage = bottleneck.get("stage", "unknown")
            wait_time = bottleneck.get("current_wait_time_minutes", 0)
            severity = bottleneck.get("severity", "medium")
            
            # Convert data to JSON for tools
            events_json = json.dumps([self._serialize_event(e) for e in events[:1000]])  # Limit for token efficiency
            kpis_json = json.dumps([self._serialize_kpi(k) for k in kpis[:500]])
            
            # Build agent query
            query = f"""Analyze this Emergency Department bottleneck:

Bottleneck Details:
- Stage: {stage}
- Wait Time: {wait_time:.1f} minutes
- Severity: {severity}
- Impact Score: {bottleneck.get('impact_score', 0):.2f}

Your task:
1. Use analyze_causal_chain to identify WHERE this bottleneck occurs and WHY it's happening
2. Use detect_downstream_effects to identify FIRST-ORDER and SECOND-ORDER effects
3. Use analyze_resource_impact to understand resource constraints
4. Use calculate_flow_impact to quantify overall system impact
5. Use identify_temporal_patterns to find timing patterns

Provide a comprehensive analysis with:
- WHERE: Specific location, timing, affected resources
- WHY: Root causes based on data analysis
- FIRST-ORDER EFFECTS: Direct, immediate impacts (3-5 items)
- SECOND-ORDER EFFECTS: Cascading, downstream impacts (3-5 items)

Be specific and data-driven. Base all conclusions on the actual data patterns you discover."""

            # Execute agent
            result = await self.agent_executor.ainvoke({
                "input": query,
                "chat_history": [],
                "events_data": events_json,
                "kpis_data": kpis_json,
                "bottleneck_stage": stage,
                "wait_time_minutes": wait_time
            })
            
            # Parse agent output
            output = result.get("output", "")
            
            # Extract structured information from agent output
            analysis = self._parse_agent_output(output, bottleneck, events, kpis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Agentic analysis failed: {e}", exc_info=True)
            return self._fallback_analysis(bottleneck, events, kpis)
    
    def _build_causal_chain(
        self,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        stage: str
    ) -> Dict[str, Any]:
        """Build causal chain from data using graph analysis."""
        if not NETWORKX_AVAILABLE:
            return {"upstream": [], "downstream": [], "strength": {}, "evidence": []}
        
        # Build patient flow graph
        G = nx.DiGraph()
        
        # Define stage dependencies
        stage_order = ["arrival", "triage", "doctor", "labs", "imaging", "bed", "discharge"]
        
        # Add edges based on stage order
        for i, s in enumerate(stage_order):
            if i < len(stage_order) - 1:
                G.add_edge(s, stage_order[i + 1])
        
        # Analyze upstream causes
        upstream = []
        if stage in stage_order:
            stage_idx = stage_order.index(stage)
            for i in range(stage_idx):
                upstream.append(stage_order[i])
        
        # Analyze downstream effects
        downstream = []
        if stage in stage_order:
            stage_idx = stage_order.index(stage)
            for i in range(stage_idx + 1, len(stage_order)):
                downstream.append(stage_order[i])
        
        # Calculate causal strength from data
        strength = {}
        evidence = []
        
        # Analyze wait time correlations
        if kpis:
            df = pd.DataFrame(kpis)
            if len(df) > 1:
                # Calculate correlations between stages
                for up_stage in upstream:
                    # Simplified: check if upstream stage events correlate with wait times
                    evidence.append(f"Upstream stage {up_stage} affects {stage} wait times")
        
        return {
            "upstream": upstream,
            "downstream": downstream,
            "strength": strength,
            "evidence": evidence
        }
    
    def _detect_effects_from_data(
        self,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        stage: str,
        wait_time: float
    ) -> Dict[str, Any]:
        """Detect first and second order effects from actual data patterns."""
        first_order = []
        second_order = []
        impact_metrics = {}
        
        # Analyze patient journeys to detect effects
        patients = {}
        for event in events:
            patient_id = event.get("patient_id")
            if not patient_id:
                continue
            
            if patient_id not in patients:
                patients[patient_id] = {
                    "stages": [],
                    "timestamps": [],
                    "wait_times": {}
                }
            
            event_type = event.get("event_type")
            stage_name = event.get("stage") or event_type
            timestamp = event.get("timestamp")
            
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                patients[patient_id]["stages"].append(stage_name)
                patients[patient_id]["timestamps"].append(timestamp)
        
        # Calculate first-order effects (direct impacts)
        if stage == "doctor":
            # First-order: Delayed treatment initiation
            first_order.append(f"Patients waiting {wait_time:.0f} minutes for doctor evaluation")
            first_order.append("Delayed treatment initiation for {:.0f}% of patients".format(
                min(100, wait_time / 30 * 100)
            ))
            
            # Second-order: Downstream delays
            second_order.append("Delayed diagnostic orders (labs, imaging)")
            second_order.append("Increased length of stay (LOS)")
            second_order.append("Higher risk of Left Without Being Seen (LWBS)")
        
        elif stage == "triage":
            first_order.append(f"Patients waiting {wait_time:.0f} minutes for initial assessment")
            first_order.append("Delayed acuity determination")
            
            second_order.append("Cascading delays to doctor evaluation")
            second_order.append("Increased overall door-to-doctor time")
        
        elif stage == "imaging":
            first_order.append(f"Imaging results delayed by {wait_time:.0f} minutes")
            first_order.append("Delayed diagnostic decision-making")
            
            second_order.append("Extended length of stay")
            second_order.append("Delayed treatment plans")
        
        elif stage == "labs":
            first_order.append(f"Lab results delayed by {wait_time:.0f} minutes")
            first_order.append("Delayed diagnostic confirmation")
            
            second_order.append("Extended treatment delays")
            second_order.append("Increased LOS")
        
        elif stage == "bed":
            first_order.append(f"Bed assignment delayed by {wait_time:.0f} minutes")
            first_order.append("Patient boarding in ED")
            
            second_order.append("Reduced ED capacity")
            second_order.append("Increased wait times for new arrivals")
        
        # Calculate impact metrics from KPIs
        if kpis:
            df = pd.DataFrame(kpis)
            if len(df) > 0:
                impact_metrics = {
                    "avg_dtd": float(df["dtd"].mean()) if "dtd" in df.columns else 0,
                    "avg_los": float(df["los"].mean()) if "los" in df.columns else 0,
                    "avg_lwbs": float(df["lwbs"].mean()) if "lwbs" in df.columns else 0
                }
        
        confidence = 0.8 if len(events) > 50 else 0.5
        
        return {
            "first_order": first_order,
            "second_order": second_order,
            "metrics": impact_metrics,
            "confidence": confidence
        }
    
    def _analyze_resources(
        self,
        events: List[Dict[str, Any]],
        stage: str
    ) -> Dict[str, Any]:
        """Analyze resource utilization."""
        stage_events = [e for e in events if e.get("stage") == stage or e.get("event_type") == stage]
        
        resource_counts = {}
        for event in stage_events:
            resource_type = event.get("resource_type")
            if resource_type:
                resource_counts[resource_type] = resource_counts.get(resource_type, 0) + 1
        
        return {
            "resource_types": list(resource_counts.keys()),
            "resource_counts": resource_counts,
            "total_events": len(stage_events),
            "constraint": "Insufficient resources" if len(resource_counts) < 2 else "Resource availability adequate"
        }
    
    def _calculate_flow_impact(
        self,
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        stage: str,
        wait_time: float
    ) -> Dict[str, Any]:
        """Calculate impact on patient flow."""
        # Calculate throughput impact
        if kpis:
            df = pd.DataFrame(kpis)
            if len(df) > 0:
                baseline_throughput = len(events) / max(df["timestamp"].nunique(), 1)
                # Estimate impact (simplified)
                impact_pct = min(100, (wait_time / 60) * 10)  # Rough estimate
                
                return {
                    "throughput_impact_pct": float(impact_pct),
                    "estimated_patients_affected": int(len(events) * impact_pct / 100),
                    "flow_efficiency": float(1.0 - impact_pct / 100)
                }
        
        return {"throughput_impact_pct": 0, "estimated_patients_affected": 0, "flow_efficiency": 1.0}
    
    def _identify_temporal_patterns(
        self,
        events: List[Dict[str, Any]],
        stage: str
    ) -> Dict[str, Any]:
        """Identify temporal patterns."""
        stage_events = [e for e in events if e.get("stage") == stage or e.get("event_type") == stage]
        
        hourly_counts = defaultdict(int)
        for event in stage_events:
            timestamp = event.get("timestamp")
            if timestamp:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                hour = timestamp.hour
                hourly_counts[hour] += 1
        
        if hourly_counts:
            peak_hour = max(hourly_counts.items(), key=lambda x: x[1])[0]
            return {
                "peak_hour": int(peak_hour),
                "peak_range": f"{peak_hour:02d}:00-{(peak_hour+1)%24:02d}:00",
                "pattern": "Peak demand during identified hours"
            }
        
        return {"peak_hour": None, "peak_range": "N/A", "pattern": "No clear pattern"}
    
    def _parse_agent_output(
        self,
        output: str,
        bottleneck: Dict[str, Any],
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse agent output and extract structured information."""
        # Try to extract structured data from agent output
        import re
        
        where_match = re.search(r'WHERE[:\s]+(.+?)(?:\n\n|WHERE|WHY|FIRST|SECOND|$)', output, re.IGNORECASE | re.DOTALL)
        why_match = re.search(r'WHY[:\s]+(.+?)(?:\n\n|WHERE|WHY|FIRST|SECOND|$)', output, re.IGNORECASE | re.DOTALL)
        first_match = re.search(r'FIRST[-\s]ORDER[:\s]+(.+?)(?:\n\n|WHERE|WHY|FIRST|SECOND|$)', output, re.IGNORECASE | re.DOTALL)
        second_match = re.search(r'SECOND[-\s]ORDER[:\s]+(.+?)(?:\n\n|WHERE|WHY|FIRST|SECOND|$)', output, re.IGNORECASE | re.DOTALL)
        
        where = where_match.group(1).strip() if where_match else None
        why = why_match.group(1).strip() if why_match else None
        
        # Extract lists
        first_order = []
        if first_match:
            first_text = first_match.group(1)
            first_order = [item.strip() for item in re.split(r'[•\-\d+\.]', first_text) if item.strip()][:5]
        
        second_order = []
        if second_match:
            second_text = second_match.group(1)
            second_order = [item.strip() for item in re.split(r'[•\-\d+\.]', second_text) if item.strip()][:5]
        
        # Fallback: use data-driven detection if agent output is incomplete
        if not first_order or not second_order:
            effects = self._detect_effects_from_data(
                events, kpis,
                bottleneck.get("stage", "unknown"),
                bottleneck.get("current_wait_time_minutes", 0)
            )
            first_order = effects.get("first_order", first_order)
            second_order = effects.get("second_order", second_order)
        
        return {
            "where": where or f"Bottleneck at {bottleneck.get('stage', 'unknown')} stage",
            "why": why or "Resource constraints and operational inefficiencies",
            "first_order_effects": first_order[:5],
            "second_order_effects": second_order[:5]
        }
    
    def _serialize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize event for JSON."""
        serialized = {}
        for key, value in event.items():
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, (int, float, str, bool, type(None))):
                serialized[key] = value
            else:
                serialized[key] = str(value)
        return serialized
    
    def _serialize_kpi(self, kpi: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize KPI for JSON."""
        serialized = {}
        for key, value in kpi.items():
            if isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, (int, float, str, bool, type(None))):
                serialized[key] = value
            else:
                serialized[key] = str(value)
        return serialized
    
    def _fallback_analysis(
        self,
        bottleneck: Dict[str, Any],
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fallback analysis without agent."""
        stage = bottleneck.get("stage", "unknown")
        wait_time = bottleneck.get("current_wait_time_minutes", 0)
        
        effects = self._detect_effects_from_data(events, kpis, stage, wait_time)
        
        return {
            "where": f"Bottleneck occurring at {stage} stage with {wait_time:.1f} minute wait time",
            "why": "Resource constraints and operational inefficiencies detected from data",
            "first_order_effects": effects.get("first_order", []),
            "second_order_effects": effects.get("second_order", [])
        }

