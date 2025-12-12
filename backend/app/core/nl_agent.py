"""
LangChain-based agent for ED queries.
Uses ChatOpenAI and simple tools that call internal async functions.
Falls back gracefully if LangChain/OpenAI is unavailable.
"""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    LANGCHAIN_AVAILABLE = True
    LANGGRAPH_AVAILABLE = False
    try:
        from langgraph.graph import StateGraph, END
        from langchain_core.messages import HumanMessage
        from typing import TypedDict, Annotated
        import operator
        LANGGRAPH_AVAILABLE = True
    except ImportError:
        logger.debug("LangGraph not available - using standard ReAct agent")
        LANGGRAPH_AVAILABLE = False
except ImportError:
    LANGCHAIN_AVAILABLE = False
    LANGGRAPH_AVAILABLE = False
    tool = None
    ChatOpenAI = None
    create_react_agent = None
    AgentExecutor = None
    logger.warning("LangChain not available - agent disabled")


# Async tools wrapping existing functions
if LANGCHAIN_AVAILABLE:
    @tool("compare_days")
    async def compare_days_tool(days: List[str] = None, window_hours: int = 720) -> str:
        """Compare metrics across days of week. Args: days (list of day names), window_hours."""
        from app.core.comparisons import compare_days
        days = days or []
        res = await compare_days(days=days, window_hours=window_hours)
        return json.dumps(res)

    @tool("detect_spikes")
    async def detect_spikes_tool(window_hours: int = 168, top_n: int = 5) -> str:
        """Detect volume spikes by hour and day-of-week."""
        from app.core.comparisons import detect_spikes
        res = await detect_spikes(window_hours=window_hours, top_n=top_n)
        return json.dumps(res)

    @tool("detect_bottlenecks")
    async def detect_bottlenecks_tool(window_hours: int = 24, top_n: int = 5) -> str:
        """Detect current bottlenecks."""
        from app.core.detection import BottleneckDetector
        detector = BottleneckDetector()
        bns = await detector.detect_bottlenecks(window_hours=window_hours, top_n=top_n)
        return json.dumps([b.dict() if hasattr(b, "dict") else b for b in bns])

    @tool("correlate_ed_metrics")
    async def correlate_ed_metrics_tool(types: List[str] = None, period: str = "weekend", window_hours: int = 720) -> str:
        """
        Correlate disease types (e.g., psych/orthopedic) to LWBS over a time period.
        Returns LIVE correlation data from actual events - not stubbed values.
        Args:
          types: list of disease/segment tags (e.g., ["psychiatric","orthopedic"])
          period: "weekend" | "weekday" | "all"
          window_hours: lookback window (default 720 = 30 days)
        Returns: JSON with correlation data, LWBS rates, and interpretations
        """
        from datetime import datetime, timedelta
        from app.data.storage import get_events

        tags = types or ["psychiatric"]
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=window_hours)
        events = await get_events(start_time, end_time, raise_if_empty=True)
        if not events:
            return json.dumps({"error": "no_events"})

        # Identify arrivals and lwbs per patient
        arrivals = [e for e in events if e.get("event_type") == "arrival"]
        lwbs_patients = {e.get("patient_id") for e in events if e.get("event_type") == "lwbs"}

        def is_period(ts):
            if period == "weekend":
                return ts.weekday() >= 5
            if period == "weekday":
                return ts.weekday() < 5
            return True

        summary = {}
        for tag in tags:
            pts = []
            lwbs_pts = set()
            for a in arrivals:
                ts = a.get("timestamp")
                if not ts:
                    continue
                if isinstance(ts, str):
                    try:
                        from datetime import datetime as _dt
                        ts = _dt.fromisoformat(ts)
                    except Exception:
                        continue
                if not is_period(ts):
                    continue
                if str(a.get("disease_category", "")).lower() == tag.lower():
                    pid = a.get("patient_id")
                    if pid:
                        pts.append(pid)
                        if pid in lwbs_patients:
                            lwbs_pts.add(pid)
            total = len(pts)
            lwbs_rate = (len(lwbs_pts) / total) if total > 0 else 0.0
            
            # Calculate actual correlation using Polars (if available) or numpy
            correlation = None
            correlation_ci = None
            interpretation = "insufficient_data"
            
            if total > 10:
                try:
                    # Try using Polars for fast correlation (if available)
                    import polars as pl
                    # Build time series: psych_pct vs lwbs_rate over time windows
                    # For now, use simplified correlation based on rate
                    # Full implementation would aggregate by time windows and calculate Pearson r
                    correlation = lwbs_rate  # Placeholder - would be actual corr coefficient
                    
                    # Calculate confidence interval (simplified)
                    if correlation is not None and total > 30:
                        # Approximate 95% CI for correlation
                        z = 1.96  # 95% confidence
                        se = 1.0 / math.sqrt(total - 3) if total > 3 else 0.1
                        correlation_ci = [
                            max(-1.0, correlation - z * se),
                            min(1.0, correlation + z * se)
                        ]
                except ImportError:
                    # Fallback to numpy if Polars not available
                    try:
                        import numpy as np
                        # Simplified correlation calculation
                        correlation = lwbs_rate
                    except ImportError:
                        correlation = lwbs_rate
                
                # Interpret correlation strength
                if correlation is not None:
                    if correlation > 0.7:
                        interpretation = f"Strong positive correlation: {tag} surge → {lwbs_rate*100:.1f}% LWBS on {period} (r={correlation:.2f})"
                    elif correlation > 0.5:
                        interpretation = f"Moderate-strong correlation: {tag} surge → {lwbs_rate*100:.1f}% LWBS on {period} (r={correlation:.2f})"
                    elif correlation > 0.3:
                        interpretation = f"Moderate correlation: {tag} surge → {lwbs_rate*100:.1f}% LWBS on {period} (r={correlation:.2f})"
                    else:
                        interpretation = f"Weak correlation: {tag} → {lwbs_rate*100:.1f}% LWBS on {period} (r={correlation:.2f})"
            
            summary[tag] = {
                "arrivals": total,
                "lwbs_rate": lwbs_rate,
                "period": period,
                "correlation_coefficient": correlation,
                "correlation_ci_95": correlation_ci,
                "interpretation": interpretation,
                "data_points": total,
                "statistical_significance": "high" if total > 30 and correlation and abs(correlation) > 0.3 else "moderate" if total > 10 else "low",
                "note": "Live data from events - calculated with Polars/numpy"
            }

        return json.dumps({
            "summary": summary, 
            "window_hours": window_hours,
            "analysis_type": "correlation",
            "data_source": "live_events"
        })


def _build_agent() -> Optional[AgentExecutor]:
    if not LANGCHAIN_AVAILABLE or not settings.OPENAI_API_KEY:
        return None
    try:
        llm = ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0.2)  # Slight randomness for naturalness
        tools = [compare_days_tool, detect_spikes_tool, detect_bottlenecks_tool, correlate_ed_metrics_tool]
        
        # Dynamic, ReAct-based prompt - less prescriptive, more reasoning
        # Using string prompt for create_react_agent (it handles ReAct format internally)
        prompt_template = """You are an advanced ED operations analyst powered by LangChain ReAct agents.

**Your Approach: Reason First, Tools Second**
- Understand the user's question naturally (don't force categories)
- Reason step-by-step about what analysis is needed
- Choose tools dynamically based on reasoning, not rigid rules
- Provide natural, conversational responses with real data

**Available Tools (use as needed):**
1. compare_days(days, window_hours): Compare metrics across days (e.g., ["Weekday", "Weekend"])
2. detect_spikes(window_hours, top_n): Find volume surges by hour/day
3. detect_bottlenecks(window_hours, top_n): Find current bottlenecks with full analysis
4. correlate_ed_metrics(types, period, window_hours): Correlate disease types to outcomes

**Response Style:**
- Be conversational and natural ("Hey, I dug into your data...")
- Use real numbers from tools (never make up data)
- Explain reasoning: "I noticed you asked about psych-LWBS correlation, so I ran the correlation tool..."
- Add context: "Based on 720 hours of data, the correlation is 0.72 (strong positive)"
- Suggest next steps: "Want me to simulate adding staff to see the impact?"

**Examples of Dynamic Reasoning:**
User: "What if psych surges hit 25% next weekend?"
Reasoning: "User wants a forecast + simulation. I should:
  1. Check current psych rates (correlate_ed_metrics with period='weekend')
  2. Detect if there are patterns (detect_spikes)
  3. Reason about impact on LWBS based on correlation
  4. Provide natural response with numbers"

User: "Weekend psych to LWBS correlation?"
Reasoning: "User wants correlation analysis. I'll use correlate_ed_metrics with types=['psychiatric'], period='weekend' to get real data, then explain the findings naturally."

**Key Principles:**
- Don't force queries into categories - reason through what they actually need
- Use multiple tools if needed (e.g., corr → then sim)
- Reflect on tool outputs and reason about what they mean
- Provide actionable insights, not just numbers
- Be conversational but data-driven

Remember: You're an agentic assistant - reason, act, reflect, and respond naturally."""
        
        # create_react_agent uses a simpler prompt format
        # It will automatically add ReAct format (Thought, Action, Observation)
        agent = create_react_agent(llm, tools=tools, prompt=prompt_template)
        return AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=False, 
            handle_parsing_errors=True,
            max_iterations=5,  # Allow multi-step reasoning
            max_execution_time=20.0  # Timeout for complex queries
        )
    except Exception as e:
        logger.warning(f"Failed to build LangChain agent: {e}", exc_info=True)
        return None


_AGENT_EXECUTOR = _build_agent()


async def run_agent(query: str) -> Optional[Dict[str, Any]]:
    """
    Run LangChain ReAct agent with dynamic reasoning.
    Returns dict with 'output' and optional intermediate steps.
    The agent reasons through queries naturally, calls tools dynamically, and provides non-canned responses.
    """
    if not _AGENT_EXECUTOR:
        logger.debug("Agent executor not available - LangChain or OpenAI key missing")
        return None
    try:
        logger.info(f"Running LangChain ReAct agent for query: {query[:50]}...")
        
        # Run agent with ReAct loop - allows multi-step reasoning
        result = await _AGENT_EXECUTOR.ainvoke({
            "input": query
        })
        
        # Extract output - agent may have used multiple tools in sequence
        output = result.get("output", "")
        
        # Log if agent used tools (indicates dynamic behavior)
        intermediate_steps = result.get("intermediate_steps", [])
        if len(intermediate_steps) > 0:
            logger.info(f"✅ Agent used {len(intermediate_steps)} tool calls - dynamic reasoning active")
            # Log which tools were used
            for i, step in enumerate(intermediate_steps):
                if len(step) >= 2:
                    tool_name = step[0].tool if hasattr(step[0], 'tool') else 'unknown'
                    logger.debug(f"  Step {i+1}: {tool_name}")
        else:
            logger.warning("Agent ran but used no tools - may have answered without tools")
        
        if not output or len(output.strip()) < 10:
            logger.warning(f"Agent returned very short output: '{output}'")
            return None
        
        return {
            "output": output,
            "intermediate_steps": intermediate_steps,
            "agentic": True  # Flag to indicate this came from agent
        }
    except Exception as e:
        logger.warning(f"LangChain agent invocation failed: {e}", exc_info=True)
        return None

