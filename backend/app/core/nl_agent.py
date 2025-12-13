"""
LangChain-based agent for ED queries.
Uses ChatOpenAI and simple tools that call internal async functions.
Falls back gracefully if LangChain/OpenAI is unavailable.
"""
import asyncio
import json
import logging
import math
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
            # Track time series data for correlation calculation
            time_series_data = []  # List of (timestamp, is_tag_patient, is_lwbs)
            
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
                
                pid = a.get("patient_id")
                is_tag_patient = str(a.get("disease_category", "")).lower() == tag.lower()
                
                if pid:
                    if is_tag_patient:
                        pts.append(pid)
                        is_lwbs = pid in lwbs_patients
                        if is_lwbs:
                            lwbs_pts.add(pid)
                        # Store for time series correlation
                        time_series_data.append((ts, 1, 1 if is_lwbs else 0))
                    else:
                        # Also track non-tag patients for comparison
                        time_series_data.append((ts, 0, 1 if pid in lwbs_patients else 0))
            
            total = len(pts)
            lwbs_rate = (len(lwbs_pts) / total) if total > 0 else 0.0
            
            # Calculate actual correlation using Polars (if available) or numpy
            correlation = None
            correlation_ci = None
            interpretation = "insufficient_data"
            
            if total > 10 and len(time_series_data) > 20:
                try:
                    # Try using Polars for fast correlation calculation
                    import polars as pl
                    
                    # Build DataFrame with hourly aggregates
                    # Group by hour and calculate tag_pct and lwbs_rate
                    hourly_data = {}
                    for ts, is_tag, is_lwbs in time_series_data:
                        hour_key = ts.replace(minute=0, second=0, microsecond=0)
                        if hour_key not in hourly_data:
                            hourly_data[hour_key] = {"tag_count": 0, "total_count": 0, "lwbs_count": 0}
                        hourly_data[hour_key]["total_count"] += 1
                        if is_tag:
                            hourly_data[hour_key]["tag_count"] += 1
                        if is_lwbs:
                            hourly_data[hour_key]["lwbs_count"] += 1
                    
                    # Convert to lists for correlation
                    tag_pcts = []
                    lwbs_rates = []
                    for hour_key, data in hourly_data.items():
                        if data["total_count"] > 0:
                            tag_pct = (data["tag_count"] / data["total_count"]) * 100
                            lwbs_rate_hour = data["lwbs_count"] / data["total_count"]
                            tag_pcts.append(tag_pct)
                            lwbs_rates.append(lwbs_rate_hour)
                    
                    # Calculate Pearson correlation if we have enough data points
                    if len(tag_pcts) >= 5 and len(lwbs_rates) >= 5:
                        try:
                            # Use Polars for correlation
                            df = pl.DataFrame({
                                "tag_pct": tag_pcts,
                                "lwbs_rate": lwbs_rates
                            })
                            correlation = df.select(pl.corr("tag_pct", "lwbs_rate")).item()
                            
                            # Calculate confidence interval using Fisher transformation
                            if correlation is not None and math.isfinite(correlation) and len(tag_pcts) > 30:
                                n = len(tag_pcts)
                                # Fisher z-transformation for CI
                                z = 1.96  # 95% confidence
                                z_corr = 0.5 * math.log((1 + correlation) / (1 - correlation)) if abs(correlation) < 0.99 else 0
                                se = 1.0 / math.sqrt(n - 3)
                                z_lower = z_corr - z * se
                                z_upper = z_corr + z * se
                                # Transform back
                                correlation_ci = [
                                    max(-1.0, (math.exp(2 * z_lower) - 1) / (math.exp(2 * z_lower) + 1)),
                                    min(1.0, (math.exp(2 * z_upper) - 1) / (math.exp(2 * z_upper) + 1))
                                ]
                        except Exception as e:
                            logger.debug(f"Polars correlation calculation failed: {e}, using fallback")
                            # Fallback to numpy
                            import numpy as np
                            correlation = float(np.corrcoef(tag_pcts, lwbs_rates)[0, 1]) if len(tag_pcts) == len(lwbs_rates) else lwbs_rate
                except ImportError:
                    # Fallback to numpy if Polars not available
                    try:
                        import numpy as np
                        if len(tag_pcts) >= 5 and len(lwbs_rates) >= 5 and len(tag_pcts) == len(lwbs_rates):
                            correlation = float(np.corrcoef(tag_pcts, lwbs_rates)[0, 1])
                        else:
                            correlation = lwbs_rate  # Simplified fallback
                    except (ImportError, ValueError, TypeError) as e:
                        logger.debug(f"NumPy correlation calculation failed: {e}, using rate as proxy")
                        correlation = lwbs_rate
                except Exception as e:
                    logger.warning(f"Error calculating correlation: {e}, using rate as proxy")
                    correlation = lwbs_rate
                
                # Validate correlation is finite
                if correlation is not None and not math.isfinite(correlation):
                    correlation = lwbs_rate  # Fallback to rate
                
                # Interpret correlation strength
                if correlation is not None and math.isfinite(correlation):
                    abs_corr = abs(correlation)
                    if abs_corr > 0.7:
                        interpretation = f"Strong {'positive' if correlation > 0 else 'negative'} correlation: {tag} surge → {lwbs_rate*100:.1f}% LWBS on {period} (r={correlation:.2f})"
                    elif abs_corr > 0.5:
                        interpretation = f"Moderate-strong {'positive' if correlation > 0 else 'negative'} correlation: {tag} surge → {lwbs_rate*100:.1f}% LWBS on {period} (r={correlation:.2f})"
                    elif abs_corr > 0.3:
                        interpretation = f"Moderate {'positive' if correlation > 0 else 'negative'} correlation: {tag} surge → {lwbs_rate*100:.1f}% LWBS on {period} (r={correlation:.2f})"
                    else:
                        interpretation = f"Weak {'positive' if correlation > 0 else 'negative'} correlation: {tag} → {lwbs_rate*100:.1f}% LWBS on {period} (r={correlation:.2f})"
            
            summary[tag] = {
                "arrivals": total,
                "lwbs_rate": lwbs_rate,
                "period": period,
                "correlation_coefficient": correlation if correlation is not None and math.isfinite(correlation) else None,
                "correlation_ci_95": correlation_ci,
                "interpretation": interpretation,
                "data_points": total,
                "time_series_points": len(time_series_data),
                "statistical_significance": "high" if total > 30 and correlation and abs(correlation) > 0.3 else "moderate" if total > 10 else "low",
                "note": "Live data from events - calculated with Polars/numpy correlation"
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

LangChain-based agent for ED queries.
Uses ChatOpenAI and simple tools that call internal async functions.
Falls back gracefully if LangChain/OpenAI is unavailable.
"""
import asyncio
import json
import logging
import math
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
            # Track time series data for correlation calculation
            time_series_data = []  # List of (timestamp, is_tag_patient, is_lwbs)
            
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
                
                pid = a.get("patient_id")
                is_tag_patient = str(a.get("disease_category", "")).lower() == tag.lower()
                
                if pid:
                    if is_tag_patient:
                        pts.append(pid)
                        is_lwbs = pid in lwbs_patients
                        if is_lwbs:
                            lwbs_pts.add(pid)
                        # Store for time series correlation
                        time_series_data.append((ts, 1, 1 if is_lwbs else 0))
                    else:
                        # Also track non-tag patients for comparison
                        time_series_data.append((ts, 0, 1 if pid in lwbs_patients else 0))
            
            total = len(pts)
            lwbs_rate = (len(lwbs_pts) / total) if total > 0 else 0.0
            
            # Calculate actual correlation using Polars (if available) or numpy
            correlation = None
            correlation_ci = None
            interpretation = "insufficient_data"
            
            if total > 10 and len(time_series_data) > 20:
                try:
                    # Try using Polars for fast correlation calculation
                    import polars as pl
                    
                    # Build DataFrame with hourly aggregates
                    # Group by hour and calculate tag_pct and lwbs_rate
                    hourly_data = {}
                    for ts, is_tag, is_lwbs in time_series_data:
                        hour_key = ts.replace(minute=0, second=0, microsecond=0)
                        if hour_key not in hourly_data:
                            hourly_data[hour_key] = {"tag_count": 0, "total_count": 0, "lwbs_count": 0}
                        hourly_data[hour_key]["total_count"] += 1
                        if is_tag:
                            hourly_data[hour_key]["tag_count"] += 1
                        if is_lwbs:
                            hourly_data[hour_key]["lwbs_count"] += 1
                    
                    # Convert to lists for correlation
                    tag_pcts = []
                    lwbs_rates = []
                    for hour_key, data in hourly_data.items():
                        if data["total_count"] > 0:
                            tag_pct = (data["tag_count"] / data["total_count"]) * 100
                            lwbs_rate_hour = data["lwbs_count"] / data["total_count"]
                            tag_pcts.append(tag_pct)
                            lwbs_rates.append(lwbs_rate_hour)
                    
                    # Calculate Pearson correlation if we have enough data points
                    if len(tag_pcts) >= 5 and len(lwbs_rates) >= 5:
                        try:
                            # Use Polars for correlation
                            df = pl.DataFrame({
                                "tag_pct": tag_pcts,
                                "lwbs_rate": lwbs_rates
                            })
                            correlation = df.select(pl.corr("tag_pct", "lwbs_rate")).item()
                            
                            # Calculate confidence interval using Fisher transformation
                            if correlation is not None and math.isfinite(correlation) and len(tag_pcts) > 30:
                                n = len(tag_pcts)
                                # Fisher z-transformation for CI
                                z = 1.96  # 95% confidence
                                z_corr = 0.5 * math.log((1 + correlation) / (1 - correlation)) if abs(correlation) < 0.99 else 0
                                se = 1.0 / math.sqrt(n - 3)
                                z_lower = z_corr - z * se
                                z_upper = z_corr + z * se
                                # Transform back
                                correlation_ci = [
                                    max(-1.0, (math.exp(2 * z_lower) - 1) / (math.exp(2 * z_lower) + 1)),
                                    min(1.0, (math.exp(2 * z_upper) - 1) / (math.exp(2 * z_upper) + 1))
                                ]
                        except Exception as e:
                            logger.debug(f"Polars correlation calculation failed: {e}, using fallback")
                            # Fallback to numpy
                            import numpy as np
                            correlation = float(np.corrcoef(tag_pcts, lwbs_rates)[0, 1]) if len(tag_pcts) == len(lwbs_rates) else lwbs_rate
                except ImportError:
                    # Fallback to numpy if Polars not available
                    try:
                        import numpy as np
                        if len(tag_pcts) >= 5 and len(lwbs_rates) >= 5 and len(tag_pcts) == len(lwbs_rates):
                            correlation = float(np.corrcoef(tag_pcts, lwbs_rates)[0, 1])
                        else:
                            correlation = lwbs_rate  # Simplified fallback
                    except (ImportError, ValueError, TypeError) as e:
                        logger.debug(f"NumPy correlation calculation failed: {e}, using rate as proxy")
                        correlation = lwbs_rate
                except Exception as e:
                    logger.warning(f"Error calculating correlation: {e}, using rate as proxy")
                    correlation = lwbs_rate
                
                # Validate correlation is finite
                if correlation is not None and not math.isfinite(correlation):
                    correlation = lwbs_rate  # Fallback to rate
                
                # Interpret correlation strength
                if correlation is not None and math.isfinite(correlation):
                    abs_corr = abs(correlation)
                    if abs_corr > 0.7:
                        interpretation = f"Strong {'positive' if correlation > 0 else 'negative'} correlation: {tag} surge → {lwbs_rate*100:.1f}% LWBS on {period} (r={correlation:.2f})"
                    elif abs_corr > 0.5:
                        interpretation = f"Moderate-strong {'positive' if correlation > 0 else 'negative'} correlation: {tag} surge → {lwbs_rate*100:.1f}% LWBS on {period} (r={correlation:.2f})"
                    elif abs_corr > 0.3:
                        interpretation = f"Moderate {'positive' if correlation > 0 else 'negative'} correlation: {tag} surge → {lwbs_rate*100:.1f}% LWBS on {period} (r={correlation:.2f})"
                    else:
                        interpretation = f"Weak {'positive' if correlation > 0 else 'negative'} correlation: {tag} → {lwbs_rate*100:.1f}% LWBS on {period} (r={correlation:.2f})"
            
            summary[tag] = {
                "arrivals": total,
                "lwbs_rate": lwbs_rate,
                "period": period,
                "correlation_coefficient": correlation if correlation is not None and math.isfinite(correlation) else None,
                "correlation_ci_95": correlation_ci,
                "interpretation": interpretation,
                "data_points": total,
                "time_series_points": len(time_series_data),
                "statistical_significance": "high" if total > 30 and correlation and abs(correlation) > 0.3 else "moderate" if total > 10 else "low",
                "note": "Live data from events - calculated with Polars/numpy correlation"
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

