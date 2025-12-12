"""
OpenAI-powered bottleneck analysis.
Generates detailed explanations of where bottlenecks occurred, why they happened,
and their first and second order effects.
"""
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None


class BottleneckAIAnalyzer:
    """Uses OpenAI to generate comprehensive bottleneck analysis."""
    
    def __init__(self):
        if OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self.model = settings.OPENAI_MODEL
        else:
            self.client = None
            self.model = None
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI package not installed - AI analysis disabled")
            elif not settings.OPENAI_API_KEY:
                logger.warning("OpenAI API key not set - AI analysis disabled")
    
    async def analyze_bottleneck(
        self,
        bottleneck: Dict[str, Any],
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Generate comprehensive AI analysis of a bottleneck.
        
        Returns:
            {
                "where": "Detailed location description",
                "why": "Root cause analysis",
                "first_order_effects": ["Effect 1", "Effect 2", ...],
                "second_order_effects": ["Effect 1", "Effect 2", ...]
            }
        """
        if not self.client:
            return self._fallback_analysis(bottleneck, events, kpis)
        
        try:
            # Prepare context for OpenAI
            context = self._prepare_context(bottleneck, events, kpis, window_hours)
            
            # Generate analysis
            analysis = await self._generate_analysis(context, bottleneck)
            
            return analysis
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}", exc_info=True)
            return self._fallback_analysis(bottleneck, events, kpis)
    
    def _prepare_context(
        self,
        bottleneck: Dict[str, Any],
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]],
        window_hours: int
    ) -> Dict[str, Any]:
        """Prepare context data for OpenAI analysis."""
        # Extract relevant metrics
        stage = bottleneck.get("stage", "unknown")
        wait_time = bottleneck.get("current_wait_time_minutes", 0)
        severity = bottleneck.get("severity", "medium")
        impact_score = bottleneck.get("impact_score", 0.5)
        
        # Filter events for this stage
        stage_events = [
            e for e in events
            if e.get("stage") == stage or e.get("event_type") == stage
        ]
        
        # Calculate stage-specific metrics
        stage_metrics = {
            "total_events": len(stage_events),
            "avg_duration": 0.0,
            "peak_hours": [],
            "resource_utilization": {}
        }
        
        if stage_events:
            durations = [
                e.get("duration_minutes", 0)
                for e in stage_events
                if e.get("duration_minutes") and e.get("duration_minutes") > 0
            ]
            if durations:
                stage_metrics["avg_duration"] = sum(durations) / len(durations)
            
            # Find peak hours
            hourly_counts = {}
            for event in stage_events:
                timestamp = event.get("timestamp")
                if timestamp:
                    if isinstance(timestamp, str):
                        from datetime import datetime
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    hour = timestamp.hour
                    hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
            
            if hourly_counts:
                peak_hour = max(hourly_counts.items(), key=lambda x: x[1])[0]
                stage_metrics["peak_hours"] = [peak_hour]
            
            # Resource utilization
            resource_counts = {}
            for event in stage_events:
                resource_type = event.get("resource_type")
                if resource_type:
                    resource_counts[resource_type] = resource_counts.get(resource_type, 0) + 1
            stage_metrics["resource_utilization"] = resource_counts
        
        # Extract KPI trends
        kpi_trends = {
            "dtd": [k.get("dtd", 0) for k in kpis if k.get("dtd")],
            "los": [k.get("los", 0) for k in kpis if k.get("los")],
            "lwbs": [k.get("lwbs", 0) for k in kpis if k.get("lwbs")],
            "bed_utilization": [k.get("bed_utilization", 0) for k in kpis if k.get("bed_utilization")]
        }
        
        # Calculate trends
        trends = {}
        for metric, values in kpi_trends.items():
            if len(values) >= 2:
                recent_avg = sum(values[-3:]) / min(3, len(values))
                earlier_avg = sum(values[:3]) / min(3, len(values))
                if earlier_avg > 0:
                    trends[metric] = ((recent_avg - earlier_avg) / earlier_avg) * 100
        
        return {
            "bottleneck": {
                "name": bottleneck.get("bottleneck_name", "Unknown"),
                "stage": stage,
                "wait_time_minutes": wait_time,
                "severity": severity,
                "impact_score": impact_score,
                "causes": bottleneck.get("causes", [])
            },
            "stage_metrics": stage_metrics,
            "kpi_trends": trends,
            "window_hours": window_hours,
            "total_events": len(events),
            "total_kpis": len(kpis)
        }
    
    async def _generate_analysis(
        self,
        context: Dict[str, Any],
        bottleneck: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate analysis using OpenAI."""
        prompt = self._build_prompt(context, bottleneck)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert Emergency Department operations analyst with deep knowledge of healthcare operations, patient flow, and bottleneck analysis. Your task is to analyze ED bottlenecks and provide detailed, actionable insights."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            try:
                analysis = json.loads(content)
            except json.JSONDecodeError:
                # If not JSON, try to extract structured info
                analysis = self._parse_text_response(content)
            
            return analysis
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return self._fallback_analysis(bottleneck, {}, [])
    
    def _build_prompt(
        self,
        context: Dict[str, Any],
        bottleneck: Dict[str, Any]
    ) -> str:
        """Build the prompt for OpenAI."""
        bn = context["bottleneck"]
        stage_metrics = context["stage_metrics"]
        trends = context["kpi_trends"]
        
        prompt = f"""Analyze this Emergency Department bottleneck and provide a comprehensive analysis in JSON format.

BOTTLENECK DETAILS:
- Name: {bn['name']}
- Stage: {bn['stage']}
- Current Wait Time: {bn['wait_time_minutes']:.1f} minutes
- Severity: {bn['severity']}
- Impact Score: {bn['impact_score']:.2f}
- Identified Causes: {', '.join(bn['causes']) if bn['causes'] else 'None specified'}

STAGE METRICS:
- Total Events: {stage_metrics['total_events']}
- Average Duration: {stage_metrics['avg_duration']:.1f} minutes
- Peak Hours: {stage_metrics['peak_hours']}
- Resource Utilization: {stage_metrics['resource_utilization']}

KPI TRENDS (percentage change):
{json.dumps(trends, indent=2)}

Please provide a JSON response with the following structure:
{{
    "where": "Detailed description of WHERE this bottleneck is occurring, including specific location, time patterns, and affected resources. Be specific about the physical location and operational context.",
    "why": "Comprehensive explanation of WHY this bottleneck is happening. Analyze root causes, contributing factors, resource constraints, and systemic issues. Be specific and data-driven.",
    "first_order_effects": [
        "Direct immediate impact 1",
        "Direct immediate impact 2",
        "Direct immediate impact 3"
    ],
    "second_order_effects": [
        "Secondary downstream impact 1",
        "Secondary downstream impact 2",
        "Secondary downstream impact 3"
    ]
}}

Guidelines:
- "where": Be specific about location, timing, and affected areas
- "why": Provide root cause analysis based on the data provided
- "first_order_effects": List 3-5 direct, immediate consequences
- "second_order_effects": List 3-5 downstream, cascading effects

Return ONLY valid JSON, no additional text."""
        
        return prompt
    
    def _parse_text_response(self, text: str) -> Dict[str, Any]:
        """Parse text response if JSON parsing fails."""
        # Try to extract JSON from text
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        # Fallback: extract structured info manually
        return {
            "where": self._extract_section(text, "where", "WHERE"),
            "why": self._extract_section(text, "why", "WHY"),
            "first_order_effects": self._extract_list(text, "first_order", "first order"),
            "second_order_effects": self._extract_list(text, "second_order", "second order")
        }
    
    def _extract_section(self, text: str, key: str, label: str) -> str:
        """Extract a section from text response."""
        import re
        patterns = [
            rf'"{key}":\s*"([^"]+)"',
            rf'{label}[:]\s*(.+?)(?:\n\n|\n[A-Z]|$)',
            rf'{label}\s*[:]\s*(.+?)(?:\n\n|\n[A-Z]|$)'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return f"Analysis for {key} unavailable"
    
    def _extract_list(self, text: str, key: str, label: str) -> List[str]:
        """Extract a list from text response."""
        import re
        patterns = [
            rf'"{key}_effects":\s*\[(.*?)\]',
            rf'{label}[_\s]*effects?[:]\s*(.+?)(?:\n\n|\n[A-Z]|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1)
                # Extract bullet points or list items
                items = re.findall(r'["\']([^"\']+)["\']|[-•]\s*(.+?)(?:\n|$)', content)
                if items:
                    return [item[0] or item[1] for item in items if item[0] or item[1]]
        return [f"Effect analysis for {key} unavailable"]
    
    def _fallback_analysis(
        self,
        bottleneck: Dict[str, Any],
        events: List[Dict[str, Any]],
        kpis: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate fallback analysis without OpenAI."""
        stage = bottleneck.get("stage", "unknown")
        wait_time = bottleneck.get("current_wait_time_minutes", 0)
        causes = bottleneck.get("causes", [])
        
        where = f"The bottleneck is occurring at the {stage} stage of the Emergency Department. Current wait time is {wait_time:.1f} minutes."
        
        why = "Root causes: " + "; ".join(causes) if causes else "Resource constraints and operational inefficiencies."
        
        first_order = [
            f"Patients waiting {wait_time:.0f} minutes at {stage} stage",
            "Increased patient dissatisfaction and complaints",
            "Delayed treatment initiation for affected patients"
        ]
        
        second_order = [
            "Downstream delays in subsequent ED stages",
            "Potential increase in Left Without Being Seen (LWBS) rates",
            "Impact on overall ED throughput and capacity"
        ]
        
        return {
            "where": where,
            "why": why,
            "first_order_effects": first_order,
            "second_order_effects": second_order
        }
