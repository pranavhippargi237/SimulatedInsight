"""
NLP parsing layer for natural language queries.
"""
import logging
import json
from typing import Dict, Any, Optional
import openai
from app.data.schemas import ParsedScenario, ScenarioChange, NLPQuery
from app.core.config import settings

logger = logging.getLogger(__name__)


class NLParser:
    """Natural language parser for ED scenario queries."""
    
    def __init__(self):
        if settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
    
    async def parse_query(self, query: NLPQuery) -> ParsedScenario:
        """
        Parse natural language query into structured scenario.
        
        Args:
            query: Natural language query
            
        Returns:
            Parsed scenario with confidence score
        """
        if not self.client:
            # Fallback to rule-based parsing
            return await self._rule_based_parse(query.query)
        
        try:
            # Use OpenAI to parse query
            prompt = self._build_parsing_prompt(query.query)
            
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a parser that converts natural language ED operations queries into structured JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Extract scenario
            scenario_dict = result.get("scenario", {})
            scenario = ScenarioChange(**scenario_dict)
            
            confidence = result.get("confidence", 0.8)
            suggestions = result.get("suggestions", [])
            
            return ParsedScenario(
                scenario=scenario,
                confidence=confidence,
                original_query=query.query,
                suggestions=suggestions
            )
        
        except Exception as e:
            logger.warning(f"OpenAI parsing failed: {e}, falling back to rule-based")
            return await self._rule_based_parse(query.query)
    
    def _build_parsing_prompt(self, query: str) -> str:
        """Build prompt for OpenAI parsing."""
        schema_example = {
            "action": "add",
            "resource_type": "nurse",
            "quantity": 2,
            "time_start": "14:00",
            "time_end": "18:00",
            "day": "Saturday"
        }
        
        return f"""
Parse this ED operations query into a structured JSON scenario:

Query: "{query}"

Output JSON schema:
{{
    "scenario": {{
        "action": "add|remove|shift|modify",
        "resource_type": "nurse|doctor|bed|tech",
        "quantity": <number>,
        "time_start": "HH:MM" or null,
        "time_end": "HH:MM" or null,
        "day": "day name" or null,
        "department": "department name" or null
    }},
    "confidence": <0.0-1.0>,
    "suggestions": ["alternative interpretations if any"]
}}

Example output:
{{
    "scenario": {schema_example},
    "confidence": 0.9,
    "suggestions": []
}}

Parse the query now:
"""
    
    async def _rule_based_parse(self, query: str) -> ParsedScenario:
        """Fallback rule-based parser."""
        import re
        
        query_lower = query.lower()
        
        # Extract action
        action = "add"
        if "remove" in query_lower or "reduce" in query_lower:
            action = "remove"
        elif "shift" in query_lower:
            action = "shift"
        elif "modify" in query_lower or "change" in query_lower:
            action = "modify"
        
        # Extract resource type
        resource_type = "nurse"
        if "doctor" in query_lower or "physician" in query_lower:
            resource_type = "doctor"
        elif "bed" in query_lower:
            resource_type = "bed"
        elif "lab tech" in query_lower or "lab technician" in query_lower or ("lab" in query_lower and "tech" in query_lower):
            resource_type = "lab_tech"  # Lab techs (separate from imaging)
        elif "imaging tech" in query_lower or "imaging technician" in query_lower or ("imaging" in query_lower and "tech" in query_lower):
            resource_type = "imaging_tech"  # Explicit imaging tech
        elif "tech" in query_lower or "technician" in query_lower:
            resource_type = "tech"  # Default: imaging tech
        elif "nurse" in query_lower:
            resource_type = "nurse"
        
        # Extract quantity
        quantity = 1
        numbers = re.findall(r'\d+', query)
        if numbers:
            quantity = int(numbers[0])
        
        # Extract time
        time_start = None
        time_end = None
        time_pattern = r'(\d{1,2}):(\d{2})'
        times = re.findall(time_pattern, query)
        if len(times) >= 1:
            time_start = f"{times[0][0]}:{times[0][1]}"
        if len(times) >= 2:
            time_end = f"{times[1][0]}:{times[1][1]}"
        
        # Extract day
        day = None
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for d in days:
            if d in query_lower:
                day = d.capitalize()
                break
        
        scenario = ScenarioChange(
            action=action,
            resource_type=resource_type,
            quantity=quantity,
            time_start=time_start,
            time_end=time_end,
            day=day
        )
        
        return ParsedScenario(
            scenario=scenario,
            confidence=0.7,  # Lower confidence for rule-based
            original_query=query,
            suggestions=["Consider using more specific language for better parsing"]
        )

