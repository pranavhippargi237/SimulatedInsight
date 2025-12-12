"""
Conversation Manager: Maintains context across multiple turns of conversation.
Enables ChatGPT-like natural language interaction with follow-up questions.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import openai
from app.core.config import settings

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation context and enables natural multi-turn interactions.
    
    Maintains:
    - Conversation history
    - Context from previous queries
    - User preferences
    - Current state (what was last discussed)
    """
    
    def __init__(self):
        if settings.OPENAI_API_KEY:
            self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        else:
            self.client = None
            logger.warning("OpenAI API key not set - conversation context disabled")
        
        # In-memory conversation storage (using SQLite for persistence)
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
    
    def get_conversation_id(self, user_id: Optional[str] = None) -> str:
        """Generate or retrieve conversation ID."""
        # For MVP, use a simple session-based approach
        # In production, use proper session management
        return user_id or "default_session"
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to conversation history."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        self.conversations[conversation_id].append(message)
        
        # Keep last 20 messages to avoid context bloat
        if len(self.conversations[conversation_id]) > 20:
            self.conversations[conversation_id] = self.conversations[conversation_id][-20:]
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversations.get(conversation_id, [])
    
    async def process_with_context(
        self,
        query: str,
        conversation_id: str,
        system_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process query with full conversation context.
        
        Uses OpenAI to:
        1. Understand the query in context of previous messages
        2. Determine if it's a follow-up question
        3. Extract what the user wants to do
        4. Generate appropriate actions
        """
        if not self.client:
            return await self._fallback_process(query, conversation_id)
        
        # Get conversation history
        history = self.get_conversation_history(conversation_id)
        
        # Build system prompt with context
        system_prompt = self._build_system_prompt(system_context)
        
        # Build messages for OpenAI
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 10 messages for context)
        for msg in history[-10:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        try:
            # Use OpenAI to understand intent and generate structured response
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"},
                functions=self._get_functions(),
                function_call="auto"
            )
            
            content = response.choices[0].message.content
            if content:
                try:
                    result = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    # If not valid JSON, treat as plain text response
                    logger.warning("OpenAI response was not valid JSON, using fallback")
                    return await self._fallback_process(query, conversation_id)
            else:
                logger.warning("OpenAI returned empty response")
                return await self._fallback_process(query, conversation_id)
            
            # Add user message to history
            self.add_message(conversation_id, "user", query)
            
            # Add assistant response to history
            assistant_response = result.get("response", "")
            self.add_message(conversation_id, "assistant", assistant_response, result)
            
            return result
        
        except Exception as e:
            logger.error(f"Context processing failed: {e}", exc_info=True)
            return await self._fallback_process(query, conversation_id)
    
    def _build_system_prompt(self, system_context: Optional[Dict[str, Any]]) -> str:
        """Build system prompt with ED operations context."""
        return """You are an intelligent assistant for an Emergency Department operations system. 
You help hospital directors and ED managers optimize their operations through natural conversation.

Your capabilities:
1. **Simulation**: Run "what-if" scenarios (e.g., "What if we add 2 nurses?")
2. **Planning**: Generate action plans and recommendations (e.g., "What should I do?")
3. **Analysis**: Detect bottlenecks and analyze current state
4. **Explanation**: Explain results and answer questions
5. **Comparison**: Compare different scenarios

You can handle:
- Direct questions: "What if we add 2 nurses?"
- Follow-up questions: "What about on weekends?" (referring to previous query)
- Clarifications: "Can you explain why DTD increased?"
- Multi-part requests: "What if we add nurses and what should I do?"

Resources you can work with:
- Nurses (triage_nurses)
- Doctors
- Beds
- Lab techs (lab_techs)
- Imaging techs (imaging_techs, tech)

Always be:
- Conversational and natural
- Helpful and actionable
- Clear about what you're doing
- Ready to answer follow-up questions

When responding, output JSON with:
{
    "intent": "simulation|plan|both|analysis|comparison|explanation|follow_up",
    "is_follow_up": true/false,
    "follow_up_context": "what the follow-up refers to",
    "actions": ["simulate", "generate_plan", etc.],
    "parameters": {
        "action": "add|remove|shift|modify",
        "resource_type": "nurse|doctor|bed|lab_tech|imaging_tech",
        "quantity": number,
        "time_start": "HH:MM" or null,
        "time_end": "HH:MM" or null,
        "day": "day name" or null
    },
    "response": "Natural language response explaining what you'll do",
    "confidence": 0.0-1.0
}"""
    
    def _get_functions(self) -> List[Dict[str, Any]]:
        """Define functions for structured output."""
        return [
            {
                "name": "run_simulation",
                "description": "Run a simulation scenario",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["add", "remove", "shift", "modify"]},
                        "resource_type": {"type": "string", "enum": ["nurse", "doctor", "bed", "lab_tech", "imaging_tech", "tech"]},
                        "quantity": {"type": "integer"},
                        "time_start": {"type": "string", "nullable": True},
                        "time_end": {"type": "string", "nullable": True},
                        "day": {"type": "string", "nullable": True}
                    },
                    "required": ["action", "resource_type", "quantity"]
                }
            },
            {
                "name": "generate_plan",
                "description": "Generate an action plan with recommendations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "window_hours": {"type": "integer", "default": 48},
                        "top_n": {"type": "integer", "default": 5}
                    }
                }
            },
            {
                "name": "detect_bottlenecks",
                "description": "Detect current bottlenecks",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "window_hours": {"type": "integer", "default": 48}
                    }
                }
            }
        ]
    
    async def _fallback_process(self, query: str, conversation_id: str) -> Dict[str, Any]:
        """Fallback processing without OpenAI - handles ALL queries intelligently."""
        query_lower = query.lower()
        
        # Bottleneck queries
        if any(word in query_lower for word in ["bottleneck", "bottlenecks", "major bottlenecks", "what are my bottlenecks"]):
            return {
                "intent": "analysis",
                "is_follow_up": False,
                "actions": ["detect_bottlenecks"],
                "parameters": {},
                "response": "I'll analyze your ED to identify current bottlenecks and their impact.",
                "confidence": 0.9
            }
        
        # LWBS queries
        if any(word in query_lower for word in ["lwbs", "left without", "why is lwbs", "lwbs high", "lwbs rate", "leave without"]):
            return {
                "intent": "analysis",
                "is_follow_up": False,
                "actions": ["detect_bottlenecks", "generate_plan"],
                "parameters": {},
                "response": "I'll analyze why your LWBS rate is high and provide recommendations to reduce it.",
                "confidence": 0.9
            }
        
        # DTD queries
        if any(word in query_lower for word in ["dtd", "door to doctor", "wait time", "waiting time", "why is dtd"]):
            return {
                "intent": "analysis",
                "is_follow_up": False,
                "actions": ["detect_bottlenecks", "generate_plan"],
                "parameters": {},
                "response": "I'll analyze your door-to-doctor times and identify what's causing delays.",
                "confidence": 0.9
            }
        
        # LOS queries
        if any(word in query_lower for word in ["los", "length of stay", "why is los", "los high"]):
            return {
                "intent": "analysis",
                "is_follow_up": False,
                "actions": ["detect_bottlenecks", "generate_plan"],
                "parameters": {},
                "response": "I'll analyze your length of stay and identify factors contributing to extended stays.",
                "confidence": 0.9
            }
        
        # Metrics/KPI queries
        if any(word in query_lower for word in ["metrics", "kpi", "performance", "how are we doing", "current state"]):
            return {
                "intent": "analysis",
                "is_follow_up": False,
                "actions": ["detect_bottlenecks"],
                "parameters": {},
                "response": "I'll analyze your current ED performance metrics and bottlenecks.",
                "confidence": 0.9
            }
        
        # Action plan queries
        if any(word in query_lower for word in ["what should i do", "recommend", "recommendations", "action plan", "what can i do", "help me"]):
            return {
                "intent": "plan",
                "is_follow_up": False,
                "actions": ["generate_plan"],
                "parameters": {},
                "response": "I'll analyze your ED and generate a comprehensive action plan with prioritized recommendations.",
                "confidence": 0.9
            }
        
        # Simulation queries - try to extract parameters
        if any(word in query_lower for word in ["what if", "simulate", "add", "remove", "if we"]):
            # Try to extract resource type and quantity
            resource_type = "nurse"
            quantity = 1
            
            if "doctor" in query_lower or "physician" in query_lower:
                resource_type = "doctor"
            elif "bed" in query_lower:
                resource_type = "bed"
            elif "lab tech" in query_lower or ("lab" in query_lower and "tech" in query_lower):
                resource_type = "lab_tech"
            elif "imaging tech" in query_lower or ("imaging" in query_lower and "tech" in query_lower):
                resource_type = "imaging_tech"
            elif "tech" in query_lower:
                resource_type = "tech"
            
            # Extract quantity
            import re
            numbers = re.findall(r'\d+', query)
            if numbers:
                quantity = int(numbers[0])
            
            action = "add"
            if "remove" in query_lower or "reduce" in query_lower:
                action = "remove"
            
            return {
                "intent": "simulation",
                "is_follow_up": False,
                "actions": ["simulate"],
                "parameters": {
                    "action": action,
                    "resource_type": resource_type,
                    "quantity": quantity
                },
                "response": f"I'll simulate {action}ing {quantity} {resource_type}(s) and show you the impact on your ED metrics.",
                "confidence": 0.8
            }
        
        # Explanation queries - analyze and explain
        if any(word in query_lower for word in ["why", "explain", "how", "what causes", "reason"]):
            return {
                "intent": "analysis",
                "is_follow_up": False,
                "actions": ["detect_bottlenecks", "generate_plan"],
                "parameters": {},
                "response": "I'll analyze your ED operations to explain what's happening and why.",
                "confidence": 0.8
            }
        
        # Comparison queries
        if any(word in query_lower for word in ["compare", "versus", "vs", "difference between"]):
            return {
                "intent": "comparison",
                "is_follow_up": False,
                "actions": ["simulate"],  # Will need to run multiple simulations
                "parameters": {},
                "response": "I'll help you compare different scenarios. Let me analyze the options.",
                "confidence": 0.7
            }
        
        # Default: Always do something useful - detect bottlenecks and generate plan
        return {
            "intent": "analysis",
            "is_follow_up": False,
            "actions": ["detect_bottlenecks", "generate_plan"],
            "parameters": {},
            "response": "I'll analyze your ED operations, detect bottlenecks, and provide recommendations.",
            "confidence": 0.7
        }
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]

