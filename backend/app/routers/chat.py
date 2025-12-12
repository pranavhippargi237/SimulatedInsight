"""
Chat endpoint - ChatGPT-like conversational interface.
"""
from fastapi import APIRouter, HTTPException, Query
from app.core.conversational_ai import ConversationalAI
from app.data.schemas import NLPQuery

router = APIRouter()
conversational_ai = ConversationalAI()


@router.post("/chat")
async def chat(
    query: NLPQuery,
    conversation_id: str = Query("default", description="Conversation session ID")
):
    """
    Main chat endpoint - ChatGPT-like natural language interface.
    
    Handles:
    - Natural language queries
    - Follow-up questions
    - Multi-turn conversations
    - Context awareness
    
    Examples:
    - "What if we add 2 nurses?"
    - "What about on weekends?" (follow-up)
    - "What should I do?"
    - "Can you explain why DTD increased?"
    
    Args:
        query: Natural language query
        conversation_id: Session ID for conversation context
        
    Returns:
        Natural language response with results
    """
    try:
        result = await conversational_ai.chat(
            query=query.query,
            conversation_id=conversation_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.post("/chat/clear")
async def clear_conversation(conversation_id: str = Query("default")):
    """Clear conversation history."""
    try:
        conversational_ai.conversation_manager.clear_conversation(conversation_id)
        return {"status": "ok", "message": "Conversation cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear conversation: {str(e)}")

