"""
Chat endpoint - ChatGPT-like conversational interface.
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
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


@router.post("/chat/stream")
async def chat_stream(
    query: NLPQuery,
    conversation_id: str = Query("default", description="Conversation session ID")
):
    """
    Streaming chat endpoint - streams response as it's generated.
    
    Returns Server-Sent Events (SSE) format:
    - data: {"type": "results", "data": {...}} - Initial results
    - data: {"type": "chunk", "content": "..."} - Text chunks
    - data: {"type": "done"} - Completion signal
    - data: {"type": "error", "message": "..."} - Error signal
    """
    async def generate():
        try:
            async for chunk in conversational_ai.chat_stream(
                query=query.query,
                conversation_id=conversation_id
            ):
                yield chunk
        except Exception as e:
            import json
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.post("/chat/clear")
async def clear_conversation(conversation_id: str = Query("default")):
    """Clear conversation history."""
    try:
        conversational_ai.conversation_manager.clear_conversation(conversation_id)
        return {"status": "ok", "message": "Conversation cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear conversation: {str(e)}")


