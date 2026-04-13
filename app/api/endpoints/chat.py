from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from app.services.rag_pipeline import get_conversational_rag
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Keep rag chain cached in memory or initialize lazily
rag_chain_instance = None

def get_chain():
    global rag_chain_instance
    if rag_chain_instance is None:
        rag_chain_instance = get_conversational_rag()
    return rag_chain_instance

class ChatRequest(BaseModel):
    session_id: str
    question: str

@router.post("/ask")
async def ask_question(request: ChatRequest):
    try:
        logger.info(f"User query: {request.question} in session: {request.session_id}")
        
        chain = get_chain()
        
        response = chain.invoke(
            {"input": request.question},
            config={"configurable": {"session_id": request.session_id}}
        )
        
        answer = response.get("answer", "")
        # Safely get sources if they exist in context doc metadata
        sources = [doc.metadata for doc in response.get("context", [])]
        
        return {
            "answer": answer,
            "session_id": request.session_id,
            "sources": sources
        }

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})
