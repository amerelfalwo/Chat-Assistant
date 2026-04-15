from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.services.rag_pipeline import get_conversational_rag
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Keep rag chains cached in memory per session
rag_chains = {}

def get_chain(session_id: str):
    if session_id not in rag_chains:
        rag_chains[session_id] = get_conversational_rag(session_id)
    return rag_chains[session_id]

class ChatRequest(BaseModel):
    session_id: str
    question: str

@router.post("/ask")
async def ask_question(request: ChatRequest):
    try:
        logger.info(f"User query: {request.question} in session: {request.session_id}")
        
        chain = get_chain(request.session_id)
        
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
