from fastapi import APIRouter
from app.api.endpoints import chat, upload

api_router = APIRouter()
api_router.include_router(chat.router, prefix="/api", tags=["chat"])
api_router.include_router(upload.router, prefix="/api", tags=["upload"])
