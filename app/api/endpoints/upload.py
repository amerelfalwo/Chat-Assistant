from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
from pathlib import Path
import os
import aiofiles
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.vectorstore import get_vectorstore
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = "./upload_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    try:
        logger.info(f"Received {len(files)} uploaded files")
        vectorstore = get_vectorstore()
        
        for file in files:
            save_path = Path(UPLOAD_DIR) / file.filename
            
            # Save file asynchronously
            async with aiofiles.open(save_path, "wb") as f:
                content = await file.read()
                await f.write(content)
                
            logger.info(f"Processing and loading {file.filename}")
            
            # Load and chunk pdf
            loader = PyPDFLoader(str(save_path))
            documents = loader.load()
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(documents)
            
            # Add metadata if needed to track source
            for idx, chunk in enumerate(chunks):
                chunk.metadata["id"] = f"{save_path.stem}-{idx}"
                chunk.metadata["source"] = file.filename
                
            # Upsert directly to vectorstore
            logger.info(f"Uploading {len(chunks)} chunks to VectorStore")
            vectorstore.add_documents(chunks)

        logger.info("Documents successfully added to vectorstore")
        return {"message": "Files processed and vectorstore updated."}

    except Exception as e:
        logger.exception("Error during PDF upload")
        return JSONResponse(status_code=500, content={"error": str(e)})
