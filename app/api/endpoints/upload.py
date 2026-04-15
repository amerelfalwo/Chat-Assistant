from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import JSONResponse
from typing import List
from pathlib import Path
import os
import asyncio
import aiofiles
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.vectorstore import get_vectorstore
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = "./upload_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Shared splitter (stateless, safe to reuse) ──────────────────────────
_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)


def _process_and_upsert(save_path: Path, session_id: str, filename: str) -> int:
    """
    Synchronous CPU-bound work: load PDF → chunk → embed → upsert.
    Designed to be called via asyncio.to_thread() so the event loop stays free.
    """
    vectorstore = get_vectorstore(namespace=session_id)

    loader = PyPDFLoader(str(save_path))
    documents = loader.load()

    chunks = _splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["id"] = f"{save_path.stem}-{idx}"
        chunk.metadata["source"] = filename

    logger.info(f"Uploading {len(chunks)} chunks to VectorStore")
    vectorstore.add_documents(chunks)

    return len(chunks)


@router.post("/upload")
async def upload_pdfs(
    session_id: str = Query(..., description="Session ID"),
    file: UploadFile = File(..., description="Select PDF file"),
):
    save_path = None
    try:
        logger.info(f"Received file {file.filename} for session {session_id}")

        save_path = Path(UPLOAD_DIR) / file.filename

        async with aiofiles.open(save_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        logger.info(f"Processing {file.filename} in background thread...")

        # Offload all CPU-bound work to a separate thread
        num_chunks = await asyncio.to_thread(
            _process_and_upsert, save_path, session_id, file.filename
        )

        logger.info("Document successfully added to vectorstore")
        return {
            "message": "File processed and vectorstore updated.",
            "session_id": session_id,
            "chunks_processed": num_chunks,
        }

    except Exception as e:
        logger.exception("Error during PDF upload")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # Clean up temp file after processing
        if save_path and save_path.exists():
            os.remove(save_path)


@router.post("/upload-multiple")
async def upload_multiple_pdfs(
    session_id: str = Query(..., description="Session ID"),
    files: List[UploadFile] = File(..., description="Select PDF files"),
):
    saved_paths = []
    total_chunks = 0
    try:
        logger.info(f"Received {len(files)} files for session {session_id}")

        for file in files:
            save_path = Path(UPLOAD_DIR) / file.filename
            saved_paths.append(save_path)

            async with aiofiles.open(save_path, "wb") as f:
                content = await file.read()
                await f.write(content)

            logger.info(f"Processing {file.filename} in background thread...")

            # Offload each file's processing to a separate thread
            num_chunks = await asyncio.to_thread(
                _process_and_upsert, save_path, session_id, file.filename
            )
            total_chunks += num_chunks

        logger.info("Documents successfully added to vectorstore")
        return {
            "message": "Files processed and vectorstore updated.",
            "session_id": session_id,
            "total_chunks_processed": total_chunks,
        }

    except Exception as e:
        logger.exception("Error during PDF upload")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # Clean up temp files after processing
        for p in saved_paths:
            if p.exists():
                os.remove(p)