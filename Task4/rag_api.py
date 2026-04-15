"""
FastAPI REST API для RAG-бота.
Endpoints для поиска в базе знаний и генерации ответов.
"""
import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Task4.rag_bot import (
    search_chunks,
    build_prompt,
    format_few_shot,
    call_llm,
    rag_query,
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
    check_ollama,
    check_groq,
    check_openrouter,
    LLM_BACKEND,
)

# ─── Config ───
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-api")


# ─── Lifespan: load models on startup ───
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load embedding model and ChromaDB on startup."""
    logger.info("Loading embedding model and ChromaDB index...")
    # Already loaded at module import, but log the info
    from Task4.rag_bot import collection, embedding_model
    logger.info("Index contains %s chunks", collection.count())
    logger.info("LLM backend: %s", LLM_BACKEND)
    yield
    logger.info("Shutting down RAG API...")


app = FastAPI(
    title="RAG Bot API",
    description="Knowledge base Q&A API with Few-shot + Chain-of-Thought",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic models ───
class QueryRequest(BaseModel):
    question: str = Field(..., description="User question", min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    backend: Optional[str] = Field(
        default=None,
        description="LLM backend: ollama, groq, openrouter, or demo. Default: auto-detect",
    )
    include_reasoning: bool = Field(
        default=True,
        description="Include Chain-of-Thought reasoning in response",
    )


class ChunkInfo(BaseModel):
    chunk_id: str
    source_file: str
    chunk_index: int
    similarity: float
    text_preview: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    reasoning: Optional[str] = None
    chunks: list[ChunkInfo]
    total_time: float
    backend: str


class HealthResponse(BaseModel):
    status: str
    index_chunks: int
    llm_backend: str
    ollama_available: bool
    groq_available: bool
    openrouter_available: bool


# ─── Endpoints ───
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and available backends."""
    from Task4.rag_bot import collection
    return HealthResponse(
        status="ok",
        index_chunks=collection.count(),
        llm_backend=LLM_BACKEND,
        ollama_available=check_ollama(),
        groq_available=check_groq(),
        openrouter_available=check_openrouter(),
    )


@app.post("/query", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Ask a question to the knowledge base.
    Returns answer with Chain-of-Thought reasoning and sources.
    """
    start_time = time.time()

    try:
        # Determine backend
        backend = request.backend
        if not backend:
            # Auto-detect
            if check_ollama():
                backend = "ollama"
            elif check_groq():
                backend = "groq"
            elif check_openrouter():
                backend = "openrouter"
            else:
                backend = "demo"

        # Search chunks
        chunks = search_chunks(request.question, top_k=request.top_k)

        if not chunks:
            raise HTTPException(status_code=404, detail="No relevant chunks found")

        # Build prompt
        few_shot_text = format_few_shot(FEW_SHOT_EXAMPLES)
        prompt = build_prompt(request.question, chunks, few_shot_text)

        # Call LLM
        answer = call_llm(SYSTEM_PROMPT, prompt, backend=backend)

        elapsed = time.time() - start_time

        # Parse reasoning if present
        reasoning = None
        final_answer = answer
        if request.include_reasoning:
            if "REASONING:" in answer and "ANSWER:" in answer:
                parts = answer.split("ANSWER:", 1)
                reasoning = parts[0].replace("REASONING:", "").strip()
                final_answer = parts[1].strip()
            elif "РАЗМЫШЛЕНИЕ:" in answer and "ОТВЕТ:" in answer:
                parts = answer.split("ОТВЕТ:", 1)
                reasoning = parts[0].replace("РАЗМЫШЛЕНИЕ:", "").strip()
                final_answer = parts[1].strip()

        return QueryResponse(
            question=request.question,
            answer=final_answer,
            reasoning=reasoning,
            chunks=[
                ChunkInfo(
                    chunk_id=c["chunk_id"],
                    source_file=c["source_file"],
                    chunk_index=c["chunk_index"],
                    similarity=c["similarity"],
                    text_preview=c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"],
                )
                for c in chunks[:3]
            ],
            total_time=round(elapsed, 2),
            backend=backend,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing query: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/search")
async def search_knowledge_base(q: str, top_k: int = 5):
    """
    Search the knowledge base without LLM generation.
    Returns matching chunks only.
    """
    chunks = search_chunks(q, top_k=top_k)
    return {
        "query": q,
        "results_count": len(chunks),
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "source_file": c["source_file"],
                "chunk_index": c["chunk_index"],
                "similarity": c["similarity"],
                "text": c["text"][:500] + "..." if len(c["text"]) > 500 else c["text"],
            }
            for c in chunks
        ],
    }


# ─── Run with uvicorn ───
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "rag_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
