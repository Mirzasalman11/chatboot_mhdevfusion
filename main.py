"""
MHDEVFUSION RAG Chatbot — FastAPI Backend
Run with: uvicorn main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_engine import RAGEngine

# ── Bootstrap ──────────────────────────────────────────────────────────────────
load_dotenv()  # Load OPENAI_API_KEY from .env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global RAG engine instance
rag = RAGEngine()


# ── Lifespan (startup / shutdown) ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the embedding index once at startup."""
    logger.info("Starting MHDEVFUSION Chatbot …")
    await rag.build_index()
    logger.info("✅  Server ready at http://localhost:8000")
    yield
    logger.info("Shutting down …")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MHDEVFUSION RAG Chatbot API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production if needed
    allow_methods=["*"],
    allow_headers=["*"],
)



# ── Schemas ────────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []


class Source(BaseModel):
    category: str
    question: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


# ─------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "index_ready": rag._index_ready}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main RAG chat endpoint.

    Body:
        message  — latest user message
        history  — previous turns [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        answer   — GPT-4o-mini response grounded in KB
        sources  — retrieved KB entries used as context (for transparency)
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        history_dicts = [m.model_dump() for m in req.history]
        result = await rag.chat(req.message, history=history_dicts)
        return ChatResponse(**result)
    except Exception as exc:
        logger.exception("Chat error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
