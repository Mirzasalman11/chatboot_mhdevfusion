"""
RAG Engine for MHDEVFUSION Chatbot
- Embeds all Q&As once at startup using text-embedding-3-small
- On each query: embed → cosine similarity → retrieve top-k → gpt-4o-mini
"""

import os
import logging
import numpy as np
from openai import AsyncOpenAI
from knowledge_base import KNOWLEDGE_BASE

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"   # cheapest + accurate
CHAT_MODEL      = "gpt-4o-mini"
TOP_K           = 3                          # retrieve top-3 Q&As per query
SIMILARITY_THRESHOLD = 0.30                 # ignore very low-relevance results

SYSTEM_PROMPT = """You are a helpful and friendly assistant for MHDEVFUSION, 
a full-service digital agency. Your job is to answer questions accurately using 
ONLY the provided context from the knowledge base.

Rules:
- Answer ONLY from the provided context. Do not invent services, prices, or features.
- Be concise, friendly, and professional.
- If the context contains a CTA (call-to-action), include it naturally at the end.
- If the question is not covered by the context, say:
  "I don't have that information right now. Please reach out to us at 
   mhdevfusion@gmail.com or book a free call at mhdevfusion.com — our team 
   will be happy to help!"
- Never make up pricing or feature details.
"""


class RAGEngine:
    """
    Manages the full RAG pipeline:
      1. Embed all KB entries at startup (cached in memory)
      2. Embed incoming user queries
      3. Cosine similarity retrieval
      4. GPT-4o-mini generation with retrieved context
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.kb = KNOWLEDGE_BASE
        # Will be populated by build_index()
        self._embeddings: np.ndarray | None = None   # shape: (N, 1536)
        self._index_ready = False

    # ── Index Building ─────────────────────────────────────────────────────────

    async def build_index(self) -> None:
        """
        Called once at FastAPI startup.
        Embeds all knowledge base questions and caches the vectors.
        """
        logger.info("Building RAG index for %d Q&A entries …", len(self.kb))
        texts = [item["question"] for item in self.kb]
        embeddings = await self._embed_batch(texts)
        self._embeddings = np.array(embeddings, dtype="float32")
        self._index_ready = True
        logger.info("RAG index ready. Embedding matrix shape: %s", self._embeddings.shape)

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in a single API call (more efficient)."""
        response = await self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        # Sort by index to guarantee order (OpenAI may reorder in batch)
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    async def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single user query."""
        response = await self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query],
        )
        return np.array(response.data[0].embedding, dtype="float32")

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def _cosine_similarity(self, query_vec: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query_vec and all KB embeddings."""
        # Both are already float32; no need for explicit casting
        kb_norm   = self._embeddings / (np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10)
        q_norm    = query_vec        / (np.linalg.norm(query_vec)                                + 1e-10)
        return kb_norm @ q_norm   # shape: (N,)

    def _retrieve(self, query_vec: np.ndarray, top_k: int = TOP_K) -> list[dict]:
        """Return top-k most relevant KB entries above the similarity threshold."""
        scores = self._cosine_similarity(query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= SIMILARITY_THRESHOLD:
                results.append({**self.kb[idx], "score": round(score, 4)})
        return results

    # ── Generation ─────────────────────────────────────────────────────────────

    def _build_context(self, retrieved: list[dict]) -> str:
        """Format retrieved Q&As into a context block for the system prompt."""
        if not retrieved:
            return "No relevant information found in the knowledge base."
        lines = []
        for i, item in enumerate(retrieved, 1):
            cta = f"\n   CTA: {item['cta']}" if item.get("cta") else ""
            lines.append(
                f"[{i}] Category: {item['category']}\n"
                f"   Q: {item['question']}\n"
                f"   A: {item['answer']}{cta}"
            )
        return "\n\n".join(lines)

    async def chat(
        self,
        user_message: str,
        history: list[dict] | None = None,
    ) -> dict:
        """
        Full RAG pipeline for a single user turn.

        Args:
            user_message: The user's latest message.
            history:      Previous conversation turns as [{"role": ..., "content": ...}].

        Returns:
            {
              "answer": str,
              "sources": [{"category": str, "question": str, "score": float}],
            }
        """
        if not self._index_ready:
            raise RuntimeError("RAG index not built yet. Call build_index() first.")

        # 1. Embed query
        query_vec = await self._embed_query(user_message)

        # 2. Retrieve relevant Q&As
        retrieved = self._retrieve(query_vec)

        # 3. Build context
        context = self._build_context(retrieved)

        # 4. Construct messages
        messages = [
            {
                "role": "system",
                "content": (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"=== KNOWLEDGE BASE CONTEXT ===\n{context}\n"
                    f"==============================\n"
                    f"Answer the user's question using ONLY the above context."
                ),
            }
        ]
        # Include conversation history (last 10 turns to save tokens)
        if history:
            messages.extend(history[-10:])

        messages.append({"role": "user", "content": user_message})

        # 5. Generate response
        response = await self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.3,    # low temp for factual accuracy
            max_tokens=512,
        )

        answer = response.choices[0].message.content.strip()

        # 6. Return answer + sources (for debug/transparency)
        sources = [
            {
                "category": r["category"],
                "question": r["question"],
                "score":    r["score"],
            }
            for r in retrieved
        ]

        logger.info(
            "Query: %r | Retrieved: %d sources | Top score: %.4f",
            user_message[:80],
            len(retrieved),
            sources[0]["score"] if sources else 0,
        )

        return {"answer": answer, "sources": sources}
