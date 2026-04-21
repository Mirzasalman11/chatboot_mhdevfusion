"""
RAG Engine for MHDEVFUSION Chatbot
- Embeds all Q&As once at startup using text-embedding-3-small
- On each query: clean → embed → cosine similarity → retrieve top-k → gpt-4o-mini
"""

import os
import logging
import numpy as np
from openai import AsyncOpenAI
from knowledge_base import KNOWLEDGE_BASE

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "text-embedding-3-small"
CHAT_MODEL           = "gpt-4o-mini"
TOP_K                = 3
SIMILARITY_THRESHOLD = 0.30

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

# ── Query Cleaner Prompt ───────────────────────────────────────────────────────
CLEAN_PROMPT = """You are a text correction assistant. 
Fix spelling mistakes, punctuation errors, and grammar in the user's message.
Return ONLY the corrected text — no explanation, no quotes, nothing else.

Examples:
Input:  "wat servises do u offfer"
Output: "What services do you offer?"

Input:  "hw mch dos the growt plan cst"
Output: "How much does the growth plan cost?"

Input:  "do u hve ai chatbot,,,for website??"
Output: "Do you have an AI chatbot for website?"
"""


class RAGEngine:
    """
    Manages the full RAG pipeline:
      1. Embed all KB entries at startup (cached in memory)
      2. Clean/fix user query (spelling + punctuation)
      3. Embed cleaned query
      4. Cosine similarity retrieval
      5. GPT-4o-mini generation with retrieved context
    """

    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.kb = KNOWLEDGE_BASE
        self._embeddings: np.ndarray | None = None
        self._index_ready = False

    # ── Index Building ─────────────────────────────────────────────────────────

    async def build_index(self) -> None:
        logger.info("Building RAG index for %d Q&A entries …", len(self.kb))
        texts = [item["question"] for item in self.kb]
        embeddings = await self._embed_batch(texts)
        self._embeddings = np.array(embeddings, dtype="float32")
        self._index_ready = True
        logger.info("RAG index ready. Embedding matrix shape: %s", self._embeddings.shape)

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    async def _embed_query(self, query: str) -> np.ndarray:
        response = await self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query],
        )
        return np.array(response.data[0].embedding, dtype="float32")

    # ── Query Cleaner ──────────────────────────────────────────────────────────

    async def _clean_query(self, raw: str) -> str:
        """
        Fix spelling, punctuation, and grammar using gpt-4o-mini.
        Returns corrected text. Falls back to original if anything fails.
        """
        try:
            response = await self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": CLEAN_PROMPT},
                    {"role": "user",   "content": raw},
                ],
                temperature=0,
                max_tokens=200,
            )
            cleaned = response.choices[0].message.content.strip()
            if cleaned and cleaned != raw:
                logger.info("Query cleaned: %r → %r", raw, cleaned)
            return cleaned or raw
        except Exception as e:
            logger.warning("Query cleaning failed, using raw: %s", e)
            return raw

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def _cosine_similarity(self, query_vec: np.ndarray) -> np.ndarray:
        kb_norm = self._embeddings / (np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10)
        q_norm  = query_vec        / (np.linalg.norm(query_vec)                                + 1e-10)
        return kb_norm @ q_norm

    def _retrieve(self, query_vec: np.ndarray, top_k: int = TOP_K) -> list[dict]:
        scores     = self._cosine_similarity(query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= SIMILARITY_THRESHOLD:
                results.append({**self.kb[idx], "score": round(score, 4)})
        return results

    # ── Generation ─────────────────────────────────────────────────────────────

    def _build_context(self, retrieved: list[dict]) -> str:
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
        """
        if not self._index_ready:
            raise RuntimeError("RAG index not built yet. Call build_index() first.")

        # 1. Clean query — fix spelling & punctuation
        cleaned_message = await self._clean_query(user_message)

        # 2. Embed cleaned query
        query_vec = await self._embed_query(cleaned_message)

        # 3. Retrieve relevant Q&As
        retrieved = self._retrieve(query_vec)

        # 4. Build context
        context = self._build_context(retrieved)

        # 5. Construct messages
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
        if history:
            messages.extend(history[-10:])

        # Use cleaned message for LLM (better comprehension)
        messages.append({"role": "user", "content": cleaned_message})

        # 6. Generate response
        response = await self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=512,
        )

        answer = response.choices[0].message.content.strip()

        sources = [
            {
                "category": r["category"],
                "question": r["question"],
                "score":    r["score"],
            }
            for r in retrieved
        ]

        logger.info(
            "Original: %r | Cleaned: %r | Sources: %d | Top score: %.4f",
            user_message[:60],
            cleaned_message[:60],
            len(retrieved),
            sources[0]["score"] if sources else 0,
        )

        return {
            "answer":          answer,
            "sources":         sources,
            "original_query":  user_message,
            "cleaned_query":   cleaned_message,
        }