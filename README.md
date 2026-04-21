# MHDEVFUSION RAG Chatbot

A production-ready RAG (Retrieval-Augmented Generation) chatbot for MHDEVFUSION
built with FastAPI + OpenAI.

## Architecture

```
User query
    │
    ▼
text-embedding-3-small          ← embed query (cheap & accurate)
    │
    ▼
Cosine Similarity               ← compare against 42 pre-embedded Q&As
    │
    ▼
Top-3 Q&A Context               ← only most relevant answers passed to LLM
    │
    ▼
gpt-4o-mini                     ← generate grounded response
    │
    ▼
Answer + Source Tags            ← returned to user
```

## Setup

### 1. Clone / copy files
```
mhdevfusion-chatbot/
├── main.py
├── rag_engine.py
├── knowledge_base.py
├── requirements.txt
├── .env.example
└── static/
    └── index.html
```

### 2. Create virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up .env
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

### 5. Run the server
```bash
uvicorn main:app --reload --port 8000
```

### 6. Open chat UI
Visit: http://localhost:8000

## API Endpoints

| Method | Path     | Description                     |
|--------|----------|---------------------------------|
| GET    | /        | Chat UI (HTML)                  |
| GET    | /health  | Health check + index status     |
| POST   | /chat    | RAG chat endpoint               |

### POST /chat

**Request:**
```json
{
  "message": "What is included in the Growth plan?",
  "history": [
    {"role": "user", "content": "What plans do you offer?"},
    {"role": "assistant", "content": "We have five plans..."}
  ]
}
```

**Response:**
```json
{
  "answer": "The Growth plan ($199) includes...",
  "sources": [
    {"category": "Pricing", "question": "What is included in the Growth plan?", "score": 0.9241}
  ]
}
```

## Cost Estimate

| Component              | Model                    | Cost              |
|------------------------|--------------------------|-------------------|
| Embeddings (startup)   | text-embedding-3-small   | ~$0.0001 one-time |
| Per query embed        | text-embedding-3-small   | ~$0.000002        |
| Per response           | gpt-4o-mini              | ~$0.0002          |
| **Per conversation**   |                          | **< $0.005**      |

## Customisation

- **Add Q&As**: Edit `knowledge_base.py` and restart (index rebuilds automatically)
- **Adjust retrieval**: Change `TOP_K` and `SIMILARITY_THRESHOLD` in `rag_engine.py`
- **System prompt**: Edit `SYSTEM_PROMPT` in `rag_engine.py`
- **Temperature**: Adjust in `rag_engine.py` (default: 0.3 for factual accuracy)
