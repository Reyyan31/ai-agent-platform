"""
conversation_memory.py
-----------------------
Persists every user turn to a dedicated ChromaDB collection so the agent
builds up a searchable conversation history over time.
"""

import os
import uuid
from datetime import datetime, timezone

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_BACKEND    = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_CHROMA_DIR = os.path.join(_BACKEND, "chroma_db")

# ── Load once at import time ───────────────────────────────────────────────────
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

_chroma_client = chromadb.PersistentClient(
    path=_CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False),
)

_collection = _chroma_client.get_or_create_collection(name="conversation_memory")


# ── Public API ─────────────────────────────────────────────────────────────────

def save_to_memory(text: str) -> None:
    """
    Embed *text* and upsert it into the conversation_memory collection.
    Skips silently if text is empty or whitespace-only.
    """
    if not text or not text.strip():
        return

    embedding = _embedder.encode([text.strip()])[0].tolist()

    try:
        results = _collection.query(
            query_embeddings=[embedding],
            n_results=1,
        )
        distances = results.get("distances", [[]])[0]
        if distances and distances[0] < 0.05:
            print(f"[MEMORY SKIPPED] near-duplicate: {text[:60]}{'...' if len(text) > 60 else ''}")
            return
    except Exception:
        pass  # collection may be empty

    doc_id    = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    _collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[{"timestamp": timestamp, "source": "chat"}],
        documents=[text.strip()],
    )

    print(f"[MEMORY SAVED] {text[:60]}{'...' if len(text) > 60 else ''}")


# Public alias so other modules can query conversation history
memory_collection = _collection
