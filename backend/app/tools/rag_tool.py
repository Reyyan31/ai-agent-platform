import os
from typing import Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from app.memory.conversation_memory import memory_collection
from app.tools.base import Tool, ToolResult


class RAGTool(Tool):
    name: str = "rag_search"
    description: str = (
        "Answers questions about the user's background, resume, or work experience, AND recalls any facts, preferences, or information the user has previously told the agent in conversation (e.g. 'what did I say my favorite X was', 'what's my Y'). "
        "Use this any time the user is asking the agent to recall something specific they've shared, not just resume questions."
    )

    def __init__(self, chroma_dir: Optional[str] = None):
        if chroma_dir is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            chroma_dir = os.path.join(base_dir, "chroma_db")

        self.chroma_dir = chroma_dir
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(
            path=self.chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name="personal_knowledge")

    def _synthesize_answer(self, query: str, raw_chunks: str) -> str:
        try:
            client = OpenAI(
                api_key=os.environ.get("GROQ_API_KEY", ""),
                base_url="https://api.groq.com/openai/v1"
            )
            prompt = (
                "Answer the user's question directly and concisely using only the provided context, "
                "in 1-3 sentences, in plain conversational text with no markdown headers; "
                "if the context doesn't actually contain an answer to the question, say so honestly instead of guessing."
            )
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                temperature=0,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Context:\\n{raw_chunks}\\n\\nQuestion:\\n{query}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[RAG SYNTHESIS ERROR] {type(e).__name__}: {e}")
            return raw_chunks

    def run(self, query: Optional[str] = None, **kwargs: Any) -> ToolResult:
        search_query = (
            query
            if query is not None
            else (kwargs.get("query") or kwargs.get("search_query") or kwargs.get("input") or kwargs.get("text") or "")
        )

        if not search_query or not search_query.strip():
            return ToolResult(
                success=False,
                output=None,
                error="No query provided for knowledge base search.",
            )

        try:
            query_embedding = self.model.encode(search_query.strip()).tolist()

            # ── Query personal_knowledge ───────────────────────────────────
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
            )
            documents = results.get("documents", [[]])
            metadatas = results.get("metadatas", [[]])

            chunk_texts = []
            if documents and documents[0]:
                chunk_texts = [doc for doc in documents[0] if doc]
            elif metadatas and metadatas[0]:
                chunk_texts = [
                    m.get("text", "")
                    for m in metadatas[0]
                    if isinstance(m, dict) and m.get("text")
                ]

            # ── Query conversation_memory ──────────────────────────────────
            try:
                mem_results = memory_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=10,
                    include=["documents", "metadatas"],
                )
                mem_docs = mem_results.get("documents", [[]])
                mem_metas = mem_results.get("metadatas", [[]])
                if mem_docs and mem_docs[0]:
                    pairs = list(zip(
                        mem_docs[0],
                        mem_metas[0] if (mem_metas and mem_metas[0]) else [{}] * len(mem_docs[0])
                    ))
                    pairs = [(doc, meta) for doc, meta in pairs if doc]
                    pairs.sort(
                        key=lambda x: x[1].get("timestamp", "") if isinstance(x[1], dict) else "",
                        reverse=True
                    )
                    top_pairs = pairs[:5]
                    if top_pairs:
                        top_ts = top_pairs[0][1].get("timestamp", "unknown") if isinstance(top_pairs[0][1], dict) else "unknown"
                        print(f"[RAG MEMORY] top memory result timestamp: {top_ts}")
                    chunk_texts.extend([doc for doc, _ in top_pairs])
            except Exception:
                pass  # conversation_memory may be empty on first run — that's fine

            if not chunk_texts:
                return ToolResult(
                    success=True,
                    output="No relevant information found in the personal knowledge base.",
                    error=None,
                )

            joined_output = "\n\n".join(chunk_texts)
            return ToolResult(
                success=True,
                output=self._synthesize_answer(search_query, joined_output),
                error=None,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"RAG search error: {str(e)}",
            )
