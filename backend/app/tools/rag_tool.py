import os
from typing import Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.tools.base import Tool, ToolResult


class RAGTool(Tool):
    name: str = "rag_search"
    description: str = (
        "Answers questions about the user's background, resume, work experience, "
        "or personal notes by searching their personal knowledge base."
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

            if not chunk_texts:
                return ToolResult(
                    success=True,
                    output="No relevant information found in the personal knowledge base.",
                    error=None,
                )

            joined_output = "\n\n".join(chunk_texts)
            return ToolResult(
                success=True,
                output=joined_output,
                error=None,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"RAG search error: {str(e)}",
            )
