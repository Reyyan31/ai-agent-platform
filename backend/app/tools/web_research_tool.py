import os
import re
from typing import Any, Optional

from tavily import TavilyClient
from openai import OpenAI

from app.tools.base import Tool, ToolResult
from app.tools.document_tool import DocumentTool


class WebResearchTool(Tool):
    name: str = "web_research"
    description: str = (
        "Searches the live web for current information on any topic. "
        "Can also save research findings as a Word document if requested."
    )

    def __init__(self):
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.groq_client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1"
        )

    def run(
        self,
        query: Optional[str] = None,
        save_as_document: bool = False,
        **kwargs: Any
    ) -> ToolResult:
        research_query = (
            query
            if query is not None
            else (kwargs.get("query") or kwargs.get("search_query") or kwargs.get("text") or "")
        )

        if not research_query or not research_query.strip():
            return ToolResult(
                success=False,
                output=None,
                error="No research query provided.",
            )

        try:
            search_response = self.client.search(research_query, max_results=5)
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Web research error: {str(e)}",
            )

        raw_context_parts = []
        results = search_response.get("results", [])
        print(f"[WEB RESEARCH] Tavily returned {len(results)} results")
        for result in results:
            content = result.get("content") or result.get("raw_content") or ""
            if content.strip():
                raw_context_parts.append(content)
        print(f"[WEB RESEARCH] Extracted {len(raw_context_parts)} content chunks")
        if not raw_context_parts:
            print(f"[WEB RESEARCH] No usable content in any of the {len(results)} results")

        if not raw_context_parts:
            return ToolResult(
                success=True,
                output="No web results found for the query.",
                error=None,
            )

        raw_context = "\n\n".join(raw_context_parts)

        # Synthesize using Groq
        system_prompt = (
            "Write a clear, well-organized answer to the user's query using only the provided web search context. "
            "Use ## for section headings and - for bullet points. "
            "Mention points in general terms (e.g., 'according to recent reports', 'sources indicate') "
            "without fabricating specific URLs. "
            "Return only the answer with no conversational preamble."
        )

        try:
            synthesis_response = self.groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{raw_context}\n\nQuery:\n{research_query}"},
                ],
            )
            synthesized_text = synthesis_response.choices[0].message.content or ""
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Synthesis error: {str(e)}",
            )

        if not synthesized_text.strip():
            synthesized_text = raw_context

        if not save_as_document:
            self._last_raw_context = raw_context
            return ToolResult(
                success=True,
                output=synthesized_text,
                error=None,
            )

        # Slugify the query for filename
        slug = re.sub(r"[^\w\s-]", "", research_query.lower())
        slug = re.sub(r"[\s]+", "_", slug)[:50]

        document_tool = DocumentTool()
        return document_tool.run(content=synthesized_text, format="docx", filename=slug)