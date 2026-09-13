from typing import Any, Optional
from app.tools.base import Tool, ToolResult


class EchoTool(Tool):
    name: str = "echo"
    description: str = "Echoes back input text."

    def run(self, text: Optional[str] = None, **kwargs: Any) -> ToolResult:
        """Echo back the provided input text."""
        input_text = text if text is not None else (kwargs.get("input") or kwargs.get("message") or "")
        return ToolResult(
            success=True,
            output=input_text,
            error=None
        )
