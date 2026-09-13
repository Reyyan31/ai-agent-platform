from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel


class ToolResult(BaseModel):
    success: bool
    output: Any = None
    error: Optional[str] = None


class Tool(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with keyword arguments and return a ToolResult."""
        pass
