from typing import Dict, Any
from app.tools.base import Tool, ToolResult
from app.tools.echo_tool import EchoTool
from app.tools.calculator_tool import CalculatorTool

TOOL_REGISTRY: Dict[str, Tool] = {
    EchoTool.name: EchoTool(),
    CalculatorTool.name: CalculatorTool(),
}


def get_tool_descriptions() -> Dict[str, str]:
    """Returns a dictionary mapping each tool name to its description."""
    return {name: tool.description for name, tool in TOOL_REGISTRY.items()}


__all__ = [
    "Tool",
    "ToolResult",
    "EchoTool",
    "CalculatorTool",
    "TOOL_REGISTRY",
    "get_tool_descriptions",
]
