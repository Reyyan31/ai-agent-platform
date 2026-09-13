import re
from typing import Any, Optional
from app.tools.base import Tool, ToolResult


class CalculatorTool(Tool):
    name: str = "calculator"
    description: str = "Evaluates basic arithmetic expressions safely."

    # Allow only digits, decimals, operators (+, -, *, /, %, **), parentheses, and whitespace
    SAFE_PATTERN = re.compile(r"^[0-9\.\+\-\*\/\%\(\)\s]+$")

    def run(self, expression: Optional[str] = None, **kwargs: Any) -> ToolResult:
        """Evaluates basic arithmetic expression safely using restricted eval."""
        expr = expression if expression is not None else (kwargs.get("expr") or kwargs.get("input") or kwargs.get("text"))
        
        if expr is None or not isinstance(expr, str) or not expr.strip():
            return ToolResult(
                success=False,
                output=None,
                error="No arithmetic expression provided."
            )

        cleaned_expr = expr.strip()

        if not self.SAFE_PATTERN.match(cleaned_expr):
            return ToolResult(
                success=False,
                output=None,
                error="Invalid expression: contains forbidden characters or instructions."
            )

        try:
            # Restricted eval with no builtins accessible
            result = eval(cleaned_expr, {"__builtins__": None}, {})
            return ToolResult(
                success=True,
                output=result,
                error=None
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Evaluation error: {str(e)}"
            )
