import os
from dotenv import load_dotenv
from openai import OpenAI

from app.tools.base import Tool, ToolResult

load_dotenv()

_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY") or "EMPTY_API_KEY",
    base_url="https://api.groq.com/openai/v1",
)

_SYSTEM_PROMPT = """\
You are an expert programming assistant.

When given a problem or question, respond with:
1. Correct, working code in the requested language, with clear inline comments explaining each key step.
2. A section headed exactly "## Explanation" followed by bullet points (each starting with "- ") that describe the key logic and any assumptions you made.

Do not include anything outside the code block and the Explanation section.
"""


class CodeHelperTool(Tool):
    name: str = "code_helper"
    description: str = (
        "Generates code solutions with explanations for programming problems, "
        "coding assignments, or technical questions. Use this when the user asks "
        "to write, generate, fix, debug, or explain code."
    )

    def run(self, problem: str = "", language: str = "python", **kwargs) -> ToolResult:
        """Calls Groq to produce a code solution + explanation for the given problem."""
        problem = problem or kwargs.get("problem", "")
        language = language or kwargs.get("language", "python")

        if not problem or not problem.strip():
            return ToolResult(
                success=False,
                output=None,
                error="No problem description provided.",
            )

        try:
            response = _client.chat.completions.create(
                model="openai/gpt-oss-20b",
                temperature=0,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Language: {language}\n\nProblem: {problem}",
                    },
                ],
            )
            result_text = response.choices[0].message.content.strip()
            return ToolResult(success=True, output=result_text, error=None)

        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Code generation failed: {type(e).__name__}: {e}",
            )
