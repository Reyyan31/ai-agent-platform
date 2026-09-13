import json
import os
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, END

from app.agent.state import AgentState
from app.tools import TOOL_REGISTRY, get_tool_descriptions

load_dotenv()

# Instantiate client pointing to Groq's OpenAI-compatible endpoint
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY") or "EMPTY_API_KEY",
    base_url="https://api.groq.com/openai/v1",
)


def route_node(state: AgentState) -> Dict[str, Any]:
    """Routes the user input to the most appropriate tool using OpenAI chat completions."""
    user_input = state.get("user_input", "")
    tool_descriptions = get_tool_descriptions()

    tools_summary = "\n".join(
        [f"- {name}: {description}" for name, description in tool_descriptions.items()]
    )

    system_prompt = (
        "You are an AI routing agent. Your job is to select the best-fitting tool to fulfill the user's request.\n\n"
        f"Available tools:\n{tools_summary}\n\n"
        "Instructions:\n"
        "- Return a JSON object in the exact format: {\"tool\": \"<tool_name or null>\", \"args\": {<tool_args>}}\n"
        "- If 'echo' tool is selected, args should be: {\"text\": \"<text to echo>\"}\n"
        "- If 'calculator' tool is selected, args should be: {\"expression\": \"<arithmetic expression>\"}\n"
        "- If 'rag_search' tool is selected, args should be: {\"query\": \"<search query>\"}\n"
        "- If no available tool matches the request, set \"tool\" to null and \"args\" to null or {}\n"
    )

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)

        raw_tool = parsed.get("tool")
        if raw_tool is not None and str(raw_tool).lower() in ("null", "none", ""):
            selected_tool = None
        else:
            selected_tool = raw_tool

        raw_args = parsed.get("args")
        if isinstance(raw_args, dict):
            tool_args = raw_args
        else:
            tool_args = {}

    except Exception as e:
        print(f"[ROUTING ERROR] {type(e).__name__}: {e}")
        selected_tool = None
        tool_args = {}

    return {
        "selected_tool": selected_tool,
        "tool_args": tool_args,
    }


def execute_node(state: AgentState) -> Dict[str, Any]:
    """Executes the selected tool from TOOL_REGISTRY with the provided arguments."""
    selected_tool = state.get("selected_tool")
    tool_args = state.get("tool_args") or {}

    if not selected_tool or selected_tool not in TOOL_REGISTRY:
        return {
            "tool_output": None,
            "final_response": f"No valid tool was selected or available for input: '{state.get('user_input', '')}'",
        }

    tool = TOOL_REGISTRY[selected_tool]
    try:
        result = tool.run(**tool_args)
        if result.success:
            output_str = str(result.output)
            final_response = output_str
        else:
            output_str = f"Error: {result.error}"
            final_response = f"Tool execution error: {result.error}"
    except Exception as e:
        output_str = f"Execution exception: {str(e)}"
        final_response = f"Failed to execute tool '{selected_tool}': {str(e)}"

    return {
        "tool_output": output_str,
        "final_response": final_response,
    }


# Build LangGraph StateGraph
graph_builder = StateGraph(AgentState)

# Add nodes
graph_builder.add_node("route", route_node)
graph_builder.add_node("execute", execute_node)

# Set entry point and edges
graph_builder.set_entry_point("route")
graph_builder.add_edge("route", "execute")
graph_builder.add_edge("execute", END)

# Compile graph
agent_graph = graph_builder.compile()
