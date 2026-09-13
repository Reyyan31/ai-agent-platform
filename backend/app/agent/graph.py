import json
import os
from typing import Any, Dict, List, Optional
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
    """Generates an execution plan (list of steps) for the user's input."""
    user_input = state.get("user_input", "")
    tool_descriptions = get_tool_descriptions()

    tools_summary = "\n".join(
        [f"- {name}: {description}" for name, description in tool_descriptions.items()]
    )

    system_prompt = (
        "You are an AI planning and routing agent. Your job is to analyze the user request and generate an ordered plan of execution steps.\n\n"
        f"Available tools:\n{tools_summary}\n\n"
        "Instructions:\n"
        "- Return a JSON object with a 'steps' array representing the sequential execution plan.\n"
        "- Each item in 'steps' must be an object with 'tool' (string) and 'args' (object): {\"steps\": [{\"tool\": \"<tool_name>\", \"args\": {<tool_args>}}]}\n"
        "- If a single tool is needed, return a 1-item list in 'steps'.\n"
        "- If a request requires background information before writing a document, create a multi-step plan.\n"
        "- Tool argument formats:\n"
        "  - 'echo': {\"text\": \"<text to echo>\"}\n"
        "  - 'calculator': {\"expression\": \"<arithmetic expression>\"}\n"
        "  - 'rag_search': {\"query\": \"<search query>\"}\n"
        "  - 'create_document': {\"format\": \"docx|pdf\", \"filename\": \"<filename>\", \"content\": \"<optional initial content>\"}\n"
        "- If no tool is appropriate, return {\"steps\": []}.\n\n"
        "Example Two-Step Plan for 'Create a Word document summary of my work experience':\n"
        "{\n"
        '  "steps": [\n'
        '    {"tool": "rag_search", "args": {"query": "work experience and career summary"}},\n'
        '    {"tool": "create_document", "args": {"format": "docx", "filename": "work_experience_summary", "content": ""}}\n'
        "  ]\n"
        "}\n"
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

        plan: List[Dict[str, Any]] = []
        if "steps" in parsed and isinstance(parsed["steps"], list):
            for step in parsed["steps"]:
                if isinstance(step, dict):
                    tool_name = step.get("tool")
                    if tool_name and str(tool_name).lower() not in ("null", "none", ""):
                        step_args = step.get("args") if isinstance(step.get("args"), dict) else {}
                        plan.append({"tool": tool_name, "args": step_args})
        elif "tool" in parsed and parsed.get("tool"):
            raw_tool = parsed.get("tool")
            if str(raw_tool).lower() not in ("null", "none", ""):
                raw_args = parsed.get("args") if isinstance(parsed.get("args"), dict) else {}
                plan.append({"tool": raw_tool, "args": raw_args})

    except Exception as e:
        print(f"[ROUTING ERROR] {type(e).__name__}: {e}")
        plan = []

    first_step = plan[0] if plan else {}
    return {
        "plan": plan,
        "step_results": [],
        "selected_tool": first_step.get("tool"),
        "tool_args": first_step.get("args", {}),
    }


def synthesize_content(user_input: str, raw_notes: str) -> str:
    """Uses LLM to synthesize raw notes into a clean, structured document body."""
    system_prompt = (
        "You are an expert technical writer. Read the raw notes provided and write a clean, "
        "well-organized document body that fulfills the user's original request.\n\n"
        "Formatting rules:\n"
        "- Prefix section headings with '## '.\n"
        "- Prefix bullet points with '- '.\n"
        "- Use normal paragraphs for standard text.\n"
        "- Do not use any other markdown syntax (no #, ###, **, *, `, etc.).\n"
        "- Return only the document body text with no conversational preamble or sign-off."
    )

    user_message = f"User Request: {user_input}\n\nRaw Notes:\n{raw_notes}"

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        synthesized_text = response.choices[0].message.content or ""
        return synthesized_text.strip() or raw_notes
    except Exception as e:
        print(f"[SYNTHESIS ERROR] {type(e).__name__}: {e}")
        return raw_notes


def execute_step(state: AgentState) -> Dict[str, Any]:
    """Pops the next step from the remaining plan, executes it, and records the result."""
    remaining_plan = list(state.get("plan") or [])
    step_results = list(state.get("step_results") or [])

    if not remaining_plan:
        return {
            "plan": [],
            "step_results": step_results,
            "tool_output": None,
            "final_response": state.get("final_response")
            or f"No steps executed for input: '{state.get('user_input', '')}'",
        }

    current_step = remaining_plan.pop(0)
    tool_name = current_step.get("tool")
    tool_args = dict(current_step.get("args") or {})

    # Automatically synthesize previous step's output into create_document content if available
    if tool_name == "create_document":
        existing_content = tool_args.get("content", "")
        if step_results:
            previous_output = step_results[-1]
            tool_args["content"] = synthesize_content(
                state.get("user_input", ""), previous_output
            )
        else:
            tool_args["content"] = existing_content

    if not tool_name or tool_name not in TOOL_REGISTRY:
        output_str = f"Tool '{tool_name}' not found or unavailable."
        final_response = f"No valid tool was found for step: '{tool_name}'"
        step_results.append(output_str)
    else:
        tool = TOOL_REGISTRY[tool_name]
        try:
            result = tool.run(**tool_args)
            if result.success:
                output_str = str(result.output)
                final_response = output_str
            else:
                output_str = f"Error: {result.error}"
                final_response = f"Tool '{tool_name}' execution error: {result.error}"
        except Exception as e:
            output_str = f"Execution exception: {str(e)}"
            final_response = f"Failed to execute tool '{tool_name}': {str(e)}"

        step_results.append(output_str)

    return {
        "plan": remaining_plan,
        "step_results": step_results,
        "selected_tool": tool_name,
        "tool_args": tool_args,
        "tool_output": output_str,
        "final_response": final_response,
    }


def should_continue(state: AgentState) -> str:
    """Checks if there are remaining steps in the execution plan."""
    remaining_plan = state.get("plan")
    if remaining_plan and len(remaining_plan) > 0:
        return "execute_step"
    return END


# Build LangGraph StateGraph with looping conditional edges
graph_builder = StateGraph(AgentState)

# Add nodes
graph_builder.add_node("route", route_node)
graph_builder.add_node("execute_step", execute_step)

# Set entry point and edges
graph_builder.set_entry_point("route")
graph_builder.add_edge("route", "execute_step")
graph_builder.add_conditional_edges(
    "execute_step",
    should_continue,
    {
        "execute_step": "execute_step",
        END: END,
    },
)

# Compile graph
agent_graph = graph_builder.compile()
