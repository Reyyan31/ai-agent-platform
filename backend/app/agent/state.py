from typing import TypedDict, Optional, Dict, Any, List


class AgentState(TypedDict):
    user_input: str
    selected_tool: Optional[str]
    tool_args: Optional[dict]
    tool_output: Optional[str]
    final_response: Optional[str]
    plan: Optional[List[Dict[str, Any]]]
    step_results: Optional[List[str]]
