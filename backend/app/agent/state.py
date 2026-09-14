from typing import TypedDict, Optional, Dict, Any, List


class AgentState(TypedDict):
    user_input: str
    selected_tool: Optional[str]
    tool_args: Optional[dict]
    tool_output: Optional[str]
    final_response: Optional[str]
    plan: Optional[List[Dict[str, Any]]]
    step_results: Optional[List[str]]
    classifier_label: Optional[str]
    classifier_confidence: Optional[float]
    routing_path: Optional[str]
    reflection_valid: Optional[bool]
    reflection_reason: Optional[str]
