from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

from app.agent.graph import agent_graph

router = APIRouter()


class AgentRequest(BaseModel):
    message: str


class AgentResponse(BaseModel):
    tool_used: Optional[str] = None
    response: str


@router.post("/agent/run", response_model=AgentResponse)
async def run_agent(req: AgentRequest) -> AgentResponse:
    result = agent_graph.invoke({"user_input": req.message})
    return AgentResponse(
        tool_used=result.get("selected_tool"),
        response=result.get("final_response") or "",
    )
