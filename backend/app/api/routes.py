import os
import uuid
from typing import Optional
from fastapi import APIRouter, Form, File, UploadFile
from pydantic import BaseModel

from app.agent.graph import agent_graph
from app.tools.ocr_tool import OCRTool

router = APIRouter()


class AgentRequest(BaseModel):
    message: str


class AgentResponse(BaseModel):
    tool_used: Optional[str] = None
    response: str


@router.post("/agent/run", response_model=AgentResponse)
async def run_agent(
    message: str = Form(...),
    image: Optional[UploadFile] = File(None),
) -> AgentResponse:
    user_input = message
    
    if image is not None:
        temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        
        file_ext = os.path.splitext(image.filename)[1] if image.filename else ".png"
        temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}{file_ext}")
        
        try:
            content = await image.read()
            with open(temp_path, "wb") as f:
                f.write(content)
            
            ocr_result = OCRTool().run(image_path=temp_path)
            
            if ocr_result.success:
                user_input = f"{message}\n\n[Extracted from uploaded image]:\n{ocr_result.output}"
            else:
                return AgentResponse(
                    tool_used="ocr_extract",
                    response=f"Failed to read image: {ocr_result.error}",
                )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    result = agent_graph.invoke({"user_input": user_input})
    return AgentResponse(
        tool_used=result.get("selected_tool"),
        response=result.get("final_response") or "",
    )
