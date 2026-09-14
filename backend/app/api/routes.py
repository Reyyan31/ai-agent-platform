import os
import uuid
import secrets
from typing import Optional
from fastapi import APIRouter, Form, File, UploadFile, HTTPException, Depends, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.agent.graph import agent_graph
from app.tools.ocr_tool import OCRTool

router = APIRouter()

_GENERATED_DOCS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "generated_docs")
)

_MEDIA_TYPES = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pdf": "application/pdf",
}


class AgentResponse(BaseModel):
    tool_used: Optional[str] = None
    response: str
    classifier_label: Optional[str] = None
    classifier_confidence: Optional[float] = None
    routing_path: Optional[str] = None
    reflection_valid: Optional[bool] = None
    reflection_reason: Optional[str] = None


def verify_api_key(x_api_key: str = Header(None)):
    """Verify the API key against the configured API_KEY."""
    expected_key = os.getenv("API_KEY")
    if not x_api_key or x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@router.get("/documents/download/{filename}")
async def download_document(
    filename: str,
    _: None = Depends(verify_api_key),
):
    # Sanitize filename: strip path separators and .. sequences
    sanitized = os.path.basename(filename.replace("..", ""))
    if not sanitized:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Build full path and verify it's inside _GENERATED_DOCS_DIR
    full_path = os.path.join(_GENERATED_DOCS_DIR, sanitized)
    resolved_path = os.path.abspath(full_path)
    
    if not resolved_path.startswith(os.path.abspath(_GENERATED_DOCS_DIR)) or not os.path.isfile(resolved_path):
        raise HTTPException(status_code=404, detail={"error": "File not found"})
    
    # Determine media type
    ext = os.path.splitext(sanitized)[1].lower()
    media_type = _MEDIA_TYPES.get(ext, "application/octet-stream")
    
    return FileResponse(
        resolved_path,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{sanitized}"'}
    )


@router.post("/agent/run", response_model=AgentResponse)
async def run_agent(
    message: str = Form(min_length=1, description="User message to the agent"),
    image: Optional[UploadFile] = File(None),
    _: None = Depends(verify_api_key),
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
        classifier_label=result.get("classifier_label"),
        classifier_confidence=result.get("classifier_confidence"),
        routing_path=result.get("routing_path"),
        reflection_valid=result.get("reflection_valid"),
        reflection_reason=result.get("reflection_reason"),
    )
