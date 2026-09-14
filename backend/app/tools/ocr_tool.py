import os
from typing import Any, Optional

from PIL import Image
import pytesseract

from app.tools.base import Tool, ToolResult

pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class OCRTool(Tool):
    name: str = "ocr_extract"
    description: str = "Extracts text from an uploaded image using OCR."

    def run(
        self,
        image_path: Optional[str] = None,
        **kwargs: Any
    ) -> ToolResult:
        img_path = (
            image_path
            if image_path is not None
            else (kwargs.get("image_path") or kwargs.get("image") or kwargs.get("path") or "")
        )

        if not img_path or not img_path.strip():
            return ToolResult(
                success=False,
                output=None,
                error="No image path provided.",
            )

        try:
            image = Image.open(img_path)
            extracted_text = pytesseract.image_to_string(image).strip()
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"OCR error: {str(e)}",
            )

        if not extracted_text or not extracted_text.strip():
            return ToolResult(
                success=False,
                output=None,
                error="No text could be extracted from the image.",
            )

        return ToolResult(
            success=True,
            output=extracted_text,
            error=None,
        )