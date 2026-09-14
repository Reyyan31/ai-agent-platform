import traceback

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from app.api.routes import router as api_router

load_dotenv()

app = FastAPI(
    title="AI Agent Platform API",
    version="1.0.0",
)

# CORS: allow localhost:3000 for Next.js dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")


# Global exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"[UNHANDLED ERROR] {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"error": "Invalid request", "detail": exc.errors()},
    )


@app.get("/health")
def health_check():
    return {"status": "ok"}
