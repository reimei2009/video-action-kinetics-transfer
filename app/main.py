"""
FastAPI Main Application

Entry point cho Video Action Recognition API:
- Khởi tạo FastAPI app
- Setup CORS, middleware
- Include routers
- Startup/shutdown events
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import time

from app.core.config import settings
from app.api.routes import router
from app.services.model_service import model_service
from app.services.video_service import video_service


# ==========================================
# Lifespan Context Manager (Startup/Shutdown)
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events: Chạy khi start/stop server
    
    Startup:
    - Model đã được load tự động khi import model_service
    - Cleanup old temp files
    
    Shutdown:
    - Cleanup temp files
    """
    # ===== STARTUP =====
    print("\n" + "=" * 60)
    print("🚀 FastAPI Server Starting...")
    print("=" * 60)
    
    # Cleanup old temp files
    video_service.cleanup_old_files(max_age_hours=1)
    
    print(f"✅ API Ready: http://{settings.HOST}:{settings.PORT}")
    print(f"✅ Docs: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"✅ Model: {settings.MODEL_NAME}")
    print("=" * 60 + "\n")
    
    yield  # Server running...
    
    # ===== SHUTDOWN =====
    print("\n" + "=" * 60)
    print("🛑 FastAPI Server Shutting Down...")
    print("=" * 60)
    
    # Cleanup temp files
    video_service.cleanup_old_files(max_age_hours=0)
    
    print("✅ Cleanup completed")
    print("=" * 60 + "\n")


# ==========================================
# FastAPI App Initialization
# ==========================================

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan,
    
    # Swagger UI configuration
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    
    # OpenAPI tags
    openapi_tags=[
        {
            "name": "Video Action Recognition",
            "description": "Endpoints for video action recognition"
        }
    ]
)


# ==========================================
# CORS Middleware
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # Frontend origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# ==========================================
# Custom Middleware (Request Logging)
# ==========================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log mỗi request với timestamp và processing time
    """
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log
    print(f"[{request.method}] {request.url.path} - {response.status_code} ({process_time:.3f}s)")
    
    # Add header với processing time
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# ==========================================
# Exception Handlers
# ==========================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors (400 Bad Request)
    
    Trả về detailed error messages
    """
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "message": "Validation error",
            "error_code": "VALIDATION_ERROR",
            "details": {
                "errors": errors
            }
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handle tất cả uncaught exceptions (500 Internal Server Error)
    """
    print(f"❌ Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "details": {
                "error": str(exc),
                "path": str(request.url.path)
            }
        }
    )


# ==========================================
# Include Routers
# ==========================================

# Main API routes
app.include_router(router)


# ==========================================
# Root Endpoint
# ==========================================

@app.get(
    "/",
    tags=["Root"],
    summary="API Welcome",
    description="Root endpoint with API information"
)
async def read_root():
    """
    Welcome endpoint
    
    Returns API info và quick links
    """
    return {
        "message": "🎥 Video Action Recognition API",
        "version": settings.API_VERSION,
        "status": "running",
        "model": {
            "name": settings.MODEL_NAME,
            "architecture": "X3D-XS",
            "num_classes": settings.NUM_CLASSES
        },
        "endpoints": {
            "api_root": "/api/v1/",
            "predict": "/api/v1/predict",
            "health": "/api/v1/health",
            "models": "/api/v1/models",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "documentation": {
            "swagger_ui": f"http://{settings.HOST}:{settings.PORT}/docs",
            "redoc": f"http://{settings.HOST}:{settings.PORT}/redoc",
            "openapi_json": f"http://{settings.HOST}:{settings.PORT}/openapi.json"
        },
        "quick_start": {
            "1": "Go to /docs for interactive API documentation",
            "2": "Use POST /api/v1/predict to upload video and get predictions",
            "3": "Check /api/v1/health for server status"
        }
    }


# ==========================================
# Health Check (Simple)
# ==========================================

@app.get(
    "/ping",
    tags=["Root"],
    summary="Simple ping",
    description="Simple health check (no model check)"
)
async def ping():
    """
    Simple ping endpoint (no heavy operations)
    """
    return {
        "status": "ok",
        "message": "pong"
    }


# ==========================================
# Development Info
# ==========================================

if __name__ == "__main__":
    """
    Development mode: Run with uvicorn
    
    Production mode: Use run_api.py hoặc:
        uvicorn app.main:app --host 0.0.0.0 --port 8000
    """
    import uvicorn
    
    print("\n" + "=" * 60)
    print("🔧 Development Mode")
    print("=" * 60)
    print(f"Starting server at http://{settings.HOST}:{settings.PORT}")
    print(f"Docs: http://{settings.HOST}:{settings.PORT}/docs")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,  # Auto-reload khi code thay đổi
        log_level=settings.LOG_LEVEL.lower()
    )
