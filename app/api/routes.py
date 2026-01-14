"""
API Routes - FastAPI Endpoints

Định nghĩa các REST API endpoints:
- POST /predict: Upload video → predict actions
- GET /health: Health check
- GET /models: Model info
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional
import time
from datetime import datetime

from app.models.schemas import (
    PredictionResponse, 
    PredictionResult,
    VideoMetadata,
    ErrorResponse, 
    HealthResponse,
    ModelInfo
)
from app.services.model_service import model_service
from app.services.video_service import video_service
from app.core.config import settings


# ==========================================
# Router Setup
# ==========================================
router = APIRouter(
    prefix="/api/v1",
    tags=["Video Action Recognition"],
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        400: {"model": ErrorResponse, "description": "Bad Request"}
    }
)

# Track server start time (for uptime)
SERVER_START_TIME = time.time()


# ==========================================
# Endpoints
# ==========================================

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict video action",
    description="""
    Upload a video file and get top-K action predictions.
    
    **Process:**
    1. Upload video (max 100MB)
    2. Preprocess: Extract 2-second clip, resize, normalize
    3. Model inference: X3D predicts actions
    4. Return top-K predictions with confidence scores
    
    **Supported formats:** MP4, AVI, MOV, MKV, WEBM, FLV
    """,
    responses={
        200: {
            "description": "Prediction successful",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "message": "Prediction completed successfully",
                        "predictions": [
                            {"label": "PlayingPiano", "confidence": 0.2100, "rank": 1},
                            {"label": "Archery", "confidence": 0.0886, "rank": 2}
                        ],
                        "model_name": "ucf101",
                        "processing_time": 1.234,
                        "video_metadata": {
                            "filename": "test.mp4",
                            "duration": 12.88,
                            "fps": 25,
                            "size_mb": 2.5
                        }
                    }
                }
            }
        }
    }
)
async def predict_video(
    file: UploadFile = File(
        ..., 
        description="Video file to analyze"
    ),
    top_k: Optional[int] = Query(
        None,
        description="Number of top predictions (default: 5)",
        ge=1,
        le=20
    )
):
    """
    **Main endpoint: Predict video action**
    
    Upload video → Get top-K action predictions
    """
    start_time = time.time()
    
    try:
        # Step 1: Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "message": "Invalid file type. Please upload a video file.",
                    "error_code": "INVALID_FILE_TYPE",
                    "details": {
                        "content_type": file.content_type,
                        "allowed_types": "video/*"
                    }
                }
            )
        
        # Step 2: Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Step 3: Validate file (extension, size)
        is_valid, error_message = video_service.validate_video_file(
            file.filename, 
            file_size
        )
        
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "message": error_message,
                    "error_code": "INVALID_VIDEO",
                    "details": {
                        "filename": file.filename,
                        "size_mb": round(file_size / (1024 * 1024), 2)
                    }
                }
            )
        
        print(f"\n{'='*60}")
        print(f"📹 Processing video: {file.filename}")
        print(f"{'='*60}")
        
        # Step 4: Process video (save → metadata → preprocess)
        video_tensor, metadata = video_service.process_video(
            file_content, 
            file.filename
        )
        
        # Step 5: Predict with model
        predictions = model_service.predict(
            video_tensor, 
            top_k=top_k or settings.TOP_K
        )
        
        # Step 6: Format response
        prediction_results = [
            PredictionResult(
                label=label,
                confidence=round(confidence, 4),
                rank=i + 1
            )
            for i, (label, confidence) in enumerate(predictions)
        ]
        
        processing_time = time.time() - start_time
        
        response = PredictionResponse(
            success=True,
            message="Prediction completed successfully",
            predictions=prediction_results,
            model_name=settings.MODEL_NAME,
            processing_time=round(processing_time, 4),
            video_metadata=VideoMetadata(**metadata),
            timestamp=datetime.now()
        )
        
        print(f"✅ Prediction completed in {processing_time:.3f}s")
        print(f"   Top prediction: {predictions[0][0]} ({predictions[0][1]:.2%})")
        print(f"{'='*60}\n")
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions (đã format sẵn)
        raise
    
    except Exception as e:
        # Catch unexpected errors
        print(f"❌ Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "Internal server error during prediction",
                "error_code": "PREDICTION_ERROR",
                "details": {
                    "error": str(e)
                }
            }
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API and model are running properly"
)
async def health_check():
    """
    **Health check endpoint**
    
    Returns server status, model info, and uptime
    """
    try:
        model_info = model_service.get_model_info()
        uptime_seconds = time.time() - SERVER_START_TIME
        
        return HealthResponse(
            status="healthy",
            api_version=settings.API_VERSION,
            model_loaded=model_service.model is not None,
            model_name=model_info["model_name"],
            device=model_info["device"],
            uptime_seconds=round(uptime_seconds, 2),
            timestamp=datetime.now()
        )
    
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            api_version=settings.API_VERSION,
            model_loaded=False,
            model_name="unknown",
            device="unknown",
            uptime_seconds=None,
            timestamp=datetime.now()
        )


@router.get(
    "/models",
    response_model=ModelInfo,
    summary="Get model information",
    description="List available models and current model details"
)
async def get_models():
    """
    **Model info endpoint**
    
    Returns available models and current model configuration
    """
    try:
        model_info = model_service.get_model_info()
        
        return ModelInfo(
            available_models=["kinetics", "ucf101"],
            current_model=settings.MODEL_NAME,
            model_details={
                "architecture": model_info["architecture"],
                "num_classes": model_info["num_classes"],
                "class_names": model_info["class_names"],
                "device": model_info["device"],
                "weights_path": model_info["weights_path"],
                "input_shape": model_info["input_shape"],
                "clip_duration": f"{settings.VIDEO_CLIP_DURATION}s",
                "num_frames": settings.VIDEO_NUM_FRAMES,
                "crop_size": settings.VIDEO_CROP_SIZE
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "Error retrieving model info",
                "error_code": "MODEL_INFO_ERROR",
                "details": {"error": str(e)}
            }
        )


@router.post(
    "/models/switch",
    summary="Switch model (kinetics ↔ ucf101)",
    description="Switch between Kinetics and UCF101 models",
    response_model=dict
)
async def switch_model(
    model_name: str = Query(
        ...,
        description="Model to switch to",
        pattern="^(kinetics|ucf101)$"
    )
):
    """
    **Switch model endpoint**
    
    Change between Kinetics-400 and UCF101 models
    """
    try:
        if model_name not in ["kinetics", "ucf101"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid model name. Use 'kinetics' or 'ucf101'"
            )
        
        if model_name == settings.MODEL_NAME:
            return {
                "success": True,
                "message": f"Already using {model_name} model",
                "current_model": model_name
            }
        
        # Reload model
        model_service.reload_model(model_name)
        
        return {
            "success": True,
            "message": f"Switched to {model_name} model successfully",
            "current_model": model_name,
            "num_classes": model_service.num_classes,
            "class_names": model_service.class_names[:5]  # First 5 classes
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "Error switching model",
                "error_code": "MODEL_SWITCH_ERROR",
                "details": {"error": str(e)}
            }
        )


# ==========================================
# Root Endpoint (Welcome)
# ==========================================

@router.get(
    "/",
    summary="API Root",
    description="Welcome message and API info"
)
async def root():
    """
    **API Root endpoint**
    
    Returns welcome message and available endpoints
    """
    return {
        "message": "🎥 Video Action Recognition API",
        "version": settings.API_VERSION,
        "model": settings.MODEL_NAME,
        "endpoints": {
            "predict": "/api/v1/predict (POST)",
            "health": "/api/v1/health (GET)",
            "models": "/api/v1/models (GET)",
            "switch_model": "/api/v1/models/switch (POST)",
            "docs": "/docs (Swagger UI)",
            "redoc": "/redoc (ReDoc)"
        },
        "quick_start": "Upload video to /api/v1/predict to get action predictions"
    }
