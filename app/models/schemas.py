"""
Pydantic Schemas for Request/Response Models

Định nghĩa cấu trúc JSON cho API endpoints:
- Validation: Tự động validate request data
- Documentation: FastAPI auto-gen Swagger UI
- Serialization: Python objects ↔ JSON
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime


# ==========================================
# Prediction Models
# ==========================================

class PredictionResult(BaseModel):
    """
    Một prediction result (1 label + confidence score)
    
    Example:
        {
            "label": "PlayingPiano",
            "confidence": 0.2100,
            "rank": 1
        }
    """
    label: str = Field(
        ..., 
        description="Tên class (action label)",
        examples=["PlayingPiano", "Basketball", "Diving"]
    )
    confidence: float = Field(
        ..., 
        description="Confidence score (0.0 - 1.0)",
        ge=0.0,  # >= 0
        le=1.0,  # <= 1.0
        examples=[0.21, 0.89, 0.15]
    )
    rank: int = Field(
        ..., 
        description="Ranking (1 = highest confidence)",
        ge=1,
        examples=[1, 2, 3]
    )
    
    class Config:
        # Example cho Swagger UI
        json_schema_extra = {
            "example": {
                "label": "PlayingPiano",
                "confidence": 0.2100,
                "rank": 1
            }
        }


class VideoMetadata(BaseModel):
    """
    Metadata của video đã upload
    
    Example:
        {
            "filename": "test_video.mp4",
            "duration": 12.88,
            "fps": 25,
            "size_mb": 2.5
        }
    """
    filename: str = Field(..., description="Tên file video")
    duration: Optional[float] = Field(None, description="Video duration (seconds)")
    fps: Optional[int] = Field(None, description="Frame per second")
    size_mb: Optional[float] = Field(None, description="File size (MB)")
    resolution: Optional[str] = Field(None, description="Video resolution", examples=["1920x1080"])


class PredictionResponse(BaseModel):
    """
    Response chính cho /predict endpoint
    
    Example:
        {
            "success": true,
            "message": "Prediction completed successfully",
            "predictions": [
                {"label": "PlayingPiano", "confidence": 0.21, "rank": 1},
                {"label": "Archery", "confidence": 0.09, "rank": 2}
            ],
            "model_name": "ucf101",
            "processing_time": 1.234,
            "video_metadata": {...},
            "timestamp": "2026-01-14T10:30:00"
        }
    """
    success: bool = Field(
        default=True, 
        description="Request thành công hay không"
    )
    message: str = Field(
        default="Prediction completed successfully",
        description="Message mô tả kết quả"
    )
    predictions: List[PredictionResult] = Field(
        ..., 
        description="Danh sách top-K predictions (sorted by confidence)",
        min_length=1
    )
    model_name: str = Field(
        ..., 
        description="Model name được sử dụng",
        examples=["kinetics", "ucf101"]
    )
    processing_time: float = Field(
        ..., 
        description="Thời gian xử lý (seconds)",
        examples=[1.234, 2.567]
    )
    video_metadata: Optional[VideoMetadata] = Field(
        None,
        description="Metadata của video"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp khi tạo response"
    )
    
    @field_validator('predictions')
    @classmethod
    def sort_predictions_by_confidence(cls, v: List[PredictionResult]) -> List[PredictionResult]:
        """Đảm bảo predictions được sort theo confidence giảm dần"""
        return sorted(v, key=lambda x: x.confidence, reverse=True)
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Prediction completed successfully",
                "predictions": [
                    {"label": "PlayingPiano", "confidence": 0.2100, "rank": 1},
                    {"label": "Archery", "confidence": 0.0886, "rank": 2},
                    {"label": "PlayingGuitar", "confidence": 0.0885, "rank": 3},
                    {"label": "Basketball", "confidence": 0.0879, "rank": 4},
                    {"label": "Drumming", "confidence": 0.0879, "rank": 5}
                ],
                "model_name": "ucf101",
                "processing_time": 1.234,
                "video_metadata": {
                    "filename": "test_video.mp4",
                    "duration": 12.88,
                    "fps": 25,
                    "size_mb": 2.5
                },
                "timestamp": "2026-01-14T10:30:00"
            }
        }


# ==========================================
# Error Response
# ==========================================

class ErrorResponse(BaseModel):
    """
    Response khi có lỗi
    
    Example:
        {
            "success": false,
            "message": "Invalid video file",
            "error_code": "INVALID_VIDEO",
            "details": {"reason": "File too large"}
        }
    """
    success: bool = Field(
        default=False,
        description="Request thất bại"
    )
    message: str = Field(
        ..., 
        description="Error message",
        examples=["Invalid video file", "Model not loaded", "Processing failed"]
    )
    error_code: Optional[str] = Field(
        None,
        description="Error code để client xử lý",
        examples=["INVALID_VIDEO", "MODEL_ERROR", "PROCESSING_ERROR"]
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Chi tiết lỗi (optional)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp khi lỗi xảy ra"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "message": "Invalid video file",
                "error_code": "INVALID_VIDEO",
                "details": {
                    "reason": "File size exceeds 100MB limit",
                    "max_size_mb": 100
                },
                "timestamp": "2026-01-14T10:30:00"
            }
        }


# ==========================================
# Health Check
# ==========================================

class HealthResponse(BaseModel):
    """
    Response cho /health endpoint
    
    Example:
        {
            "status": "healthy",
            "api_version": "1.0.0",
            "model_loaded": true,
            "model_name": "ucf101",
            "device": "cpu",
            "uptime_seconds": 3600
        }
    """
    status: str = Field(
        ..., 
        description="Service status",
        examples=["healthy", "unhealthy"]
    )
    api_version: str = Field(
        ..., 
        description="API version"
    )
    model_loaded: bool = Field(
        ..., 
        description="Model đã load thành công chưa"
    )
    model_name: str = Field(
        ..., 
        description="Model name đang sử dụng"
    )
    device: str = Field(
        ..., 
        description="Device (cpu/cuda)"
    )
    uptime_seconds: Optional[float] = Field(
        None,
        description="Server uptime (seconds)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "api_version": "1.0.0",
                "model_loaded": True,
                "model_name": "ucf101",
                "device": "cpu",
                "uptime_seconds": 3600.5,
                "timestamp": "2026-01-14T10:30:00"
            }
        }


# ==========================================
# Request Models (Optional)
# ==========================================

class PredictRequest(BaseModel):
    """
    Optional request body cho /predict
    (Thực tế dùng UploadFile từ FastAPI)
    
    Chỉ dùng để document optional parameters
    """
    top_k: Optional[int] = Field(
        5,
        description="Number of top predictions to return",
        ge=1,
        le=10
    )
    threshold: Optional[float] = Field(
        0.01,
        description="Minimum confidence threshold",
        ge=0.0,
        le=1.0
    )


# ==========================================
# Batch Prediction (Future Feature)
# ==========================================

class BatchPredictionResponse(BaseModel):
    """
    Response cho batch prediction (multiple videos)
    
    Future feature: Upload nhiều videos cùng lúc
    """
    success: bool = Field(default=True)
    message: str = Field(default="Batch prediction completed")
    results: List[PredictionResponse] = Field(
        ...,
        description="Danh sách predictions cho từng video"
    )
    total_videos: int = Field(..., description="Tổng số videos")
    failed_videos: int = Field(default=0, description="Số videos xử lý thất bại")
    total_processing_time: float = Field(..., description="Tổng thời gian (seconds)")


# ==========================================
# Model Info
# ==========================================

class ModelInfo(BaseModel):
    """
    Response cho /models endpoint (list available models)
    
    Example:
        {
            "available_models": ["kinetics", "ucf101"],
            "current_model": "ucf101",
            "model_details": {...}
        }
    """
    available_models: List[str] = Field(
        ...,
        description="Danh sách models có sẵn"
    )
    current_model: str = Field(
        ...,
        description="Model đang được sử dụng"
    )
    model_details: Dict[str, Any] = Field(
        ...,
        description="Chi tiết model (architecture, classes, etc.)"
    )


if __name__ == "__main__":
    # Test schemas
    print("Testing Pydantic Schemas...")
    
    # Test PredictionResult
    pred = PredictionResult(label="PlayingPiano", confidence=0.21, rank=1)
    print(f"✅ PredictionResult: {pred.model_dump_json(indent=2)}")
    
    # Test PredictionResponse
    response = PredictionResponse(
        predictions=[
            PredictionResult(label="PlayingPiano", confidence=0.21, rank=1),
            PredictionResult(label="Archery", confidence=0.09, rank=2)
        ],
        model_name="ucf101",
        processing_time=1.234
    )
    print(f"✅ PredictionResponse: {response.model_dump_json(indent=2)}")
    
    # Test ErrorResponse
    error = ErrorResponse(
        message="Invalid video file",
        error_code="INVALID_VIDEO"
    )
    print(f"✅ ErrorResponse: {error.model_dump_json(indent=2)}")
    
    print("\n✅ All schemas validated successfully!")
