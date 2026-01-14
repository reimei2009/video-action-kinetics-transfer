"""
Configuration Settings for Video Action Recognition API

Quản lý tất cả cấu hình: model paths, classes, server settings
Có thể override bằng environment variables (.env file)
"""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings
    
    Tất cả settings có thể override qua environment variables
    Ví dụ: export MODEL_NAME="kinetics" hoặc tạo file .env
    """
    
    # ==========================================
    # Project Paths
    # ==========================================
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    WEIGHTS_DIR: Path = PROJECT_ROOT / "weights"
    
    # ==========================================
    # Model Configuration
    # ==========================================
    MODEL_NAME: str = "ucf101"  # "kinetics" hoặc "ucf101"
    
    # Weights file paths
    KINETICS_WEIGHTS: Path = WEIGHTS_DIR / "x3d_kinetics_subset_best.pth"
    UCF101_WEIGHTS: Path = WEIGHTS_DIR / "x3d_ucf101_best.pth"
    
    # X3D Model Architecture
    X3D_VERSION: str = "xs"  # xs, s, m, l
    NUM_CLASSES: int = 10    # 10 classes cho cả Kinetics và UCF101
    
    # ==========================================
    # Class Names
    # ==========================================
    # Kinetics-400 subset classes (10 classes)
    KINETICS_CLASSES: List[str] = [
        "playing piano",
        "playing guitar", 
        "playing drums",
        "playing violin",
        "playing keyboard",
        "playing saxophone",
        "playing trumpet",
        "playing flute",
        "playing cello",
        "playing clarinet"
    ]
    
    # UCF101 selected classes (10 classes)
    UCF101_CLASSES: List[str] = [
        "Archery",
        "Basketball", 
        "Biking",
        "Diving",
        "Drumming",
        "GolfSwing",
        "HorseRiding",
        "PlayingGuitar",
        "PlayingPiano",
        "TennisSwing"
    ]
    
    # ==========================================
    # Video Processing Settings
    # ==========================================
    VIDEO_CLIP_DURATION: float = 2.0    # Lấy 2 giây từ video (khớp với training)
    VIDEO_FPS: int = 25                 # Frame per second
    VIDEO_NUM_FRAMES: int = 16          # X3D input: 16 frames
    VIDEO_RESIZE: int = 256             # Resize shorter side to 256
    VIDEO_CROP_SIZE: int = 224          # Center crop 224x224
    
    # Mean & Std cho normalization (Kinetics-400 dataset)
    VIDEO_MEAN: List[float] = [0.45, 0.45, 0.45]
    VIDEO_STD: List[float] = [0.225, 0.225, 0.225]
    
    # ==========================================
    # Inference Settings
    # ==========================================
    TOP_K: int = 5                      # Trả về top-5 predictions
    CONFIDENCE_THRESHOLD: float = 0.01  # Min confidence để hiển thị
    BATCH_SIZE: int = 1                 # Process 1 video at a time
    
    # Device
    DEVICE: str = "cpu"                 # "cpu" hoặc "cuda" (auto-detect)
    
    # ==========================================
    # API Server Settings
    # ==========================================
    API_TITLE: str = "Video Action Recognition API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = """
    🎥 **Video Action Recognition API** sử dụng X3D model
    
    **Features:**
    - Upload video → Predict top-K action labels
    - Support 2 models: Kinetics-400 subset, UCF101
    - REST API + Gradio UI
    
    **Model Info:**
    - Architecture: X3D (Facebook Research)
    - Input: 16 frames × 224×224 pixels
    - Output: Top-5 action labels with confidence scores
    """
    
    # Server host & port
    HOST: str = "0.0.0.0"               # Bind to all interfaces
    PORT: int = 8000                    # Default port
    RELOAD: bool = True                 # Auto-reload khi code thay đổi (dev only)
    
    # CORS (Cross-Origin Resource Sharing)
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",        # React frontend
        "http://localhost:8000",        # FastAPI docs
        "http://127.0.0.1:8000",
    ]
    
    # ==========================================
    # Upload Settings
    # ==========================================
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100 MB
    ALLOWED_VIDEO_EXTENSIONS: List[str] = [
        ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"
    ]
    TEMP_UPLOAD_DIR: Path = PROJECT_ROOT / "temp_uploads"
    
    # ==========================================
    # Logging
    # ==========================================
    LOG_LEVEL: str = "INFO"             # DEBUG, INFO, WARNING, ERROR
    LOG_FILE: Path = PROJECT_ROOT / "logs" / "api.log"
    
    # ==========================================
    # Model Config Loading (from SettingsConfigDict)
    # ==========================================
    model_config = SettingsConfigDict(
        env_file=".env",                # Đọc từ file .env nếu có
        env_file_encoding="utf-8",
        case_sensitive=False,           # Không phân biệt hoa/thường
        extra="ignore"                  # Bỏ qua env vars không định nghĩa
    )
    
    # ==========================================
    # Computed Properties
    # ==========================================
    @property
    def weights_path(self) -> Path:
        """Trả về weights path dựa trên MODEL_NAME"""
        if self.MODEL_NAME == "kinetics":
            return self.KINETICS_WEIGHTS
        elif self.MODEL_NAME == "ucf101":
            return self.UCF101_WEIGHTS
        else:
            raise ValueError(f"Invalid MODEL_NAME: {self.MODEL_NAME}")
    
    @property
    def class_names(self) -> List[str]:
        """Trả về class names dựa trên MODEL_NAME"""
        if self.MODEL_NAME == "kinetics":
            return self.KINETICS_CLASSES
        elif self.MODEL_NAME == "ucf101":
            return self.UCF101_CLASSES
        else:
            raise ValueError(f"Invalid MODEL_NAME: {self.MODEL_NAME}")
    
    def validate_paths(self) -> None:
        """Kiểm tra các paths có tồn tại không"""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
        
        # Tạo temp upload dir nếu chưa có
        self.TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Tạo logs dir nếu chưa có
        self.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


# ==========================================
# Global Settings Instance
# ==========================================
settings = Settings()

# Validate paths khi import module
settings.validate_paths()


# ==========================================
# Helper Functions
# ==========================================
def get_settings() -> Settings:
    """
    Dependency injection cho FastAPI routes
    
    Usage trong routes:
        @app.get("/config")
        def get_config(settings: Settings = Depends(get_settings)):
            return settings
    """
    return settings


def print_settings():
    """Debug: In ra tất cả settings"""
    print("=" * 60)
    print("🔧 Video Action Recognition API - Configuration")
    print("=" * 60)
    print(f"Model Name:      {settings.MODEL_NAME}")
    print(f"Weights Path:    {settings.weights_path}")
    print(f"Num Classes:     {settings.NUM_CLASSES}")
    print(f"Class Names:     {', '.join(settings.class_names[:3])}...")
    print(f"Device:          {settings.DEVICE}")
    print(f"Server:          http://{settings.HOST}:{settings.PORT}")
    print(f"Clip Duration:   {settings.VIDEO_CLIP_DURATION}s")
    print(f"Top-K:           {settings.TOP_K}")
    print("=" * 60)


if __name__ == "__main__":
    # Test config
    print_settings()
