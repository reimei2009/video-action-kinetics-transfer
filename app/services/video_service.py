"""
Video Service - Xử lý video upload và preprocessing

Chức năng:
- Save video từ request
- Extract metadata (duration, FPS, resolution)
- Preprocess video → tensor cho model
- Cleanup temp files
"""

import torch
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import time
import os
import sys

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from pytorchvideo.data.encoded_video import EncodedVideo
from datasets.kinetics_subset import get_kinetics_transforms
from app.core.config import settings


class VideoService:
    """
    Service xử lý video
    
    Usage:
        video_service = VideoService()
        video_tensor, metadata = video_service.process_video(video_bytes)
    """
    
    def __init__(self):
        """Khởi tạo service"""
        # Ensure temp upload dir exists
        settings.TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✓ Video Service initialized (temp dir: {settings.TEMP_UPLOAD_DIR})")
    
    def save_uploaded_file(
        self, 
        file_content: bytes, 
        filename: str
    ) -> Path:
        """
        Lưu video từ request vào temp folder
        
        Args:
            file_content: Video file bytes
            filename: Original filename
        
        Returns:
            Path to saved file
        """
        try:
            # Generate unique filename với timestamp
            timestamp = int(time.time() * 1000)
            safe_filename = f"{timestamp}_{filename}"
            file_path = settings.TEMP_UPLOAD_DIR / safe_filename
            
            # Write to disk
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            file_size_mb = len(file_content) / (1024 * 1024)
            print(f"✓ Saved video: {file_path} ({file_size_mb:.2f} MB)")
            
            return file_path
        
        except Exception as e:
            print(f"❌ Error saving file: {e}")
            raise
    
    def extract_metadata(
        self, 
        video_path: Path
    ) -> Dict[str, Any]:
        """
        Extract video metadata
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dict với duration, fps, size_mb, resolution (nếu có)
        """
        try:
            # Load video
            video = EncodedVideo.from_path(str(video_path))
            
            # Get metadata
            metadata = {
                "filename": video_path.name,
                "duration": round(video.duration, 2),
                "size_mb": round(video_path.stat().st_size / (1024 * 1024), 2)
            }
            
            # Try to get FPS and resolution (không phải tất cả video đều có)
            try:
                if hasattr(video, '_video_reader') and video._video_reader:
                    reader = video._video_reader
                    if hasattr(reader, 'get_metadata'):
                        meta = reader.get_metadata()
                        if 'video' in meta and 'fps' in meta['video']:
                            metadata['fps'] = int(meta['video']['fps'][0])
            except:
                pass  # FPS không lấy được thì thôi
            
            print(f"✓ Metadata: {metadata}")
            return metadata
        
        except Exception as e:
            print(f"❌ Error extracting metadata: {e}")
            return {
                "filename": video_path.name,
                "duration": None,
                "fps": None,
                "size_mb": None
            }
    
    def preprocess_video(
        self, 
        video_path: Path,
        clip_duration: float = None,
        num_frames: int = None,
        crop_size: int = None
    ) -> torch.Tensor:
        """
        Preprocess video → tensor cho model
        
        Pipeline:
        1. Load video với EncodedVideo
        2. Sample clip (clip_duration seconds) từ giữa video
        3. Apply transforms (resize, crop, normalize)
        4. Return tensor shape (1, 3, 16, 224, 224)
        
        Args:
            video_path: Path to video file
            clip_duration: Clip duration (default: settings.VIDEO_CLIP_DURATION)
            num_frames: Number of frames (default: settings.VIDEO_NUM_FRAMES)
            crop_size: Crop size (default: settings.VIDEO_CROP_SIZE)
        
        Returns:
            Video tensor shape (1, C, T, H, W)
        """
        if clip_duration is None:
            clip_duration = settings.VIDEO_CLIP_DURATION
        if num_frames is None:
            num_frames = settings.VIDEO_NUM_FRAMES
        if crop_size is None:
            crop_size = settings.VIDEO_CROP_SIZE
        
        try:
            start_time = time.time()
            
            # Step 1: Load video
            video = EncodedVideo.from_path(str(video_path))
            video_duration = video.duration
            
            # Step 2: Sample clip từ giữa video
            start_sec = max(0, (video_duration - clip_duration) / 2)
            end_sec = min(video_duration, start_sec + clip_duration)
            
            print(f"✓ Loading video clip: {start_sec:.2f}s - {end_sec:.2f}s (total: {video_duration:.2f}s)")
            
            # Get clip
            video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
            video_tensor = video_data['video']  # Shape: (C, T, H, W)
            
            print(f"  Raw video shape: {video_tensor.shape}")
            
            # Step 3: Apply transforms
            transform = get_kinetics_transforms(
                num_frames=num_frames, 
                crop_size=crop_size
            )
            video_tensor = transform(video_tensor)  # Shape: (C, T, H, W)
            
            # Step 4: Add batch dimension
            video_tensor = video_tensor.unsqueeze(0)  # Shape: (1, C, T, H, W)
            
            preprocess_time = time.time() - start_time
            print(f"✓ Preprocessed shape: {video_tensor.shape} (time: {preprocess_time:.3f}s)")
            
            return video_tensor
        
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            raise
    
    def process_video(
        self, 
        file_content: bytes, 
        filename: str
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Full pipeline: Save → Extract metadata → Preprocess
        
        Args:
            file_content: Video bytes từ request
            filename: Original filename
        
        Returns:
            (video_tensor, metadata)
            - video_tensor: Shape (1, 3, 16, 224, 224)
            - metadata: Dict với duration, fps, size_mb, etc.
        """
        video_path = None
        
        try:
            # Step 1: Save file
            video_path = self.save_uploaded_file(file_content, filename)
            
            # Step 2: Extract metadata
            metadata = self.extract_metadata(video_path)
            
            # Step 3: Preprocess
            video_tensor = self.preprocess_video(video_path)
            
            return video_tensor, metadata
        
        except Exception as e:
            print(f"❌ Video processing error: {e}")
            raise
        
        finally:
            # Step 4: Cleanup (xóa temp file)
            if video_path and video_path.exists():
                try:
                    video_path.unlink()
                    print(f"✓ Cleaned up temp file: {video_path.name}")
                except Exception as e:
                    print(f"⚠ Warning: Could not delete temp file: {e}")
    
    def validate_video_file(
        self, 
        filename: str, 
        file_size: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate video file trước khi xử lý
        
        Args:
            filename: Original filename
            file_size: File size in bytes
        
        Returns:
            (is_valid, error_message)
        """
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in settings.ALLOWED_VIDEO_EXTENSIONS:
            return False, f"Invalid file type. Allowed: {', '.join(settings.ALLOWED_VIDEO_EXTENSIONS)}"
        
        # Check file size
        if file_size > settings.MAX_UPLOAD_SIZE:
            max_mb = settings.MAX_UPLOAD_SIZE / (1024 * 1024)
            current_mb = file_size / (1024 * 1024)
            return False, f"File too large: {current_mb:.1f}MB (max: {max_mb:.0f}MB)"
        
        return True, None
    
    def cleanup_old_files(self, max_age_hours: int = 1):
        """
        Xóa các temp files cũ hơn max_age_hours
        
        Args:
            max_age_hours: Delete files older than this (hours)
        """
        try:
            current_time = time.time()
            deleted_count = 0
            
            for file_path in settings.TEMP_UPLOAD_DIR.glob("*"):
                if file_path.is_file():
                    file_age_hours = (current_time - file_path.stat().st_mtime) / 3600
                    
                    if file_age_hours > max_age_hours:
                        file_path.unlink()
                        deleted_count += 1
            
            if deleted_count > 0:
                print(f"✓ Cleaned up {deleted_count} old temp files")
        
        except Exception as e:
            print(f"⚠ Warning: Cleanup error: {e}")


# ==========================================
# Global Singleton Instance
# ==========================================
video_service = VideoService()


# ==========================================
# Helper Functions
# ==========================================

def get_video_service() -> VideoService:
    """
    Dependency injection cho FastAPI
    
    Usage:
        @app.post("/predict")
        def predict(service: VideoService = Depends(get_video_service)):
            return service.process_video(...)
    """
    return video_service


if __name__ == "__main__":
    """Test video service"""
    print("\n" + "=" * 60)
    print("Testing Video Service...")
    print("=" * 60)
    
    # Test validation
    print("\nTest 1: Validate video file")
    is_valid, error = video_service.validate_video_file("test.mp4", 10 * 1024 * 1024)
    print(f"  test.mp4 (10MB): {'✓ Valid' if is_valid else f'✗ Invalid - {error}'}")
    
    is_valid, error = video_service.validate_video_file("test.txt", 1024)
    print(f"  test.txt: {'✓ Valid' if is_valid else f'✗ Invalid - {error}'}")
    
    is_valid, error = video_service.validate_video_file("huge.mp4", 200 * 1024 * 1024)
    print(f"  huge.mp4 (200MB): {'✓ Valid' if is_valid else f'✗ Invalid - {error}'}")
    
    # Test với video thật (nếu có)
    test_video_path = project_root / "4088191-hd_1920_1080_25fps.mp4"
    if test_video_path.exists():
        print(f"\nTest 2: Process real video")
        print(f"  Video: {test_video_path.name}")
        
        # Read file
        with open(test_video_path, 'rb') as f:
            file_content = f.read()
        
        # Process
        video_tensor, metadata = video_service.process_video(
            file_content, 
            test_video_path.name
        )
        
        print(f"\n  Result:")
        print(f"    Tensor shape: {video_tensor.shape}")
        print(f"    Metadata: {metadata}")
    
    print("\n✅ Video Service test completed!")
