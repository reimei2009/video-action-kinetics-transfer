"""
Model Service - Load X3D model và thực hiện inference

Singleton service để:
- Load model 1 lần duy nhất khi start server
- Tái sử dụng model cho tất cả requests
- Tách biệt business logic khỏi API routes
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import time
from typing import List, Tuple
import sys

# Add src to path để import models
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from models.x3d_wrapper import build_x3d
from app.core.config import settings


class ModelService:
    """
    Service quản lý X3D model
    
    Pattern: Singleton - Chỉ tạo 1 instance duy nhất
    Usage:
        model_service = ModelService()
        predictions = model_service.predict(video_tensor)
    """
    
    def __init__(self):
        """
        Khởi tạo service:
        1. Auto-detect device (CPU/CUDA)
        2. Load X3D model từ weights
        3. Set model to eval mode
        """
        print("=" * 60)
        print("🚀 Initializing Model Service...")
        print("=" * 60)
        
        # Step 1: Detect device
        self.device = self._get_device()
        
        # Step 2: Load model
        self.model = None
        self.model_name = settings.MODEL_NAME
        self.class_names = settings.class_names
        self.num_classes = settings.NUM_CLASSES
        
        self._load_model()
        
        print("=" * 60)
        print("✅ Model Service initialized successfully!")
        print("=" * 60)
    
    def _get_device(self) -> torch.device:
        """
        Auto-detect best device
        
        Priority: CUDA > MPS (Mac M1) > CPU
        """
        if settings.DEVICE != "cpu" and torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✓ Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print("✓ Using MPS device (Apple Silicon)")
        else:
            device = torch.device('cpu')
            print("✓ Using CPU device")
        
        return device
    
    def _load_model(self):
        """
        Load X3D model từ weights file
        
        Steps:
        1. Build X3D architecture
        2. Load pretrained weights
        3. Move to device
        4. Set eval mode (disable dropout, batchnorm)
        """
        try:
            weights_path = settings.weights_path
            print(f"Loading model: {self.model_name}")
            print(f"Weights path: {weights_path}")
            print(f"Num classes: {self.num_classes}")
            
            # Step 1: Build architecture
            self.model = build_x3d(
                num_classes=self.num_classes,
                model_name='x3d_xs',  # X3D-XS architecture
                pretrained=False
            )
            
            # Step 2: Load weights
            checkpoint = torch.load(weights_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                # Training checkpoint format
                self.model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', 'unknown')
                val_acc = checkpoint.get('val_acc', 0)
                print(f"✓ Loaded checkpoint from epoch {epoch}")
                print(f"  Validation Accuracy: {val_acc:.2f}%")
            else:
                # Raw state dict
                self.model.load_state_dict(checkpoint)
                print("✓ Loaded raw state dict")
            
            # Step 3: Move to device
            self.model = self.model.to(self.device)
            
            # Step 4: Set eval mode
            self.model.eval()
            
            print(f"✓ Model loaded successfully!")
            print(f"  Classes: {', '.join(self.class_names[:3])}... ({len(self.class_names)} total)")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict(
        self, 
        video_tensor: torch.Tensor, 
        top_k: int = None
    ) -> List[Tuple[str, float]]:
        """
        Predict top-K actions cho video
        
        Args:
            video_tensor: Shape (1, C, T, H, W) - preprocessed video
            top_k: Number of top predictions (default: settings.TOP_K)
        
        Returns:
            List of (label, confidence) sorted by confidence descending
            Example: [("PlayingPiano", 0.21), ("Archery", 0.09), ...]
        """
        if top_k is None:
            top_k = settings.TOP_K
        
        try:
            # Move to device
            video_tensor = video_tensor.to(self.device)
            
            # Inference (no gradient computation)
            with torch.no_grad():
                start_time = time.time()
                
                # Forward pass
                logits = self.model(video_tensor)  # Shape: (1, num_classes)
                
                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=1)  # Shape: (1, num_classes)
                
                inference_time = time.time() - start_time
            
            # Get top-K predictions
            top_probs, top_indices = torch.topk(probs, k=min(top_k, self.num_classes), dim=1)
            
            # Convert to list of (label, confidence)
            predictions = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                label = self.class_names[idx.item()]
                confidence = prob.item()
                predictions.append((label, confidence))
            
            print(f"✓ Inference completed in {inference_time:.3f}s")
            
            return predictions
        
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            raise
    
    def predict_batch(
        self, 
        video_tensors: torch.Tensor, 
        top_k: int = None
    ) -> List[List[Tuple[str, float]]]:
        """
        Predict cho batch of videos (future feature)
        
        Args:
            video_tensors: Shape (B, C, T, H, W)
            top_k: Number of top predictions
        
        Returns:
            List of predictions for each video
        """
        if top_k is None:
            top_k = settings.TOP_K
        
        try:
            video_tensors = video_tensors.to(self.device)
            
            with torch.no_grad():
                logits = self.model(video_tensors)  # (B, num_classes)
                probs = F.softmax(logits, dim=1)
            
            batch_predictions = []
            for i in range(video_tensors.shape[0]):
                top_probs, top_indices = torch.topk(probs[i], k=min(top_k, self.num_classes))
                
                predictions = [
                    (self.class_names[idx.item()], prob.item())
                    for prob, idx in zip(top_probs, top_indices)
                ]
                batch_predictions.append(predictions)
            
            return batch_predictions
        
        except Exception as e:
            print(f"❌ Batch prediction error: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """
        Lấy thông tin model
        
        Returns:
            Dict chứa model metadata
        """
        return {
            "model_name": self.model_name,
            "architecture": "X3D-XS",
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "device": str(self.device),
            "weights_path": str(settings.weights_path),
            "input_shape": "(1, 3, 16, 224, 224)",  # (B, C, T, H, W)
        }
    
    def reload_model(self, model_name: str = None):
        """
        Reload model (switch between kinetics/ucf101)
        
        Args:
            model_name: "kinetics" hoặc "ucf101"
        """
        if model_name and model_name != self.model_name:
            # Update settings
            settings.MODEL_NAME = model_name
            self.model_name = model_name
            self.class_names = settings.class_names
            
            print(f"Switching to model: {model_name}")
            self._load_model()
        else:
            print(f"Reloading current model: {self.model_name}")
            self._load_model()


# ==========================================
# Global Singleton Instance
# ==========================================
# Tạo 1 instance duy nhất khi import module
# Tất cả routes sẽ dùng chung instance này

print("\n" + "=" * 60)
print("🔄 Creating global ModelService instance...")
print("=" * 60)

model_service = ModelService()

print("✅ Global model_service ready!")
print("=" * 60 + "\n")


# ==========================================
# Helper Functions
# ==========================================

def get_model_service() -> ModelService:
    """
    Dependency injection cho FastAPI
    
    Usage trong routes:
        @app.post("/predict")
        def predict(service: ModelService = Depends(get_model_service)):
            return service.predict(...)
    """
    return model_service


if __name__ == "__main__":
    """Test model service"""
    print("\n" + "=" * 60)
    print("Testing Model Service...")
    print("=" * 60)
    
    # Print model info
    info = model_service.get_model_info()
    print("\nModel Info:")
    for key, value in info.items():
        if key == "class_names":
            print(f"  {key}: {value[:3]}... ({len(value)} total)")
        else:
            print(f"  {key}: {value}")
    
    # Test prediction with dummy data
    print("\nTesting prediction with dummy video tensor...")
    dummy_video = torch.randn(1, 3, 16, 224, 224)  # (B, C, T, H, W)
    predictions = model_service.predict(dummy_video, top_k=5)
    
    print("\nTop-5 Predictions:")
    for i, (label, conf) in enumerate(predictions, 1):
        print(f"  {i}. {label}: {conf:.4f}")
    
    print("\n✅ Model Service test completed!")
