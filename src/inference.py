"""
Inference script - Chạy dự đoán trên video (local)
"""

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from pytorchvideo.transforms import (
    UniformTemporalSubsample,
    ShortSideScale,
    CenterCrop,
    Normalize,
)
from torchvision.transforms import Compose

from models.x3d_wrapper import create_x3d_model


class VideoActionPredictor:
    """
    Predictor cho video action recognition
    """
    
    def __init__(
        self,
        model_path,
        class_names,
        model_name='x3d_xs',
        num_frames=16,
        crop_size=224,
        device='cpu'
    ):
        self.device = torch.device(device)
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.class_names = class_names
        
        # Load model
        print(f'Loading model from {model_path}...')
        self.model = create_x3d_model(
            model_name=model_name,
            num_classes=len(class_names),
            pretrained=False
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        print('✓ Model loaded')
        
        # Transform
        self.transform = Compose([
            UniformTemporalSubsample(num_frames),
            ShortSideScale(size=256),
            CenterCrop(crop_size),
            Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        ])
    
    def preprocess_video(self, video_path, max_frames=None):
        """
        Đọc video và preprocess thành tensor (C, T, H, W)
        
        Args:
            video_path: đường dẫn video
            max_frames: số frame tối đa để đọc (None = đọc hết)
        
        Returns:
            video_tensor: shape (C, T, H, W)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_count += 1
            
            if max_frames and frame_count >= max_frames:
                break
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f'Could not read video: {video_path}')
        
        # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
        video = np.stack(frames, axis=0)  # (T, H, W, C)
        video = torch.from_numpy(video).permute(3, 0, 1, 2).float()  # (C, T, H, W)
        
        # Normalize to [0, 1]
        video = video / 255.0
        
        return video
    
    def predict(self, video_path, top_k=3):
        """
        Dự đoán action cho 1 video
        
        Args:
            video_path: đường dẫn video
            top_k: số lớp top có xác suất cao nhất
        
        Returns:
            predictions: list of (class_name, score)
        """
        # Load & preprocess video
        print(f'Processing video: {video_path}')
        video_tensor = self.preprocess_video(video_path)
        
        # Apply transform
        video_tensor = self.transform(video_tensor)
        
        # Add batch dimension: (C, T, H, W) -> (1, C, T, H, W)
        video_tensor = video_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(video_tensor)
            probs = torch.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.class_names)))
        
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            predictions.append({
                'class': self.class_names[idx.item()],
                'score': prob.item()
            })
        
        return predictions
    
    def predict_batch(self, video_paths, top_k=3):
        """Dự đoán batch nhiều video"""
        results = {}
        for video_path in video_paths:
            try:
                preds = self.predict(video_path, top_k=top_k)
                results[video_path] = preds
            except Exception as e:
                print(f'Error processing {video_path}: {e}')
                results[video_path] = None
        return results


def main():
    parser = argparse.ArgumentParser(description='Video Action Recognition Inference')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights (.pth)')
    parser.add_argument('--classes', type=str, required=True, help='Comma-separated class names')
    parser.add_argument('--model_name', type=str, default='x3d_xs', help='Model architecture')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or cuda')
    parser.add_argument('--top_k', type=int, default=3, help='Top-K predictions')
    
    args = parser.parse_args()
    
    # Parse class names
    class_names = [c.strip() for c in args.classes.split(',')]
    
    # Create predictor
    predictor = VideoActionPredictor(
        model_path=args.model,
        class_names=class_names,
        model_name=args.model_name,
        device=args.device
    )
    
    # Predict
    predictions = predictor.predict(args.video, top_k=args.top_k)
    
    # Print results
    print(f'\n=== Predictions for {args.video} ===')
    for i, pred in enumerate(predictions, 1):
        print(f'{i}. {pred["class"]}: {pred["score"]:.4f} ({pred["score"]*100:.2f}%)')


if __name__ == '__main__':
    main()
