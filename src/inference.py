"""
Inference script - Chạy dự đoán trên video (local)
"""

import argparse
import torch
from pathlib import Path


def predict_video(video_path, weights_path, class_names, device='cpu'):
    """
    Dự đoán action cho 1 video
    
    Args:
        video_path: đường dẫn tới file video
        weights_path: đường dẫn tới model weights (.pth)
        class_names: list tên các classes
        device: 'cpu' hoặc 'cuda'
    
    Returns:
        predictions: list of (class_name, score)
    """
    print(f"\n=== Video Action Prediction ===")
    print(f"Video: {video_path}")
    print(f"Weights: {weights_path}")
    print(f"Classes: {class_names}")
    print(f"Device: {device}")
    
    # Check files exist
    if not Path(video_path).exists():
        print(f"⚠ Video file not found: {video_path}")
        return None
    
    if not Path(weights_path).exists():
        print(f"⚠ Weights file not found: {weights_path}")
        print(f"  Download from Kaggle after training")
        return None
    
    # TODO: Load model
    print(f"✓ [TODO] Load X3D model with {len(class_names)} classes")
    
    # TODO: Load weights
    print(f"✓ [TODO] Load weights from {weights_path}")
    
    # TODO: Preprocess video
    print(f"✓ [TODO] Load and preprocess video frames")
    
    # TODO: Inference
    print(f"✓ [TODO] Run inference")
    
    # Dummy predictions
    predictions = [
        {'class': class_names[0], 'score': 0.85},
        {'class': class_names[1], 'score': 0.10},
        {'class': class_names[2], 'score': 0.05},
    ]
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Video Action Recognition Inference')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--model', type=str, help='Path to model weights (.pth)')
    parser.add_argument('--classes', type=str, help='Comma-separated class names')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or cuda')
    parser.add_argument('--top_k', type=int, default=3, help='Top-K predictions')
    
    args = parser.parse_args()
    
    if not args.video or not args.model or not args.classes:
        print("\n=== Inference Script (Skeleton) ===")
        print("Usage:")
        print("  python src/inference.py \\")
        print("    --video path/to/video.mp4 \\")
        print("    --model weights/x3d_nsar_best.pth \\")
        print("    --classes 'basketball,soccer,tennis' \\")
        print("    --device cpu")
        print("\nNote: Weights file will be available after Kaggle training")
        return
    
    # Parse class names
    class_names = [c.strip() for c in args.classes.split(',')]
    
    # Predict
    predictions = predict_video(
        video_path=args.video,
        weights_path=args.model,
        class_names=class_names,
        device=args.device
    )
    
    # Print results
    if predictions:
        print(f'\n=== Top-{args.top_k} Predictions ====')
        for i, pred in enumerate(predictions[:args.top_k], 1):
            print(f'{i}. {pred["class"]}: {pred["score"]:.4f} ({pred["score"]*100:.2f}%)')


if __name__ == '__main__':
    main()
