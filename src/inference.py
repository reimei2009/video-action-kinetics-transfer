"""
Inference script - Chạy dự đoán trên video (local)
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Add src to path for imports
src_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(src_dir))

from models.x3d_wrapper import build_x3d
from datasets.kinetics_subset import get_kinetics_transforms
from pytorchvideo.data.encoded_video import EncodedVideo


def load_model(weights_path, num_classes, device='cpu'):
    """Load model from checkpoint"""
    print(f"Loading model...")
    
    # Build model architecture
    model = build_x3d(
        num_classes=num_classes,
        model_name='x3d_xs',
        pretrained=False
    )
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'val_acc' in checkpoint:
            print(f"  Validation Acc: {checkpoint['val_acc']:.2f}%")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def load_and_preprocess_video(video_path, num_frames=16, crop_size=224):
    """Load video and apply transforms"""
    print(f"Loading video: {video_path}")
    
    # Load video
    video = EncodedVideo.from_path(video_path)
    
    # Get video duration and sample a clip from the middle
    video_duration = video.duration
    start_sec = max(0, (video_duration - 2.0) / 2)  # 2-second clip from middle
    end_sec = start_sec + 2.0
    
    # Get video clip
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
    video_tensor = video_data['video']  # Shape: (C, T, H, W)
    
    print(f"  Video shape: {video_tensor.shape}")
    print(f"  Duration: {video_duration:.2f}s, Using clip: {start_sec:.2f}s - {end_sec:.2f}s")
    
    # Apply transforms
    transform = get_kinetics_transforms(num_frames=num_frames, crop_size=crop_size)
    video_tensor = transform(video_tensor)
    
    # Add batch dimension: (1, C, T, H, W)
    video_tensor = video_tensor.unsqueeze(0)
    
    print(f"✓ Preprocessed shape: {video_tensor.shape}")
    
    return video_tensor


def predict_video(video_path, weights_path, class_names, device='cpu', num_frames=16, crop_size=224):
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
    print(f"\n{'='*60}")
    print(f"=== Video Action Recognition ===")
    print(f"{'='*60}\n")
    print(f"Video: {video_path}")
    print(f"Weights: {weights_path}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Device: {device}\n")
    
    # Check files exist
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return None
    
    if not Path(weights_path).exists():
        print(f"❌ Weights file not found: {weights_path}")
        return None
    
    # Load model
    model = load_model(weights_path, len(class_names), device)
    
    # Load and preprocess video
    video_tensor = load_and_preprocess_video(video_path, num_frames, crop_size)
    video_tensor = video_tensor.to(device)
    
    # Inference
    print(f"\nRunning inference...")
    with torch.no_grad():
        outputs = model(video_tensor)
        probabilities = F.softmax(outputs, dim=1)
    
    # Get predictions
    probs = probabilities[0].cpu().numpy()
    predictions = [
        {'class': class_names[i], 'score': float(probs[i])}
        for i in range(len(class_names))
    ]
    
    # Sort by score
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description='Video Action Recognition Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default usage with 10 Kinetics classes
  python src/inference.py --video 4088191-hd_1920_1080_25fps.mp4 --weights weights/x3d_kinetics_subset_best.pth
  
  # Custom classes
  python src/inference.py --video test.mp4 --weights weights/model.pth --classes "basketball,soccer,tennis"
  
  # Use GPU
  python src/inference.py --video test.mp4 --weights weights/model.pth --device cuda
        """
    )
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights (.pth)')
    parser.add_argument('--classes', type=str, 
                       default='playing guitar,playing violin,cooking on campfire,playing basketball,bench pressing,doing aerobics,playing piano,skipping rope,dribbling basketball,washing dishes',
                       help='Comma-separated class names (default: 10 Kinetics classes)')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or cuda')
    parser.add_argument('--top_k', type=int, default=5, help='Show top-K predictions')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames to sample')
    parser.add_argument('--crop_size', type=int, default=224, help='Spatial crop size')
    
    args = parser.parse_args()
    
    # Parse class names
    class_names = [c.strip() for c in args.classes.split(',')]
    
    # Predict
    predictions = predict_video(
        video_path=args.video,
        weights_path=args.weights,
        class_names=class_names,
        device=args.device,
        num_frames=args.num_frames,
        crop_size=args.crop_size
    )
    
    # Print results
    if predictions:
        print(f'\n{"="*60}')
        print(f'=== Top-{args.top_k} Predictions ===')
        print(f'{"="*60}\n')
        for i, pred in enumerate(predictions[:args.top_k], 1):
            bar_length = int(pred["score"] * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            print(f'{i}. {pred["class"]:30s} {pred["score"]*100:5.2f}% {bar}')
        print()


if __name__ == '__main__':
    main()
    main()
