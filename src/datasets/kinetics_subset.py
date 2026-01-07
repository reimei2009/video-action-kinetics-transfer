"""
Kinetics Subset Dataset Loader
Sử dụng cho pretrain/fine-tune trên Kinetics 5% (Kaggle)
"""

import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pytorchvideo.data.encoded_video import EncodedVideo
import json


class KineticsSubsetDataset(Dataset):
    """
    Dataset loader cho Kinetics subset (5%)
    
    Args:
        data_root: đường dẫn tới thư mục chứa video (train/<class>/<video>.mp4)
        selected_classes: list các class muốn train (10-20 classes)
        clip_duration: độ dài clip (giây)
        num_frames: số frame mỗi clip (16, 32...)
        split: 'train' hoặc 'val'
    """
    
    def __init__(
        self,
        data_root,
        selected_classes=None,
        clip_duration=2.0,
        num_frames=16,
        split='train',
        transform=None
    ):
        self.data_root = data_root
        self.clip_duration = clip_duration
        self.num_frames = num_frames
        self.split = split
        self.transform = transform
        
        # Nếu không chỉ định, chọn 10 classes phổ biến
        if selected_classes is None:
            self.selected_classes = [
                'abseiling', 'basketball', 'climbing', 'dancing',
                'playing_guitar', 'running', 'swimming', 'tennis',
                'volleyball', 'walking'
            ]
        else:
            self.selected_classes = selected_classes
            
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.selected_classes)}
        
        # Scan video files
        self.samples = self._scan_videos()
        
        if len(self.samples) == 0:
            print(f"\n⚠ WARNING: No videos found!")
            print(f"  Data root: {self.data_root}")
            print(f"  Split: {self.split}")
            print(f"  Split path: {os.path.join(self.data_root, self.split)}")
            print(f"  Path exists: {os.path.exists(self.data_root)}")
            if os.path.exists(self.data_root):
                print(f"  Contents: {os.listdir(self.data_root)[:10]}")
        
    def _scan_videos(self):
        """Quét tất cả video trong data_root/split/<class>/<video>.mp4"""
        samples = []
        split_path = os.path.join(self.data_root, self.split)
        
        if not os.path.exists(split_path):
            print(f"⚠ Split path not found: {split_path}")
            # Try without split subdirectory
            split_path = self.data_root
            print(f"  Trying root path: {split_path}")
        
        for class_name in self.selected_classes:
            class_path = os.path.join(split_path, class_name)
            if not os.path.exists(class_path):
                print(f"⚠ Class path not found: {class_path}")
                continue
                
            video_files = [f for f in os.listdir(class_path) if f.endswith(('.mp4', '.avi', '.mkv'))]
            print(f"  {class_name}: found {len(video_files)} videos")
            
            for video_name in video_files:
                video_path = os.path.join(class_path, video_name)
                samples.append({
                    'path': video_path,
                    'label': self.class_to_idx[class_name],
                    'class': class_name
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            video: tensor shape (C, T, H, W)
            label: int
        """
        sample = self.samples[idx]
        video_path = sample['path']
        label = sample['label']
        
        # Load video bằng PyTorchVideo
        video = EncodedVideo.from_path(video_path)
        
        # Lấy clip từ giây 0 đến clip_duration
        video_data = video.get_clip(start_sec=0, end_sec=self.clip_duration)
        video_tensor = video_data['video']  # shape: (C, T, H, W)
        
        if self.transform:
            video_tensor = self.transform(video_tensor)
        
        return video_tensor, label


def get_kinetics_transforms(num_frames=16, crop_size=224, mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]):
    """
    Transform chuẩn cho Kinetics dataset
    Compatible với torchvision (không dùng pytorchvideo transforms)
    """
    import torchvision.transforms as T
    
    # Simple transform pipeline using only torchvision
    # Video tensor shape: (C, T, H, W) or (T, C, H, W)
    return T.Compose([
        # Input sẽ là (C, T, H, W) từ EncodedVideo
        # Chuyển về (T, C, H, W) để dễ xử lý
        T.Lambda(lambda x: x.permute(1, 0, 2, 3)),  # (C,T,H,W) -> (T,C,H,W)
        
        # Temporal subsample: lấy num_frames frames đều đặn
        T.Lambda(lambda x: temporal_subsample(x, num_frames)),
        
        # Spatial transforms (áp dụng cho từng frame)
        T.Lambda(lambda x: torch.stack([
            T.Compose([
                T.Resize(256),
                T.CenterCrop(crop_size),
                T.Normalize(mean=mean, std=std)
            ])(frame) for frame in x
        ])),
        
        # Chuyển về (C, T, H, W) cho model
        T.Lambda(lambda x: x.permute(1, 0, 2, 3)),  # (T,C,H,W) -> (C,T,H,W)
    ])


def temporal_subsample(video_tensor, num_frames):
    """
    Uniformly subsample num_frames from video_tensor
    video_tensor shape: (T, C, H, W)
    """
    total_frames = video_tensor.shape[0]
    if total_frames <= num_frames:
        # Pad if not enough frames
        pad_size = num_frames - total_frames
        padding = video_tensor[-1:].repeat(pad_size, 1, 1, 1)
        return torch.cat([video_tensor, padding], dim=0)
    
    # Uniform sampling
    indices = torch.linspace(0, total_frames - 1, num_frames).long()
    return video_tensor[indices]
