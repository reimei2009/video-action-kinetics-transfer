"""
UCF101 Dataset Loader
UCF-101: 101 action classes with 13,320 videos
"""

import os
import torch
from torch.utils.data import Dataset
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose
import csv
import json


class UCF101Dataset(Dataset):
    """
    Dataset loader cho UCF101 (101 action classes)
    Structure: /kaggle/input/ucf101-action-recognition/UCF-101/<class_name>/<video>.avi
    
    Args:
        data_root: đường dẫn tới thư mục UCF-101
        annotation_file: file csv/json chứa mapping video -> label (optional)
        clip_duration: độ dài clip (giây)
        num_frames: số frame mỗi clip
        split: 'train' hoặc 'val'
        selected_classes: list classes muốn dùng (None = all 101 classes)
    """
    
    def __init__(
        self,
        data_root,
        annotation_file=None,
        clip_duration=2.0,
        num_frames=16,
        split='train',
        transform=None,
        selected_classes=None,
        train_val_split=0.8
    ):
        self.data_root = data_root
        self.clip_duration = clip_duration
        self.num_frames = num_frames
        self.split = split
        self.transform = transform
        self.selected_classes = selected_classes
        self.train_val_split = train_val_split
        
        # Load dataset structure
        self.samples = self._load_ucf101()
        
        # Build class list từ detected samples
        detected_classes = sorted(list(set([s['class'] for s in self.samples])))
        self.class_names = detected_classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
    def _load_ucf101(self):
        """
        Load UCF101 dataset
        Structure options:
        1. data_root/train/<class_name>/<video>.avi (pre-split)
        2. data_root/UCF-101/<class_name>/<video>.avi (single folder)
        3. data_root/<class_name>/<video>.avi (direct)
        """
        samples = []
        
        # Check for train/val/test split structure first
        train_path = os.path.join(self.data_root, 'train')
        val_path = os.path.join(self.data_root, 'val')
        ucf_path = os.path.join(self.data_root, 'UCF-101')
        
        if os.path.exists(train_path) and os.path.exists(val_path):
            # Pre-split structure
            search_root = train_path if self.split == 'train' else val_path
            print(f"  Using pre-split structure: {os.path.basename(search_root)}/")
        elif os.path.exists(ucf_path):
            # UCF-101 subfolder
            search_root = ucf_path
            print(f"  Found UCF-101 subfolder")
        else:
            # Direct structure
            search_root = self.data_root
            print(f"  Using root folder")
        
        # Scan class folders
        if not os.path.isdir(search_root):
            print(f"  ❌ Path not found: {search_root}")
            return samples
        
        class_folders = [d for d in os.listdir(search_root)
                        if os.path.isdir(os.path.join(search_root, d))]
        
        if not class_folders:
            print(f"  ❌ No class folders found in {search_root}")
            return samples
        
        # Filter selected classes
        if self.selected_classes:
            class_folders = [c for c in class_folders if c in self.selected_classes]
            print(f"  Using {len(class_folders)} selected classes")
        else:
            print(f"  Found {len(class_folders)} total classes")
        
        # Load videos from each class
        for class_name in sorted(class_folders):
            class_dir = os.path.join(search_root, class_name)
            video_files = [f for f in os.listdir(class_dir)
                          if f.endswith(('.avi', '.mp4', '.AVI', '.MP4'))]
            
            if video_files:
                # For pre-split structure, use all videos (already split)
                # For single folder structure, do manual split
                if os.path.exists(train_path) and os.path.exists(val_path):
                    # Pre-split: use all
                    split_files = video_files
                else:
                    # Manual split
                    split_idx = int(len(video_files) * self.train_val_split)
                    if self.split == 'train':
                        split_files = video_files[:split_idx]
                    else:
                        split_files = video_files[split_idx:]
                
                print(f"    {class_name}: {len(split_files)} videos")
                
                for video_file in split_files:
                    video_path = os.path.join(class_dir, video_file)
                    samples.append({
                        'path': video_path,
                        'label': -1,  # Will be assigned later
                        'class': class_name
                    })
        
        # Assign labels
        if samples:
            detected_classes = sorted(list(set([s['class'] for s in samples])))
            class_to_idx_temp = {cls: idx for idx, cls in enumerate(detected_classes)}
            for sample in samples:
                sample['label'] = class_to_idx_temp[sample['class']]
            print(f"  ✓ Total {len(samples)} videos loaded for {self.split}")
        else:
            print(f"  ⚠ No videos found!")
        
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
        
        # Load video
        video = EncodedVideo.from_path(video_path)
        
        # Sample clip từ video
        video_data = video.get_clip(start_sec=0, end_sec=self.clip_duration)
        video_tensor = video_data['video']  # (C, T, H, W)
        
        if self.transform:
            video_tensor = self.transform(video_tensor)
        
        return video_tensor, label


def get_ucf101_transforms(num_frames=16, crop_size=224):
    """Transform cho UCF101 (tương tự Kinetics)"""
    import torchvision.transforms as T
    
    return T.Compose([
        # (C,T,H,W) -> (T,C,H,W)
        T.Lambda(lambda x: x.permute(1, 0, 2, 3)),
        
        # Temporal subsample
        T.Lambda(lambda x: temporal_subsample(x, num_frames)),
        
        # Spatial transforms
        T.Lambda(lambda x: torch.stack([
            T.Compose([
                T.Resize(256),
                T.CenterCrop(crop_size),
                T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
            ])(frame) for frame in x
        ])),
        
        # (T,C,H,W) -> (C,T,H,W)
        T.Lambda(lambda x: x.permute(1, 0, 2, 3)),
    ])


def temporal_subsample(video_tensor, num_frames):
    """Subsample video to num_frames uniformly"""
    T_in = video_tensor.shape[0]
    if T_in <= num_frames:
        return video_tensor
    indices = torch.linspace(0, T_in - 1, num_frames).long()
    return video_tensor[indices]
