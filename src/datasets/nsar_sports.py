"""
NSARPMD Sports Dataset Loader
National Sports Action Recognition Dataset
"""

import os
import torch
from torch.utils.data import Dataset
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose
import csv
import json


class NSARSportsDataset(Dataset):
    """
    Dataset loader cho NSARPMD (National Sports Action Recognition Dataset)
    
    Args:
        data_root: đường dẫn tới thư mục chứa video
        annotation_file: file csv/json chứa mapping video -> label
        clip_duration: độ dài clip (giây)
        num_frames: số frame mỗi clip
        split: 'train' hoặc 'val'
    """
    
    def __init__(
        self,
        data_root,
        annotation_file=None,
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
        
        # NSARPMD có các lớp sports
        self.sports_classes = [
            'basketball', 'soccer', 'tennis', 'volleyball',
            'badminton', 'cricket', 'hockey', 'swimming'
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.sports_classes)}
        
        # Load annotations
        self.samples = self._load_annotations(annotation_file)
        
    def _load_annotations(self, annotation_file):
        """
        Load annotations từ file CSV hoặc parse từ tên file
        Format: <sport_name>_<id>.mp4 hoặc CSV với columns [filename, label]
        """
        samples = []
        
        if annotation_file and os.path.exists(annotation_file):
            # Đọc từ CSV
            with open(annotation_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('split') == self.split or 'split' not in row:
                        video_path = os.path.join(self.data_root, row['filename'])
                        if os.path.exists(video_path):
                            samples.append({
                                'path': video_path,
                                'label': self.class_to_idx[row['label']],
                                'class': row['label']
                            })
        else:
            # Parse từ tên file (fallback)
            for video_name in os.listdir(self.data_root):
                if not video_name.endswith(('.mp4', '.avi', '.mkv')):
                    continue
                    
                # Thử parse: <sport>_<id>.mp4
                sport_name = video_name.split('_')[0].lower()
                if sport_name in self.class_to_idx:
                    video_path = os.path.join(self.data_root, video_name)
                    samples.append({
                        'path': video_path,
                        'label': self.class_to_idx[sport_name],
                        'class': sport_name
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
        
        # Load video
        video = EncodedVideo.from_path(video_path)
        
        # Sample clip từ video
        video_data = video.get_clip(start_sec=0, end_sec=self.clip_duration)
        video_tensor = video_data['video']  # (C, T, H, W)
        
        if self.transform:
            video_tensor = self.transform(video_tensor)
        
        return video_tensor, label


def get_nsar_transforms(num_frames=16, crop_size=224):
    """Transform cho NSARPMD (giống Kinetics)"""
    from pytorchvideo.transforms import (
        UniformTemporalSubsample,
        ShortSideScale,
        CenterCrop,
        Normalize,
    )
    
    return Compose([
        UniformTemporalSubsample(num_frames),
        ShortSideScale(size=256),
        CenterCrop(crop_size),
        Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])
