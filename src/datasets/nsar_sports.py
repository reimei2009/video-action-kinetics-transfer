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
        transform=None,
        auto_detect_classes=True
    ):
        self.data_root = data_root
        self.clip_duration = clip_duration
        self.num_frames = num_frames
        self.split = split
        self.transform = transform
        self.auto_detect_classes = auto_detect_classes
        
        # NSARPMD có các lớp sports (default)
        self.sports_classes = [
            'basketball', 'soccer', 'tennis', 'volleyball',
            'badminton', 'cricket', 'hockey', 'swimming'
        ]
        
        # Load annotations và auto-detect classes
        self.samples = self._load_annotations(annotation_file)
        
        # Update class mapping sau khi load
        if self.auto_detect_classes and len(self.samples) > 0:
            detected_classes = sorted(list(set([s['class'] for s in self.samples])))
            if detected_classes != self.sports_classes:
                print(f"  Auto-detected classes: {detected_classes}")
                self.sports_classes = detected_classes
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.sports_classes)}
        
    def _load_annotations(self, annotation_file):
        """
        Load annotations từ file CSV hoặc parse từ cấu trúc thư mục
        Hỗ trợ nhiều formats:
        - CSV: [filename, label]
        - Folder structure: data_root/<class_name>/*.mp4
        - Flat structure: data_root/<class>_<id>.mp4
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
            # Try folder structure: data_root/<class_name>/*.mp4
            if os.path.isdir(self.data_root):
                print(f"  Scanning dataset structure: {self.data_root}")
                
                # Check if data_root has subdirectories
                subdirs = [d for d in os.listdir(self.data_root) 
                          if os.path.isdir(os.path.join(self.data_root, d))]
                
                if subdirs:
                    # Folder structure - mỗi subfolder là 1 class
                    print(f"  Found {len(subdirs)} subdirectories")
                    
                    # Collect all videos từ mọi subfolder
                    for subdir in subdirs:
                        class_dir = os.path.join(self.data_root, subdir)
                        
                        # Recursively tìm videos trong subfolder
                        video_files = []
                        for root, dirs, files in os.walk(class_dir):
                            for f in files:
                                if f.endswith(('.mp4', '.avi', '.mkv', '.MP4', '.AVI')):
                                    video_files.append(os.path.join(root, f))
                        
                        if video_files:
                            # Sử dụng subfolder name làm class
                            class_name = subdir.strip().replace(' ', '_').lower()
                            print(f"    {subdir}: {len(video_files)} videos -> class '{class_name}'")
                            
                            for video_path in video_files:
                                samples.append({
                                    'path': video_path,
                                    'label': -1,  # Will be reassigned after auto-detect
                                    'class': class_name
                                })
                else:
                    # Flat structure: <sport>_<id>.mp4
                    print(f"  Flat structure detected")
                    video_files = [f for f in os.listdir(self.data_root) 
                                  if f.endswith(('.mp4', '.avi', '.mkv', '.MP4'))]
                    print(f"  Found {len(video_files)} video files")
                    
                    for video_name in video_files:
                        # Thử parse: <sport>_<id>.mp4
                        sport_name = video_name.split('_')[0].lower()
                        if sport_name in self.class_to_idx:
                            video_path = os.path.join(self.data_root, video_name)
                            samples.append({
                                'path': video_path,
                                'label': self.class_to_idx[sport_name],
                                'class': sport_name
                            })
        
        if len(samples) == 0:
            print(f"  ⚠ WARNING: No videos found!")
            print(f"  Please check:")
            print(f"    - data_root exists: {os.path.exists(self.data_root)}")
            if os.path.exists(self.data_root):
                contents = os.listdir(self.data_root)[:10]
                print(f"    - Contents (first 10): {contents}")
        
        else:
            # Reassign labels after auto-detection
            detected_classes = sorted(list(set([s['class'] for s in samples])))
            class_to_idx_new = {cls: idx for idx, cls in enumerate(detected_classes)}
            for sample in samples:
                sample['label'] = class_to_idx_new[sample['class']]
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
