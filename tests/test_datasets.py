"""
Unit tests for dataset loaders
"""

import pytest
import torch
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import shutil


@pytest.mark.unit
class TestKineticsSubsetDataset:
    """Test cases for Kinetics subset dataset"""
    
    def test_dataset_initialization(self, temp_dir):
        """Test dataset can be initialized"""
        from src.datasets.kinetics_subset import KineticsSubsetDataset
        
        # Create dummy directory structure
        train_dir = temp_dir / 'train'
        train_dir.mkdir()
        
        selected_classes = ['basketball', 'swimming']
        for class_name in selected_classes:
            (train_dir / class_name).mkdir()
        
        dataset = KineticsSubsetDataset(
            data_root=str(temp_dir),
            selected_classes=selected_classes,
            split='train'
        )
        
        assert len(dataset.selected_classes) == 2
        assert dataset.class_to_idx == {'basketball': 0, 'swimming': 1}
    
    def test_dataset_default_classes(self, temp_dir):
        """Test dataset with default classes"""
        from src.datasets.kinetics_subset import KineticsSubsetDataset
        
        dataset = KineticsSubsetDataset(
            data_root=str(temp_dir),
            split='train'
        )
        
        # Should have 10 default classes
        assert len(dataset.selected_classes) == 10
        assert 'basketball' in dataset.selected_classes
    
    def test_dataset_scan_videos(self, temp_dir):
        """Test video scanning functionality"""
        from src.datasets.kinetics_subset import KineticsSubsetDataset
        
        # Create directory structure with dummy videos
        train_dir = temp_dir / 'train'
        train_dir.mkdir()
        
        basketball_dir = train_dir / 'basketball'
        basketball_dir.mkdir()
        
        # Create dummy video files
        (basketball_dir / 'video1.mp4').touch()
        (basketball_dir / 'video2.mp4').touch()
        (basketball_dir / 'video3.avi').touch()
        
        dataset = KineticsSubsetDataset(
            data_root=str(temp_dir),
            selected_classes=['basketball'],
            split='train'
        )
        
        # Should find 3 videos
        assert len(dataset.samples) == 3
        assert all(s['class'] == 'basketball' for s in dataset.samples)
        assert all(s['label'] == 0 for s in dataset.samples)
    
    def test_dataset_length(self, temp_dir):
        """Test __len__ method"""
        from src.datasets.kinetics_subset import KineticsSubsetDataset
        
        train_dir = temp_dir / 'train'
        train_dir.mkdir()
        
        for class_name in ['basketball', 'swimming']:
            class_dir = train_dir / class_name
            class_dir.mkdir()
            for i in range(5):
                (class_dir / f'video{i}.mp4').touch()
        
        dataset = KineticsSubsetDataset(
            data_root=str(temp_dir),
            selected_classes=['basketball', 'swimming'],
            split='train'
        )
        
        assert len(dataset) == 10  # 5 videos per class


@pytest.mark.unit
class TestNSARSportsDataset:
    """Test cases for NSAR Sports dataset"""
    
    def test_nsar_initialization(self, temp_dir):
        """Test NSAR dataset initialization"""
        from src.datasets.nsar_sports import NSARSportsDataset
        
        dataset = NSARSportsDataset(
            data_root=str(temp_dir),
            split='train'
        )
        
        # Should have 8 default sports classes
        assert len(dataset.sports_classes) == 8
        assert 'basketball' in dataset.sports_classes
        assert 'soccer' in dataset.sports_classes
    
    def test_nsar_with_annotation_file(self, temp_dir):
        """Test NSAR with CSV annotation file"""
        from src.datasets.nsar_sports import NSARSportsDataset
        import csv
        
        # Create dummy videos
        (temp_dir / 'basketball_01.mp4').touch()
        (temp_dir / 'soccer_02.mp4').touch()
        
        # Create annotation CSV
        csv_path = temp_dir / 'annotations.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'label', 'split'])
            writer.writeheader()
            writer.writerow({'filename': 'basketball_01.mp4', 'label': 'basketball', 'split': 'train'})
            writer.writerow({'filename': 'soccer_02.mp4', 'label': 'soccer', 'split': 'train'})
        
        dataset = NSARSportsDataset(
            data_root=str(temp_dir),
            annotation_file=str(csv_path),
            split='train'
        )
        
        assert len(dataset.samples) == 2
    
    def test_nsar_parse_from_filename(self, temp_dir):
        """Test parsing labels from filename"""
        from src.datasets.nsar_sports import NSARSportsDataset
        
        # Create dummy videos with sport name in filename
        (temp_dir / 'basketball_game1.mp4').touch()
        (temp_dir / 'tennis_match2.mp4').touch()
        (temp_dir / 'unknown_sport.mp4').touch()  # Should be ignored
        
        dataset = NSARSportsDataset(
            data_root=str(temp_dir),
            annotation_file=None,  # No CSV, parse from filename
            split='train'
        )
        
        # Should find 2 valid sports videos
        assert len(dataset.samples) == 2
        classes_found = [s['class'] for s in dataset.samples]
        assert 'basketball' in classes_found
        assert 'tennis' in classes_found


@pytest.mark.unit
class TestTransforms:
    """Test transform functions"""
    
    def test_kinetics_transforms(self):
        """Test Kinetics transform pipeline"""
        from src.datasets.kinetics_subset import get_kinetics_transforms
        
        transform = get_kinetics_transforms(
            num_frames=16,
            crop_size=224
        )
        
        # Create dummy video tensor (C, T, H, W)
        dummy_video = torch.randn(3, 32, 256, 256)
        
        # Apply transform
        transformed = transform(dummy_video)
        
        # Check output shape: (C, 16, 224, 224)
        assert transformed.shape[0] == 3  # Channels
        assert transformed.shape[1] == 16  # Frames
        assert transformed.shape[2] == 224  # Height
        assert transformed.shape[3] == 224  # Width
    
    def test_nsar_transforms(self):
        """Test NSAR transform pipeline"""
        from src.datasets.nsar_sports import get_nsar_transforms
        
        transform = get_nsar_transforms(
            num_frames=16,
            crop_size=224
        )
        
        dummy_video = torch.randn(3, 48, 320, 320)
        transformed = transform(dummy_video)
        
        # Should match target dimensions
        assert transformed.shape == (3, 16, 224, 224)


@pytest.mark.integration
class TestDatasetIntegration:
    """Integration tests for datasets"""
    
    def test_dataset_with_dataloader(self, temp_dir):
        """Test dataset works with PyTorch DataLoader"""
        from src.datasets.kinetics_subset import KineticsSubsetDataset
        from torch.utils.data import DataLoader
        
        # Create dummy data
        train_dir = temp_dir / 'train'
        train_dir.mkdir()
        class_dir = train_dir / 'basketball'
        class_dir.mkdir()
        
        for i in range(4):
            (class_dir / f'video{i}.mp4').touch()
        
        dataset = KineticsSubsetDataset(
            data_root=str(temp_dir),
            selected_classes=['basketball'],
            split='train'
        )
        
        # Create DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Should be able to get length
        assert len(loader) == 2  # 4 samples / batch_size 2
    
    def test_multiple_classes_labeling(self, temp_dir):
        """Test correct labeling with multiple classes"""
        from src.datasets.kinetics_subset import KineticsSubsetDataset
        
        train_dir = temp_dir / 'train'
        train_dir.mkdir()
        
        classes = ['basketball', 'swimming', 'tennis']
        for idx, class_name in enumerate(classes):
            class_dir = train_dir / class_name
            class_dir.mkdir()
            (class_dir / 'video.mp4').touch()
        
        dataset = KineticsSubsetDataset(
            data_root=str(temp_dir),
            selected_classes=classes,
            split='train'
        )
        
        # Check labels are assigned correctly
        labels = [s['label'] for s in dataset.samples]
        assert set(labels) == {0, 1, 2}
        
        # Check class mapping
        assert dataset.class_to_idx['basketball'] == 0
        assert dataset.class_to_idx['swimming'] == 1
        assert dataset.class_to_idx['tennis'] == 2
