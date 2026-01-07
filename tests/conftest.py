"""
Pytest configuration and shared fixtures
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Temporary directory for test files"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def dummy_video_tensor():
    """Dummy video tensor (B, C, T, H, W)"""
    return torch.randn(2, 3, 16, 224, 224)


@pytest.fixture
def dummy_labels():
    """Dummy labels for classification"""
    return torch.tensor([0, 1])


@pytest.fixture
def sample_config():
    """Sample training config"""
    return {
        'model_name': 'x3d_xs',
        'num_frames': 16,
        'crop_size': 224,
        'batch_size': 2,
        'learning_rate': 0.001,
        'epochs': 2,
        'clip_duration': 2.0,
    }


@pytest.fixture
def mock_model():
    """Mock X3D model"""
    class MockX3D(torch.nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.fc = torch.nn.Linear(192, num_classes)
            
        def forward(self, x):
            # x: (B, C, T, H, W)
            B = x.size(0)
            # Simulate pooling
            out = torch.randn(B, 192)
            return self.fc(out)
    
    return MockX3D


@pytest.fixture
def mock_dataloader(dummy_video_tensor, dummy_labels):
    """Mock DataLoader"""
    class MockDataLoader:
        def __init__(self):
            self.dataset_size = 4
            
        def __iter__(self):
            for _ in range(2):  # 2 batches
                yield dummy_video_tensor, dummy_labels
                
        def __len__(self):
            return 2
    
    return MockDataLoader()


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility"""
    torch.manual_seed(42)
    np.random.seed(42)
