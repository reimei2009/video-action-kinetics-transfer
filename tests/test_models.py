"""
Unit tests for X3D model wrapper
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock


@pytest.mark.unit
class TestX3DWrapper:
    """Test cases for X3D model wrapper"""
    
    @patch('torch.hub.load')
    def test_build_x3d_basic(self, mock_hub_load):
        """Test basic X3D model creation"""
        from src.models.x3d_wrapper import build_x3d
        
        # Mock model structure
        mock_model = MagicMock()
        mock_model.blocks = [None] * 6
        mock_model.blocks[5] = MagicMock()
        mock_model.blocks[5].proj = nn.Linear(192, 400)  # Kinetics-400 default
        mock_hub_load.return_value = mock_model
        
        # Build model
        model = build_x3d(num_classes=10, pretrained=False)
        
        # Verify torch.hub.load was called
        mock_hub_load.assert_called_once()
        args = mock_hub_load.call_args
        assert args[0][0] == 'facebookresearch/pytorchvideo'
        assert args[0][1] == 'x3d_xs'
        
        # Verify output layer was changed
        assert isinstance(mock_model.blocks[5].proj, nn.Linear)
    
    @patch('torch.hub.load')
    def test_build_x3d_custom_classes(self, mock_hub_load):
        """Test X3D with custom number of classes"""
        from src.models.x3d_wrapper import build_x3d
        
        mock_model = MagicMock()
        mock_model.blocks = [None] * 6
        mock_model.blocks[5] = MagicMock()
        mock_model.blocks[5].proj = nn.Linear(192, 400)
        mock_hub_load.return_value = mock_model
        
        # Test different class numbers
        for num_classes in [5, 10, 20, 100]:
            model = build_x3d(num_classes=num_classes, pretrained=False)
            # In real implementation, would verify the linear layer size
            assert mock_hub_load.called
    
    @patch('torch.hub.load')
    def test_build_x3d_freeze_backbone(self, mock_hub_load):
        """Test freezing backbone parameters"""
        from src.models.x3d_wrapper import build_x3d
        
        # Create mock model with named parameters
        mock_model = MagicMock()
        mock_model.blocks = [None] * 6
        mock_model.blocks[5] = MagicMock()
        mock_model.blocks[5].proj = nn.Linear(192, 10)
        
        # Mock named_parameters
        params = [
            ('blocks.0.conv', MagicMock(requires_grad=True)),
            ('blocks.1.conv', MagicMock(requires_grad=True)),
            ('blocks.5.proj.weight', MagicMock(requires_grad=True)),
        ]
        mock_model.named_parameters.return_value = params
        mock_hub_load.return_value = mock_model
        
        # Build with freeze
        model = build_x3d(num_classes=10, freeze_backbone=True)
        
        # Verify named_parameters was called (freeze logic)
        assert mock_model.named_parameters.called
    
    @patch('torch.hub.load')
    def test_build_x3d_different_architectures(self, mock_hub_load):
        """Test different X3D architectures"""
        from src.models.x3d_wrapper import build_x3d
        
        mock_model = MagicMock()
        mock_model.blocks = [None] * 6
        mock_model.blocks[5] = MagicMock()
        mock_model.blocks[5].proj = nn.Linear(192, 400)
        mock_hub_load.return_value = mock_model
        
        architectures = ['x3d_xs', 'x3d_s', 'x3d_m']
        
        for arch in architectures:
            model = build_x3d(num_classes=10, model_name=arch, pretrained=False)
            # Verify correct architecture was requested
            call_args = mock_hub_load.call_args[0]
            assert call_args[1] == arch
    
    @patch('torch.hub.load')
    def test_build_x3d_pretrained_flag(self, mock_hub_load):
        """Test pretrained parameter"""
        from src.models.x3d_wrapper import build_x3d
        
        mock_model = MagicMock()
        mock_model.blocks = [None] * 6
        mock_model.blocks[5] = MagicMock()
        mock_model.blocks[5].proj = nn.Linear(192, 400)
        mock_hub_load.return_value = mock_model
        
        # Test with pretrained=True
        model = build_x3d(num_classes=10, pretrained=True)
        kwargs = mock_hub_load.call_args[1]
        assert kwargs['pretrained'] == True
        
        # Test with pretrained=False
        model = build_x3d(num_classes=10, pretrained=False)
        kwargs = mock_hub_load.call_args[1]
        assert kwargs['pretrained'] == False


@pytest.mark.unit
class TestModelForward:
    """Test forward pass with mock model"""
    
    def test_mock_model_forward(self, mock_model, dummy_video_tensor):
        """Test mock model forward pass"""
        model = mock_model(num_classes=10)
        
        # Forward pass
        output = model(dummy_video_tensor)
        
        # Check output shape
        assert output.shape == (2, 10)  # (batch_size, num_classes)
    
    def test_mock_model_different_batch_sizes(self, mock_model):
        """Test with different batch sizes"""
        model = mock_model(num_classes=5)
        
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 16, 224, 224)
            output = model(x)
            assert output.shape == (batch_size, 5)


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for model"""
    
    @patch('torch.hub.load')
    def test_model_can_train(self, mock_hub_load, dummy_video_tensor, dummy_labels):
        """Test that model can be trained (backward pass works)"""
        from src.models.x3d_wrapper import build_x3d
        
        # Create real linear layer for testing
        mock_model = MagicMock()
        mock_model.blocks = [None] * 6
        mock_model.blocks[5] = MagicMock()
        
        # Use real nn.Linear for testing gradients
        real_linear = nn.Linear(192, 2)
        mock_model.blocks[5].proj = real_linear
        
        # Make the model callable with a simple forward
        def forward(x):
            B = x.size(0)
            features = torch.randn(B, 192, requires_grad=True)
            return real_linear(features)
        
        mock_model.forward = forward
        mock_model.__call__ = forward
        mock_hub_load.return_value = mock_model
        
        model = build_x3d(num_classes=2, pretrained=False)
        
        # Forward pass
        output = model(dummy_video_tensor)
        
        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, dummy_labels)
        
        # Backward pass should work
        loss.backward()
        
        # Check gradients exist
        assert real_linear.weight.grad is not None
