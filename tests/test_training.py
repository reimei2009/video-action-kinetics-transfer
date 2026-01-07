"""
Unit tests for training scripts
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import patch, MagicMock, mock_open
import yaml
from pathlib import Path


@pytest.mark.unit
class TestTrainKinetics:
    """Test cases for train_kinetics.py"""
    
    def test_train_one_epoch_returns_metrics(self, mock_model, mock_dataloader):
        """Test train_one_epoch returns loss and accuracy"""
        from src.train_kinetics import train_one_epoch
        
        model = mock_model(num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        device = torch.device('cpu')
        
        loss, acc = train_one_epoch(
            model, mock_dataloader, criterion, optimizer, device, epoch=1
        )
        
        # Should return float values
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        
        # Dummy values from skeleton
        assert loss == 0.5
        assert acc == 75.0
    
    def test_evaluate_returns_metrics(self, mock_model, mock_dataloader):
        """Test evaluate returns validation metrics"""
        from src.train_kinetics import evaluate
        
        model = mock_model(num_classes=2)
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cpu')
        
        loss, acc = evaluate(model, mock_dataloader, criterion, device)
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss == 0.6
        assert acc == 70.0
    
    def test_main_loads_config(self, temp_dir, sample_config):
        """Test main function loads config correctly"""
        from src.train_kinetics import main
        
        # Create config file
        config_path = temp_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
        
        # Mock print to capture output
        with patch('builtins.print') as mock_print:
            main(str(config_path))
            
            # Verify config was loaded
            calls = [str(call) for call in mock_print.call_args_list]
            assert any('Config loaded' in str(call) for call in calls)
    
    def test_main_handles_missing_config(self, temp_dir):
        """Test main handles missing config file gracefully"""
        from src.train_kinetics import main
        
        fake_path = temp_dir / 'nonexistent.yaml'
        
        with patch('builtins.print') as mock_print:
            main(str(fake_path))
            
            # Should print warning
            calls = [str(call) for call in mock_print.call_args_list]
            assert any('not found' in str(call) for call in calls)
    
    @patch('torch.cuda.is_available')
    def test_main_detects_device(self, mock_cuda, temp_dir, sample_config):
        """Test device detection (CPU/CUDA)"""
        from src.train_kinetics import main
        
        config_path = temp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
        
        # Test CPU
        mock_cuda.return_value = False
        with patch('builtins.print') as mock_print:
            main(str(config_path))
            calls = [str(call) for call in mock_print.call_args_list]
            assert any('cpu' in str(call).lower() for call in calls)
        
        # Test CUDA
        mock_cuda.return_value = True
        with patch('builtins.print') as mock_print:
            main(str(config_path))
            calls = [str(call) for call in mock_print.call_args_list]
            assert any('cuda' in str(call).lower() for call in calls)


@pytest.mark.unit
class TestTrainNSAR:
    """Test cases for train_nsar.py"""
    
    def test_train_one_epoch_nsar(self, mock_model, mock_dataloader):
        """Test NSAR train_one_epoch"""
        from src.train_nsar import train_one_epoch
        
        model = mock_model(num_classes=8)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        device = torch.device('cpu')
        
        loss, acc = train_one_epoch(
            model, mock_dataloader, criterion, optimizer, device, epoch=1
        )
        
        # NSAR should return different dummy values
        assert loss == 0.4
        assert acc == 80.0
    
    def test_evaluate_nsar(self, mock_model, mock_dataloader):
        """Test NSAR evaluate"""
        from src.train_nsar import evaluate
        
        model = mock_model(num_classes=8)
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cpu')
        
        loss, acc = evaluate(model, mock_dataloader, criterion, device)
        
        assert loss == 0.5
        assert acc == 78.0
    
    def test_nsar_main_loads_config(self, temp_dir):
        """Test NSAR main loads config with transfer settings"""
        from src.train_nsar import main
        
        config = {
            'model_name': 'x3d_xs',
            'freeze_backbone': True,
            'kinetics_weights': '/path/to/weights.pth',
            'epochs': 3,
        }
        
        config_path = temp_dir / 'nsar_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with patch('builtins.print') as mock_print:
            main(str(config_path))
            
            calls = [str(call) for call in mock_print.call_args_list]
            # Should mention transfer learning
            assert any('Transfer' in str(call) for call in calls)
            assert any('Freeze' in str(call) or 'freeze' in str(call) for call in calls)


@pytest.mark.integration
class TestTrainingIntegration:
    """Integration tests for training pipeline"""
    
    def test_training_loop_completes(self, mock_model, mock_dataloader, sample_config):
        """Test complete training loop runs without errors"""
        from src.train_kinetics import train_one_epoch, evaluate
        
        model = mock_model(num_classes=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        device = torch.device('cpu')
        
        # Simulate 2 epochs
        for epoch in range(1, 3):
            train_loss, train_acc = train_one_epoch(
                model, mock_dataloader, criterion, optimizer, device, epoch
            )
            val_loss, val_acc = evaluate(
                model, mock_dataloader, criterion, device
            )
            
            # Metrics should be returned
            assert train_loss is not None
            assert val_acc is not None
    
    def test_config_parsing_all_fields(self, temp_dir):
        """Test config with all possible fields"""
        from src.train_kinetics import main
        
        config = {
            'model_name': 'x3d_xs',
            'num_frames': 16,
            'crop_size': 224,
            'batch_size': 4,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'epochs': 5,
            'clip_duration': 2.0,
            'selected_classes': ['basketball', 'swimming'],
            'data_root': '/kaggle/input/kinetics',
        }
        
        config_path = temp_dir / 'full_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Should load without errors
        with patch('builtins.print'):
            try:
                main(str(config_path))
                success = True
            except Exception as e:
                success = False
                pytest.fail(f"Config parsing failed: {e}")
        
        assert success


@pytest.mark.unit
class TestOptimizationComponents:
    """Test optimization components"""
    
    def test_loss_computation(self, mock_model, dummy_video_tensor, dummy_labels):
        """Test loss can be computed"""
        model = mock_model(num_classes=2)
        criterion = nn.CrossEntropyLoss()
        
        output = model(dummy_video_tensor)
        loss = criterion(output, dummy_labels)
        
        assert loss.item() >= 0
        assert isinstance(loss, torch.Tensor)
    
    def test_optimizer_step(self, mock_model):
        """Test optimizer can update parameters"""
        model = mock_model(num_classes=2)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Get initial params
        initial_params = [p.clone() for p in model.parameters()]
        
        # Forward + backward
        x = torch.randn(2, 3, 16, 224, 224)
        output = model(x)
        loss = output.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Params should be updated (at least one)
        params_changed = any(
            not torch.equal(p1, p2) 
            for p1, p2 in zip(initial_params, model.parameters())
        )
        assert params_changed


@pytest.mark.slow
class TestLongRunningTraining:
    """Tests that simulate longer training scenarios"""
    
    def test_multiple_epochs_training(self, mock_model, mock_dataloader):
        """Test training over multiple epochs"""
        from src.train_kinetics import train_one_epoch, evaluate
        
        model = mock_model(num_classes=5)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        device = torch.device('cpu')
        
        best_acc = 0.0
        
        # Train for 5 epochs
        for epoch in range(1, 6):
            train_loss, train_acc = train_one_epoch(
                model, mock_dataloader, criterion, optimizer, device, epoch
            )
            val_loss, val_acc = evaluate(
                model, mock_dataloader, criterion, device
            )
            
            # Track best accuracy (even if dummy)
            if val_acc > best_acc:
                best_acc = val_acc
        
        # Should have tracked some accuracy
        assert best_acc > 0
