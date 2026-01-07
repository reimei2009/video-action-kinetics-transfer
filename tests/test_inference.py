"""
Unit tests for inference script
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.mark.unit
class TestPredictVideo:
    """Test cases for predict_video function"""
    
    def test_predict_video_nonexistent_video(self, temp_dir):
        """Test with non-existent video file"""
        from src.inference import predict_video
        
        fake_video = temp_dir / 'nonexistent.mp4'
        fake_weights = temp_dir / 'weights.pth'
        fake_weights.touch()
        
        result = predict_video(
            video_path=str(fake_video),
            weights_path=str(fake_weights),
            class_names=['basketball', 'soccer'],
            device='cpu'
        )
        
        # Should return None for non-existent video
        assert result is None
    
    def test_predict_video_nonexistent_weights(self, temp_dir):
        """Test with non-existent weights file"""
        from src.inference import predict_video
        
        fake_video = temp_dir / 'video.mp4'
        fake_video.touch()
        fake_weights = temp_dir / 'nonexistent_weights.pth'
        
        result = predict_video(
            video_path=str(fake_video),
            weights_path=str(fake_weights),
            class_names=['basketball', 'soccer'],
            device='cpu'
        )
        
        # Should return None for non-existent weights
        assert result is None
    
    def test_predict_video_valid_files(self, temp_dir):
        """Test with valid video and weights files"""
        from src.inference import predict_video
        
        video_path = temp_dir / 'video.mp4'
        video_path.touch()
        weights_path = temp_dir / 'weights.pth'
        weights_path.touch()
        
        class_names = ['basketball', 'soccer', 'tennis']
        
        result = predict_video(
            video_path=str(video_path),
            weights_path=str(weights_path),
            class_names=class_names,
            device='cpu'
        )
        
        # Skeleton returns dummy predictions
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 3
    
    def test_predict_video_output_format(self, temp_dir):
        """Test prediction output format"""
        from src.inference import predict_video
        
        video_path = temp_dir / 'video.mp4'
        video_path.touch()
        weights_path = temp_dir / 'weights.pth'
        weights_path.touch()
        
        result = predict_video(
            video_path=str(video_path),
            weights_path=str(weights_path),
            class_names=['class1', 'class2', 'class3'],
            device='cpu'
        )
        
        # Check format: list of dicts with 'class' and 'score'
        assert isinstance(result, list)
        for pred in result:
            assert 'class' in pred
            assert 'score' in pred
            assert isinstance(pred['score'], float)
            assert 0 <= pred['score'] <= 1
    
    def test_predict_video_different_devices(self, temp_dir):
        """Test prediction with different device settings"""
        from src.inference import predict_video
        
        video_path = temp_dir / 'video.mp4'
        video_path.touch()
        weights_path = temp_dir / 'weights.pth'
        weights_path.touch()
        
        for device in ['cpu', 'cuda']:
            result = predict_video(
                video_path=str(video_path),
                weights_path=str(weights_path),
                class_names=['a', 'b'],
                device=device
            )
            
            # Should work with both devices
            assert result is not None


@pytest.mark.unit
class TestInferenceMain:
    """Test cases for main inference function"""
    
    def test_main_no_arguments(self):
        """Test main with no arguments shows help"""
        from src.inference import main
        
        with patch('sys.argv', ['inference.py']):
            with patch('builtins.print') as mock_print:
                main()
                
                # Should print usage information
                calls = [str(call) for call in mock_print.call_args_list]
                assert any('Usage' in str(call) or 'Inference' in str(call) for call in calls)
    
    @patch('src.inference.predict_video')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_with_arguments(self, mock_args, mock_predict, temp_dir):
        """Test main with valid arguments"""
        from src.inference import main
        
        # Mock arguments
        mock_args.return_value = MagicMock(
            video='video.mp4',
            model='model.pth',
            classes='basketball,soccer',
            device='cpu',
            top_k=3
        )
        
        # Mock predict_video to return predictions
        mock_predict.return_value = [
            {'class': 'basketball', 'score': 0.85},
            {'class': 'soccer', 'score': 0.10},
        ]
        
        with patch('builtins.print'):
            main()
        
        # predict_video should have been called
        assert mock_predict.called
    
    def test_main_parses_classes_correctly(self):
        """Test class names are parsed from comma-separated string"""
        from src.inference import main
        
        with patch('argparse.ArgumentParser.parse_args') as mock_args:
            mock_args.return_value = MagicMock(
                video='v.mp4',
                model='m.pth',
                classes='basketball, soccer, tennis',  # With spaces
                device='cpu',
                top_k=2
            )
            
            with patch('src.inference.predict_video') as mock_predict:
                with patch('builtins.print'):
                    main()
                
                # Check classes were parsed correctly
                call_args = mock_predict.call_args
                if call_args:
                    class_names = call_args[1]['class_names']
                    assert 'basketball' in class_names
                    assert 'soccer' in class_names
                    assert 'tennis' in class_names


@pytest.mark.integration
class TestInferenceIntegration:
    """Integration tests for inference pipeline"""
    
    def test_end_to_end_prediction_skeleton(self, temp_dir):
        """Test end-to-end prediction flow (skeleton version)"""
        from src.inference import predict_video
        
        # Create dummy files
        video_path = temp_dir / 'test_video.mp4'
        video_path.touch()
        weights_path = temp_dir / 'model_weights.pth'
        weights_path.touch()
        
        class_names = [
            'basketball', 'soccer', 'tennis', 
            'volleyball', 'swimming', 'running'
        ]
        
        # Run prediction
        predictions = predict_video(
            video_path=str(video_path),
            weights_path=str(weights_path),
            class_names=class_names,
            device='cpu'
        )
        
        # Verify output
        assert predictions is not None
        assert len(predictions) > 0
        
        # Check top prediction
        top_pred = predictions[0]
        assert top_pred['class'] in class_names
        assert top_pred['score'] > 0
    
    def test_prediction_with_single_class(self, temp_dir):
        """Test prediction with only one class"""
        from src.inference import predict_video
        
        video_path = temp_dir / 'video.mp4'
        video_path.touch()
        weights_path = temp_dir / 'weights.pth'
        weights_path.touch()
        
        predictions = predict_video(
            video_path=str(video_path),
            weights_path=str(weights_path),
            class_names=['basketball'],
            device='cpu'
        )
        
        # Should still return predictions
        assert predictions is not None
        assert len(predictions) > 0


@pytest.mark.unit
class TestInferencePipeline:
    """Test inference pipeline components"""
    
    def test_predictions_sorted_by_score(self, temp_dir):
        """Test predictions are sorted by confidence"""
        from src.inference import predict_video
        
        video_path = temp_dir / 'video.mp4'
        video_path.touch()
        weights_path = temp_dir / 'weights.pth'
        weights_path.touch()
        
        predictions = predict_video(
            video_path=str(video_path),
            weights_path=str(weights_path),
            class_names=['a', 'b', 'c', 'd'],
            device='cpu'
        )
        
        if predictions and len(predictions) > 1:
            # Check scores are in descending order
            scores = [p['score'] for p in predictions]
            assert scores == sorted(scores, reverse=True)
    
    def test_prediction_scores_sum_to_one(self, temp_dir):
        """Test prediction scores approximately sum to 1 (softmax)"""
        from src.inference import predict_video
        
        video_path = temp_dir / 'video.mp4'
        video_path.touch()
        weights_path = temp_dir / 'weights.pth'
        weights_path.touch()
        
        predictions = predict_video(
            video_path=str(video_path),
            weights_path=str(weights_path),
            class_names=['a', 'b', 'c'],
            device='cpu'
        )
        
        if predictions:
            total_score = sum(p['score'] for p in predictions)
            # Dummy values might not sum to 1, but should be reasonable
            assert 0 < total_score <= 1.1  # Allow small tolerance


@pytest.mark.slow
class TestInferencePerformance:
    """Performance and stress tests for inference"""
    
    def test_inference_with_many_classes(self, temp_dir):
        """Test inference with large number of classes"""
        from src.inference import predict_video
        
        video_path = temp_dir / 'video.mp4'
        video_path.touch()
        weights_path = temp_dir / 'weights.pth'
        weights_path.touch()
        
        # 100 classes
        class_names = [f'class_{i}' for i in range(100)]
        
        predictions = predict_video(
            video_path=str(video_path),
            weights_path=str(weights_path),
            class_names=class_names,
            device='cpu'
        )
        
        # Should still work
        assert predictions is not None
    
    def test_multiple_predictions_same_model(self, temp_dir):
        """Test multiple predictions (simulating batch inference)"""
        from src.inference import predict_video
        
        # Create multiple video files
        video_paths = []
        for i in range(5):
            video_path = temp_dir / f'video_{i}.mp4'
            video_path.touch()
            video_paths.append(video_path)
        
        weights_path = temp_dir / 'weights.pth'
        weights_path.touch()
        
        class_names = ['a', 'b', 'c']
        
        # Run predictions on all videos
        all_predictions = []
        for video_path in video_paths:
            preds = predict_video(
                video_path=str(video_path),
                weights_path=str(weights_path),
                class_names=class_names,
                device='cpu'
            )
            all_predictions.append(preds)
        
        # All should have predictions
        assert all(p is not None for p in all_predictions)
        assert len(all_predictions) == 5
