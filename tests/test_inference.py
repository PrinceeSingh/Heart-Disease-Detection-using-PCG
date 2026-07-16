"""Unit tests for inference module."""

import pytest
import torch
import numpy as np
from src.inference import PCGPredictor
from src.model import PCGClassifier


class TestPCGPredictor:
    """Test PCGPredictor inference wrapper."""
    
    @pytest.fixture
    def model_path(self, tmp_path):
        """Create and save a dummy model."""
        model = PCGClassifier()
        path = str(tmp_path / "test_model.pt")
        torch.save(model.state_dict(), path)
        return path
    
    @pytest.fixture
    def predictor(self, model_path):
        """Create predictor instance."""
        return PCGPredictor(model_path, threshold=0.5, device='cpu')
    
    def test_predictor_initialization(self, model_path):
        """Test predictor can be initialized."""
        predictor = PCGPredictor(model_path, device='cpu')
        assert predictor is not None
        assert predictor.threshold == 0.5
    
    def test_predict_numpy(self, predictor):
        """Test prediction on numpy array."""
        spectrogram = np.random.randn(64, 251).astype(np.float32)
        pred, prob = predictor.predict(spectrogram, return_probabilities=True)
        
        assert pred in [0, 1]
        assert 0 <= prob <= 1
    
    def test_predict_torch(self, predictor):
        """Test prediction on torch tensor."""
        spectrogram = torch.randn(1, 64, 251)
        pred, prob = predictor.predict(spectrogram, return_probabilities=True)
        
        assert pred in [0, 1]
        assert 0 <= prob <= 1
    
    def test_predict_batch(self, predictor):
        """Test batch prediction."""
        spectrograms = np.random.randn(8, 64, 251).astype(np.float32)
        preds, probs = predictor.predict_batch(spectrograms)
        
        assert preds.shape == (8,)
        assert probs.shape == (8,)
        assert np.all((preds == 0) | (preds == 1))
        assert np.all((probs >= 0) & (probs <= 1))
    
    def test_threshold_update(self, predictor):
        """Test threshold can be updated."""
        assert predictor.threshold == 0.5
        predictor.set_threshold(0.7)
        assert predictor.threshold == 0.7
    
    def test_threshold_validation(self, predictor):
        """Test invalid threshold raises error."""
        with pytest.raises(AssertionError):
            predictor.set_threshold(1.5)  # Out of range
        
        with pytest.raises(AssertionError):
            predictor.set_threshold(-0.1)  # Out of range
    
    def test_prediction_deterministic(self, predictor):
        """Test predictions are deterministic in eval mode."""
        spectrogram = np.random.randn(64, 251).astype(np.float32)
        
        pred1, prob1 = predictor.predict(spectrogram, return_probabilities=True)
        pred2, prob2 = predictor.predict(spectrogram, return_probabilities=True)
        
        assert pred1 == pred2
        assert np.isclose(prob1, prob2)
