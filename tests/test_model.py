"""Unit tests for PCG model architecture."""

import pytest
import torch
import numpy as np
from src.model import PCGClassifier, count_parameters


class TestPCGClassifier:
    """Test PCGClassifier model."""
    
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return PCGClassifier(n_mels=64, n_frames=251, dropout=0.4)
    
    @pytest.fixture
    def dummy_input(self):
        """Create dummy spectrogram batch."""
        return torch.randn(4, 1, 64, 251)  # (B=4, C=1, H=64, W=251)
    
    def test_model_initialization(self, model):
        """Test model can be initialized."""
        assert model is not None
        assert isinstance(model, PCGClassifier)
    
    def test_forward_pass(self, model, dummy_input):
        """Test forward pass produces correct output shape."""
        output = model(dummy_input)
        assert output.shape == (4,), f"Expected shape (4,), got {output.shape}"
    
    def test_output_dtype(self, model, dummy_input):
        """Test output dtype is float32."""
        output = model(dummy_input)
        assert output.dtype == torch.float32
    
    def test_output_range(self, model, dummy_input):
        """Test logit outputs are in reasonable range (unbounded, but check finite)."""
        output = model(dummy_input)
        assert torch.isfinite(output).all()
    
    def test_model_eval_mode(self, model, dummy_input):
        """Test model works in eval mode."""
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (4,)
    
    def test_model_train_mode(self, model, dummy_input):
        """Test model works in train mode (dropout active)."""
        model.train()
        output = model(dummy_input)
        assert output.shape == (4,)
    
    def test_count_parameters(self, model):
        """Test parameter counting."""
        num_params = count_parameters(model)
        assert num_params > 0
        assert isinstance(num_params, int)
        # CNN-BiLSTM should have ~350K parameters
        assert 300_000 < num_params < 400_000
    
    def test_model_gradients(self, model, dummy_input):
        """Test gradients can be computed."""
        model.train()
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_attention_weights(self, model, dummy_input):
        """Test attention weights method."""
        model.eval()
        with torch.no_grad():
            logits, attn_weights = model.get_attention_weights(dummy_input)
        
        assert logits.shape == (4,)
        assert attn_weights.shape == (4, 62)  # 62 time steps
        
        # Attention weights should sum to 1 (softmax)
        attn_sum = attn_weights.sum(dim=1)
        assert torch.allclose(attn_sum, torch.ones(4), atol=1e-5)
    
    def test_single_sample(self):
        """Test forward pass with single sample."""
        model = PCGClassifier()
        x = torch.randn(1, 1, 64, 251)
        output = model(x)
        assert output.shape == (1,)
    
    def test_different_batch_sizes(self, model):
        """Test model with different batch sizes."""
        for batch_size in [1, 8, 16, 32]:
            x = torch.randn(batch_size, 1, 64, 251)
            output = model(x)
            assert output.shape == (batch_size,)
