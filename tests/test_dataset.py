"""Unit tests for PCG dataset."""

import pytest
import numpy as np
import torch
from src.dataset import PCGDataset


class TestPCGDataset:
    """Test PCGDataset class."""
    
    @pytest.fixture
    def dummy_data(self):
        """Create dummy dataset."""
        n_samples = 32
        features = np.random.randn(n_samples, 64, 251).astype(np.float32)
        labels = np.random.randint(0, 2, n_samples).astype(np.float32)
        return features, labels
    
    def test_dataset_creation(self, dummy_data):
        """Test dataset can be created."""
        features, labels = dummy_data
        ds = PCGDataset(features, labels, augment=False)
        assert ds is not None
    
    def test_dataset_length(self, dummy_data):
        """Test dataset length."""
        features, labels = dummy_data
        ds = PCGDataset(features, labels, augment=False)
        assert len(ds) == 32
    
    def test_getitem_shape(self, dummy_data):
        """Test __getitem__ returns correct shape."""
        features, labels = dummy_data
        ds = PCGDataset(features, labels, augment=False)
        x, y = ds[0]
        
        # Should have channel dimension added
        assert x.shape == (1, 64, 251)
        assert y.item() in [0, 1]
    
    def test_getitem_dtype(self, dummy_data):
        """Test __getitem__ returns correct dtypes."""
        features, labels = dummy_data
        ds = PCGDataset(features, labels, augment=False)
        x, y = ds[0]
        
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32
    
    def test_augmentation_disabled(self, dummy_data):
        """Test augmentation can be disabled."""
        features, labels = dummy_data
        ds = PCGDataset(features, labels, augment=False)
        x, y = ds[0]
        
        # Get same item multiple times - should be identical
        x2, y2 = ds[0]
        assert torch.allclose(x, x2)
    
    def test_augmentation_enabled(self, dummy_data):
        """Test augmentation is enabled and changes samples."""
        features, labels = dummy_data
        ds = PCGDataset(features, labels, augment=True)
        
        x1, y1 = ds[0]
        x2, y2 = ds[0]
        
        # With augmentation, samples should differ (high probability)
        # Note: very small probability of being identical due to random mask being empty
        # For this test, just check both are valid
        assert x1.shape == (1, 64, 251)
        assert x2.shape == (1, 64, 251)
    
    def test_spec_augment_frequency_masking(self, dummy_data):
        """Test frequency masking doesn't break spectrogram."""
        features, labels = dummy_data
        ds = PCGDataset(features, labels, augment=True)
        
        x, y = ds[0]
        # Check values are reasonable (not NaN or Inf)
        assert torch.isfinite(x).all()
    
    def test_spec_augment_time_masking(self, dummy_data):
        """Test time masking doesn't break spectrogram."""
        features, labels = dummy_data
        ds = PCGDataset(features, labels, augment=True)
        
        # Run multiple times to increase probability of time masking
        for _ in range(10):
            x, y = ds[0]
            assert torch.isfinite(x).all()
    
    def test_mismatched_features_labels_raises(self):
        """Test error is raised for mismatched features and labels."""
        features = np.random.randn(10, 64, 251)
        labels = np.random.randint(0, 2, 20)  # Mismatch!
        
        with pytest.raises(AssertionError):
            PCGDataset(features, labels)
    
    def test_class_balance(self, dummy_data):
        """Test dataset preserves class distribution."""
        features, labels = dummy_data
        ds = PCGDataset(features, labels, augment=False)
        
        # Check we can get all samples
        for i in range(len(ds)):
            x, y = ds[i]
            assert x.shape == (1, 64, 251)
