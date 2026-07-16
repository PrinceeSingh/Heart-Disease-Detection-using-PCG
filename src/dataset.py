"""
dataset.py
PCG Dataset with SpecAugment Data Augmentation

SpecAugment randomly masks frequency bands and time frames to simulate
real-world signal degradation (poor stethoscope contact, motion artifacts).
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class PCGDataset(Dataset):
    """
    PyTorch Dataset for PCG spectrograms with optional SpecAugment.
    
    SpecAugment applies two types of random masking:
    - Frequency masking: removes up to 8 mel bands (12.5% of spectrum)
    - Time masking: removes up to 30 frames (~480ms at 16ms/frame)
    
    This helps the model learn to make predictions even with partial signal loss.
    Augmentation is applied ONLY during training (augment=True).
    """
    
    # SpecAugment hyperparameters
    MAX_FREQ_MASK = 8      # max consecutive mel bands to mask
    MAX_TIME_MASK = 30     # max consecutive time frames to mask
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, augment: bool = False):
        """
        Args:
            features: (N, 64, 251) log-mel spectrograms
            labels: (N,) binary labels (0=normal, 1=abnormal)
            augment: if True, apply SpecAugment during __getitem__
        """
        # Add channel dimension for CNN: (N, 64, 251) → (N, 1, 64, 251)
        self.X = torch.tensor(features, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(labels, dtype=torch.float32)
        self.augment = augment
        
        assert len(self.X) == len(self.y), "Features and labels must have same length"
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Returns (spectrogram, label) pair.
        If augment=True, applies SpecAugment to spectrogram.
        """
        x = self.X[idx].clone()  # (1, 64, 251)
        
        if self.augment:
            x = self._spec_augment(x)
        
        return x, self.y[idx]
    
    def _spec_augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment: random frequency and time masking.
        
        Reference: Park et al. (2019). "SpecAugment: A Simple Data Augmentation
        Method for Automatic Speech Recognition"
        
        Args:
            x: (1, 64, 251) spectrogram tensor
            
        Returns:
            augmented: (1, 64, 251) spectrogram with masked regions
        """
        x = x.clone()
        _, n_mels, n_frames = x.shape
        
        # Frequency masking: blank out up to MAX_FREQ_MASK consecutive mel bands
        # Simulates poor stethoscope-skin contact or spectral attenuation
        f_mask_len = np.random.randint(0, self.MAX_FREQ_MASK)
        if f_mask_len > 0:
            f_start = np.random.randint(0, max(1, n_mels - f_mask_len))
            x[:, f_start:f_start + f_mask_len, :] = x.mean()
        
        # Time masking: blank out up to MAX_TIME_MASK consecutive frames
        # Simulates brief stethoscope repositioning or motion artifacts
        t_mask_len = np.random.randint(0, self.MAX_TIME_MASK)
        if t_mask_len > 0:
            t_start = np.random.randint(0, max(1, n_frames - t_mask_len))
            x[:, :, t_start:t_start + t_mask_len] = x.mean()
        
        return x
