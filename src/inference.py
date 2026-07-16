"""
inference.py
Inference utilities for PCG Heart Disease Classifier

Provides:
- Single sample inference
- Batch inference
- Probability calibration
- Confidence scoring
"""

import numpy as np
import torch
from typing import Union, Tuple
from model import PCGClassifier


class PCGPredictor:
    """
    Wrapper for inference with trained PCGClassifier model.
    
    Usage:
        predictor = PCGPredictor(model_path, device='cuda')
        prob, pred = predictor.predict(spectrogram)
    """
    
    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model_path: path to saved model weights (.pt file)
            threshold: decision threshold for binary classification
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        self.threshold = threshold
        
        # Load model
        self.model = PCGClassifier(n_mels=64, n_frames=251, dropout=0.4)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
        print(f"Device: {self.device}")
    
    def predict(
        self,
        spectrogram: Union[np.ndarray, torch.Tensor],
        return_probabilities: bool = False
    ) -> Union[Tuple[int, float], Tuple[int, float, float]]:
        """
        Predict on a single spectrogram.
        
        Args:
            spectrogram: (64, 251) log-mel spectrogram (or (1, 64, 251))
            return_probabilities: if True, return (pred, prob)
                                  if False, return (pred,)
            
        Returns:
            prediction: 0 (normal) or 1 (abnormal)
            probability: confidence score [0, 1]
        """
        # Ensure correct shape
        if isinstance(spectrogram, np.ndarray):
            spectrogram = torch.from_numpy(spectrogram).float()
        
        if spectrogram.dim() == 2:  # (64, 251)
            spectrogram = spectrogram.unsqueeze(0)  # (1, 64, 251)
        
        if spectrogram.dim() == 3:  # (1, 64, 251)
            spectrogram = spectrogram.unsqueeze(0)  # (1, 1, 64, 251)
        
        spectrogram = spectrogram.to(self.device)
        
        with torch.no_grad():
            logits = self.model(spectrogram)  # (1,)
            prob = torch.sigmoid(logits).item()  # scalar
        
        pred = 1 if prob >= self.threshold else 0
        
        if return_probabilities:
            return pred, prob
        return pred, prob
    
    def predict_batch(
        self,
        spectrograms: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on a batch of spectrograms.
        
        Args:
            spectrograms: (B, 64, 251) or (B, 1, 64, 251) spectrograms
            
        Returns:
            predictions: (B,) binary predictions
            probabilities: (B,) confidence scores
        """
        if isinstance(spectrograms, np.ndarray):
            spectrograms = torch.from_numpy(spectrograms).float()
        
        if spectrograms.dim() == 3:  # (B, 64, 251)
            spectrograms = spectrograms.unsqueeze(1)  # (B, 1, 64, 251)
        
        spectrograms = spectrograms.to(self.device)
        
        with torch.no_grad():
            logits = self.model(spectrograms)  # (B,)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B,)
        
        preds = (probs >= self.threshold).astype(int)
        
        return preds, probs
    
    def set_threshold(self, threshold: float):
        """
        Update decision threshold.
        
        Args:
            threshold: new threshold in [0, 1]
        """
        assert 0 <= threshold <= 1, "Threshold must be in [0, 1]"
        self.threshold = threshold
        print(f"Updated decision threshold to {threshold:.3f}")
