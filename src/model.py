"""
model.py
PCG Heart Disease Classifier Model Architecture

CNN-BiLSTM with self-attention pooling for binary classification on log-mel spectrograms.
Total parameters: ~351K (deployment-friendly)
"""

import torch
import torch.nn as nn
from typing import Tuple


class PCGClassifier(nn.Module):
    """
    CNN-BiLSTM with Attention Pooling for PCG Classification
    
    Architecture:
    - CNN Encoder: 3 convolutional blocks (1→32→64→128 channels)
    - BiLSTM: 2 layers, hidden_size=64 (bidirectional → 128 output)
    - Attention: Learnable pooling across time steps
    - Classifier: 2-layer MLP (128→64→1)
    
    Input:  (B, 1, 64, 251)  # log-mel spectrogram: 64 freq bins × 251 time frames
    Output: (B,)             # binary logits (apply sigmoid at inference)
    """
    
    def __init__(self, n_mels: int = 64, n_frames: int = 251, dropout: float = 0.4):
        super().__init__()
        
        # ─── CNN Encoder (3 blocks) ───────────────────────────────────
        # Block 1: 1→32 channels, MaxPool(2,2)
        # Block 2: 32→64 channels, MaxPool(2,2)
        # Block 3: 64→128 channels, MaxPool(16,1) - collapses freq axis
        # Output: (B, 128, 1, 62)
        self.cnn = nn.Sequential(
            # Block 1: Coarse pattern extraction
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # → (B, 32, 32, 125)
            nn.Dropout2d(0.2),
            
            # Block 2: Finer discriminative features
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # → (B, 64, 16, 62)
            nn.Dropout2d(0.2),
            
            # Block 3: Collapse frequency axis, preserve time
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((16, 1)),   # → (B, 128, 1, 62)
        )
        
        # ─── Bidirectional LSTM (2 layers) ────────────────────────────
        # Input:  (B, 62, 128)  — 62 time steps, 128-dim features
        # Output: (B, 62, 128)  — bidirectional: 64×2=128
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # ─── Self-Attention Pooling ──────────────────────────────────
        # Computes scalar attention weight for each time step
        # Then performs weighted sum: (B, 62, 128) → (B, 128)
        self.attention = nn.Linear(128, 1)
        
        # ─── Classification Head ──────────────────────────────────────
        # 128 → 64 → 1 (logit, not sigmoid)
        # BCEWithLogitsLoss applies sigmoid internally for stability
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: (B, 1, 64, 251) log-mel spectrogram
            
        Returns:
            logits: (B,) binary logits (0 = normal, 1 = abnormal)
        """
        # CNN encoding
        x = self.cnn(x)                              # (B, 128, 1, 62)
        x = x.squeeze(2).permute(0, 2, 1)            # (B, 62, 128)
        
        # BiLSTM
        x, _ = self.lstm(x)                          # (B, 62, 128)
        
        # Attention pooling
        attn_weights = torch.softmax(self.attention(x), dim=1)  # (B, 62, 1)
        x = (x * attn_weights).sum(dim=1)            # (B, 128)
        
        # Classification
        return self.classifier(x).squeeze(1)         # (B,)
    
    def get_attention_weights(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention weights for interpretability (used in Grad-CAM).
        
        Args:
            x: (B, 1, 64, 251) log-mel spectrogram
            
        Returns:
            logits: (B,) model predictions
            attn_weights: (B, 62) attention weights across time steps
        """
        x = self.cnn(x)                              # (B, 128, 1, 62)
        x = x.squeeze(2).permute(0, 2, 1)            # (B, 62, 128)
        x, _ = self.lstm(x)                          # (B, 62, 128)
        attn_weights = torch.softmax(self.attention(x), dim=1).squeeze(-1)  # (B, 62)
        
        # For classification
        x_pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, 128)
        logits = self.classifier(x_pooled).squeeze(1)            # (B,)
        
        return logits, attn_weights


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
