"""
train.py
Training loop for PCG Heart Disease Classifier

Features:
- Early stopping based on validation AUC
- Class-weighted loss for imbalanced data (3.78:1 normal:abnormal)
- Learning rate scheduling (ReduceLROnPlateau)
- Gradient clipping for LSTM stability
- Model checkpointing (best AUC only)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from typing import Dict, Tuple

from model import PCGClassifier
from dataset import PCGDataset


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """
    Single training epoch.
    
    Args:
        model: PCGClassifier
        train_loader: training data loader
        criterion: loss function (BCEWithLogitsLoss with pos_weight)
        optimizer: Adam optimizer
        device: torch.device (cpu or cuda)
        
    Returns:
        avg_loss: average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_samples = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for LSTM stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item() * len(y)
        num_samples += len(y)
    
    return total_loss / num_samples


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Validation epoch.
    
    Args:
        model: PCGClassifier
        val_loader: validation data loader
        criterion: loss function
        device: torch.device
        
    Returns:
        avg_loss: average validation loss
        auc: ROC-AUC score
        f1: F1 score (at threshold 0.5)
    """
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            
            # Get probabilities via sigmoid
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    avg_loss = total_loss / len(all_labels)
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, (all_probs >= 0.5).astype(int))
    
    return avg_loss, auc, f1


def train(
    logmel: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: str = "./models",
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    early_stop_patience: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict:
    """
    Full training pipeline.
    
    Args:
        logmel: (N, 64, 251) log-mel spectrograms
        labels: (N,) binary labels
        train_idx: indices for training set
        val_idx: indices for validation set
        test_idx: indices for test set
        output_dir: directory to save model checkpoints
        epochs: max epochs to train
        batch_size: training batch size
        learning_rate: initial learning rate (Adam)
        weight_decay: L2 regularization weight
        early_stop_patience: epochs without AUC improvement before stopping
        device: "cuda" or "cpu"
        
    Returns:
        results: dict with training history and test metrics
    """
    device = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)
    
    # ─── Prepare datasets ─────────────────────────────────────────────
    train_ds = PCGDataset(logmel[train_idx], labels[train_idx], augment=True)
    val_ds = PCGDataset(logmel[val_idx], labels[val_idx], augment=False)
    test_ds = PCGDataset(logmel[test_idx], labels[test_idx], augment=False)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    # ─── Initialize model ────────────────────────────────────────────
    model = PCGClassifier(n_mels=64, n_frames=251, dropout=0.4).to(device)
    
    # ─── Class-weighted loss (handle imbalance) ──────────────────────
    n_neg = (labels[train_idx] == 0).sum()
    n_pos = (labels[train_idx] == 1).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # ─── Optimizer + Scheduler ───────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=4,
        verbose=False
    )
    
    # ─── Training loop ───────────────────────────────────────────────
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
        'val_f1': []
    }
    
    best_val_auc = 0.0
    patience_counter = 0
    best_epoch = 0
    best_model_path = os.path.join(output_dir, 'best_model.pt')
    
    print(f"Training on {device}")
    print(f"Training set: {n_neg:,} normal | {n_pos:,} abnormal (pos_weight: {pos_weight.item():.2f})")
    print(f"Validation set: {len(val_ds)}")
    print(f"Test set: {len(test_ds)}")
    print("\n" + "="*80)
    
    for epoch in range(1, epochs + 1):
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_auc, val_f1 = validate_epoch(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)
        
        # Learning rate scheduling
        scheduler.step(val_auc)
        
        # Checkpoint best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            flag = "✓ saved"
        else:
            patience_counter += 1
            flag = f"(patience {patience_counter}/{early_stop_patience})"
        
        print(f"Epoch {epoch:2d} | loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_AUC {val_auc:.4f} | "
              f"val_F1 {val_f1:.4f} | {flag}")
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    print("="*80)
    print(f"Best validation AUC: {best_val_auc:.4f} (epoch {best_epoch})\n")
    
    # ─── Evaluate on test set ────────────────────────────────────────
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    test_loss, test_auc, test_f1 = validate_epoch(model, test_loader, criterion, device)
    
    results = {
        'best_epoch': int(best_epoch),
        'best_val_auc': float(best_val_auc),
        'test_auc': float(test_auc),
        'test_loss': float(test_loss),
        'test_f1': float(test_f1),
        'pos_weight': float(pos_weight.item()),
        'history': history,
        'model_path': best_model_path
    }
    
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test F1:  {test_f1:.4f}")
    
    return results
