#!/usr/bin/env python3
"""
Command-line training script for PCG Heart Disease Classifier

Usage:
    python train_cli.py --features features.npy --labels labels.npy \
                       --output ./models --epochs 40 --batch_size 64
"""

import argparse
import numpy as np
import torch
import json
from src.train import train


def main():
    parser = argparse.ArgumentParser(
        description="Train PCG Heart Disease Classifier"
    )
    parser.add_argument(
        "--features",
        required=True,
        type=str,
        help="Path to features.npy (N, 64, 251)"
    )
    parser.add_argument(
        "--labels",
        required=True,
        type=str,
        help="Path to labels.npy (N,) binary labels"
    )
    parser.add_argument(
        "--train-idx",
        required=True,
        type=str,
        help="Path to train_idx.npy"
    )
    parser.add_argument(
        "--val-idx",
        required=True,
        type=str,
        help="Path to val_idx.npy"
    )
    parser.add_argument(
        "--test-idx",
        required=True,
        type=str,
        help="Path to test_idx.npy"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models",
        help="Output directory for model checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Max epochs to train"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 regularization weight"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: 'cuda' or 'cpu'"
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    logmel = np.load(args.features)
    labels = np.load(args.labels)
    train_idx = np.load(args.train_idx)
    val_idx = np.load(args.val_idx)
    test_idx = np.load(args.test_idx)
    
    print(f"Loaded features: {logmel.shape}")
    print(f"Loaded labels: {labels.shape}")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Train
    print("\nStarting training...\n")
    results = train(
        logmel=logmel,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        early_stop_patience=args.patience,
        device=args.device
    )
    
    # Save results
    results_path = f"{args.output}/results.json"
    # Make serializable
    results_json = {
        'best_epoch': results['best_epoch'],
        'best_val_auc': results['best_val_auc'],
        'test_auc': results['test_auc'],
        'test_loss': results['test_loss'],
        'test_f1': results['test_f1'],
        'pos_weight': results['pos_weight'],
        'model_path': results['model_path']
    }
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(f"Model saved to {results['model_path']}")


if __name__ == "__main__":
    main()
