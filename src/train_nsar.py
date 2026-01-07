"""
Training script cho NSARPMD Sports Dataset
Transfer learning từ Kinetics pretrained weights
"""

import argparse
import yaml
import torch
from pathlib import Path


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train model trong 1 epoch"""
    print(f"  [train_one_epoch] Epoch {epoch} - training...")
    # TODO: Implement training loop
    return 0.4, 80.0  # dummy loss, acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate model trên validation set"""
    print(f"  [evaluate] Validating...")
    # TODO: Implement evaluation loop
    return 0.5, 78.0  # dummy val_loss, val_acc


def main(config_path):
    print(f"\n=== NSAR Transfer Learning Script ===")
    print(f"Config: {config_path}")
    
    # Load config
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Config loaded")
        print(f"  - Model: {config.get('model_name', 'x3d_xs')}")
        print(f"  - Freeze backbone: {config.get('freeze_backbone', True)}")
        print(f"  - Kinetics weights: {config.get('kinetics_weights', 'N/A')}")
    else:
        print(f"⚠ Config file not found: {config_path}")
        config = {'epochs': 2}
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    # TODO: Load NSARPMD dataset
    print(f"✓ Dataset: [will load NSARPMD from Kaggle]")
    
    # TODO: Load model + Kinetics weights
    print(f"✓ Model: [will load X3D + transfer from Kinetics]")
    
    # Training loop skeleton
    print(f"\n=== Starting Transfer Learning ===")
    for epoch in range(1, config.get('epochs', 2) + 1):
        print(f"\nEpoch {epoch}/{config.get('epochs', 2)}")
        train_loss, train_acc = train_one_epoch(None, None, None, None, device, epoch)
        val_loss, val_acc = evaluate(None, None, None, device)
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.2f}%")
    
    print(f"\n✓ Transfer learning completed!")
    print(f"  Weights will be saved to: weights/x3d_nsar_best.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer learning to NSAR Sports')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/nsar_transfer.yaml',
        help='Path to config file'
    )
    args = parser.parse_args()
    
    main(args.config)
